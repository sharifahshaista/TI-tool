"""
Web Scraper Module

Main scraper class that coordinates all components to crawl websites
and extract content while respecting robots.txt rules.
"""

import requests
import logging
import time
from urllib.parse import urlparse, urljoin
from collections import deque

from .robots_handler import RobotsHandler
from .content_extractor import ContentExtractor
from .url_tracker import URLTracker
from .csv_writer import CSVWriter
from .url_utils import normalize_url, same_domain, ensure_scheme

logger = logging.getLogger(__name__)


class WebScraper:
    """Main web scraper coordinating all components."""

    def __init__(self, user_agent='PythonWebScraper/1.0', default_delay=1.0, clear_tracking=False, output_dir='crawled_data'):
        """
        Initialize the WebScraper.

        Args:
            user_agent (str): User agent string for requests
            default_delay (float): Default delay between requests in seconds
            clear_tracking (bool): If True, clear previous crawl history before starting
            output_dir (str): Directory for output CSV files
        """
        self.user_agent = user_agent
        self.default_delay = default_delay

        # Initialize components
        self.robots_handler = RobotsHandler(user_agent=user_agent)
        self.content_extractor = ContentExtractor()
        self.url_tracker = URLTracker()
        self.csv_writer = CSVWriter(output_dir=output_dir)

        # Clear tracking if requested
        if clear_tracking:
            logger.info("Clearing previous crawl tracking...")
            self.url_tracker.clear()

        # Request session for connection pooling
        self.session = requests.Session()

        # Set realistic browser headers to help bypass basic bot detection
        # Since robots.txt often allows crawling, these headers help the scraper
        # be recognized as a legitimate client
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',  # Do Not Track
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        })

        logger.info("WebScraper initialized")

    def _get_crawl_delay(self, url):
        """
        Get the appropriate crawl delay for a URL.

        Args:
            url (str): URL to check

        Returns:
            float: Delay in seconds
        """
        delay = self.robots_handler.get_crawl_delay(url)
        if delay is None:
            delay = self.default_delay
        logger.debug(f"Using crawl delay of {delay}s for {url}")
        return delay

    def _fetch_page(self, url, timeout=10):
        """
        Fetch a webpage.

        Args:
            url (str): URL to fetch
            timeout (int): Request timeout in seconds

        Returns:
            str or None: HTML content or None if fetch failed
        """
        try:
            logger.info(f"Fetching: {url}")
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()

            logger.info(f"Successfully fetched {url} ({len(response.content)} bytes)")

            # Explicitly decode as UTF-8 with error replacement
            # This prevents encoding auto-detection failures that cause corruption
            return response.content.decode('utf-8', errors='replace')

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def crawl_url(self, url, extract_links=False, skip_if_crawled=True, should_stop=None):
        """
        Crawl a single URL.

        Args:
            url (str): URL to crawl
            extract_links (bool): Whether to extract links from the page
            skip_if_crawled (bool): If True, skip URLs that were previously crawled (default: True)
            should_stop (callable, optional): Callback function that returns True to stop crawling

        Returns:
            dict: Crawl result with success status, content, links, and error info
        """
        # Normalize URL for consistency
        url = normalize_url(ensure_scheme(url))

        result = {
            'url': url,
            'success': False,
            'content': None,
            'links': [],
            'error': None,
            'warning': False
        }

        # Check if already crawled (using normalized URL) - only if skip_if_crawled is True
        if skip_if_crawled and self.url_tracker.is_crawled(url):
            logger.debug(f"URL already crawled: {url}")
            result['error'] = "Already crawled"
            result['warning'] = True

            # NEW: Extract links even from previously crawled URLs
            if extract_links:
                # Respect crawl delay even for link extraction
                delay = self._get_crawl_delay(url)
                time.sleep(delay)

                # Fetch page only for link extraction
                html_content = self._fetch_page(url)
                if html_content:
                    result['links'] = self.content_extractor.extract_links(html_content, url)
                    logger.info(f"Extracted {len(result['links'])} links from previously crawled URL: {url}")
                else:
                    logger.warning(f"Failed to fetch previously crawled URL for link extraction: {url}")

            return result

        # Check for user cancellation before proceeding
        if should_stop and should_stop():
            logger.info(f"Crawl cancelled by user before fetching: {url}")
            result['error'] = "Stopped by user"
            return result

        # Check robots.txt
        if not self.robots_handler.can_fetch(url):
            logger.warning(f"robots.txt disallows crawling: {url}")
            result['error'] = "Disallowed by robots.txt"
            result['warning'] = True
            self.url_tracker.mark_crawled(url, success=False, error=result['error'])
            return result

        # Respect crawl delay with interruptible sleep
        delay = self._get_crawl_delay(url)

        # Sleep in small increments so we can check for stop requests
        start_time = time.time()
        while time.time() - start_time < delay:
            # Check if user wants to stop during the delay
            if should_stop and should_stop():
                logger.info(f"Crawl stopped by user during delay for {url}")
                result['error'] = "Stopped by user"
                return result

            # Sleep in small increments (check every 0.5 seconds)
            remaining = delay - (time.time() - start_time)
            if remaining > 0:
                time.sleep(min(0.5, remaining))

        # Fetch page
        html_content = self._fetch_page(url)
        if html_content is None:
            result['error'] = "Failed to fetch page"
            self.url_tracker.mark_crawled(url, success=False, error=result['error'])
            return result

        # Extract links if requested
        if extract_links:
            result['links'] = self.content_extractor.extract_links(html_content, url)
            logger.debug(f"Extracted {len(result['links'])} links from {url}")

        # Extract content
        text_content = self.content_extractor.extract(html_content)
        if not text_content:
            result['error'] = "No content extracted"
            self.url_tracker.mark_crawled(url, success=False, error=result['error'])
            return result

        # Write to CSV
        if self.csv_writer.write_result(url, text_content):
            result['success'] = True
            result['content'] = text_content
            self.url_tracker.mark_crawled(url, success=True)
            logger.info(f"Successfully crawled and saved: {url}")
        else:
            result['error'] = "Failed to write to CSV"
            self.url_tracker.mark_crawled(url, success=False, error=result['error'])

        return result

    def crawl_urls(self, urls, extract_links=False):
        """
        Crawl multiple URLs.

        Args:
            urls (list): List of URLs to crawl
            extract_links (bool): Whether to extract links from pages (default: False)

        Returns:
            dict: Summary of crawl results with extracted links if requested
        """
        logger.info(f"Starting crawl of {len(urls)} URLs")

        results = []
        successful = 0
        failed = 0
        warnings = 0
        all_links = set()  # Collect all unique links found

        for url in urls:
            result = self.crawl_url(url, extract_links=extract_links)
            results.append(result)

            # Collect links if extraction was enabled
            if extract_links and result.get('links'):
                all_links.update(result['links'])

            if result['success']:
                successful += 1
            else:
                failed += 1
                if result.get('warning', False):
                    warnings += 1

        summary = {
            'total': len(urls),
            'successful': successful,
            'failed': failed,
            'warnings': warnings,
            'results': results,
            'total_links_found': len(all_links) if extract_links else 0,
            'unique_links': list(all_links) if extract_links else []
        }

        logger.info(f"Crawl complete: {successful} successful, {failed} failed, {warnings} warnings")
        if extract_links:
            logger.info(f"Found {len(all_links)} unique links across all pages")
        return summary

    def get_stats(self):
        """
        Get scraper statistics.

        Returns:
            dict: Statistics about crawled URLs
        """
        return {
            'total_crawled': self.url_tracker.get_crawled_count(),
            'successful': self.url_tracker.get_successful_count(),
            'failed': self.url_tracker.get_failed_count(),
            'output_files': self.csv_writer.get_output_files()
        }

    def crawl_site(self, start_url, max_pages=100, max_depth=3, same_domain_only=True, skip_if_crawled=True, should_stop=None):
        """
        Crawl an entire website starting from a URL, following links.

        Args:
            start_url (str): Starting URL to begin crawling
            max_pages (int): Maximum number of pages to crawl
            max_depth (int): Maximum depth to crawl (0 = start page only)
            same_domain_only (bool): Only crawl pages on the same domain
            skip_if_crawled (bool): If True, skip URLs that were previously crawled (default: True)
                                   Set to False to re-crawl URLs with new parameters
            should_stop (callable, optional): Callback function that returns True to stop crawling.
                                             Called at the start of each iteration to check for cancellation.

        Returns:
            dict: Summary of crawl results
        """
        logger.info(f"Starting site crawl from {start_url}")
        logger.info(f"Settings: max_pages={max_pages}, max_depth={max_depth}, same_domain={same_domain_only}, skip_if_crawled={skip_if_crawled}")

        # Normalize start URL
        start_url = normalize_url(ensure_scheme(start_url))

        # Queue of (url, depth) tuples
        queue = deque([(start_url, 0)])
        visited = set()

        results = []
        successful = 0
        failed = 0
        warnings = 0
        pages_crawled = 0

        while queue and pages_crawled < max_pages:
            # Check for user cancellation
            if should_stop and should_stop():
                logger.info("Crawl stopped by user request")
                break

            url, depth = queue.popleft()

            # Normalize URL
            url = normalize_url(ensure_scheme(url))

            # Skip if already visited
            if url in visited:
                continue

            # Skip if depth exceeded
            if depth > max_depth:
                logger.debug(f"Skipping {url} - max depth exceeded")
                continue

            visited.add(url)

            # Skip if different domain (if same_domain_only)
            if same_domain_only:
                if not same_domain(url, start_url):
                    logger.debug(f"Skipping {url} - different domain")
                    continue

            # Crawl the page - ALWAYS extract links if we haven't reached max depth
            # The fix: depth <= max_depth (was: depth < max_depth)
            # This ensures we extract links from the starting page (depth 0)
            extract_links = (depth < max_depth)  # Extract links if we can still go deeper
            result = self.crawl_url(url, extract_links=extract_links, skip_if_crawled=skip_if_crawled, should_stop=should_stop)
            results.append(result)
            pages_crawled += 1

            if result['success']:
                successful += 1
                logger.info(f"[{pages_crawled}/{max_pages}] Crawled: {url} (depth {depth})")
            else:
                failed += 1
                if result.get('warning', False):
                    warnings += 1
                logger.warning(f"[{pages_crawled}/{max_pages}] Failed: {url} - {result['error']}")

            # Add discovered links to queue (regardless of success/failure)
            # This is crucial for the fix: even if URL was already crawled, we want its links
            if result['links']:
                # Ensure all links have schemes and normalize them
                valid_links = []
                for link in result['links']:
                    try:
                        # Ensure scheme first, then normalize
                        link_with_scheme = ensure_scheme(link) if '://' not in link else link
                        normalized = normalize_url(link_with_scheme)
                        valid_links.append(normalized)
                    except Exception as e:
                        logger.debug(f"Skipping invalid link {link}: {e}")
                        continue

                # Filter out already visited links
                new_links = [link for link in valid_links if link not in visited]
                for link in new_links:
                    queue.append((link, depth + 1))
                logger.debug(f"Added {len(new_links)} new links to queue (from {len(result['links'])} found)")

        summary = {
            'start_url': start_url,
            'total': pages_crawled,
            'successful': successful,
            'failed': failed,
            'warnings': warnings,
            'max_pages_reached': pages_crawled >= max_pages,
            'results': results
        }

        logger.info(f"Site crawl complete: {successful} successful, {failed} failed, {warnings} warnings out of {pages_crawled} pages")
        return summary

    def close(self):
        """Clean up resources."""
        self.session.close()
        logger.info("WebScraper closed")