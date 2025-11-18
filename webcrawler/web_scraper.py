#!/usr/bin/env python3
"""
Web Scraper with robots.txt compliance, URL tracking, and content filtering.

This module provides a robust web scraping solution that:
- Respects robots.txt rules
- Tracks visited URLs to prevent duplicate crawling
- Filters out ads and images
- Exports data to CSV format
- Implements comprehensive error handling
"""

import csv
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Set, List, Dict, Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup


class WebScraper:
    """
    A web scraper that respects robots.txt and tracks visited URLs.

    Attributes:
        config (dict): Configuration settings for the scraper
        visited_urls (set): Set of URLs that have been crawled
        robot_parsers (dict): Cache of RobotFileParser objects per domain
        session (requests.Session): Persistent HTTP session
        logger (logging.Logger): Logger instance
    """

    def __init__(self, config: Dict[str, any]):
        """
        Initialize the web scraper with configuration.

        Args:
            config: Dictionary containing scraper configuration
        """
        self.config = config
        self.visited_urls: Set[str] = set()
        self.robot_parsers: Dict[str, RobotFileParser] = {}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config.get('user_agent', 'WebScraperBot/1.0')
        })

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

        # Load visited URLs if exists
        self._load_visited_urls()

    def _setup_logging(self) -> None:
        """Configure logging with appropriate level and format."""
        log_level = getattr(logging, self.config.get('log_level', 'INFO'))
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data/scraper.log'),
                logging.StreamHandler()
            ]
        )

    def _load_visited_urls(self) -> None:
        """Load previously visited URLs from tracking file."""
        tracking_file = Path(self.config.get('visited_urls_file', 'data/visited_urls.txt'))
        if tracking_file.exists():
            try:
                with open(tracking_file, 'r') as f:
                    self.visited_urls = set(line.strip() for line in f if line.strip())
                self.logger.info(f"Loaded {len(self.visited_urls)} visited URLs from {tracking_file}")
            except Exception as e:
                self.logger.error(f"Failed to load visited URLs: {e}")

    def _save_visited_url(self, url: str) -> None:
        """
        Save a visited URL to the tracking file.

        Args:
            url: The URL to save
        """
        tracking_file = Path(self.config.get('visited_urls_file', 'data/visited_urls.txt'))
        try:
            tracking_file.parent.mkdir(parents=True, exist_ok=True)
            with open(tracking_file, 'a') as f:
                f.write(f"{url}\n")
        except Exception as e:
            self.logger.error(f"Failed to save visited URL {url}: {e}")

    def _get_robot_parser(self, url: str) -> Optional[RobotFileParser]:
        """
        Get or create a RobotFileParser for the given URL's domain.

        Args:
            url: URL to get the robot parser for

        Returns:
            RobotFileParser instance or None if robots.txt cannot be fetched
        """
        parsed_url = urlparse(url)
        domain = f"{parsed_url.scheme}://{parsed_url.netloc}"

        if domain not in self.robot_parsers:
            robots_url = urljoin(domain, '/robots.txt')
            rp = RobotFileParser()
            rp.set_url(robots_url)

            try:
                rp.read()
                self.robot_parsers[domain] = rp
                self.logger.info(f"Loaded robots.txt from {domain}")
            except Exception as e:
                self.logger.warning(f"Could not load robots.txt from {domain}: {e}")
                # Create permissive parser if robots.txt unavailable
                rp = RobotFileParser()
                rp.parse([])
                self.robot_parsers[domain] = rp

        return self.robot_parsers[domain]

    def can_fetch(self, url: str) -> bool:
        """
        Check if the URL can be fetched according to robots.txt.

        Args:
            url: URL to check

        Returns:
            True if the URL can be fetched, False otherwise
        """
        rp = self._get_robot_parser(url)
        if rp is None:
            return True

        user_agent = self.config.get('user_agent', '*')
        can_fetch = rp.can_fetch(user_agent, url)

        if not can_fetch:
            self.logger.info(f"Robots.txt disallows fetching: {url}")

        return can_fetch

    def _is_valid_content(self, element) -> bool:
        """
        Check if an HTML element contains valid content (not ads or images).

        Args:
            element: BeautifulSoup element to check

        Returns:
            True if element contains valid content, False if it's an ad or image
        """
        # Filter out common ad-related classes and IDs
        ad_indicators = ['ad', 'advertisement', 'promo', 'sponsored', 'banner']

        if element.name in ['img', 'picture', 'svg']:
            return False

        # Check element attributes
        for attr in ['class', 'id']:
            values = element.get(attr, [])
            if isinstance(values, str):
                values = [values]

            for value in values:
                if any(indicator in value.lower() for indicator in ad_indicators):
                    return False

        return True

    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """
        Extract clean text content from HTML, filtering out ads and images.

        Args:
            soup: BeautifulSoup object of the page

        Returns:
            Cleaned text content
        """
        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'img', 'picture', 'svg', 'video', 'audio']):
            element.decompose()

        # Remove ad-related elements
        ad_indicators = ['ad', 'advertisement', 'promo', 'sponsored', 'banner']
        for indicator in ad_indicators:
            for element in soup.find_all(class_=lambda x: x and indicator in x.lower()):
                element.decompose()
            for element in soup.find_all(id=lambda x: x and indicator in x.lower()):
                element.decompose()

        # Extract text
        text = soup.get_text(separator=' ', strip=True)

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        return text

    def scrape_url(self, url: str, skip_if_crawled: bool = True) -> Optional[Dict[str, str]]:
        """
        Scrape a single URL and return the extracted data.

        Args:
            url: URL to scrape
            skip_if_crawled: If True, skip URLs that were previously crawled (default: True)

        Returns:
            Dictionary containing date, url, and text_content, or None if scraping failed
        """
        # Check if already visited - only if skip_if_crawled is True
        if skip_if_crawled and url in self.visited_urls:
            self.logger.info(f"Skipping already visited URL: {url}")
            return None

        # Check robots.txt
        if not self.can_fetch(url):
            self.visited_urls.add(url)
            self._save_visited_url(url)
            return None

        # Respect rate limiting
        rate_limit = self.config.get('rate_limit_seconds', 1.0)
        time.sleep(rate_limit)

        try:
            # Fetch the page
            response = self.session.get(
                url,
                timeout=self.config.get('timeout_seconds', 30),
                allow_redirects=True
            )
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract clean text content
            text_content = self._extract_text_content(soup)

            # Mark as visited
            self.visited_urls.add(url)
            self._save_visited_url(url)

            # Return scraped data
            return {
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'url': url,
                'text_content': text_content
            }

        except requests.exceptions.HTTPError as e:
            self.logger.error(f"HTTP error for {url}: {e}")
        except requests.exceptions.Timeout as e:
            self.logger.error(f"Timeout for {url}: {e}")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error for {url}: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error scraping {url}: {e}")

        # Mark as visited even on error to avoid retry loops
        self.visited_urls.add(url)
        self._save_visited_url(url)

        return None

    def scrape_urls(self, urls: List[str], output_file: str = 'data/scraped_data.csv', skip_if_crawled: bool = True) -> None:
        """
        Scrape multiple URLs and save results to CSV.

        Args:
            urls: List of URLs to scrape
            output_file: Path to output CSV file
            skip_if_crawled: If True, skip URLs that were previously crawled (default: True)
        """
        self.logger.info(f"Starting scrape of {len(urls)} URLs")

        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if file exists to determine if we need to write headers
        file_exists = output_path.exists()

        # Open CSV file for writing
        try:
            with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['date', 'url', 'text_content']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Write header if file is new
                if not file_exists:
                    writer.writeheader()

                # Scrape each URL
                success_count = 0
                for url in urls:
                    self.logger.info(f"Scraping: {url}")
                    data = self.scrape_url(url, skip_if_crawled=skip_if_crawled)

                    if data:
                        writer.writerow(data)
                        csvfile.flush()  # Ensure data is written immediately
                        success_count += 1
                        self.logger.info(f"Successfully scraped and saved: {url}")

                self.logger.info(f"Scraping complete. Successfully scraped {success_count}/{len(urls)} URLs")

        except Exception as e:
            self.logger.error(f"Error writing to CSV file: {e}")
            raise

    def get_crawl_delay(self, url: str) -> float:
        """
        Get the crawl delay specified in robots.txt for the URL's domain.

        Args:
            url: URL to check crawl delay for

        Returns:
            Crawl delay in seconds, or default rate limit if not specified
        """
        rp = self._get_robot_parser(url)
        if rp:
            user_agent = self.config.get('user_agent', '*')
            crawl_delay = rp.crawl_delay(user_agent)
            if crawl_delay:
                return float(crawl_delay)

        return self.config.get('rate_limit_seconds', 1.0)


def main():
    """Example usage of the WebScraper."""
    # Load configuration
    config = {
        'user_agent': 'WebScraperBot/1.0 (+http://example.com/bot)',
        'rate_limit_seconds': 1.0,
        'timeout_seconds': 30,
        'log_level': 'INFO',
        'visited_urls_file': 'data/visited_urls.txt'
    }

    # Initialize scraper
    scraper = WebScraper(config)

    # Example URLs to scrape
    urls = [
        'https://example.com',
        'https://example.org',
    ]

    # Scrape URLs and save to CSV
    scraper.scrape_urls(urls, output_file='data/scraped_data.csv')


if __name__ == '__main__':
    main()