"""
Enhanced Web Crawler
Enriched with features from crawler_monitor, deepcrawl_example, virtual_scrolling, and builtin_browser_example

Features:
- Deep crawl strategies (BFS, DFS, Best-First)
- Filters and scorers for targeted crawling
- Virtual scroll support for infinite scroll pages
- Builtin browser mode for better performance
- Real-time progress tracking
- Streaming support
- S3 storage integration for data persistence
"""

import asyncio
import logging
import os
import re
import time
import requests
import gzip
import xml.etree.ElementTree as ET
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from urllib.parse import urlparse, urljoin
from dataclasses import dataclass, field
from enum import Enum
from bs4 import BeautifulSoup

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
    LXMLWebScrapingStrategy,
    VirtualScrollConfig
)
from crawl4ai.deep_crawling import (
    BFSDeepCrawlStrategy,
    DFSDeepCrawlStrategy,
    BestFirstCrawlingStrategy
)
from crawl4ai.deep_crawling.filters import (
    FilterChain,
    URLPatternFilter,
    DomainFilter,
    ContentTypeFilter
)
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer

# Import S3 storage
try:
    from aws_storage import get_storage
    HAS_S3_STORAGE = True
except ImportError:
    HAS_S3_STORAGE = False
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter


class CrawlMode(Enum):
    """Available crawl modes"""
    SIMPLE = "simple"  # Discover links and crawl (original)
    SITEMAP = "sitemap"  # Sitemap-based crawl
    PAGINATION = "pagination"  # Two-phase pagination crawl
    CATEGORY = "category"  # Category/topic-focused crawl
    DEEP_BFS = "deep_bfs"  # Breadth-first deep crawl
    DEEP_DFS = "deep_dfs"  # Depth-first deep crawl
    BEST_FIRST = "best_first"  # Best-first with scoring


@dataclass
class CrawlResult:
    """Result of a crawling operation"""
    success: bool
    pages_found: int
    pages_saved: int
    message: str
    duration: float = 0.0
    failed_urls: List[Dict] = field(default_factory=list)
    mode: str = "simple"
    structured_data: List[Dict] = field(default_factory=list)  # For multi-format export


class EnhancedWebCrawler:
    """
    Enhanced web crawler with deep crawl, filtering, and virtual scroll support
    """
    
    # URL patterns to exclude - focuses crawler on news/articles/blogs
    EXCLUDE_URL_PATTERNS = [
        # Navigation and utility pages
        'about-us', 'about', 'aboutus',
        'contact-us', 'contact', 'contactus',
        'faq', 'faqs', 'frequently-asked',
        'author', 'authors', 'writer', 'writers',
        'team', 'staff', 'people',
        
        # Commercial/marketing pages  
        'advertisement', 'advertise', 'advertising', 'ads',
        'sponsor', 'partners', 'partnership',
        'careers', 'jobs', 'employment',
        'subscribe', 'subscription', 'newsletter',
        'shop', 'store', 'product', 'products',
        'pricing', 'plans',
        
        # Legal/policy pages
        'privacy', 'privacy-policy', 'cookie-policy',
        'terms', 'terms-of-service', 'terms-and-conditions',
        'legal', 'disclaimer', 'dmca',
        
        # Technical/admin pages
        'login', 'signin', 'signup', 'register',
        'account', 'profile', 'dashboard',
        'search', 'sitemap',
        'feed', 'rss', 'atom',
        
        # Media/file extensions
        '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.svg',
        '.mp4', '.mp3', '.avi', '.mov',
        '.zip', '.rar', '.tar', '.gz',
        '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx'
    ]
    
    def __init__(self, start_url: str, output_dir: str = "crawled_data/saved_md", strip_links: bool = True):
        self.start_url = start_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.strip_links = strip_links
        
        # Parse URL
        parsed = urlparse(start_url)
        self.parsed_url = parsed  # Store parsed URL for later use
        self.base_domain = parsed.netloc
        self.base_url = f"{parsed.scheme}://{self.base_domain}"
        
        # State tracking
        self.visited_urls = set()
        self.cancelled = False

        # Logging setup
        self.log_path = self.output_dir / "crawl.log"
        if not self.log_path.exists():
            try:
                self.log_path.write_text("timestamp,event,mode,url,status,info\n", encoding='utf-8')
            except Exception:
                pass

    def _log(self, event: str, mode: str = "", url: str = "", status: str = "", info: str = "") -> None:
        """Append a CSV log line to crawl.log with ISO timestamp."""
        try:
            ts = datetime.now().isoformat()
            # Escape commas in free text fields
            safe_url = url.replace(',', ' ')
            safe_status = (status or "").replace(',', ' ')
            safe_info = (info or "").replace('\n', ' ').replace(',', ' ')
            line = f"{ts},{event},{mode},{safe_url},{safe_status},{safe_info}\n"
            with self.log_path.open('a', encoding='utf-8') as f:
                f.write(line)
        except Exception:
            # Don't break crawl on logging issues
            pass
    
    def _should_exclude_url(self, url: str) -> bool:
        """
        Check if a URL should be excluded from crawling.
        Returns True if URL matches any exclusion pattern.
        
        This method uses path segment analysis to avoid false positives
        (e.g., excludes '/about' but not '/news/about-climate-change')
        
        Args:
            url: The URL to check
            
        Returns:
            bool: True if URL should be excluded, False otherwise
        """
        url_lower = url.lower()
        parsed = urlparse(url)
        
        # Keep the homepage
        if not parsed.path or parsed.path == '/':
            return False
        
        # Extract path segments for more accurate matching
        path = parsed.path.strip('/')
        path_segments = [seg for seg in path.split('/') if seg]
        
        # File extension check (highest priority)
        file_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.svg',
                          '.mp4', '.mp3', '.avi', '.mov',
                          '.zip', '.rar', '.tar', '.gz',
                          '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
                          '.xml']  # Include .xml for sitemap files
        for ext in file_extensions:
            if url_lower.endswith(ext):
                return True
        
        # Check for exclusion patterns as path segments or exact matches
        # This prevents false positives like "about-climate-change" matching "about"
        exclude_segments = [
            'about-us', 'aboutus',
            'contact-us', 'contactus',
            'faq', 'faqs', 'frequently-asked',
            'author', 'authors', 'writer', 'writers',
            'team', 'staff', 'people',
            'advertisement', 'advertise', 'advertising', 'ads',
            'sponsor', 'partners', 'partnership',
            'careers', 'jobs', 'employment',
            'subscribe', 'subscription', 'newsletter',
            'shop', 'store', 'product', 'products',
            'pricing', 'plans',
            'privacy', 'privacy-policy', 'cookie-policy',
            'terms', 'terms-of-service', 'terms-and-conditions',
            'legal', 'disclaimer', 'dmca',
            'login', 'signin', 'signup', 'register',
            'account', 'profile', 'dashboard',
            'search', 'sitemap',
            'feed', 'rss', 'atom',
            'category', 'categories', 'tag', 'tags',
            'archive', 'archives'
        ]
        
        # Check if any path segment exactly matches an exclusion pattern
        for segment in path_segments:
            if segment in exclude_segments:
                return True
        
        # Check for numeric-only segments (often pagination like /page/2)
        # Exclude if a segment is purely numeric and follows 'page' or is standalone
        for i, segment in enumerate(path_segments):
            if segment.isdigit():
                # If it's the only segment or follows 'page', exclude it
                if len(path_segments) == 1 or (i > 0 and path_segments[i-1] in ['page', 'pages', 'p']):
                    return True
        
        # Special check for single-word segments that are common non-content pages
        # Only exclude if they're the ONLY segment or the FIRST segment
        single_word_excludes = ['about', 'contact', 'page', 'pages']
        if len(path_segments) > 0:
            first_segment = path_segments[0]
            # Exclude if it's a standalone page (e.g., /about, /contact)
            if len(path_segments) == 1 and first_segment in single_word_excludes:
                return True
            # Also exclude if it's the first segment followed by another non-content segment
            # (e.g., /about/team, /contact/form)
            if (len(path_segments) == 2 and 
                first_segment in single_word_excludes and
                path_segments[1] in exclude_segments):
                return True
        
        # Check for query parameters that suggest non-article pages
        if parsed.query:
            query_lower = parsed.query.lower()
            exclude_params = ['search', 's=', 'q=', 'filter', 'sort']
            for param in exclude_params:
                if param in query_lower:
                    return True
        
        return False
    
    def _get_browser_config(self, use_builtin_browser: bool = True) -> BrowserConfig:
        """
        Get browser configuration with enhanced error handling settings.
        
        Args:
            use_builtin_browser: Whether to use builtin browser mode
            
        Returns:
            BrowserConfig with optimized settings for avoiding ERR_ABORTED and bot detection
        """
        return BrowserConfig(
            browser_mode="builtin" if use_builtin_browser else "default",
            headless=True,
            verbose=False,
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            extra_args=[
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
                "--disable-blink-features=AutomationControlled",
                "--ignore-certificate-errors",
                "--disable-extensions",
                "--disable-popup-blocking"
            ],
        )
    
    async def crawl(
        self,
        max_pages: int = 100,
        progress_callback: Optional[Callable] = None,
        mode: CrawlMode = CrawlMode.SIMPLE,
        max_depth: int = 5,
        max_pagination_pages: int = 20,
        keywords: Optional[List[str]] = None,
        enable_virtual_scroll: bool = False,
        virtual_scroll_selector: str = None,
        use_builtin_browser: bool = True,
        cancel_callback: Optional[Callable[[], bool]] = None
    ) -> CrawlResult:
        """
        Main crawl method with multiple modes
        
        Args:
            max_pages: Maximum number of pages to crawl
            progress_callback: Optional callback for progress updates (msg, current, total)
            mode: Crawl mode (SIMPLE, DEEP_BFS, DEEP_DFS, BEST_FIRST)
            max_depth: Maximum depth for deep crawl modes
            max_pagination_pages: Maximum number of pagination pages to crawl (for PAGINATION mode)
            keywords: Keywords for relevance scoring (Best-First mode)
            enable_virtual_scroll: Enable virtual scrolling for infinite scroll pages
            virtual_scroll_selector: CSS selector for scroll container
            use_builtin_browser: Use builtin browser mode for better performance
        
        Returns:
            CrawlResult with statistics
        """
        start_time = time.time()
        
        def update(msg, current=0, total=1):
            if progress_callback:
                progress_callback(msg, current, total)
        
        update(f"Starting {mode.value} crawl of: {self.start_url}")
        self._log(event="start", mode=mode.value, url=self.start_url)
        
        # Allow external cancellation
        if cancel_callback and cancel_callback():
            self.cancelled = True

        # Choose crawl method based on mode
        if mode == CrawlMode.SIMPLE:
            result = await self._simple_crawl(max_pages, update, enable_virtual_scroll, virtual_scroll_selector, use_builtin_browser)
        elif mode == CrawlMode.SITEMAP:
            result = await self._sitemap_crawl(max_pages, update, enable_virtual_scroll, virtual_scroll_selector, use_builtin_browser)
        elif mode == CrawlMode.PAGINATION:
            result = await self._pagination_crawl(max_pages, max_pagination_pages, update, enable_virtual_scroll, virtual_scroll_selector, use_builtin_browser)
        elif mode == CrawlMode.CATEGORY:
            result = await self._category_crawl(max_pages, max_depth, update, enable_virtual_scroll, virtual_scroll_selector, use_builtin_browser)
        elif mode in [CrawlMode.DEEP_BFS, CrawlMode.DEEP_DFS, CrawlMode.BEST_FIRST]:
            result = await self._deep_crawl(
                mode, max_pages, max_depth, keywords, update, 
                enable_virtual_scroll, virtual_scroll_selector, use_builtin_browser
            )
        else:
            result = await self._simple_crawl(max_pages, update, enable_virtual_scroll, virtual_scroll_selector, use_builtin_browser)
        
        result.duration = time.time() - start_time
        result.mode = mode.value
        self._log(event="finish", mode=mode.value, url=self.start_url, status="success" if result.success else "failed", info=f"saved={result.pages_saved} found={result.pages_found} duration={result.duration:.2f}s")
        
        # Save structured data in multiple formats
        if hasattr(result, 'structured_data') and result.structured_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_structured_data(result.structured_data, timestamp)
            self.generate_crawl_summary(result.structured_data)
        
        return result
    
    async def _simple_crawl(
        self,
        max_pages: int,
        update: Callable,
        enable_virtual_scroll: bool,
        virtual_scroll_selector: str,
        use_builtin_browser: bool
    ) -> CrawlResult:
        """
        Simple crawl: Discover links and crawl them
        """
        results = {"total": 0, "success": 0, "failed": []}
        discovered_urls = []
        structured_data = []  # For multi-format export
        
        # Configure browser using helper method
        browser_config = self._get_browser_config(use_builtin_browser)
        
        # Configure virtual scroll if enabled
        virtual_scroll_config = None
        if enable_virtual_scroll:
            virtual_scroll_config = VirtualScrollConfig(
                container_selector=virtual_scroll_selector or "body",
                scroll_count=50,
                scroll_by="container_height",
                wait_after_scroll=0.3
            )
        
        # Configure crawler with timeout settings
        crawl_config = CrawlerRunConfig(
            check_robots_txt=True,
            cache_mode=CacheMode.BYPASS,
            excluded_tags=["nav", "footer", "header", "aside", "form", "script", "style"],
            exclude_external_links=True,
            scraping_strategy=LXMLWebScrapingStrategy(),
            only_text=True,
            remove_forms=True,
            wait_until="networkidle",  # Wait for network to be idle instead of just DOM
            simulate_user=True,
            scan_full_page=True,
            verbose=False,
            screenshot=False,
            virtual_scroll_config=virtual_scroll_config,
            page_timeout=90000,  # 90 second timeout (increased from 60)
            delay_before_return_html=2.0  # Increased delay for stability (was 0.5)
        )
        
        # Use context manager for proper resource cleanup
        async with AsyncWebCrawler(config=browser_config) as crawler:
            try:
                # Step 1: Crawl starting URL
                update(f"Crawling starting URL: {self.start_url}")
                
                try:
                    result = await crawler.arun(url=self.start_url, config=crawl_config)
                    
                    if result.success:
                        filename = self._url_to_filename(self.start_url)
                        filepath = self.output_dir / filename
                        markdown_content = self._add_metadata(result.markdown.raw_markdown, self.start_url, result.status_code)
                        filepath.write_text(markdown_content, encoding='utf-8')
                        update(f"✓ Saved: {filename}", 1, max_pages)
                        self._log(event="saved", mode="simple", url=self.start_url, status=str(result.status_code))
                        results["success"] += 1
                        self.visited_urls.add(self.start_url)
                        
                        # Collect structured data
                        structured_data.append(self._extract_result_data(self.start_url, result))
                        
                        # Discover links
                        if result.html:
                            from bs4 import BeautifulSoup
                            soup = BeautifulSoup(result.html, 'html.parser')
                            for link in soup.find_all('a', href=True):
                                href = link['href']
                                full_url = urljoin(self.start_url, href)
                                if urlparse(full_url).netloc == self.base_domain:
                                    if not full_url.startswith(('javascript:', 'mailto:', '#')):
                                        full_url = full_url.split('#')[0]
                                        # Apply URL filtering
                                        if (full_url not in discovered_urls and 
                                            full_url not in self.visited_urls and
                                            not self._should_exclude_url(full_url)):
                                            discovered_urls.append(full_url)
                            
                            update(f"Discovered {len(discovered_urls)} links to crawl")
                    else:
                        results["failed"].append({"url": self.start_url, "error": result.error_message or "Unknown error"})
                        self._log(event="failed", mode="simple", url=self.start_url, status=str(result.status_code), info=result.error_message or "Unknown error")
                
                except Exception as e:
                    results["failed"].append({"url": self.start_url, "error": str(e)})
                    self._log(event="error", mode="simple", url=self.start_url, info=str(e))
                
                # Step 2: Crawl discovered links
                total_to_crawl = min(len(discovered_urls), max_pages - 1)
                for i, url in enumerate(discovered_urls[:max_pages - 1]):
                    if self.cancelled:
                        break
                    
                    if url in self.visited_urls:
                        continue
                    
                    # Yield control more frequently to prevent UI hanging
                    await asyncio.sleep(0)
                    
                    try:
                        current_index = results["success"] + 1
                        update(f"Crawling page {current_index}/{max_pages}: {url}", current_index, max_pages)
                        
                        result = await crawler.arun(url=url, config=crawl_config)
                        
                        if result.success:
                            filename = self._url_to_filename(url)
                            filepath = self.output_dir / filename
                            markdown_content = self._add_metadata(result.markdown.raw_markdown, url, result.status_code)
                            filepath.write_text(markdown_content, encoding='utf-8')
                            update(f"✓ Saved: {filename}", current_index, max_pages)
                            self._log(event="saved", mode="simple", url=url, status=str(result.status_code))
                            results["success"] += 1
                            self.visited_urls.add(url)
                            
                            # Collect structured data
                            structured_data.append(self._extract_result_data(url, result))
                        else:
                            results["failed"].append({"url": url, "error": result.error_message or "Unknown error"})
                            self._log(event="failed", mode="simple", url=url, status=str(result.status_code), info=result.error_message or "Unknown error")
                            # Still collect data for failed pages
                            structured_data.append(self._extract_result_data(url, result))
                    
                    except Exception as e:
                        results["failed"].append({"url": url, "error": str(e)})
                        self._log(event="error", mode="simple", url=url, info=str(e))
                    
                    # Yield control to allow UI updates
                    await asyncio.sleep(0.1)
            
            except Exception as e:
                # Catch any unexpected errors in the entire crawl process
                self._log(event="critical_error", mode="simple", info=str(e))
                results["failed"].append({"url": "crawl_process", "error": str(e)})
        
        return CrawlResult(
            success=results["success"] > 0,
            pages_found=len(discovered_urls) + 1,
            pages_saved=results["success"],
            message=f"Crawled {results['success']} pages successfully",
            failed_urls=results["failed"],
            structured_data=structured_data
        )
    
    async def _deep_crawl(
        self,
        mode: CrawlMode,
        max_pages: int,
        max_depth: int,
        keywords: Optional[List[str]],
        update: Callable,
        enable_virtual_scroll: bool,
        virtual_scroll_selector: str,
        use_builtin_browser: bool
    ) -> CrawlResult:
        """
        Deep crawl with BFS, DFS, or Best-First strategy
        """
        update(f"Starting {mode.value} deep crawl (depth={max_depth}, max_pages={max_pages})")
        
        # Configure browser using helper method
        browser_config = self._get_browser_config(use_builtin_browser)
        
        # Configure filters
        filter_chain = FilterChain([
            DomainFilter(allowed_domains=[self.base_domain]),
            ContentTypeFilter(allowed_types=["text/html"])
        ])
        
        # Configure deep crawl strategy
        if mode == CrawlMode.DEEP_BFS:
            strategy = BFSDeepCrawlStrategy(
                max_depth=max_depth,
                include_external=False,
                filter_chain=filter_chain,
                max_pages=max_pages
            )
        elif mode == CrawlMode.DEEP_DFS:
            strategy = DFSDeepCrawlStrategy(
                max_depth=max_depth,
                include_external=False,
                filter_chain=filter_chain,
                max_pages=max_pages
            )
        else:  # BEST_FIRST
            scorer = None
            if keywords:
                scorer = KeywordRelevanceScorer(keywords=keywords, weight=1.0)
                update(f"Using keyword scorer with keywords: {', '.join(keywords)}")
            else:
                update("⚠️ No keywords provided - using default scoring")
            
            strategy = BestFirstCrawlingStrategy(
                max_depth=max_depth,
                include_external=False,
                filter_chain=filter_chain,
                url_scorer=scorer,
                max_pages=max_pages
            )
        
        # Configure virtual scroll if enabled
        virtual_scroll_config = None
        if enable_virtual_scroll:
            virtual_scroll_config = VirtualScrollConfig(
                container_selector=virtual_scroll_selector or "body",
                scroll_count=50,
                scroll_by="container_height",
                wait_after_scroll=0.3
            )
        
        # Configure crawler with robust error handling settings
        crawl_config = CrawlerRunConfig(
            deep_crawl_strategy=strategy,
            scraping_strategy=LXMLWebScrapingStrategy(),
            cache_mode=CacheMode.BYPASS,
            excluded_tags=["nav", "footer", "header", "aside", "form", "script", "style"],
            verbose=False,
            stream=True,  # Stream results as they come
            virtual_scroll_config=virtual_scroll_config,
            page_timeout=90000,  # 90 second timeout (increased from 60)
            wait_until="networkidle",  # Wait for network to be idle instead of just DOM
            simulate_user=True,
            magic=True  # Enable anti-detection features
        )
        
        results_list = []
        failed = []
        structured_data = []  # For multi-format export
        
        # Use async context manager for proper resource cleanup
        async with AsyncWebCrawler(config=browser_config) as crawler:
            try:
                # Execute deep crawl with streaming and robust error handling
                current = 0
                consecutive_failures = 0
                max_consecutive_failures = 5
                
                try:
                    async for result in await crawler.arun(url=self.start_url, config=crawl_config):
                        # Yield control to prevent UI blocking
                        await asyncio.sleep(0)
                        
                        if self.cancelled:
                            update("Crawl cancelled by user", current, max_pages)
                            break
                        
                        current += 1
                        depth = result.metadata.get("depth", 0) if result.metadata else 0
                        score = result.metadata.get("score", 0) if result.metadata else 0
                        
                        try:
                            if result.success:
                                filename = self._url_to_filename(result.url)
                                filepath = self.output_dir / filename
                                markdown_content = self._add_metadata(result.markdown.raw_markdown, result.url, result.status_code)
                                filepath.write_text(markdown_content, encoding='utf-8')
                                
                                if mode == CrawlMode.BEST_FIRST and score > 0:
                                    update(f"✓ Saved (depth={depth}, score={score:.2f}): {filename}", current, max_pages)
                                else:
                                    update(f"✓ Saved (depth={depth}): {filename}", current, max_pages)
                                
                                results_list.append(result)
                                consecutive_failures = 0  # Reset failure counter on success
                                
                                # Collect structured data
                                structured_data.append(self._extract_result_data(result.url, result))
                                self._log(event="saved", mode=mode.value, url=result.url, status=str(result.status_code))
                            else:
                                error_msg = result.error_message or "Unknown error"
                                failed.append({"url": result.url, "error": error_msg})
                                update(f"✗ Failed (depth={depth}): {result.url} - {error_msg}", current, max_pages)
                                consecutive_failures += 1
                                self._log(event="failed", mode=mode.value, url=result.url, status=str(result.status_code), info=error_msg)
                                
                                # Still collect data for failed pages
                                structured_data.append(self._extract_result_data(result.url, result))
                                
                                # If too many consecutive failures, stop
                                if consecutive_failures >= max_consecutive_failures:
                                    update(f"⚠️ Stopping: {max_consecutive_failures} consecutive failures", current, max_pages)
                                    break
                        
                        except Exception as e:
                            error_msg = f"Error processing result: {str(e)}"
                            failed.append({"url": result.url if hasattr(result, 'url') else 'unknown', "error": error_msg})
                            update(f"✗ Processing error: {error_msg}", current, max_pages)
                            consecutive_failures += 1
                            self._log(event="error", mode=mode.value, url=result.url if hasattr(result, 'url') else 'unknown', info=error_msg)
                            
                            if consecutive_failures >= max_consecutive_failures:
                                update(f"⚠️ Stopping: {max_consecutive_failures} consecutive failures", current, max_pages)
                                break
                        
                        # Add small delay to avoid overwhelming the server
                        await asyncio.sleep(0.5)
                
                except RuntimeError as e:
                    # Handle navigation errors gracefully
                    error_msg = str(e)
                    if "Failed on navigating" in error_msg or "ACS-GOTO" in error_msg or "Connection closed" in error_msg:
                        update(f"⚠️ Navigation/Connection error: {error_msg[:100]}", current, max_pages)
                        update("Tip: This may be due to network issues, rate limiting, or site protection", current, max_pages)
                    else:
                        raise  # Re-raise if it's a different RuntimeError
                
                except Exception as e:
                    # Catch any other errors during crawling
                    error_msg = str(e)
                    if "Connection closed" in error_msg:
                        update(f"⚠️ Browser connection lost: {error_msg[:100]}", current, max_pages)
                    else:
                        update(f"⚠️ Crawl error: {error_msg}", current, max_pages)
            
            except Exception as e:
                # Catch top-level errors
                self._log(event="critical_error", mode=mode.value, info=str(e))
        
        return CrawlResult(
            success=len(results_list) > 0,
            pages_found=len(results_list) + len(failed),
            pages_saved=len(results_list),
            message=f"Deep crawl ({mode.value}): {len(results_list)} pages saved",
            failed_urls=failed,
            structured_data=structured_data
        )
    
    def _url_to_filename(self, url: str) -> str:
        """Convert URL to a safe filename"""
        parsed = urlparse(url)
        path = parsed.path.strip('/')
        
        if not path:
            path = parsed.netloc.replace('.', '_')
        else:
            path = path.replace('.html', '').replace('.php', '').replace('.aspx', '')
            path = path.replace('/', '-')
            path = re.sub(r'[^\w\-]', '', path)
        
        if len(path) > 200:
            path = path[:200]
        
        return f"{path}.md"
    
    def _add_metadata(self, markdown: str, url: str, status_code: int) -> str:
        """Add metadata header to markdown content"""
        from datetime import datetime
        
        metadata = f"""# {url}

Status: {status_code}
Crawled: {datetime.now().isoformat()}
---

"""
        content = markdown
        if self.strip_links and content:
            try:
                # Remove markdown links [text](url) -> text
                content = re.sub(r"\[([^\]]+)\]\((?:https?:\/\/|www\.)[^)]+\)", r"\1", content)
                # Remove bare URLs
                content = re.sub(r"(?i)\b(?:https?:\/\/|www\.)\S+", "", content)
            except Exception:
                # Fallback silently if regex fails
                pass
        return metadata + content
    
    async def _category_crawl(
        self,
        max_pages: int,
        max_depth: int,
        update: Callable,
        enable_virtual_scroll: bool,
        virtual_scroll_selector: str,
        use_builtin_browser: bool
    ) -> CrawlResult:
        """
        Category/topic-focused deep crawl
        Stays within the specified category/topic path
        """
        update("Starting category-focused crawl...")
        
        # Extract the focused path from the URL
        focused_path = self.parsed_url.path
        
        # Detect if URL contains category/topic indicators
        topic_indicators = ['/articles/', '/posts/', '/blog/', '/news/', '/category/', '/reports/', '/topics/', '/tag/']
        is_category_url = any(indicator in focused_path.lower() for indicator in topic_indicators)
        
        if is_category_url:
            update(f"  ✓ Detected category/topic-specific URL: {focused_path}")
        else:
            update(f"  Using focused path: {focused_path}")
        
        try:
            # Build URL pattern filters based on focused path
            url_patterns = [
                f"*{self.base_domain}{focused_path}*",
                f"*{focused_path}/p[0-9]*",
                f"*{focused_path}?*page=*",
            ]
            
            update(f"  Filtering to URLs matching: {focused_path}")
            
            # Configure filters
            filter_chain = FilterChain([
                DomainFilter(allowed_domains=[self.base_domain]),
                URLPatternFilter(patterns=url_patterns),
                ContentTypeFilter(allowed_types=["text/html"])
            ])
            
            # Configure markdown generation with pruning filter (Canary Media style)
            markdown_generator = DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(
                    threshold=0.48,  # Relevance threshold for content filtering
                    threshold_type="dynamic",  # Adaptive threshold
                ),
                options={
                    "ignore_links": False,
                    "citations": True
                }
            )
            
            # Configure browser
            browser_config = BrowserConfig(
                browser_mode="builtin" if use_builtin_browser else "default",
                headless=True,
                verbose=False,
                extra_args=[
                    "--disable-gpu", 
                    "--disable-dev-shm-usage", 
                    "--no-sandbox",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-web-security"
                ],
            )
            
            # Use BFS strategy for category crawling (depth 5 like Canary Media)
            strategy = BFSDeepCrawlStrategy(
                max_depth=max_depth,
                include_external=False,
                filter_chain=filter_chain,
                max_pages=max_pages
            )
            
            # Configure virtual scroll if enabled
            virtual_scroll_config = None
            if enable_virtual_scroll:
                virtual_scroll_config = VirtualScrollConfig(
                    container_selector=virtual_scroll_selector or "body",
                    scroll_count=50,
                    scroll_by="container_height",
                    wait_after_scroll=0.3
                )
            
            # Configure crawler with content pruning (Canary Media optimization)
            crawl_config = CrawlerRunConfig(
                deep_crawl_strategy=strategy,
                scraping_strategy=LXMLWebScrapingStrategy(),
                markdown_generator=markdown_generator,  # Add pruning through markdown generator
                cache_mode=CacheMode.BYPASS,
                excluded_tags=["nav", "footer", "header", "aside", "form", "script", "style"],
                exclude_external_links=True,  # Remove external links from content
                verbose=False,
                stream=True,
                virtual_scroll_config=virtual_scroll_config,
                page_timeout=90000,  # 90 second timeout (increased from 60)
                wait_until="networkidle",  # Wait for network to be idle instead of just DOM
                simulate_user=True,
                magic=True
            )
            
            results_list = []
            failed = []
            structured_data = []  # For multi-format export
            
            crawler = AsyncWebCrawler(config=browser_config)
            
            try:
                if not use_builtin_browser:
                    await crawler.start()
                
                # Execute category-focused crawl
                current = 0
                consecutive_failures = 0
                max_consecutive_failures = 5
                
                try:
                    async for result in await crawler.arun(url=self.start_url, config=crawl_config):
                        # Yield control to prevent UI blocking
                        await asyncio.sleep(0)
                        
                        if self.cancelled:
                            update("Crawl cancelled by user", current, max_pages)
                            break
                        
                        current += 1
                        depth = result.metadata.get("depth", 0) if result.metadata else 0
                        
                        try:
                            if result.success:
                                filename = self._url_to_filename(result.url)
                                filepath = self.output_dir / filename
                                # Use filtered markdown if available (Canary Media style pruning)
                                content = result.markdown.fit_markdown or result.markdown.raw_markdown
                                # Only save if content meets minimum threshold
                                if content and len(content.strip()) > 100:
                                    markdown_content = self._add_metadata(content, result.url, result.status_code)
                                    filepath.write_text(markdown_content, encoding='utf-8')
                                    update(f"✓ Saved (depth={depth}): {filename}", current, max_pages)
                                    results_list.append(result)
                                    consecutive_failures = 0
                                    structured_data.append(self._extract_result_data(result.url, result))
                                else:
                                    update(f"⚠️ Skipped (insufficient content): {filename}", current, max_pages)
                                    consecutive_failures += 1
                                    structured_data.append(self._extract_result_data(result.url, result))
                            else:
                                error_msg = result.error_message or "Unknown error"
                                failed.append({"url": result.url, "error": error_msg})
                                update(f"✗ Failed (depth={depth}): {result.url}", current, max_pages)
                                consecutive_failures += 1
                                self._log(event="failed", mode="category", url=result.url, status=str(getattr(result, 'status_code', '')), info=error_msg)
                                if consecutive_failures >= max_consecutive_failures:
                                    update(f"⚠️ Stopping: {max_consecutive_failures} consecutive failures", current, max_pages)
                                    break
                        except Exception as e:
                            error_msg = f"Error processing result: {str(e)}"
                            failed.append({"url": result.url if hasattr(result, 'url') else 'unknown', "error": error_msg})
                            update(f"✗ Processing error: {error_msg}", current, max_pages)
                            consecutive_failures += 1
                            self._log(event="error", mode="category", url=result.url if hasattr(result, 'url') else 'unknown', info=error_msg)
                            if consecutive_failures >= max_consecutive_failures:
                                update(f"⚠️ Stopping: {max_consecutive_failures} consecutive failures", current, max_pages)
                                break
                        
                        await asyncio.sleep(0.5)
                
                except RuntimeError as e:
                    error_msg = str(e)
                    if "Failed on navigating" in error_msg or "ACS-GOTO" in error_msg:
                        update(f"⚠️ Navigation error: {error_msg}", current, max_pages)
                        update("Tip: This may be due to network issues or rate limiting", current, max_pages)
                    else:
                        raise
                
                except Exception as e:
                    update(f"⚠️ Crawl error: {str(e)}", current, max_pages)
            
            finally:
                if not use_builtin_browser:
                    await crawler.close()
            
            return CrawlResult(
                success=len(results_list) > 0,
                pages_found=len(results_list),
                pages_saved=len(results_list),
                message=f"Category-focused crawl: {len(results_list)} pages saved from {focused_path}",
                failed_urls=failed,
                structured_data=structured_data
            )
        
        except Exception as e:
            return CrawlResult(
                success=False,
                pages_found=0,
                pages_saved=0,
                message=f"Category crawl failed: {str(e)}",
                failed_urls=[]
            )
    
    async def _detect_pagination_pattern(self, update: Callable) -> tuple[str, str, str]:
        """
        Dynamically detect the correct pagination pattern by testing different formats
        Returns: (pagination_type, pagination_pattern, base_list_url)
        """
        import httpx
        
        # Extract base URL
        if '?' in self.start_url:
            base_url = self.start_url.split('?')[0]
        else:
            base_url = self.start_url.rstrip('/')
        
        # Remove existing pagination from URL if present
        base_url = re.sub(r'/(page|p)\d+/?$', '', base_url)
        base_url = re.sub(r'/p\d+/?$', '', base_url)
        
        update(f"  Testing pagination patterns for: {base_url}")
        
        # Test different pagination patterns
        test_patterns = [
            # Query-based patterns
            ('query', '?page=', f"{base_url}?page=2"),
            ('query', '?p=', f"{base_url}?p=2"),
            ('query', '?_page=', f"{base_url}?_page=2"),
            
            # Path-based patterns
            ('path', '/page/', f"{base_url}/page/2/"),
            ('path', '/p/', f"{base_url}/p/2/"),
            ('path', '/p', f"{base_url}/p2"),  # Quaise.com style
            ('path', '/', f"{base_url}/2/"),
        ]
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for pagination_type, pattern, test_url in test_patterns:
                try:
                    update(f"    Testing: {test_url}")
                    response = await client.get(test_url)
                    
                    if response.status_code == 200:
                        update(f"    ✅ Found working pattern: {pattern} (Status: {response.status_code})")
                        return pagination_type, pattern, base_url
                    else:
                        update(f"    ❌ Failed: {pattern} (Status: {response.status_code})")
                        
                except Exception as e:
                    update(f"    ❌ Error testing {pattern}: {str(e)}")
                    continue
        
        # Fallback to default if no pattern works
        update("    ⚠️ No working pagination pattern found, using default /page/")
        return 'path', '/page/', base_url
    
    async def _pagination_crawl(
        self,
        max_pages: int,
        max_pagination_pages: int,
        update: Callable,
        enable_virtual_scroll: bool,
        virtual_scroll_selector: str,
        use_builtin_browser: bool
    ) -> CrawlResult:
        """
        Two-phase pagination crawl
        Phase 1: Extract article URLs from pagination pages
        Phase 2: Crawl discovered articles
        
        Args:
            max_pages: Maximum number of article pages to crawl
            max_pagination_pages: Maximum number of pagination pages to check
            update: Progress callback function
            enable_virtual_scroll: Enable virtual scrolling
            virtual_scroll_selector: CSS selector for virtual scroll
            use_builtin_browser: Use builtin browser mode
        """
        update("Starting pagination-based crawl...")
        
        try:
            # Dynamically detect pagination pattern by testing different formats
            pagination_type, pagination_pattern, base_list_url = await self._detect_pagination_pattern(update)
            
            update(f"  Detected {pagination_type}-based pagination: {pagination_pattern}")
            update(f"  Base URL: {base_list_url}")
            update(f"  Phase 1: Extracting article URLs from up to {max_pagination_pages} pagination pages")
            
            # Phase 1: Extract article URLs
            all_article_urls = set()
            
            # Configure browser
            browser_config = BrowserConfig(
                browser_mode="builtin" if use_builtin_browser else "default",
                headless=True,
                verbose=False,
                extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
            )
            
            crawl_config = CrawlerRunConfig(
                scraping_strategy=LXMLWebScrapingStrategy(),
                cache_mode=CacheMode.BYPASS,
                verbose=False,
                page_timeout=60000,  # 60 second timeout
                wait_until="domcontentloaded",  # Wait for DOM content to be loaded
                simulate_user=True,
                delay_before_return_html=0.5
            )
            
            crawler = AsyncWebCrawler(config=browser_config)
            
            try:
                if not use_builtin_browser:
                    await crawler.start()
                
                # Crawl pagination pages (up to max_pagination_pages)
                for page_num in range(1, max_pagination_pages + 1):
                    # Generate pagination URL based on detected pattern
                    if pagination_type == 'query':
                        page_url = f"{base_list_url}{pagination_pattern}{page_num}"
                    else:  # path-based
                        # Handle different path patterns
                        if pagination_pattern == '/p':
                            # Quaise.com style: /p2, /p3 (no trailing slash)
                            page_url = f"{base_list_url}{pagination_pattern}{page_num}"
                        elif pagination_pattern == '/':
                            # Numeric only: /2/, /3/
                            page_url = f"{base_list_url}{pagination_pattern}{page_num}/"
                        else:
                            # Standard style: /page/2/, /p/2/
                            page_url = f"{base_list_url}{pagination_pattern}{page_num}/"
                    
                    update(f"    Extracting from page {page_num}: {page_url}")
                    
                    result = await crawler.arun(page_url, config=crawl_config)
                    
                    if result.success and result.html:
                        soup = BeautifulSoup(result.html, 'html.parser')
                        
                        for link in soup.find_all('a', href=True):
                            href = link['href']
                            full_url = urljoin(page_url, href)
                            
                            # Filter out pagination URLs based on detected pattern
                            is_pagination_url = False
                            
                            if pagination_type == 'query':
                                is_pagination_url = pagination_pattern in full_url
                            else:  # path-based
                                if pagination_pattern == '/p':
                                    # Quaise.com style: /p2, /p3
                                    is_pagination_url = re.search(r'/p\d+/?$', full_url) is not None
                                elif pagination_pattern == '/':
                                    # Numeric only: /2/, /3/
                                    is_pagination_url = re.search(r'/\d+/?$', full_url) is not None
                                else:
                                    # Standard style: /page/2/, /p/2/
                                    is_pagination_url = (
                                        pagination_pattern in full_url or
                                        full_url.rstrip('/').endswith(pagination_pattern.rstrip('/'))
                                    )
                            
                            # Only include URLs from same domain that look like articles
                            if (full_url.startswith(self.base_url) and
                                not is_pagination_url and
                                not self._should_exclude_url(full_url)):
                                all_article_urls.add(full_url)
                    
                    await asyncio.sleep(1)  # Rate limiting
            
            finally:
                if not use_builtin_browser:
                    await crawler.close()
            
            article_list = list(all_article_urls)[:max_pages]
            update(f"  Discovered {len(all_article_urls)} unique article URLs")
            update(f"  Phase 2: Crawling {len(article_list)} articles (limited to max_pages)")
            
            # Phase 2: Crawl the discovered articles
            saved_count, structured_data = await self._crawl_url_batch(
                article_list, max_pages, update,
                enable_virtual_scroll, virtual_scroll_selector, use_builtin_browser
            )
            
            return CrawlResult(
                success=saved_count > 0,
                pages_found=len(article_list),
                pages_saved=saved_count,
                message=f"Pagination crawl ({pagination_type}-based): {saved_count}/{len(article_list)} articles saved",
                failed_urls=[],
                structured_data=structured_data
            )
        
        except Exception as e:
            return CrawlResult(
                success=False,
                pages_found=0,
                pages_saved=0,
                message=f"Pagination crawl failed: {str(e)}",
                failed_urls=[]
            )
    
    async def _sitemap_crawl(
        self,
        max_pages: int,
        update: Callable,
        enable_virtual_scroll: bool,
        virtual_scroll_selector: str,
        use_builtin_browser: bool
    ) -> CrawlResult:
        """
        Sitemap-based crawl with numbered sitemap discovery
        Prioritizes highest-numbered post-sitemaps first
        """
        update("Starting sitemap discovery...")
        
        # Discover sitemaps
        sitemap_urls = await self._discover_sitemaps(update)
        
        if not sitemap_urls:
            update("⚠️ No sitemaps found")
            return CrawlResult(
                success=False,
                pages_found=0,
                pages_saved=0,
                message="No sitemaps found",
                failed_urls=[]
            )
        
        # Determine if using numbered sitemaps
        numbered_sitemaps = [s for s in sitemap_urls if re.search(r'post-sitemap\d+\.xml', s)]
        
        if numbered_sitemaps:
            highest_num = max([int(re.search(r'post-sitemap(\d+)\.xml', s).group(1)) for s in numbered_sitemaps])
            update(f"🔍 STRATEGY: Numbered Post-Sitemap Crawl (found {len(numbered_sitemaps)} sitemaps, highest: {highest_num})")
        else:
            update(f"🔍 STRATEGY: Standard Sitemap-Based Crawl")
        
        update(f"Starting sitemap crawl with {len(sitemap_urls)} sitemap(s)...")
        
        try:
            # Fetch all URLs from sitemaps
            all_urls = []
            is_post_sitemap_crawl = False
            
            for idx, sitemap_url in enumerate(sitemap_urls, 1):
                sitemap_name = sitemap_url.split('/')[-1]
                update(f"  [{idx}/{len(sitemap_urls)}] Fetching: {sitemap_name}")
                urls = await self._fetch_sitemap_urls(sitemap_url)
                
                # Check if this is a post-sitemap
                if 'post-sitemap' in sitemap_name.lower():
                    is_post_sitemap_crawl = True
                
                all_urls.extend(urls)
                update(f"  ✓ Loaded {len(urls)} URLs from {sitemap_name}")
            
            update(f"  Total found: {len(all_urls)} URLs from all sitemap(s)")
            
            # For numbered sitemaps, URLs are already in correct order
            # (highest numbered sitemap processed first = newest posts first)
            if is_post_sitemap_crawl and numbered_sitemaps:
                update(f"  ✓ URLs ordered by sitemap number (highest/newest first)")
            
            # Filter unwanted URLs using centralized filtering
            filtered_urls = []
            for url in all_urls:
                url_str = url if isinstance(url, str) else url.get("url", "")
                if not self._should_exclude_url(url_str):
                    filtered_urls.append(url_str)
            
            update(f"  After filtering: {len(filtered_urls)} URLs to crawl")
            
            # Limit to max_pages
            filtered_urls = filtered_urls[:max_pages]
            update(f"  Limiting to max_pages: {len(filtered_urls)} URLs will be crawled")
            
            # Crawl URLs
            saved_count, structured_data = await self._crawl_url_batch(
                filtered_urls, max_pages, update, 
                enable_virtual_scroll, virtual_scroll_selector, use_builtin_browser
            )
            for d in structured_data:
                try:
                    self._log(event="saved", mode="sitemap", url=d.get('url', ''), status=str(d.get('status_code', '')))
                except Exception:
                    pass
            
            strategy_name = "Numbered Post-Sitemap" if numbered_sitemaps else "Standard Sitemap"
            
            return CrawlResult(
                success=saved_count > 0,
                pages_found=len(filtered_urls),
                pages_saved=saved_count,
                message=f"{strategy_name} crawl: {saved_count}/{len(filtered_urls)} pages saved from {len(sitemap_urls)} sitemap(s)",
                failed_urls=[],
                structured_data=structured_data
            )
        
        except Exception as e:
            return CrawlResult(
                success=False,
                pages_found=0,
                pages_saved=0,
                message=f"Sitemap crawl failed: {str(e)}",
                failed_urls=[]
            )
    
    async def _discover_sitemaps(self, update: Callable) -> List[str]:
        """
        Discover sitemaps with prioritization:
        1. Numbered content sitemaps (news, article, post) - highest first
        2. Non-numbered content sitemaps (only if no numbered ones found)
        3. Standard sitemap.xml and sitemap_index.xml
        """
        sitemap_urls = []
        
        # Check for numbered content sitemaps (news, article, post)
        content_keywords = ["post", "news", "article", "blog", "content"]
        update("  Checking for numbered content sitemaps...")
        
        for keyword in content_keywords:
            numbered_sitemaps = await self._discover_numbered_sitemaps(f"{keyword}-sitemap", update)
            if numbered_sitemaps:
                sitemap_urls.extend(numbered_sitemaps)
                update(f"  ✓ Found {len(numbered_sitemaps)} numbered {keyword}-sitemap(s)")
        
        # Check for non-numbered content sitemaps ONLY if no numbered ones found
        if not sitemap_urls:
            update("  Checking for non-numbered content sitemaps...")
            headers = {"User-Agent": "Mozilla/5.0"}
            
            for keyword in content_keywords:
                try:
                    non_numbered_url = f"{self.base_url}/{keyword}-sitemap.xml"
                    response = requests.get(non_numbered_url, headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        try:
                            content = response.content
                            if non_numbered_url.endswith('.gz'):
                                content = gzip.decompress(content)
                            ET.fromstring(content)
                            sitemap_urls.append(non_numbered_url)
                            update(f"  ✓ Found non-numbered {keyword}-sitemap.xml")
                        except ET.ParseError:
                            pass
                except Exception:
                    pass
        
        # If no content sitemaps found, check standard sitemaps
        if not sitemap_urls:
            update("  Checking for standard sitemaps...")
            standard_sitemaps = [
                f"{self.base_url}/sitemap_index.xml",
                f"{self.base_url}/sitemap.xml"
            ]
            
            for sitemap_url in standard_sitemaps:
                try:
                    response = requests.get(sitemap_url, headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        try:
                            content = response.content
                            if sitemap_url.endswith('.gz'):
                                content = gzip.decompress(content)
                            ET.fromstring(content)
                            sitemap_urls.append(sitemap_url)
                            update(f"  ✓ Found {sitemap_url.split('/')[-1]}")
                        except ET.ParseError:
                            pass
                except Exception:
                    pass
        
        return sitemap_urls
    
    async def _discover_numbered_sitemaps(self, base_name: str, update: Callable, max_check: int = 100) -> List[str]:
        """
        Discover numbered sitemaps (e.g., post-sitemap1.xml, post-sitemap2.xml, ...)
        Returns list sorted by number (highest first)
        """
        found_sitemaps = []
        headers = {"User-Agent": "Mozilla/5.0"}
        
        for i in range(1, max_check + 1):
            sitemap_url = f"{self.base_url}/{base_name}{i}.xml"
            
            try:
                response = requests.get(sitemap_url, headers=headers, timeout=5)
                
                if response.status_code == 200:
                    try:
                        content = response.content
                        if sitemap_url.endswith('.gz'):
                            content = gzip.decompress(content)
                        ET.fromstring(content)
                        found_sitemaps.append((i, sitemap_url))
                    except ET.ParseError:
                        continue
                elif response.status_code == 404:
                    # Continue checking a bit more in case of gaps
                    if len(found_sitemaps) > 0 and i > max(s[0] for s in found_sitemaps) + 5:
                        break
            except Exception:
                continue
        
        if found_sitemaps:
            # Sort by number (highest first) and return URLs
            found_sitemaps.sort(key=lambda x: x[0], reverse=True)
            urls = [url for _, url in found_sitemaps]
            return urls
        
        return []
    
    async def _fetch_sitemap_urls(self, sitemap_url: str) -> List[str]:
        """Fetch URLs from a sitemap XML file"""
        urls = []
        
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(sitemap_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                content = response.content
                if sitemap_url.endswith('.gz'):
                    content = gzip.decompress(content)
                
                root = ET.fromstring(content)
                
                # Handle sitemap index
                namespaces = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
                
                # Check if it's a sitemap index
                sitemaps = root.findall(".//sm:sitemap/sm:loc", namespaces)
                if sitemaps:
                    # Recursively fetch from child sitemaps
                    for sitemap in sitemaps:
                        child_urls = await self._fetch_sitemap_urls(sitemap.text)
                        urls.extend(child_urls)
                else:
                    # Regular sitemap with URLs
                    url_elements = root.findall(".//sm:url/sm:loc", namespaces)
                    for url_elem in url_elements:
                        urls.append(url_elem.text)
        
        except Exception as e:
            print(f"Error fetching sitemap {sitemap_url}: {e}")
        
        return urls
    
    async def _crawl_url_batch(
        self,
        urls: List[str],
        max_pages: int,
        update: Callable,
        enable_virtual_scroll: bool,
        virtual_scroll_selector: str,
        use_builtin_browser: bool
    ) -> tuple[int, List[Dict]]:
        """Crawl a batch of URLs and return count of saved pages and structured data"""
        saved_count = 0
        structured_data = []
        
        # Configure browser
        browser_config = BrowserConfig(
            browser_mode="builtin" if use_builtin_browser else "default",
            headless=True,
            verbose=False,
            extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
        )
        
        # Configure virtual scroll if enabled
        virtual_scroll_config = None
        if enable_virtual_scroll:
            virtual_scroll_config = VirtualScrollConfig(
                container_selector=virtual_scroll_selector or "body",
                scroll_count=50,
                scroll_by="container_height",
                wait_after_scroll=0.3
            )
        
        # Configure crawler
        crawl_config = CrawlerRunConfig(
            check_robots_txt=True,
            cache_mode=CacheMode.BYPASS,
            excluded_tags=["nav", "footer", "header", "aside", "form", "script", "style"],
            exclude_external_links=True,
            scraping_strategy=LXMLWebScrapingStrategy(),
            only_text=True,
            remove_forms=True,
            wait_until="networkidle",  # Wait for network to be idle instead of just DOM
            simulate_user=True,
            scan_full_page=True,
            verbose=False,
            screenshot=False,
            virtual_scroll_config=virtual_scroll_config,
            page_timeout=90000,  # 90 second timeout
            delay_before_return_html=2.0  # Increased delay for stability
        )
        
        crawler = AsyncWebCrawler(config=browser_config)
        
        try:
            if not use_builtin_browser:
                await crawler.start()
            
            for i, url in enumerate(urls):
                # Yield control to prevent UI blocking
                await asyncio.sleep(0)
                
                if self.cancelled:
                    break
                
                if url in self.visited_urls:
                    continue
                
                try:
                    current = i + 1
                    update(f"Crawling page {current}/{len(urls)}: {url}", current, max_pages)
                    
                    result = await crawler.arun(url=url, config=crawl_config)
                    
                    if result.success:
                        filename = self._url_to_filename(url)
                        filepath = self.output_dir / filename
                        markdown_content = self._add_metadata(result.markdown.raw_markdown, url, result.status_code)
                        filepath.write_text(markdown_content, encoding='utf-8')
                        update(f"✓ Saved: {filename}", current, max_pages)
                        saved_count += 1
                        self.visited_urls.add(url)
                        
                        # Collect structured data
                        structured_data.append(self._extract_result_data(url, result))
                    else:
                        update(f"✗ Failed: {url}", current, max_pages)
                        # Still collect data for failed pages
                        structured_data.append(self._extract_result_data(url, result))
                
                except Exception as e:
                    update(f"Error crawling {url}: {str(e)}", current, max_pages)
                
                await asyncio.sleep(0.5)
        
        finally:
            if not use_builtin_browser:
                await crawler.close()
        
        return saved_count, structured_data
    
    def _extract_result_data(self, url: str, result, title: str = "") -> Dict[str, Any]:
        """Extract relevant data from CrawlResult for multi-format export."""
        # Create filename from URL
        filename = self._url_to_filename(url)
        if not filename.endswith('.md'):
            filename += '.md'
        
        # Get markdown content
        markdown_content = result.markdown.raw_markdown if result.markdown else ""
        
        return {
            "file_name": filename,
            "content": markdown_content,
            "url": url,
            "title": title or self._extract_title_from_html(result.html) if result.html else "",
            "status": "success" if result.success else "failed",
            "status_code": result.status_code,
            "timestamp": datetime.now().isoformat(),
            "error_message": result.error_message if hasattr(result, 'error_message') else None
        }
    
    def _extract_title_from_html(self, html: str) -> str:
        """Extract title from HTML."""
        if not html:
            return "No Title"
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            title_tag = soup.find('title')
            if title_tag:
                return title_tag.get_text().strip()
        except Exception:
            pass
        
        # Fallback regex extraction
        match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return "No Title"
    
    def _create_safe_filename(self, url: str, idx: int = None) -> str:
        """Create a safe filename from URL."""
        # Remove protocol and clean URL
        safe_name = re.sub(r'https?://', '', url)
        safe_name = re.sub(r'[^\w\-_.]', '_', safe_name)
        safe_name = safe_name[:100]  # Limit length
        
        if idx is not None:
            return f"{idx:03d}_{safe_name}"
        return safe_name
    
    def save_structured_data(self, structured_data: List[Dict], timestamp: str = None):
        """Save structured data in CSV and JSON formats."""
        if not structured_data:
            return
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create subdirectories for different formats
        csv_dir = self.output_dir / "csv"
        json_dir = self.output_dir / "json"
        csv_dir.mkdir(exist_ok=True)
        json_dir.mkdir(exist_ok=True)
        
        # Save CSV
        self._save_csv(structured_data, csv_dir, timestamp)
        
        # Save JSON
        self._save_json(structured_data, json_dir, timestamp)
    
    def _save_csv(self, structured_data: List[Dict], output_dir: Path, timestamp: str):
        """Save structured data to CSV file."""
        filename = f"crawl_results_{timestamp}.csv"
        filepath = output_dir / filename
        
        # Define CSV columns - simplified structure
        csv_columns = [
            "file_name", "content", "url", "title", 
            "status", "status_code", "timestamp", "error_message"
        ]
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(structured_data)
        
        print(f"✓ CSV saved: {filepath}")
        
        # Upload to S3 if available
        if HAS_S3_STORAGE:
            try:
                storage = get_storage()
                # Extract domain from start_url for S3 key
                domain = urlparse(self.start_url).netloc.replace('www.', '').replace('.', '_')
                s3_key = f"crawled_data/{domain}_{timestamp}.csv"
                storage.upload_file(str(filepath), s3_key)
                print(f"✓ CSV uploaded to S3: {s3_key}")
            except Exception as e:
                print(f"⚠️ Failed to upload CSV to S3: {e}")
    
    def _save_json(self, structured_data: List[Dict], output_dir: Path, timestamp: str, include_content: bool = True):
        """Save structured data to JSON file."""
        filename = f"crawl_results_{timestamp}.json"
        filepath = output_dir / filename
        
        # Optionally filter out content for smaller JSON
        if not include_content:
            results_to_save = [
                {k: v for k, v in result.items() 
                 if k not in ['content']}
                for result in structured_data
            ]
        else:
            results_to_save = structured_data
        
        # Calculate statistics
        successful = sum(1 for r in structured_data if r.get("status") == "success")
        failed = len(structured_data) - successful
        
        json_data = {
            "crawl_metadata": {
                "timestamp": timestamp,
                "start_url": self.start_url,
                "output_dir": str(self.output_dir),
                "total_files": len(structured_data),
                "successful": successful,
                "failed": failed,
                "success_rate": f"{(successful/len(structured_data)*100):.1f}%" if structured_data else "0%"
            },
            "results": results_to_save
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ JSON saved: {filepath}")
        
        # Upload to S3 if available
        if HAS_S3_STORAGE:
            try:
                storage = get_storage()
                # Extract domain from start_url for S3 key
                domain = urlparse(self.start_url).netloc.replace('www.', '').replace('.', '_')
                s3_key = f"crawled_data/{domain}_{timestamp}.json"
                storage.upload_file(str(filepath), s3_key)
                print(f"✓ JSON uploaded to S3: {s3_key}")
            except Exception as e:
                print(f"⚠️ Failed to upload JSON to S3: {e}")
    
    def generate_crawl_summary(self, structured_data: List[Dict]):
        """Generate and print a summary of the crawling session."""
        if not structured_data:
            print("No results to summarize!")
            return
        
        successful = [r for r in structured_data if r.get("status") == "success"]
        failed = [r for r in structured_data if r.get("status") != "success"]
        
        print("\n" + "="*60)
        print("CRAWLING SUMMARY")
        print("="*60)
        print(f"Start URL: {self.start_url}")
        print(f"Total files processed: {len(structured_data)}")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")
        print(f"\nOutput Directory: {self.output_dir.absolute()}")
        print(f"  Markdown files: {self.output_dir}")
        print(f"  CSV files: {self.output_dir / 'csv'}")
        print(f"  JSON files: {self.output_dir / 'json'}")
        print("="*60 + "\n")
    
    def cancel(self):
        """Request cancellation of the crawl"""
        self.cancelled = True


# Alias for backwards compatibility
SimplifiedWebCrawler = EnhancedWebCrawler