"""
URL Tracker Module

This module tracks which URLs have been crawled to avoid duplicates
and manage crawl state using JSON file persistence.
"""

import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class URLTracker:
    """Tracks crawled URLs using JSON file persistence."""

    def __init__(self, storage_path='data/crawled_urls.json'):
        """
        Initialize the URLTracker.

        Args:
            storage_path (str): Path to JSON file for storing crawled URLs
        """
        self.storage_path = storage_path
        self.crawled_urls = {}
        self._ensure_directory()
        self._load()

    def _ensure_directory(self):
        """Ensure the storage directory exists."""
        directory = os.path.dirname(self.storage_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

    def _load(self):
        """Load crawled URLs from JSON file."""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    self.crawled_urls = json.load(f)
                logger.info(f"Loaded {len(self.crawled_urls)} crawled URLs from {self.storage_path}")
            else:
                self.crawled_urls = {}
                logger.info(f"No existing URL tracker found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading URL tracker: {e}")
            self.crawled_urls = {}

    def _save(self):
        """Save crawled URLs to JSON file."""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.crawled_urls, f, indent=2)
            logger.debug(f"Saved {len(self.crawled_urls)} crawled URLs to {self.storage_path}")
        except Exception as e:
            logger.error(f"Error saving URL tracker: {e}")

    def is_crawled(self, url):
        """
        Check if a URL has been crawled.

        Args:
            url (str): URL to check

        Returns:
            bool: True if URL has been crawled, False otherwise
        """
        return url in self.crawled_urls

    def mark_crawled(self, url, success=True, error=None):
        """
        Mark a URL as crawled.

        Args:
            url (str): URL that was crawled
            success (bool): Whether the crawl was successful
            error (str): Error message if crawl failed
        """
        self.crawled_urls[url] = {
            'timestamp': datetime.now().isoformat(),
            'success': success,
            'error': error
        }
        self._save()
        logger.debug(f"Marked URL as crawled: {url} (success={success})")

    def get_crawled_count(self):
        """
        Get the total number of crawled URLs.

        Returns:
            int: Number of crawled URLs
        """
        return len(self.crawled_urls)

    def get_successful_count(self):
        """
        Get the number of successfully crawled URLs.

        Returns:
            int: Number of successful crawls
        """
        return sum(1 for info in self.crawled_urls.values() if info.get('success', False))

    def get_failed_count(self):
        """
        Get the number of failed crawls.

        Returns:
            int: Number of failed crawls
        """
        return sum(1 for info in self.crawled_urls.values() if not info.get('success', True))

    def get_crawl_info(self, url):
        """
        Get crawl information for a specific URL.

        Args:
            url (str): URL to get info for

        Returns:
            dict or None: Crawl information or None if not crawled
        """
        return self.crawled_urls.get(url)

    def clear(self):
        """Clear all tracked URLs."""
        self.crawled_urls = {}
        self._save()
        logger.info("Cleared all tracked URLs")

    def get_all_urls(self):
        """
        Get all tracked URLs.

        Returns:
            list: List of all crawled URLs
        """
        return list(self.crawled_urls.keys())