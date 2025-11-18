"""
Robots.txt Handler Module

This module handles parsing and checking robots.txt files to ensure
compliant web scraping behavior.
"""

import urllib.robotparser
from urllib.parse import urlparse
import logging
import requests

logger = logging.getLogger(__name__)


class RobotsHandler:
    """Handles robots.txt parsing and URL permission checking."""

    def __init__(self, user_agent='PythonWebScraper/1.0'):
        """
        Initialize the RobotsHandler.

        Args:
            user_agent (str): User agent string to identify the scraper
        """
        self.user_agent = user_agent
        self.robot_parsers = {}  # Cache parsers by domain

    def _get_robots_url(self, url):
        """
        Get the robots.txt URL for a given page URL.

        Args:
            url (str): The webpage URL

        Returns:
            str: The robots.txt URL for the domain
        """
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    def _get_domain(self, url):
        """
        Extract domain from URL for caching purposes.

        Args:
            url (str): The webpage URL

        Returns:
            str: The domain (scheme + netloc)
        """
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def can_fetch(self, url):
        """
        Check if the given URL can be fetched according to robots.txt.

        Args:
            url (str): The URL to check

        Returns:
            bool: True if fetching is allowed, False otherwise
        """
        try:
            domain = self._get_domain(url)

            # Use cached parser if available
            if domain not in self.robot_parsers:
                robots_url = self._get_robots_url(url)
                parser = urllib.robotparser.RobotFileParser()
                parser.set_url(robots_url)

                try:
                    # Fetch robots.txt manually to avoid parsing issues
                    response = requests.get(robots_url, timeout=10,
                                          headers={'User-Agent': self.user_agent})
                    response.raise_for_status()

                    # Parse the content manually
                    parser.parse(response.text.splitlines())
                    self.robot_parsers[domain] = parser
                    logger.info(f"Successfully loaded robots.txt from {robots_url}")
                except Exception as e:
                    logger.warning(f"Could not read robots.txt from {robots_url}: {e}")
                    # If robots.txt cannot be read, assume allowed (fail open)
                    self.robot_parsers[domain] = None

            parser = self.robot_parsers[domain]

            # If no parser (robots.txt unavailable), allow crawling
            if parser is None:
                return True

            can_fetch = parser.can_fetch(self.user_agent, url)
            logger.debug(f"robots.txt check for {url}: {can_fetch}")
            return can_fetch

        except Exception as e:
            logger.error(f"Error checking robots.txt for {url}: {e}")
            # On error, fail open and allow crawling
            return True

    def get_crawl_delay(self, url):
        """
        Get the crawl delay specified in robots.txt.

        Args:
            url (str): The URL to check

        Returns:
            float or None: Crawl delay in seconds, or None if not specified
        """
        try:
            domain = self._get_domain(url)

            if domain not in self.robot_parsers:
                # Trigger robots.txt loading
                self.can_fetch(url)

            parser = self.robot_parsers.get(domain)

            if parser is None:
                return None

            delay = parser.crawl_delay(self.user_agent)
            return delay

        except Exception as e:
            logger.error(f"Error getting crawl delay for {url}: {e}")
            return None