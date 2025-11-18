"""
Content Extractor Module

This module extracts clean text content from HTML pages, removing
ads, scripts, styles, and other unwanted elements.
"""

from bs4 import BeautifulSoup
import logging
import re

logger = logging.getLogger(__name__)


class ContentExtractor:
    """Extracts clean text content from HTML."""

    # Common ad-related class patterns
    AD_PATTERNS = [
        'ad', 'ads', 'advertisement', 'banner', 'sponsored',
        'promo', 'promotion', 'sidebar', 'widget', 'popup'
    ]

    # Tags to remove entirely
    REMOVE_TAGS = ['script', 'style', 'img', 'iframe', 'noscript', 'svg']

    def __init__(self):
        """Initialize the ContentExtractor."""
        pass

    def _is_ad_element(self, element):
        """
        Check if an element is likely an advertisement.

        Args:
            element: BeautifulSoup element

        Returns:
            bool: True if element appears to be an ad
        """
        if not element.name:
            return False

        # Skip Elementor elements (not ads, they're content builders)
        classes = element.get('class', [])
        for cls in classes:
            if 'elementor' in cls.lower():
                return False

        # Skip WordPress theme elements
        element_id = element.get('id', '').lower()
        if any(wp_pattern in element_id for wp_pattern in ['wp-', 'post-', 'page-', 'entry-']):
            return False

        # Check class names with word boundaries
        for pattern in self.AD_PATTERNS:
            # Check each class individually (word boundary)
            for cls in classes:
                cls_lower = cls.lower()
                # Match if pattern is the whole class or at word boundaries
                if (cls_lower == pattern or
                    cls_lower.startswith(pattern + '-') or
                    cls_lower.startswith(pattern + '_') or
                    cls_lower.endswith('-' + pattern) or
                    cls_lower.endswith('_' + pattern)):
                    return True

        # Check id attribute with word boundaries
        for pattern in self.AD_PATTERNS:
            if (element_id == pattern or
                element_id.startswith(pattern + '-') or
                element_id.startswith(pattern + '_') or
                element_id.endswith('-' + pattern) or
                element_id.endswith('_' + pattern)):
                return True

        return False

    def _remove_unwanted_elements(self, soup):
        """
        Remove unwanted elements from the soup.

        Args:
            soup: BeautifulSoup object

        Returns:
            BeautifulSoup: Modified soup with elements removed
        """
        # Remove script, style, img, iframe tags
        for tag in self.REMOVE_TAGS:
            for element in soup.find_all(tag):
                element.decompose()

        # Remove ad-related elements (only divs and asides to avoid over-filtering)
        for tag_name in ['div', 'aside', 'section']:
            for element in soup.find_all(tag_name):
                if self._is_ad_element(element):
                    element.decompose()

        return soup

    def _clean_text(self, text):
        """
        Clean extracted text by removing extra whitespace and invalid characters.

        Args:
            text (str): Raw text

        Returns:
            str: Cleaned text
        """
        # Remove replacement characters and other problematic Unicode
        text = text.replace('\ufffd', '')  # Remove replacement character �
        text = text.replace('\x00', '')    # Remove null bytes

        # Remove non-printable characters except common whitespace
        text = ''.join(char for char in text
                      if char.isprintable() or char in '\n\r\t ')

        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def extract(self, html_content):
        """
        Extract clean text content from HTML.

        Args:
            html_content (str): Raw HTML content

        Returns:
            str: Extracted and cleaned text content
        """
        try:
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')

            # Remove skip-to-content links FIRST (before other processing)
            for skip_link in soup.find_all('a', class_=lambda x: x and 'skip' in ' '.join(x).lower()):
                skip_link.decompose()

            # Also remove common skip link patterns by text
            for link in soup.find_all('a', href=lambda x: x and x.startswith('#')):
                link_text = link.get_text(strip=True).lower()
                if link_text in ['skip to content', 'skip to main content', 'skip navigation', 'skip to main']:
                    link.decompose()

            # Remove unwanted elements
            soup = self._remove_unwanted_elements(soup)

            # Remove navigation elements (but not header divs which might contain content)
            for nav in soup.find_all('nav'):
                nav.decompose()

            # Try to find main content area first
            main_content = None
            # Expanded selectors including Elementor and WordPress patterns
            selectors = [
                'main',
                'article',
                '[role="main"]',
                '.content',
                '#content',
                '.main',
                '[data-elementor-type="wp-page"]',  # Elementor pages
                '[data-elementor-type="wp-post"]',  # Elementor posts
                '.elementor-widget-theme-post-content',  # Elementor content widget
                '.entry-content',  # WordPress default
                '#primary',  # Common WP theme pattern
                '.site-main',  # Common WP theme pattern
            ]

            for selector in selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    logger.debug(f"Found main content using selector: {selector}")
                    break

            # If no main content found, use the body
            if not main_content:
                main_content = soup.find('body') or soup

            # Extract text
            text = main_content.get_text(separator=' ')

            # Clean text
            text = self._clean_text(text)

            logger.info(f"Successfully extracted {len(text)} characters of text")
            return text

        except Exception as e:
            logger.error(f"Error extracting content: {e}")
            return ""

    def extract_links(self, html_content, base_url):
        """
        Extract all links from HTML content.

        Args:
            html_content (str): Raw HTML content
            base_url (str): Base URL for resolving relative links

        Returns:
            list: List of absolute URLs found in the page (deduplicated)
        """
        try:
            from urllib.parse import urljoin, urlparse, urldefrag

            soup = BeautifulSoup(html_content, 'html.parser')
            links = set()  # Use set to avoid duplicates

            for link in soup.find_all('a', href=True):
                href = link['href'].strip()

                # Skip empty hrefs and anchors
                if not href or href.startswith('#'):
                    continue

                # Skip mailto:, tel:, javascript: links
                if any(href.startswith(prefix) for prefix in ['mailto:', 'tel:', 'javascript:', 'data:']):
                    continue

                # Convert relative URLs to absolute
                absolute_url = urljoin(base_url, href)

                # Remove URL fragments (e.g., #section)
                absolute_url, _ = urldefrag(absolute_url)

                # Only include http/https links
                parsed = urlparse(absolute_url)
                if parsed.scheme in ['http', 'https']:
                    links.add(absolute_url)

            # Convert back to list for return
            links_list = list(links)
            logger.debug(f"Found {len(links_list)} unique links in page")
            return links_list

        except Exception as e:
            logger.error(f"Error extracting links: {e}")
            return []