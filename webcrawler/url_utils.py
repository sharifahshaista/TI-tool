"""
URL Utilities Module

Provides URL normalization and comparison utilities to handle
variations like www/non-www, http/https, trailing slashes, etc.
"""

from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
import logging

logger = logging.getLogger(__name__)


def normalize_url(url):
    """
    Normalize a URL to a canonical form for comparison.

    Handles:
    - www/non-www (removes www)
    - http/https (prefers https)
    - Trailing slashes (removes them)
    - URL parameters (sorts them)
    - Fragments (removes them)
    - Case (lowercases domain)

    Args:
        url (str): URL to normalize

    Returns:
        str: Normalized URL

    Examples:
        >>> normalize_url('http://www.example.com/')
        'https://example.com'
        >>> normalize_url('HTTPS://Example.COM/Page')
        'https://example.com/Page'
        >>> normalize_url('http://example.com/path?b=2&a=1')
        'https://example.com/path?a=1&b=2'
    """
    if not url:
        return url

    try:
        # Parse URL
        parsed = urlparse(url)

        # Normalize scheme (prefer https)
        scheme = 'https' if parsed.scheme in ['http', 'https'] else parsed.scheme

        # Normalize domain (lowercase, remove www)
        netloc = parsed.netloc.lower()
        if netloc.startswith('www.'):
            netloc = netloc[4:]

        # Normalize path (remove trailing slash for consistency)
        path = parsed.path
        if path == '/':
            # Root path should be empty for consistency
            path = ''
        elif path and path.endswith('/'):
            # Remove trailing slashes from other paths
            path = path.rstrip('/')
        elif not path:
            path = ''

        # Normalize query parameters (sort them)
        query = parsed.query
        if query:
            params = parse_qs(query, keep_blank_values=True)
            # Sort parameters for consistent ordering
            query = urlencode(sorted(params.items()), doseq=True)

        # Remove fragment (not needed for crawling)
        fragment = ''

        # Reconstruct URL
        normalized = urlunparse((scheme, netloc, path, parsed.params, query, fragment))

        return normalized

    except Exception as e:
        logger.warning(f"Failed to normalize URL {url}: {e}")
        return url


def urls_equal(url1, url2):
    """
    Check if two URLs are equivalent after normalization.

    Args:
        url1 (str): First URL
        url2 (str): Second URL

    Returns:
        bool: True if URLs are equivalent

    Examples:
        >>> urls_equal('http://www.example.com', 'https://example.com')
        True
        >>> urls_equal('example.com/page/', 'example.com/page')
        True
    """
    return normalize_url(url1) == normalize_url(url2)


def same_domain(url1, url2):
    """
    Check if two URLs are on the same domain.

    Handles www/non-www variations automatically.

    Args:
        url1 (str): First URL
        url2 (str): Second URL

    Returns:
        bool: True if same domain

    Examples:
        >>> same_domain('http://www.example.com/page1', 'https://example.com/page2')
        True
        >>> same_domain('example.com', 'other.com')
        False
    """
    try:
        # Normalize and extract domains
        domain1 = urlparse(normalize_url(url1)).netloc
        domain2 = urlparse(normalize_url(url2)).netloc

        return domain1 == domain2

    except Exception as e:
        logger.warning(f"Failed to compare domains: {e}")
        return False


def ensure_scheme(url):
    """
    Ensure URL has a scheme (add https:// if missing).

    Args:
        url (str): URL that may be missing a scheme

    Returns:
        str: URL with scheme

    Examples:
        >>> ensure_scheme('example.com')
        'https://example.com'
        >>> ensure_scheme('http://example.com')
        'http://example.com'
    """
    if not url:
        return url

    # Check if URL already has a scheme
    if '://' in url:
        return url

    # Add https:// by default
    return f'https://{url}'


def get_domain(url):
    """
    Extract the domain from a URL (normalized).

    Args:
        url (str): URL to extract domain from

    Returns:
        str: Domain name (without www)

    Examples:
        >>> get_domain('https://www.example.com/page')
        'example.com'
        >>> get_domain('http://blog.example.com')
        'blog.example.com'
    """
    try:
        normalized = normalize_url(url)
        return urlparse(normalized).netloc
    except Exception as e:
        logger.warning(f"Failed to extract domain from {url}: {e}")
        return ''


def is_valid_url(url):
    """
    Check if a string is a valid URL.

    Args:
        url (str): String to validate

    Returns:
        bool: True if valid URL

    Examples:
        >>> is_valid_url('https://example.com')
        True
        >>> is_valid_url('not a url')
        False
    """
    if not url or not isinstance(url, str):
        return False

    try:
        # Try to parse it
        result = urlparse(url)

        # Must have scheme and netloc
        return bool(result.scheme) and bool(result.netloc)

    except Exception:
        return False