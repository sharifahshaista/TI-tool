"""
Web Scraper Package

A Python web scraper that respects robots.txt, extracts clean content,
and saves results to CSV files.
"""

from .scraper import WebScraper
from .robots_handler import RobotsHandler
from .content_extractor import ContentExtractor
from .url_tracker import URLTracker
from .csv_writer import CSVWriter

__version__ = '1.1.0'
__all__ = [
    'WebScraper',
    'RobotsHandler',
    'ContentExtractor',
    'URLTracker',
    'CSVWriter'
]