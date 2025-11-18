"""
CSV Writer Module

This module handles writing crawl results to CSV files with
proper formatting and error handling.
"""

import csv
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class CSVWriter:
    """Writes crawl results to CSV files."""

    FIELDNAMES = ['date', 'url', 'text_content']

    def __init__(self, output_dir='data'):
        """
        Initialize the CSVWriter.

        Args:
            output_dir (str): Directory for output CSV files
        """
        self.output_dir = output_dir
        self._ensure_directory()

    def _ensure_directory(self):
        """Ensure the output directory exists."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")

    def _get_domain_name(self, url):
        """
        Extract domain name from URL for filename.

        Args:
            url (str): URL to extract domain from

        Returns:
            str: Sanitized domain name suitable for filename
        """
        from urllib.parse import urlparse

        parsed = urlparse(url)
        domain = parsed.netloc

        # Remove 'www.' prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]

        # Replace dots with underscores for filename
        domain = domain.replace('.', '_')

        return domain

    def _get_output_path(self, url):
        """
        Get the output CSV path for a given URL's domain.

        Args:
            url (str): URL to get output path for

        Returns:
            str: Path to output CSV file
        """
        domain = self._get_domain_name(url)
        date_str = datetime.now().strftime('%Y%m%d')
        filename = f"{domain}_{date_str}.csv"
        return os.path.join(self.output_dir, filename)

    def _file_exists(self, filepath):
        """Check if file exists and has content."""
        return os.path.exists(filepath) and os.path.getsize(filepath) > 0

    def write_result(self, url, text_content):
        """
        Write a single crawl result to CSV.

        Args:
            url (str): The crawled URL
            text_content (str): Extracted text content

        Returns:
            bool: True if write was successful, False otherwise
        """
        try:
            output_path = self._get_output_path(url)
            file_exists = self._file_exists(output_path)

            # Prepare row data
            row = {
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'url': url,
                'text_content': text_content
            }

            # Write to CSV
            with open(output_path, 'a', newline='', encoding='utf-8', errors='replace') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=self.FIELDNAMES,
                    quoting=csv.QUOTE_NONNUMERIC  # Quote all non-numeric fields
                )

                # Write header if file is new
                if not file_exists:
                    writer.writeheader()
                    logger.info(f"Created new CSV file: {output_path}")

                writer.writerow(row)

            logger.info(f"Wrote result to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error writing to CSV: {e}")
            return False

    def write_results_batch(self, results):
        """
        Write multiple crawl results to CSV.

        Args:
            results (list): List of (url, text_content) tuples

        Returns:
            int: Number of successfully written results
        """
        success_count = 0

        for url, text_content in results:
            if self.write_result(url, text_content):
                success_count += 1

        logger.info(f"Wrote {success_count}/{len(results)} results to CSV")
        return success_count

    def get_output_files(self):
        """
        Get list of all output CSV files.

        Returns:
            list: List of CSV file paths in output directory
        """
        try:
            files = []
            for filename in os.listdir(self.output_dir):
                if filename.endswith('.csv'):
                    files.append(os.path.join(self.output_dir, filename))
            return files
        except Exception as e:
            logger.error(f"Error listing output files: {e}")
            return []