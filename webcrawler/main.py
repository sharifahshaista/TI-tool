"""
Main CLI Interface

Command-line interface for the web scraper that prompts for URLs
and orchestrates the crawling process.
"""

import logging
import sys
from .scraper import WebScraper


def setup_logging(level=logging.INFO):
    """
    Set up logging configuration.

    Args:
        level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('data/scraper.log')
        ]
    )


def prompt_for_urls():
    """
    Prompt user for URLs to crawl.

    Returns:
        list: List of URLs entered by user
    """
    print("\n=== Python Web Scraper ===")
    print("\nEnter URLs to crawl (one per line).")
    print("Press Enter twice when done, or Ctrl+C to exit.\n")

    urls = []
    empty_count = 0

    while True:
        try:
            url = input(f"URL {len(urls) + 1}: ").strip()

            if not url:
                empty_count += 1
                if empty_count >= 2:
                    break
                continue

            empty_count = 0

            # Basic URL validation
            if not url.startswith(('http://', 'https://')):
                print("  Warning: URL should start with http:// or https://")
                confirm = input("  Add anyway? (y/n): ").lower()
                if confirm != 'y':
                    continue

            urls.append(url)
            print(f"  Added: {url}")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)

    return urls


def display_results(summary):
    """
    Display crawl results summary.

    Args:
        summary (dict): Crawl summary from scraper
    """
    print("\n=== Crawl Results ===")
    print(f"\nTotal URLs processed: {summary['total']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")

    if summary['failed'] > 0:
        print("\nFailed URLs:")
        for result in summary['results']:
            if not result['success']:
                print(f"  - {result['url']}")
                print(f"    Reason: {result['error']}")

    print("\n=== Content saved to CSV files in data/ directory ===\n")


def display_stats(scraper):
    """
    Display scraper statistics.

    Args:
        scraper: WebScraper instance
    """
    stats = scraper.get_stats()

    print("\n=== Scraper Statistics ===")
    print(f"\nTotal URLs crawled (all time): {stats['total_crawled']}")
    print(f"Successful crawls: {stats['successful']}")
    print(f"Failed crawls: {stats['failed']}")

    if stats['output_files']:
        print(f"\nOutput files ({len(stats['output_files'])}):")
        for filepath in stats['output_files']:
            print(f"  - {filepath}")

    print()


def main():
    """Main CLI entry point."""
    # Set up logging
    setup_logging()

    # Initialize scraper
    scraper = WebScraper(user_agent='PythonWebScraper/1.0', default_delay=1.0)

    try:
        # Prompt for URLs
        urls = prompt_for_urls()

        if not urls:
            print("\nNo URLs provided. Exiting.")
            return

        print(f"\nStarting crawl of {len(urls)} URL(s)...")
        print("This may take a while depending on crawl delays...\n")

        # Crawl URLs
        summary = scraper.crawl_urls(urls)

        # Display results
        display_results(summary)
        display_stats(scraper)

    except Exception as e:
        logging.error(f"Error in main: {e}", exc_info=True)
        print(f"\nError: {e}")
        sys.exit(1)

    finally:
        scraper.close()


if __name__ == '__main__':
    main()