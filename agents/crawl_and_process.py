"""
Unified Crawl and Process Workflow
Combines intelligent web crawling with AI-powered post-processing
"""

import asyncio
from pathlib import Path
from typing import Optional
import argparse

from agents.web_crawler import IntelligentWebCrawler, standardize_url
from agents.markdown_post_processor import MarkdownPostProcessor


class CrawlAndProcessWorkflow:
    """
    Unified workflow for crawling websites and post-processing markdown files
    """
    
    def __init__(self, url: str, output_dir: str = "saved_md"):
        self.url = standardize_url(url)
        self.output_dir = Path(output_dir)
        self.crawler = IntelligentWebCrawler(self.url, str(self.output_dir))
        self.processor = MarkdownPostProcessor()
    
    async def run(
        self,
        max_pages: int = 100,
        date_filter_months: Optional[int] = 6,
        process_output_dir: str = "processed_content",
        progress_callback=None
    ):
        """
        Run complete workflow: crawl → post-process → export
        
        Args:
            max_pages: Maximum pages to crawl
            date_filter_months: Filter articles from last N months (None = no filter)
            process_output_dir: Directory for processed output
            progress_callback: Optional callback for progress updates
        """
        print("\n" + "="*80)
        print("CRAWL AND PROCESS WORKFLOW")
        print("="*80)
        print(f"URL: {self.url}")
        print(f"Max pages: {max_pages}")
        print(f"Output dir: {self.output_dir}")
        if date_filter_months:
            print(f"Date filter: Last {date_filter_months} months")
        print("="*80 + "\n")
        
        # Step 1: Crawl website
        print("STEP 1: CRAWLING WEBSITE")
        print("-" * 80)
        
        results = await self.crawler.crawl(
            max_pages=max_pages,
            progress_callback=progress_callback
        )
        
        total_saved = sum(r.pages_saved for r in results)
        
        print(f"\nCrawl complete! Saved {total_saved} pages to {self.output_dir}/")
        
        if total_saved == 0:
            print("No pages were crawled. Exiting.")
            return None, None
        
        # Step 2: Post-process markdown files
        print("\n" + "="*80)
        print("STEP 2: POST-PROCESSING MARKDOWN FILES")
        print("-" * 80)
        
        df, stats = self.processor.process_folder(
            folder_path=self.output_dir,
            output_dir=Path(process_output_dir),
            date_filter_months=date_filter_months,
            auto_detect=True
        )
        
        # Step 3: Summary
        print("\n" + "="*80)
        print("WORKFLOW COMPLETE")
        print("="*80)
        print(f"Pages crawled: {total_saved}")
        print(f"Pages processed: {stats['processed']}")
        print(f"Skipped (old): {stats['skipped_old']}")
        print(f"Skipped (error): {stats['skipped_error']}")
        print(f"\nProcessed data saved to: {process_output_dir}/")
        print("="*80)
        
        return df, stats


async def main():
    """Command-line interface for unified workflow"""
    parser = argparse.ArgumentParser(
        description="Intelligent Web Crawler with AI-powered Post-Processing"
    )
    parser.add_argument(
        'url',
        type=str,
        help='URL to crawl'
    )
    parser.add_argument(
        '-n', '--max-pages',
        type=int,
        default=50,
        help='Maximum pages to crawl (default: 50)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='saved_md',
        help='Output directory for crawled markdown (default: saved_md)'
    )
    parser.add_argument(
        '-p', '--processed-output',
        type=str,
        default='processed_content',
        help='Output directory for processed data (default: processed_content)'
    )
    parser.add_argument(
        '-m', '--months',
        type=int,
        default=6,
        help='Filter articles from last N months (default: 6, 0 = no filter)'
    )
    
    args = parser.parse_args()
    
    # Convert months=0 to None (no filter)
    date_filter = args.months if args.months > 0 else None
    
    # Run workflow
    workflow = CrawlAndProcessWorkflow(args.url, args.output)
    
    await workflow.run(
        max_pages=args.max_pages,
        date_filter_months=date_filter,
        process_output_dir=args.processed_output
    )


if __name__ == "__main__":
    asyncio.run(main())

