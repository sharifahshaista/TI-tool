"""
ThinkGeoEnergy Crawler
Uses web crawler with optimized settings for TGE content
"""

from agents.web_crawler import EnhancedWebCrawler, CrawlMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter

async def run_thinkgeoenergy_crawler(base_output_dir, progress_callback=None):
    """Run optimized crawler for ThinkGeoEnergy."""
    
    # Initialize crawler with link stripping enabled
    crawler = EnhancedWebCrawler(
        start_url="https://www.thinkgeoenergy.com/post-sitemap.xml",
        output_dir=base_output_dir,
        strip_links=True  # Enable link stripping
    )
    
    # Configure markdown generation to ignore links
    markdown_generator = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(
            threshold=0.48,
            threshold_type="dynamic"
        ),
        options={
            "ignore_links": True,  # Ignore links in markdown generation
            "citations": True
        }
    )
    
    # Run the crawler in sitemap mode
    result = await crawler.crawl(
        max_pages=500,
        progress_callback=progress_callback,
        mode=CrawlMode.SITEMAP
    )
    
    return result
