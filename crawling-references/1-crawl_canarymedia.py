"""
Step 1: Crawling the Canary Media website

Run this script to crawl the Canary Media website. It will crawl each and every one of the 60 topics 
as their corresponding URLs are listed in the array of URLs to scrape near the end of this script. 
"""


import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling.filters import (
    FilterChain,
    DomainFilter,
    URLPatternFilter,
    ContentTypeFilter
)
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
from urllib.parse import urlparse
import os
import re
import time
from datetime import datetime

def create_safe_folder_name(path_segment):
    """
    Convert URL path segment to a safe folder name.
    """
    # Remove leading/trailing slashes and convert to safe folder name
    safe_name = path_segment.strip('/')
    # Replace special characters with underscores
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', safe_name)
    # Remove multiple consecutive underscores
    safe_name = re.sub(r'_+', '_', safe_name)
    # Remove leading/trailing underscores
    safe_name = safe_name.strip('_')
    
    # If empty after cleaning, use a default name
    if not safe_name:
        safe_name = "root"
    
    return safe_name

def get_theme_folder_name(url, base_focused_path):
    """
    Extract theme folder name from URL based on the focused path for better organisation of crawl output
    """
    parsed_url = urlparse(url)
    url_path = parsed_url.path
    
    # If the URL contains the focused path, extract the theme
    if base_focused_path in url_path:
        # Get the part after the focused path
        theme_part = url_path.replace(base_focused_path, '', 1).strip('/')
        
        # If there's additional path after the focused path, use the first segment
        if theme_part:
            # Split by '/' and take the first meaningful part
            theme_segments = [seg for seg in theme_part.split('/') if seg]
            if theme_segments:
                return create_safe_folder_name(theme_segments[0])
    
    # Fallback: use the focused path itself as folder name
    return create_safe_folder_name(base_focused_path)

async def crawl_single_url(crawler, url, base_output_dir="canary_media_crawl_output", delay_between_requests=1):
    """
    Crawl a single URL and save results to themed folder.
    """
    print(f"\n{'='*80}")
    print(f"STARTING CRAWL: {url}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Parse the starting URL to get the base domain
        parsed_start_url = urlparse(url)
        base_domain = parsed_start_url.netloc # canarymedia.com
        focused_path = parsed_start_url.path # /articles/clean-aluminum
        scheme = parsed_start_url.scheme #http or https
        base_domain_path = base_domain + focused_path # canarymedia.com/articles/clean-aluminum

        print(f"Base domain: {base_domain}")
        print(f"Focused path: {focused_path}")
        print(f"Scheme: {scheme}")
        print(f"Base domain path: {base_domain_path}")
        
        # Extract theme from focused path for folder naming
        theme_folder = get_theme_folder_name(url, focused_path)
        print(f"Theme folder: {theme_folder}")
        
        # Create a filter chain with domain and pattern filters
        filter_chain = FilterChain([
            # Domain filter - stays within the same domain
            DomainFilter(
                allowed_domains=[base_domain],  # Only crawl within the starting domain
            ),
            
            # URL pattern filter - include only article/blog pages
            URLPatternFilter(
                patterns=[
                    f"*{base_domain}{focused_path}*",                    # Anything containing this path
                    f"*{focused_path}/p[0-9]*",                          # Numbered pagination
                    f"*{focused_path}?*page=*",                          # Query param pagination
                ],
            ),
            
            # Content type filter - only HTML content
            ContentTypeFilter(
                allowed_types=["text/html"],
                check_extension=True
            )
        ])
        
        # Configure markdown generation with pruning filter
        markdown_generator = DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(
                threshold=0.48,  # Relevance threshold (0.0 to 1.0)
                threshold_type="dynamic",  # or "dynamic" for adaptive threshold
            ),
            options={
                "ignore_links": False,
                "citations": True
            }
        )
        
        # Configure the deep crawl strategy with filters
        config = CrawlerRunConfig(
            deep_crawl_strategy=BFSDeepCrawlStrategy(
                max_depth=5,
                filter_chain=filter_chain,  # Apply the filter chain
                include_external=False,  # Don't follow external links
            ),
            scraping_strategy=LXMLWebScrapingStrategy(),
            markdown_generator=markdown_generator,  # Add pruning through markdown generator
            verbose=True,
            # Additional crawl configuration
            exclude_external_links=True,  # Remove external links from content
            excluded_tags=['nav', 'footer', 'header', 'aside'],  # Skip navigation elements
        )

        results = await crawler.arun(url, config=config)
        print(f"Crawled {len(results)} pages in total")

        # Track statistics
        filtered_count = 0
        saved_count = 0

        # Create the main output directory structure
        theme_output_dir = os.path.join(base_output_dir, theme_folder)
        os.makedirs(theme_output_dir, exist_ok=True)
        
        print(f"Saving content to: {theme_output_dir}")

        # Access individual results
        for i, result in enumerate(results):
            print(f"URL: {result.url}")
            print(f"Status: {result.status_code}")
            
            # Check if content was successfully extracted
            if result.success and result.markdown:
                # Use filtered markdown if available, otherwise raw markdown
                content = result.markdown.fit_markdown or result.markdown.raw_markdown
                
                # Only save if content meets minimum threshold
                if content and len(content.strip()) > 100:  # Minimum content length
                    # Create a clean filename from the URL
                    parsed_url = urlparse(result.url)
                    
                    # Create filename - more descriptive approach
                    url_path = parsed_url.path.strip('/')
                    if url_path:
                        # Use the last meaningful part of the path as filename
                        path_parts = [part for part in url_path.split('/') if part]
                        if path_parts:
                            filename_base = path_parts[-1]  # Use last part of path
                        else:
                            filename_base = "index"
                    else:
                        filename_base = "index"
                    
                    # Clean filename and ensure it's unique
                    filename_base = re.sub(r'[<>:"/\\|?*]', '_', filename_base)
                    filename_base = re.sub(r'_+', '_', filename_base).strip('_')
                    
                    # Add index if filename is empty
                    if not filename_base:
                        filename_base = f"page_{i}"
                    
                    filename = f"{filename_base}.md"
                    
                    # Handle duplicate filenames by adding a counter
                    counter = 1
                    original_filename = filename
                    filepath = os.path.join(theme_output_dir, filename)
                    while os.path.exists(filepath):
                        name_parts = original_filename.rsplit('.', 1)
                        if len(name_parts) == 2:
                            filename = f"{name_parts[0]}_{counter}.{name_parts[1]}"
                        else:
                            filename = f"{original_filename}_{counter}"
                        filepath = os.path.join(theme_output_dir, filename)
                        counter += 1

                    # Save the markdown content
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(f"# {result.url}\n\n")
                        f.write(f"Status: {result.status_code}\n")
                        f.write(f"Crawl Depth: {result.metadata.get('depth', 0)}\n\n")
                        f.write("---\n\n")
                        f.write(content)
                    
                    saved_count += 1
                    print(f"  ✓ Saved: {theme_folder}/{filename}")
                else:
                    filtered_count += 1
                    print(f"  ✗ Filtered out (insufficient content)")
            else:
                print(f"  ✗ Failed or no content")
        
        end_time = time.time()
        crawl_duration = end_time - start_time
        
        print(f"\nSummary for {url}:")
        print(f"  Total pages crawled: {len(results)}")
        print(f"  Pages saved: {saved_count}")
        print(f"  Pages filtered: {filtered_count}")
        print(f"  Output directory: {theme_output_dir}")
        print(f"  Crawl duration: {crawl_duration:.2f} seconds")
        
        # Create a summary file
        summary_file = os.path.join(theme_output_dir, "_crawl_summary.txt")
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"Crawl Summary\n")
            f.write(f"=============\n\n")
            f.write(f"Starting URL: {url}\n")
            f.write(f"Base domain: {base_domain}\n")
            f.write(f"Focused path: {focused_path}\n")
            f.write(f"Theme folder: {theme_folder}\n")
            f.write(f"Total pages crawled: {len(results)}\n")
            f.write(f"Pages saved: {saved_count}\n")
            f.write(f"Pages filtered: {filtered_count}\n")
            f.write(f"Crawl duration: {crawl_duration:.2f} seconds\n")
            f.write(f"Crawl date: {datetime.now().isoformat()}\n")
        
        print(f"  Summary saved: {summary_file}")
        
        return {
            'url': url,
            'theme_folder': theme_folder,
            'total_pages': len(results),
            'saved_count': saved_count,
            'filtered_count': filtered_count,
            'duration': crawl_duration,
            'success': True
        }
        
    except Exception as e:
        print(f"ERROR crawling {url}: {str(e)}")
        return {
            'url': url,
            'theme_folder': None,
            'total_pages': 0,
            'saved_count': 0,
            'filtered_count': 0,
            'duration': time.time() - start_time,
            'success': False,
            'error': str(e)
        }

async def crawl_multiple_urls(urls, base_output_dir="crawl_output", delay_between_crawls=5, delay_between_requests=1):
    """
    Crawl multiple URLs sequentially with delays between crawls.
    
    Args:
        urls: List of URLs to crawl
        base_output_dir: Base directory for all crawl outputs
        delay_between_crawls: Seconds to wait between different URL crawls
        delay_between_requests: Seconds to wait between individual requests
    """
    print(f"MULTI-URL CRAWL STARTED")
    print(f"URLs to crawl: {len(urls)}")
    print(f"Base output directory: {base_output_dir}")
    print(f"Delay between crawls: {delay_between_crawls}s")
    print(f"Delay between requests: {delay_between_requests}s")
    
    total_start_time = time.time()
    crawl_results = []
    
    # Create base output directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    async with AsyncWebCrawler() as crawler:
        for i, url in enumerate(urls, 1):
            print(f"\n{'#'*100}")
            print(f"CRAWL {i}/{len(urls)}: {url}")
            print(f"{'#'*100}")
            
            # Crawl the URL
            result = await crawl_single_url(
                crawler, 
                url, 
                base_output_dir=base_output_dir,
                delay_between_requests=delay_between_requests
            )
            crawl_results.append(result)
            
            # Add delay between crawls (except for the last one)
            if i < len(urls) and delay_between_crawls > 0:
                print(f"\n Wait {delay_between_crawls} seconds before next crawl.")
                await asyncio.sleep(delay_between_crawls)
    
    # Generate overall summary
    total_duration = time.time() - total_start_time
    successful_crawls = sum(1 for r in crawl_results if r['success'])
    total_pages_saved = sum(r['saved_count'] for r in crawl_results)
    total_pages_crawled = sum(r['total_pages'] for r in crawl_results)
    
    print(f"\n{'='*100}")
    print(f"MULTI-URL CRAWL COMPLETED")
    print(f"{'='*100}")
    print(f"Total URLs processed: {len(urls)}")
    print(f"Successful crawls: {successful_crawls}")
    print(f"Failed crawls: {len(urls) - successful_crawls}")
    print(f"Total pages crawled: {total_pages_crawled}")
    print(f"Total pages saved: {total_pages_saved}")
    print(f"Total duration: {total_duration:.2f} seconds")
    
    # Save overall summary
    overall_summary_file = os.path.join(base_output_dir, "_multi_crawl_summary.txt")
    with open(overall_summary_file, "w", encoding="utf-8") as f:
        f.write(f"Multi-URL Crawl Summary\n")
        f.write(f"=======================\n\n")
        f.write(f"Crawl date: {datetime.now().isoformat()}\n")
        f.write(f"Total URLs processed: {len(urls)}\n")
        f.write(f"Successful crawls: {successful_crawls}\n")
        f.write(f"Failed crawls: {len(urls) - successful_crawls}\n")
        f.write(f"Total pages crawled: {total_pages_crawled}\n")
        f.write(f"Total pages saved: {total_pages_saved}\n")
        f.write(f"Total duration: {total_duration:.2f} seconds\n\n")
        
        f.write("Individual Crawl Results:\n")
        f.write("=" * 40 + "\n")
        for result in crawl_results:
            f.write(f"\nURL: {result['url']}\n")
            f.write(f"Success: {result['success']}\n")
            if result['success']:
                f.write(f"Theme folder: {result['theme_folder']}\n")
                f.write(f"Pages crawled: {result['total_pages']}\n")
                f.write(f"Pages saved: {result['saved_count']}\n")
                f.write(f"Duration: {result['duration']:.2f}s\n")
            else:
                f.write(f"Error: {result.get('error', 'Unknown error')}\n")
            f.write("-" * 40 + "\n")
    
    print(f"Overall summary saved: {overall_summary_file}")
    
    # Show per-URL results
    print(f"\nPer-URL Results:")
    print(f"{'-'*100}")
    for result in crawl_results:
        status = "✓" if result['success'] else "✗"
        if result['success']:
            print(f"{status} {result['url']:<50} | {result['theme_folder']:<20} | Pages: {result['saved_count']:<3} | {result['duration']:.1f}s")
        else:
            print(f"{status} {result['url']:<50} | ERROR: {result.get('error', 'Unknown')}")
    
    return crawl_results

async def main():
    """
    Main function with URL list configuration.
    """
    # Configure your list of URLs here
    urls_to_crawl = [
        "https://www.canarymedia.com/articles/air-travel",
        "https://www.canarymedia.com/articles/batteries",
        "https://www.canarymedia.com/articles/canary-media",
        "https://www.canarymedia.com/articles/carbon-capture",
        "https://www.canarymedia.com/articles/carbon-removal",
        "https://www.canarymedia.com/articles/carbon-free-buildings",
        "https://www.canarymedia.com/articles/clean-aluminum",
        "https://www.canarymedia.com/articles/clean-energy",
        "https://www.canarymedia.com/articles/clean-energy-jobs",
        "https://www.canarymedia.com/articles/clean-energy-manufacturing",
        "https://www.canarymedia.com/articles/clean-energy-supply-chain",
        "https://www.canarymedia.com/articles/clean-fleets",
        "https://www.canarymedia.com/articles/clean-industry",
        "https://www.canarymedia.com/articles/climate-crisis",
        "https://www.canarymedia.com/articles/climate-justice",
        "https://www.canarymedia.com/articles/climatetech-finance",
        "https://www.canarymedia.com/articles/corporate-procurement",
        "https://www.canarymedia.com/articles/culture",
        "https://www.canarymedia.com/articles/distributed-energy-resources",
        "https://www.canarymedia.com/articles/electric-vehicles",
        "https://www.canarymedia.com/articles/electrification",
        "https://www.canarymedia.com/articles/emissions-reduction",
        "https://www.canarymedia.com/articles/energy-efficiency",
        "https://www.canarymedia.com/articles/energy-equity",
        "https://www.canarymedia.com/articles/energy-markets",
        "https://www.canarymedia.com/articles/energy-storage",
        "https://www.canarymedia.com/articles/enn",
        "https://www.canarymedia.com/articles/ev-charging",
        "https://www.canarymedia.com/articles/food-and-farms",
        "https://www.canarymedia.com/articles/fossil-fuels",
        "https://www.canarymedia.com/articles/fun-stuff",
        "https://www.canarymedia.com/articles/geothermal",
        "https://www.canarymedia.com/articles/green-steel",
        "https://www.canarymedia.com/articles/grid-edge",
        "https://www.canarymedia.com/articles/guides-and-how-tos",
        "https://www.canarymedia.com/articles/heat-pumps",
        "https://www.canarymedia.com/articles/hydrogen",
        "https://www.canarymedia.com/articles/hydropower",
        "https://www.canarymedia.com/articles/just-transition",
        "https://www.canarymedia.com/articles/land-use",
        "https://www.canarymedia.com/articles/liquefied-natural-gas",
        "https://www.canarymedia.com/articles/long-duration-energy-storage",
        "https://www.canarymedia.com/articles/sea-transport",
        "https://www.canarymedia.com/articles/methane",
        "https://www.canarymedia.com/articles/nuclear",
        "https://www.canarymedia.com/articles/ocean-energy",
        "https://www.canarymedia.com/articles/offshore-wind",
        "https://www.canarymedia.com/articles/policy-regulation",
        "https://www.canarymedia.com/articles/politics",
        "https://www.canarymedia.com/articles/public-transit",
        "https://www.canarymedia.com/articles/recycling-renewables",
        "https://www.canarymedia.com/articles/solar",
        "https://www.canarymedia.com/articles/sponsored",
        "https://www.canarymedia.com/articles/transmission",
        "https://www.canarymedia.com/articles/transportation",
        "https://www.canarymedia.com/articles/utilities",
        "https://www.canarymedia.com/articles/virtual-power-plants",
        "https://www.canarymedia.com/articles/wind",
        "https://www.canarymedia.com/articles/workforce-diversity",
    
    ]
    
    # Configuration options
    base_output_dir = "crawl_output"  # Where all crawl results will be saved
    
    # Start the multi-URL crawl
    results = await crawl_multiple_urls(
        urls=urls_to_crawl,
        base_output_dir=base_output_dir,
  
    )
    
    return results

if __name__ == "__main__":
    # Run the multi-URL crawler
    results = asyncio.run(main())