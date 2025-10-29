"""
Quick Start Crawlers
Integrates the original crawling reference scripts for use in the Streamlit app.
These scripts run with their original parameters and configurations.
"""

import asyncio
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add the project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import functions from the original scripts
import psutil
import requests
import re
import gzip
import fnmatch
from typing import List, Callable, Optional
from xml.etree import ElementTree
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import FilterChain, DomainFilter, URLPatternFilter, ContentTypeFilter


# ===========================
# CANARY MEDIA CRAWLER
# ===========================

def create_safe_folder_name(path_segment):
    """Convert URL path segment to a safe folder name."""
    safe_name = path_segment.strip('/')
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', safe_name)
    safe_name = re.sub(r'_+', '_', safe_name)
    safe_name = safe_name.strip('_')
    if not safe_name:
        safe_name = "root"
    return safe_name


def get_theme_folder_name(url, base_focused_path):
    """Extract theme folder name from URL based on the focused path."""
    parsed_url = urlparse(url)
    url_path = parsed_url.path
    
    if base_focused_path in url_path:
        theme_part = url_path.replace(base_focused_path, '', 1).strip('/')
        if theme_part:
            theme_segments = [seg for seg in theme_part.split('/') if seg]
            if theme_segments:
                return create_safe_folder_name(theme_segments[0])
    
    return create_safe_folder_name(base_focused_path)


def create_simplified_structured_data(crawl_results, output_dir):
    """
    Convert crawl results to simplified structured data format.
    
    Args:
        crawl_results: List of crawl result dictionaries
        output_dir: Base output directory path
    
    Returns:
        List of simplified structured data dictionaries
    """
    import csv
    import json
    
    structured_data = []
    
    for result in crawl_results:
        if result.get('success', False):
            # Get the theme folder and find markdown files
            theme_folder = result.get('theme_folder', '')
            
            # If theme_folder exists, use it; otherwise use the output_dir directly
            if theme_folder:
                search_path = os.path.join(output_dir, theme_folder)
            else:
                search_path = output_dir
            
            if os.path.exists(search_path):
                # Find all markdown files in the directory (recursively if needed)
                for root, dirs, files in os.walk(search_path):
                    for filename in files:
                        if filename.endswith('.md') and not filename.startswith('_'):
                            file_path = os.path.join(root, filename)
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                
                                # Extract title from content (first # heading)
                                title = "No Title"
                                lines = content.split('\n')
                                for line in lines:
                                    if line.startswith('# '):
                                        title = line[2:].strip()
                                        break
                                
                                structured_data.append({
                                    "file_name": filename,
                                    "content": content,
                                    "url": result.get('url', ''),
                                    "title": title,
                                    "status": "success",
                                    "status_code": 200,
                                    "timestamp": datetime.now().isoformat(),
                                    "error_message": None
                                })
                            except Exception as e:
                                # If we can't read the file, still include it with error
                                structured_data.append({
                                    "file_name": filename,
                                    "content": "",
                                    "url": result.get('url', ''),
                                    "title": "Error reading file",
                                    "status": "failed",
                                    "status_code": 0,
                                    "timestamp": datetime.now().isoformat(),
                                    "error_message": str(e)
                                })
        else:
            # For failed crawls, create a placeholder entry
            structured_data.append({
                "file_name": f"failed_{result.get('url', 'unknown').replace('/', '_').replace(':', '_')}.md",
                "content": "",
                "url": result.get('url', ''),
                "title": "Failed to crawl",
                "status": "failed",
                "status_code": 0,
                "timestamp": datetime.now().isoformat(),
                "error_message": "Crawl failed"
            })
    
    return structured_data


def save_structured_data_csv(structured_data, output_dir, timestamp):
    """Save structured data to CSV file."""
    import csv
    
    csv_dir = os.path.join(output_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    
    filename = f"crawl_results_{timestamp}.csv"
    filepath = os.path.join(csv_dir, filename)
    
    # Define CSV columns - simplified structure
    csv_columns = [
        "file_name", "content", "url", "title", 
        "status", "status_code", "timestamp", "error_message"
    ]
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(structured_data)
    
    print(f"✓ CSV saved: {filepath}")


def save_structured_data_json(structured_data, output_dir, timestamp):
    """Save structured data to JSON file."""
    import json
    
    json_dir = os.path.join(output_dir, "json")
    os.makedirs(json_dir, exist_ok=True)
    
    filename = f"crawl_results_{timestamp}.json"
    filepath = os.path.join(json_dir, filename)
    
    # Calculate statistics
    successful = sum(1 for r in structured_data if r.get("status") == "success")
    failed = len(structured_data) - successful
    
    json_data = {
        "crawl_metadata": {
            "timestamp": timestamp,
            "output_dir": output_dir,
            "total_files": len(structured_data),
            "successful": successful,
            "failed": failed,
            "success_rate": f"{(successful/len(structured_data)*100):.1f}%" if structured_data else "0%"
        },
        "results": structured_data
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ JSON saved: {filepath}")


async def crawl_canary_media_single_url(
    crawler,
    url,
    base_output_dir="canary_media_crawl_output",
    progress_callback: Optional[Callable] = None
):
    """Crawl a single Canary Media URL (original implementation)."""
    start_time = time.time()
    
    if progress_callback:
        progress_callback(f"Starting crawl: {url}", 0, 1)
    
    try:
        parsed_start_url = urlparse(url)
        base_domain = parsed_start_url.netloc
        focused_path = parsed_start_url.path
        scheme = parsed_start_url.scheme
        
        theme_folder = get_theme_folder_name(url, focused_path)
        
        filter_chain = FilterChain([
            DomainFilter(allowed_domains=[base_domain]),
            URLPatternFilter(patterns=[
                f"*{base_domain}{focused_path}*",
                f"*{focused_path}/p[0-9]*",
                f"*{focused_path}?*page=*",
            ]),
            ContentTypeFilter(allowed_types=["text/html"], check_extension=True)
        ])
        
        markdown_generator = DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(threshold=0.48, threshold_type="dynamic"),
            options={"ignore_links": False, "citations": True}
        )
        
        config = CrawlerRunConfig(
            deep_crawl_strategy=BFSDeepCrawlStrategy(
                max_depth=5,
                filter_chain=filter_chain,
                include_external=False,
            ),
            scraping_strategy=LXMLWebScrapingStrategy(),
            markdown_generator=markdown_generator,
            verbose=True,
            exclude_external_links=True,
            excluded_tags=['nav', 'footer', 'header', 'aside'],
        )
        
        results = await crawler.arun(url, config=config)
        
        filtered_count = 0
        saved_count = 0
        theme_output_dir = os.path.join(base_output_dir, theme_folder)
        os.makedirs(theme_output_dir, exist_ok=True)
        
        for i, result in enumerate(results):
            if progress_callback:
                progress_callback(f"Processing page {i+1}/{len(results)}: {result.url[:60]}...", i+1, len(results))
            
            if result.success and result.markdown:
                content = result.markdown.fit_markdown or result.markdown.raw_markdown
                
                if content and len(content.strip()) > 100:
                    parsed_url = urlparse(result.url)
                    url_path = parsed_url.path.strip('/')
                    if url_path:
                        path_parts = [part for part in url_path.split('/') if part]
                        filename_base = path_parts[-1] if path_parts else "index"
                    else:
                        filename_base = "index"
                    
                    filename_base = re.sub(r'[<>:"/\\|?*]', '_', filename_base)
                    filename_base = re.sub(r'_+', '_', filename_base).strip('_')
                    if not filename_base:
                        filename_base = f"page_{i}"
                    
                    filename = f"{filename_base}.md"
                    counter = 1
                    original_filename = filename
                    filepath = os.path.join(theme_output_dir, filename)
                    while os.path.exists(filepath):
                        name_parts = original_filename.rsplit('.', 1)
                        filename = f"{name_parts[0]}_{counter}.{name_parts[1]}" if len(name_parts) == 2 else f"{original_filename}_{counter}"
                        filepath = os.path.join(theme_output_dir, filename)
                        counter += 1
                    
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(f"# {result.url}\n\nStatus: {result.status_code}\nCrawl Depth: {result.metadata.get('depth', 0)}\n\n---\n\n{content}")
                    
                    saved_count += 1
                else:
                    filtered_count += 1
        
        duration = time.time() - start_time
        
        # Save summary
        summary_file = os.path.join(theme_output_dir, "_crawl_summary.txt")
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"Crawl Summary\n=============\n\n")
            f.write(f"Starting URL: {url}\nBase domain: {base_domain}\nFocused path: {focused_path}\n")
            f.write(f"Theme folder: {theme_folder}\nTotal pages crawled: {len(results)}\n")
            f.write(f"Pages saved: {saved_count}\nPages filtered: {filtered_count}\n")
            f.write(f"Crawl duration: {duration:.2f} seconds\nCrawl date: {datetime.now().isoformat()}\n")
        
        return {
            'url': url,
            'theme_folder': theme_folder,
            'total_pages': len(results),
            'saved_count': saved_count,
            'filtered_count': filtered_count,
            'duration': duration,
            'success': True
        }
        
    except Exception as e:
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


async def run_canary_media_crawler(
    base_output_dir="crawled_data/canarymedia",
    progress_callback: Optional[Callable] = None
):
    """Run the Canary Media multi-URL crawler with original settings."""
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
    
    total_start_time = time.time()
    crawl_results = []
    os.makedirs(base_output_dir, exist_ok=True)
    
    async with AsyncWebCrawler() as crawler:
        for i, url in enumerate(urls_to_crawl, 1):
            if progress_callback:
                progress_callback(f"Topic {i}/{len(urls_to_crawl)}: {url}", i-1, len(urls_to_crawl))
            
            result = await crawl_canary_media_single_url(
                crawler,
                url,
                base_output_dir=base_output_dir,
                progress_callback=lambda msg, curr, tot: progress_callback(
                    f"Topic {i}/{len(urls_to_crawl)} - {msg}", i-1, len(urls_to_crawl)
                ) if progress_callback else None
            )
            crawl_results.append(result)
            
            if i < len(urls_to_crawl):
                await asyncio.sleep(5)  # Original delay between crawls
    
    total_duration = time.time() - total_start_time
    successful_crawls = sum(1 for r in crawl_results if r['success'])
    total_pages_saved = sum(r['saved_count'] for r in crawl_results)
    total_pages_crawled = sum(r['total_pages'] for r in crawl_results)
    
    # Save overall summary
    overall_summary_file = os.path.join(base_output_dir, "_multi_crawl_summary.txt")
    with open(overall_summary_file, "w", encoding="utf-8") as f:
        f.write(f"Multi-URL Crawl Summary\n=======================\n\n")
        f.write(f"Crawl date: {datetime.now().isoformat()}\nTotal URLs processed: {len(urls_to_crawl)}\n")
        f.write(f"Successful crawls: {successful_crawls}\nFailed crawls: {len(urls_to_crawl) - successful_crawls}\n")
        f.write(f"Total pages crawled: {total_pages_crawled}\nTotal pages saved: {total_pages_saved}\n")
        f.write(f"Total duration: {total_duration:.2f} seconds\n\n")
        
        f.write("Individual Crawl Results:\n" + "=" * 40 + "\n")
        for result in crawl_results:
            f.write(f"\nURL: {result['url']}\nSuccess: {result['success']}\n")
            if result['success']:
                f.write(f"Theme folder: {result['theme_folder']}\nPages crawled: {result['total_pages']}\n")
                f.write(f"Pages saved: {result['saved_count']}\nDuration: {result['duration']:.2f}s\n")
            else:
                f.write(f"Error: {result.get('error', 'Unknown error')}\n")
            f.write("-" * 40 + "\n")
    
    return {
        'total_urls': len(urls_to_crawl),
        'successful_crawls': successful_crawls,
        'total_pages_crawled': total_pages_crawled,
        'total_pages_saved': total_pages_saved,
        'total_duration': total_duration,
        'crawl_results': crawl_results,
        'output_dir': base_output_dir
    }


# ===========================
# CLEANTECHNICA CRAWLER
# ===========================

def create_safe_filename_ct(url, index=0):
    """Create a safe filename from a URL."""
    parsed_url = urlparse(url)
    path = parsed_url.path.strip('/')
    filename_base = path.split('/')[-1] if path else "index"
    filename_base = re.sub(r'[<>:"/\\|?*]', '_', filename_base)
    filename_base = re.sub(r'_+', '_', filename_base).strip('_')
    if not filename_base:
        filename_base = f"page_{index}"
    return f"{filename_base}.md"


def get_cleantechnica_urls(date_filters=None, sitemap_pattern="post-sitemap", max_sitemaps=5):
    """
    Fetch URLs from CleanTechnica's sitemap.
    Only processes the LAST few sitemaps (most recent articles).
    """
    sitemap_index_url = "https://cleantechnica.com/sitemap_index.xml"
    all_urls = []
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/xml'}
        print(f"Fetching sitemap index: {sitemap_index_url}")
        response = requests.get(sitemap_index_url, headers=headers, timeout=30)
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
        ns = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        sitemap_locs = [loc.text for loc in root.findall('.//ns:sitemap/ns:loc', ns)]
        filtered_sitemaps = [s for s in sitemap_locs if sitemap_pattern in s]
        
        print(f"Found {len(sitemap_locs)} sitemaps | {len(filtered_sitemaps)} '{sitemap_pattern}' sitemaps")
        
        # IMPORTANT: Only process the LAST few sitemaps (most recent articles)
        # The first sitemaps contain very old articles (2008+)
        recent_sitemaps = filtered_sitemaps[-max_sitemaps:]
        print(f"📋 Processing LAST {len(recent_sitemaps)} sitemaps (most recent articles)")
        
        for idx, sitemap_url in enumerate(recent_sitemaps, 1):
            print(f"  [{idx}/{len(recent_sitemaps)}] {sitemap_url}")
            try:
                sm_response = requests.get(sitemap_url, headers=headers, timeout=30)
                sm_response.raise_for_status()
                sm_root = ElementTree.fromstring(sm_response.content)
                urls = [loc.text for loc in sm_root.findall('.//ns:url/ns:loc', ns)]
                
                if date_filters:
                    original_count = len(urls)
                    urls = [u for u in urls if any(df in u for df in date_filters)]
                    print(f"    Collected {len(urls)} URLs (filtered from {original_count})")
                else:
                    print(f"    Collected {len(urls)} URLs (no date filter)")
                all_urls.extend(urls)
            except Exception as e:
                print(f"    Error parsing {sitemap_url}: {e}")
                continue
        
        print(f"\n✓ Total URLs collected: {len(all_urls)}")
        return all_urls
    except Exception as e:
        print(f"Error fetching sitemap index: {e}")
        return []


async def crawl_batch_ct(crawler, urls, base_output_dir, batch_num, total_batches, crawl_config, dispatcher, progress_callback=None):
    """Crawl a batch of URLs and save results (CleanTechnica)."""
    if progress_callback:
        progress_callback(f"Batch {batch_num}/{total_batches}: Crawling {len(urls)} URLs", batch_num, total_batches)
    
    results = await crawler.arun_many(urls=urls, config=crawl_config, dispatcher=dispatcher)
    
    success_count = saved_count = filtered_count = fail_count = 0
    
    for i, result in enumerate(results):
        if result.success:
            success_count += 1
            if result.markdown:
                content = result.markdown.fit_markdown or result.markdown.raw_markdown
                if content and len(content.strip()) > 100:
                    filename = create_safe_filename_ct(result.url, index=i)
                    filepath = os.path.join(base_output_dir, filename)
                    
                    # handle duplicate filenames
                    counter, original_filepath = 1, filepath
                    while os.path.exists(filepath):
                        name_parts = original_filepath.rsplit('.', 1)
                        filepath = f"{name_parts[0]}_{counter}.{name_parts[1]}"
                        counter += 1
                    
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(f"# {result.url}\n\nStatus: {result.status_code}\n\n---\n\n{content}")
                    saved_count += 1
                else:
                    filtered_count += 1
            else:
                filtered_count += 1
        else:
            fail_count += 1
    
    return {'success': success_count, 'saved': saved_count, 'filtered': filtered_count, 'failed': fail_count}


async def run_cleantechnica_crawler(
    base_output_dir="crawled_data/cleantechnica",
    progress_callback: Optional[Callable] = None,
    max_pages: int = 500
):
    """Run the CleanTechnica crawler - crawls recent articles (last 5 sitemaps)."""
    # Date filters for 2025 and 2026 articles
    date_filters = ["/2025/", "/2026/"]
    sitemap_pattern = "post-sitemap"
    max_concurrent = 10
    batch_size = 50  # Reduced from 100 for better stability
    max_sitemaps = 3  # Reduced from 5 to avoid overwhelming the system
    
    if progress_callback:
        progress_callback(f"Fetching URLs from sitemap (filters: {', '.join(date_filters)}, last {max_sitemaps} sitemaps)...", 0, 1)
    
    urls = get_cleantechnica_urls(
        date_filters=date_filters, 
        sitemap_pattern=sitemap_pattern,
        max_sitemaps=max_sitemaps
    )
    
    if not urls:
        if progress_callback:
            progress_callback("⚠️ No URLs found in recent sitemaps", 0, 1)
        return {
            'error': 'No URLs found to crawl from recent sitemaps.',
            'total_urls': 0,
            'total_stats': {'success': 0, 'saved': 0, 'filtered': 0, 'failed': 0},
            'total_duration': 0,
            'peak_memory_mb': 0,
            'output_dir': base_output_dir,
            'crawl_results': []
        }
    
    # Limit the number of URLs to crawl
    if len(urls) > max_pages:
        if progress_callback:
            progress_callback(f"⚠️ Limiting to {max_pages} URLs (found {len(urls)})", 0, 1)
        urls = urls[:max_pages]
    
    if progress_callback:
        progress_callback(f"✓ Will crawl {len(urls)} recent URLs", 0, 1)
    
    os.makedirs(base_output_dir, exist_ok=True)
    peak_memory = 0
    process = psutil.Process(os.getpid())
    
    # Simplified browser config to avoid launch issues
    browser_config = BrowserConfig(
        headless=True, 
        verbose=False
    )
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        scraping_strategy=LXMLWebScrapingStrategy(),
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(threshold=0.2, threshold_type="dynamic"),
            options={"ignore_links": False}
        ),
        exclude_external_links=True,
        excluded_tags=['nav', 'footer', 'header', 'aside'],
        verbose=False,
    )
    dispatcher = MemoryAdaptiveDispatcher(memory_threshold_percent=70.0, check_interval=1.0, max_session_permit=max_concurrent)
    
    batches = [urls[i:i + batch_size] for i in range(0, len(urls), batch_size)]
    total_stats = {'success': 0, 'saved': 0, 'filtered': 0, 'failed': 0}
    start_time = time.time()
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        for batch_num, batch_urls in enumerate(batches, 1):
            stats = await crawl_batch_ct(
                crawler, batch_urls, base_output_dir, 
                batch_num, len(batches), crawl_config, 
                dispatcher, progress_callback
            )
            
            for key in total_stats:
                total_stats[key] += stats[key]
            
            current_mem = process.memory_info().rss
            peak_memory = max(peak_memory, current_mem)
            
            if batch_num < len(batches):
                await asyncio.sleep(1)
    
    total_elapsed = time.time() - start_time
    
    # Save summary
    with open(os.path.join(base_output_dir, "_crawl_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Crawl Summary\n=============\n\n")
        f.write(f"Total URLs: {len(urls)}\n")
        f.write(f"Success: {total_stats['success']} | Saved: {total_stats['saved']} | Filtered: {total_stats['filtered']} | Failed: {total_stats['failed']}\n")
        f.write(f"Batch size: {batch_size} | Concurrency: {max_concurrent}\n")
        f.write(f"Time: {total_elapsed:.2f}s ({total_elapsed/60:.1f}m) | Avg per page: {total_elapsed/len(urls):.2f}s\n")
        f.write(f"Peak memory: {peak_memory // (1024*1024)} MB\n")
        f.write(f"Crawl date: {datetime.now().isoformat()}\n")
    
    # Create crawl_results structure for compatibility with other crawlers
    crawl_results = [{
        'url': base_output_dir,
        'theme_folder': '',
        'total_pages': total_stats['saved'],
        'saved_count': total_stats['saved'],
        'success': total_stats['saved'] > 0
    }]
    
    return {
        'total_urls': len(urls),
        'total_pages_crawled': len(urls),
        'total_pages_saved': total_stats['saved'],
        'successful_crawls': 1 if total_stats['saved'] > 0 else 0,
        'total_stats': total_stats,
        'total_duration': total_elapsed,
        'peak_memory_mb': peak_memory // (1024*1024),
        'output_dir': base_output_dir,
        'crawl_results': crawl_results
    }


# ===========================
# COOL COALITION CRAWLER
# ===========================

def create_safe_filename(url, index=0):
    """Create a safe filename from a URL."""
    parsed_url = urlparse(url)
    path = parsed_url.path.strip("/")
    filename_base = path.split("/")[-1] if path else "index"
    filename_base = re.sub(r'[<>:"/\\|?*]', "_", filename_base)
    filename_base = re.sub(r"_+", "_", filename_base).strip("_")
    if not filename_base:
        filename_base = f"page_{index}"
    return f"{filename_base}.md"


async def extract_article_urls_from_page(crawler, page_url, config):
    """Extract all article URLs from a news listing page."""
    print(f"  Extracting articles from: {page_url}")
    
    try:
        result = await crawler.arun(page_url, config=config)
        
        if not result.success or not result.html:
            print(f"    ✗ Failed to load page")
            return []
        
        soup = BeautifulSoup(result.html, 'html.parser')
        article_urls = set()
        
        # Find all links on the page
        all_links = soup.find_all('a', href=True)
        
        for link in all_links:
            href = link['href']
            full_url = urljoin(page_url, href)
            
            # Article URLs follow pattern: https://coolcoalition.org/article-slug-here/
            # They are direct children of the root domain (not in subdirectories)
            # and don't contain special markers like ?_page=, /news-stories/, etc.
            
            if (full_url.startswith('https://coolcoalition.org/') and 
                '?_page=' not in full_url and
                '/news-stories/' not in full_url and
                '/news-events/' not in full_url and
                '/category/' not in full_url and
                '/tag/' not in full_url and
                '/author/' not in full_url and
                '/#' not in full_url and
                full_url != 'https://coolcoalition.org/' and
                full_url != 'https://coolcoalition.org'):
                
                # Parse the URL to check structure
                parsed = urlparse(full_url)
                path_parts = [p for p in parsed.path.strip('/').split('/') if p]
                
                # Article URLs typically have 1+ path segments (the slug)
                # and don't end with known non-article patterns
                if (len(path_parts) >= 1 and 
                    not any(x in full_url.lower() for x in ['.pdf', '.jpg', '.png', '.gif', 
                                                              'privacy', 'terms', 'about', 
                                                              'contact', 'login', 'register'])):
                    article_urls.add(full_url)
        
        article_list = sorted(list(article_urls))
        print(f"    ✓ Found {len(article_list)} article URLs")
        
        # Debug: print first few URLs found
        if article_list:
            print(f"    Sample URLs:")
            for url in article_list[:3]:
                print(f"      - {url}")
        
        return article_list
        
    except Exception as e:
        print(f"    ✗ Error extracting articles: {e}")
        return []


async def crawl_single_url(crawler, url, config, url_type="article"):
    """Crawl a single URL and return the result."""
    try:
        result = await crawler.arun(url, config=config)
        
        if result.success and result.markdown:
            content = result.markdown.fit_markdown or result.markdown.raw_markdown
            
            if content and len(content.strip()) > 100:
                return {
                    "url": url,
                    "success": True,
                    "content": content,
                    "status_code": result.status_code,
                    "url_type": url_type,
                }
        
        return {
            "url": url,
            "success": False,
            "error": "Insufficient content",
            "url_type": url_type,
        }
        
    except Exception as e:
        return {
            "url": url,
            "success": False,
            "error": str(e),
            "url_type": url_type,
        }


async def crawl_cool_coalition_news(
    base_url="https://coolcoalition.org/news-events/news-stories",
    max_pages=10,
    base_output_dir="cool_coalition_crawl",
    delay_between_pages=2,
    delay_between_articles=1,
):
    """
    Crawl Cool Coalition news pages with pagination and extract all articles.
    
    Args:
        base_url: Base URL for news stories
        max_pages: Maximum number of pagination pages to crawl
        base_output_dir: Output directory for saved content
        delay_between_pages: Delay between pagination pages (seconds)
        delay_between_articles: Delay between article crawls (seconds)
    """
    
    print("\n" + "=" * 100)
    print("COOL COALITION NEWS CRAWLER")
    print("=" * 100)
    
    start_time = time.time()
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Setup crawler configuration
    markdown_generator = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(threshold=0.48, threshold_type="dynamic"),
        options={"ignore_links": False, "citations": True},
    )
    
    config = CrawlerRunConfig(
        scraping_strategy=LXMLWebScrapingStrategy(),
        markdown_generator=markdown_generator,
        verbose=False,  # Reduce noise
        excluded_tags=['nav', 'footer', 'header', 'aside', 'script', 'style'],
        exclude_external_links=True,
    )
    
    all_article_urls = set()
    pagination_urls = []
    saved_count = 0
    failed_count = 0
    
    # Generate pagination URLs
    for page_num in range(1, max_pages + 1):
        pagination_urls.append(f"{base_url}/?_page={page_num}")
    
    print(f"\nPlan: Crawl {len(pagination_urls)} pagination pages\n")
    
    async with AsyncWebCrawler() as crawler:
        # Phase 1: Extract article URLs from all pagination pages
        print("=" * 100)
        print("PHASE 1: EXTRACTING ARTICLE URLs FROM PAGINATION PAGES")
        print("=" * 100)
        
        for i, page_url in enumerate(pagination_urls, 1):
            print(f"\n[{i}/{len(pagination_urls)}] Processing pagination page:")
            
            article_urls = await extract_article_urls_from_page(crawler, page_url, config)
            
            if article_urls:
                all_article_urls.update(article_urls)
            else:
                print(f"    ⚠ No articles found - may have reached the last page")
                # If we get no articles, likely reached the end
                if i > 3:  # Allow first few pages to be empty (edge case)
                    print(f"    Stopping pagination crawl at page {i}")
                    break
            
            if i < len(pagination_urls) and delay_between_pages > 0:
                await asyncio.sleep(delay_between_pages)
        
        print(f"\n✓ Total unique articles discovered: {len(all_article_urls)}")
        
        # Phase 2: Crawl all discovered articles
        print("\n" + "=" * 100)
        print("PHASE 2: CRAWLING ARTICLE CONTENT")
        print("=" * 100 + "\n")
        
        article_list = sorted(list(all_article_urls))
        
        for i, article_url in enumerate(article_list, 1):
            print(f"[{i}/{len(article_list)}] Crawling: {article_url}")
            
            result = await crawl_single_url(crawler, article_url, config, url_type="article")
            
            if result["success"]:
                filename = create_safe_filename(article_url, saved_count)
                filepath = os.path.join(base_output_dir, filename)
                
                # Handle duplicate filenames
                counter = 1
                original_filepath = filepath
                while os.path.exists(filepath):
                    name_parts = original_filepath.rsplit(".", 1)
                    filepath = f"{name_parts[0]}_{counter}.{name_parts[1]}"
                    counter += 1
                
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"# {result['url']}\n\n")
                    f.write(f"Status: {result['status_code']}\n")
                    f.write(f"Crawled: {datetime.now().isoformat()}\n")
                    f.write(f"---\n\n")
                    f.write(result["content"])
                
                saved_count += 1
                print(f"  ✓ Saved: {filename}")
            else:
                failed_count += 1
                error = result.get("error", "Unknown error")
                print(f"  ✗ Failed: {error}")
            
            # Rate limiting
            if i < len(article_list) and delay_between_articles > 0:
                await asyncio.sleep(delay_between_articles)
    
    end_time = time.time()
    crawl_duration = end_time - start_time
    
    # Save summary
    summary_file = os.path.join(base_output_dir, "_crawl_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("Cool Coalition News Crawl Summary\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Base URL: {base_url}\n")
        f.write(f"Pagination pages checked: {min(len(pagination_urls), max_pages)}\n")
        f.write(f"Total unique articles discovered: {len(all_article_urls)}\n")
        f.write(f"Articles saved: {saved_count}\n")
        f.write(f"Articles failed: {failed_count}\n")
        f.write(f"Success rate: {saved_count}/{len(all_article_urls)} ({100*saved_count/max(1,len(all_article_urls)):.1f}%)\n")
        f.write(f"Crawl duration: {crawl_duration:.2f} seconds\n")
        f.write(f"Crawl date: {datetime.now().isoformat()}\n\n")
        
        f.write("All discovered article URLs:\n")
        f.write("-" * 100 + "\n")
        for i, url in enumerate(sorted(all_article_urls), 1):
            f.write(f"{i:3d}. {url}\n")
    
    print("\n" + "=" * 100)
    print("CRAWL COMPLETE")
    print("=" * 100)
    print(f"Articles discovered: {len(all_article_urls)}")
    print(f"Articles saved: {saved_count}")
    print(f"Articles failed: {failed_count}")
    print(f"Duration: {crawl_duration:.2f} seconds")
    print(f"Output directory: {base_output_dir}")
    print(f"Summary file: {summary_file}")
    
    return {
        "total_articles": len(all_article_urls),
        "saved_count": saved_count,
        "failed_count": failed_count,
        "duration": crawl_duration,
    }


async def run_cool_coalition_crawler(
    base_output_dir="crawled_data/coolcoalition",
    progress_callback: Optional[Callable] = None
):
    """Run the Cool Coalition crawler with original settings."""
    result = await crawl_cool_coalition_news(
        base_url="https://coolcoalition.org/news-events/news-stories",
        max_pages=10,  # Adjust this to crawl more/fewer pagination pages
        base_output_dir=base_output_dir,
        delay_between_pages=2,  # Delay between pagination pages
        delay_between_articles=1,  # Delay between article crawls
    )
    
    return {
        'total_articles': result["total_articles"],
        'saved_count': result["saved_count"],
        'failed_count': result["failed_count"],
        'duration': result["duration"],
        'output_dir': base_output_dir
    }


# ===========================
# CARBON CAPTURE MAGAZINE CRAWLER
# ===========================

class ExcludeURLPatternFilter:
    """Custom filter to exclude URLs matching certain patterns."""
    def __init__(self, exclude_patterns):
        self.exclude_patterns = exclude_patterns
    
    def should_crawl(self, url):
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(url.lower(), pattern.lower()):
                return False
        return True


def get_sitemap_urls(base_url):
    """Fetch all URLs from sitemap(s) of a website."""
    def parse_sitemap(sitemap_url):
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(sitemap_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            content = response.content
            if sitemap_url.endswith(".gz"):
                content = gzip.decompress(content)
            
            root = ElementTree.fromstring(content)
            namespaces = {"sitemap": "http://www.sitemaps.org/schemas/sitemap/0.9"}
            
            urls, sitemaps = [], []
            
            sitemap_elements = root.findall(".//sitemap:sitemap", namespaces)
            if sitemap_elements:
                for elem in sitemap_elements:
                    loc_elem = elem.find("sitemap:loc", namespaces)
                    if loc_elem is not None:
                        sitemaps.append(loc_elem.text.strip())
            
            url_elements = root.findall(".//sitemap:url", namespaces)
            if url_elements:
                for elem in url_elements:
                    loc_elem = elem.find("sitemap:loc", namespaces)
                    if loc_elem is not None:
                        url = loc_elem.text.strip()
                        lastmod = elem.find("sitemap:lastmod", namespaces)
                        urls.append({
                            "url": url,
                            "lastmod": lastmod.text if lastmod is not None else None,
                        })
            
            return urls, sitemaps
        except Exception:
            return [], []
    
    base_url = base_url.rstrip("/")
    sitemap_candidates = [base_url]
    all_urls = []
    processed, to_process = set(), sitemap_candidates.copy()
    
    while to_process:
        sitemap_url = to_process.pop(0)
        if sitemap_url in processed:
            continue
        processed.add(sitemap_url)
        
        urls, child_sitemaps = parse_sitemap(sitemap_url)
        all_urls.extend(urls)
        to_process.extend([s for s in child_sitemaps if s not in processed])
    
    url_list = [u["url"] for u in all_urls]
    return all_urls, url_list


async def crawl_single_url_simple(crawler, url, config, url_data):
    """Crawl a single URL with simple configuration."""
    try:
        result = await crawler.arun(url, config=config)
        if result.success and result.markdown:
            content = result.markdown.raw_markdown
            if content and len(content.strip()) > 100:
                return {
                    "url": url,
                    "success": True,
                    "content": content,
                    "status_code": result.status_code,
                    "sitemap_data": url_data,
                }
        return {"url": url, "success": False, "error": "No content", "sitemap_data": url_data}
    except Exception as e:
        return {"url": url, "success": False, "error": str(e), "sitemap_data": url_data}


async def run_carbon_capture_magazine_crawler(
    base_output_dir="crawled_data/carboncapturemagazine",
    progress_callback: Optional[Callable] = None
):
    """Run the Carbon Capture Magazine crawler with sitemap-based approach."""
    base_url = "https://carboncapturemagazine.com/sitemap.xml"
    batch_size = 30
    delay_between_batches = 3
    
    if progress_callback:
        progress_callback("Fetching URLs from sitemap...", 0, 1)
    
    urls_data, url_list = get_sitemap_urls(base_url)
    if not url_list:
        return {'error': 'No URLs found in sitemap'}
    
    exclude_patterns = ["*/about*", "*/contact*", "*.pdf", "*.jpg", "*.png"]
    exclude_filter = ExcludeURLPatternFilter(exclude_patterns)
    filtered_urls = [u for u in urls_data if exclude_filter.should_crawl(u["url"])]
    
    markdown_generator = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(threshold=0.48, threshold_type="dynamic"),
        options={"ignore_links": False, "citations": True},
    )
    
    config = CrawlerRunConfig(
        scraping_strategy=LXMLWebScrapingStrategy(),
        markdown_generator=markdown_generator,
        verbose=True,
    )
    
    os.makedirs(base_output_dir, exist_ok=True)
    saved_count = 0
    failed_count = 0
    
    async with AsyncWebCrawler() as crawler:
        total_batches = -(-len(filtered_urls)//batch_size)
        for i in range(0, len(filtered_urls), batch_size):
            batch = filtered_urls[i:i + batch_size]
            if progress_callback:
                progress_callback(f"Processing batch {i//batch_size+1}/{total_batches}", i//batch_size, total_batches)
            
            tasks = [crawl_single_url_simple(crawler, u["url"], config, u) for u in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for url_data, result in zip(batch, batch_results):
                url = url_data["url"]
                if isinstance(result, Exception) or not result.get("success"):
                    failed_count += 1
                else:
                    filename = create_safe_filename_ct(url, saved_count)
                    filepath = os.path.join(base_output_dir, filename)
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(f"# {url}\n\nStatus: {result['status_code']}\n---\n\n{result['content']}")
                    saved_count += 1
            
            if i + batch_size < len(filtered_urls) and delay_between_batches > 0:
                await asyncio.sleep(delay_between_batches)
    
    return {
        'total_urls': len(filtered_urls),
        'saved_count': saved_count,
        'failed_count': failed_count,
        'output_dir': base_output_dir
    }


# ===========================
# THINKGEOENERGY CRAWLER
# ===========================


def get_sitemap_urls(base_url):
    """Fetch all URLs from sitemap(s) of a website."""
    def parse_sitemap(sitemap_url):
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(sitemap_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            content = response.content
            if sitemap_url.endswith(".gz"):
                content = gzip.decompress(content)
            
            root = ElementTree.fromstring(content)
            namespaces = {"sitemap": "http://www.sitemaps.org/schemas/sitemap/0.9"}
            
            urls, sitemaps = [], []
            
            sitemap_elements = root.findall(".//sitemap:sitemap", namespaces)
            if sitemap_elements:
                for elem in sitemap_elements:
                    loc_elem = elem.find("sitemap:loc", namespaces)
                    if loc_elem is not None:
                        sitemaps.append(loc_elem.text.strip())
            
            url_elements = root.findall(".//sitemap:url", namespaces)
            if url_elements:
                for elem in url_elements:
                    loc_elem = elem.find("sitemap:loc", namespaces)
                    if loc_elem is not None:
                        url = loc_elem.text.strip()
                        lastmod = elem.find("sitemap:lastmod", namespaces)
                        urls.append({
                            "url": url,
                            "lastmod": lastmod.text if lastmod is not None else None,
                        })
            
            return urls, sitemaps
        except Exception:
            return [], []
    
    base_url = base_url.rstrip("/")
    sitemap_candidates = [base_url]
    all_urls = []
    processed, to_process = set(), sitemap_candidates.copy()
    
    while to_process:
        sitemap_url = to_process.pop(0)
        if sitemap_url in processed:
            continue
        processed.add(sitemap_url)
        
        urls, child_sitemaps = parse_sitemap(sitemap_url)
        all_urls.extend(urls)
        to_process.extend([s for s in child_sitemaps if s not in processed])
    
    url_list = [u["url"] for u in all_urls]
    return all_urls, url_list


async def run_thinkgeoenergy_crawler(
    base_output_dir="crawled_data/thinkgeoenergy",
    progress_callback: Optional[Callable] = None
):
    """Run the ThinkGeoEnergy crawler with original settings."""
    base_url = "https://www.thinkgeoenergy.com/post-sitemap.xml"
    batch_size = 30
    delay_between_batches = 3
    
    if progress_callback:
        progress_callback("Fetching URLs from sitemap...", 0, 1)
    
    urls_data, url_list = get_sitemap_urls(base_url)
    if not url_list:
        return {'error': 'No URLs found in sitemap'}
    
    exclude_patterns = ["*/about*", "*/contact*", "*.pdf", "*.jpg", "*.png"]
    exclude_filter = ExcludeURLPatternFilter(exclude_patterns)
    filtered_urls = [u for u in urls_data if exclude_filter.should_crawl(u["url"])]
    
    markdown_generator = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(threshold=0.48, threshold_type="dynamic"),
        options={"ignore_links": False, "citations": True},
    )
    
    config = CrawlerRunConfig(
        scraping_strategy=LXMLWebScrapingStrategy(),
        markdown_generator=markdown_generator,
        verbose=True,
    )
    
    os.makedirs(base_output_dir, exist_ok=True)
    saved_count = 0
    failed_count = 0
    
    async with AsyncWebCrawler() as crawler:
        total_batches = -(-len(filtered_urls)//batch_size)
        for i in range(0, len(filtered_urls), batch_size):
            batch = filtered_urls[i:i + batch_size]
            if progress_callback:
                progress_callback(f"Processing batch {i//batch_size+1}/{total_batches}", i//batch_size, total_batches)
            
            tasks = [crawler.arun(u["url"], config=config) for u in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for url_data, result in zip(batch, batch_results):
                url = url_data["url"]
                if isinstance(result, Exception) or not result.success or not result.markdown:
                    failed_count += 1
                else:
                    content = result.markdown.raw_markdown
                    if content and len(content.strip()) > 100:
                        filename = create_safe_filename_ct(url, saved_count)
                        filepath = os.path.join(base_output_dir, filename)
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(f"# {url}\n\nStatus: {result.status_code}\n---\n\n{content}")
                        saved_count += 1
                    else:
                        failed_count += 1
            
            if i + batch_size < len(filtered_urls) and delay_between_batches > 0:
                await asyncio.sleep(delay_between_batches)
    
    return {
        'total_urls': len(filtered_urls),
        'saved_count': saved_count,
        'failed_count': failed_count,
        'output_dir': base_output_dir
    }

# ===========================
# PRESET CRAWLER DISPATCHER
# ===========================

async def run_preset_crawl(preset_name, max_pages=500, progress_callback=None):
    """
    Run a preset crawler based on the preset name.
    
    Args:
        preset_name: Name of the preset (e.g., "Canary Media (60 Topics)", "CleanTechnica", etc.)
        max_pages: Maximum number of pages to crawl
        progress_callback: Optional callback for progress updates
    
    Returns:
        Dictionary with crawl results
    """
    # Map preset names to their corresponding crawler functions
    preset_mapping = {
        "Canary Media (60 Topics)": run_canary_media_crawler,
        "CleanTechnica": run_cleantechnica_crawler,
        "Cool Coalition": run_cool_coalition_crawler,
        "Carbon Capture Magazine": run_carbon_capture_magazine_crawler,
        "ThinkGeoEnergy": run_thinkgeoenergy_crawler,
    }
    
    if preset_name not in preset_mapping:
        raise ValueError(f"Unknown preset: {preset_name}. Available presets: {list(preset_mapping.keys())}")
    
    # Get the appropriate crawler function
    crawler_func = preset_mapping[preset_name]
    
    # Set up output directory based on preset
    output_dir_mapping = {
        "Canary Media (60 Topics)": "crawled_data/canarymedia",
        "CleanTechnica": "crawled_data/cleantechnica", 
        "Cool Coalition": "crawled_data/coolcoalition",
        "Carbon Capture Magazine": "crawled_data/carboncapturemagazine",
        "ThinkGeoEnergy": "crawled_data/thinkgeoenergy",
    }
    
    output_dir = output_dir_mapping[preset_name]
    
    # Run the crawler
    if progress_callback:
        progress_callback(f"Starting {preset_name} crawler...", 0, 1)
    
    result = await crawler_func(
        base_output_dir=output_dir,
        progress_callback=progress_callback
    )
    
    # Add simplified structured data to the result
    if 'crawl_results' in result:
        structured_data = create_simplified_structured_data(result['crawl_results'], output_dir)
        result['structured_data'] = structured_data
        
        # Save CSV and JSON files
        if structured_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_structured_data_csv(structured_data, output_dir, timestamp)
            save_structured_data_json(structured_data, output_dir, timestamp)
    
    return result
