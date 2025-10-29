"""
Step 1: Crawling the CleanTechnica website
3-crawl_docs_FAST.py
--------------------
Batch-crawls a list of documentation URLs in parallel using Crawl4AI's arun_many and a memory-adaptive dispatcher.
Tracks memory usage, prints a summary of successes/failures, and is suitable for large-scale doc scraping jobs.
Usage: Call main() or run as a script. Adjust max_concurrent for parallelism.
"""
"""
3-crawl_docs_FAST.py
--------------------
Batch-crawls a list of documentation URLs in parallel using Crawl4AI's arun_many and a memory-adaptive dispatcher.
Tracks memory usage, prints a summary of successes/failures, and is suitable for large-scale doc scraping jobs.
Usage: Call main() or run as a script. Adjust max_concurrent for parallelism.
"""

import os
import sys
import psutil
import asyncio
import requests
import re
from typing import List
from xml.etree import ElementTree
from datetime import datetime
from urllib.parse import urlparse
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter


def create_safe_filename(url, index=0):
    """Create a safe filename from a URL."""
    parsed_url = urlparse(url)
    path = parsed_url.path.strip('/')
    filename_base = path.split('/')[-1] if path else "index"
    filename_base = re.sub(r'[<>:"/\\|?*]', '_', filename_base)
    filename_base = re.sub(r'_+', '_', filename_base).strip('_')
    if not filename_base:
        filename_base = f"page_{index}"
    return f"{filename_base}.md"


async def crawl_batch(crawler, urls, base_output_dir, batch_num, total_batches, crawl_config, dispatcher):
    """Crawl a batch of URLs and save results."""
    print(f"\n{'='*80}\nBATCH {batch_num}/{total_batches}: Crawling {len(urls)} URLs\n{'='*80}")
    results = await crawler.arun_many(urls=urls, config=crawl_config, dispatcher=dispatcher)

    success_count = saved_count = filtered_count = fail_count = 0

    for i, result in enumerate(results):
        if result.success:
            success_count += 1
            if result.markdown:
                content = result.markdown.fit_markdown or result.markdown.raw_markdown
                if content and len(content.strip()) > 100:
                    filename = create_safe_filename(result.url, index=i)
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
            if fail_count <= 3:
                print(f"  ✗ Error: {result.url[:60]}...")

    print(f"\nBatch {batch_num} Results: ✓ {success_count} | 💾 {saved_count} | ⊘ {filtered_count} | ✗ {fail_count}")
    return {'success': success_count, 'saved': saved_count, 'filtered': filtered_count, 'failed': fail_count}


async def crawl_parallel(urls: List[str], base_output_dir="crawl_output", max_concurrent=10, batch_size=10):
    """Crawl URLs in batches for better progress visibility."""
    os.makedirs(base_output_dir, exist_ok=True)
    peak_memory, process = 0, psutil.Process(os.getpid())

    def log_memory(prefix=""):
        nonlocal peak_memory
        current_mem = process.memory_info().rss
        peak_memory = max(peak_memory, current_mem)
        print(f"{prefix} Memory: {current_mem // (1024*1024)} MB | Peak: {peak_memory // (1024*1024)} MB")

    browser_config = BrowserConfig(headless=True, verbose=False, extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"])
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
    total_stats, start_time = {'success': 0, 'saved': 0, 'filtered': 0, 'failed': 0}, datetime.now().timestamp()

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for batch_num, batch_urls in enumerate(batches, 1):
            stats = await crawl_batch(crawler, batch_urls, base_output_dir, batch_num, len(batches), crawl_config, dispatcher)
            for key in total_stats:
                total_stats[key] += stats[key]
            log_memory("Current")
            if batch_num < len(batches):
                await asyncio.sleep(1)

    total_elapsed = datetime.now().timestamp() - start_time
    with open(os.path.join(base_output_dir, "_crawl_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Crawl Summary\n=============\n\n"
                f"Total URLs: {len(urls)}\n"
                f"Success: {total_stats['success']} | Saved: {total_stats['saved']} | Filtered: {total_stats['filtered']} | Failed: {total_stats['failed']}\n"
                f"Batch size: {batch_size} | Concurrency: {max_concurrent}\n"
                f"Time: {total_elapsed:.2f}s ({total_elapsed/60:.1f}m) | Avg per page: {total_elapsed/len(urls):.2f}s\n"
                f"Peak memory: {peak_memory // (1024*1024)} MB\n"
                f"Crawl date: {datetime.now().isoformat()}\n")

    print(f"\n{'='*80}\nCRAWL COMPLETE\n{'='*80}")


def get_cleantechnica_urls(date_filters=None, sitemap_pattern="post-sitemap"):
    """Fetch URLs from CleanTechnica's sitemap, filtering by sitemap pattern + date filters."""
    sitemap_index_url, all_urls = "https://cleantechnica.com/sitemap_index.xml", []
    try:
        headers = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/xml'}
        print(f"Fetching sitemap index: {sitemap_index_url}")
        response = requests.get(sitemap_index_url, headers=headers, timeout=30)
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
        ns = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        sitemap_locs = [loc.text for loc in root.findall('.//ns:sitemap/ns:loc', ns)]
        filtered_sitemaps = [s for s in sitemap_locs if sitemap_pattern in s]

        print(f"Found {len(sitemap_locs)} sitemaps | Keeping {len(filtered_sitemaps)} '{sitemap_pattern}'")

        for idx, sitemap_url in enumerate(filtered_sitemaps, 1):
            print(f"  [{idx}/{len(filtered_sitemaps)}] {sitemap_url}")
            try:
                sm_response = requests.get(sitemap_url, headers=headers, timeout=30)
                sm_response.raise_for_status()
                sm_root = ElementTree.fromstring(sm_response.content)
                urls = [loc.text for loc in sm_root.findall('.//ns:url/ns:loc', ns)]

                if date_filters:
                    urls = [u for u in urls if any(df in u for df in date_filters)]
                print(f"    Collected {len(urls)} URLs")
                all_urls.extend(urls)
            except Exception as e:
                print(f"    Error parsing {sitemap_url}: {e}")
                continue

        print(f"\n✓ Total URLs collected: {len(all_urls)}")
        return all_urls
    except Exception as e:
        print(f"Error fetching sitemap index: {e}")
        return []


async def main():
    base_output_dir, max_concurrent, batch_size = "crawl_output", 10, 100
    date_filters = ["/2025/06/", "/2025/07/", "/2025/08/", "/2025/09/"]
    sitemap_pattern = "post-sitemap"

    urls = get_cleantechnica_urls(date_filters=date_filters, sitemap_pattern=sitemap_pattern)
    if urls:
        await crawl_parallel(urls, base_output_dir=base_output_dir, max_concurrent=max_concurrent, batch_size=batch_size)
    else:
        print("No URLs found to crawl")


if __name__ == "__main__":
    asyncio.run(main())
