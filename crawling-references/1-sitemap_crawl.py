"""
This script is to test the effectiveness of sitemap-based crawling to enhance the efficiency of crawling by minimising
the "noise".

It is also more deterministic in terms of which URLs are articles/posts and which are not. More can be
done to optimise the pre-crawling strategy
"""


import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
from urllib.parse import urlparse
import requests
import xml.etree.ElementTree as ET
import os
import re
import time
import fnmatch
from datetime import datetime
import gzip


class ExcludeURLPatternFilter:
    """Custom filter to exclude URLs matching certain patterns."""

    def __init__(self, exclude_patterns):
        self.exclude_patterns = exclude_patterns

    def should_crawl(self, url):
        """Return False if URL matches any exclude pattern."""
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(url.lower(), pattern.lower()):
                return False
        return True


def get_sitemap_urls(base_url):
    """Fetch all URLs from sitemap(s) of a website."""

    def parse_sitemap(sitemap_url):
        try:
            print(f"  Fetching sitemap: {sitemap_url}")

            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(sitemap_url, headers=headers, timeout=30)
            response.raise_for_status()

            content = response.content
            if sitemap_url.endswith(".gz"):
                content = gzip.decompress(content)

            root = ET.fromstring(content)

            namespaces = {
                "sitemap": "http://www.sitemaps.org/schemas/sitemap/0.9",
            }

            urls, sitemaps = [], []

            # Sitemap index
            sitemap_elements = root.findall(".//sitemap:sitemap", namespaces)
            if sitemap_elements:
                print(f"    Found sitemap index with {len(sitemap_elements)} sitemaps")
                for elem in sitemap_elements:
                    loc_elem = elem.find("sitemap:loc", namespaces)
                    if loc_elem is not None:
                        sitemaps.append(loc_elem.text.strip())

            # Extract URLs
            url_elements = root.findall(".//sitemap:url", namespaces)
            if url_elements:
                print(f"    Found {len(url_elements)} URLs")
                for elem in url_elements:
                    loc_elem = elem.find("sitemap:loc", namespaces)
                    if loc_elem is not None:
                        url = loc_elem.text.strip()
                        lastmod = elem.find("sitemap:lastmod", namespaces)
                        urls.append(
                            {
                                "url": url,
                                "lastmod": lastmod.text if lastmod is not None else None,
                            }
                        )

            return urls, sitemaps

        except Exception as e:
            print(f"    Error parsing sitemap {sitemap_url}: {e}")
            return [], []

    print(f"Discovering URLs from sitemap for: {base_url}")
    base_url = base_url.rstrip("/")

    sitemap_candidates = [base_url]

    all_urls, all_sitemaps = [], []
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

    print(f"\nSitemap discovery complete!")
    print(f"  Total sitemaps processed: {len(processed)}")
    print(f"  Total URLs found: {len(url_list)}")

    return all_urls, url_list


def create_safe_filename(url, index=0):
    parsed_url = urlparse(url)
    path = parsed_url.path.strip("/")
    filename_base = path.split("/")[-1] if path else "index"
    filename_base = re.sub(r'[<>:"/\\|?*]', "_", filename_base) or f"page_{index}"
    return f"{filename_base}.md"


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


async def crawl_urls_from_sitemap(urls_data, base_output_dir="crawl_output", batch_size=50, delay_between_batches=5):
    """Crawl URLs discovered from sitemap."""

    print("\n" + "=" * 80)
    print("STARTING SITEMAP-BASED CRAWL")
    print("=" * 80)

    start_time = time.time()

    exclude_patterns = ["*/about*", "*/contact*", "*.pdf", "*.jpg", "*.png"]
    exclude_filter = ExcludeURLPatternFilter(exclude_patterns)

    filtered_urls = [u for u in urls_data if exclude_filter.should_crawl(u["url"])]
    excluded_urls = [u for u in urls_data if not exclude_filter.should_crawl(u["url"])]

    print(f"After filtering: {len(filtered_urls)} URLs to crawl, {len(excluded_urls)} excluded")

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
    results, saved_count, failed_count = [], 0, 0

    async with AsyncWebCrawler() as crawler:
        for i in range(0, len(filtered_urls), batch_size):
            batch = filtered_urls[i : i + batch_size]
            print(f"\nProcessing batch {i//batch_size+1}/{-(-len(filtered_urls)//batch_size)}")

            tasks = [crawl_single_url_simple(crawler, u["url"], config, u) for u in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for url_data, result in zip(batch, batch_results):
                url = url_data["url"]
                if isinstance(result, Exception) or not result.get("success"):
                    print(f"  ✗ Failed: {url}")
                    failed_count += 1
                else:
                    filename = create_safe_filename(url, saved_count)
                    filepath = os.path.join(base_output_dir, filename)
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(f"# {url}\n\n")
                        f.write(f"Status: {result['status_code']}\n")
                        f.write(f"---\n\n")
                        f.write(result["content"])
                    saved_count += 1
                    print(f"  ✓ Saved: {filename}")
                results.append(result)

            if i + batch_size < len(filtered_urls) and delay_between_batches > 0:
                print(f"  Waiting {delay_between_batches}s before next batch...")
                await asyncio.sleep(delay_between_batches)

    print("\n" + "=" * 80)
    print("SITEMAP CRAWL COMPLETE")
    print("=" * 80)
    print(f"URLs crawled: {len(filtered_urls)} | Saved: {saved_count} | Failed: {failed_count}")

    return results


async def main():
    base_url = "https://carboncapturemagazine.com/sitemap.xml"
    output_dir = "carboncapture_sitemap_crawl"

    urls_data, url_list = get_sitemap_urls(base_url)
    if not url_list:
        print("No URLs found in sitemap. Exiting.")
        return

    await crawl_urls_from_sitemap(urls_data, base_output_dir=output_dir)


if __name__ == "__main__":
    asyncio.run(main())
