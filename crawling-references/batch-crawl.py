import asyncio
import traceback
from typing import Any, Dict
from urllib.parse import urlparse
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
    LXMLWebScrapingStrategy
)
async def process_website_urls(urls: list[str]) -> Dict[str, Any]:
    results = {"total": len(urls), "success": 0, "failed": []}
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(
        check_robots_txt=True,
        cache_mode=CacheMode.BYPASS,
        excluded_tags=["nav", "footer", "header", "aside", "form", "script", "style"],
        exclude_external_links=True,
        scraping_strategy=LXMLWebScrapingStrategy(),
        only_text=True,
        remove_forms=True,
        wait_until="domcontentloaded",
        simulate_user=True,
        scan_full_page=True,
        verbose=True,
        screenshot=False,
    )
    crawler = AsyncWebCrawler(config=browser_config)
    try:
        await crawler.start()
        for url in urls:
            try:
                result = await crawler.arun(url=url, config=crawl_config)
                if result.success:
                    hostname = urlparse(url).netloc.replace(".", "_")
                    filename = f"{hostname}.md"
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(result.markdown)
                    print(f"✅ Saved {filename} ({len(result.markdown)} chars)")
                    results["success"] += 1
                else:
                    print(f"⚠️ Failed to crawl {url}: {result.error_message}")
                    results["failed"].append({"url": url, "error": result.error_message})
            except asyncio.TimeoutError:
                print(f"⏳ Timeout for {url}")
                results["failed"].append({"url": url, "error": "Timeout"})
            except Exception as e:
                print(f"💥 Unexpected error for {url}: {e}")
                traceback.print_exc()
                results["failed"].append({"url": url, "error": str(e)})
    finally:
        await crawler.close()
    return results

if __name__ == "__main__":
    urls_to_crawl = [
        "https://example.com",
        "https://www.wikipedia.org/"
    ]
    asyncio.run(process_website_urls(urls_to_crawl))