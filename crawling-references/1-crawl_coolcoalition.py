"""
Cool Coalition News Crawler
Crawls pagination pages and extracts all articles from each page
"""

import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import os
import re
import time
from datetime import datetime


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


async def main():
    """Main execution function."""
    result = await crawl_cool_coalition_news(
        base_url="https://coolcoalition.org/news-events/news-stories",
        max_pages=10,  # Adjust this to crawl more/fewer pagination pages
        base_output_dir="cool_coalition_crawl",
        delay_between_pages=2,  # Delay between pagination pages
        delay_between_articles=1,  # Delay between article crawls
    )
    
    return result


if __name__ == "__main__":
    asyncio.run(main())