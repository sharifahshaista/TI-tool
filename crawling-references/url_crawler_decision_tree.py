"""
URL Crawler Decision Tree
=========================
This script analyzes a user-provided URL and automatically determines the best 
Crawl4AI strategy to use based on the website's structure.

It implements a decision tree based on patterns from existing crawling scripts:
1. Sitemap availability detection
2. URL structure analysis
3. Pagination pattern detection
4. Content organization assessment

Usage:
    python url_crawler_decision_tree.py "https://example.com/articles"
"""

import asyncio
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Tuple
import gzip
import re


class CrawlingStrategy:
    """Enum-like class for crawling strategies"""
    SITEMAP_BASED = "sitemap_based"
    SITEMAP_WITH_DATE_FILTER = "sitemap_with_date_filter"
    DEEP_CRAWL_FOCUSED_PATH = "deep_crawl_focused_path"
    PAGINATION_THEN_ARTICLES = "pagination_then_articles"
    CATEGORY_DEEP_CRAWL = "category_deep_crawl"
    SIMPLE_DEEP_CRAWL = "simple_deep_crawl"
    SINGLE_PAGE_CRAWL = "single_page_crawl"


class URLAnalyzer:
    """Analyzes URLs to determine optimal crawling strategy"""
    
    def __init__(self, url: str):
        self.url = url
        self.parsed_url = urlparse(url)
        self.base_domain = self.parsed_url.netloc
        self.path = self.parsed_url.path
        self.base_url = f"{self.parsed_url.scheme}://{self.base_domain}"
        
        # Analysis results
        self.has_sitemap = False
        self.sitemap_urls = []
        self.has_pagination = False
        self.has_categories = False
        self.has_date_structure = False
        self.is_topic_specific = False
        self.url_pattern_type = None
        
    def analyze(self) -> Dict:
        """Perform comprehensive URL analysis"""
        print("\n" + "="*80)
        print("URL ANALYSIS")
        print("="*80)
        print(f"Target URL: {self.url}")
        print(f"Base Domain: {self.base_domain}")
        print(f"Path: {self.path}")
        
        # Run all analysis steps
        self._check_sitemap()
        self._analyze_url_structure()
        self._detect_pagination()
        self._detect_categories()
        self._detect_date_patterns()
        
        # Determine strategy
        strategy = self._determine_strategy()
        
        analysis_result = {
            'url': self.url,
            'base_domain': self.base_domain,
            'base_url': self.base_url,
            'path': self.path,
            'has_sitemap': self.has_sitemap,
            'sitemap_urls': self.sitemap_urls,
            'has_pagination': self.has_pagination,
            'has_categories': self.has_categories,
            'has_date_structure': self.has_date_structure,
            'is_topic_specific': self.is_topic_specific,
            'url_pattern_type': self.url_pattern_type,
            'recommended_strategy': strategy,
            'strategy_explanation': self._get_strategy_explanation(strategy),
            'config_template': self._get_config_template(strategy)
        }
        
        self._print_analysis_summary(analysis_result)
        
        return analysis_result
    
    def _check_sitemap(self):
        """Check if website has a sitemap"""
        print("\n[1] Checking for sitemap...")
        
        sitemap_candidates = [
            f"{self.base_url}/sitemap.xml",
            f"{self.base_url}/sitemap_index.xml",
            f"{self.base_url}/post-sitemap.xml",
            f"{self.base_url}/page-sitemap.xml",
        ]
        
        for sitemap_url in sitemap_candidates:
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                response = requests.get(sitemap_url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    # Try to parse as XML
                    try:
                        content = response.content
                        if sitemap_url.endswith('.gz'):
                            content = gzip.decompress(content)
                        
                        root = ET.fromstring(content)
                        self.has_sitemap = True
                        self.sitemap_urls.append(sitemap_url)
                        print(f"  ✓ Found sitemap: {sitemap_url}")
                    except ET.ParseError:
                        continue
                        
            except Exception as e:
                continue
        
        if not self.has_sitemap:
            print(f"  ✗ No sitemap found")
    
    def _analyze_url_structure(self):
        """Analyze URL structure to detect patterns"""
        print("\n[2] Analyzing URL structure...")
        
        path_lower = self.path.lower()
        
        # Check if it's topic/category specific
        topic_indicators = ['/articles/', '/posts/', '/blog/', '/news/', '/category/']
        for indicator in topic_indicators:
            if indicator in path_lower:
                self.is_topic_specific = True
                # Extract topic name if possible
                parts = [p for p in self.path.split('/') if p]
                if len(parts) > 1:
                    print(f"  ✓ Topic-specific URL detected: {parts[-1]}")
                break
        
        if not self.is_topic_specific:
            print(f"  • URL appears to be root/general level")
    
    def _detect_pagination(self):
        """Detect if URL uses pagination"""
        print("\n[3] Detecting pagination patterns...")
        
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(self.url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Look for pagination indicators
                pagination_patterns = [
                    '?page=',
                    '?_page=',
                    '/page/',
                    '/p/',
                    'pagination',
                    'next-page',
                    'load-more'
                ]
                
                page_html = response.text.lower()
                for pattern in pagination_patterns:
                    if pattern in page_html:
                        self.has_pagination = True
                        print(f"  ✓ Pagination detected: {pattern}")
                        return
                
                print(f"  ✗ No pagination detected")
        except Exception as e:
            print(f"  ! Could not check pagination: {e}")
    
    def _detect_categories(self):
        """Detect if URL is part of a category system"""
        print("\n[4] Detecting category structure...")
        
        category_indicators = ['/category/', '/categories/', '/topic/', '/topics/', '/tag/']
        
        for indicator in category_indicators:
            if indicator in self.path.lower():
                self.has_categories = True
                print(f"  ✓ Category structure detected: {indicator}")
                return
        
        print(f"  ✗ No category structure detected")
    
    def _detect_date_patterns(self):
        """Detect if URLs contain date patterns (year/month)"""
        print("\n[5] Detecting date-based URL patterns...")
        
        # Check if sitemap URLs contain dates
        if self.sitemap_urls:
            for sitemap_url in self.sitemap_urls[:3]:  # Check first few
                try:
                    headers = {"User-Agent": "Mozilla/5.0"}
                    response = requests.get(sitemap_url, headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        # Look for date patterns in sitemap URLs
                        content = response.content
                        if sitemap_url.endswith('.gz'):
                            content = gzip.decompress(content)
                        
                        root = ET.fromstring(content)
                        namespaces = {"sitemap": "http://www.sitemaps.org/schemas/sitemap/0.9"}
                        
                        url_elements = root.findall(".//sitemap:url/sitemap:loc", namespaces)[:10]
                        
                        for elem in url_elements:
                            url_text = elem.text
                            # Look for /YYYY/MM/ or /YYYY-MM-DD/ patterns
                            if re.search(r'/20\d{2}/(0[1-9]|1[0-2])/', url_text):
                                self.has_date_structure = True
                                print(f"  ✓ Date-based URL structure detected")
                                print(f"    Example: {url_text}")
                                return
                except Exception:
                    continue
        
        # Also check the provided URL itself
        if re.search(r'/20\d{2}/(0[1-9]|1[0-2])/', self.url):
            self.has_date_structure = True
            print(f"  ✓ Date pattern in provided URL")
            return
        
        print(f"  ✗ No date-based structure detected")
    
    def _determine_strategy(self) -> str:
        """
        Decision tree to determine optimal crawling strategy
        
        Decision Logic:
        1. If sitemap exists + date structure → Sitemap with date filter
        2. If sitemap exists → Sitemap-based crawl
        3. If pagination detected → Pagination then articles
        4. If category structure → Category deep crawl
        5. If topic-specific path → Focused path deep crawl
        6. If root domain → Simple deep crawl
        7. Default → Single page crawl
        """
        print("\n[6] Determining optimal strategy...")
        
        # Decision tree
        if self.has_sitemap and self.has_date_structure:
            return CrawlingStrategy.SITEMAP_WITH_DATE_FILTER
        
        elif self.has_sitemap:
            return CrawlingStrategy.SITEMAP_BASED
        
        elif self.has_pagination:
            return CrawlingStrategy.PAGINATION_THEN_ARTICLES
        
        elif self.has_categories:
            return CrawlingStrategy.CATEGORY_DEEP_CRAWL
        
        elif self.is_topic_specific:
            return CrawlingStrategy.DEEP_CRAWL_FOCUSED_PATH
        
        elif self.path in ['/', '']:
            return CrawlingStrategy.SIMPLE_DEEP_CRAWL
        
        else:
            return CrawlingStrategy.SINGLE_PAGE_CRAWL
    
    def _get_strategy_explanation(self, strategy: str) -> str:
        """Get human-readable explanation of the strategy"""
        explanations = {
            CrawlingStrategy.SITEMAP_WITH_DATE_FILTER: 
                "Sitemap-based crawl with date filtering for efficient, targeted crawling of recent content.",
            
            CrawlingStrategy.SITEMAP_BASED: 
                "Sitemap-based crawl for deterministic, comprehensive site coverage with minimal noise.",
            
            CrawlingStrategy.PAGINATION_THEN_ARTICLES: 
                "Two-phase crawl: First extract article URLs from pagination pages, then crawl articles.",
            
            CrawlingStrategy.CATEGORY_DEEP_CRAWL: 
                "Deep crawl starting from category page, following links to discover all related articles.",
            
            CrawlingStrategy.DEEP_CRAWL_FOCUSED_PATH: 
                "Deep crawl focused on specific path/topic with URL pattern filtering.",
            
            CrawlingStrategy.SIMPLE_DEEP_CRAWL: 
                "Simple BFS deep crawl from root domain with basic filtering.",
            
            CrawlingStrategy.SINGLE_PAGE_CRAWL: 
                "Single page crawl without following links (most conservative approach)."
        }
        return explanations.get(strategy, "Custom crawling strategy")
    
    def _get_config_template(self, strategy: str) -> Dict:
        """Get configuration template for the recommended strategy"""
        
        templates = {
            CrawlingStrategy.SITEMAP_WITH_DATE_FILTER: {
                'approach': 'sitemap_with_date_filter',
                'reference_file': 'CleanTechnica/1-crawl_cleantechnica.py',
                'key_features': [
                    'Parse sitemap XML first',
                    'Filter URLs by date patterns',
                    'Batch parallel crawling',
                    'Memory-adaptive dispatcher'
                ],
                'config_params': {
                    'sitemap_url': self.sitemap_urls[0] if self.sitemap_urls else f"{self.base_url}/sitemap.xml",
                    'date_filters': ['/2025/06/', '/2025/07/', '/2025/08/', '/2025/09/'],
                    'batch_size': 100,
                    'max_concurrent': 10,
                    'sitemap_pattern': 'post-sitemap'
                }
            },
            
            CrawlingStrategy.SITEMAP_BASED: {
                'approach': 'sitemap_based',
                'reference_file': 'CarbonCaptureMagazine/1-sitemap_crawl.py or ThinkGeoEnergy/1-crawl_thinkgeoenergy.py',
                'key_features': [
                    'Fetch all URLs from sitemap',
                    'Exclude patterns (about, contact, etc.)',
                    'Batch processing with delays',
                    'Simple content filtering'
                ],
                'config_params': {
                    'sitemap_url': self.sitemap_urls[0] if self.sitemap_urls else f"{self.base_url}/sitemap.xml",
                    'batch_size': 50,
                    'delay_between_batches': 5,
                    'exclude_patterns': ['*/about*', '*/contact*', '*.pdf', '*.jpg', '*.png']
                }
            },
            
            CrawlingStrategy.PAGINATION_THEN_ARTICLES: {
                'approach': 'pagination_then_articles',
                'reference_file': 'CoolCoalition/1-crawl_coolcoalition.py',
                'key_features': [
                    'Phase 1: Extract article URLs from pagination pages',
                    'Phase 2: Crawl discovered articles',
                    'Two-phase approach for clean separation',
                    'Rate limiting between phases'
                ],
                'config_params': {
                    'base_url': self.url,
                    'max_pages': 10,
                    'delay_between_pages': 2,
                    'delay_between_articles': 1,
                    'pagination_pattern': '?_page=' if '?_page=' in self.url else '/page/'
                }
            },
            
            CrawlingStrategy.CATEGORY_DEEP_CRAWL: {
                'approach': 'category_deep_crawl',
                'reference_file': 'TechCrunch-Climate/1-crawl_techcrunch.py',
                'key_features': [
                    'Start from category page',
                    'Follow links to discover articles',
                    'BFS deep crawl with filters',
                    'Track crawl depth'
                ],
                'config_params': {
                    'start_url': self.url,
                    'max_depth': 20,
                    'max_pages': 500,
                    'include_external': False,
                    'url_patterns': [f"*{self.base_domain}/category/*", f"*{self.base_domain}/20*"]
                }
            },
            
            CrawlingStrategy.DEEP_CRAWL_FOCUSED_PATH: {
                'approach': 'deep_crawl_focused_path',
                'reference_file': 'CanaryMedia/1-crawl_canarymedia.py or crawl_specific.py',
                'key_features': [
                    'Deep crawl with focused path filtering',
                    'Domain and URL pattern filters',
                    'Content pruning for relevance',
                    'Organized output by theme'
                ],
                'config_params': {
                    'start_url': self.url,
                    'base_domain': self.base_domain,
                    'focused_path': self.path,
                    'max_depth': 5,
                    'pruning_threshold': 0.48,
                    'url_patterns': [
                        f"*{self.base_domain}{self.path}*",
                        f"*{self.path}/p[0-9]*",
                        f"*{self.path}?*page=*"
                    ]
                }
            },
            
            CrawlingStrategy.SIMPLE_DEEP_CRAWL: {
                'approach': 'simple_deep_crawl',
                'reference_file': 'crawl.py',
                'key_features': [
                    'Basic BFS deep crawl',
                    'Minimal configuration',
                    'Stay within domain',
                    'No external links'
                ],
                'config_params': {
                    'start_url': self.url,
                    'max_depth': 1,
                    'include_external': False
                }
            },
            
            CrawlingStrategy.SINGLE_PAGE_CRAWL: {
                'approach': 'single_page',
                'reference_file': 'N/A - Basic AsyncWebCrawler usage',
                'key_features': [
                    'Crawl single page only',
                    'No link following',
                    'Fast and simple',
                    'Good for testing'
                ],
                'config_params': {
                    'url': self.url,
                    'follow_links': False
                }
            }
        }
        
        return templates.get(strategy, {})
    
    def _print_analysis_summary(self, analysis: Dict):
        """Print a formatted summary of the analysis"""
        print("\n" + "="*80)
        print("ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\n📊 Site Characteristics:")
        print(f"  • Has Sitemap: {'Yes ✓' if analysis['has_sitemap'] else 'No ✗'}")
        if analysis['sitemap_urls']:
            print(f"    Sitemaps found: {', '.join(analysis['sitemap_urls'])}")
        print(f"  • Has Pagination: {'Yes ✓' if analysis['has_pagination'] else 'No ✗'}")
        print(f"  • Has Categories: {'Yes ✓' if analysis['has_categories'] else 'No ✗'}")
        print(f"  • Date Structure: {'Yes ✓' if analysis['has_date_structure'] else 'No ✗'}")
        print(f"  • Topic Specific: {'Yes ✓' if analysis['is_topic_specific'] else 'No ✗'}")
        
        print(f"\n🎯 Recommended Strategy: {analysis['recommended_strategy'].upper()}")
        print(f"\n💡 Explanation:")
        print(f"  {analysis['strategy_explanation']}")
        
        config = analysis['config_template']
        if config:
            print(f"\n📁 Reference Implementation:")
            print(f"  {config.get('reference_file', 'N/A')}")
            
            print(f"\n⚙️  Key Features:")
            for feature in config.get('key_features', []):
                print(f"  • {feature}")
            
            print(f"\n🔧 Recommended Configuration:")
            for key, value in config.get('config_params', {}).items():
                if isinstance(value, list):
                    print(f"  • {key}:")
                    for item in value:
                        print(f"      - {item}")
                else:
                    print(f"  • {key}: {value}")


def generate_crawler_code(analysis: Dict) -> str:
    """Generate ready-to-use crawler code based on analysis"""
    
    strategy = analysis['recommended_strategy']
    config = analysis['config_template']
    
    code_templates = {
        CrawlingStrategy.SITEMAP_WITH_DATE_FILTER: f'''
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
import requests
import xml.etree.ElementTree as ET

# Based on analysis of: {analysis['url']}

async def main():
    # Configuration
    sitemap_url = "{config['config_params']['sitemap_url']}"
    date_filters = {config['config_params']['date_filters']}
    output_dir = "crawl_output"
    
    # Fetch URLs from sitemap
    print(f"Fetching sitemap: {{sitemap_url}}")
    response = requests.get(sitemap_url, headers={{'User-Agent': 'Mozilla/5.0'}}, timeout=30)
    root = ET.fromstring(response.content)
    
    # Extract URLs and filter by date
    ns = {{'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}}
    all_urls = [loc.text for loc in root.findall('.//ns:url/ns:loc', ns)]
    filtered_urls = [u for u in all_urls if any(df in u for df in date_filters)]
    
    print(f"Found {{len(filtered_urls)}} URLs matching date filters")
    
    # Configure crawler
    config = CrawlerRunConfig(
        verbose=True,
        exclude_external_links=True,
        excluded_tags=['nav', 'footer', 'header', 'aside']
    )
    
    # Crawl in batches
    async with AsyncWebCrawler() as crawler:
        for url in filtered_urls[:10]:  # Limit for testing
            print(f"Crawling: {{url}}")
            result = await crawler.arun(url, config=config)
            if result.success:
                print(f"  ✓ Success")

if __name__ == "__main__":
    asyncio.run(main())
''',

        CrawlingStrategy.SITEMAP_BASED: f'''
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
import requests
import xml.etree.ElementTree as ET

# Based on analysis of: {analysis['url']}

async def main():
    # Configuration
    sitemap_url = "{config['config_params']['sitemap_url']}"
    output_dir = "crawl_output"
    exclude_patterns = {config['config_params']['exclude_patterns']}
    
    # Fetch sitemap URLs
    print(f"Fetching sitemap: {{sitemap_url}}")
    response = requests.get(sitemap_url, headers={{'User-Agent': 'Mozilla/5.0'}})
    root = ET.fromstring(response.content)
    ns = {{'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}}
    
    urls = [loc.text for loc in root.findall('.//ns:url/ns:loc', ns)]
    print(f"Found {{len(urls)}} URLs in sitemap")
    
    # Configure crawler
    config = CrawlerRunConfig(
        verbose=True,
        exclude_external_links=True
    )
    
    # Crawl URLs
    async with AsyncWebCrawler() as crawler:
        for url in urls[:10]:  # Limit for testing
            print(f"Crawling: {{url}}")
            result = await crawler.arun(url, config=config)
            if result.success:
                print(f"  ✓ Success")

if __name__ == "__main__":
    asyncio.run(main())
''',

        CrawlingStrategy.DEEP_CRAWL_FOCUSED_PATH: f'''
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import FilterChain, DomainFilter, URLPatternFilter

# Based on analysis of: {analysis['url']}

async def main():
    url = "{analysis['url']}"
    
    # Configure filters
    filter_chain = FilterChain([
        DomainFilter(allowed_domains=["{analysis['base_domain']}"]),
        URLPatternFilter(patterns={config['config_params']['url_patterns']})
    ])
    
    # Configure deep crawl
    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth={config['config_params']['max_depth']},
            filter_chain=filter_chain,
            include_external=False
        ),
        verbose=True,
        exclude_external_links=True,
        excluded_tags=['nav', 'footer', 'header', 'aside']
    )
    
    # Crawl
    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun(url, config=config)
        print(f"Crawled {{len(results)}} pages")
        
        for result in results:
            if result.success:
                print(f"✓ {{result.url}}")

if __name__ == "__main__":
    asyncio.run(main())
''',

        CrawlingStrategy.PAGINATION_THEN_ARTICLES: f'''
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Based on analysis of: {analysis['url']}

async def extract_articles(crawler, page_url, config):
    """Extract article URLs from pagination page"""
    result = await crawler.arun(page_url, config=config)
    if not result.success:
        return []
    
    soup = BeautifulSoup(result.html, 'html.parser')
    article_urls = set()
    
    for link in soup.find_all('a', href=True):
        full_url = urljoin(page_url, link['href'])
        # Add your article detection logic here
        article_urls.add(full_url)
    
    return list(article_urls)

async def main():
    base_url = "{analysis['url']}"
    max_pages = {config['config_params']['max_pages']}
    
    config = CrawlerRunConfig(verbose=True)
    
    async with AsyncWebCrawler() as crawler:
        # Phase 1: Extract article URLs
        all_articles = set()
        for page_num in range(1, max_pages + 1):
            page_url = f"{{base_url}}{config['config_params']['pagination_pattern']}{{page_num}}"
            print(f"Extracting from: {{page_url}}")
            articles = await extract_articles(crawler, page_url, config)
            all_articles.update(articles)
        
        print(f"\\nFound {{len(all_articles)}} articles")
        
        # Phase 2: Crawl articles
        for article_url in list(all_articles)[:10]:  # Limit for testing
            print(f"Crawling: {{article_url}}")
            result = await crawler.arun(article_url, config=config)
            if result.success:
                print(f"  ✓ Success")

if __name__ == "__main__":
    asyncio.run(main())
'''
    }
    
    return code_templates.get(strategy, "# No template available for this strategy")


def main():
    """Main function for command-line usage"""
    import sys
    
    print("\n" + "="*80)
    print("CRAWL4AI - INTELLIGENT URL CRAWLER DECISION TREE")
    print("="*80)
    
    if len(sys.argv) < 2:
        print("\nUsage: python url_crawler_decision_tree.py <URL>")
        print("\nExample:")
        print("  python url_crawler_decision_tree.py https://www.example.com/articles/solar")
        sys.exit(1)
    
    target_url = sys.argv[1]
    
    # Analyze URL
    analyzer = URLAnalyzer(target_url)
    analysis = analyzer.analyze()
    
    # Generate code
    print("\n" + "="*80)
    print("GENERATED CRAWLER CODE")
    print("="*80)
    code = generate_crawler_code(analysis)
    print(code)
    
    # Save to file
    output_file = "generated_crawler.py"
    with open(output_file, 'w') as f:
        f.write(code)
    
    print("\n" + "="*80)
    print(f"✅ Code saved to: {output_file}")
    print("="*80)
    
    print("\n📝 Next Steps:")
    print("  1. Review the generated code")
    print("  2. Adjust parameters as needed")
    print("  3. Run: python generated_crawler.py")
    print("  4. Check crawl_output/ directory for results")


if __name__ == "__main__":
    main()

