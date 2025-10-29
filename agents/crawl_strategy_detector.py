"""
Auto-detection system for optimal web crawling strategy.
Tests different strategies and recommends the best approach.
"""

import asyncio
import httpx
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse
from typing import Dict, Optional, List
import re
from bs4 import BeautifulSoup


class CrawlStrategyDetector:
    """Detect the best crawling strategy for a given URL."""
    
    def __init__(self, url: str):
        self.url = url
        self.parsed_url = urlparse(url)
        self.base_url = f"{self.parsed_url.scheme}://{self.parsed_url.netloc}"
        self.results = {}
    
    async def detect_best_strategy(self) -> Dict:
        """
        Test all strategies and return the best one with justification.
        
        Returns:
            {
                'recommended_strategy': str,
                'justification': str,
                'confidence': str (high/medium/low),
                'details': dict with test results for each strategy
            }
        """
        print("🔍 Starting auto-detection...")
        
        # Test strategies in priority order
        await self._test_pagination()
        await self._test_deep_crawl()
        await self._test_sitemap()
        await self._test_simple_discovery()
        await self._test_category()
        
        # Determine best strategy
        return self._determine_best_strategy()
    
    async def _test_pagination(self) -> Dict:
        """Test if pagination pattern exists."""
        print("  📄 Testing Pagination strategy...")
        
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                response = await client.get(self.url, headers={'User-Agent': 'Mozilla/5.0'})
                
                if response.status_code != 200:
                    self.results['pagination'] = {
                        'viable': False,
                        'reason': f"HTTP {response.status_code}",
                        'score': 0
                    }
                    return self.results['pagination']
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Look for pagination indicators
                pagination_patterns = [
                    r'/page/\d+',      # /page/2, /page/3
                    r'\?page=\d+',     # ?page=2, ?page=3
                    r'/p/\d+',         # /p/2, /p/3
                    r'/p\d+',          # /p2, /p3 (Quaise.com style)
                    r'&page=\d+',      # &page=2, &page=3
                    r'/\d+/$'          # /2/, /3/
                ]
                
                pagination_found = False
                pagination_type = None
                
                # Check links for pagination patterns
                links = soup.find_all('a', href=True)
                for link in links:
                    href = link.get('href', '')
                    for pattern in pagination_patterns:
                        if re.search(pattern, href):
                            pagination_found = True
                            pagination_type = pattern
                            break
                    if pagination_found:
                        break
                
                # Check for pagination keywords in classes/ids
                pagination_keywords = ['pagination', 'pager', 'page-numbers', 'next-page', 'prev-page']
                pagination_elements = 0
                for keyword in pagination_keywords:
                    pagination_elements += len(soup.find_all(class_=re.compile(keyword, re.I)))
                    pagination_elements += len(soup.find_all(id=re.compile(keyword, re.I)))
                
                # Count article-like links
                article_count = len([l for l in links if self._looks_like_article(l.get('href', ''))])
                
                if pagination_found and article_count > 5:
                    self.results['pagination'] = {
                        'viable': True,
                        'reason': f"Found pagination pattern ({pagination_type}) with {article_count} articles",
                        'score': 90,
                        'details': {
                            'pattern': pagination_type,
                            'article_count': article_count,
                            'pagination_elements': pagination_elements
                        }
                    }
                elif pagination_elements > 0 and article_count > 5:
                    self.results['pagination'] = {
                        'viable': True,
                        'reason': f"Found {pagination_elements} pagination elements with {article_count} articles",
                        'score': 75,
                        'details': {
                            'article_count': article_count,
                            'pagination_elements': pagination_elements
                        }
                    }
                else:
                    self.results['pagination'] = {
                        'viable': False,
                        'reason': f"No clear pagination ({article_count} articles found)",
                        'score': 20
                    }
        
        except Exception as e:
            self.results['pagination'] = {
                'viable': False,
                'reason': f"Error: {str(e)[:50]}",
                'score': 0
            }
        
        return self.results['pagination']
    
    async def _test_deep_crawl(self) -> Dict:
        """Test if deep crawl (BFS/DFS) is viable."""
        print("  🌳 Testing Deep Crawl strategy...")
        
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                response = await client.get(self.url, headers={'User-Agent': 'Mozilla/5.0'})
                
                if response.status_code != 200:
                    self.results['deep_crawl'] = {
                        'viable': False,
                        'reason': f"HTTP {response.status_code}",
                        'score': 0
                    }
                    return self.results['deep_crawl']
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Count internal links
                links = soup.find_all('a', href=True)
                internal_links = [l for l in links if self._is_internal_link(l.get('href', ''))]
                article_links = [l for l in internal_links if self._looks_like_article(l.get('href', ''))]
                
                # Check depth potential (subdirectories)
                depth_levels = set()
                for link in internal_links:
                    href = link.get('href', '')
                    path = urlparse(urljoin(self.base_url, href)).path
                    depth = path.count('/')
                    depth_levels.add(depth)
                
                max_depth = max(depth_levels) if depth_levels else 0
                
                if len(internal_links) > 20 and max_depth > 2:
                    self.results['deep_crawl'] = {
                        'viable': True,
                        'reason': f"Rich link structure: {len(internal_links)} internal links, max depth {max_depth}",
                        'score': 70,
                        'details': {
                            'internal_links': len(internal_links),
                            'article_links': len(article_links),
                            'max_depth': max_depth
                        }
                    }
                else:
                    self.results['deep_crawl'] = {
                        'viable': False,
                        'reason': f"Limited link structure: {len(internal_links)} links, depth {max_depth}",
                        'score': 30
                    }
        
        except Exception as e:
            self.results['deep_crawl'] = {
                'viable': False,
                'reason': f"Error: {str(e)[:50]}",
                'score': 0
            }
        
        return self.results['deep_crawl']
    
    async def _test_sitemap(self) -> Dict:
        """Test if sitemap is available."""
        print("  🗺️  Testing Sitemap strategy...")
        
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                # Try different sitemap locations (including content-specific sitemaps)
                content_keywords = ["post", "news", "article"]
                sitemap_urls = [
                    f"{self.base_url}/sitemap.xml",
                    f"{self.base_url}/sitemap_index.xml",
                    f"{self.base_url}/sitemap-index.xml"
                ]
                
                # Add content-specific sitemaps
                for keyword in content_keywords:
                    sitemap_urls.extend([
                        f"{self.base_url}/{keyword}-sitemap.xml",
                        f"{self.base_url}/{keyword}-sitemap1.xml",
                        f"{self.base_url}/{keyword}-sitemap2.xml"
                    ])
                
                found_sitemap = None
                url_count = 0
                
                for sitemap_url in sitemap_urls:
                    try:
                        response = await client.get(sitemap_url, headers={'User-Agent': 'Mozilla/5.0'})
                        if response.status_code == 200 and 'xml' in response.headers.get('content-type', ''):
                            # Parse sitemap
                            root = ET.fromstring(response.content)
                            
                            # Count URLs or sitemaps
                            ns = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
                            urls = root.findall('.//ns:url/ns:loc', ns)
                            sitemaps = root.findall('.//ns:sitemap/ns:loc', ns)
                            
                            count = len(urls) + len(sitemaps)
                            if count > 0:
                                found_sitemap = sitemap_url
                                url_count = count
                                break
                    except:
                        continue
                
                # Check robots.txt for sitemap
                if not found_sitemap:
                    try:
                        robots_response = await client.get(f"{self.base_url}/robots.txt")
                        if robots_response.status_code == 200:
                            for line in robots_response.text.split('\n'):
                                if line.lower().startswith('sitemap:'):
                                    sitemap_url = line.split(':', 1)[1].strip()
                                    sitemap_response = await client.get(sitemap_url)
                                    if sitemap_response.status_code == 200:
                                        found_sitemap = sitemap_url
                                        # Try to count URLs
                                        try:
                                            root = ET.fromstring(sitemap_response.content)
                                            ns = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
                                            urls = root.findall('.//ns:url/ns:loc', ns)
                                            sitemaps = root.findall('.//ns:sitemap/ns:loc', ns)
                                            url_count = len(urls) + len(sitemaps)
                                        except:
                                            url_count = 1  # At least one sitemap exists
                                        break
                    except:
                        pass
                
                if found_sitemap:
                    # Check if it's a content-specific sitemap for bonus scoring
                    sitemap_name = found_sitemap.split('/')[-1].lower()
                    is_content_sitemap = any(keyword in sitemap_name for keyword in content_keywords)
                    
                    base_score = 90
                    if is_content_sitemap:
                        base_score = 95  # Higher score for content-specific sitemaps
                        reason_suffix = f" (content-specific: {sitemap_name})"
                    else:
                        reason_suffix = ""
                    
                    self.results['sitemap'] = {
                        'viable': True,
                        'reason': f"Found sitemap at {found_sitemap.split('/')[-1]} with {url_count}+ URLs{reason_suffix}",
                        'score': base_score,
                        'details': {
                            'sitemap_url': found_sitemap,
                            'url_count': url_count,
                            'is_content_sitemap': is_content_sitemap
                        }
                    }
                else:
                    self.results['sitemap'] = {
                        'viable': False,
                        'reason': "No sitemap found",
                        'score': 0
                    }
        
        except Exception as e:
            self.results['sitemap'] = {
                'viable': False,
                'reason': f"Error: {str(e)[:50]}",
                'score': 0
            }
        
        return self.results['sitemap']
    
    async def _test_simple_discovery(self) -> Dict:
        """Test if simple discovery is viable (fallback)."""
        print("  🔍 Testing Simple Discovery strategy...")
        
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                response = await client.get(self.url, headers={'User-Agent': 'Mozilla/5.0'})
                
                if response.status_code != 200:
                    self.results['simple'] = {
                        'viable': False,
                        'reason': f"HTTP {response.status_code}",
                        'score': 0
                    }
                    return self.results['simple']
                
                soup = BeautifulSoup(response.text, 'html.parser')
                links = soup.find_all('a', href=True)
                internal_links = [l for l in links if self._is_internal_link(l.get('href', ''))]
                
                # Simple discovery is always viable as a fallback
                self.results['simple'] = {
                    'viable': True,
                    'reason': f"Basic crawl of {len(internal_links)} discoverable links (fallback option)",
                    'score': 50,  # Low score, only used as fallback
                    'details': {
                        'link_count': len(internal_links)
                    }
                }
        
        except Exception as e:
            self.results['simple'] = {
                'viable': True,  # Always viable as fallback
                'reason': "Fallback option (limited testing due to error)",
                'score': 40
            }
        
        return self.results['simple']
    
    async def _test_category(self) -> Dict:
        """Test if category-based crawling is viable."""
        print("  📁 Testing Category strategy...")
        
        try:
            # Check if URL path suggests a category
            path = self.parsed_url.path.lower()
            
            category_indicators = [
                '/articles/', '/category/', '/topics/', '/tags/',
                '/news/', '/blog/', '/posts/', '/archive/'
            ]
            
            is_category_path = any(indicator in path for indicator in category_indicators)
            
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                response = await client.get(self.url, headers={'User-Agent': 'Mozilla/5.0'})
                
                if response.status_code != 200:
                    self.results['category'] = {
                        'viable': False,
                        'reason': f"HTTP {response.status_code}",
                        'score': 0
                    }
                    return self.results['category']
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Count links that share the same path prefix
                links = soup.find_all('a', href=True)
                path_prefix = '/'.join(path.split('/')[:3])  # Get first 2-3 path segments
                
                matching_links = []
                for link in links:
                    href = link.get('href', '')
                    full_url = urljoin(self.url, href)
                    if self._is_internal_link(href) and path_prefix in full_url:
                        matching_links.append(href)
                
                if is_category_path and len(matching_links) > 10:
                    self.results['category'] = {
                        'viable': True,
                        'reason': f"Category URL detected with {len(matching_links)} related articles",
                        'score': 80,
                        'details': {
                            'category_path': path_prefix,
                            'matching_links': len(matching_links)
                        }
                    }
                elif len(matching_links) > 20:
                    self.results['category'] = {
                        'viable': True,
                        'reason': f"Focused content area with {len(matching_links)} related links",
                        'score': 65,
                        'details': {
                            'matching_links': len(matching_links)
                        }
                    }
                else:
                    self.results['category'] = {
                        'viable': False,
                        'reason': f"Not a clear category page ({len(matching_links)} matching links)",
                        'score': 25
                    }
        
        except Exception as e:
            self.results['category'] = {
                'viable': False,
                'reason': f"Error: {str(e)[:50]}",
                'score': 0
            }
        
        return self.results['category']
    
    def _determine_best_strategy(self) -> Dict:
        """Analyze test results and determine the best strategy."""
        
        # Priority order with scoring
        strategy_priority = [
            ('pagination', 'Pagination Crawl'),
            ('sitemap', 'Sitemap Crawl'),
            ('category', 'Category Crawl'),
            ('deep_crawl', 'Deep Crawl (BFS)'),
            ('simple', 'Simple Discovery')
        ]
        
        # Find highest scoring viable strategy
        best_strategy = None
        best_score = 0
        
        for strategy_key, strategy_name in strategy_priority:
            if strategy_key in self.results:
                result = self.results[strategy_key]
                if result['viable'] and result['score'] > best_score:
                    best_strategy = strategy_key
                    best_score = result['score']
        
        # Fallback to simple if nothing else works
        if not best_strategy:
            best_strategy = 'simple'
            best_score = 40
        
        # Map to CrawlMode
        strategy_map = {
            'pagination': 'Pagination Crawl',
            'sitemap': 'Sitemap Crawl',
            'category': 'Category Crawl',
            'deep_crawl': 'Deep Crawl (BFS)',
            'simple': 'Simple Discovery'
        }
        
        recommended = strategy_map[best_strategy]
        result_data = self.results.get(best_strategy, {})
        
        # Determine confidence
        if best_score >= 80:
            confidence = "HIGH"
            confidence_emoji = "🟢"
        elif best_score >= 60:
            confidence = "MEDIUM"
            confidence_emoji = "🟡"
        else:
            confidence = "LOW"
            confidence_emoji = "🟠"
        
        # Build justification
        justification = self._build_justification(best_strategy, result_data)
        
        return {
            'recommended_strategy': recommended,
            'justification': justification,
            'confidence': confidence,
            'confidence_emoji': confidence_emoji,
            'score': best_score,
            'details': self.results,
            'tested_url': self.url
        }
    
    def _build_justification(self, strategy: str, result_data: Dict) -> str:
        """Build a human-readable justification for the strategy choice."""
        
        reason = result_data.get('reason', 'No specific reason')
        details = result_data.get('details', {})
        
        justifications = {
            'sitemap': f"**Sitemap-Based Crawl** is recommended because:\n"
                      f"- {reason}\n"
                      f"- Most efficient method for comprehensive coverage\n"
                      f"- Structured, organized URL discovery\n"
                      f"- Typically contains all published articles",
            
            'pagination': f"**Pagination Crawl** is recommended because:\n"
                         f"- {reason}\n"
                         f"- Site uses pagination for article listing\n"
                         f"- Efficient article discovery through paginated pages\n"
                         f"- Follows natural site navigation",
            
            'category': f"**Category Crawl** is recommended because:\n"
                       f"- {reason}\n"
                       f"- URL indicates focused content category\n"
                       f"- Enables targeted crawling within topic\n"
                       f"- Content filtering for relevance",
            
            'deep_crawl': f"**Deep Crawl (BFS)** is recommended because:\n"
                         f"- {reason}\n"
                         f"- Rich internal linking structure detected\n"
                         f"- Breadth-first exploration of site hierarchy\n"
                         f"- Discovers content through link following",
            
            'simple': f"**Simple Discovery** is recommended because:\n"
                     f"- {reason}\n"
                     f"- Basic crawl suitable for simple site structures\n"
                     f"- Fallback option when specialized strategies not viable\n"
                     f"- Direct link discovery from starting page"
        }
        
        return justifications.get(strategy, f"Recommended based on: {reason}")
    
    def _is_internal_link(self, href: str) -> bool:
        """Check if a link is internal to the domain."""
        if not href or href.startswith('#') or href.startswith('mailto:') or href.startswith('tel:'):
            return False
        
        try:
            full_url = urljoin(self.base_url, href)
            parsed = urlparse(full_url)
            return parsed.netloc == self.parsed_url.netloc or parsed.netloc == ''
        except:
            return False
    
    def _looks_like_article(self, href: str) -> bool:
        """Heuristic to check if a link looks like an article."""
        if not href:
            return False
        
        # Exclude common non-article patterns
        exclude_patterns = [
            r'/author/', r'/tag/', r'/category/', r'/page/',
            r'/about', r'/contact', r'/privacy', r'/terms',
            r'\.(jpg|png|gif|pdf|zip)', r'/feed', r'/rss'
        ]
        
        for pattern in exclude_patterns:
            if re.search(pattern, href.lower()):
                return False
        
        # Look for article-like patterns
        article_patterns = [
            r'/\d{4}/\d{2}/',  # Date-based URLs
            r'/articles?/',
            r'/posts?/',
            r'/news/',
            r'/blog/',
            r'-\d+$'  # Ends with ID
        ]
        
        for pattern in article_patterns:
            if re.search(pattern, href.lower()):
                return True
        
        # If URL has reasonable length and depth, consider it
        path = urlparse(href).path
        if 3 < len(path.split('/')) < 8 and len(path) > 20:
            return True
        
        return False

