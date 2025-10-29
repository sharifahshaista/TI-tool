"""
Intelligent Markdown Post-Processor
Automatically extracts metadata from crawled markdown files, filters by date,
cleans content, and exports to CSV/JSON
"""

import re
import os
import csv
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
import pandas as pd

from agents.markdown_metadata_detector import MarkdownMetadataDetector, MetadataPattern


class MarkdownPostProcessor:
    """
    Post-processes crawled markdown files with intelligent metadata extraction
    """
    
    def __init__(self, patterns: Optional[MetadataPattern] = None, blocked_urls: Optional[set] = None):
        """
        Initialize post-processor
        
        Args:
            patterns: Pre-defined metadata patterns. If None, will auto-detect.
            blocked_urls: Set of URLs to skip (category/landing pages)
        """
        self.patterns = patterns
        self.detector = MarkdownMetadataDetector()
        self.using_fallback_patterns = False  # Track if using non-AI fallback patterns
        self.blocked_urls = blocked_urls or set()  # URLs to skip (category pages, etc.)
    
    def process_folder(
        self,
        folder_path: Path,
        output_dir: Path = Path("processed_content"),
        date_filter_months: Optional[int] = None,
        auto_detect: bool = True,
        pattern_file: Optional[Path] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Process all markdown files in folder
        
        Args:
            folder_path: Path to folder containing markdown files
            output_dir: Path to save processed output
            date_filter_months: Filter articles within last N months (None = no filter)
            auto_detect: Whether to auto-detect patterns using AI
            pattern_file: Path to saved pattern JSON file
            
        Returns:
            Tuple of (DataFrame with extracted metadata, processing stats)
        """
        folder_path = Path(folder_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get or detect patterns
        if pattern_file and pattern_file.exists():
            print(f"Loading patterns from: {pattern_file}")
            self.patterns = self.detector.load_patterns(pattern_file)
            self.using_fallback_patterns = False
        elif self.patterns is None and auto_detect:
            print("Auto-detecting patterns from markdown files...")
            md_files = list(folder_path.glob('*.md'))
            if not md_files:
                raise ValueError(f"No markdown files found in {folder_path}")
            self.patterns = self.detector.analyze_samples(md_files, num_samples=3)
            # Save detected patterns
            pattern_output = output_dir / "detected_patterns.json"
            self.detector.save_patterns(self.patterns, pattern_output)
            self.using_fallback_patterns = False
        elif self.patterns is None:
            # Use fallback patterns if no patterns provided and auto_detect=False
            print("⚠️ No patterns provided and auto_detect=False. Using fallback patterns...")
            print("For better results, enable 'AI Pattern Detection' or provide a saved patterns file.")
            print("Titles will be extracted from markdown filenames in sentence case.")
            self.patterns = self.detector._get_fallback_patterns()
            self.using_fallback_patterns = True
        
        # Find all markdown files
        md_files = list(folder_path.rglob('*.md'))
        print(f"\nFound {len(md_files)} markdown files")
        
        # Process each file
        all_metadata = []
        stats = {
            'total_files': len(md_files),
            'processed': 0,
            'skipped_old': 0,
            'skipped_error': 0,
            'date_cutoff': None
        }
        
        # Set date cutoff if filtering
        if date_filter_months:
            stats['date_cutoff'] = datetime.now() - timedelta(days=date_filter_months * 30)
            print(f"Filtering articles from last {date_filter_months} months (after {stats['date_cutoff'].strftime('%Y-%m-%d')})")
        
        for md_file in md_files:
            try:
                metadata = self.extract_metadata(md_file)
                
                # Skip blocked URLs (category/landing pages)
                if metadata.get('url') and metadata['url'] in self.blocked_urls:
                    stats['skipped_old'] += 1  # Count as skipped
                    print(f"  Skipping (blocked): {md_file.name}")
                    continue
                
                # Date filtering
                if date_filter_months and metadata.get('date'):
                    pub_date = self._parse_date(metadata['date'])
                    if pub_date and pub_date < stats['date_cutoff']:
                        stats['skipped_old'] += 1
                        print(f"  Skipping (old): {md_file.name}")
                        continue
                
                all_metadata.append(metadata)
                stats['processed'] += 1
                print(f"  ✓ Processed: {md_file.name}")
                
            except Exception as e:
                stats['skipped_error'] += 1
                print(f"  ✗ Error: {md_file.name} - {str(e)}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(all_metadata)
        
        # Save outputs
        if not df.empty:
            timestamp = datetime.now().strftime('%Y%m%d')
            folder_name = folder_path.name
            
            # CSV output
            csv_file = output_dir / f"{folder_name}_{timestamp}.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8')
            print(f"\n✓ CSV saved to: {csv_file}")
            
            # JSON output
            json_file = output_dir / f"{folder_name}_{timestamp}.json"
            df.to_json(json_file, orient='records', indent=2)
            print(f"✓ JSON saved to: {json_file}")
            
            # Stats file
            stats_file = output_dir / f"{folder_name}_{timestamp}_stats.txt"
            with open(stats_file, 'w', encoding='utf-8') as f:
                f.write("Markdown Post-Processing Summary\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Source folder: {folder_path}\n")
                f.write(f"Processing date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"Total markdown files: {stats['total_files']}\n")
                f.write(f"Successfully processed: {stats['processed']}\n")
                f.write(f"Skipped (old): {stats['skipped_old']}\n")
                f.write(f"Skipped (error): {stats['skipped_error']}\n")
                if stats['date_cutoff']:
                    f.write(f"\nDate filter: Articles after {stats['date_cutoff'].strftime('%Y-%m-%d')}\n")
                f.write(f"\nOutput files:\n")
                f.write(f"  - CSV: {csv_file.name}\n")
                f.write(f"  - JSON: {json_file.name}\n")
            
            print(f"✓ Stats saved to: {stats_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("PROCESSING SUMMARY")
        print("="*80)
        print(f"Total files: {stats['total_files']}")
        print(f"Processed: {stats['processed']}")
        print(f"Skipped (old): {stats['skipped_old']}")
        print(f"Skipped (error): {stats['skipped_error']}")
        print("="*80)
        
        return df, stats
    
    def extract_metadata(self, md_file: Path) -> Dict:
        """
        Extract metadata from a single markdown file using detected patterns
        
        Args:
            md_file: Path to markdown file
            
        Returns:
            Dictionary with extracted metadata
        """
        with open(md_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        metadata = {
            'filename': md_file.name,
            'filepath': str(md_file),
            'url': None,
            'title': None,
            'author': None,
            'date': None,
            'content': None,
            'tags': None
        }
        
        # Extract URL
        if self.patterns.url_pattern:
            url_match = re.search(self.patterns.url_pattern, text, re.MULTILINE)
            if url_match:
                metadata['url'] = url_match.group(1).strip()
        
        # Extract title with improved logic
        if self.using_fallback_patterns:
            # For non-AI mode: Use filename directly in sentence case
            # This ensures consistency and avoids URL/heading extraction
            metadata['title'] = self._filename_to_title(md_file.stem)
        else:
            # For AI mode: Use intelligent extraction
            if self.patterns.title_pattern:
                title_match = re.search(self.patterns.title_pattern, text, re.MULTILINE)
                if title_match:
                    potential_title = title_match.group(1).strip()
                    # Ensure the matched title is not a URL
                    if not potential_title.startswith('http://') and not potential_title.startswith('https://'):
                        metadata['title'] = potential_title
            
            # Fallback: Try to extract title from first meaningful heading
            if not metadata['title']:
                metadata['title'] = self._extract_title_from_headings(text)
            
            # Final fallback: Convert URL or filename to title
            if not metadata['title']:
                if metadata['url']:
                    metadata['title'] = self._url_to_title(metadata['url'])
                else:
                    metadata['title'] = self._filename_to_title(md_file.stem)
        
        # Extract author
        if self.patterns.author_pattern:
            author_match = re.search(self.patterns.author_pattern, text, re.MULTILINE)
            if author_match:
                metadata['author'] = author_match.group(1).strip()
        
        # Extract date
        if self.patterns.date_pattern:
            date_match = re.search(self.patterns.date_pattern, text, re.MULTILINE)
            if date_match:
                metadata['date'] = date_match.group(1).strip()
        
        # Extract tags
        if self.patterns.tags_pattern:
            tag_matches = re.findall(self.patterns.tags_pattern, text, re.MULTILINE)
            if tag_matches:
                metadata['tags'] = ', '.join(tag_matches)
        
        # Extract content
        metadata['content'] = self._extract_content(text)
        
        # Clean content
        metadata['content'] = self._clean_content(metadata['content'])
        
        return metadata
    
    def _extract_content(self, text: str) -> str:
        """Extract main content using markers or fallback"""
        
        lines = text.splitlines()
        content_lines = []
        
        # Try marker-based extraction
        if self.patterns.content_start_marker or self.patterns.content_end_marker:
            in_content = False if self.patterns.content_start_marker else True
            
            for line in lines:
                # Check for start marker
                if self.patterns.content_start_marker and self.patterns.content_start_marker in line:
                    in_content = True
                    continue
                
                # Check for end marker
                if self.patterns.content_end_marker and self.patterns.content_end_marker in line:
                    in_content = False
                    break
                
                if in_content:
                    # Filter out crawl metadata even within content
                    if not self._is_crawl_metadata_line(line):
                        content_lines.append(line)
        else:
            # Fallback: Skip metadata at top, take everything else
            skip_lines = 0
            in_metadata_section = True
            
            for i, line in enumerate(lines):
                stripped = line.strip()
                
                # Skip initial metadata lines (URL, title, status, crawl info, etc.)
                if in_metadata_section:
                    # Check if this is a metadata line
                    if self._is_crawl_metadata_line(line):
                        skip_lines = i + 1
                    elif stripped.startswith('#') or stripped.startswith('---'):
                        skip_lines = i + 1
                    elif not stripped:  # Empty line
                        skip_lines = i + 1
                    else:
                        # First non-metadata content line found
                        in_metadata_section = False
                        break
            
            # Collect remaining content lines, filtering out any crawl metadata
            for line in lines[skip_lines:]:
                if not self._is_crawl_metadata_line(line):
                    content_lines.append(line)
        
        return '\n'.join(content_lines)
    
    def _is_crawl_metadata_line(self, line: str) -> bool:
        """Check if a line contains crawl metadata/status information"""
        stripped = line.strip().lower()
        
        # Patterns to identify crawl metadata lines
        crawl_metadata_indicators = [
            'status:',
            'crawled:',
            'crawl date:',
            'crawl time:',
            'last crawled:',
            'fetched:',
            'retrieved:',
            'crawl status:',
            'success:',
            'failed:',
            'http status:',
            'response code:',
            'crawler:',
            'user-agent:',
            'crawl4ai',
            'crawling',
            'scraped:',
            'extraction date:',
            'processed:',
            'generated by:',
            'created by crawler',
            'auto-generated',
        ]
        
        # Check if line starts with or contains crawl metadata indicators
        for indicator in crawl_metadata_indicators:
            if stripped.startswith(indicator) or f' {indicator}' in stripped:
                return True
        
        # Check for timestamp patterns (common in crawl metadata)
        # Pattern: YYYY-MM-DD HH:MM:SS or similar
        timestamp_pattern = r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}'
        if re.search(timestamp_pattern, stripped) and len(stripped) < 100:
            # Likely a timestamp metadata line (short line with timestamp)
            return True
        
        return False
    
    def _clean_content(self, content: str) -> str:
        """Clean content by removing noise patterns"""
        
        if not content:
            return ""
        
        # Apply noise removal patterns
        for pattern in self.patterns.noise_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove images (standard markdown)
        content = re.sub(r'!\[.*?\]\(https?://[^\)]+\)', '', content)
        
        # Keep link text but remove URLs
        content = re.sub(r'\[([^\]]+)\]\(https?://[^\)]+\)', r'\1', content)
        
        # Remove any remaining crawl metadata patterns
        # Remove lines with "Status: ..." or "Crawled: ..."
        content = re.sub(r'^(Status|Crawled|Fetched|Retrieved|Scraped|Processed|Generated by|Created by crawler):.*$', '', content, flags=re.MULTILINE | re.IGNORECASE)
        
        # Remove standalone timestamps
        content = re.sub(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}.*$', '', content, flags=re.MULTILINE)
        
        # Remove crawl4ai references
        content = re.sub(r'crawl4ai|auto-generated.*crawler', '', content, flags=re.IGNORECASE)
        
        # Remove excessive whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r' {2,}', ' ', content)
        
        # Remove leading/trailing whitespace on each line
        lines = [line.strip() for line in content.splitlines()]
        content = '\n'.join(line for line in lines if line)
        
        return content.strip()
    
    def _extract_title_from_headings(self, text: str) -> str:
        """
        Extract title from first meaningful heading in markdown
        Skips metadata headings and looks for actual content title
        """
        lines = text.splitlines()
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Skip the first few lines (likely metadata)
            if i < 6:
                continue
            
            # Look for H1 or H2 headings (# or ##)
            if stripped.startswith('# ') and not stripped.startswith('# http'):
                # Extract heading text
                title = stripped.lstrip('#').strip()
                
                # Validate it's not metadata
                if not self._is_crawl_metadata_line(line) and len(title) > 5:
                    return title
            
            elif stripped.startswith('## ') and not stripped.startswith('## http'):
                # Extract heading text
                title = stripped.lstrip('#').strip()
                
                # Validate it's not navigation or metadata
                skip_patterns = ['share', 'related', 'subscribe', 'follow', 'menu', 'navigation']
                if (not self._is_crawl_metadata_line(line) and 
                    len(title) > 5 and
                    not any(pattern in title.lower() for pattern in skip_patterns)):
                    return title
        
        return None
    
    def _filename_to_title(self, filename: str) -> str:
        """
        Convert filename to readable title with smart capitalization
        """
        # Replace underscores and hyphens with spaces
        title = filename.replace('_', ' ').replace('-', ' ')
        
        # Split into words
        words = title.split()
        
        # Handle special cases
        special_upper = {'ai', 'ev', 'suv', 'ceo', 'cfo', 'us', 'usa', 'uk', 'eu', 'co2', 'h2', 'ch4', 'nh3'}
        lower_words = {'and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'as'}
        
        title_words = []
        for i, word in enumerate(words):
            if word.lower() in special_upper:
                title_words.append(word.upper())
            elif word.lower() in lower_words and i != 0:
                title_words.append(word.lower())
            else:
                title_words.append(word.capitalize())
        
        return ' '.join(title_words)
    
    def _url_to_title(self, url: str) -> str:
        """Convert URL slug to readable title"""
        parsed = urlparse(url)
        slug = parsed.path.rstrip('/').split('/')[-1]
        
        # Remove file extensions
        slug = slug.replace('.html', '').replace('.md', '')
        
        # Use the same logic as _filename_to_title
        return self._filename_to_title(slug)
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime object"""
        
        if not date_str:
            return None
        
        # Try common formats
        formats = [
            '%d %B %Y',  # 23 October 2025
            '%d %b %Y',  # 23 Oct 2025
            '%B %d, %Y',  # October 23, 2025
            '%b %d, %Y',  # Oct 23, 2025
            '%Y-%m-%d',  # 2025-10-23
            '%d/%m/%Y',  # 23/10/2025
            '%m/%d/%Y',  # 10/23/2025
            '%Y/%m/%d',  # 2025/10/23
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        
        return None
    
    def add_blocked_urls(self, urls: set):
        """
        Add URLs to blocked list (category/landing pages)
        
        Args:
            urls: Set of URLs to block
        """
        self.blocked_urls.update(urls)
    
    def detect_category_pages(self, folder_path: Path) -> set:
        """
        Auto-detect category/landing pages by analyzing URL patterns
        
        Args:
            folder_path: Path to folder containing markdown files
            
        Returns:
            Set of detected category page URLs
        """
        category_urls = set()
        category_indicators = [
            '/articles/',
            '/category/',
            '/topics/',
            '/tags/',
            '/archive/',
        ]
        
        md_files = list(folder_path.glob('*.md'))[:20]  # Sample first 20 files
        
        for md_file in md_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Extract URL
                url_match = re.search(r'^#\s+(https?://[^\s]+)', text, re.MULTILINE)
                if url_match:
                    url = url_match.group(1).strip()
                    
                    # Check if URL looks like a category page
                    # Category pages typically end with a category slug (no specific article identifier)
                    parsed_url = urlparse(url)
                    path_parts = [p for p in parsed_url.path.split('/') if p]
                    
                    # If path has category indicator and ends without numeric/date identifier
                    for indicator in category_indicators:
                        if indicator.strip('/') in path_parts:
                            # Check if last part looks like a category (not an article)
                            if path_parts and not re.search(r'\d', path_parts[-1]):
                                category_urls.add(url)
                                break
            except Exception:
                continue
        
        return category_urls


def main():
    """Command-line interface for post-processor"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Intelligent Markdown Post-Processor with AI-powered metadata extraction"
    )
    parser.add_argument(
        'folder',
        type=str,
        help='Path to folder containing markdown files'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='processed_content',
        help='Output directory for processed files (default: processed_content)'
    )
    parser.add_argument(
        '-m', '--months',
        type=int,
        default=None,
        help='Filter articles from last N months (default: no filter)'
    )
    parser.add_argument(
        '-p', '--patterns',
        type=str,
        default=None,
        help='Path to saved pattern JSON file (default: auto-detect)'
    )
    parser.add_argument(
        '--no-auto-detect',
        action='store_true',
        help='Disable auto-detection of patterns'
    )
    
    args = parser.parse_args()
    
    folder_path = Path(args.folder)
    output_dir = Path(args.output)
    
    if not folder_path.exists():
        print(f"Error: Folder '{folder_path}' does not exist")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("INTELLIGENT MARKDOWN POST-PROCESSOR")
    print("="*80)
    print(f"Source folder: {folder_path}")
    print(f"Output directory: {output_dir}")
    if args.months:
        print(f"Date filter: Last {args.months} months")
    print("="*80 + "\n")
    
    # Process
    processor = MarkdownPostProcessor()
    
    pattern_file = Path(args.patterns) if args.patterns else None
    
    df, stats = processor.process_folder(
        folder_path=folder_path,
        output_dir=output_dir,
        date_filter_months=args.months,
        auto_detect=not args.no_auto_detect,
        pattern_file=pattern_file
    )
    
    # Display sample
    if not df.empty:
        print("\n" + "="*80)
        print("SAMPLE EXTRACTED DATA")
        print("="*80)
        sample = df.iloc[0]
        print(f"URL: {sample.get('url', 'N/A')}")
        print(f"Title: {sample.get('title', 'N/A')}")
        print(f"Author: {sample.get('author', 'N/A')}")
        print(f"Date: {sample.get('date', 'N/A')}")
        print(f"Tags: {sample.get('tags', 'N/A')}")
        content_preview = sample.get('content', '')
        if content_preview:
            print(f"Content preview: {content_preview[:200]}...")
        print("="*80)


if __name__ == "__main__":
    main()

