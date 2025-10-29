"""
Preset-based metadata extractors for specific news sites.
These extractors use hardcoded patterns from the reference implementations.
NO AI DETECTION - strictly pattern-based extraction.
"""

import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from urllib.parse import urlparse


# ========================================
# THINKGEOENERGY EXTRACTOR
# ========================================
class ThinkGeoEnergyExtractor:
    """Extract metadata from ThinkGeoEnergy markdown files."""
    
    @staticmethod
    def extract_url(lines: List[str]) -> str:
        """Extract URL from the first line after #"""
        for line in lines:
            if line.strip().startswith('# https://') or line.strip().startswith('# http://'):
                return line.strip()[2:].strip()
        return ""
    
    @staticmethod
    def extract_title(lines: List[str]) -> str:
        """Extract title after the # symbol"""
        for line in lines:
            stripped = line.strip()
            # Skip the URL line
            if stripped.startswith('# https://') or stripped.startswith('# http://'):
                continue
            # Find the actual title line (starts with # but not a URL)
            if stripped.startswith('# ') and not stripped.startswith('# http'):
                return stripped[2:].strip()
        return ""
    
    @staticmethod
    def extract_author(lines: List[str]) -> str:
        """Extract author name between [] and ** pair"""
        for line in lines:
            # Look for pattern [**Author Name**]
            match = re.search(r'\[\*\*([^\]]+)\*\*\]', line)
            if match:
                return match.group(1).strip()
        return ""
    
    @staticmethod
    def extract_date(lines: List[str]) -> str:
        """Extract date after the author profile link"""
        for line in lines:
            # Find the author link line
            if '**' in line and '[' in line and ']' in line:
                # Look for date pattern after the author link
                date_match = re.search(r'(\d{1,2}\s+\w{3}\s+\d{4})', line)
                if date_match:
                    return date_match.group(1).strip()
        return ""
    
    @staticmethod
    def extract_content(lines: List[str]) -> str:
        """Extract main content text between title and 'Source:'"""
        content_lines = []
        title_found = False
        
        for line in lines:
            stripped = line.strip()
            
            # Start collecting after we find the title (not the URL line)
            if not title_found and stripped.startswith('# ') and not stripped.startswith('# http'):
                title_found = True
                continue
            
            # Stop at "Source:"
            if stripped.startswith('Source:'):
                break
            
            # Start collecting content after title is found
            if title_found:
                # Skip images, links to images, social media, advertisements, javascript
                if any(skip in stripped.lower() for skip in ['![', 'javascript:', '](https://ads.', 'social_', '.jpg', '.png', '.gif', 'delivery/avw', 'delivery/lg']):
                    continue
                
                # Skip navigation and header elements
                if stripped.startswith('##') or stripped.startswith('[Homepage]') or stripped.startswith('Status:'):
                    continue
                
                # Skip empty lines at the start
                if not content_lines and not stripped:
                    continue
                
                # Add non-empty content lines
                if stripped and not stripped.startswith('[') or (stripped.startswith('[') and '**' in stripped):
                    content_lines.append(stripped)
        
        # Join and clean up the content
        content = ' '.join(content_lines)
        
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        
        return content
    
    @staticmethod
    def extract_tags(lines: List[str]) -> str:
        """Extract tags from [] under 'Tags' section"""
        tags = []
        in_tags_section = False
        
        for line in lines:
            stripped = line.strip()
            
            # Find the Tags section
            if stripped == 'Tags':
                in_tags_section = True
                continue
            
            # Stop when we hit the next section (like SHARE)
            if in_tags_section and (stripped in ['SHARE', ''] or stripped.startswith('##')):
                break
            
            # Extract tags
            if in_tags_section:
                # Find all [tag] patterns
                tag_matches = re.findall(r'\[([^\]]+)\]', line)
                for tag in tag_matches:
                    # Skip URLs
                    if 'http' not in tag.lower():
                        tags.append(tag.strip())
        
        return ', '.join(tags)
    
    @classmethod
    def extract(cls, file_path: Path) -> Dict[str, str]:
        """Extract all metadata from a ThinkGeoEnergy markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            return {
                'url': cls.extract_url(lines),
                'title': cls.extract_title(lines),
                'author': cls.extract_author(lines),
                'date': cls.extract_date(lines),
                'content': cls.extract_content(lines),
                'tags': cls.extract_tags(lines)
            }
        except Exception as e:
            print(f"Error extracting from {file_path}: {e}")
            return {}


# ========================================
# CANARYMEDIA EXTRACTOR
# ========================================
class CanaryMediaExtractor:
    """Extract metadata from CanaryMedia markdown files."""
    
    @staticmethod
    def extract(file_path: Path) -> Dict[str, str]:
        """Parse a CanaryMedia markdown file into structured metadata."""
        data = {
            "url": None,
            "title": None,
            "date": None,
            "author": None,
            "content": None,
            "tags": ""
        }

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        lines = text.splitlines()

        # --- URL ---
        for line in lines:
            if line.startswith("# http"):
                data["url"] = line.lstrip("# ").strip()
                break

        # --- Title ---
        title_match = re.search(r'scrolledValue\s*["\']>\s*#\s+(.*)', text)
        if title_match:
            data["title"] = title_match.group(1).strip()

        # --- Author ---
        author_match = re.search(
            r"\[([^\]]+)\]\(https://www\.canarymedia\.com/about/people/[^\)]+\)", text
        )
        if author_match:
            data["author"] = author_match.group(1).strip()

        # --- Date ---
        date_match = re.search(r"\b(\d{1,2}\s+[A-Z][a-z]+\s+\d{4})\b", text)
        if date_match:
            data["date"] = date_match.group(1).strip()

        # --- Tags ---
        tag_matches = re.findall(r"\*\s*\[([^\]]+)\]\([^)]+\)", text)
        if tag_matches:
            data["tags"] = ", ".join(t.strip() for t in tag_matches)

        # --- Content ---
        content_section = []
        in_content = False
        for line in lines:
            if line.strip().startswith("scrolledValue"):
                in_content = True
                continue
            if in_content:
                # Stop at tags section
                if line.strip().startswith("* ["):
                    break

                # Skip title line (starts with '# ')
                if line.strip().startswith("# "):
                    continue
                # Skip author line
                if re.match(r"^By\s+\[.*?\]\(", line.strip()):
                    continue
                # Skip date line
                if re.match(r"^\d{1,2}\s+[A-Z][a-z]+\s+\d{4}$", line.strip()):
                    continue
                # Skip image captions
                if re.match(r"^\(.*Getty Images.*\)$", line.strip()) or re.match(r"^\([^)]*\)$", line.strip()):
                    continue

                content_section.append(line)

        data["content"] = "\n".join(content_section).strip()

        return data


# ========================================
# CLEANTECHNICA EXTRACTOR
# ========================================
class CleanTechnicaExtractor:
    """Extract metadata from CleanTechnica markdown files."""
    
    @staticmethod
    def is_numbered_duplicate(file_path: Path) -> bool:
        """Check if file is a numbered duplicate (e.g., filename_1.md, filename-2.md)."""
        pattern = re.compile(r'^(.+)[-_](\d+)\.md$')
        return bool(pattern.match(file_path.name))
    
    @staticmethod
    def extract_date_from_url(text: str) -> Optional[datetime]:
        """Extract date from URL: /YYYY/MM/DD/"""
        match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', text)
        if match:
            year, month, day = match.groups()
            try:
                return datetime(int(year), int(month), int(day))
            except ValueError:
                return None
        return None
    
    @staticmethod
    def is_within_six_months(article_date: datetime) -> bool:
        """Check if article is within the last 6 months."""
        cutoff_date = datetime.now() - timedelta(days=180)
        return article_date >= cutoff_date
    
    @staticmethod
    def is_valid_article_url(url: str) -> bool:
        """Check if URL is a valid article URL pattern."""
        if not url:
            return False
        # Must match: https://cleantechnica.com/YYYY/MM/DD/...
        article_pattern = re.compile(r"^https://cleantechnica\.com/\d{4}/\d{2}/\d{2}/.+")
        return bool(article_pattern.match(url))
    
    @staticmethod
    def url_to_title(url: str) -> Optional[str]:
        """Convert URL slug to title case."""
        if not url:
            return None
        slug = url.rstrip("/").split("/")[-1]
        words = slug.replace(".md", "").split("-")

        special_upper = {"EV", "SUV", "BMW", "VW", "ID", "CEO", "CFO", "AI", "US", "USA", "UK"}
        lower_words = {"and", "or", "the", "a", "an", "in", "on", "at", "to", "for", "of", "with", "by"}

        title_words = []
        for i, w in enumerate(words):
            if w.upper() in special_upper:
                title_words.append(w.upper())
            elif w.lower() in lower_words and i != 0:
                title_words.append(w.lower())
            else:
                title_words.append(w.capitalize())

        return " ".join(title_words)
    
    @staticmethod
    def extract_between_third_asterisks(text: str) -> Optional[str]:
        """Extract content between the 3rd and 4th *** (or * * *) separators."""
        # Match both '***' and '* * *'
        parts = re.split(r'\n\s*(?:\*{3}|(?:\*\s*){3})\s*\n', text)

        # parts[3] = after third pair → main content
        if len(parts) >= 4:
            return parts[3].strip()
        else:
            return None
    
    @staticmethod
    def clean_content(content: Optional[str]) -> Optional[str]:
        """Remove footers, links, images, and boilerplate from extracted content."""
        if not content:
            return None

        # Remove markdown images
        content = re.sub(r"!\[.*?\]\(https?://[^\)]+\)", "", content)

        # Keep only link text from markdown hyperlinks
        content = re.sub(r"\[([^\]]+)\]\(https?://[^\)]+\)", r"\1", content)

        # Remove affiliate and footer text
        content = re.sub(
            r"(?si)(support cleantechnica|sign up|get our free daily|advertisement|subscribe|watch our|contact us here|have a tip).*",
            "",
            content,
        )

        # Remove promotional underscores and "Filed Under"/"Tags"
        content = re.sub(r"(_){2,}", "", content)
        content = re.sub(r"(Filed Under:|Tags:).*", "", content, flags=re.IGNORECASE)

        # Collapse excessive whitespace
        content = re.sub(r"\s+", " ", content).strip()
        return content
    
    @classmethod
    def should_process_file(cls, file_path: Path) -> tuple[bool, str]:
        """
        Check if file should be processed based on CleanTechnica filtering rules.
        Returns (should_process, skip_reason)
        """
        # Step 1: Skip numbered duplicates (from 3-file_cleaning.py)
        if cls.is_numbered_duplicate(file_path):
            return False, "numbered_duplicate"
        
        # Step 2: Check date from URL (from 2-date_filter.py)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
            
            article_date = cls.extract_date_from_url(first_line)
            if not article_date:
                return False, "no_date_in_url"
            
            if not cls.is_within_six_months(article_date):
                return False, "older_than_6_months"
        except Exception as e:
            return False, f"read_error: {e}"
        
        return True, ""
    
    @classmethod
    def extract(cls, file_path: Path) -> Optional[Dict[str, str]]:
        """Extract metadata fields from a CleanTechnica markdown file."""
        # Check if file should be processed
        should_process, skip_reason = cls.should_process_file(file_path)
        if not should_process:
            return None
        
        data = {
            "url": None,
            "title": None,
            "date": None,
            "author": None,
            "content": None,
            "tags": ""
        }

        text = Path(file_path).read_text(encoding="utf-8")

        # --- URL ---
        url_match = re.search(r'^#\s+(https://cleantechnica\.com/[^\s]+)', text, re.MULTILINE)
        if url_match:
            data["url"] = url_match.group(1).strip()
        
        # Step 3: Validate article URL pattern (from 5-CSV_cleaning.py)
        if not cls.is_valid_article_url(data["url"]):
            return None

        # --- Title ---
        if data["url"]:
            data["title"] = cls.url_to_title(data["url"])

        # --- Date ---
        if data["url"]:
            date_match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', data["url"])
            if date_match:
                y, m, d = date_match.groups()
                try:
                    data["date"] = datetime(int(y), int(m), int(d)).strftime("%d %B %Y")
                except ValueError:
                    data["date"] = f"{d}/{m}/{y}"

        # --- Author ---
        author_match = re.search(
            r'\[\s*([A-Za-z\s]+)\s*\]\(https://cleantechnica\.com/author/[^\)]+\)',
            text,
            re.IGNORECASE,
        )
        if author_match:
            data["author"] = author_match.group(1).strip()

        # --- Main content (between 3rd pair of *** separators) ---
        raw_content = cls.extract_between_third_asterisks(text)
        data["content"] = cls.clean_content(raw_content)

        return data


# ========================================
# CARBONCAPTUREMAGAZINE EXTRACTOR
# ========================================
class CarbonCaptureMagazineExtractor:
    """Extract metadata from CarbonCaptureMagazine markdown files."""
    
    @staticmethod
    def clean_content(text: str) -> str:
        """Clean content by removing ads, related stories, etc."""
        # Remove 'BY ...' line but keep the rest
        text = re.sub(r'^BY [^\n]+\n', '', text, flags=re.MULTILINE)

        # Remove '## Related Stories' and everything after
        text = re.split(r'## Related Stories', text, maxsplit=1, flags=re.IGNORECASE)[0]

        # Remove advertisement blocks
        lines = text.splitlines()
        cleaned_lines = []
        skip = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("Advertisement"):
                skip = True
                continue
            if skip:
                # Skip image lines
                if re.match(r'(!\[.*?\]\(.*?\)|\[!\[.*?\]\(.*?\)\]\(.*?\))', stripped):
                    continue
                else:
                    skip = False  # Stop skipping after non-ad/image line
            cleaned_lines.append(line)

        # Join back lines and clean extra newlines/whitespace
        cleaned_text = "\n".join(cleaned_lines)
        cleaned_text = re.sub(r'\n{2,}', '\n\n', cleaned_text).strip()

        return cleaned_text
    
    @staticmethod
    def extract_publication_date(text: str) -> Optional[datetime]:
        """Extract publication date in Month Day, Year format."""
        date_match = re.search(r'([A-Za-z]+ \d{1,2}, \d{4})', text)
        if date_match:
            try:
                pub_date = datetime.strptime(date_match.group(1), "%B %d, %Y")
                return pub_date
            except ValueError:
                return None
        return None
    
    @staticmethod
    def is_recent(pub_date: Optional[datetime], months: int = 6) -> bool:
        """Check if the publication date is within the last `months` months."""
        if not pub_date:
            return False
        cutoff_date = datetime.now() - timedelta(days=months*30)
        return pub_date >= cutoff_date
    
    @classmethod
    def extract(cls, file_path: Path, date_filter_months: int = 6) -> Optional[Dict[str, str]]:
        """Extract metadata from a CarbonCaptureMagazine markdown file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Check date filter
        pub_date = cls.extract_publication_date(text)
        if not cls.is_recent(pub_date, months=date_filter_months):
            return None  # Skip older articles

        # Extract URL
        url_match = re.search(r'^#\s*(https?://\S+)', text, re.MULTILINE)
        url = url_match.group(1).strip() if url_match else None

        # Extract title
        title_match = re.search(r'^##\s*(.+)', text, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else None

        # Extract author
        author_match = re.search(r'\nBY ([^\n]+)', text)
        author = author_match.group(1).strip() if author_match else None

        # Extract date
        date_match = re.search(r'([A-Za-z]+ \d{1,2}, \d{4})', text)
        date = date_match.group(1).strip() if date_match else None

        # Extract content
        content_start = None
        content_end = None
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("BY "):
                content_start = i
            if line.strip().startswith("## Upcoming Events"):
                content_end = i
                break
        if content_start is not None:
            if content_end is None:
                content_end = len(lines)
            content_lines = lines[content_start:content_end]
            content_text = "\n".join(content_lines)
        else:
            # fallback
            split_marker = re.split(r'^---\s*$', text, maxsplit=1, flags=re.MULTILINE)
            if len(split_marker) == 2:
                content_text = split_marker[1].strip()
            else:
                content_text = text.strip()

        # Clean the content
        content = cls.clean_content(content_text)

        # Fallback title from URL if missing
        if not title and url:
            title = urlparse(url).path.rstrip('/').split('/')[-1].replace('-', ' ').title()

        return {
            'url': url,
            'title': title,
            'date': date,
            'author': author,
            'content': content,
            'tags': ""
        }


# ========================================
# COOLCOALITION EXTRACTOR
# ========================================
class CoolCoalitionExtractor:
    """Extract metadata from CoolCoalition markdown files."""
    
    @staticmethod
    def extract(file_path: Path) -> Dict[str, str]:
        """Extract URL, title, and content from a CoolCoalition markdown file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Extract URL
        url_match = re.search(r'^#\s*(https?://coolcoalition\.org/[^\s]+)', text, re.MULTILINE)
        url = url_match.group(1).strip() if url_match else None

        # Marker for "standard" Cool Coalition articles
        marker = ("Share\nWe Will: Efficient, Climate-Friendly Cooling for All\n"
                  "Receive latest stories, news on efficient, climate-friendly cooling and join the movement!")

        # Check if standard article
        standard_article = ('#' in text and marker in text)

        if standard_article:
            # Extract title: first line after '---' that starts with '#'
            title_match = re.search(r'---\s*\n#\s*(.+)', text, re.MULTILINE)
            title = title_match.group(1).strip() if title_match else None

            # Extract content: text between title and marker
            if title:
                try:
                    start_idx = text.index(title) + len(title)
                    end_idx = text.index(marker)
                    content = text[start_idx:end_idx].strip()
                except ValueError:
                    # fallback if slicing fails
                    content = text.split('---', 1)[-1].strip()
            else:
                # fallback: use everything after '---'
                content = text.split('---', 1)[-1].strip()
                # fallback title from URL
                if url:
                    title = urlparse(url).path.rstrip('/').split('/')[-1].replace('-', ' ').title()
                else:
                    title = None
        else:
            # Non-standard article: use last part of URL as title
            if url:
                title = urlparse(url).path.rstrip('/').split('/')[-1].replace('-', ' ').title()
            else:
                title = None
            # Everything after first '---' as content
            split_marker = re.split(r'^---\s*$', text, maxsplit=1, flags=re.MULTILINE)
            if len(split_marker) == 2:
                content = split_marker[1].strip()
            else:
                content = text.strip()

        return {
            'url': url,
            'title': title,
            'content': content,
            'date': "",
            'author': "",
            'tags': ""
        }


# ========================================
# PRESET EXTRACTOR REGISTRY
# ========================================
PRESET_EXTRACTORS = {
    'ThinkGeoEnergy': ThinkGeoEnergyExtractor,
    'CanaryMedia': CanaryMediaExtractor,
    'CleanTechnica': CleanTechnicaExtractor,
    'CarbonCaptureMagazine': CarbonCaptureMagazineExtractor,
    'CoolCoalition': CoolCoalitionExtractor,
}


def get_preset_extractor(preset_name: str):
    """Get the extractor class for a given preset."""
    return PRESET_EXTRACTORS.get(preset_name)

