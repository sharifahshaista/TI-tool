"""
AI-Powered Markdown Metadata Pattern Detector
Uses LLM to analyze markdown files and automatically generate regex patterns
for extracting structured metadata (URL, title, author, date, content, tags)
"""

import re
import os
import json
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI, AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class MetadataPattern(BaseModel):
    """Regex patterns for extracting metadata from markdown files"""
    url_pattern: str = Field(description="Regex pattern to extract URL")
    title_pattern: str = Field(description="Regex pattern to extract title")
    author_pattern: Optional[str] = Field(description="Regex pattern to extract author name", default=None)
    date_pattern: Optional[str] = Field(description="Regex pattern to extract publication date", default=None)
    content_start_marker: Optional[str] = Field(description="Marker indicating start of main content", default=None)
    content_end_marker: Optional[str] = Field(description="Marker indicating end of main content", default=None)
    tags_pattern: Optional[str] = Field(description="Regex pattern to extract tags/categories", default=None)
    noise_patterns: List[str] = Field(description="List of patterns to remove (ads, navigation, etc.)", default_factory=list)


class MarkdownMetadataDetector:
    """
    Intelligently detects metadata patterns in markdown files using AI
    """
    
    def __init__(self):
        # Try to initialize Azure OpenAI first, fall back to regular OpenAI
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        
        if azure_api_key and azure_endpoint:
            self.client = AzureOpenAI(
                api_key=azure_api_key,
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
                azure_endpoint=azure_endpoint
            )
            self.model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4")
            self.is_azure = True
        else:
            self.client = OpenAI()
            self.model = "gpt-4o-mini"
            self.is_azure = False
    
    def analyze_samples(self, markdown_files: List[Path], num_samples: int = 3) -> MetadataPattern:
        """
        Analyze sample markdown files and generate regex patterns using AI
        
        Args:
            markdown_files: List of markdown file paths to analyze
            num_samples: Number of sample files to analyze
            
        Returns:
            MetadataPattern object with detected regex patterns
        """
        # Read sample files
        samples = []
        for md_file in markdown_files[:num_samples]:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Limit sample size for efficiency
                    samples.append({
                        'filename': md_file.name,
                        'content': content[:1000000]  # First 1000000 chars
                    })
            except Exception as e:
                continue
        
        if not samples:
            raise ValueError("No valid markdown files found for analysis")
        
        # Construct AI prompt
        prompt = self._build_analysis_prompt(samples)
        
        # Get AI response using OpenAI API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            
            # Parse response into MetadataPattern
            pattern = self._parse_ai_response(response.choices[0].message.content)
        except Exception as e:
            print(f"AI analysis failed: {e}")
            print("Using fallback patterns...")
            pattern = self._get_fallback_patterns()
        
        return pattern
    
    def _build_analysis_prompt(self, samples: List[Dict]) -> str:
        """Build enhanced prompt for AI to analyze markdown structure"""
        
        samples_text = "\n\n".join([
            f"=== SAMPLE {i+1}: {s['filename']} ===\n{s['content']}\n"
            for i, s in enumerate(samples)
        ])
        
        prompt = f"""You are an expert in analyzing markdown file structures and creating robust regex patterns for metadata extraction from web-crawled content.

Analyze these markdown file samples carefully and identify patterns for extracting:

1. **URL** - Usually appears after # at the start (e.g., "# https://example.com/article")
2. **Title** - Main article title (H1 or H2 heading, NOT the URL)
3. **Author** - Author name (often in a byline, link, or after "By")
4. **Publication Date** - Date in various formats (DD Month YYYY, Month DD YYYY, etc.)
5. **Main Content** - The actual article text (identify clear start/end markers if possible)
6. **Tags/Categories** - Article tags or categories (often in links or lists)
7. **Noise Patterns** - Elements to remove (ads, navigation, social buttons, footers, etc.)

MARKDOWN SAMPLES:
{samples_text}

INSTRUCTIONS:
1. Create SPECIFIC and ACCURATE regex patterns that match the actual structure in the samples
2. For URL: Extract the full URL from the line starting with #
3. For Title: Look for patterns like "scrolledValue'> # Title" or "## Title" (NOT URLs)
4. For Author: Look for author links like "[Name](url/people/..)" or "By [Name]"
5. For Date: Match date formats like "23 October 2025" or "October 23, 2025"
6. For Content: Identify unique start/end markers (like "scrolledValue" or "***" separators)
7. For Tags: Look for tag links like "* [Tag](url)" or similar patterns
8. For Noise: Include patterns for ads, social media, promotions, etc.

VALIDATION RULES:
- Test each pattern mentally against the samples provided
- Ensure capturing groups () only capture the desired data
- Use non-capturing groups (?:...) for structure matching
- Make date patterns flexible for variations
- Avoid overly generic patterns that might catch unwanted text

OUTPUT FORMAT (valid JSON only):
{{
  "url_pattern": "r'^#\\s+(https?://[^\\s]+)'",
  "title_pattern": "r'scrolledValue[^>]*>\\s*#\\s+(.+)'",
  "author_pattern": "r'\\[([^\\]]+)\\]\\(https?://[^/]+/(?:about/)?people/[^\\)]+\\)'" or null,
  "date_pattern": "r'(\\d{{1,2}}\\s+[A-Z][a-z]+\\s+\\d{{4}})'" or null,
  "content_start_marker": "scrolledValue" or null or "!",
  "content_end_marker": "* [" or null,
  "tags_pattern": "r'\\*\\s*\\[([^\\]]+)\\]\\([^\\)]+\\)'" or null,
  noise_patterns = [
    r'!\[.*?\]\(https?://[^\)]+\)',  # image markdown with URL
    r'(?i)(advertisement|subscribe|follow us|share this|sign up|get our|support us)',  # common promotional text
    r'\[Share\].*',  # lines starting with [Share]
    r'\[Tweet\].*',  # lines starting with [Tweet]
    r'(!\[\]\()+' ,  # multiple or repeated empty markdown image patterns like ![]( ![](
    r'#+\s*news\b.*',  # headings like ## News, # News, etc.
    r'\b\d+\.\s*(homepage|technology|news|topics|sections?)\b',  # numbered navigation lists
]
}}

RESPOND WITH VALID JSON ONLY (no markdown, no explanations, just the JSON object)."""
        
        return prompt
    
    def _parse_ai_response(self, response: str) -> MetadataPattern:
        """Parse AI response into MetadataPattern"""
        
        # Extract JSON from response (might be wrapped in markdown)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if not json_match:
            # Fallback to basic patterns
            return self._get_fallback_patterns()
        
        try:
            data = json.loads(json_match.group(0))
            
            # Convert string "null" to None
            for key in data:
                if data[key] == "null" or data[key] == "None":
                    data[key] = None
            
            # Ensure noise_patterns is a list
            if 'noise_patterns' not in data or data['noise_patterns'] is None:
                data['noise_patterns'] = []
            
            return MetadataPattern(**data)
        
        except Exception as e:
            print(f"Error parsing AI response: {e}")
            return self._get_fallback_patterns()
    
    def _get_fallback_patterns(self) -> MetadataPattern:
        """Return generic fallback patterns if AI analysis fails"""
        return MetadataPattern(
            url_pattern=r'^#\s*(https?://\S+)',
            title_pattern=r'^##?\s*(.+)',
            author_pattern=r'\[([^\]]+)\]\(.*?/author/.*?\)',
            date_pattern=r'(\d{1,2}\s+\w+\s+\d{4})',
            content_start_marker=None,
            content_end_marker=None,
            tags_pattern=None,
            noise_patterns=[
                r'!\[.*?\]\(https?://[^\)]+\)',  # Images
                r'Advertisement',
                r'## Related',
                r'## Share',
                r'Follow us on',
                r'Subscribe',
            ]
        )
    
    def save_patterns(self, patterns: MetadataPattern, output_file: Path):
        """Save detected patterns to JSON file for reuse"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(patterns.model_dump(), f, indent=2)
        print(f"Patterns saved to: {output_file}")
    
    def load_patterns(self, pattern_file: Path) -> MetadataPattern:
        """Load previously saved patterns from JSON file"""
        with open(pattern_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return MetadataPattern(**data)


def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 2:
        print("\nUsage: python markdown_metadata_detector.py <markdown_folder>")
        print("\nExample:")
        print("  python markdown_metadata_detector.py saved_md/")
        sys.exit(1)
    
    folder = Path(sys.argv[1])
    
    if not folder.exists():
        print(f"Error: Folder '{folder}' does not exist")
        sys.exit(1)
    
    # Find markdown files
    md_files = list(folder.glob('*.md'))
    
    if not md_files:
        print(f"No markdown files found in '{folder}'")
        sys.exit(1)
    
    print(f"\nFound {len(md_files)} markdown files")
    print("Analyzing structure using AI...")
    
    # Detect patterns
    detector = MarkdownMetadataDetector()
    patterns = detector.analyze_samples(md_files, num_samples=min(3, len(md_files)))
    
    # Display results
    print("\n" + "="*80)
    print("DETECTED PATTERNS")
    print("="*80)
    print(json.dumps(patterns.model_dump(), indent=2))
    
    # Save patterns
    output_file = folder / "metadata_patterns.json"
    detector.save_patterns(patterns, output_file)


if __name__ == "__main__":
    main()

