"""
Quick Start Post-Processing Presets
Pre-configured metadata extraction patterns for common news sites.
"""

from typing import Dict, List, Optional

# ========================================
# THINKGEOENERGY PATTERNS
# ========================================
THINKGEOENERGY_PATTERNS = {
    'url': {
        'type': 'regex',
        'pattern': r'^#\s+(https?://www\.thinkgeoenergy\.com/[^\s]+)',
        'flags': 'MULTILINE'
    },
    'title': {
        'type': 'regex',
        'pattern': r'^#\s+(?!https?://)(.+)$',
        'flags': 'MULTILINE'
    },
    'author': {
        'type': 'regex',
        'pattern': r'\[\*\*([^\]]+)\*\*\]',
        'flags': ''
    },
    'date': {
        'type': 'regex',
        'pattern': r'(\d{1,2}\s+\w{3}\s+\d{4})',
        'flags': ''
    },
    'content': {
        'type': 'between_markers',
        'start_marker': r'^#\s+(?!https?://)(.+)$',  # Title line
        'end_marker': r'^Source:',
        'flags': 'MULTILINE'
    },
    'tags': {
        'type': 'section',
        'section_marker': 'Tags',
        'pattern': r'\[([^\]]+)\]',
        'flags': ''
    }
}

THINKGEOENERGY_BLOCKED_URLS = []


# ========================================
# CANARYMEDIA PATTERNS
# ========================================
CANARYMEDIA_PATTERNS = {
    'url': {
        'type': 'regex',
        'pattern': r'^#\s+(https?://www\.canarymedia\.com/[^\s]+)',
        'flags': 'MULTILINE'
    },
    'title': {
        'type': 'regex',
        'pattern': r"scrolledValue[\"']>\s*#\s+(.+)",
        'flags': ''
    },
    'author': {
        'type': 'regex',
        'pattern': r'\[([^\]]+)\]\(https://www\.canarymedia\.com/about/people/[^\)]+\)',
        'flags': ''
    },
    'date': {
        'type': 'regex',
        'pattern': r'\b(\d{1,2}\s+[A-Z][a-z]+\s+\d{4})\b',
        'flags': ''
    },
    'content': {
        'type': 'between_markers',
        'start_marker': r'scrolledValue',
        'end_marker': r'^\*\s*\[',
        'flags': 'MULTILINE'
    },
    'tags': {
        'type': 'regex',
        'pattern': r'\*\s*\[([^\]]+)\]\([^\)]+\)',
        'flags': ''
    }
}

# All 60 category/landing pages
CANARYMEDIA_BLOCKED_URLS = [
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


# ========================================
# CLEANTECHNICA PATTERNS
# ========================================
CLEANTECHNICA_PATTERNS = {
    'url': {
        'type': 'regex',
        'pattern': r'^#\s+(https://cleantechnica\.com/[^\s]+)',
        'flags': 'MULTILINE'
    },
    'title': {
        'type': 'from_url_slug',  # Special: convert URL slug to title
        'pattern': '',
        'flags': ''
    },
    'author': {
        'type': 'regex',
        'pattern': r'\[\s*([A-Za-z\s]+)\s*\]\(https://cleantechnica\.com/author/[^\)]+\)',
        'flags': ''
    },
    'date': {
        'type': 'from_url_date',  # Special: extract from /YYYY/MM/DD/ in URL
        'pattern': r'/(\d{4})/(\d{2})/(\d{2})/',
        'flags': ''
    },
    'content': {
        'type': 'between_asterisks',  # Special: between 3rd and 4th *** separators
        'pattern': r'\n\s*(?:\*{3}|(?:\*\s*){3})\s*\n',
        'flags': ''
    },
    'tags': {
        'type': 'none',  # CleanTechnica doesn't have consistent tag extraction
        'pattern': '',
        'flags': ''
    }
}

CLEANTECHNICA_BLOCKED_URLS = []


# ========================================
# CARBONCAPTUREMAGAZINE PATTERNS
# ========================================
CARBONCAPTUREMAGAZINE_PATTERNS = {
    'url': {
        'type': 'regex',
        'pattern': r'^#\s+(https?://[^\s]+)',
        'flags': 'MULTILINE'
    },
    'title': {
        'type': 'regex',
        'pattern': r'^##\s*(.+)',
        'flags': 'MULTILINE'
    },
    'author': {
        'type': 'regex',
        'pattern': r'\nBY\s+([^\n]+)',
        'flags': ''
    },
    'date': {
        'type': 'regex',
        'pattern': r'([A-Za-z]+\s+\d{1,2},\s+\d{4})',
        'flags': ''
    },
    'content': {
        'type': 'between_markers',
        'start_marker': r'\nBY\s+',
        'end_marker': r'^##\s+Upcoming Events',
        'flags': 'MULTILINE'
    },
    'tags': {
        'type': 'none',
        'pattern': '',
        'flags': ''
    }
}

CARBONCAPTUREMAGAZINE_BLOCKED_URLS = []


# ========================================
# COOLCOALITION PATTERNS
# ========================================
COOLCOALITION_PATTERNS = {
    'url': {
        'type': 'regex',
        'pattern': r'^#\s+(https?://coolcoalition\.org/[^\s]+)',
        'flags': 'MULTILINE'
    },
    'title': {
        'type': 'regex',
        'pattern': r'---\s*\n#\s*(.+)',
        'flags': 'MULTILINE'
    },
    'author': {
        'type': 'none',  # Cool Coalition doesn't have author info
        'pattern': '',
        'flags': ''
    },
    'date': {
        'type': 'none',  # Cool Coalition doesn't have consistent dates
        'pattern': '',
        'flags': ''
    },
    'content': {
        'type': 'between_markers',
        'start_marker': r'---\s*\n#\s*.+',
        'end_marker': r'Share\nWe Will:',
        'flags': 'MULTILINE'
    },
    'tags': {
        'type': 'none',
        'pattern': '',
        'flags': ''
    }
}

COOLCOALITION_BLOCKED_URLS = []


# ========================================
# PRESET CONFIGURATIONS
# ========================================
QUICK_START_PRESETS = {
    'ThinkGeoEnergy': {
        'name': 'ThinkGeoEnergy',
        'description': 'Geothermal energy news and analysis',
        'patterns': THINKGEOENERGY_PATTERNS,
        'blocked_urls': THINKGEOENERGY_BLOCKED_URLS,
        'content_filters': {
            'skip_images': True,
            'skip_links_to_images': True,
            'skip_navigation': True,
            'skip_social': True,
            'skip_ads': True
        }
    },
    'CanaryMedia': {
        'name': 'CanaryMedia',
        'description': 'Clean energy journalism (60 topics)',
        'patterns': CANARYMEDIA_PATTERNS,
        'blocked_urls': CANARYMEDIA_BLOCKED_URLS,
        'content_filters': {
            'skip_images': True,
            'skip_author_lines': True,
            'skip_date_lines': True,
            'skip_captions': True
        }
    },
    'CleanTechnica': {
        'name': 'CleanTechnica',
        'description': 'Clean tech and electric vehicle news (multi-step filtering)',
        'patterns': CLEANTECHNICA_PATTERNS,
        'blocked_urls': CLEANTECHNICA_BLOCKED_URLS,
        'content_filters': {
            'remove_numbered_duplicates': True,  # filename_1.md, filename-2.md
            'date_filter_6_months': True,  # Only articles from last 6 months
            'validate_article_url': True,  # Must match /YYYY/MM/DD/ pattern
            'remove_images': True,
            'remove_affiliate_text': True,
            'remove_footer': True,
            'remove_filed_under': True
        },
        'date_filter_months': 6  # Built-in 6-month filter
    },
    'CarbonCaptureMagazine': {
        'name': 'CarbonCaptureMagazine',
        'description': 'Carbon capture and storage news',
        'patterns': CARBONCAPTUREMAGAZINE_PATTERNS,
        'blocked_urls': CARBONCAPTUREMAGAZINE_BLOCKED_URLS,
        'content_filters': {
            'remove_advertisements': True,
            'remove_related_stories': True,
            'remove_by_lines': True
        },
        'date_filter_months': 6  # Only articles from last 6 months
    },
    'CoolCoalition': {
        'name': 'CoolCoalition',
        'description': 'Efficient cooling and climate solutions',
        'patterns': COOLCOALITION_PATTERNS,
        'blocked_urls': COOLCOALITION_BLOCKED_URLS,
        'content_filters': {
            'remove_share_sections': True,
            'fallback_title_from_url': True
        }
    }
}


def get_preset(preset_name: str) -> Optional[Dict]:
    """Get a quick start preset by name."""
    return QUICK_START_PRESETS.get(preset_name)


def get_available_presets() -> List[str]:
    """Get list of available preset names."""
    return list(QUICK_START_PRESETS.keys())


def get_preset_description(preset_name: str) -> str:
    """Get the description for a preset."""
    preset = QUICK_START_PRESETS.get(preset_name)
    return preset['description'] if preset else ""


async def run_preset_processing(preset_name: str, markdown_folder: str, output_folder: str, date_filter: Optional[int] = None, progress_callback=None):
    """
    Run preset post-processing based on the preset name.
    
    Args:
        preset_name: Name of the preset (e.g., "ThinkGeoEnergy", "CanaryMedia", etc.)
        markdown_folder: Path to folder containing markdown files
        output_folder: Path to output folder for processed files
        date_filter: Optional date filter in months
        progress_callback: Optional callback for progress updates
    
    Returns:
        Dictionary with processing results
    """
    import asyncio
    from pathlib import Path
    from agents.markdown_post_processor import MarkdownPostProcessor
    
    # Get preset configuration
    preset = get_preset(preset_name)
    if not preset:
        raise ValueError(f"Unknown preset: {preset_name}")
    
    # Create processor with preset configuration (only supported args)
    processor = MarkdownPostProcessor(
        blocked_urls=set(preset.get('blocked_urls', []))
    )
    
    # The MarkdownPostProcessor.process_folder is synchronous; run in a thread for async compatibility
    from functools import partial
    loop = asyncio.get_running_loop()
    df, stats = await loop.run_in_executor(
        None,
        partial(
            processor.process_folder,
            folder_path=Path(markdown_folder),
            output_dir=Path(output_folder),
            date_filter_months=date_filter,
            auto_detect=False,
            pattern_file=None
        )
    )
    
    return {
        'dataframe': df,
        'metadata': stats,
        'output_dir': str(Path(output_folder))
    }

