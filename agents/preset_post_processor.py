"""
Preset-based post-processor that uses hardcoded extractors for specific sites.
NO AI DETECTION - uses reference implementation patterns only.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Set
import os
from datetime import datetime

from agents.preset_extractors import get_preset_extractor
from agents.quick_start_post_processing import get_preset


class PresetPostProcessor:
    """Process markdown files using preset-specific extractors."""
    
    def __init__(self, preset_name: str, blocked_urls: Optional[Set[str]] = None):
        """
        Initialize the preset post-processor.
        
        Args:
            preset_name: Name of the preset (ThinkGeoEnergy, CanaryMedia, etc.)
            blocked_urls: Set of URLs to skip (category pages, landing pages)
        """
        self.preset_name = preset_name
        self.blocked_urls = blocked_urls or set()
        
        # Load preset configuration
        self.preset_config = get_preset(preset_name)
        if not self.preset_config:
            raise ValueError(f"Unknown preset: {preset_name}")
        
        # Add preset's blocked URLs
        if self.preset_config.get('blocked_urls'):
            self.blocked_urls.update(self.preset_config['blocked_urls'])
        
        # Get the extractor class
        self.extractor_class = get_preset_extractor(preset_name)
        if not self.extractor_class:
            raise ValueError(f"No extractor found for preset: {preset_name}")
    
    def process_folder(
        self,
        folder_path: Path,
        output_dir: Path = Path("processed_content"),
        date_filter_months: Optional[int] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Process all markdown files in folder using preset extractor.
        
        Args:
            folder_path: Path to folder containing markdown files
            output_dir: Path to save processed output
            date_filter_months: Filter articles within last N months (None = no filter)
        
        Returns:
            Tuple of (DataFrame, stats dictionary)
        """
        # Use preset's date filter if not specified
        if date_filter_months is None and self.preset_config.get('date_filter_months'):
            date_filter_months = self.preset_config['date_filter_months']
        
        # Find all markdown files
        md_files = list(folder_path.rglob('*.md'))
        
        stats = {
            'total_files': len(md_files),
            'processed': 0,
            'skipped_blocked': 0,
            'skipped_old': 0,
            'skipped_duplicates': 0,
            'skipped_invalid_url': 0,
            'skipped_error': 0
        }
        
        all_data = []
        
        for md_file in md_files:
            try:
                # Extract metadata using preset-specific extractor
                if self.preset_name == 'CarbonCaptureMagazine':
                    # CarbonCaptureMagazine has date filtering built into extractor
                    metadata = self.extractor_class.extract(
                        md_file, 
                        date_filter_months=date_filter_months or 6
                    )
                    if metadata is None:
                        # Filtered out by date
                        stats['skipped_old'] += 1
                        continue
                elif self.preset_name == 'CleanTechnica':
                    # CleanTechnica has multiple built-in filters
                    metadata = self.extractor_class.extract(md_file)
                    if metadata is None:
                        # Check why it was skipped (for better stats)
                        should_process, skip_reason = self.extractor_class.should_process_file(md_file)
                        if 'numbered_duplicate' in skip_reason:
                            stats['skipped_duplicates'] += 1
                        elif 'older_than_6_months' in skip_reason:
                            stats['skipped_old'] += 1
                        elif skip_reason == "" and metadata is None:
                            # Filtered by invalid URL pattern
                            stats['skipped_invalid_url'] += 1
                        else:
                            stats['skipped_error'] += 1
                        continue
                else:
                    # Other extractors
                    metadata = self.extractor_class.extract(md_file)
                
                if not metadata:
                    stats['skipped_error'] += 1
                    continue
                
                # Check if URL is blocked
                url = metadata.get('url')
                if url and url in self.blocked_urls:
                    stats['skipped_blocked'] += 1
                    continue
                
                # Add file info
                metadata['file_name'] = md_file.name
                metadata['folder'] = md_file.parent.name
                
                all_data.append(metadata)
                stats['processed'] += 1
                
            except Exception as e:
                print(f"Error processing {md_file.name}: {e}")
                stats['skipped_error'] += 1
                continue
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        # Save outputs
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = output_dir / f"{self.preset_name.lower()}_{timestamp}.csv"
        json_file = output_dir / f"{self.preset_name.lower()}_{timestamp}.json"
        
        # Save CSV
        df.to_csv(csv_file, index=False)
        print(f"✅ CSV saved to: {csv_file}")
        
        # Save JSON
        df.to_json(json_file, orient='records', indent=2, force_ascii=False)
        print(f"✅ JSON saved to: {json_file}")
        
        # Update stats with output files
        stats['csv_file'] = str(csv_file)
        stats['json_file'] = str(json_file)
        
        return df, stats

