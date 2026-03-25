#!/usr/bin/env python3
"""
Test script to verify embedding file naming format
"""

import re
from pathlib import Path

def extract_source_name(filename: str) -> str:
    """Extract source name from filename."""
    # Remove extension
    stem = Path(filename).stem
    
    # Remove all date patterns (YYYYMMDD)
    stem_no_dates = re.sub(r'_?\d{8}', '', stem)
    
    # Remove common suffixes
    stem_no_dates = re.sub(r'_(com|org|net|filtered|crawl|processed).*$', '', stem_no_dates)
    
    # Clean up multiple underscores
    stem_no_dates = re.sub(r'_+', '_', stem_no_dates).strip('_')
    
    return stem_no_dates.lower() if stem_no_dates else stem

def extract_date_from_filename(filename: str) -> str:
    """Extract last date from filename."""
    # Remove extension
    stem = Path(filename).stem
    
    # Find ALL date patterns (YYYYMMDD)
    date_matches = re.findall(r'\d{8}', stem)
    
    if date_matches:
        # Return the LAST date found (processed date)
        return date_matches[-1]
    
    return ""

def generate_embedding_filename(json_filename: str) -> str:
    """Generate embedding filename from JSON filename."""
    source_name = extract_source_name(json_filename)
    date_str = extract_date_from_filename(json_filename)
    
    if date_str:
        return f"{source_name}_embeddings_{date_str}.pkl"
    else:
        return f"{source_name}_embeddings.pkl"

# Test cases
test_files = [
    "techcrunch_com_20251204.json",
    "pv-magazine_com_20251204.json",
    "hydrogen-central_com_20251204.json",
    "techcrunch_com_20251127_filtered_20251127.json",
    "pv-magazine_com_20251204_filtered_20251204.json",
    "hydrogen-central_com_20251204_filtered_20251204.json",
    "canarymedia_com_crawl_filtered_20251119.json",
]

print("=" * 80)
print("EMBEDDING FILE NAMING TEST")
print("=" * 80)
print()

for filename in test_files:
    source = extract_source_name(filename)
    date = extract_date_from_filename(filename)
    embedding_file = generate_embedding_filename(filename)
    
    print(f"Input:     {filename}")
    print(f"Source:    {source}")
    print(f"Date:      {date}")
    print(f"Output:    {embedding_file}")
    print()

print("=" * 80)
print("Expected format: <source>_embeddings_<processed_date>.pkl")
print("=" * 80)
