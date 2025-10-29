import os
import re
import csv
from pathlib import Path

def extract_url(lines):
    """Extract URL from the first line after #"""
    for line in lines:
        if line.strip().startswith('# https://') or line.strip().startswith('# http://'):
            return line.strip()[2:].strip()
    return ""

def extract_title(lines):
    """Extract title after the # symbol"""
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Skip the URL line
        if stripped.startswith('# https://') or stripped.startswith('# http://'):
            continue
        # Find the actual title line (starts with # but not a URL)
        if stripped.startswith('# ') and not stripped.startswith('# http'):
            return stripped[2:].strip()
    return ""

def extract_author(lines):
    """Extract author name between [] and ** pair"""
    for line in lines:
        # Look for pattern [**Author Name**]
        match = re.search(r'\[\*\*([^\]]+)\*\*\]', line)
        if match:
            return match.group(1).strip()
    return ""

def extract_date(lines):
    """Extract date after the author profile link"""
    for i, line in enumerate(lines):
        # Find the author link line
        if '**' in line and '[' in line and ']' in line:
            # Look for date pattern after the author link
            # Common patterns: "23 Jul 2025", "Jul 23, 2025", etc.
            date_match = re.search(r'(\d{1,2}\s+\w{3}\s+\d{4})', line)
            if date_match:
                return date_match.group(1).strip()
    return ""

def extract_content(lines):
    """Extract main content text between title and 'Source:'"""
    content_lines = []
    in_content = False
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

def extract_tags(lines):
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

def extract_metadata_from_file(filepath):
    """Extract all metadata from a single markdown file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        metadata = {
            'filename': os.path.basename(filepath),
            'url': extract_url(lines),
            'title': extract_title(lines),
            'author': extract_author(lines),
            'date': extract_date(lines),
            'content': extract_content(lines),
            'tags': extract_tags(lines)
        }
        
        return metadata
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return None

def process_folder(folder_path, output_csv='metadata_output.csv'):
    """Process all markdown files in the specified folder"""
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    # Find all markdown files
    md_files = list(folder.glob('*.md')) + list(folder.glob('*.markdown'))
    
    if not md_files:
        print(f"No markdown files found in '{folder_path}'.")
        return
    
    print(f"Found {len(md_files)} markdown file(s). Processing...")
    
    # Extract metadata from all files
    all_metadata = []
    for md_file in md_files:
        print(f"Processing: {md_file.name}")
        metadata = extract_metadata_from_file(md_file)
        if metadata:
            all_metadata.append(metadata)
    
    # Write to CSV
    if all_metadata:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['filename', 'url', 'title', 'author', 'date', 'content', 'tags']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for metadata in all_metadata:
                writer.writerow(metadata)
        
        print(f"\nSuccess! Metadata extracted to '{output_csv}'")
        print(f"Processed {len(all_metadata)} file(s).")
    else:
        print("No metadata could be extracted.")

if __name__ == "__main__":
    # Get folder path from user
    folder_path = input("Enter the folder path containing markdown files: ").strip()
    
    # Optional: customize output filename
    output_file = input("Enter output CSV filename (press Enter for 'metadata_output.csv'): ").strip()
    if not output_file:
        output_file = 'metadata_output.csv'
    
    if not output_file.endswith('.csv'):
        output_file += '.csv'
    
    process_folder(folder_path, output_file)