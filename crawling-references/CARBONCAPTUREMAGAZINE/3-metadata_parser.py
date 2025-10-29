import os
import re
import json
import csv
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime, timedelta

def clean_content(text):
    """
    Clean content by:
    - Keeping the introductory paragraph and all article text
    - Removing advertisement blocks starting with 'Advertisement'
    - Removing images/linked images in ad blocks
    - Removing '## Related Stories' and everything after
    - Cleaning extra newlines
    """
    # Remove 'BY ...' line but keep the rest
    text = re.sub(r'^BY [^\n]+\n', '', text, flags=re.MULTILINE)

    # Remove '## Related Stories' and everything after
    text = re.split(r'## Related Stories', text, maxsplit=1, flags=re.IGNORECASE)[0]

    # Remove advertisement blocks: lines that start with 'Advertisement' and any following images/links
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




def extract_publication_date(text):
    """Extract publication date in Month Day, Year format."""
    date_match = re.search(r'([A-Za-z]+ \d{1,2}, \d{4})', text)
    if date_match:
        try:
            pub_date = datetime.strptime(date_match.group(1), "%B %d, %Y")
            return pub_date
        except ValueError:
            return None
    return None

def is_recent(pub_date, months=6):
    """Check if the publication date is within the last `months` months."""
    if not pub_date:
        return False
    cutoff_date = datetime.now() - timedelta(days=months*30)
    return pub_date >= cutoff_date

def extract_metadata(md_file):
    """Extract URL, title, date, author, and cleaned content from a markdown file."""
    with open(md_file, 'r', encoding='utf-8') as f:
        text = f.read()

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

    # Extract content: from author line to '## Upcoming Events' or end of file
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
        # fallback: everything after first '---'
        split_marker = re.split(r'^---\s*$', text, maxsplit=1, flags=re.MULTILINE)
        if len(split_marker) == 2:
            content_text = split_marker[1].strip()
        else:
            content_text = text.strip()

    # Clean the content
    content = clean_content(content_text)

    # fallback title from URL if missing
    if not title and url:
        title = urlparse(url).path.rstrip('/').split('/')[-1].replace('-', ' ').title()

    return {
        'file': str(md_file),
        'url': url,
        'title': title,
        'date': date,
        'author': author,
        'content': content
    }

def process_markdown_folder(folder_path, json_file="carboncapture.json", csv_file="carboncapture.csv"):
    folder = Path(folder_path)
    all_metadata = []

    for md_file in folder.rglob("*.md"):
        with open(md_file, 'r', encoding='utf-8') as f:
            text = f.read()
        pub_date = extract_publication_date(text)
        if not is_recent(pub_date):
            continue  # skip older articles
        metadata = extract_metadata(md_file)
        all_metadata.append(metadata)

    # Save JSON
    with open(json_file, 'w', encoding='utf-8') as f_json:
        json.dump(all_metadata, f_json, ensure_ascii=False, indent=2)
    print(f"JSON metadata saved to {json_file}")

    # Save CSV
    with open(csv_file, 'w', encoding='utf-8', newline='') as f_csv:
        fieldnames = ['file', 'url', 'title', 'date', 'author', 'content']
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for data in all_metadata:
            row = {k: (v.replace('\n', ' ') if isinstance(v, str) else v) for k, v in data.items()}
            writer.writerow(row)
    print(f"CSV metadata saved to {csv_file}")

    print(f"\nProcessed {len(all_metadata)} markdown files within 6 months.")

if __name__ == "__main__":
    folder = input("Enter path to markdown folder: ").strip()
    process_markdown_folder(folder)
