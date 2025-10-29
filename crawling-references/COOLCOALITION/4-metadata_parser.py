import os
import re
import json
import csv
from pathlib import Path
from urllib.parse import urlparse

def extract_metadata(md_file):
    """Extract URL, title, and content from a markdown file."""
    with open(md_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Extract URL: first line after '#' that looks like a URL
    url_match = re.search(r'^#\s*(https?://\S+)', text, re.MULTILINE)
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

        # Extract content: text between title and marker, fallback if title missing
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
        # Non-standard article: use last part of URL as title, everything after '---' as content
        if url:
            title = urlparse(url).path.rstrip('/').split('/')[-1].replace('-', ' ').title()
        else:
            title = None
        # Everything after first '---' as content
        split_marker = re.split(r'^---\s*$', text, maxsplit=1, flags=re.MULTILINE)
        if len(split_marker) == 2:
            content = split_marker[1].strip()
        else:
            content = text.strip()  # fallback: use whole text

    return {
        'file': str(md_file),
        'url': url,
        'title': title,
        'content': content
    }


def process_markdown_folder(folder_path, json_file="coolcoalition.json", csv_file="coolcoalition.csv"):
    folder = Path(folder_path)
    all_metadata = []

    for md_file in folder.rglob("*.md"):
        metadata = extract_metadata(md_file)
        all_metadata.append(metadata)

    # Save to JSON
    with open(json_file, 'w', encoding='utf-8') as f_json:
        json.dump(all_metadata, f_json, ensure_ascii=False, indent=2)
    print(f"JSON metadata saved to {json_file}")

    # Save to CSV
    with open(csv_file, 'w', encoding='utf-8', newline='') as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=['file', 'url', 'title', 'content'])
        writer.writeheader()
        for data in all_metadata:
            row = {k: (v.replace('\n', ' ') if isinstance(v, str) else v) for k, v in data.items()}
            writer.writerow(row)
    print(f"CSV metadata saved to {csv_file}")

    print(f"\nProcessed {len(all_metadata)} files.")

if __name__ == "__main__":
    folder = input("Enter path to markdown folder: ").strip()
    process_markdown_folder(folder)

