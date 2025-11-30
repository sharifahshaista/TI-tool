import json
import re
from pathlib import Path

def extract_title_and_date(url):
    """
    Extract title and publication date from TechCrunch URL.
    URL format: https://techcrunch.com/YYYY/MM/DD/slug
    """
    # Match the pattern: /YYYY/MM/DD/slug
    pattern = r'/(\d{4})/(\d{2})/(\d{2})/(.+)'
    match = re.search(pattern, url)

    if match:
        year, month, day, slug = match.groups()
        # Format date as YYYY-MM-DD
        publication_date = f"{year}-{month}-{day}"

        # Convert slug to title: replace hyphens with spaces and title case
        title = slug.replace('-', ' ').title()

        return title, publication_date
    else:
        # If pattern doesn't match, return None values
        return None, None

def fix_json_file(file_path):
    """
    Read JSON file, fix title and publication_date for each record, and write back.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for record in data:
        if 'url' in record:
            title, pub_date = extract_title_and_date(record['url'])
            if title and pub_date:
                record['title'] = title
                record['publication_date'] = pub_date

    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # Path to the JSON file
    file_path = Path("/Users/sharifahshaista/TI-tool/processed_data/techcrunch_com_20251127_filtered_20251127.json")

    if file_path.exists():
        fix_json_file(file_path)
        print(f"Fixed titles and publication dates in {file_path}")
    else:
        print(f"File not found: {file_path}")