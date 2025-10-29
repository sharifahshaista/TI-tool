"""

Step 2: Date Filtering
This is to minimise the number of markdown files that need to be processed by the metadata parser in step 3.

The 6-month threshold is the default, it is configurable.

Fortunately, publication dates are within the URL of each article. It is quite standardised to identify the dates using
regular expressions in the article URL. This script will also return a list of Markdown files without dates, for further inspection
"""


import os
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import re

def extract_date_from_url(md_file):
    """Extract date from the first line of the markdown file URL: /YYYY/MM/DD/"""
    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
        
        match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', first_line)
        if match:
            year, month, day = match.groups()
            return datetime(int(year), int(month), int(day))
    except Exception as e:
        print(f"Error reading {md_file}: {e}")
    return None

def is_within_six_months(article_date, reference_date=None):
    if reference_date is None:
        reference_date = datetime.now()
    return article_date >= reference_date - timedelta(days=180)

def process_markdown_files(root_folder, archive_folder):
    """Filter markdown files: keep recent, move older than 6 months to archive."""
    root_path = Path(root_folder)
    archive_path = Path(archive_folder)
    archive_path.mkdir(exist_ok=True, parents=True)

    recent_articles, old_articles = [], []

    for md_file in root_path.rglob("*.md"):
        article_date = extract_date_from_url(md_file)
        if article_date:
            if is_within_six_months(article_date):
                recent_articles.append({'file': md_file, 'date': article_date})
            else:
                old_articles.append({'file': md_file, 'date': article_date})
                # Move to archive, preserving subfolder structure
                archive_sub = archive_path / md_file.parent.name
                archive_sub.mkdir(exist_ok=True)
                shutil.move(str(md_file), str(archive_sub / md_file.name))
        else:
            print(f"⚠️ Could not extract date from URL in {md_file.name}")

    print(f"\nRecent articles (within 6 months): {len(recent_articles)}")
    print(f"Old articles moved to archive: {len(old_articles)}")
    return recent_articles, old_articles

def main():
    cwd = os.getcwd()
    crawl_root_folder = input(f"Enter path to markdown root folder (default: {cwd}): ").strip() or cwd
    archive_folder = input(f"Enter path for archive folder (default: {cwd}/archive): ").strip() or os.path.join(cwd, "archive")

    recent, old = process_markdown_files(crawl_root_folder, archive_folder)
    print("\n✓ Processing complete.")

if __name__ == "__main__":
    main()

