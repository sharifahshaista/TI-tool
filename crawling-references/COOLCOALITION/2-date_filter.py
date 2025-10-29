import os
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import re

def extract_date_from_markdown(md_file):
    """Extract the first date after a '#' header in the markdown file."""
    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Match date patterns like '07 December 2023' after a header '#'
        match = re.search(r'^\s*#.*?\n\s*(\d{1,2}\s+\w+\s+\d{4})', content, re.MULTILINE)
        if match:
            date_str = match.group(1).strip()
            try:
                return datetime.strptime(date_str, "%d %B %Y")
            except ValueError:
                return None
    except Exception as e:
        print(f"Error reading {md_file}: {e}")
    return None

def is_within_six_months(article_date, reference_date=None):
    if reference_date is None:
        reference_date = datetime.now()
    return article_date >= reference_date - timedelta(days=180)  # approx 6 months

def process_markdown_files(root_folder, archive_folder):
    """Filter markdown files: keep recent, move older than 6 months to archive."""
    root_path = Path(root_folder)
    archive_path = Path(archive_folder)
    archive_path.mkdir(exist_ok=True, parents=True)

    recent_articles, old_articles = [], []

    for md_file in root_path.rglob("*.md"):
        article_date = extract_date_from_markdown(md_file)
        if article_date:
            if is_within_six_months(article_date):
                recent_articles.append({'file': md_file, 'date': article_date})
            else:
                old_articles.append({'file': md_file, 'date': article_date})
                # Move to archive, preserving subfolder structure
                relative_sub = md_file.relative_to(root_path).parent
                archive_sub = archive_path / relative_sub
                archive_sub.mkdir(exist_ok=True, parents=True)
                shutil.move(str(md_file), str(archive_sub / md_file.name))
        else:
            print(f"Could not extract date from {md_file.name}; keeping file.")

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
