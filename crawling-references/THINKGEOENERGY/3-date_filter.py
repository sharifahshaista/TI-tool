import os
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path

def extract_date_from_markdown(content):
    """
    Extract publication date from markdown content.
    Only looks for date patterns that come immediately after an author profile.
    Example:
        [**Carlo Cariaga**](author-link) 20 Oct 2023
    """
    # Regex: match author link, then capture a date right after it
    pattern = r'\[.*?\]\(.*?/author/.*?\)\s+(\d{1,2}\s+\w+\s+\d{4})'

    match = re.search(pattern, content, re.MULTILINE)
    if match:
        date_str = match.group(1).strip()
        for fmt in ("%d %B %Y", "%d %b %Y"):
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
    return None

def is_within_six_months(article_date, reference_date=None):
    """
    Check if article date is within the last 6 months from reference date.
    """
    if reference_date is None:
        reference_date = datetime.now()
    
    six_months_ago = reference_date - timedelta(days=180)  # ~6 months
    return article_date >= six_months_ago

def process_markdown_files(root_folder, archive_folder=None, dry_run=True):
    """
    Process markdown files in subdomain folders and filter by date.
    Retain articles with author+date within 6 months.
    Move older ones to archive folder.
    """
    root_path = Path(root_folder)
    recent_articles = []
    old_articles = []
    
    if not root_path.exists():
        print(f"ERROR: Root folder does not exist: {root_folder}")
        return recent_articles, old_articles
    
    if archive_folder and not dry_run:
        archive_path = Path(archive_folder)
        archive_path.mkdir(exist_ok=True)
    
    subdomain_folders = [item for item in root_path.iterdir() if item.is_dir()]
    if not subdomain_folders:
        subdomain_folders = [root_path]
    
    for subdomain_folder in subdomain_folders:
        markdown_files = list(subdomain_folder.glob("*.md"))
        
        for md_file in markdown_files:
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()
                
                article_date = extract_date_from_markdown(content)
                
                if article_date:
                    if is_within_six_months(article_date):
                        recent_articles.append({
                            "file": md_file,
                            "date": article_date,
                            "subdomain": subdomain_folder.name
                        })
                        print(f"✓ Recent: {md_file.name} ({article_date.strftime('%d %b %Y')}) - KEEPING")
                    else:
                        old_articles.append({
                            "file": md_file,
                            "date": article_date,
                            "subdomain": subdomain_folder.name
                        })
                        print(f"✗ Old: {md_file.name} ({article_date.strftime('%d %b %Y')}) - MOVING to archive")
                        
                        if archive_folder and not dry_run:
                            archive_subdomain = Path(archive_folder) / subdomain_folder.name
                            archive_subdomain.mkdir(exist_ok=True)
                            shutil.move(str(md_file), str(archive_subdomain / md_file.name))
                
                else:
                    print(f"? No author+date found: {md_file.name} (IGNORED)")
            
            except Exception as e:
                print(f"Error processing {md_file.name}: {e}")
    
    return recent_articles, old_articles

def main():
    """
    Main function to run the script.
    """
    current_dir = "/Users/sharifahshaista/GenAI-for-TI/canarymedia/crawl_output"
    crawl_root_folder = input("Enter path to crawl4ai output folder (press Enter for default): ").strip() or current_dir
    archive_folder = input("Enter path for archive folder (press Enter for default): ").strip() or "/Users/sharifahshaista/GenAI-for-TI/canarymedia/archives"
    
    print("Markdown Date Filter Script")
    print("="*50)
    print("Running in DEBUG mode (no files moved)...")
    
    recent, old = process_markdown_files(
        root_folder=crawl_root_folder,
        archive_folder=archive_folder,
        dry_run=True
    )
    
    if old:
        print(f"\nFOUND {len(old)} old files that would be MOVED to archive.")
        confirm = input("Type 'ARCHIVE' to confirm moving: ").strip()
        
        if confirm == "ARCHIVE":
            print("\nRunning in LIVE mode - MOVING old files...")
            _, old_live = process_markdown_files(
                root_folder=crawl_root_folder,
                archive_folder=archive_folder,
                dry_run=False
            )
            print(f"✓ Moved {len(old_live)} old files to archive.")
        else:
            print("Archiving cancelled.")
    else:
        print("\nNo old files found. Nothing to archive.")

if __name__ == "__main__":
    main()
