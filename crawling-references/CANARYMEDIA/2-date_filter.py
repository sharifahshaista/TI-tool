"""
Step 2: Date Filtering
This is to minimise the number of markdown files that need to be processed by the metadata parser in step 3.

Run this script by copying the PATH to the folder containing the markdown files. The objective of this script is to remove any markdown files that are older than 6 months.
The 6-month threshold is the default, it is configurable.

You MUST input the path to the crawl output folder and the path to the archive folder.
"""

import os
import re
from datetime import datetime, timedelta
from pathlib import Path
import shutil

def extract_date_from_markdown(content):
    """
    Extract publication date from markdown content.
    Looks for date patterns after author line and before image references.
    """
    # Pattern to find date after author line (format: "DD Month YYYY")
    date_patterns = [
        r'By \[.*?\]\(.*?\)\s*\n(\d{1,2}\s+\w+\s+\d{4})',  # After author link
        r'By .*?\n(\d{1,2}\s+\w+\s+\d{4})',                # After author name
        r'(\d{1,2}\s+\w+\s+\d{4})',                        # Standalone date pattern
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            date_str = match.group(1).strip()
            try:
                # Parse date string like "10 October 2022"
                parsed_date = datetime.strptime(date_str, '%d %B %Y')
                return parsed_date
            except ValueError:
                try:
                    # Try alternative format like "10 Oct 2022"
                    parsed_date = datetime.strptime(date_str, '%d %b %Y')
                    return parsed_date
                except ValueError:
                    continue
    
    return None

def is_within_six_months(article_date, reference_date=None):
    """
    Check if article date is within the last 6 months from reference date.
    """
    if reference_date is None:
        reference_date = datetime.now()
    
    six_months_ago = reference_date - timedelta(days=180)  # Approximately 6 months
    return article_date >= six_months_ago

def process_markdown_files(root_folder, archive_folder=None, dry_run=True):
    """
    Process markdown files in subdomain folders and filter by date.
    
    Args:
        root_folder: Path to the root folder containing subdomain folders
        archive_folder: Path to archive folder for old articles (older than 6 months)
        dry_run: If True, only print what would be done without actually moving files
    """
    root_path = Path(root_folder)
    recent_articles = []
    old_articles = []
    
    # Debug: Check if root folder exists
    if not root_path.exists():
        print(f"ERROR: Root folder does not exist: {root_folder}")
        return recent_articles, old_articles
    
    print(f"DEBUG: Root folder found: {root_folder}")
    
    if archive_folder and not dry_run:
        archive_path = Path(archive_folder)
        archive_path.mkdir(exist_ok=True)
        print(f"DEBUG: Archive folder created/verified: {archive_folder}")
    
    # Debug: List all items in root folder
    all_items = list(root_path.iterdir())
    print(f"DEBUG: Found {len(all_items)} items in root folder")
    
    subdomain_folders = [item for item in all_items if item.is_dir()]
    print(f"DEBUG: Found {len(subdomain_folders)} subdomain folders")
    
    if not subdomain_folders:
        print("DEBUG: No subdomain folders found. Checking for markdown files in root folder...")
        # Check if markdown files are directly in root folder
        root_md_files = list(root_path.glob("*.md"))
        print(f"DEBUG: Found {len(root_md_files)} markdown files in root folder")
        if root_md_files:
            subdomain_folders = [root_path]  # Process root as a single folder
    
    # Iterate through all subdomain folders
    for subdomain_folder in subdomain_folders:
        print(f"\nProcessing folder: {subdomain_folder.name}")
        
        # Process all markdown files in this subdomain folder
        markdown_files = list(subdomain_folder.glob("*.md"))
        print(f"DEBUG: Found {len(markdown_files)} markdown files in {subdomain_folder.name}")
        
        if not markdown_files:
            # Check for other file types
            all_files = list(subdomain_folder.glob("*"))
            print(f"DEBUG: Total files in folder: {len(all_files)}")
            if all_files:
                print(f"DEBUG: File types found: {[f.suffix for f in all_files[:5]]}")  # Show first 5 file extensions
        
        for md_file in markdown_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                print(f"DEBUG: Processing file: {md_file.name}")
                
                # Debug: Show first few lines of content to understand structure
                lines = content.split('\n')[:15]  # First 15 lines
                print(f"DEBUG: First few lines of {md_file.name}:")
                for i, line in enumerate(lines):
                    print(f"  {i+1:2d}: {line}")
                
                # Extract date from content
                article_date = extract_date_from_markdown(content)
                print(f"DEBUG: Extracted date: {article_date}")
                
                if article_date:
                    if is_within_six_months(article_date):
                        recent_articles.append({
                            'file': md_file,
                            'date': article_date,
                            'subdomain': subdomain_folder.name
                        })
                        print(f"  ✓ Recent: {md_file.name} ({article_date.strftime('%d %B %Y')}) - KEEPING in original folder")
                            
                    else:
                        old_articles.append({
                            'file': md_file,
                            'date': article_date,
                            'subdomain': subdomain_folder.name
                        })
                        print(f"  ✗ Old: {md_file.name} ({article_date.strftime('%d %B %Y')}) - MOVING to archive")
                        
                        # Move old files to archive folder if not in dry run mode
                        if archive_folder and not dry_run:
                            try:
                                # Create subdomain folder in archive
                                archive_subdomain = archive_path / subdomain_folder.name
                                archive_subdomain.mkdir(exist_ok=True)
                                
                                # Move file to archive
                                archive_file_path = archive_subdomain / md_file.name
                                shutil.move(str(md_file), str(archive_file_path))
                                print(f"    MOVED: {md_file.name} → archive/{subdomain_folder.name}/")
                                
                            except Exception as move_error:
                                print(f"    ERROR moving {md_file.name}: {move_error}")
                        
                else:
                    print(f"  ? No date found: {md_file.name}")
                    # Debug: Try to find any date-like patterns
                    import re
                    date_like = re.findall(r'\d{1,2}\s+\w+\s+\d{4}', content)
                    if date_like:
                        print(f"    DEBUG: Found date-like patterns: {date_like[:3]}")  # Show first 3
                    
            except Exception as e:
                print(f"  Error processing {md_file.name}: {e}")
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"SUMMARY:")
    print(f"Recent articles (within 6 months): {len(recent_articles)} - KEPT in original folders")
    print(f"Old articles (older than 6 months): {len(old_articles)} - {'MOVED to archive' if not dry_run else 'WOULD BE MOVED to archive'}")
    if not dry_run and archive_folder:
        print(f"Archive location: {archive_folder}")
    print(f"{'='*50}")
    
    # Sort and display recent articles by date
    if recent_articles:
        print(f"\nRECENT ARTICLES (kept in original folders):")
        recent_articles.sort(key=lambda x: x['date'], reverse=True)
        for article in recent_articles[:10]:  # Show top 10 most recent
            print(f"  {article['date'].strftime('%d %B %Y')} - {article['subdomain']} - {article['file'].name}")
    
    # Show old articles that were/would be moved to archive
    if old_articles:
        action = "MOVED to archive" if not dry_run else "WOULD MOVE to archive"
        print(f"\nOLD ARTICLES ({action}):")
        old_articles.sort(key=lambda x: x['date'], reverse=True)
        for article in old_articles[:10]:  # Show oldest 10
            status = "ARCHIVED" if not dry_run else "MARKED FOR ARCHIVE"
            if not dry_run and archive_folder:
                print(f"  {article['date'].strftime('%d %B %Y')} - archive/{article['subdomain']} - {article['file'].name} [{status}]")
            else:
                print(f"  {article['date'].strftime('%d %B %Y')} - {article['subdomain']} - {article['file'].name} [{status}]")
    
    return recent_articles, old_articles

def main():
    """
    Main function to run the script.
    Modify the paths below according to your setup.
    """
    # Get the current working directory for reference
    current_dir = "/Users/sharifahshaista/GenAI-for-TI/canarymedia/crawl_output"
    print(f"Current working directory: {current_dir}")
    
    # Configure these paths according to your setup
    crawl_root_folder = input("Enter the path to your crawl4ai output folder: ").strip()
    
    # If user just presses enter, use current directory for testing
    if not crawl_root_folder:
        crawl_root_folder = current_dir
        print(f"Using current directory: {crawl_root_folder}")
    
    # Get archive folder path
    archive_folder = input("Enter the path for archive folder (or press Enter for 'archive' in current dir): ").strip()
    if not archive_folder:
        archive_folder = "/Users/sharifahshaista/GenAI-for-TI/canarymedia/archives"
        print(f"Using archive folder: {archive_folder}")
    
    print("Markdown Date Filter Script")
    print("="*50)
    
    # Run in debug mode to see what's happening
    print("Running in DEBUG mode (no files will be moved)...")
    recent, old = process_markdown_files(
        root_folder=crawl_root_folder,
        archive_folder=archive_folder,
        dry_run=True
    )
    
    # Ask user if they want to proceed with archiving
    if old:
        print(f"\n{'='*60}")
        print(f"FOUND: {len(old)} old files that would be MOVED to archive!")
        print(f"Archive destination: {archive_folder}")
        print(f"{'='*60}")
        
        confirm = input("\nDo you want to proceed with MOVING these old files to archive? (type 'ARCHIVE' to confirm): ").strip()
        
        if confirm == 'ARCHIVE':
            print("\nRunning in LIVE mode - MOVING old files to archive...")
            recent_live, old_live = process_markdown_files(
                root_folder=crawl_root_folder,
                archive_folder=archive_folder,
                dry_run=False
            )
            print(f"\n✓ Archiving complete. {len(old_live)} files were moved to archive.")
            print(f"✓ Recent files remain in their original subdomain folders.")
        else:
            print("\nArchiving cancelled. No files were moved.")
    else:
        print("\nNo old files found. Nothing to archive.")
    
    # Ask if user wants to process only one file for testing
    if not recent and not old:
        test_file = input("\nNo files processed. Enter path to a single markdown file to test (or press Enter to skip): ").strip()
        if test_file and os.path.exists(test_file):
            print(f"\nTesting single file: {test_file}")
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                print("File content preview:")
                lines = content.split('\n')[:20]
                for i, line in enumerate(lines):
                    print(f"  {i+1:2d}: {line}")
                
                date = extract_date_from_markdown(content)
                print(f"\nExtracted date: {date}")
                
                if date:
                    is_recent = is_within_six_months(date)
                    print(f"Is within 6 months: {is_recent}")
                else:
                    # Try to find any dates
                    import re
                    all_dates = re.findall(r'\d{1,2}\s+\w+\s+\d{4}', content)
                    print(f"All date-like patterns found: {all_dates}")
                    
            except Exception as e:
                print(f"Error testing file: {e}")


if __name__ == "__main__":
    main()