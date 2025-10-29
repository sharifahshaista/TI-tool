import os
from pathlib import Path

def extract_first_line_url(filepath):
    """Extract URL from the first line of a markdown file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            # Check if first line starts with # and contains a URL
            if first_line.startswith('# http'):
                url = first_line[2:].strip()  # Remove '# ' prefix
                return url
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return None

def delete_non_article_files(folder_path, dry_run=True):
    """Delete markdown files that don't contain 'articles' in their URL."""
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    # Find all markdown files
    md_files = list(folder.glob('*.md')) + list(folder.glob('*.markdown'))
    
    if not md_files:
        print(f"No markdown files found in '{folder_path}'.")
        return
    
    print(f"Found {len(md_files)} markdown file(s).")
    print(f"{'DRY RUN - ' if dry_run else ''}Checking for files without 'articles' in URL...\n")
    
    to_delete = []
    to_keep = []
    no_url = []
    
    for md_file in md_files:
        url = extract_first_line_url(md_file)
        
        if url is None:
            no_url.append(md_file)
            print(f"⚠ No URL found: {md_file.name}")
        elif 'articles' in url.lower():
            to_keep.append((md_file, url))
        else:
            to_delete.append((md_file, url))
            print(f"{'[DRY RUN] Would delete' if dry_run else '🗑 Deleting'}: {md_file.name}")
            print(f"  URL: {url}")
    
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total files: {len(md_files)}")
    print(f"Files with 'articles' in URL (keeping): {len(to_keep)}")
    print(f"Files without 'articles' in URL (deleting): {len(to_delete)}")
    print(f"Files with no URL (skipping): {len(no_url)}")
    
    if not dry_run and to_delete:
        print(f"\n{'=' * 80}")
        print("DELETING FILES")
        print(f"{'=' * 80}")
        deleted_count = 0
        for md_file, url in to_delete:
            try:
                os.remove(md_file)
                deleted_count += 1
                print(f"✓ Deleted: {md_file.name}")
            except Exception as e:
                print(f"✗ Failed to delete {md_file.name}: {e}")
        
        print(f"\nSuccessfully deleted {deleted_count} file(s).")
    elif dry_run and to_delete:
        print(f"\nThis was a DRY RUN. Run with dry_run=False to actually delete files.")
    else:
        print(f"\nNo files to delete.")

if __name__ == "__main__":
    # Get folder path from user
    folder_path = input("Enter the folder path containing markdown files: ").strip()
    
    # Ask if user wants to do a dry run first
    dry_run_choice = input("Do you want to do a dry run first? (y/n, default=y): ").strip().lower()
    dry_run = dry_run_choice != 'n'
    
    delete_non_article_files(folder_path, dry_run=dry_run)
    
    # If it was a dry run, ask if user wants to proceed with actual deletion
    if dry_run:
        proceed = input("\nDo you want to proceed with actual deletion? (y/n): ").strip().lower()
        if proceed == 'y':
            delete_non_article_files(folder_path, dry_run=False)
        else:
            print("Deletion cancelled.")