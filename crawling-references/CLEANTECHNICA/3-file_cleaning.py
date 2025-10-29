#!/usr/bin/env python3
"""
Script to delete markdown files with numbered suffixes.
Keeps files like 'filename.md' and deletes files like 'filename_1.md', 'filename-2.md', etc.
"""

import os
import re
import argparse
from pathlib import Path


def find_numbered_markdown_files(folder_path):
    """
    Find all markdown files with numbered suffixes in the given folder.
    
    Args:
        folder_path (str): Path to the folder containing markdown files
        
    Returns:
        list: List of file paths that match the numbered pattern
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Pattern to match files ending with _number.md or -number.md
    # This matches patterns like: filename_1.md, filename-2.md, filename_123.md
    pattern = re.compile(r'^(.+)[-_](\d+)\.md$')
    
    numbered_files = []
    
    for file_path in folder.glob('*.md'):
        match = pattern.match(file_path.name)
        if match:
            base_name, number = match.groups()
            numbered_files.append({
                'path': file_path,
                'base_name': base_name,
                'number': number,
                'full_name': file_path.name
            })
    
    return numbered_files


def delete_files(files_to_delete, dry_run=True):
    """
    Delete the specified files.
    
    Args:
        files_to_delete (list): List of file dictionaries to delete
        dry_run (bool): If True, only print what would be deleted without actually deleting
    """
    if not files_to_delete:
        print("No numbered markdown files found to delete.")
        return
    
    print(f"Found {len(files_to_delete)} numbered markdown files:")
    print("-" * 50)
    
    for file_info in files_to_delete:
        print(f"  {file_info['full_name']}")
    
    print("-" * 50)
    
    if dry_run:
        print("\n[DRY RUN] The above files would be deleted.")
        print("Run with --execute to actually delete them.")
    else:
        print("\nDeleting files...")
        deleted_count = 0
        for file_info in files_to_delete:
            try:
                file_info['path'].unlink()
                print(f"  ✓ Deleted: {file_info['full_name']}")
                deleted_count += 1
            except Exception as e:
                print(f"  ✗ Error deleting {file_info['full_name']}: {e}")
        
        print(f"\nSuccessfully deleted {deleted_count} out of {len(files_to_delete)} files.")


def main():
    # Configuration - Change these values as needed
    FOLDER_PATH = "/Users/sharifahshaista/GenAI-for-TI/C-baseURL/ii-cleantechnica/crawl_output"  # Change this to your folder path, or use "." for current directory
    EXECUTE_DELETE = True  # Change to True to actually delete files (False = dry run)
    
    # Try to use command line arguments if provided, otherwise use the config above
    try:
        parser = argparse.ArgumentParser(
            description="Delete markdown files with numbered suffixes",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python delete_numbered_md.py /path/to/folder                # Dry run (preview only)
  python delete_numbered_md.py /path/to/folder --execute      # Actually delete files
  python delete_numbered_md.py .                              # Process current directory
            """
        )
        
        parser.add_argument(
            'folder',
            nargs='?',  # Make folder argument optional
            default=FOLDER_PATH,
            help='Path to the folder containing markdown files'
        )
        
        parser.add_argument(
            '--execute',
            action='store_true',
            help='Actually delete the files (default is dry run)'
        )
        
        args = parser.parse_args()
        folder_path = args.folder
        execute_delete = args.execute or EXECUTE_DELETE
        
    except SystemExit:
        # If argument parsing fails (e.g., running in IDE), use config values
        print("Using configuration values (not command line arguments)")
        folder_path = FOLDER_PATH
        execute_delete = EXECUTE_DELETE
    
    print(f"Processing folder: {folder_path}")
    print(f"Mode: {'EXECUTE' if execute_delete else 'DRY RUN'}")
    print()
    
    try:
        # Find numbered markdown files
        numbered_files = find_numbered_markdown_files(folder_path)
        
        # Delete files (or show what would be deleted)
        delete_files(numbered_files, dry_run=not execute_delete)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())