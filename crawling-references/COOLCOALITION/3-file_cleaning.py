import os
import re
import glob
from pathlib import Path

def remove_duplicate_markdown_files(folder_path="."):
    """
    Remove duplicate markdown files that have numbers at the end of their filename.
    Keeps the original file (without number) and removes numbered duplicates.
    
    Args:
        folder_path (str): Path to the folder containing markdown files
    """
    
    # Convert to Path object for easier handling
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    # Pattern to match markdown files with numbers at the end
    # Examples: file_1.md, file_2.md, article_10.md, etc.
    pattern = re.compile(r'^(.+)_(\d+)\.md$')
    
    # Get all markdown files in the folder
    md_files = list(folder.glob("*.md"))
    
    files_to_delete = []
    originals_found = []
    
    print(f"Scanning {len(md_files)} markdown files in '{folder_path}'...")
    print("-" * 60)
    
    for file_path in md_files:
        filename = file_path.name
        match = pattern.match(filename)
        
        if match:
            base_name = match.group(1)  # Part before the number
            number = match.group(2)     # The number
            
            # Check if original file (without number) exists
            original_filename = f"{base_name}.md"
            original_path = folder / original_filename
            
            if original_path.exists():
                files_to_delete.append(file_path)
                print(f"Will DELETE: {filename} (duplicate of {original_filename})")
            else:
                # Original doesn't exist, keep track for info
                print(f"KEEP: {filename} (no original {original_filename} found)")
    
    print("-" * 60)
    
    if not files_to_delete:
        print("No duplicate files found to delete.")
        return
    
    print(f"\nFound {len(files_to_delete)} duplicate files to delete:")
    for file_path in files_to_delete:
        print(f"  - {file_path.name}")
    
    # Confirm deletion
    response = input(f"\nDo you want to delete these {len(files_to_delete)} duplicate files? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        deleted_count = 0
        errors = []
        
        for file_path in files_to_delete:
            try:
                file_path.unlink()  # Delete the file
                print(f"✓ Deleted: {file_path.name}")
                deleted_count += 1
            except Exception as e:
                error_msg = f"✗ Error deleting {file_path.name}: {str(e)}"
                print(error_msg)
                errors.append(error_msg)
        
        print(f"\nSummary:")
        print(f"✓ Successfully deleted: {deleted_count} files")
        if errors:
            print(f"✗ Errors: {len(errors)} files")
            for error in errors:
                print(f"  {error}")
    else:
        print("Deletion cancelled.")

def preview_duplicates(folder_path="."):
    """
    Preview duplicate files without deleting them.
    
    Args:
        folder_path (str): Path to the folder containing markdown files
    """
    
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    pattern = re.compile(r'^(.+)_(\d+)\.md$')
    md_files = list(folder.glob("*.md"))
    
    duplicates = {}
    
    for file_path in md_files:
        filename = file_path.name
        match = pattern.match(filename)
        
        if match:
            base_name = match.group(1)
            number = int(match.group(2))
            
            if base_name not in duplicates:
                duplicates[base_name] = []
            duplicates[base_name].append((filename, number))
    
    if not duplicates:
        print("No duplicate markdown files found.")
        return
    
    print(f"Preview of duplicate groups in '{folder_path}':")
    print("=" * 60)
    
    for base_name, files in duplicates.items():
        original_file = f"{base_name}.md"
        original_exists = (folder / original_file).exists()
        
        print(f"\nGroup: {base_name}")
        print(f"Original file: {original_file} {'(EXISTS)' if original_exists else '(MISSING)'}")
        print("Duplicates:")
        
        # Sort by number
        files.sort(key=lambda x: x[1])
        for filename, number in files:
            print(f"  - {filename}")

if __name__ == "__main__":
    import sys
    
    # Default to current directory if no argument provided
    folder_path = sys.argv[1] if len(sys.argv) > 1 else "."
    folder_path = "/Users/sharifahshaista/GenAI-for-TI/C-baseURL/iv-coolcoalition/cool_coalition_crawl"
    
    print("Duplicate Markdown File Remover")
    print("=" * 40)
    print("This script removes markdown files with numbers at the end (e.g., file_1.md, file_2.md)")
    print("It keeps the original file (file.md) and removes numbered duplicates.\n")
    
    # First show preview
    print("PREVIEW MODE:")
    preview_duplicates(folder_path)
    
    print("\n" + "=" * 60)
    
    # Then ask if user wants to proceed with deletion
    proceed = input("Do you want to proceed with deletion? (y/N): ").strip().lower()
    if proceed in ['y', 'yes']:
        print("\nDELETION MODE:")
        remove_duplicate_markdown_files(folder_path)
    else:
        print("Operation cancelled.")