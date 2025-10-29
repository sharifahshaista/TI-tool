"""
This script will remove the toggle navigation portion in every Markdown file.
This is because I enabled "include all external links". Some articles have hyperlinks
within the content which I don't want to omit.

Modify the path to the folder containing the crawled content.
"""

import os
import re

def clean_markdown_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".md"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Remove the large navigation/menu block
            # Captures everything from "Toggle navigation" up to "## News"
            cleaned_content = re.sub(
                r"Toggle navigation.*?## News",
                "## News",  # replace with just the section header
                content,
                flags=re.DOTALL
            )

            # Save cleaned content back
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(cleaned_content)

            print(f"Cleaned: {filename}")

# 🔧 Change this to your markdown folder path
folder_path = "/Users/sharifahshaista/GenAI-for-TI/B-sitemaps/iv-thinkgeoenergy/thinkgeoenergy_sitemap_crawl"
clean_markdown_files(folder_path)
