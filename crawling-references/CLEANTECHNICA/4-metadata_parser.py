#!/usr/bin/env python3
"""
Script to extract metadata from CleanTechnica markdown files.
Extracts URL, title, publication date, author, and main content (between 3rd pair of *** separators).
"""

import re
from pathlib import Path
from datetime import datetime
import pandas as pd

# Optional: Add URLs to skip
BLOCKED_URLS = set([])


# ---------- Utility: Convert URL slug to readable title ----------
def url_to_title(url):
    """Convert URL slug to title case."""
    if not url:
        return None
    slug = url.rstrip("/").split("/")[-1]
    words = slug.replace(".md", "").split("-")

    special_upper = {"EV", "SUV", "BMW", "VW", "ID", "CEO", "CFO", "AI", "US", "USA", "UK"}
    lower_words = {"and", "or", "the", "a", "an", "in", "on", "at", "to", "for", "of", "with", "by"}

    title_words = []
    for i, w in enumerate(words):
        if w.upper() in special_upper:
            title_words.append(w.upper())
        elif w.lower() in lower_words and i != 0:
            title_words.append(w.lower())
        else:
            title_words.append(w.capitalize())

    return " ".join(title_words)


# ---------- Main content extraction ----------
def extract_between_third_asterisks(text):
    """
    Extract content between the 3rd and 4th *** (or * * *) separators in the markdown.
    """
    # Match both '***' and '* * *'
    parts = re.split(r'\n\s*(?:\*{3}|(?:\*\s*){3})\s*\n', text)

    # parts[0] = preamble
    # parts[1] = after first pair
    # parts[2] = after second pair
    # parts[3] = after third pair → main content
    if len(parts) >= 4:
        return parts[3].strip()
    else:
        return None


def clean_content(content):
    """Remove footers, links, images, and boilerplate from extracted content."""
    if not content:
        return None

    # Remove markdown images
    content = re.sub(r"!\[.*?\]\(https?://[^\)]+\)", "", content)

    # Keep only link text from markdown hyperlinks
    content = re.sub(r"\[([^\]]+)\]\(https?://[^\)]+\)", r"\1", content)

    # Remove affiliate and footer text
    content = re.sub(
        r"(?si)(support cleantechnica|sign up|get our free daily|advertisement|subscribe|watch our|contact us here|have a tip).*",
        "",
        content,
    )

    # Remove promotional underscores and "Filed Under"/"Tags"
    content = re.sub(r"(_){2,}", "", content)
    content = re.sub(r"(Filed Under:|Tags:).*", "", content, flags=re.IGNORECASE)

    # Collapse excessive whitespace
    content = re.sub(r"\s+", " ", content).strip()
    return content


# ---------- Main parsing function ----------
def parse_markdown(file_path):
    """Extract metadata fields from a single markdown file."""
    data = {
        "url": None,
        "title": None,
        "date": None,
        "author": None,
        "content": None,
    }

    text = Path(file_path).read_text(encoding="utf-8")

    # --- URL ---
    url_match = re.search(r'^#\s+(https://cleantechnica\.com/[^\s]+)', text, re.MULTILINE)
    if url_match:
        data["url"] = url_match.group(1).strip()

    # --- Title ---
    if data["url"]:
        data["title"] = url_to_title(data["url"])

    # --- Date ---
    if data["url"]:
        date_match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', data["url"])
        if date_match:
            y, m, d = date_match.groups()
            try:
                data["date"] = datetime(int(y), int(m), int(d)).strftime("%d %B %Y")
            except ValueError:
                data["date"] = f"{d}/{m}/{y}"

    # --- Author ---
    author_match = re.search(
        r'\[\s*([A-Za-z\s]+)\s*\]\(https://cleantechnica\.com/author/[^\)]+\)',
        text,
        re.IGNORECASE,
    )
    if author_match:
        data["author"] = author_match.group(1).strip()

    # --- Main content (between 3rd pair of *** separators) ---
    raw_content = extract_between_third_asterisks(text)
    data["content"] = clean_content(raw_content)

    return data


# ---------- Folder processing ----------
def process_markdown_folder(root_folder):
    """Walk through folder, parse markdown files, and return DataFrame."""
    rows = []
    root = Path(root_folder)

    for md_file in root.rglob("*.md"):
        try:
            parsed = parse_markdown(md_file)
            if not parsed["url"] or not parsed["title"]:
                continue
            if parsed["url"] in BLOCKED_URLS:
                continue

            parsed["file_name"] = md_file.name
            parsed["folder"] = md_file.parent.name
            rows.append(parsed)
        except Exception as e:
            print(f"⚠️ Error processing {md_file}: {e}")

    df = pd.DataFrame(rows)
    return df


# ---------- Entry point ----------
if __name__ == "__main__":
    folder = "/Users/sharifahshaista/GenAI-for-TI/C-baseURL/ii-cleantechnica/crawl_output"
    df = process_markdown_folder(folder)

    df.to_csv("cleantechnica_raw.csv", index=False)
    print(f"✅ Processed {len(df)} articles into cleantechnica_raw.csv")

    if not df.empty:
        sample = df.iloc[0]
        print("\nSample extracted data:")
        print("-" * 50)
        print(f"URL: {sample['url']}")
        print(f"Title: {sample['title']}")
        print(f"Date: {sample['date']}")
        print(f"Author: {sample['author']}")
        print(f"Content length: {len(sample['content']) if sample['content'] else 0}")
        print(f"Content preview: {sample['content'][:200] if sample['content'] else 'None'}...")
