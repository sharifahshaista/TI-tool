import os
import re
import pandas as pd
from pathlib import Path

# --- Blocked URLs (category/landing pages) ---
BLOCKED_URLS = {
    "https://www.canarymedia.com/articles/air-travel",
    "https://www.canarymedia.com/articles/batteries",
    "https://www.canarymedia.com/articles/canary-media",
    "https://www.canarymedia.com/articles/carbon-capture",
    "https://www.canarymedia.com/articles/carbon-removal",
    "https://www.canarymedia.com/articles/carbon-free-buildings",
    "https://www.canarymedia.com/articles/clean-aluminum",
    "https://www.canarymedia.com/articles/clean-energy",
    "https://www.canarymedia.com/articles/clean-energy-jobs",
    "https://www.canarymedia.com/articles/clean-energy-manufacturing",
    "https://www.canarymedia.com/articles/clean-energy-supply-chain",
    "https://www.canarymedia.com/articles/clean-fleets",
    "https://www.canarymedia.com/articles/clean-industry",
    "https://www.canarymedia.com/articles/climate-crisis",
    "https://www.canarymedia.com/articles/climate-justice",
    "https://www.canarymedia.com/articles/climatetech-finance",
    "https://www.canarymedia.com/articles/corporate-procurement",
    "https://www.canarymedia.com/articles/culture",
    "https://www.canarymedia.com/articles/distributed-energy-resources",
    "https://www.canarymedia.com/articles/electric-vehicles",
    "https://www.canarymedia.com/articles/electrification",
    "https://www.canarymedia.com/articles/emissions-reduction",
    "https://www.canarymedia.com/articles/energy-efficiency",
    "https://www.canarymedia.com/articles/energy-equity",
    "https://www.canarymedia.com/articles/energy-markets",
    "https://www.canarymedia.com/articles/energy-storage",
    "https://www.canarymedia.com/articles/enn",
    "https://www.canarymedia.com/articles/ev-charging",
    "https://www.canarymedia.com/articles/food-and-farms",
    "https://www.canarymedia.com/articles/fossil-fuels",
    "https://www.canarymedia.com/articles/fun-stuff",
    "https://www.canarymedia.com/articles/geothermal",
    "https://www.canarymedia.com/articles/green-steel",
    "https://www.canarymedia.com/articles/grid-edge",
    "https://www.canarymedia.com/articles/guides-and-how-tos",
    "https://www.canarymedia.com/articles/heat-pumps",
    "https://www.canarymedia.com/articles/hydrogen",
    "https://www.canarymedia.com/articles/hydropower",
    "https://www.canarymedia.com/articles/just-transition",
    "https://www.canarymedia.com/articles/land-use",
    "https://www.canarymedia.com/articles/liquefied-natural-gas",
    "https://www.canarymedia.com/articles/long-duration-energy-storage",
    "https://www.canarymedia.com/articles/sea-transport",
    "https://www.canarymedia.com/articles/methane",
    "https://www.canarymedia.com/articles/nuclear",
    "https://www.canarymedia.com/articles/ocean-energy",
    "https://www.canarymedia.com/articles/offshore-wind",
    "https://www.canarymedia.com/articles/policy-regulation",
    "https://www.canarymedia.com/articles/politics",
    "https://www.canarymedia.com/articles/public-transit",
    "https://www.canarymedia.com/articles/recycling-renewables",
    "https://www.canarymedia.com/articles/solar",
    "https://www.canarymedia.com/articles/sponsored",
    "https://www.canarymedia.com/articles/transmission",
    "https://www.canarymedia.com/articles/transportation",
    "https://www.canarymedia.com/articles/utilities",
    "https://www.canarymedia.com/articles/virtual-power-plants",
    "https://www.canarymedia.com/articles/wind",
    "https://www.canarymedia.com/articles/workforce-diversity",
}


def parse_markdown(file_path):
    """Parse a markdown file into structured metadata."""
    data = {
        "url": None,
        "title": None,
        "date": None,
        "author": None,
        "content": None,
        "tags": ""
    }

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    lines = text.splitlines()

    # --- URL ---
    for line in lines:
        if line.startswith("# http"):
            data["url"] = line.lstrip("# ").strip()
            break

    # --- Title ---
    title_match = re.search(r'scrolledValue\s*["\']>\s*#\s+(.*)', text)
    if title_match:
        data["title"] = title_match.group(1).strip()

    # --- Author ---
    author_match = re.search(
        r"\[([^\]]+)\]\(https://www\.canarymedia\.com/about/people/[^\)]+\)", text
    )
    if author_match:
        data["author"] = author_match.group(1).strip()

    # --- Date ---
    date_match = re.search(r"\b(\d{1,2}\s+[A-Z][a-z]+\s+\d{4})\b", text)
    if date_match:
        data["date"] = date_match.group(1).strip()

    # --- Tags ---
    tag_matches = re.findall(r"\*\s*\[([^\]]+)\]\([^)]+\)", text)
    if tag_matches:
        data["tags"] = ", ".join(t.strip() for t in tag_matches)

    # --- Content ---
    content_section = []
    in_content = False
    for line in lines:
        if line.strip().startswith("scrolledValue"):
            in_content = True
            continue
        if in_content:
            # Stop at tags section
            if line.strip().startswith("* ["):
                break

            # Skip title line (starts with '# ')
            if line.strip().startswith("# "):
                continue
            # Skip author line
            if re.match(r"^By\s+\[.*?\]\(", line.strip()):
                continue
            # Skip date line
            if re.match(r"^\d{1,2}\s+[A-Z][a-z]+\s+\d{4}$", line.strip()):
                continue
            # Skip image captions
            if re.match(r"^\(.*Getty Images.*\)$", line.strip()) or re.match(r"^\([^)]*\)$", line.strip()):
                continue

            content_section.append(line)

    data["content"] = "\n".join(content_section).strip()

    return data


def process_markdown_folder(root_folder):
    """Process all markdown files into a DataFrame, filtering out blocked URLs."""
    root = Path(root_folder)
    rows = []

    for md_file in root.rglob("*.md"):
        parsed = parse_markdown(md_file)

        # Skip if URL is in BLOCKED_URLS
        if parsed["url"] in BLOCKED_URLS:
            continue

        parsed["file_name"] = md_file.name
        parsed["folder"] = md_file.parent.name
        rows.append(parsed)

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    # Ask user for folder path
    folder = input("Enter the path to your crawl4ai output folder: ").strip()

    if not folder:
        folder = "crawl_output"  # default
        print(f"No path entered. Using default: {folder}")

    df = process_markdown_folder(folder)

    out_csv = "canary_media_raw.csv"
    df.to_csv(out_csv, index=False)

    print(f"Processed {len(df)} articles after filtering.")
    print(f"Saved to: {out_csv}")
    print(df.head())
