"""
Step 4: CSV Cleaning
This is to clean the CSV file by removing rows where the date is missing or empty, and rows where the title contains '404 Page Not Found'.
Note that these entries to sieve were "eyeballed" by me. 
"""


import pandas as pd
from pathlib import Path


def clean_csv(input_csv, output_csv=None):
    """
    Remove rows where:
      1. 'date' is missing or empty
      2. 'title' contains '404 Page Not Found'
    """
    try:
        df = pd.read_csv(input_csv)
        print(f"Loaded {len(df)} entries from {input_csv}")
    except FileNotFoundError:
        print(f"Error: File '{input_csv}' not found")
        return None
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

    initial_count = len(df)

    # 1. Drop rows with missing or empty date
    df = df.dropna(subset=["date"])
    df = df[df["date"].astype(str).str.strip() != ""]
    after_date_filter = len(df)
    print(f"Removed {initial_count - after_date_filter} rows with missing/empty date")

    # 2. Drop rows where 'title' contains "404 Page Not Found"
    if "title" in df.columns:
        mask_404 = df["title"].astype(str).str.contains("404 Page Not Found", case=False, na=False)
        df = df[~mask_404]
        after_404_filter = len(df)
        print(f"Removed {after_date_filter - after_404_filter} rows with '404 Page Not Found' in title")
    else:
        print("Warning: 'title' column not found — skipping 404 filtering")

    # Save cleaned CSV
    if output_csv is None:
        input_path = Path(input_csv)
        output_csv = input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}"

    try:
        df.to_csv(output_csv, index=False)
        print(f"\nCleaned CSV saved to: {output_csv}")
    except Exception as e:
        print(f"Error saving CSV: {e}")

    print(f"Final entries: {len(df)}")
    return df


if __name__ == "__main__":
    INPUT_CSV = "canary_media_raw.csv"
    OUTPUT_CSV = "canary_media.csv"
    clean_csv(INPUT_CSV, OUTPUT_CSV)

