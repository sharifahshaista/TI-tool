import pandas as pd
import re
from pathlib import Path
from datetime import datetime, timedelta


def filter_csv_by_article_urls_and_date(input_csv, output_csv=None):
    """
    Filter CSV file by:
      1. Keeping only article URLs of the form: https://cleantechnica.com/YYYY/MM/DD/...
      2. Removing entries older than 6 months based on 'date' column.
    
    Args:
        input_csv (str): Path to input CSV file
        output_csv (str): Path to output CSV file (optional, defaults to input_articles_recent.csv)
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    # Regex pattern to match article URLs
    article_pattern = re.compile(r"^https://cleantechnica\.com/\d{4}/\d{2}/\d{2}/.+")
    
    # Read the CSV
    try:
        df = pd.read_csv(input_csv)
        print(f"Loaded CSV with {len(df)} entries")
    except FileNotFoundError:
        print(f"Error: File '{input_csv}' not found")
        return None
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

    if 'url' not in df.columns:
        print("Error: 'url' column not found in CSV")
        print(f"Available columns: {list(df.columns)}")
        return None

    if 'date' not in df.columns:
        print("Error: 'date' column not found in CSV")
        print(f"Available columns: {list(df.columns)}")
        return None

    # Step 1: Filter only valid article URLs
    mask = df['url'].str.match(article_pattern, na=False)
    df = df[mask]
    print(f"Remaining after URL filter: {len(df)}")

    # Step 2: Filter out old articles (older than 6 months)
    # Parse 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Define cutoff
    cutoff_date = datetime.now() - timedelta(days=6*30)  # ~6 months
    mask_recent = df['date'] >= cutoff_date
    filtered_df = df[mask_recent]

    print(f"Remaining after date filter: {len(filtered_df)}")

    # Save filtered CSV
    if output_csv is None:
        input_path = Path(input_csv)
        output_csv = input_path.parent / f"{input_path.stem}_articles_recent{input_path.suffix}"

    try:
        filtered_df.to_csv(output_csv, index=False)
        print(f"\nFiltered CSV saved to: {output_csv}")
    except Exception as e:
        print(f"Error saving filtered CSV: {e}")

    return filtered_df


def main():
    # Config
    INPUT_CSV = "cleantechnica_raw.csv"  # Change this to your CSV file name
    OUTPUT_CSV = "cleantechnica.csv"    
    print()
    print("CleanTechnica CSV URL + Date Filter")
    print("=" * 40)

    filtered_df = filter_csv_by_article_urls_and_date(INPUT_CSV, OUTPUT_CSV)

    if filtered_df is not None:
        print(f"\nFiltering successful!")

        # Show sample of remaining data
        if len(filtered_df) > 0:
            print(f"\nSample of remaining entries:")
            print("-" * 50)
            for i, row in filtered_df.head(3).iterrows():
                print(f"Title: {row['title']}")
                print(f"URL: {row['url']}")
                print(f"Date: {row['date']}")
                print("-" * 50)
    else:
        print("Filtering failed!")


if __name__ == "__main__":
    main()
