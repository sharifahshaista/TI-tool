import json
import csv
from pathlib import Path

def update_csv_from_json(json_file, csv_file):
    """
    Update CSV file with title and publication_date from JSON file.
    Matches records by URL.
    """
    # Read JSON data
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # Create a mapping of URL to (title, publication_date) from JSON
    url_to_data = {}
    for record in json_data:
        if 'url' in record and 'title' in record and 'publication_date' in record:
            url_to_data[record['url']] = (record['title'], record['publication_date'])

    # Read CSV data
    csv_data = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            csv_data.append(row)

    # Update CSV data with JSON data
    for row in csv_data:
        if row['url'] in url_to_data:
            title, pub_date = url_to_data[row['url']]
            row['title'] = title
            row['publication_date'] = pub_date

    # Write back to CSV
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames) # type: ignore
        writer.writeheader()
        writer.writerows(csv_data)

if __name__ == "__main__":
    json_file = Path("/Users/sharifahshaista/TI-tool/processed_data/techcrunch_com_20251127_filtered_20251127.json")
    csv_file = Path("/Users/sharifahshaista/TI-tool/processed_data/techcrunch_com_20251127_filtered_20251127.csv")

    if json_file.exists() and csv_file.exists():
        update_csv_from_json(json_file, csv_file)
        print(f"Updated {csv_file} with data from {json_file}")
    else:
        print(f"Files not found: {json_file} or {csv_file}")