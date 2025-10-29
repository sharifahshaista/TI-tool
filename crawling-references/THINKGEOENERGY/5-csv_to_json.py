import csv
import json

csv_file = "thinkgeoenergy.csv"
json_file = "thinkgeoenergy.json"

data = []
with open(csv_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

with open(json_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"CSV converted to JSON and saved as {json_file}")
