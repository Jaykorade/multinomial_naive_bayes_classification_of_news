import pandas as pd
import json 
# Path to the JSON file
json_file = 'News_Category_Dataset.json'

# Read JSON file line by line and convert to DataFrame
with open(json_file, 'r') as file:
    data = [json.loads(line) for line in file]

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
csv_file = 'output.csv'
df.to_csv(csv_file, index=False)

print(f"JSON data has been converted to CSV and saved as '{csv_file}'.")

