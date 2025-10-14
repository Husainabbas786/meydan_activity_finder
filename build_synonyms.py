import pandas as pd
import json
from collections import Counter
import glob

# Find the CSV file automatically
csv_files = glob.glob('enhanced_activities_*.csv')
if not csv_files:
    print("ERROR: No CSV file found!")
    exit()

csv_file = csv_files[0]
print(f"Loading: {csv_file}")

# Load your generated keywords
df = pd.read_csv(csv_file)

# Extract all keywords
all_keywords = []
for keywords in df['Generated Keywords']:
    if keywords and keywords != 'ERROR' and keywords != '-':
        kw_list = [k.strip().lower() for k in str(keywords).split(',')]
        all_keywords.extend(kw_list)

# Count frequency
keyword_freq = Counter(all_keywords)

# Print top 50 most common keywords
print("\nTop 50 Most Common Keywords:")
print("="*60)
for kw, count in keyword_freq.most_common(50):
    print(f"{kw}: {count}")

# Build synonym dictionary
synonym_dict = {
    "software": ["application", "app", "program", "coding", "development", "software development"],
    "website": ["web", "site", "web development", "web design", "online platform"],
    "repair": ["fix", "fixing", "maintenance", "service", "servicing"],
    "car": ["vehicle", "auto", "automobile", "motor vehicle"],
    "food": ["restaurant", "cafe", "dining", "catering", "eatery"],
    "import": ["importing", "trade", "international trade"],
    "export": ["exporting", "trade", "international sales"],
    "wholesale": ["bulk", "distributor", "trading"],
    "retail": ["shop", "store", "selling"],
    "consulting": ["consultant", "advisory", "advice", "consultancy"],
    "design": ["designing", "designer", "creative"],
    "manufacturing": ["production", "factory", "manufacturing"],
    "clinic": ["medical center", "health center", "healthcare"],
    "training": ["education", "courses", "learning", "teaching"]
}

# Save
with open('synonym_dictionary.json', 'w') as f:
    json.dump(synonym_dict, f, indent=2)

print(f"\n✅ Synonym dictionary saved to synonym_dictionary.json")
print(f"✅ Contains {len(synonym_dict)} synonym groups")