
import os
import json

# Define the directory containing your JSON files
json_dir = '/media/pro/28d83ea8-54ca-4f9d-b09d-604d9edec39f/probir/img_seg/data/train/json_file'

# Set to store unique labels and dictionary for mapping label to first encountered JSON filename
unique_labels = set()
label_first_file = {}

# Iterate over all files in the directory
for filename in os.listdir(json_dir):
    # Process only files ending with '.json' (case-insensitive)
    if filename.lower().endswith('.json'):
        file_path = os.path.join(json_dir, filename)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Assuming the JSON structure has an "objects" list containing annotations
            for obj in data.get('objects', []):
                label = obj.get('label', '').strip()  # Get the label and remove extra whitespace
                if label:
                    label_lower = label.lower()  # Lowercase to normalize duplicates
                    unique_labels.add(label_lower)
                    # If this label has not been seen before, record the file that provides this label
                    if label_lower not in label_first_file:
                        label_first_file[label_lower] = filename
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

# Print out the unique labels along with an example JSON filename for each label
print("Unique labels and their sample JSON file:")
for label in sorted(unique_labels):
    sample_file = label_first_file.get(label, "N/A")
    print(f"{label}: {sample_file}")

print("\nTotal number of unique classes:", len(unique_labels))

