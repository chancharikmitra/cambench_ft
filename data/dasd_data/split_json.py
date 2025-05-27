import json
import math
import os

# Path to your large JSON file
input_file = "llava_cot_sum_dpo.json"

# Read the JSON file
print(f"Reading {input_file}...")
with open(input_file, 'r') as f:
    data = json.load(f)

original_count = len(data)
print(f"Original number of samples: {original_count}")

# Function to check if a sample has non-string values where strings are expected
def is_valid_sample(sample):
    # Check if required fields exist
    if not all(key in sample for key in ["conversations", "chosen", "rejected"]):
        return False
    
    # Check if "conversations" is a list with at least one item
    if not isinstance(sample["conversations"], list) or len(sample["conversations"]) == 0:
        return False
    
    # Check if each conversation item has "from" and "value" as strings
    for conv in sample["conversations"]:
        if not isinstance(conv, dict):
            return False
        if not all(key in conv for key in ["from", "value"]):
            return False
        if not isinstance(conv["from"], str) or not isinstance(conv["value"], str):
            return False
    
    # Check if "chosen" and "rejected" have "from" and "value" as strings
    for role in ["chosen", "rejected"]:
        if not isinstance(sample[role], dict):
            return False
        if not all(key in sample[role] for key in ["from", "value"]):
            return False
        if not isinstance(sample[role]["from"], str) or not isinstance(sample[role]["value"], str):
            return False
    
    # If we reach here, the sample is valid
    return True

# Filter out invalid samples
valid_data = [sample for sample in data if is_valid_sample(sample)]
removed_count = original_count - len(valid_data)
print(f"Removed {removed_count} invalid samples")
print(f"Remaining valid samples: {len(valid_data)}")

# Calculate the size of each part
total_items = len(valid_data)
items_per_file = math.ceil(total_items / 3)
print(f"Total valid items: {total_items}, Items per file: {items_per_file}")

# Split and save the data into three files
for i in range(3):
    start_idx = i * items_per_file
    end_idx = min((i + 1) * items_per_file, total_items)
    part_data = valid_data[start_idx:end_idx]
    
    # Create output filename
    output_file = f"llava_cot_sum_dpo_part{i+1}.json"
    
    # Save to file
    print(f"Writing part {i+1} with {len(part_data)} items to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(part_data, f, ensure_ascii=False, indent=2)

print("Done! Files created:")
for i in range(3):
    file_name = f"llava_cot_sum_dpo_part{i+1}.json"
    file_size = os.path.getsize(file_name) / (1024 * 1024)  # Size in MB
    print(f"  - {file_name} ({file_size:.2f} MB)")

print(f"\nSummary:")
print(f"  - Original samples: {original_count}")
print(f"  - Invalid samples removed: {removed_count}")
print(f"  - Valid samples processed: {len(valid_data)}")