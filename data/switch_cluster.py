#!/usr/bin/env python3

import json
import os

# Config - file to modify
json_file = "dataset_info.json"
from_cluster = "cam_motion_TRINITY"
to_cluster = "cam_motion_LAMBDA"

print(f"Updating paths in {json_file}: {from_cluster} → {to_cluster}")

try:
    # Read the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    modified = False
    modified_entries = []
    
    # Process each entry
    for key, value in data.items():
        # Check if it's a cam_motion entry
        if key.startswith("cam_motion") and "file_name" in value:
            file_path = value["file_name"]
            if from_cluster in file_path:
                # Update the path
                new_path = file_path.replace(from_cluster, to_cluster)
                value["file_name"] = new_path
                modified = True
                modified_entries.append(key)
    
    # Write back only if modified
    if modified:
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Updated {len(modified_entries)} entries in {json_file}")
        print("Modified entries:")
        for entry in modified_entries:
            print(f"  - {entry}")
    else:
        print(f"No changes needed in {json_file}")

except Exception as e:
    print(f"Error processing {json_file}: {e}")