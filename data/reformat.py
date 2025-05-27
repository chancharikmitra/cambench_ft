# #!/usr/bin/env python3

# import json
# import os
# import glob

# # Configuration
# json_dir = "cam_motion_FLAME"
# old_path = "/data3/zhiqiul/video_annotation/videos/"
# new_path = "/mnt/localssd/video_annotation/videos/"

# # Count files for progress tracking
# json_files = glob.glob(os.path.join(json_dir, "**/*.json"), recursive=True)
# total_files = len(json_files)
# processed = 0

# print(f"Found {total_files} JSON files to process.")
# print(f"Replacing video paths from {old_path} to {new_path}")

# # Process each JSON file
# for json_file in json_files:
#     try:
#         # Read the JSON file
#         with open(json_file, 'r') as f:
#             data = json.load(f)
        
#         # Check if it's a list of entries
#         if isinstance(data, list):
#             modified = False
            
#             # Process each entry in the list
#             for entry in data:
#                 if "videos" in entry and isinstance(entry["videos"], list):
#                     # Update video paths
#                     for i, video_path in enumerate(entry["videos"]):
#                         if video_path.startswith(old_path):
#                             entry["videos"][i] = video_path.replace(old_path, new_path)
#                             modified = True
            
#             # Write back to file if modified
#             if modified:
#                 with open(json_file, 'w') as f:
#                     json.dump(data, f, indent=2)
        
#         # Update progress counter
#         processed += 1
#         if processed % 100 == 0 or processed == total_files:
#             print(f"Processed {processed} of {total_files} files.")
    
#     except Exception as e:
#         print(f"Error processing {json_file}: {e}")

# print("Path replacement complete. All video paths updated.")

#!/usr/bin/env python3
import json
import os
import shutil
from pathlib import Path

# Define source and destination directories
SOURCE_DIR = "cam_motion_ORCHARD"
DEST_DIR = "cam_motion_LAMBDA"

# Create destination directory if it doesn't exist
os.makedirs(DEST_DIR, exist_ok=True)

# Get all JSON files in the source directory
json_files = list(Path(SOURCE_DIR).glob("*.json"))

# Process each JSON file
for json_file in json_files:
    # Get the filename
    filename = json_file.name
    
    # Define output path
    output_path = Path(DEST_DIR) / filename
    
    # Read the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # For each item in the JSON array
    for item in data:
        # Check if there are videos in the item
        if "videos" in item:
            # Replace the path in each video
            for i, video_path in enumerate(item["videos"]):
                if "/mnt/localssd" in video_path:
                    item["videos"][i] = video_path.replace("/mnt/localssd", "/home/ubuntu/lambdadata")
    
    # Write the modified data to the output file
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Processed: {filename}")

print(f"All files have been processed and saved to {DEST_DIR}/")