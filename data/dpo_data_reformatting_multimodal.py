import os
import pandas as pd
import shutil
from pathlib import Path
from PIL import Image
import io
import json

# Load the dataset files:
data_root = 'llava_cot'
files = os.listdir(data_root)
print(f'Files {files}')

df = pd.concat([pd.read_parquet(os.path.join(data_root, parquet_file)) for parquet_file in files])

# Create the folder for images
# output_image_folder = Path("llava_cot")
# output_image_folder.mkdir(parents=True, exist_ok=True)

# Function to save image to folder
def save_image(image_bytes, row_number):
    img = Image.open(io.BytesIO(image_bytes))
    img_path = output_image_folder / f'{row_number}.jpg'
    img.save(img_path, format='JPEG')
    return f"llava_cot/{row_number}.jpg"

# Function to create DPO formatted data
def create_dpo(df):
    dpo_data = []
    
    for row_number, row in df.iterrows():
        # Save the image if it exists
        images = [f'multimodal_cot_images/{row_number}.jpg']
        # if 'image' in row and row['image'] is not None:
        #     image_path = save_image(row['image']['bytes'], row_number)
        #     images.append(image_path)
        
        # Create the conversation format with images
        dpo_entry = {
            "conversations": [
                {   
                    "from": "human",
                    "value": f"<image>{row['question']} Let's think step by step and output the final answer within \\boxed{{}}." if images else f"{row['problem']} Let's think step by step and output the final answer within \\boxed{{}}.",
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": row['ecot']
            },
            "rejected": {
                "from": "gpt",
                "value": row['long_cot']
            },
            "images": images
        }
        dpo_data.append(dpo_entry)
    
    return dpo_data

# Create the DPO formatted data
dpo_data = create_dpo(df)

# Save the DPO formatted data into a JSON file
with open('llava_cot_sum_dpo.json', 'w') as f:
    json.dump(dpo_data, f, indent=4)

print("Images saved and dataset formatted to DPO conversation format.")
