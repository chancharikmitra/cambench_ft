import os
import pandas as pd
import shutil
from pathlib import Path
from PIL import Image
import io
import json

# Load the dataset files:
data_root = 'openmathreasoning-125k'
files = os.listdir(data_root)
print(f'Files {files}')

df = pd.concat([pd.read_parquet(os.path.join(data_root, parquet_file)) for parquet_file in files])

# print(df.head())
# print(df.shape)
# exit()
# Create the folder for images
# output_image_folder = Path("multimodal_cot_images")
# output_image_folder.mkdir(parents=True, exist_ok=True)

# # Function to save image to folder
# def save_image(image_bytes, row_number):
#     img = Image.open(io.BytesIO(image_bytes))
#     img_path = output_image_folder / f'{row_number}.png'
#     img.save(img_path)

# Function to create Alpaca formatted data
def create_alpaca_format(df):
    alpaca_data = []
    
    for row_number, row in df.iterrows():
        # Save the image using the row number as the filename
        # save_image(row['image']['bytes'], row_number)
        
        # Construct the instruction and output
        # system_instruction = "detailed thinking on"
        
        # Prepare the data in Alpaca format
        alpaca_entry = {
            # "system": system_instruction,
            "instruction": row['problem'] + " Let's think step by step and output the final answer within \boxed{}." ,
            "input": "",
            "output": row['ecot']
        }
        alpaca_data.append(alpaca_entry)
    
    return alpaca_data

# Create the Alpaca formatted data
alpaca_data = create_alpaca_format(df)

# Save the Alpaca formatted data into a JSON file
with open('nemotron_sum_sft.json', 'w') as f:
    json.dump(alpaca_data, f, indent=4)

print("Images saved and dataset formatted to Alpaca format.")