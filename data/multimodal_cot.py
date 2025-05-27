# import os
# import pandas as pd
# import shutil
# from pathlib import Path
# from PIL import Image
# import io
# import json

# # Load the dataset
# data_root = 'llava_cot'
# files = os.listdir(data_root)
# print(f'Files {files}')

# df = pd.concat([pd.read_parquet(os.path.join(data_root, parquet_file)) for parquet_file in files])

# # Create the folder for images
# output_image_folder = Path("multimodal_cot_images")
# output_image_folder.mkdir(parents=True, exist_ok=True)

# # Function to save image to folder
# def save_image(image_bytes, row_number):
#     img = Image.open(io.BytesIO(image_bytes))
#     img_path = output_image_folder / f'{row_number}.png'
#     img.save(img_path)

# # Function to create Alpaca formatted data
# def create_alpaca_format(df):
#     alpaca_data = []
    
#     for row_number, row in df.iterrows():
#         # Save the image using the row number as the filename
#         # save_image(row['image']['bytes'], row_number)
        
#         # Construct the instruction and output
#         instruction = f"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. In particular, the reasoning length should depend on the difficulty (i.e. reason less for easier questions and more for harder questions). The reasoning process and answer are enclosed within   <think> and   <answer> tags, respectively, i.e., <think>reasoning process here</think>   <answer>answer here</answer>."
        
#         # Prepare the data in Alpaca format
#         alpaca_entry = {
#             "instruction": instruction,
#             "input": row['question'] + "Let's think step by step and output the final answer within \\boxed{}." ,
#             "output": f"{row['ecot']} <answer>\\boxed{{{row['ground_truth']}}}</answer>"
#         }
#         alpaca_data.append(alpaca_entry)
    
#     return alpaca_data

# # Create the Alpaca formatted data
# alpaca_data = create_alpaca_format(df)

# # Save the Alpaca formatted data into a JSON file
# with open('alpaca_dataset.json', 'w') as f:
#     json.dump(alpaca_data, f, indent=4)

# print("Images saved and dataset formatted to Alpaca format.")
import os
import pandas as pd
import shutil
from pathlib import Path
from PIL import Image
import io
import json
from tqdm import tqdm

# Load the dataset files:
data_root = 'llava_cot'
files = os.listdir(data_root)
print(f'Files {files}')

# Load parquet files with progress bar
print("Loading parquet files...")
dfs = []
for parquet_file in tqdm(files, desc="Loading files"):
    dfs.append(pd.read_parquet(os.path.join(data_root, parquet_file)))

df = pd.concat(dfs, ignore_index=True)  # Reset index to create continuous numbering
print(f"Total rows loaded: {len(df)}")

# Create the folder for images
output_image_folder = Path("multimodal_cot_images")
output_image_folder.mkdir(parents=True, exist_ok=True)

# Function to save image to folder
def save_image(image_bytes, row_number):
    img = Image.open(io.BytesIO(image_bytes))
    img_path = output_image_folder / f'{row_number}.jpg'
    img.save(img_path, format='JPEG')
    return f"multimodal_cot_images/{row_number}.jpg"

# Function to create conversation formatted data
def create_conversation_format(df):
    conversation_data = []
    
    # Reset dataframe index to ensure continuous numbering
    df_reset = df.reset_index(drop=True)
    
    # Iterate through rows with progress bar
    for row_number, row in tqdm(df_reset.iterrows(), total=len(df_reset), desc="Processing rows"):
        # Save the image using the row number as the filename
        image_path = save_image(row['image']['bytes'], row_number)
        
        # Create the conversation format
        conversation_entry = {
            "messages": [
                {
                    "content": f"<image>{row['question']} Let's think step by step and output the final answer within \\boxed{{}}.",
                    "role": "user"
                },
                {
                    "content": row['ecot'],
                    "role": "assistant"
                }
            ],
            "images": [
                image_path
            ]
        }
        conversation_data.append(conversation_entry)
    
    print(f'Dataset Length {len(conversation_data)}')
    return conversation_data

# Create the conversation formatted data
conversation_data = create_conversation_format(df)

# Save the conversation formatted data into a JSON file
print("Saving JSON file...")
with open('multimodal_sum_sft.json', 'w') as f:
    json.dump(conversation_data, f, indent=4)

print("Images saved and dataset formatted to conversation format.")