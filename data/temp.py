import json
import os

def convert_to_sharegpt4v_format(input_file, output_file):
    """
    Convert from one format to the ShareGPT-4V message format with images.
    
    Args:
        input_file: Path to the input JSON file
        output_file: Path to save the converted format JSON
    """
    # Load the input dataset
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    # Initialize the output dataset
    output_data = []
    
    # Convert each entry
    for entry in input_data:
        # Extract the relevant information
        image_path = entry["image"]
        
        # Get the human question
        human_message = entry["conversations"][0]["value"]
        
        # Get the GPT response
        gpt_response = entry["conversations"][1]["value"]
        
        # Create the new format entry
        new_entry = {
            "messages": [
                {
                    "role": "user",
                    "content": f"<image>{human_message}"
                },
                {
                    "role": "assistant",
                    "content": gpt_response
                }
            ],
            "images": [
                image_path
            ]
        }
        
        # Add to the output dataset
        output_data.append(new_entry)
    
    # Save the converted dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Converted {len(input_data)} entries to the target format")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    # Set your file paths here
    input_file = "multimodal_sum_final.json"
    output_file = "multimodal_sum_final.json"
    
    # Run the conversion
    convert_to_sharegpt4v_format(input_file, output_file)