import json

def process_alpaca_data(input_file, output_file):
    """
    Process an Alpaca format JSON file to empty the instruction field.
    
    Args:
        input_file (str): Path to the input JSON file in Alpaca format
        output_file (str): Path to save the processed dataset
    """
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check if it's a list or single item
    if not isinstance(data, list):
        data = [data]
    
    processed_data = []
    
    for item in data:
        # Create new item with empty instruction field
        processed_item = {
            "instruction": "",
            "input": item["input"],
            "output": item["output"]
        }
        
        processed_data.append(processed_item)
    
    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    print(f"Processed {len(processed_data)} items and saved to {output_file}")

# Example usage
if __name__ == "__main__":
    input_file = "alpaca_formatted_data_final_summarized.json"
    output_file = "alpaca_formatted_data_final_summarized_nosys.json"
    process_alpaca_data(input_file, output_file)