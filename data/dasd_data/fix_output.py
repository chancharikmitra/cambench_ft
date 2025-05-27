import json

def replace_chosen_outputs(alpaca_file, dpo_file, output_file):
    """
    Replace the 'chosen' outputs in a DPO dataset with the 'output' from an Alpaca dataset.
    
    Args:
        alpaca_file: Path to the Alpaca format JSON file
        dpo_file: Path to the DPO format JSON file
        output_file: Path to save the modified DPO JSON
    """
    # Load the Alpaca dataset
    with open(alpaca_file, 'r', encoding='utf-8') as f:
        alpaca_data = json.load(f)
    
    # Load the DPO dataset
    with open(dpo_file, 'r', encoding='utf-8') as f:
        dpo_data = json.load(f)
    
    # Check if datasets have the same number of samples
    if len(alpaca_data) != len(dpo_data):
        print(f"Warning: Alpaca dataset has {len(alpaca_data)} samples, but DPO dataset has {len(dpo_data)} samples.")
        print("Will use the minimum number of samples.")
    
    # Replace 'chosen' outputs with Alpaca 'output'
    num_samples = min(len(alpaca_data), len(dpo_data))
    
    for i in range(num_samples):
        # Get the Alpaca output
        alpaca_output = alpaca_data[i]["output"]
        
        # Replace the DPO chosen output
        dpo_data[i]["chosen"]["value"] = alpaca_output
    
    # Save the modified DPO dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dpo_data, f, indent=2, ensure_ascii=False)
    
    print(f"Replaced 'chosen' outputs for {num_samples} samples.")
    print(f"Saved modified DPO dataset to {output_file}")

# Example usage
if __name__ == "__main__":
    alpaca_file = "alpaca_formatted_data_final_summarized_bbox.json"  # Your Alpaca format file
    dpo_file = "dpo_dataset_qd.json"        # Your DPO format file
    output_file = "dpo_dataset_qd.json"
    
    replace_chosen_outputs(alpaca_file, dpo_file, output_file)