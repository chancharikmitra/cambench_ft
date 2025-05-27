import json
import time
from openai import OpenAI

def add_boxed_answer_and_update_input(api_key, data_file, output_file):
    """
    Process the file to:
    1. Add the prompt about boxed answers to each input
    2. Use GPT-4o to add boxed answers to each output
    """
    # Configure OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Load the data
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)[:500]
    
    # Define the prompt text
    prompt_text = """For the input I provide, I want you to change nothing about it except adding the final answer in a Latex box (i.e. \\boxed{}) right before the the last answer token "</answer>". For example, this response: "<think>\n1. Identify the arithmetic sequence: 991, 993, 995, 997, 999.\n2. Calculate the sum of the sequence: 4975.\n3. Compute 5000 - 4975 to find N.\n4. Conclude N = 25.\n</think>\n<answer>25</answer>" Should become "<think>\n1. Identify the arithmetic sequence: 991, 993, 995, 997, 999.\n2. Calculate the sum of the sequence: 4975.\n3. Compute 5000 - 4975 to find N.\n4. Conclude N = 25.\n</think>\n<answer>\\boxed{25}</answer>" Do this for the following response: """
    
    # Process each item
    for i, item in enumerate(data):
        print(f"Processing item {i+1}/{len(data)}...")
        
        # Update the input field by adding the prompt at the end
        item["input"] = item["input"].rstrip() + " Let's think step by step and output the final answer within \\boxed{}."
        
        # Create the full prompt
        full_prompt = prompt_text + item["output"]
        
        try:
            # Call GPT-4o API using the Responses API
            response = client.responses.create(
                model="gpt-4o",
                # system="You are a helpful assistant that adds LaTeX boxed expressions to mathematical answers.",
                input=full_prompt,
                # temperature=0.0,
                # max_tokens=2048
            )
            
            # Update the item with the new output
            item["output"] = response.output_text
            
            # Add a small delay to avoid rate limits
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error processing item {i+1}: {e}")
            print(f"Full error: {str(e)}")
    
    # Save the modified data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Processed {len(data)} items and saved results to {output_file}")

# Example usage
if __name__ == "__main__":
    api_key = "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"  # Replace with your OpenAI API key
    input_file = "alpaca_formatted_data_final_summarized.json"
    output_file = "alpaca_formatted_data_final_summarized_bbox.json"
    
    add_boxed_answer_and_update_input(api_key, input_file, output_file)