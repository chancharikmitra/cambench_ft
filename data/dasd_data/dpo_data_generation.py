# import json
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from tqdm import tqdm

# def generate_dpo_datasets(input_file, output_files):
#     """
#     Generate DPO datasets from multiple models
    
#     Args:
#         input_file: Path to the input JSON file in alpaca format
#         output_files: List of output file paths for each model
#     """
#     # Load the data
#     with open(input_file, 'r', encoding='utf-8') as f:
#         data = json.load(f)
    
#     # Check if it's a list or single item
#     if not isinstance(data, list):
#         data = [data]
    
#     # Define models to use
#     models = [
#         "agentica-org/DeepScaleR-1.5B-Preview",
#         "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
#         "chancharikm/dasd_qwen_distill_1.5_lora_sum_3poch",
#         "chancharikm/dasd_deepscaler_1.5_lora_sum_3poch"
#     ]
    
#     # Process each model
#     for i, model_name in enumerate(models):
#         print(f"Processing model: {model_name}")
        
#         try:
#             # Load model and tokenizer
#             tokenizer = AutoTokenizer.from_pretrained(model_name)
#             model = AutoModelForCausalLM.from_pretrained(
#                 model_name, 
#                 torch_dtype=torch.bfloat16,
#                 device_map="auto",
#                 attn_implementation="flash_attention_2"
#             )
            
#             # Create DPO dataset for this model
#             dpo_data = []
            
#             for item in tqdm(data, desc=f"Generating with {model_name}"):
#                 # Extract the prompt
#                 prompt = item["input"] + "Let's think step by step and output the final answer within \\boxed{}."
#                 reference_output = item["output"]
                
#                 # Generate text
#                 inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#                 outputs = model.generate(
#                     **inputs,
#                     max_new_tokens=4096,
#                     temperature=0.7,
#                     do_sample=True,
#                     top_p=0.9,
#                 )
#                 generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
#                 # Create DPO example
#                 dpo_example = {
#                     "conversations": [
#                         {
#                             "from": "human",
#                             "value": prompt
#                         }
#                     ],
#                     "chosen": {
#                         "from": "gpt",
#                         "value": reference_output
#                     },
#                     "rejected": {
#                         "from": "gpt",
#                         "value": generated_text
#                     }
#                 }
                
#                 dpo_data.append(dpo_example)
            
#             # Save the dataset
#             output_file = output_files[i]
#             with open(output_file, 'w', encoding='utf-8') as f:
#                 json.dump(dpo_data, indent=2, ensure_ascii=False)
            
#             print(f"Saved DPO dataset to {output_file}")
            
#             # Free up memory
#             del model
#             del tokenizer
#             torch.cuda.empty_cache()
            
#         except Exception as e:
#             print(f"Error processing model {model_name}: {e}")
    
#     print("All datasets generated successfully!")

# # Example usage
# if __name__ == "__main__":
#     input_file = "../data/language_only/train_final/alpaca_formatted_data_final_summarized.json"  # Your input file with alpaca format
    
#     # Define output files for each model
#     output_files = [
#         "dpo_dataset_DeepScaleR.json",
#         "dpo_dataset_DeepSeek.json",
#         "dpo_dataset_dasd_qwen.json",
#         "dpo_dataset_dasd_deepscaler.json"
#     ]
    
#     generate_dpo_datasets(input_file, output_files)

import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import concurrent.futures
import numpy as np
from torch.multiprocessing import Pool, set_start_method

# Try to set start method to 'spawn' for better process handling
try:
    set_start_method('spawn')
except RuntimeError:
    pass

def process_data_chunk(args):
    """Process a chunk of data on a specific GPU"""
    model_name, data_chunk, gpu_id = args
    
    # Set the visible GPU for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    try:
        # Load model and tokenizer with flash attention
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        
        # Process data chunk
        results = []
        
        for item in tqdm(data_chunk, desc=f"GPU {gpu_id} processing chunk"):
            # Extract the prompt
            prompt = item["input"] + "Let's think step by step and output the final answer within \\boxed{}."
            reference_output = item["output"]
            
            # Generate text with flash attention
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                use_cache=True
            )
            # generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
            generated_text_clipped = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:-1], skip_special_tokens=False)
            # print(f'Generated Text {generated_text}\n\n\n')
            # print(f'Generated Text Cliped {generated_text_clipped}')
            # Create DPO example
            dpo_example = {
                "conversations": [
                    {
                        "from": "human",
                        "value": prompt
                    }
                ],
                "chosen": {
                    "from": "gpt",
                    "value": reference_output
                },
                "rejected": {
                    "from": "gpt",
                    "value": generated_text_clipped
                }
            }
            
            results.append(dpo_example)
        
        # Free up memory
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
        return results
        
    except Exception as e:
        print(f"Error processing on GPU {gpu_id}: {e}")
        return []

def process_model_with_data_parallelism(model_name, data, output_file, num_gpus):
    """Process a single model by splitting data across multiple GPUs"""
    print(f"Processing model: {model_name} with data parallelism across {num_gpus} GPUs")
    
    # Split data into chunks for each GPU
    chunk_size = len(data) // num_gpus
    if chunk_size == 0:
        chunk_size = 1
    
    data_chunks = []
    for i in range(0, len(data), chunk_size):
        end = min(i + chunk_size, len(data))
        data_chunks.append(data[i:end])
    
    # Pad with empty chunks if needed
    while len(data_chunks) < num_gpus:
        data_chunks.append([])
    
    # If we have more chunks than GPUs, combine the extra chunks
    if len(data_chunks) > num_gpus:
        data_chunks[num_gpus-1].extend([item for chunk in data_chunks[num_gpus:] for item in chunk])
        data_chunks = data_chunks[:num_gpus]
    
    # Create arguments for each GPU worker
    args_list = [(model_name, chunk, gpu_id) for gpu_id, chunk in enumerate(data_chunks) if len(chunk) > 0]
    
    # Process data in parallel across GPUs
    all_results = []
    with Pool(processes=len(args_list)) as pool:
        chunk_results = pool.map(process_data_chunk, args_list)
        
        # Combine results from all chunks
        for result in chunk_results:
            all_results.extend(result)
    
    # Save the combined results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"Saved DPO dataset to {output_file} with {len(all_results)} examples")
    return len(all_results) > 0

def generate_dpo_datasets_with_data_parallelism(input_file, output_files, num_gpus=8):
    """
    Generate DPO datasets by processing each model sequentially,
    but with data parallelism across GPUs for each model
    
    Args:
        input_file: Path to the input JSON file in alpaca format
        output_files: List of output file paths for each model
        num_gpus: Number of GPUs to use for data parallelism
    """
    # Load the data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)[:500]
    
    # Check if it's a list or single item
    if not isinstance(data, list):
        data = [data]
    
    # Define models to use
    models = [
        
        "chancharikm/dasd_deepscaler_1.5_lora_sum_3poch",
        "chancharikm/dasd_qwen_distill_1.5_lora_sum_3poch",
        "agentica-org/DeepScaleR-1.5B-Preview",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        
    ]
    
    # Process each model sequentially, but with data parallelism
    success = True
    for i, model_name in enumerate(models):
        model_success = process_model_with_data_parallelism(
            model_name, 
            data, 
            output_files[i], 
            num_gpus
        )
        if not model_success:
            print(f"Failed to process model: {model_name}")
            success = False
    
    if success:
        print("All datasets generated successfully!")
    else:
        print("Some datasets failed to generate. Check logs for details.")

# Example usage
if __name__ == "__main__":
    input_file = "../data/language_only/train_final/alpaca_formatted_data_final_summarized.json"
    
    # Define output files for each model
    output_files = [
        "dpo_dataset_DeepScaleR.json",
        "dpo_dataset_DeepSeek.json",
        "dpo_dataset_dasd_qwen.json",
        "dpo_dataset_dasd_deepscaler.json"
    ]
    
    # Run with data parallelism across all available GPUs
    generate_dpo_datasets_with_data_parallelism(input_file, output_files, num_gpus=1)