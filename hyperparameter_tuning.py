# #!/usr/bin/env python3
# import os
# import subprocess
# import glob
# from datetime import datetime

# import gc
# import time
# import torch

# # Directory containing YAML configs
# config_dir = "examples/train_lora/hyperparameter_tuning"

# # Find all config files
# config_files = sorted(glob.glob(os.path.join(config_dir, "*.yaml")))

# print(f"Found {len(config_files)} config files to process")

# # Process all configs sequentially
# for i, config_path in enumerate(config_files):
#     print(f"\nRunning config {i+1}/{len(config_files)}: {config_path}")
    
#     # Create log filename based on config name and timestamp
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     log_filename = f"log_{os.path.basename(config_path)}_{timestamp}.txt"
    
#     # Run training command
#     command = ["llamafactory-cli", "train", config_path]
    
#     # Execute command and log output
#     with open(log_filename, "w") as log_file:
#         process = subprocess.run(
#             command,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.STDOUT,
#             text=True,
#             universal_newlines=True
#         )
        
#         # Print output to console and write to log file
#         print(process.stdout)
#         log_file.write(process.stdout)
        

#         # Garbage collection:
#         process = None
#         gc.collect()
#         # GPU memory cleanup (with safety check)
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#             torch.cuda.reset_peak_memory_stats()
#         time.sleep(5)

    
#     print(f"Finished config {config_path}. Log saved to {log_filename}")

#!/usr/bin/env python3
import os
import subprocess
import glob
from datetime import datetime
import sys
import gc
import time
import torch

# Directory containing YAML configs
config_dir = "examples/train_lora/hyperparameter_tuning"

# Find all config files
config_files = sorted(glob.glob(os.path.join(config_dir, "*.yaml")))

print(f"Found {len(config_files)} config files to process")

# Process all configs sequentially
for i, config_path in enumerate(config_files):
    print(f"\nRunning config {i+1}/{len(config_files)}: {config_path}")
    
    # Create log filename based on config name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"log_{os.path.basename(config_path)}_{timestamp}.txt"
    
    # Run training command
    command = ["llamafactory-cli", "train", config_path]
    
    # Create log file and set up live streaming of output
    with open(log_filename, "w", buffering=1) as log_file:
        # Use Popen instead of run to get real-time output
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output to both console and log file in real-time
        for line in process.stdout:
            print(line, end='')  # Print to console
            log_file.write(line)  # Write to log file
            sys.stdout.flush()    # Force flush to ensure real-time output
        
        # Wait for process to complete
        process.wait()
        
        # Garbage collection:
        process = None
        gc.collect()
        # GPU memory cleanup (with safety check)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        time.sleep(5)
    
    print(f"Finished config {config_path}. Log saved to {log_filename}")