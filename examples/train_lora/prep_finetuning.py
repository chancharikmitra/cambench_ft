# import yaml
# import os

# # Open the base yaml file:
# with open('bal_imb_cap_fps8.yaml', 'r') as reffile:
#     config = yaml.safe_load(reffile)

# # Hyperparameters to tune:
# ranks = [8, 32, 64]
# lrs = [5e-5, 1.0e-4, 2e-4, 1e-3]

# for rank in ranks:
#     for lr in lrs:
#         # Create new config
#         filename = f'bal_imb_cap_rank{rank}_lr{str(lr)}.yaml'
#         config['lora_rank'] = rank
#         config['learning_rate'] = lr

#         # Save config
#         with open(os.path.join('hyperparameter_tuning', filename), 'w') as writefile:
#             yaml.dump(config, writefile, default_flow_style=False)

import yaml
import os
import copy

# Open the base yaml file:
with open('bal_imb_cap_fps8.yaml', 'r') as reffile:
    base_config = yaml.safe_load(reffile)

# Hyperparameters to tune:
ranks = [64]
epoch_nums = [10.0]
lrs = [5e-5, 1.0e-4, 2e-4]
freeze_vision = [True, False]
# Create output directory if it doesn't exist
os.makedirs('hyperparameter_tuning', exist_ok=True)
for vision_setting in freeze_vision:
    for epoch_num in epoch_nums:
        for rank in ranks:
            for lr in lrs:
                # Create a fresh copy of the config for each iteration
                config = copy.deepcopy(base_config)
                
                # Create filename with formatted learning rate to avoid scientific notation
                if lr == 5e-5:
                    lr_str = "5e-5"
                elif lr == 1.0e-4:
                    lr_str = "1e-4"
                elif lr == 2e-4:
                    lr_str = "2e-4"
                
                filename = f'bal_imb_cap_rank{rank}_lr{lr_str}_epoch{epoch_num}_freezevis{vision_setting}_fps8.yaml'
                output_dir =f'saves/qwen2.5_vl-7b/lora/bal_imb_cap_rank{rank}_lr{lr_str}_epoch{epoch_num}_freezevis{vision_setting}_fps8'
                # Update config with current hyperparameters
                config['lora_rank'] = rank
                config['learning_rate'] = lr
                config['output_dir'] = output_dir
                config['num_train_epochs'] = epoch_num

                if vision_setting:
                    config['freeze_vision_tower'] = True
                    config['freeze_multi_modal_projector'] = True
                else:
                    config['freeze_vision_tower'] = False
                    config['freeze_multi_modal_projector'] = False

                # Save config
                filepath = os.path.join('hyperparameter_tuning', filename)
                with open(filepath, 'w') as writefile:
                    yaml.dump(config, writefile, default_flow_style=False)
                
                print(f"Created {filepath} with rank={rank}, lr={lr}, epoch={epoch_num}")