#!/bin/bash
# worker_7b.sh - Worker script that runs on each node for 7B model

set -euxo pipefail

echo "=== 7B Worker Script Starting on $(hostname) - Node $SLURM_NODEID ==="
echo "Timestamp: $(date)"

# Force flush output after each line
export PYTHONUNBUFFERED=1

# Configuration
TYPE="train_full"
MODEL="qwen2.5-vl-7b-cambench"
NAME="cambench_1_3_6"
EXP_PATH=${TYPE}/${MODEL}/${NAME}.yaml
SAVE_PATH="saves_7b_1_3_6_multinode"
HF_REPO="chancharikm/saves_7b_1_3_6"

# Paths
GCS_BUCKET="gs://cmu-gpucloud-cmitra"
LOCAL_PATH="/tmp"
CONDA_ENV="lfact"

# Set HuggingFace token
export HF_TOKEN="YOUR_HF_TOKEN"

# Set up cache directories
mkdir -p $LOCAL_PATH/cmitra_cache
export HF_HOME="$LOCAL_PATH/cache"

echo "Node $SLURM_NODEID: Basic system info..."
nvidia-smi
pwd
ls
free -g

echo "Node $SLURM_NODEID: Activating conda environment ${CONDA_ENV}..."
source /home/cmitra/anaconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
echo "Node $SLURM_NODEID: Active conda environment: $(conda info --envs | grep '*' || echo 'None')"

# Set GPU environment variables
echo "Node $SLURM_NODEID: Setting GPU environment variables..."
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "Node $SLURM_NODEID: Available GPUs:"
nvidia-smi

# Set up distributed training environment variables
echo "Node $SLURM_NODEID: Setting up distributed training environment..."
export FORCE_TORCHRUN=1
export NNODES=$SLURM_NNODES
export NODE_RANK=$SLURM_NODEID
export NPROC_PER_NODE=$SLURM_GPUS_ON_NODE

echo "Node $SLURM_NODEID: Distributed training setup:"
echo "  RDZV_BACKEND: $RDZV_BACKEND"
echo "  RDZV_ENDPOINT: $RDZV_ENDPOINT"
echo "  FORCE_TORCHRUN: $FORCE_TORCHRUN"
echo "  NNODES: $NNODES"
echo "  NODE_RANK: $NODE_RANK"
echo "  NPROC_PER_NODE: $NPROC_PER_NODE"

# Download and setup data
echo "Node $SLURM_NODEID: Setting up data..."
cd $LOCAL_PATH

if [ ! -d "video_annotation" ]; then
    echo "Node $SLURM_NODEID: Data not found, downloading..."
    gcloud storage cp "${GCS_BUCKET}/video_annotation.tar.gz" ${LOCAL_PATH}/
    tar -xzf video_annotation.tar.gz
    mv data3/zhiqiul/video_annotation video_annotation
    rm video_annotation.tar.gz
    echo "Node $SLURM_NODEID: Data download and extraction completed"
else
    echo "Node $SLURM_NODEID: Data already exists, skipping download"
fi

# Create save folder for weights
mkdir -p ${LOCAL_PATH}/${SAVE_PATH}

# Change to LLaMA-Factory directory
cd ~/LLaMA-Factory
echo "Node $SLURM_NODEID: Now in directory: $(pwd)"

# Check for existing checkpoints and download if they exist
echo "Node $SLURM_NODEID: Checking for existing checkpoints..."
RESUME_FROM_CHECKPOINT="false"

if huggingface-cli repo info "$HF_REPO" >/dev/null 2>&1; then
    echo "Node $SLURM_NODEID: Remote HF repository exists. Downloading for checkpoint resume..."
    
    huggingface-cli download "$HF_REPO" --local-dir "${LOCAL_PATH}/${SAVE_PATH}" --local-dir-use-symlinks False
    
    if [ "$(find ${LOCAL_PATH}/${SAVE_PATH} -name 'checkpoint-*' -type d 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "Node $SLURM_NODEID: Found checkpoints in HF repo, will resume training"
        RESUME_FROM_CHECKPOINT="true"
    else
        echo "Node $SLURM_NODEID: No checkpoints found in HF repo, starting fresh training"
    fi
else
    echo "Node $SLURM_NODEID: Remote HF repository does not exist yet, starting fresh training"
fi

echo "Node $SLURM_NODEID: Resume from checkpoint: $RESUME_FROM_CHECKPOINT"

# Create dynamic YAML config
echo "Node $SLURM_NODEID: Creating dynamic YAML configuration..."
DYNAMIC_YAML="examples/${TYPE}/${MODEL}/dynamic_${NAME}_node${SLURM_NODEID}.yaml"

cp examples/${EXP_PATH} $DYNAMIC_YAML

# Use Python to modify the YAML
python << EOF
import yaml

config_path = "$DYNAMIC_YAML"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Update paths
config['output_dir'] = '${LOCAL_PATH}/${SAVE_PATH}'
config['hub_model_id'] = '${HF_REPO}'

# Set resume from checkpoint if needed
if "$RESUME_FROM_CHECKPOINT" == "true":
    config['resume_from_checkpoint'] = True
    print("Node $SLURM_NODEID: Set resume_from_checkpoint to: True")
else:
    config.pop('resume_from_checkpoint', None)
    print("Node $SLURM_NODEID: Starting fresh training (no resume_from_checkpoint)")

# Write back the config
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print(f"Node $SLURM_NODEID: Dynamic YAML created at: {config_path}")
EOF

# Set LLaMA-Factory specific environment variables
echo "Node $SLURM_NODEID: Setting LLaMA-Factory specific environment variables..."
export DISABLE_VERSION_CHECK=1
export FORCE_QWENVL_VIDEO_READER=torchvision

echo "Node $SLURM_NODEID: Environment variables set:"
echo "  DISABLE_VERSION_CHECK=$DISABLE_VERSION_CHECK"
echo "  FORCE_QWENVL_VIDEO_READER=$FORCE_QWENVL_VIDEO_READER"

# Run the finetuning
echo "Node $SLURM_NODEID: Starting 7B training with config: $DYNAMIC_YAML"
if [ "$RESUME_FROM_CHECKPOINT" = "true" ]; then
    echo "Node $SLURM_NODEID: RESUMING from existing checkpoint"
else
    echo "Node $SLURM_NODEID: STARTING fresh training"
fi


# GPU Memory Monitoring
echo "Node $SLURM_NODEID: Starting GPU memory monitoring..."
(
    while true; do
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU Memory Usage:" >> ${LOCAL_PATH}/gpu_memory_node_${SLURM_NODEID}.log
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >> ${LOCAL_PATH}/gpu_memory_node_${SLURM_NODEID}.log
        echo "" >> ${LOCAL_PATH}/gpu_memory_node_${SLURM_NODEID}.log
        sleep 60  # Check every minute
    done
) &
GPU_MONITOR_PID=$!
##

llamafactory-cli train $DYNAMIC_YAML 2>&1 | tee -a ${LOCAL_PATH}/training_node_${SLURM_NODEID}.log

# Stop monitoring when training finishes
kill $GPU_MONITOR_PID 2>/dev/null || true
echo "Node $SLURM_NODEID: GPU monitoring stopped"

# Clean up the dynamic YAML file after training
echo "Node $SLURM_NODEID: Cleaning up dynamic YAML file..."
rm -f $DYNAMIC_YAML

# Copy results to cloud storage (only from master node)
if [ "$SLURM_NODEID" -eq 0 ]; then
    echo "Master node: Copying updated weights to cloud storage..."
    cd ${LOCAL_PATH}
    tar -czvf ${SAVE_PATH}.tar.gz ${SAVE_PATH}
    gcloud storage cp ${SAVE_PATH}.tar.gz ${GCS_BUCKET}/${SAVE_PATH}.tar.gz
    
    echo "Master node: Copying training logs..."
    tar -czvf training_logs_${SLURM_JOB_ID}.tar.gz training_node_*.log
    gcloud storage cp training_logs_${SLURM_JOB_ID}.tar.gz ${GCS_BUCKET}/
    
    echo "Master node: All files uploaded to cloud storage successfully"
fi

echo "=== 7B Worker Script Completed on $(hostname) - Node $SLURM_NODEID at $(date) ==="