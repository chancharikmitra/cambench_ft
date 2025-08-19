# Multinode Training Setup

This guide explains how to run distributed training across multiple nodes using SLURM on the FLAME cluster.

## Prerequisites

### 1. Environment Setup

Follow all setup instructions for LLaMA-Factory from the main README, then install additional dependencies:

```bash
# Install LLaMA-Factory with required extras
pip install -e ".[torch,metrics]"

# Install specific transformers version
pip install git+https://github.com/huggingface/transformers.git@refs/pull/36188/head

# Install distributed training dependencies
pip install deepspeed --use-pep517
pip install flash-attn --no-build-isolation --use-pep517

# Install additional utilities
pip install hf_xet

# Disable version checking
export DISABLE_VERSION_CHECK=1
```

**Note:** You may need to set your `HF_TOKEN` environment variable if you encounter processor-related errors.

### 2. Set up GCS Data Storage

The directions to set up GCS data storage can be found here:

https://drive.google.com/file/d/1VpfI0UhSaoCZURJynpxC13SW1C2lgm2t/view

### 3. Configure Multinode Script

Before running, update the `cambench_7b_multinode.sh` script with your HuggingFace token:

```bash
export HF_TOKEN="your_token_here"
```

And your GCS bucket and save path:

```bash
SAVE_PATH="saves_7b_1_3_6_multinode"
HF_REPO="your_token_here"

# Paths
GCS_BUCKET="gs://cmu-gpucloud-cmitra"
```

Your downloaded folders and paths may be different, so remember to change or double check this part in the script:

```bash
if [ ! -d "video_annotation" ]; then
    echo "Node $SLURM_NODEID: Data not found, downloading..."
    gcloud storage cp "${GCS_BUCKET}/video_annotation.tar.gz" ${LOCAL_PATH}/
    tar -xzf video_annotation.tar.gz
    rm video_annotation.tar.gz
    echo "Node $SLURM_NODEID: Data download and extraction completed"
else
    echo "Node $SLURM_NODEID: Data already exists, skipping download"
fi
```

## SLURM Configuration

The `launcher.sh` script contains the following key SLURM parameters:

| Parameter | Value | Description |
|-----------|--------|-------------|
| `--job-name` | `7b-multinode` | Job identifier |
| `--partition` | `preempt` | Partition to use (see limits below) |
| `--gres` | `gpu:8` | 8 GPUs per node |
| `--cpus-per-gpu` | `8` | CPU cores per GPU |
| `--nodes` | `4` | Number of compute nodes |
| `--ntasks-per-node` | `1` | One task per node |
| `--mem` | `1600G` | Memory per node |
| `--time` | `48:00:00` | Maximum runtime (48 hours) |

## FLAME Cluster Important Notes

### Storage & Execution
- **No shared storage** between nodes - each node has its own local storage
- Commands only run on the head node unless executed with `srun`
- Use `srun` to execute scripts across all allocated nodes simultaneously
- **Data is stored in the `/tmp` folder of each node and is cleaned up after each run on that node** - this is especially important to consider when a job is restarted from requeueing or preempted (but this is handled in the current script - always redownloads data on each node).

### Partition Limits
| Partition | Max Time | Max Nodes | Priority | Preemption Risk |
|-----------|----------|-----------|----------|-----------------|
| `preempt` | 48 hours | 4 nodes | Standard | Higher (more likely to be requeued) |
| `flame` | 28 days | 3 nodes | High | Lower (less likely to be requeued) |

## Running Training

Submit the job to SLURM:

```bash
sbatch launcher_7b.sh
```

### Monitoring

Check job status:
```bash
squeue -u [USER]
```

View job output:
```bash
tail -f slurm-7b-multinode-<job_id>.out
```

**Note:** All other standard SLURM commands apply (e.g., `scancel`, `sinfo`, `sacct`, etc.).

## Distributed Training Environment

The launcher script automatically sets up distributed training variables:
- `RDZV_BACKEND=c10d` - PyTorch distributed backend
- `RDZV_ENDPOINT` - Rendezvous endpoint using the first allocated node
- Total GPUs = `SLURM_NNODES × SLURM_GPUS_ON_NODE`

The training will be distributed across all allocated GPUs using the configuration specified in your training script.