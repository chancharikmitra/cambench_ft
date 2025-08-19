#!/bin/bash
#SBATCH --job-name=7b-multinode
#SBATCH -p preempt
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8 
#SBATCH --qos=preempt_qos
#SBATCH --account=deva
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1600G   
#SBATCH --time=48:00:00 
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Set up distributed training environment variables
export RDZV_BACKEND=c10d
export RDZV_ENDPOINT=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1):29500

echo "Starting 7B multi-node workflow at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "RDZV_ENDPOINT: $RDZV_ENDPOINT"
echo "TOTAL_GPUS: $((SLURM_NNODES * SLURM_GPUS_ON_NODE))"

# Run the worker script on all nodes
srun -W 0 cambench_1_3_6_7b_multinode.sh