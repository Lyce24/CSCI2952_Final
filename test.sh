#!/bin/bash -l

#SBATCH -J ssl_run
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2        # 2 tasks = 2 Lightning processes
#SBATCH --cpus-per-task=6          # 12 total CPUs
#SBATCH --gres=gpu:2               # 2 GPUs on the same node
#SBATCH --mem=64g
#SBATCH --constraint="a5000|a5500|geforce3090"
#SBATCH --time=48:00:00

#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

echo "Nodes: $SLURM_JOB_NODELIST"
echo "ntasks-per-node=$SLURM_NTASKS_PER_NODE cpus-per-task=$SLURM_CPUS_PER_TASK"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Load conda for non-interactive shells
source ~/miniconda3/etc/profile.d/conda.sh
conda activate SSL

echo "Python: $(which python)"
python -V
which python

# launch two ranks with the absolute interpreter
srun --export=ALL python Stage1_MAE.py