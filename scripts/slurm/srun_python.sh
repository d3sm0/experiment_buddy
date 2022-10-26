#!/bin/bash
#SBATCH --job-name=spython
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --time=0-12:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=long
#SBATCH --get-user-env=L

module purge
module load python cuda/11.1/cudnn/8.0 gcc
# export XLA_FLAGS=--xla_gpu_cuda_data_dir=/cvmfs/ai.mila.quebec/apps/x86_64/common/cuda/10.1/
export LD_LIBRARY_PATH=/cvmfs/ai.mila.quebec/apps/x86_64/debian/gcc/9.3.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export NCCL_DEBUG="INFO"
export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"

source $HOME/venv/bin/activate
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
master_port=$((4440 + $RANDOM % 20))
export MASTER_PORT=$master_port
exec "$@"
