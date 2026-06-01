#!/bin/bash
#SBATCH --job-name=sd3-pom-progressive
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

mkdir -p logs

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

srun torchrun \
    --nproc_per_node=4 \
    --nnodes=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_progressive.py \
        --resume \
        --model_id /path/to/models/sd3.5-medium \
        --embeddings_dir /path/to/embeddings \
        --output_dir /path/to/progressive-output \
        --n_pom_blocks_start 1 \
        --replacement_step_schedule 1000 \
        --batch_size 2 \
        --grad_accum_steps 4 \
        --lr 1e-4 \
        --warmup_steps 500 \
        --max_steps 30000 \
        --teacher_steps_max 28 \
        --amortize_k 2 \
        --block_loss_weight 0.1 \
        --log_every 50 \
        --save_every 2000 \
        --sample_every 2000 \
        --wandb_project sd3-pom \
        --wandb_offline
# After training, sync runs from the frontend:
#   wandb sync ./wandb/run-*
