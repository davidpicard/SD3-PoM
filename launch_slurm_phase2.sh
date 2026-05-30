#!/bin/bash
#SBATCH --job-name=sd3-pom-phase2
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=24:00:00
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
    train_phase2.py \
        --resume_from /path/to/output/final \
        --model_id /path/to/models/sd3.5-medium \
        --embeddings_dir /path/to/embeddings \
        --output_dir /path/to/output_phase2 \
        --batch_size 4 \
        --grad_accum_steps 2 \
        --lr 1e-5 \
        --warmup_steps 500 \
        --max_steps 50000 \
        --block_loss_weight 0.1 \
        --pom_degree 4 \
        --pom_expand 2 \
        --pom_n_groups 1 \
        --pom_n_sel_heads 24 \
        --log_every 50 \
        --save_every 2000 \
        --sample_every 2000 \
        --wandb_project sd3-pom \
        --wandb_offline
