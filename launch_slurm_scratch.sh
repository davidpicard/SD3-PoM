#!/bin/bash
#SBATCH --job-name=sd3-pom-scratch
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --signal=B:USR1@300   # send SIGUSR1 to this script 5 min before wall time

mkdir -p logs

# Set OUTPUT_DIR and (for phase 2+) INIT_FROM to the previous phase's final checkpoint.
# Phase 1:  OUTPUT_DIR=.../phase1  INIT_FROM=""
# Phase 2:  OUTPUT_DIR=.../phase2  INIT_FROM=.../phase1/final
# --resume + --init_from together: if a phase-N checkpoint already exists in OUTPUT_DIR,
# --resume wins (requeue resume); on first launch of phase N, --init_from loads phase N-1 weights.
OUTPUT_DIR=/path/to/scratch-phase1
INIT_FROM=""   # set to previous phase final dir when starting a new resolution stage

_requeue() {
    if [ -d "$OUTPUT_DIR/final" ]; then
        echo "$(date): Training complete — skipping requeue."
    else
        echo "$(date): Wall time approaching — requeueing job $SLURM_JOB_ID ..."
        scontrol requeue "$SLURM_JOB_ID"
    fi
}
trap _requeue USR1

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

srun torchrun \
    --nproc_per_node=4 \
    --nnodes=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_scratch.py \
        --resume \
        ${INIT_FROM:+--init_from "$INIT_FROM"} \
        --model_id /path/to/models/sd3.5-medium \
        --output_dir "$OUTPUT_DIR" \
        --dataset_name stanford-vision-lab/gpic \
        --dataset_dir /path/to/gpic \
        --dataset_split train \
        --caption_type all \
        --image_size 1024 \
        --pom_degree 4 \
        --pom_expand 2 \
        --pom_n_groups 1 \
        --pom_n_sel_heads 24 \
        --lora_rank 0 \
        --gradient_checkpointing \
        --batch_size 2 \
        --grad_accum_steps 4 \
        --lr 1e-4 \
        --warmup_steps 2000 \
        --max_steps 500000 \
        --log_every 50 \
        --save_every 5000 \
        --sample_every 5000 \
        --wandb_project sd3-pom-scratch \
        --wandb_offline &

wait $!
# After training, sync wandb runs:
#   wandb sync ./wandb/run-*
