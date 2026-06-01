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
#SBATCH --signal=B:USR1@300   # send SIGUSR1 to this batch script 5 min before wall time

mkdir -p logs

OUTPUT_DIR=/path/to/progressive-output

# --- Auto-requeue on timeout ---
# Slurm delivers SIGUSR1 (--signal=B:USR1@300) before the wall time expires.
# We requeue the job so it continues from the latest checkpoint via --resume.
# Skip if training already finished (train_progressive.py writes output_dir/final/).
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
    train_progressive.py \
        --resume \
        --model_id /path/to/models/sd3.5-medium \
        --embeddings_dir /path/to/embeddings \
        --output_dir "$OUTPUT_DIR" \
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
        --wandb_offline &   # run in background so the USR1 trap can fire while training runs

wait $!   # wait for srun; SIGUSR1 interrupts this, runs _requeue, then the job ends at wall time
# After training, sync runs from the frontend:
#   wandb sync ./wandb/run-*
