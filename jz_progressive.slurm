#!/bin/bash
#SBATCH --job-name=pom-progressive
#SBATCH -C h100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --signal=B:USR1@300

set -x

module purge
module load arch/h100
module load python/3.12.2
source /lustre/fswork/projects/rech/gja/udd69tt/venv/torch_h100/bin/activate

export TMPDIR=$JOBSCRATCH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

mkdir -p logs

OUTPUT_DIR=./output/progressive-512
INIT_FROM=""

rm -f "$OUTPUT_DIR/.save_and_exit"

_requeue() {
    echo "$(date): Wall time approaching — flagging for checkpoint save ..."
    touch "$OUTPUT_DIR/.save_and_exit"
}
trap _requeue USR1

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29503

srun torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_progressive.py \
        --gpus_per_node 4 \
        --resume \
        ${INIT_FROM:+--init_from "$INIT_FROM"} \
        --model_id ./shared/models/sd3.5-medium \
        --output_dir "$OUTPUT_DIR" \
        --dataset_name stanford-vision-lab/gpic \
        --dataset_dir /lustre/fsmisc/dataset/HuggingFace/stanford-vision-lab/gpic/ \
        --num_workers 8 \
        --dataset_split train \
        --caption_type all \
        --caption_dropout 0.1 \
        --image_size 512 \
        --pom_degree 10 \
        --pom_expand 2 \
        --pom_n_groups 1 \
        --pom_n_sel_heads 24 \
        --pom_rope_max_seq_len 16384 \
        --phase_steps 10000 \
        --pretrained_lr_scale 0.1 \
        --block_warmup_steps 1000 \
        --min_phase_steps 5000 \
        --plateau_threshold 0.005 \
        --plateau_window 2000 \
        --consolidation_lr_scale 0.5 \
        --batch_size 32 \
        --grad_accum_steps 4 \
        --lr 1e-4 \
        --warmup_steps 2000 \
        --logit_normal_mean 0.0 \
        --logit_normal_std 0.8 \
        --max_steps 500000 \
        --log_every 500 \
        --save_every 10000 \
        --sample_every 20000 \
        --val_every 10000 \
        --n_val_images 256 \
        --gradient_checkpointing \
        --snr_gamma 5.0 \
        --ema_decay 0.9999 \
        --num_sample_prompts 25 \
        --wandb_project pom-progressive \
        --wandb_offline &
SRUN_PID=$!

wait $SRUN_PID
while [ $? -ge 128 ]; do
    wait $SRUN_PID
done

rm -f "$OUTPUT_DIR/.save_and_exit"
if [ ! -d "$OUTPUT_DIR/final" ]; then
    echo "$(date): Requeueing job $SLURM_JOB_ID ..."
    scontrol requeue "$SLURM_JOB_ID"
fi
# After training, sync wandb runs:
#   wandb sync ./wandb/run-*
