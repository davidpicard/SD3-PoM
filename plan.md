# SD3 with PoM

This repository is a research project aiming at replacing the Attention blocks in SD3.5 with PoM (https://github.com/davidpicard/pom, https://arxiv.org/abs/2604.06129)

Starting from a pretrained SD3.5 medium model, we replace all attention layers (both self-attention and cross-attention) with PoM, following SD3.5's existing positional encoding conventions (no RoPE), and distill the original model to get the same output per block and at the final output.

## Steps

### 1. Model Architecture
- Create a PoM variant of SD3.5 Medium that replaces every attention layer (self- and cross-attention) with PoM, staying as close to the SD3.5 architecture as possible in all other respects.
- **Weight initialization:** All non-attention weights (MLPs, norms, embeddings, etc.) are loaded directly from the pretrained SD3.5 checkpoint. PoM-specific parameters are randomly initialized.

### 2. Dataset
- Find a suitable caption-only dataset (e.g. a subset of LAION, DataComp, or similar). Only captions are needed — the training target is the teacher model's output, not real images.
- The dataset should be large enough for diversity but small enough for a short training run. A few million captions should suffice.

### 3. Training Script
- **Distillation loss:** Run the student (PoM-SD3.5) and the frozen teacher (original SD3.5) in parallel on the same batch. Match intermediate activations at each block boundary and at the final output using MSE + MAE.
- **Infrastructure:** DDP across 4 nodes × 4 H100s (16 GPUs total - using SLURM). Both models live in memory simultaneously; the 16-GPU setup is expected to provide sufficient batch size.
- **Logging:** Weights & Biases (wandb) for losses and periodic image samples to track visual progress.

### 4. Export
- Subclass `SD3Transformer2DModel` from diffusers, keeping the same forward interface as a drop-in replacement.
- Publish only the transformer weights to the HuggingFace Hub. Users load the rest (VAE, text encoders, scheduler) from the original `stabilityai/stable-diffusion-3.5-medium` repo.
- Usage:
  ```python
  from diffusers import StableDiffusion3Pipeline
  from pom_sd3 import PomSD3Transformer

  transformer = PomSD3Transformer.from_pretrained("you/pom-sd35-transformer")
  pipe = StableDiffusion3Pipeline.from_pretrained(
      "stabilityai/stable-diffusion-3.5-medium",
      transformer=transformer
  )
  images = pipe("a photo of a cat").images[0]
  ```


The complete offline workflow is now:

  On the frontend (internet-connected):
  # 1. Download captions (text only, fast)
  python download_data.py --captions \
      --dataset_name laion/laion-aesthetics-v2-5plus \
      --caption_column TEXT \
      --max_samples 5000000 \
      --captions_dir /shared/captions

  # 2. Download model weights (transformer + text encoders, no VAE)
  python download_data.py --model \
      --model_id stabilityai/stable-diffusion-3.5-medium \
      --model_dir /shared/models/sd3.5-medium

  On compute nodes (no internet):
  # 3. Precompute embeddings (needs text encoders + captions)
  python precompute_embeddings.py \
      --model_dir /shared/models/sd3.5-medium \
      --captions_dir /shared/captions \
      --output_dir /shared/embeddings 
   
  # 4. Train (needs transformer weights + embeddings)
  sbatch launch_slurm.sh   # points to /shared/models/sd3.5-medium
