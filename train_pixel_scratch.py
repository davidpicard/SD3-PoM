"""Pixel-space flow matching training script (JiT-style, no VAE).

Time convention (JiT / rectified flow):
  t = 0  → pure noise    z_0 = ε,     ε ~ N(0, noise_scale² · I)
  t = 1  → clean image   z_1 = x_0

Forward process:  z_t = t · x_0 + (1−t) · ε

Model predicts x_pred (clean image). Loss:
  L = E_t[ ‖x_pred − x_0‖² / (1−t)² ]  ← equivalent to v-prediction MSE

Timestep conditioning: t_int = round((1−t_cont) · 999)  (high int = high noise,
consistent with SD3's CombinedTimestepTextProjEmbeddings convention).

Noise scale: ε ~ N(0, (patch_size/16)² · I)  — matches JiT's SNR normalisation.
"""

import argparse
import contextlib
import functools
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from diffusers import StableDiffusion3Pipeline
from safetensors.torch import save_file as safetensors_save_file
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from torchvision import transforms

from pom_sd3.convert import build_pixel_grouped
from pom_sd3.model import PomSD3Transformer2DModel


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_ddp() -> int:
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
    return 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_fsdp(model: torch.nn.Module, local_rank: int,
                    gpus_per_node: int | None = None) -> torch.nn.Module:
    if not dist.is_initialized():
        return model
    from diffusers.models.attention import JointTransformerBlock
    from pom_sd3.blocks import JointPoMBlock
    mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={JointPoMBlock, JointTransformerBlock},
    )
    if gpus_per_node is not None and gpus_per_node < dist.get_world_size():
        rank     = dist.get_rank()
        ws       = dist.get_world_size()
        num_nodes = ws // gpus_per_node
        node_idx  = rank // gpus_per_node
        all_intra = [
            dist.new_group(list(range(n * gpus_per_node, (n + 1) * gpus_per_node)))
            for n in range(num_nodes)
        ]
        all_inter = [
            dist.new_group(list(range(g, ws, gpus_per_node)))
            for g in range(gpus_per_node)
        ]
        intra_group = all_intra[node_idx]
        inter_group = all_inter[rank % gpus_per_node]
        if is_main():
            print(f"HYBRID_SHARD: {gpus_per_node} GPUs/node × {num_nodes} nodes")
        return FSDP(
            model,
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            process_group=(intra_group, inter_group),
            auto_wrap_policy=wrap_policy,
            mixed_precision=mp,
            device_id=local_rank,
            forward_prefetch=True,
        )
    return FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp,
        device_id=local_rank,
        forward_prefetch=True,
    )


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def lr_schedule(step: int, warmup_steps: int, max_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def find_latest_checkpoint(out_dir: Path) -> Path | None:
    ckpts = sorted(
        (p for p in out_dir.glob("step_*") if p.is_dir() and (p / "config.json").exists()),
        key=lambda p: int(p.name.split("_")[1]),
    )
    return ckpts[-1] if ckpts else None


def save_checkpoint(model, optimizer, step: int, ckpt_dir: Path) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(model, FSDP):
        fsdp_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, fsdp_cfg):
            full_sd = model.state_dict()
        _prefix = "_fsdp_wrapped_module."
        full_sd = {
            (k[len(_prefix):] if k.startswith(_prefix) else k): v
            for k, v in full_sd.items()
        }
        full_osd = FSDP.full_optim_state_dict(model, optimizer)
        if is_main():
            safetensors_save_file(full_sd, ckpt_dir / "diffusion_pytorch_model.safetensors")
            inner = getattr(model, "_fsdp_wrapped_module", model)
            inner.save_config(ckpt_dir)
            torch.save(full_osd, ckpt_dir / "optimizer.pt")
            (ckpt_dir / "train_state.json").write_text(json.dumps({"step": step}))
        dist.barrier()
    else:
        model.save_pretrained(ckpt_dir)
        param_to_name = {id(p): n for n, p in model.named_parameters()}
        named_state = {
            param_to_name[id(p)]: {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in state.items()
            }
            for p, state in optimizer.state.items()
            if id(p) in param_to_name
        }
        torch.save({"named_state": named_state}, ckpt_dir / "optimizer.pt")
        (ckpt_dir / "train_state.json").write_text(json.dumps({"step": step}))


def load_checkpoint_optimizer(opt_data: dict, optimizer, model, device) -> None:
    named_state = opt_data["named_state"]
    param_to_name = {id(p): n for n, p in model.named_parameters()}
    all_params = [p for g in optimizer.param_groups for p in g["params"]]
    new_state: dict = {}
    for i, p in enumerate(all_params):
        name = param_to_name.get(id(p))
        if name and name in named_state:
            new_state[i] = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in named_state[name].items()
            }
    pos = 0
    new_groups = []
    for g in optimizer.param_groups:
        ng = {k: v for k, v in g.items() if k != "params"}
        ng["params"] = list(range(pos, pos + len(g["params"])))
        new_groups.append(ng)
        pos += len(g["params"])
    optimizer.load_state_dict({"state": new_state, "param_groups": new_groups})


def load_optimizer_fsdp(model: FSDP, optimizer, ckpt_dir: Path) -> None:
    full_osd = torch.load(ckpt_dir / "optimizer.pt", map_location="cpu") if is_main() else None
    sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, model, optim=optimizer)
    optimizer.load_state_dict(sharded_osd)


# ---------------------------------------------------------------------------
# Dataset  (identical preprocessing to train_scratch.py — images in [-1, 1])
# ---------------------------------------------------------------------------

class GPicDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dataset_name: str,
        split: str,
        image_size: int,
        rank: int,
        world_size: int,
        caption_type: str | None = None,
        dataset_dir: str | None = None,
    ):
        self._caption_type = caption_type if caption_type != "all" else None
        self._image_size = image_size

        if dataset_dir:
            import glob as _glob
            all_tars = sorted(_glob.glob(os.path.join(dataset_dir, split, "*.tar")))
            if not all_tars:
                raise FileNotFoundError(f"No .tar shards in {os.path.join(dataset_dir, split)}")
            self._tar_files = all_tars[rank::world_size]
            self._ds = None
        else:
            from datasets import load_dataset
            hf_ds = load_dataset(dataset_name, split=split, streaming=True)
            if world_size > 1:
                hf_ds = hf_ds.shard(num_shards=world_size, index=rank)
            self._ds = hf_ds
            self._tar_files = None

    def _preprocess_img(self, img):
        orig_w, orig_h = img.size
        img = transforms.functional.resize(
            img, self._image_size,
            interpolation=transforms.InterpolationMode.BICUBIC,
        )
        i, j, h, w = transforms.RandomCrop.get_params(img, (self._image_size, self._image_size))
        img = transforms.functional.crop(img, i, j, h, w)
        img = transforms.functional.to_tensor(img)
        img = transforms.functional.normalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        crop_str = f"[crop: {orig_h}x{orig_w}, offset: {i},{j}]"
        return img, crop_str

    def __iter__(self):
        if self._tar_files is not None:
            import json as _json
            import tarfile
            from io import BytesIO
            from PIL import Image as PILImage
            for tar_path in self._tar_files:
                try:
                    pending: dict = {}
                    with tarfile.open(tar_path, "r:") as tf:
                        for member in tf:
                            if not member.isfile() or "." not in member.name:
                                continue
                            base, ext = member.name.rsplit(".", 1)
                            f = tf.extractfile(member)
                            if f is None:
                                continue
                            data = f.read()
                            entry = pending.setdefault(base, {})
                            if ext == "json":
                                entry["json"] = data
                            elif ext in ("jpg", "jpeg", "png"):
                                entry["img"] = data
                            if "json" in entry and "img" in entry:
                                del pending[base]
                                try:
                                    meta = _json.loads(entry["json"])
                                    img = PILImage.open(BytesIO(entry["img"])).convert("RGB")
                                except Exception:
                                    continue
                                caption_type = meta.get("caption_type")
                                caption = meta.get("caption", "")
                                if self._caption_type and caption_type != self._caption_type:
                                    continue
                                if not caption:
                                    continue
                                pixel_values, crop_str = self._preprocess_img(img)
                                yield {"pixel_values": pixel_values, "caption": caption,
                                       "crop_str": crop_str}
                except Exception:
                    continue
        else:
            for sample in self._ds:
                try:
                    img = sample["image"].convert("RGB")
                except Exception:
                    continue
                caption = sample.get("caption", "")
                if not caption:
                    continue
                if self._caption_type and sample.get("caption_type") != self._caption_type:
                    continue
                pixel_values, crop_str = self._preprocess_img(img)
                yield {"pixel_values": pixel_values, "caption": caption, "crop_str": crop_str}


def gpic_collate(batch: list[dict]) -> dict:
    result = {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "caption": [b["caption"] for b in batch],
    }
    if "crop_str" in batch[0]:
        result["crop_str"] = [b["crop_str"] for b in batch]
    return result


# ---------------------------------------------------------------------------
# Sample prompts
# ---------------------------------------------------------------------------

SAMPLE_PROMPTS = [
    "a serene mountain landscape at sunrise, photorealistic",
    "a cyberpunk city at night, neon lights reflecting on wet streets",
    "a portrait of a fox in a business suit, oil painting",
    "abstract colorful geometric shapes, vibrant, high contrast",
    "an astronaut floating in space, Earth visible in the background",
    "a cozy library with warm lighting and shelves full of books",
    "a dragon perched on a medieval castle tower, fantasy art",
    "a bowl of ramen with steam rising, food photography",
    "a watercolor painting of a Venice canal at dusk",
    "a robot tending to a flower garden, whimsical illustration",
    "dense rainforest with rays of sunlight piercing the canopy",
    "a black and white portrait of an elderly woman, cinematic",
    "a futuristic space station interior, hard sci-fi concept art",
    "cherry blossom trees along a river in spring, Japan",
    "a close-up of a honeybee on a sunflower, macro photography",
]


# ---------------------------------------------------------------------------
# Text encoder noise suppression
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _silence_encoding_noise():
    """Suppress tokenizer noise during encode_prompt.

    Covers two channels:
    - fd 2 / sys.stderr: the Rust fast-tokenizer and diffusers CLIP-truncation
      warning both write directly to fd 2, bypassing Python's logging system.
      (<|endoftext|> padding tokens printed when caption exceeds 77 CLIP tokens.)
    - transformers.tokenization_utils_base logger: emits the
      "Token indices sequence length is longer than …" warning via Python
      logging, which wandb captures through its root-logger handler.
      Raising the level to ERROR for the duration prevents it from
      appearing in wandb logs.
    """
    import logging as _logging
    tok_logger = _logging.getLogger("transformers.tokenization_utils_base")
    old_level = tok_logger.level
    tok_logger.setLevel(_logging.ERROR)

    old_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stderr.flush()
        sys.stderr.close()
        sys.stderr = old_stderr
        os.dup2(old_fd, 2)
        os.close(old_fd)
        tok_logger.setLevel(old_level)


# ---------------------------------------------------------------------------
# Parallel text encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def fast_encode_prompt(text_pipe, captions, max_sequence_length, device):
    """Drop-in replacement for text_pipe.encode_prompt() with ~3× lower latency.

    encode_prompt tokenises sequentially between each GPU call, leaving the GPU
    idle while the CPU prepares the next tokeniser batch.  This function
    pre-tokenises all three inputs on the CPU before touching the GPU, then
    dispatches T5 and [CLIP-L + CLIP-G] onto separate CUDA streams so they
    overlap.  Total latency collapses from ~3× T5 to ~1× T5.

    Output matches encode_prompt (with negative_* omitted):
      enc_hs : (B, 77 + max_sequence_length, 4096)
               CLIP-L‖CLIP-G tokens (zero-padded to 4096) ++ T5 tokens
      pooled  : (B, 2048)   CLIP-L pooled (768) ++ CLIP-G pooled (1280)
    """
    # 1. Tokenise all three on CPU — no GPU work yet, no stall bubbles later.
    clip_ids_1 = text_pipe.tokenizer(
        captions, padding="max_length", max_length=77,
        truncation=True, return_tensors="pt",
    ).input_ids.to(device)

    clip_ids_2 = text_pipe.tokenizer_2(
        captions, padding="max_length", max_length=77,
        truncation=True, return_tensors="pt",
    ).input_ids.to(device)

    t5_ids = text_pipe.tokenizer_3(
        captions, padding="max_length", max_length=max_sequence_length,
        truncation=True, return_tensors="pt",
    ).input_ids.to(device)

    # 2. Launch T5 and CLIPs on separate streams — they overlap on the GPU.
    stream_t5   = torch.cuda.Stream(device=device)
    stream_clip = torch.cuda.Stream(device=device)

    with torch.cuda.stream(stream_t5):
        # No attention_mask: matches encode_prompt, T5 attends to all positions.
        t5_tok = text_pipe.text_encoder_3(input_ids=t5_ids).last_hidden_state  # (B, max_seq, 4096)

    with torch.cuda.stream(stream_clip):
        out1       = text_pipe.text_encoder(input_ids=clip_ids_1, output_hidden_states=True)
        clip1_tok  = out1.hidden_states[-2]               # (B, 77, 768) penultimate
        clip1_pool = out1.text_embeds                     # (B, 768) projected CLS

        out2       = text_pipe.text_encoder_2(input_ids=clip_ids_2, output_hidden_states=True)
        clip2_tok  = out2.hidden_states[-2]               # (B, 77, 1280) penultimate
        clip2_pool = out2.text_embeds                     # (B, 1280) projected CLS

    # 3. Sync back to the default stream, then assemble.
    curr = torch.cuda.current_stream(device=device)
    curr.wait_stream(stream_t5)
    curr.wait_stream(stream_clip)

    # CLIP tokens: concat on feature dim → pad to T5's 4096 → cat on seq dim with T5
    clip_tok = torch.cat([clip1_tok, clip2_tok], dim=-1)           # (B, 77, 2048)
    clip_tok = F.pad(clip_tok, (0, t5_tok.shape[-1] - clip_tok.shape[-1]))  # (B, 77, 4096)
    enc_hs   = torch.cat([clip_tok, t5_tok], dim=1)                # (B, 77+max_seq, 4096)
    pooled   = torch.cat([clip1_pool, clip2_pool], dim=-1)         # (B, 2048)

    return enc_hs.to(dtype=torch.bfloat16), pooled.to(dtype=torch.bfloat16)


# ---------------------------------------------------------------------------
# Euler sampler (JiT: t: 0→1, x-prediction + CFG)
# ---------------------------------------------------------------------------

def generate_samples_pixel(
    model,
    text_pipe,
    step: int,
    device,
    image_size: int,
    patch_size: int,
    null_enc_hs,
    null_pooled,
    num_prompts: int = 4,
    num_steps: int = 50,
    guidance_scale: float = 4.0,
    cfg_start_t: float = 0.1,
):
    """Euler ODE sampler (t: 0→1) with CFG for t > cfg_start_t.

    All ranks participate in model.forward() (FSDP collective).
    Only rank 0 saves and logs images to wandb.
    """
    if text_pipe is None:
        return
    if is_main():
        print(f"Generating pixel samples at step {step} ...")

    noise_scale = patch_size / 16.0
    prompts = SAMPLE_PROMPTS[:num_prompts]

    # All ranks encode text independently (not collective), same inputs → same outputs
    with _silence_encoding_noise():
        enc_hs, pooled = fast_encode_prompt(text_pipe, prompts, 77, device)
    enc_hs  = enc_hs.to(device=device,  dtype=torch.bfloat16)
    pooled  = pooled.to(device=device,  dtype=torch.bfloat16)
    null_eh = null_enc_hs.expand(num_prompts, -1, -1).to(device=device, dtype=torch.bfloat16)
    null_p  = null_pooled.expand(num_prompts, -1).to(device=device,  dtype=torch.bfloat16)

    B = num_prompts
    dt = 1.0 / num_steps

    # Fixed seed so all ranks generate the same initial noise
    g = torch.Generator(device=device).manual_seed(step)
    z = torch.randn(B, 3, image_size, image_size, device=device,
                    dtype=torch.bfloat16, generator=g) * noise_scale

    with torch.no_grad():
        for i in range(num_steps):
            t_cont = i * dt                                    # scalar in [0, (N-1)/N]
            t_int  = int(round((1.0 - t_cont) * 999))         # high int = high noise
            t_tensor = torch.full((B,), t_int, device=device, dtype=torch.long)
            one_minus_t = max(1.0 - t_cont, 1e-4)

            x_pred_cond = model(
                hidden_states=z,
                encoder_hidden_states=enc_hs,
                pooled_projections=pooled,
                timestep=t_tensor,
            ).sample

            if t_cont >= cfg_start_t:
                x_pred_uncond = model(
                    hidden_states=z,
                    encoder_hidden_states=null_eh,
                    pooled_projections=null_p,
                    timestep=t_tensor,
                ).sample
                x_pred = x_pred_uncond + guidance_scale * (x_pred_cond - x_pred_uncond)
            else:
                x_pred = x_pred_cond

            # v = (x_pred - z_t) / (1-t)  (velocity toward clean image)
            v = (x_pred - z) / one_minus_t
            z = z + dt * v  # Euler step

    z = z.clamp(-1.0, 1.0)

    if is_main():
        tmpdir = tempfile.mkdtemp()
        try:
            from PIL import Image as PILImage
            for i in range(B):
                arr = z[i].float().cpu()
                arr = ((arr + 1.0) / 2.0).clamp(0, 1)
                arr = (arr.permute(1, 2, 0).numpy() * 255).astype("uint8")
                PILImage.fromarray(arr).save(
                    os.path.join(tmpdir, f"{i:03d}.jpg"), format="JPEG", quality=85
                )
            images = [
                wandb.Image(os.path.join(tmpdir, f"{i:03d}.jpg"), caption=prompts[i])
                for i in range(B)
            ]
            wandb.log({"samples": images}, step=step)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Pixel-space JiT flow matching trainer")
    p.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium",
                   help="SD3.5 model path (text encoders only — no VAE loaded)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--resume", action="store_true",
                   help="Auto-resume from the latest checkpoint in output_dir")
    p.add_argument("--resume_from", default=None,
                   help="Resume from an explicit checkpoint path")
    p.add_argument("--init_from", default=None,
                   help="Load weights from a checkpoint but start training fresh (step=0)")

    # Dataset
    p.add_argument("--dataset_name", default="stanford-vision-lab/gpic")
    p.add_argument("--dataset_dir", default=None)
    p.add_argument("--dataset_split", default="train")
    p.add_argument("--caption_type", default="all",
                   choices=["all", "tag", "short", "medium", "long"])
    p.add_argument("--image_size", type=int, default=512,
                   help="Training resolution in pixels")

    # Model
    p.add_argument("--patch_size", type=int, default=32,
                   help="Patch size in pixels (32 → 256 tokens at 512px, 1024 tokens at 1024px)")
    p.add_argument("--bottleneck_dim", type=int, default=256,
                   help="Intermediate dimension in the PixelPatchEmbed bottleneck projection")
    p.add_argument("--pom_degree", type=int, default=4)
    p.add_argument("--pom_expand", type=int, default=2)
    p.add_argument("--pom_n_groups", type=int, default=1)
    p.add_argument("--pom_n_sel_heads", type=int, default=24)
    p.add_argument("--pom_rope_max_seq_len", type=int, default=8192)

    # Training
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum_steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--caption_dropout", type=float, default=0.1,
                   help="Fraction of captions replaced with empty string for CFG training")
    p.add_argument("--max_sequence_length", type=int, default=77)
    p.add_argument("--crop_str_dropout", type=float, default=0.1)
    p.add_argument("--logit_normal_mean", type=float, default=-0.8,
                   help="Mean of logit-normal t-sampling. JiT default -0.8 → mean t≈0.31")
    p.add_argument("--logit_normal_std", type=float, default=0.8,
                   help="Std of logit-normal t-sampling")
    p.add_argument("--max_steps", type=int, default=1_000_000)
    p.add_argument("--warmup_steps", type=int, default=2_000)

    # Logging / checkpointing
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=40_000)
    p.add_argument("--sample_every", type=int, default=20_000)
    p.add_argument("--num_sample_prompts", type=int, default=4)
    p.add_argument("--num_sample_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--wandb_project", default="pixel-pom-g16")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_offline", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--gpus_per_node", type=int, default=None)
    p.add_argument("--smoke_test", action="store_true",
                   help="5 steps on random data with tiny model, then exit")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    out_dir = Path(args.output_dir)
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)

    if is_main():
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            mode="offline" if args.wandb_offline else "online",
            settings=wandb.Settings(console="off"),
        )

    # --- Checkpoint resolution ---
    resume_dir: Path | None = None
    init_dir: Path | None = None
    if args.resume_from:
        resume_dir = Path(args.resume_from)
    elif args.resume:
        resume_dir = find_latest_checkpoint(out_dir)
        if resume_dir is None:
            if args.init_from:
                init_dir = Path(args.init_from)
                if is_main():
                    print(f"No checkpoint found — loading weights from {init_dir} (fresh state)")
            elif is_main():
                print("No checkpoint found — starting fresh")
    elif args.init_from:
        init_dir = Path(args.init_from)

    # --- Text encoders (no VAE, no transformer from SD3) ---
    if not args.smoke_test:
        _local = Path(args.model_id).exists()
        print(f"[rank {rank}] Loading text encoders ...")
        text_pipe = StableDiffusion3Pipeline.from_pretrained(
            args.model_id, transformer=None, vae=None,
            torch_dtype=torch.bfloat16,
            local_files_only=_local,
        ).to(device)
        for enc in (text_pipe.text_encoder, text_pipe.text_encoder_2, text_pipe.text_encoder_3):
            if enc is not None:
                enc.requires_grad_(False)
        if text_pipe.text_encoder is not None:
            text_pipe.text_encoder = torch.compile(text_pipe.text_encoder, dynamic=True)
        if text_pipe.text_encoder_2 is not None:
            text_pipe.text_encoder_2 = torch.compile(text_pipe.text_encoder_2, dynamic=True)
        if text_pipe.text_encoder_3 is not None:
            text_pipe.text_encoder_3 = torch.compile(text_pipe.text_encoder_3, dynamic=True)

        # Pre-compute null embeddings for CFG dropout + sampler
        with _silence_encoding_noise():
            null_enc_hs, null_pooled = fast_encode_prompt(
                text_pipe, [""], args.max_sequence_length, device,
            )
        null_enc_hs = null_enc_hs.to(device=device, dtype=torch.bfloat16)
        null_pooled = null_pooled.to(device=device, dtype=torch.bfloat16)
    else:
        text_pipe = null_enc_hs = null_pooled = None

    # --- Model ---
    if resume_dir is not None:
        print(f"[rank {rank}] Resuming from {resume_dir} ...")
        model = PomSD3Transformer2DModel.from_pretrained(resume_dir).to(
            device=device, dtype=torch.bfloat16
        )
    elif init_dir is not None:
        print(f"[rank {rank}] Loading weights from {init_dir} (new training phase) ...")
        model = PomSD3Transformer2DModel.from_pretrained(init_dir).to(
            device=device, dtype=torch.bfloat16
        )
    elif not args.smoke_test:
        print(f"[rank {rank}] Building pixel-space g16 model from scratch ...")
        model = build_pixel_grouped(
            patch_size=args.patch_size,
            bottleneck_dim=args.bottleneck_dim,
            pom_degree=args.pom_degree,
            pom_expand=args.pom_expand,
            pom_n_groups=args.pom_n_groups,
            pom_n_sel_heads=args.pom_n_sel_heads,
            pom_rope_max_seq_len=args.pom_rope_max_seq_len,
            torch_dtype=torch.bfloat16,
            device=device,
        )
    else:
        # Tiny smoke-test model
        model = PomSD3Transformer2DModel(
            sample_size=64, patch_size=16, in_channels=3, num_layers=4,
            attention_head_dim=16, num_attention_heads=4,
            joint_attention_dim=4096, caption_projection_dim=64,
            pooled_projection_dim=2048, out_channels=3,
            pos_embed_max_size=16, dual_attention_layers=(0,),
            pom_layers=(1, 2, 3),
            pom_degree=2, pom_expand=2, pom_n_groups=1, pom_n_sel_heads=1,
            lora_rank=0,
            pixel_patch_bottleneck_dim=32,
        ).to(device=device, dtype=torch.bfloat16)

    if is_main():
        n_total = sum(p.numel() for p in model.parameters())
        print(f"Model: {n_total/1e6:.1f}M parameters")
        cfg = dict(model.config)
        wandb.config.update({"model_" + k: v for k, v in cfg.items()}, allow_val_change=True)

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    model = wrap_model_fsdp(model, local_rank, gpus_per_node=args.gpus_per_node)
    model.train()

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        list(model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # --- Restore optimizer and step ---
    step = 0
    if resume_dir is not None:
        opt_path = resume_dir / "optimizer.pt"
        if opt_path.exists():
            if isinstance(model, FSDP):
                load_optimizer_fsdp(model, optimizer, resume_dir)
            else:
                load_checkpoint_optimizer(
                    torch.load(opt_path, map_location="cpu"), optimizer, model, device
                )
            if is_main():
                print(f"  Optimizer state restored from {opt_path}")
        state_path = resume_dir / "train_state.json"
        if state_path.exists():
            step = json.loads(state_path.read_text())["step"] + 1
            if is_main():
                print(f"  Resuming at step {step}")

    # --- Dataset ---
    if args.smoke_test:
        class _SmokeStream(torch.utils.data.IterableDataset):
            def __iter__(self):
                while True:
                    yield {
                        "pixel_values": torch.randn(3, args.image_size, args.image_size),
                        "caption": "a test image",
                    }
        dataset = _SmokeStream()
    else:
        dataset = GPicDataset(
            dataset_name=args.dataset_name,
            split=args.dataset_split,
            image_size=args.image_size,
            rank=rank,
            world_size=world_size,
            caption_type=args.caption_type,
            dataset_dir=args.dataset_dir,
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0 if args.smoke_test else args.num_workers,
        pin_memory=not args.smoke_test,
        collate_fn=gpic_collate,
    )

    # --- Training loop ---
    noise_scale = args.patch_size / 16.0  # ε ~ N(0, noise_scale² · I)
    start_step = step
    t0 = time.time()

    optimizer.zero_grad(set_to_none=True)
    batch_iter = iter(loader)

    while step < args.max_steps:
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(loader)
            batch = next(batch_iter)

        lr = lr_schedule(step, args.warmup_steps, args.max_steps, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        pixel_values = batch["pixel_values"]
        captions = batch["caption"]
        B = pixel_values.shape[0]

        # Optional: append crop info to captions
        crop_strs = batch.get("crop_str")
        if crop_strs is not None and args.crop_str_dropout < 1.0:
            captions = [
                cap + " " + cs if random.random() > args.crop_str_dropout else cap
                for cap, cs in zip(captions, crop_strs)
            ]

        # x_0: clean pixel image in [-1, 1]
        x_0 = pixel_values.to(device=device, dtype=torch.bfloat16)

        # Text embeddings
        if text_pipe is not None:
            with _silence_encoding_noise():
                enc_hs, pooled = fast_encode_prompt(
                    text_pipe, captions, args.max_sequence_length, device,
                )
            # CFG dropout: replace some captions with null embedding
            if args.caption_dropout > 0 and null_enc_hs is not None:
                drop = torch.rand(B, device=device) < args.caption_dropout
                if drop.any():
                    n = int(drop.sum())
                    enc_hs[drop] = null_enc_hs.expand(n, -1, -1)
                    pooled[drop] = null_pooled.expand(n, -1)
        else:
            enc_hs = torch.randn(B, 8, 4096, device=device, dtype=torch.bfloat16)
            pooled = torch.randn(B, 2048, device=device, dtype=torch.bfloat16)

        # JiT flow matching
        # Sample t ~ LogitNormal(μ, σ²) with μ=logit_normal_mean, σ=logit_normal_std
        t_cont = torch.sigmoid(
            torch.randn(B, device=device) * args.logit_normal_std + args.logit_normal_mean
        ).clamp(1e-4, 1 - 1e-4)  # (B,) in (0, 1)

        # Timestep integer for model conditioning: high t_int ↔ high noise level
        # (1−t_cont) maps JiT's "amount of signal" to SD3-style "sigma/noise level")
        t_int = ((1.0 - t_cont) * 999).long().clamp(0, 999)  # (B,)

        # Noisy sample: z_t = t · x_0 + (1−t) · ε
        eps = torch.randn_like(x_0) * noise_scale
        t_view = t_cont.view(B, 1, 1, 1)
        z_t = t_view * x_0 + (1.0 - t_view) * eps

        # Forward (suppress FSDP reduce-scatter on non-final micro-batches)
        is_last_accum = (step + 1) % args.grad_accum_steps == 0
        sync_ctx = (
            contextlib.nullcontext()
            if not isinstance(model, FSDP) or is_last_accum
            else model.no_sync()
        )
        with sync_ctx:
            x_pred = model(
                hidden_states=z_t,
                encoder_hidden_states=enc_hs,
                pooled_projections=pooled,
                timestep=t_int,
            ).sample  # (B, 3, H, W) — predicted clean image

            # x-prediction loss with (1−t)² weighting ≡ v-prediction MSE
            # Clamp denominator to 5e-2 (JiT paper): caps max weight at 400,
            # preventing rare near-clean samples from dominating gradients.
            one_minus_t = (1.0 - t_cont).clamp(min=5e-2)           # (B,)
            loss_per = F.mse_loss(x_pred, x_0, reduction="none").mean(dim=(1, 2, 3))  # (B,)
            weighted_loss = (loss_per / one_minus_t.pow(2)).mean()
            (weighted_loss / args.grad_accum_steps).backward()

        if is_last_accum:
            if isinstance(model, FSDP):
                model.clip_grad_norm_(1.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # --- Logging ---
        if is_main() and step % args.log_every == 0:
            elapsed = time.time() - t0
            sps = (step + 1 - start_step) * args.batch_size * world_size / elapsed
            t_cpu = t_cont.detach().cpu()
            lp = loss_per.detach().cpu()
            # JiT: t=0 is high noise, t=1 is low noise (reversed from SD3)
            high_noise = t_cpu < 0.33
            mid_noise  = (t_cpu >= 0.33) & (t_cpu < 0.67)
            low_noise  = t_cpu >= 0.67
            log = {
                "loss":             weighted_loss.item(),
                "loss_high_noise":  lp[high_noise].mean().item() if high_noise.any() else float("nan"),
                "loss_mid_noise":   lp[mid_noise].mean().item()  if mid_noise.any()  else float("nan"),
                "loss_low_noise":   lp[low_noise].mean().item()  if low_noise.any()  else float("nan"),
                "lr": lr,
                "step": step,
                "samples_per_sec": sps,
            }
            wandb.log(log, step=step)
            print(f"step={step:7d}  loss={log['loss']:.4f}  lr={lr:.2e}  {sps:.1f} samp/s")

        # --- Checkpointing ---
        if step > 0 and step % args.save_every == 0:
            ckpt_dir = out_dir / f"step_{step:07d}"
            save_checkpoint(model, optimizer, step, ckpt_dir)
            if is_main():
                print(f"Saved checkpoint to {ckpt_dir}")

        # --- Sample generation ---
        if step > 0 and step % args.sample_every == 0 and not args.smoke_test:
            model.eval()
            generate_samples_pixel(
                model=model,
                text_pipe=text_pipe,
                step=step,
                device=device,
                image_size=args.image_size,
                patch_size=args.patch_size,
                null_enc_hs=null_enc_hs,
                null_pooled=null_pooled,
                num_prompts=args.num_sample_prompts,
                num_steps=args.num_sample_steps,
                guidance_scale=args.guidance_scale,
            )
            model.train()

        step += 1

        if (out_dir / ".save_and_exit").exists():
            ckpt_dir = out_dir / f"step_{step:07d}"
            save_checkpoint(model, optimizer, step, ckpt_dir)
            if is_main():
                (out_dir / ".save_and_exit").unlink(missing_ok=True)
                print(f"Wall-time signal — saved checkpoint to {ckpt_dir}. Exiting.")
            cleanup_ddp()
            sys.exit(0)

        if args.smoke_test and step >= 5:
            print("Smoke test passed — 5 steps completed successfully.")
            cleanup_ddp()
            return

    # --- Final save ---
    final_dir = out_dir / "final"
    if isinstance(model, FSDP):
        fsdp_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, fsdp_cfg):
            full_sd = model.state_dict()
        _prefix = "_fsdp_wrapped_module."
        full_sd = {(k[len(_prefix):] if k.startswith(_prefix) else k): v for k, v in full_sd.items()}
        if is_main():
            final_dir.mkdir(parents=True, exist_ok=True)
            safetensors_save_file(full_sd, final_dir / "diffusion_pytorch_model.safetensors")
            inner = getattr(model, "_fsdp_wrapped_module", model)
            inner.save_config(final_dir)
            (final_dir / "train_state.json").write_text(json.dumps({"step": step}))
        if dist.is_initialized():
            dist.barrier()
    else:
        if is_main():
            model.save_pretrained(final_dir)
            (final_dir / "train_state.json").write_text(json.dumps({"step": step}))

    if is_main():
        print(f"Training complete. Final model saved to {final_dir}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
