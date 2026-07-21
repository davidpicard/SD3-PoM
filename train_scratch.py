"""Train a fully-PoM SD3.5 model from scratch on the gpic image-caption dataset.

No teacher / distillation: real images are VAE-encoded on the fly, captions are
text-encoded on the fly, and the full PoM transformer (all 24 blocks, random init)
is trained end-to-end with a standard flow-matching velocity-prediction objective.

Usage (single GPU):
    python train_scratch.py \
        --model_id /path/to/sd3.5-medium \
        --output_dir ./scratch-output \
        --smoke_test

Usage (SLURM / torchrun multi-node):
    See launch_slurm_scratch.sh
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
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from safetensors.torch import save_file as safetensors_save_file

import wandb
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    StableDiffusion3Pipeline,
)
from torchvision import transforms

from pom_sd3 import PomSD3Transformer2DModel
from pom_sd3.convert import SD35_MEDIUM_CONFIG


# ---------------------------------------------------------------------------
# Helpers (shared with train_progressive.py)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _silence_encoding_noise():
    """Suppress tokenizer noise during encode_prompt.

    Covers two channels:
    - fd 2 / sys.stderr: the Rust fast-tokenizer writes directly to fd 2,
      bypassing Python's logging system.
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


def print_model_summary(model: torch.nn.Module, label: str = "") -> None:
    """Print a per-layer-type parameter count table to stdout."""
    groups = {
        "PoM operators":   lambda n: ".pom." in n or ".pom2." in n,
        "FF LoRA":         lambda n: ".ff_lora_" in n or ".ff_context_lora_" in n,
        "proj_out LoRA":   lambda n: "proj_out_lora_" in n,
        "norm_out":        lambda n: "norm_out." in n,
        "Attention":       lambda n: ".attn." in n,
        "Feed-forward":    lambda n: ((".ff." in n or ".ff_context." in n)
                                      and ".ff_lora_" not in n
                                      and ".ff_context_lora_" not in n),
        "Block norms":     lambda n: ".norm1" in n or ".norm2" in n,
        "Embeddings":      lambda n: any(k in n for k in (
                               "pos_embed", "time_text_embed",
                               "context_embedder", "patch_embed")),
        "proj_out (base)": lambda n: "proj_out" in n and "proj_out_lora_" not in n,
    }
    totals: dict[str, int] = {g: 0 for g in groups}
    trainable: dict[str, int] = {g: 0 for g in groups}
    totals["Other"] = trainable["Other"] = 0
    for name, param in model.named_parameters():
        n = param.numel()
        t = n if param.requires_grad else 0
        matched = False
        for g, pred in groups.items():
            if pred(name):
                totals[g] += n
                trainable[g] += t
                matched = True
                break
        if not matched:
            totals["Other"] += n
            trainable["Other"] += t
    all_groups = list(groups) + ["Other"]
    header = f"Model summary{' — ' + label if label else ''}"
    print(f"\n{header}")
    print(f"  {'Layer type':<22}  {'Total':>10}  {'Trainable':>10}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*10}")
    for first_trainable in (True, False):
        for g in all_groups:
            if (trainable[g] > 0) != first_trainable or totals[g] == 0:
                continue
            print(f"  {g:<22}  {totals[g]/1e6:>9.2f}M  {trainable[g]/1e6:>9.2f}M")
    grand_total = sum(totals.values())
    grand_trainable = sum(trainable.values())
    print(f"  {'─'*22}  {'─'*10}  {'─'*10}")
    print(f"  {'TOTAL':<22}  {grand_total/1e6:>9.2f}M  {grand_trainable/1e6:>9.2f}M\n")


def print_model_layers(model: torch.nn.Module) -> None:
    """Print a per-layer description: block type, trainable params, and role."""
    from diffusers.models.attention import JointTransformerBlock
    try:
        from pom_sd3.blocks import JointPoMBlock, JointLocalAttnBlock
    except ImportError:
        JointPoMBlock = JointLocalAttnBlock = None

    def _tr(mod):
        return sum(p.numel() for p in mod.parameters() if p.requires_grad)

    cfg = model.config
    W = 38  # width of the name/type column

    print("Model layers")
    print(f"  {'Layer':<{W}}  {'Trainable':>10}  Description")
    print(f"  {'─'*W}  {'─'*10}  {'─'*50}")

    # --- Input layers ---
    for attr, desc in [
        ("pos_embed",        f"latent patches → image tokens (patch_size={cfg.patch_size})"),
        ("time_text_embed",  f"timestep + pooled({cfg.pooled_projection_dim}) → temb({model.inner_dim})"),
        ("context_embedder", f"text enc({cfg.joint_attention_dim}) → ctx({cfg.caption_projection_dim})"),
    ]:
        mod = getattr(model, attr, None)
        if mod is None:
            continue
        name_col = f"{attr}  [{type(mod).__name__}]"
        t = _tr(mod)
        print(f"  {name_col:<{W}}  {t/1e6:>9.2f}M  {desc}")

    print(f"  {'─'*W}  {'─'*10}  {'─'*50}")

    # --- Transformer blocks ---
    for i, blk in enumerate(model.transformer_blocks):
        bname = type(blk).__name__
        t = _tr(blk)
        tr_str = f"{t/1e6:9.2f}M" if t > 0 else "  (frozen)"

        dual = getattr(blk, "use_dual_attention", False)
        cpo  = getattr(blk, "context_pre_only", False)

        if JointPoMBlock is not None and isinstance(blk, JointPoMBlock):
            mix = f"joint PoM deg={cfg.pom_degree} exp={cfg.pom_expand}"
            if dual:
                mix += " + dual PoM"
        elif JointLocalAttnBlock is not None and isinstance(blk, JointLocalAttnBlock):
            mix = f"local attn window={getattr(blk, 'window_m', '?')}"
            if dual:
                mix += " + dual local attn"
        elif isinstance(blk, JointTransformerBlock):
            mix = "joint full attn"
            if dual:
                mix += " + dual attn"
        else:
            mix = bname

        ff = "img FF + txt FF" if not cpo else "img FF only"
        flags = []
        if dual:
            flags.append("dual")
        if cpo:
            flags.append("ctx-pre-only")
        flag_str = f"  [{', '.join(flags)}]" if flags else ""

        name_col = f"[{i:2d}] {bname}"
        print(f"  {name_col:<{W}}  {tr_str:>10}  {mix}, {ff}{flag_str}")

    print(f"  {'─'*W}  {'─'*10}  {'─'*50}")

    # --- Output layers ---
    for attr, desc in [
        ("norm_out", "AdaLN output norm conditioned on temb"),
        ("proj_out", f"Linear({model.inner_dim} → {cfg.patch_size**2 * cfg.out_channels})  unpatchify"),
    ]:
        mod = getattr(model, attr, None)
        if mod is None:
            continue
        name_col = f"{attr}  [{type(mod).__name__}]"
        t = _tr(mod)
        print(f"  {name_col:<{W}}  {t/1e6:>9.2f}M  {desc}")

    lora_A = getattr(model, "proj_out_lora_A", None)
    if lora_A is not None:
        rank = lora_A.out_features
        t = _tr(lora_A) + _tr(model.proj_out_lora_B)
        name_col = f"proj_out_lora  [LoRA rank={rank}]"
        print(f"  {name_col:<{W}}  {t/1e6:>9.2f}M  LoRA on proj_out (merged at end of training)")

    print()


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
    return 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_fsdp(model: torch.nn.Module, local_rank: int) -> torch.nn.Module:
    """Wrap model in FSDP with FULL_SHARD when running distributed; no-op otherwise."""
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
    return FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp,
        device_id=local_rank,
    )


def lr_schedule(step: int, warmup_steps: int, max_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def find_latest_checkpoint(out_dir: Path) -> Path | None:
    ckpts = sorted(
        (p for p in out_dir.glob("step_*") if p.is_dir() and (p / "config.json").exists()),
        key=lambda p: int(p.name.split("_")[1]),
    )
    return ckpts[-1] if ckpts else None


def save_checkpoint(model, optimizer, step: int, ckpt_dir: Path) -> None:
    """Save model weights, optimizer state, and step counter.

    When FSDP is active all ranks must call this (collective state_dict gathering).
    Only rank 0 writes to disk; a dist.barrier() at the end synchronises ranks.
    Weights are saved in diffusers format (config.json + model.safetensors) so
    --init_from / from_pretrained keep working.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(model, FSDP):
        # --- gather full model state dict (rank 0 only, CPU-offloaded) ---
        fsdp_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, fsdp_cfg):
            full_sd = model.state_dict()
        # Strip "_fsdp_wrapped_module." prefix so from_pretrained can load the weights
        _prefix = "_fsdp_wrapped_module."
        full_sd = {
            (k[len(_prefix):] if k.startswith(_prefix) else k): v
            for k, v in full_sd.items()
        }

        # --- gather full optimizer state dict (collective; rank 0 only) ---
        full_osd = FSDP.full_optim_state_dict(model, optimizer)

        if is_main():
            safetensors_save_file(full_sd, ckpt_dir / "diffusion_pytorch_model.safetensors")
            # config.json — access inner module for ConfigMixin.save_config
            inner = getattr(model, "_fsdp_wrapped_module", model)
            inner.save_config(ckpt_dir)
            torch.save(full_osd, ckpt_dir / "optimizer.pt")
            (ckpt_dir / "train_state.json").write_text(json.dumps({"step": step}))

        dist.barrier()
    else:
        # Single-GPU path: use diffusers save_pretrained and name-keyed optimizer state
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
    """Scatter a saved full optimizer state dict across FSDP ranks."""
    full_osd = torch.load(ckpt_dir / "optimizer.pt", map_location="cpu") if is_main() else None
    sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, model, optim=optimizer)
    optimizer.load_state_dict(sharded_osd)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GPicDataset(torch.utils.data.IterableDataset):
    """Streams stanford-vision-lab/gpic, preprocesses images, yields (pixel_values, caption).

    Two backends:
    - Local (dataset_dir set): reads WebDataset tar shards directly from disk.
      Each tar contains paired <hash>.json + <hash>.jpg/png files.
      Shards are distributed across ranks by round-robin.
    - Hub (dataset_dir None): HF datasets streaming from the Hub.
    """

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
                raise FileNotFoundError(
                    f"No .tar shards found in {os.path.join(dataset_dir, split)}"
                )
            # Distribute shards across DDP ranks (round-robin)
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
        orig_w, orig_h = img.size  # PIL: (W, H)
        img = transforms.functional.resize(
            img, self._image_size,
            interpolation=transforms.InterpolationMode.BICUBIC,
        )
        i, j, h, w = transforms.RandomCrop.get_params(
            img, (self._image_size, self._image_size)
        )
        img = transforms.functional.crop(img, i, j, h, w)
        img = transforms.functional.to_tensor(img)
        img = transforms.functional.normalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        # (i, j) = (top, left) in the resized image's pixel coords
        crop_str = f"[crop: {orig_h}x{orig_w}, offset: {i},{j}]"
        return img, crop_str

    def __iter__(self):
        if self._tar_files is not None:
            import tarfile
            import json as _json
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
                                yield {
                                    "pixel_values": pixel_values,
                                    "caption": caption,
                                    "crop_str": crop_str,
                                }
                except Exception:
                    continue
        else:
            for sample in self._ds:
                # Hub-streamed: 'image' is a decoded PIL image, metadata is flat
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
                yield {
                    "pixel_values": pixel_values,
                    "caption": caption,
                    "crop_str": crop_str,
                }


def gpic_collate(batch: list[dict]) -> dict:
    result = {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "caption": [b["caption"] for b in batch],
    }
    if "crop_str" in batch[0]:
        result["crop_str"] = [b["crop_str"] for b in batch]
    return result


# ---------------------------------------------------------------------------
# Sample generation
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
    "a surrealist painting of melting clocks in a desert landscape",
    "a Viking longship on a stormy sea, dramatic lighting",
    "a bustling street market in Marrakech, golden hour",
    "an Art Deco poster of a luxury ocean liner",
    "a snowy owl in flight against a pale winter sky",
    "an underwater coral reef teeming with colorful fish",
    "a steampunk airship over a Victorian city, detailed illustration",
    "a minimalist ink drawing of a mountain range",
    "a wolf howling at the full moon in a pine forest, night",
    "a child blowing dandelion seeds in a summer meadow, soft focus",
]


def generate_samples(model, vae, text_pipe, step: int, device, num_prompts: int = 4, resolution: int = 1024):
    if text_pipe is None or vae is None:
        return
    if is_main():
        print(f"Generating sample images at step {step} ...")
    # All ranks must participate: the FSDP model.forward() is a collective.
    # We temporarily attach transformer and vae so the pipeline can run.
    text_pipe.transformer = model
    text_pipe.vae = vae
    text_pipe.set_progress_bar_config(disable=True)
    tmpdir = tempfile.mkdtemp() if is_main() else None
    try:
        for i, prompt in enumerate(SAMPLE_PROMPTS[:num_prompts]):
            with torch.no_grad():
                img = text_pipe(prompt, num_inference_steps=28, guidance_scale=8.0,
                               height=resolution, width=resolution).images[0]
            if is_main():
                path = os.path.join(tmpdir, f"{i:03d}.jpg")
                img.save(path, format="JPEG", quality=85)
        if is_main():
            images = [
                wandb.Image(os.path.join(tmpdir, f"{i:03d}.jpg"),
                            caption=SAMPLE_PROMPTS[i])
                for i in range(min(num_prompts, len(SAMPLE_PROMPTS)))
            ]
            wandb.log({"samples": images}, step=step)
    finally:
        text_pipe.transformer = None
        text_pipe.vae = None
        torch.cuda.empty_cache()
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium",
                   help="SD3.5 model path (provides VAE and text encoders)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--resume", action="store_true",
                   help="Auto-resume from the latest checkpoint in output_dir "
                        "(takes priority over --init_from; put both in the Slurm script "
                        "so phase-2 rerequeus resume from phase-2 checkpoints)")
    p.add_argument("--resume_from", default=None,
                   help="Resume from an explicit checkpoint path "
                        "(restores model weights, optimizer state, and step counter)")
    p.add_argument("--init_from", default=None,
                   help="Load model weights from a checkpoint but start training "
                        "fresh (step=0, new optimizer). Use for resolution-stage "
                        "transitions: --init_from ./phase1/final --output_dir ./phase2")

    # Dataset
    p.add_argument("--dataset_name", default="stanford-vision-lab/gpic")
    p.add_argument("--dataset_dir", default=None,
                   help="Local directory containing the dataset files. "
                        "Passed as data_dir to load_dataset; use when the cluster "
                        "has a local copy of the dataset.")
    p.add_argument("--dataset_split", default="train")
    p.add_argument("--caption_type", default="all",
                   choices=["all", "tag", "short", "medium", "long"],
                   help="Filter captions by type (default: use all types)")
    p.add_argument("--image_size", type=int, default=512,
                   help="Training resolution (images resized+cropped to this size)")

    # PoM architecture
    p.add_argument("--pom_degree", type=int, default=4)
    p.add_argument("--pom_expand", type=int, default=2)
    p.add_argument("--pom_n_groups", type=int, default=1)
    p.add_argument("--pom_n_sel_heads", type=int, default=24)
    p.add_argument("--pom_rope_max_seq_len", type=int, default=8192,
                   help="Max sequence length for PoMRoPE frequency tables (N_img + N_txt). "
                        "8192 covers 512px (1024 patches) and 1024px (4096 patches). "
                        "Use 32768 for 2048px.")
    p.add_argument("--lora_rank", type=int, default=0,
                   help="LoRA rank for FF layers (0 = no LoRA, recommended for from-scratch)")
    p.add_argument("--hybrid_n", type=int, default=1,
                   help="Block period: 1=full PoM, 0=full local attention, "
                        "k≥2=PoM every k layers with (k-1) LocalAttn in between")
    p.add_argument("--attention_window_m", type=int, default=4,
                   help="Half-side of 2D local attention window; each image token attends "
                        "to a (2m+1)×(2m+1) neighbourhood. Ignored when hybrid_n=1.")

    # Training
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--caption_dropout", type=float, default=0.1,
                   help="Fraction of captions replaced with empty string for CFG training "
                        "(trains the unconditional score; required for guidance_scale > 1 at inference)")
    p.add_argument("--crop_str_dropout", type=float, default=0.1,
                   help="Probability of NOT appending the crop info string to captions "
                        "(0=always append, 1=never; 0.1 means model sees it 90%% of the time)")
    p.add_argument("--logit_normal_mean", type=float, default=None,
                   help="Mean of the logit-normal timestep distribution. "
                        "Default: log(ref_image_size/image_size) — shifts sampling towards "
                        "high-noise steps at lower resolutions. Pass 0.0 for standard SD3.")
    p.add_argument("--ref_image_size", type=int, default=1024,
                   help="Reference resolution for auto logit-normal mean (default 1024)")
    p.add_argument("--max_steps", type=int, default=500_000)
    p.add_argument("--warmup_steps", type=int, default=2_000)

    # Logging / checkpointing
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=5_000)
    p.add_argument("--sample_every", type=int, default=5_000)
    p.add_argument("--num_sample_prompts", type=int, default=25)
    p.add_argument("--wandb_project", default="sd3-pom-scratch")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_offline", action="store_true")
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader worker processes for image loading and preprocessing")
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Recompute per-block activations to reduce activation memory "
                        "(~30%% extra compute; recommended at 1024px+)")
    p.add_argument("--smoke_test", action="store_true",
                   help="5 steps on random data with tiny model, then exit")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.logit_normal_mean is None:
        args.logit_normal_mean = math.log(args.ref_image_size / args.image_size)
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

    # --- Resolve checkpoint paths ---
    # resume_dir: full restore (weights + optimizer + step); used for requeue/interruption
    # init_dir:   weights only, fresh training state; used for resolution-stage transitions
    # Priority: resume_from > --resume auto-detect > init_from > random init
    resume_dir: Path | None = None
    init_dir: Path | None = None

    if args.resume_from:
        resume_dir = Path(args.resume_from)
    elif args.resume:
        resume_dir = find_latest_checkpoint(out_dir)
        if resume_dir is None:
            # No phase-2 checkpoint yet — fall back to --init_from if provided
            if args.init_from:
                init_dir = Path(args.init_from)
                if is_main():
                    print(f"No checkpoint in output_dir — loading weights from {init_dir} "
                          f"(fresh training state).")
            elif is_main():
                print("No checkpoint found in output_dir — starting fresh.")
    elif args.init_from:
        init_dir = Path(args.init_from)

    # --- VAE ---
    if not args.smoke_test:
        _local = Path(args.model_id).exists()
        print(f"[rank {rank}] Loading VAE ...")
        vae = AutoencoderKL.from_pretrained(
            args.model_id, subfolder="vae",
            torch_dtype=torch.bfloat16,
            local_files_only=_local,
        ).to(device)
        for p in vae.parameters():
            p.requires_grad_(False)
        vae.eval()
        vae = torch.compile(vae, dynamic=False)
    else:
        vae = None

    # --- Text encoders ---
    if not args.smoke_test:
        print(f"[rank {rank}] Loading text encoders ...")
        text_pipe = StableDiffusion3Pipeline.from_pretrained(
            args.model_id, transformer=None, vae=None,
            torch_dtype=torch.bfloat16,
            local_files_only=_local,
        ).to(device)
        for encoder in (text_pipe.text_encoder, text_pipe.text_encoder_2, text_pipe.text_encoder_3):
            if encoder is not None:
                encoder.requires_grad_(False)
        # Compile text encoders: CLIP has fixed 77-token inputs (dynamic=False);
        # T5 has variable-length captions (dynamic=True).
        if text_pipe.text_encoder is not None:
            text_pipe.text_encoder = torch.compile(text_pipe.text_encoder, dynamic=False)
        if text_pipe.text_encoder_2 is not None:
            text_pipe.text_encoder_2 = torch.compile(text_pipe.text_encoder_2, dynamic=False)
        if text_pipe.text_encoder_3 is not None:
            text_pipe.text_encoder_3 = torch.compile(text_pipe.text_encoder_3, dynamic=True)
        # Pre-compute null (unconditional) embeddings for CFG dropout — done once, reused every step.
        if args.caption_dropout > 0:
            with torch.no_grad(), _silence_encoding_noise():
                null_enc_hs, _, null_pooled, _ = text_pipe.encode_prompt(
                    prompt=[""], prompt_2=[""], prompt_3=[""],
                )
            null_enc_hs = null_enc_hs.to(device=device, dtype=torch.bfloat16)
            null_pooled = null_pooled.to(device=device, dtype=torch.bfloat16)
        else:
            null_enc_hs = null_pooled = None
    else:
        text_pipe = None
        null_enc_hs = null_pooled = None

    # --- Model ---
    num_layers = 24  # SD3.5 Medium
    latent_channels = 16

    if resume_dir is not None:
        print(f"[rank {rank}] Resuming from {resume_dir} ...")
        model = PomSD3Transformer2DModel.from_pretrained(resume_dir).to(
            device=device, dtype=torch.bfloat16
        )
        num_layers = model.config.num_layers
    elif init_dir is not None:
        print(f"[rank {rank}] Loading weights from {init_dir} (new training phase) ...")
        model = PomSD3Transformer2DModel.from_pretrained(init_dir).to(
            device=device, dtype=torch.bfloat16
        )
        num_layers = model.config.num_layers
    elif not args.smoke_test:
        print(f"[rank {rank}] Initializing PoM model from scratch (all 24 blocks random) ...")
        model = PomSD3Transformer2DModel(
            **SD35_MEDIUM_CONFIG,
            n_pom_blocks=24,
            pom_degree=args.pom_degree,
            pom_expand=args.pom_expand,
            pom_n_groups=args.pom_n_groups,
            pom_n_sel_heads=args.pom_n_sel_heads,
            lora_rank=args.lora_rank,
            pom_rope_max_seq_len=args.pom_rope_max_seq_len,
            hybrid_n=args.hybrid_n,
            attention_window_m=args.attention_window_m,
        ).to(device=device, dtype=torch.bfloat16)
    else:
        model = PomSD3Transformer2DModel(
            sample_size=32, patch_size=2, in_channels=16, num_layers=4,
            attention_head_dim=16, num_attention_heads=4,
            joint_attention_dim=4096, caption_projection_dim=64,
            pooled_projection_dim=2048, out_channels=16,
            pos_embed_max_size=32, dual_attention_layers=(0,),
            pom_degree=2, pom_expand=2, pom_n_groups=1, pom_n_sel_heads=1,
            lora_rank=0, n_pom_blocks=4,
            hybrid_n=args.hybrid_n,
            attention_window_m=args.attention_window_m,
        ).to(device=device, dtype=torch.bfloat16)
        num_layers = 4

    if is_main():
        print_model_summary(model, label="all blocks PoM, all trainable")
        print_model_layers(model)
        import io as _io
        _buf = _io.StringIO()
        _saved = sys.stdout; sys.stdout = _buf
        print_model_summary(model, label="all blocks PoM, all trainable")
        print_model_layers(model)
        sys.stdout = _saved
        wandb.log({"model_structure": wandb.Html(f"<pre>{_buf.getvalue()}</pre>")}, step=0)
        model_cfg = dict(model.config)
        print("Model config:", json.dumps(model_cfg, indent=2, default=str))
        wandb.config.update({"model_" + k: v for k, v in model_cfg.items()}, allow_val_change=True)

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # FSDP wrapping shards params, grads, and optimizer states across ranks.
    # Must happen before optimizer creation so the optimizer sees sharded params.
    model = wrap_model_fsdp(model, local_rank)
    model.train()

    # --- Optimizer (created on sharded params when FSDP is active) ---
    all_params = list(model.parameters())
    optimizer = torch.optim.AdamW(
        all_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999),
    )

    # --- Restore from checkpoint ---
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

    # --- Dataset & DataLoader ---
    latent_size = args.image_size // 8  # VAE spatial downsampling factor

    if args.smoke_test:
        # Dummy iterable for smoke test — no HF dataset needed
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

        # Append crop info string to captions (dropped with probability crop_str_dropout)
        crop_strs = batch.get("crop_str")
        if crop_strs is not None and args.crop_str_dropout < 1.0:
            captions = [
                cap + " " + cs if random.random() > args.crop_str_dropout else cap
                for cap, cs in zip(captions, crop_strs)
            ]

        # --- Encode image → latent x_0 ---
        if vae is not None:
            with torch.no_grad():
                latents = vae.encode(
                    pixel_values.to(device=device, dtype=torch.bfloat16)
                ).latent_dist.sample()
                x_0 = (latents - vae.config.shift_factor) * vae.config.scaling_factor
        else:
            # smoke test: random latents
            x_0 = torch.randn(B, latent_channels, latent_size, latent_size,
                              device=device, dtype=torch.bfloat16)

        # --- Encode captions → text embeddings ---
        if text_pipe is not None:
            with torch.no_grad(), _silence_encoding_noise():
                enc_hs, _, pooled, _ = text_pipe.encode_prompt(
                    prompt=captions, prompt_2=captions, prompt_3=captions,
                )
            enc_hs = enc_hs.to(dtype=torch.bfloat16)
            pooled = pooled.to(dtype=torch.bfloat16)
            # CFG conditioning dropout: randomly replace captions with null embedding so
            # the model learns the unconditional score (required for guidance_scale > 1).
            if args.caption_dropout > 0 and null_enc_hs is not None:
                drop = torch.rand(B, device=device) < args.caption_dropout
                if drop.any():
                    n = int(drop.sum())
                    enc_hs[drop] = null_enc_hs.expand(n, -1, -1)
                    pooled[drop] = null_pooled.expand(n, -1)
        else:
            # smoke test: random embeddings
            enc_hs = torch.randn(B, 8, 4096, device=device, dtype=torch.bfloat16)
            pooled = torch.randn(B, 2048, device=device, dtype=torch.bfloat16)

        # --- Flow matching loss ---
        # Logit-normal timestep sampling with resolution-adjusted mean.
        # mean=0: standard SD3 (concentrates at t≈500).
        # mean>0: shifts towards high t (global structure); used at lower resolutions.
        u = torch.sigmoid(torch.randn(B, device=device) + args.logit_normal_mean)
        t = (u * 999).clamp(1, 999).long()
        sigma = (t.float() / 1000).view(B, 1, 1, 1)
        eps = torch.randn_like(x_0)
        x_t = ((1 - sigma) * x_0 + sigma * eps).to(x_0.dtype)

        # Suppress FSDP gradient reduce-scatter on all but the last micro-batch so
        # that the collective fires once per optimizer step, not once per backward.
        is_last_accum = (step + 1) % args.grad_accum_steps == 0
        sync_ctx = (
            contextlib.nullcontext()
            if not isinstance(model, FSDP) or is_last_accum
            else model.no_sync()
        )
        with sync_ctx:
            v_pred = model(
                hidden_states=x_t,
                encoder_hidden_states=enc_hs,
                pooled_projections=pooled,
                timestep=t,
            ).sample
            v_target = (eps - x_0).to(v_pred.dtype)
            loss_per = F.mse_loss(v_pred, v_target, reduction="none").mean(dim=(1, 2, 3))
            loss = loss_per.mean()
            (loss / args.grad_accum_steps).backward()

        if is_last_accum:
            # FSDP handles gradient synchronisation internally; clip using its
            # all-ranks norm computation.  Plain model falls back to the standard call.
            if isinstance(model, FSDP):
                model.clip_grad_norm_(1.0)
            else:
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # --- Logging ---
        if is_main() and step % args.log_every == 0:
            elapsed = time.time() - t0
            sps = (step + 1 - start_step) * args.batch_size * world_size / elapsed
            t_cpu = t.cpu().float()
            lp = loss_per.detach().cpu()
            low  = t_cpu < 334
            mid  = (t_cpu >= 334) & (t_cpu < 667)
            high = t_cpu >= 667
            log = {
                "loss":        loss.item(),
                "loss_low_t":  lp[low].mean().item()  if low.any()  else float("nan"),
                "loss_mid_t":  lp[mid].mean().item()  if mid.any()  else float("nan"),
                "loss_high_t": lp[high].mean().item() if high.any() else float("nan"),
                "lr": lr,
                "step": step,
                "samples_per_sec": sps,
            }
            wandb.log(log, step=step)
            print(f"step={step:7d}  loss={log['loss']:.4f}  lr={lr:.2e}  {sps:.1f} samp/s")

        # --- Checkpointing ---
        # All ranks must participate when FSDP is active (collective state_dict ops).
        if step > 0 and step % args.save_every == 0:
            ckpt_dir = out_dir / f"step_{step:07d}"
            save_checkpoint(model, optimizer, step, ckpt_dir)
            if is_main():
                print(f"Saved checkpoint to {ckpt_dir}")

        # --- Sample generation ---
        if step > 0 and step % args.sample_every == 0 and not args.smoke_test:
            model.eval()
            generate_samples(model, vae, text_pipe, step, device, args.num_sample_prompts,
                             resolution=args.image_size)
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
            getattr(model, "_fsdp_wrapped_module", model).save_config(final_dir)
        dist.barrier()
    else:
        if is_main():
            model.save_pretrained(final_dir)
    if is_main():
        print(f"Training complete. Model saved to {final_dir}")
        wandb.finish()

    cleanup_ddp()


if __name__ == "__main__":
    main()
