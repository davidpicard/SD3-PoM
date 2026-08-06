#!/usr/bin/env python3
"""
Bit-perfect comparison of encode_prompt vs fast_encode_prompt.

Verifies that the parallel CUDA-stream implementation produces identical
outputs to the sequential diffusers baseline across varied caption lengths.

Usage:
    python check_encoding.py --model_id ./shared/models/sd3.5-medium
    python check_encoding.py --model_id ./shared/models/sd3.5-medium --max_seq 256 --device cuda:1
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import StableDiffusion3Pipeline

TEST_CAPTIONS = [
    # empty (null conditioning)
    "",
    # short
    "a cat",
    # medium
    "a photorealistic portrait of a woman with red hair, golden hour lighting",
    # long — approaching T5 truncation limit at seq=77
    "an extremely detailed oil painting of a medieval castle perched on a rocky cliff "
    "overlooking a turbulent sea during a violent thunderstorm, with lightning striking "
    "the tallest tower, dark dramatic clouds swirling overhead, waves crashing on jagged "
    "rocks below, torches flickering in the windows, by J.M.W Turner style",
    # crop conditioning string (used during training)
    "target_size:(512, 512), original_size:(512, 512), crop_coords:(0, 0), a dog sitting in a park",
    # unicode / multilingual
    "une femme élégante marchant dans les rues de Paris au coucher du soleil",
]


@torch.no_grad()
def fast_encode_prompt(text_pipe, captions, max_sequence_length, device):
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

    stream_t5   = torch.cuda.Stream(device=device)
    stream_clip = torch.cuda.Stream(device=device)

    with torch.cuda.stream(stream_t5):
        t5_tok = text_pipe.text_encoder_3(input_ids=t5_ids).last_hidden_state

    with torch.cuda.stream(stream_clip):
        out1       = text_pipe.text_encoder(input_ids=clip_ids_1, output_hidden_states=True)
        clip1_tok  = out1.hidden_states[-2]
        clip1_pool = out1.text_embeds

        out2       = text_pipe.text_encoder_2(input_ids=clip_ids_2, output_hidden_states=True)
        clip2_tok  = out2.hidden_states[-2]
        clip2_pool = out2.text_embeds

    curr = torch.cuda.current_stream(device=device)
    curr.wait_stream(stream_t5)
    curr.wait_stream(stream_clip)

    clip_tok = torch.cat([clip1_tok, clip2_tok], dim=-1)
    clip_tok = F.pad(clip_tok, (0, t5_tok.shape[-1] - clip_tok.shape[-1]))
    enc_hs   = torch.cat([clip_tok, t5_tok], dim=1)
    pooled   = torch.cat([clip1_pool, clip2_pool], dim=-1)

    return enc_hs.to(dtype=torch.bfloat16), pooled.to(dtype=torch.bfloat16)


def compare(name, a, b):
    """Print comparison stats. Returns True if tensors are bit-exact."""
    equal = torch.equal(a, b)
    diff  = (a.float() - b.float()).abs()
    max_d = diff.max().item()
    mean_d = diff.mean().item()
    # BF16 machine epsilon is ~3.9e-3 (2^-7 relative); 1 ULP in BF16 ≈ 0.004 for values ~1
    status = "PASS" if equal else ("CLOSE" if max_d < 1e-2 else "FAIL")
    print(f"  {name:10s}  exact={str(equal):5s}  max_abs={max_d:.2e}  mean_abs={mean_d:.2e}  [{status}]")
    return equal, max_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--max_seq", type=int, default=77)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    local = Path(args.model_id).exists()

    print(f"Loading text encoders from {args.model_id} ...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_id,
        transformer=None,
        vae=None,
        torch_dtype=torch.bfloat16,
        local_files_only=local,
    ).to(device)
    for enc in (pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
        if enc is not None:
            enc.requires_grad_(False).eval()
    print("Encoders loaded.\n")

    all_exact = True
    worst_max = 0.0

    for caption in TEST_CAPTIONS:
        label = f'"{caption[:40]}{"…" if len(caption) > 40 else ""}"'
        print(f"Caption: {label}")

        # --- reference: sequential encode_prompt ---
        with torch.no_grad():
            ref_enc_hs, _, ref_pooled, _ = pipe.encode_prompt(
                prompt=caption,
                prompt_2=caption,
                prompt_3=caption,
                device=device,
                do_classifier_free_guidance=False,
                max_sequence_length=args.max_seq,
            )

        # --- fast: parallel streams ---
        fast_enc_hs, fast_pooled = fast_encode_prompt(
            pipe, [caption], args.max_seq, device,
        )

        exact_enc, max_enc   = compare("enc_hs",  ref_enc_hs,  fast_enc_hs)
        exact_pool, max_pool = compare("pooled",   ref_pooled,  fast_pooled)

        if not (exact_enc and exact_pool):
            all_exact = False
        worst_max = max(worst_max, max_enc, max_pool)
        print()

    print("=" * 60)
    if all_exact:
        print("RESULT: BIT-EXACT MATCH on all captions.")
    else:
        bf16_eps = 2 ** -7  # ~7.8e-3, one ULP in BF16 for values around 1
        if worst_max < bf16_eps:
            print(f"RESULT: Not bit-exact, but within BF16 precision (max_diff={worst_max:.2e} < {bf16_eps:.2e}).")
        else:
            print(f"RESULT: MISMATCH — max_diff={worst_max:.2e} exceeds BF16 precision. Check implementation.")
            sys.exit(1)


if __name__ == "__main__":
    main()
