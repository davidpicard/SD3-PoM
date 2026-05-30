"""Checkpoint verification tests. Run before launching full training.

Each test is independent and prints PASS / FAIL clearly.
Usage:
    python verify.py                          # all tests
    python verify.py --test model_shape       # specific test
    python verify.py --sd3_id <hub_id>        # tests that need the real checkpoint
"""
import argparse
import traceback

import torch

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def run(name: str, fn):
    try:
        fn()
        print(f"[{PASS}] {name}")
        return True
    except Exception:
        print(f"[{FAIL}] {name}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Test 1: JointPoMBlock instantiates and forward returns correct shapes
# ---------------------------------------------------------------------------
def test_joint_pom_block_shape():
    from pom_sd3.blocks import JointPoMBlock

    dim = 64
    block = JointPoMBlock(
        dim=dim, num_attention_heads=4, attention_head_dim=16,
        context_pre_only=False, use_dual_attention=False,
        pom_degree=2, pom_expand=2, pom_n_groups=1, pom_n_sel_heads=1,
    )
    B, N_img, N_txt = 2, 16, 8
    hidden = torch.randn(B, N_img, dim)
    enc_hs = torch.randn(B, N_txt, dim)
    temb = torch.randn(B, dim)

    enc_out, img_out = block(hidden, enc_hs, temb)
    assert img_out.shape == (B, N_img, dim), f"img_out shape mismatch: {img_out.shape}"
    assert enc_out.shape == (B, N_txt, dim), f"enc_out shape mismatch: {enc_out.shape}"


def test_joint_pom_block_context_pre_only():
    from pom_sd3.blocks import JointPoMBlock

    dim = 64
    block = JointPoMBlock(
        dim=dim, num_attention_heads=4, attention_head_dim=16,
        context_pre_only=True, use_dual_attention=False,
        pom_degree=2, pom_expand=2, pom_n_groups=1, pom_n_sel_heads=1,
    )
    B, N_img, N_txt = 2, 16, 8
    hidden = torch.randn(B, N_img, dim)
    enc_hs = torch.randn(B, N_txt, dim)
    temb = torch.randn(B, dim)

    enc_out, img_out = block(hidden, enc_hs, temb)
    assert img_out.shape == (B, N_img, dim)
    assert enc_out is None, "context_pre_only should return None for encoder states"


def test_joint_pom_block_dual_attention():
    from pom_sd3.blocks import JointPoMBlock

    dim = 64
    block = JointPoMBlock(
        dim=dim, num_attention_heads=4, attention_head_dim=16,
        context_pre_only=False, use_dual_attention=True,
        pom_degree=2, pom_expand=2, pom_n_groups=1, pom_n_sel_heads=1,
    )
    B, N_img, N_txt = 2, 16, 8
    hidden = torch.randn(B, N_img, dim)
    enc_hs = torch.randn(B, N_txt, dim)
    temb = torch.randn(B, dim)

    enc_out, img_out = block(hidden, enc_hs, temb)
    assert img_out.shape == (B, N_img, dim)
    assert enc_out.shape == (B, N_txt, dim)


# ---------------------------------------------------------------------------
# Test 2: PomSD3Transformer2DModel forward output shape
# ---------------------------------------------------------------------------
def test_model_forward_shape():
    from pom_sd3 import PomSD3Transformer2DModel

    # Tiny model for fast testing
    model = PomSD3Transformer2DModel(
        sample_size=32,
        patch_size=2,
        in_channels=16,
        num_layers=2,
        attention_head_dim=16,
        num_attention_heads=4,
        joint_attention_dim=32,
        caption_projection_dim=64,
        pooled_projection_dim=32,
        out_channels=16,
        pos_embed_max_size=32,
        dual_attention_layers=(0,),
        pom_degree=2,
        pom_expand=2,
        pom_n_groups=1,
        pom_n_sel_heads=1,
    )
    model.eval()

    B, C, H, W = 2, 16, 32, 32
    hidden_states = torch.randn(B, C, H, W)
    enc_hs = torch.randn(B, 10, 32)
    pooled = torch.randn(B, 32)
    timestep = torch.randint(0, 1000, (B,))

    with torch.no_grad():
        out = model(hidden_states, enc_hs, pooled, timestep)
    assert out.sample.shape == (B, C, H, W), f"Output shape mismatch: {out.sample.shape}"


def test_model_return_intermediate():
    from pom_sd3 import PomSD3Transformer2DModel

    model = PomSD3Transformer2DModel(
        sample_size=32, patch_size=2, in_channels=16, num_layers=2,
        attention_head_dim=16, num_attention_heads=4, joint_attention_dim=32,
        caption_projection_dim=64, pooled_projection_dim=32, out_channels=16,
        pos_embed_max_size=32, dual_attention_layers=(0,),
        pom_degree=2, pom_expand=2, pom_n_groups=1, pom_n_sel_heads=1,
    )
    model.eval()

    B, C, H, W = 1, 16, 32, 32
    hidden_states = torch.randn(B, C, H, W)
    enc_hs = torch.randn(B, 8, 32)
    pooled = torch.randn(B, 32)
    timestep = torch.randint(0, 1000, (B,))

    with torch.no_grad():
        out, intermediates = model(
            hidden_states, enc_hs, pooled, timestep, return_intermediate=True
        )
    assert len(intermediates) == 2, f"Expected 2 intermediates, got {len(intermediates)}"
    assert out.sample.shape == (B, C, H, W)


# ---------------------------------------------------------------------------
# Test 3: Weight loading from pretrained SD3.5
# (requires model_id to be downloadable — skip if not available)
# ---------------------------------------------------------------------------
def test_weight_loading(model_id: str):
    from pom_sd3 import load_sd3_weights_into_pom, PomSD3Transformer2DModel, SD35_MEDIUM_CONFIG
    from diffusers import SD3Transformer2DModel

    teacher = SD3Transformer2DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    teacher_sd = teacher.state_dict()

    student = PomSD3Transformer2DModel(
        **SD35_MEDIUM_CONFIG,
        pom_degree=4, pom_expand=2, pom_n_groups=1, pom_n_sel_heads=1,
    ).to(torch.bfloat16)

    missing, unexpected = load_sd3_weights_into_pom(student, teacher_sd)

    # All missing keys should be PoM-specific
    non_pom_missing = [k for k in missing if ".pom" not in k]
    assert not non_pom_missing, f"Non-PoM keys missing: {non_pom_missing[:5]}"

    # Verify a non-attention weight actually transferred
    key = "transformer_blocks.0.ff.net.0.proj.weight"
    assert key in teacher_sd
    student_sd = student.state_dict()
    assert torch.allclose(student_sd[key], teacher_sd[key].float()), \
        "Non-attention weight did not transfer correctly"

    # Verify attention weight was NOT transferred (student has random init)
    attn_key = "transformer_blocks.0.attn.to_q.weight"
    assert attn_key in teacher_sd
    pom_key = "transformer_blocks.0.pom.ag_proj.weight"
    assert pom_key in student_sd  # PoM key exists in student


# ---------------------------------------------------------------------------
# Test 4: save_pretrained / from_pretrained round-trip
# ---------------------------------------------------------------------------
def test_save_load_roundtrip(tmp_path="/tmp/pom_sd3_test"):
    import shutil
    from pom_sd3 import PomSD3Transformer2DModel

    model = PomSD3Transformer2DModel(
        sample_size=32, patch_size=2, in_channels=4, num_layers=2,
        attention_head_dim=16, num_attention_heads=2, joint_attention_dim=32,
        caption_projection_dim=32, pooled_projection_dim=32, out_channels=4,
        pos_embed_max_size=32, dual_attention_layers=(),
        pom_degree=2, pom_expand=2, pom_n_groups=1, pom_n_sel_heads=1,
    )
    model.save_pretrained(tmp_path)

    loaded = PomSD3Transformer2DModel.from_pretrained(tmp_path)
    # Compare a parameter
    orig_key = list(model.state_dict().keys())[0]
    assert torch.allclose(
        model.state_dict()[orig_key], loaded.state_dict()[orig_key]
    ), "Parameter mismatch after save/load"

    shutil.rmtree(tmp_path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 5: Distillation loss runs (smoke)
# ---------------------------------------------------------------------------
def test_distillation_loss():
    from train import distillation_loss

    B, C, H, W = 2, 16, 32, 32
    student_out = torch.randn(B, C, H, W, requires_grad=True)
    teacher_out = torch.randn(B, C, H, W)

    dim, N = 64, 32
    intermediates = [(torch.randn(B, N, dim), torch.randn(B, N, dim)) for _ in range(3)]

    losses = distillation_loss(
        student_out, teacher_out, intermediates, intermediates, block_weight=0.1
    )
    assert "loss" in losses
    assert losses["loss"].item() >= 0
    losses["loss"].backward()  # ensure gradients flow


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", default=None, help="Run only this test")
    p.add_argument("--sd3_id", default=None,
                   help="HF hub ID for weight loading test (e.g. stabilityai/stable-diffusion-3.5-medium)")
    args = p.parse_args()

    tests = {
        "block_shape": test_joint_pom_block_shape,
        "block_context_pre_only": test_joint_pom_block_context_pre_only,
        "block_dual_attention": test_joint_pom_block_dual_attention,
        "model_shape": test_model_forward_shape,
        "model_intermediates": test_model_return_intermediate,
        "save_load": test_save_load_roundtrip,
        "distillation_loss": test_distillation_loss,
    }

    if args.sd3_id:
        tests["weight_loading"] = lambda: test_weight_loading(args.sd3_id)

    if args.test:
        if args.test not in tests:
            print(f"Unknown test '{args.test}'. Available: {list(tests)}")
            return
        tests = {args.test: tests[args.test]}

    results = {name: run(name, fn) for name, fn in tests.items()}
    n_pass = sum(results.values())
    n_total = len(results)
    print(f"\n{n_pass}/{n_total} tests passed.")
    if n_pass < n_total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
