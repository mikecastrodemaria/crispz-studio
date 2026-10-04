"""Unit tests for the LoRA state_dict repair (alpha folding + fused attention remap).

Regression guard for a Z-Image LoRA from an external trainer (fused `attention.qkv`, bare
`attention.out`, `lora_A`/`lora_B` suffixes + `.alpha`), which diffusers 0.39.dev refused
with:
    ValueError: `state_dict` should be empty at this point but has
                dict_keys(['layers.0.attention.to_out.0.alpha', ...])
and which, once that error was worked around by folding the alphas alone, loaded WITHOUT
applying anything to the attention (`attention.qkv` / `attention.out` match no module in
the diffusers model -- only feed_forward and adaLN landed).

No model is loaded: the state dicts are synthetic and tiny.

Run:  .venv/Scripts/python tests/test_lora_alpha.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import cz_pipeline as P  # noqa: E402

DIM, RANK = 16, 4


def _trainer_sd(n_layers=2, alpha=8.0):
    """The layout of the faulty file: 'diffusion_model.' prefix, lora_A/lora_B, one alpha
    per module, a FUSED qkv (3 x DIM out) and a bare 'out'."""
    sd = {}
    for i in range(n_layers):
        pre = f"diffusion_model.layers.{i}"
        for mod, out_dim in (("attention.qkv", 3 * DIM), ("attention.out", DIM),
                             ("feed_forward.w1", DIM)):
            sd[f"{pre}.{mod}.lora_A.weight"] = torch.randn(RANK, DIM)
            sd[f"{pre}.{mod}.lora_B.weight"] = torch.randn(out_dim, RANK)
            sd[f"{pre}.{mod}.alpha"] = torch.tensor(alpha)
    return sd


def test_fold_alpha_scales_the_up_weight_and_removes_the_key():
    sd = _trainer_sd(n_layers=1, alpha=8.0)
    up_before = sd["diffusion_model.layers.0.attention.out.lora_B.weight"].clone()
    out, folded = P.fold_lora_alpha(sd)
    assert folded == 3, folded
    assert not [k for k in out if k.endswith(".alpha")], "the alphas must be gone"
    # alpha 8 / rank 4 -> x2 on the up weight, the down weight untouched.
    up_after = out["diffusion_model.layers.0.attention.out.lora_B.weight"]
    assert torch.allclose(up_after, up_before * 2.0), "alpha/rank not applied"
    assert torch.equal(out["diffusion_model.layers.0.attention.out.lora_A.weight"],
                       sd["diffusion_model.layers.0.attention.out.lora_A.weight"])


def test_fold_alpha_keeps_the_product_identical():
    """What reaches the model is B @ A: folding must not change it (alpha used to be
    applied at runtime by peft, it is in the weights now)."""
    sd = _trainer_sd(n_layers=1, alpha=2.0)
    a = sd["diffusion_model.layers.0.feed_forward.w1.lora_A.weight"]
    b = sd["diffusion_model.layers.0.feed_forward.w1.lora_B.weight"]
    expected = (b @ a) * (2.0 / RANK)                 # peft: (B @ A) * alpha/rank
    out, _ = P.fold_lora_alpha(sd)
    got = (out["diffusion_model.layers.0.feed_forward.w1.lora_B.weight"]
           @ out["diffusion_model.layers.0.feed_forward.w1.lora_A.weight"])
    assert torch.allclose(got, expected, atol=1e-6)


def test_fold_alpha_drops_an_orphan_alpha():
    sd = {"layers.0.attention.out.alpha": torch.tensor(4.0)}   # no weights
    out, folded = P.fold_lora_alpha(sd)
    assert folded == 0 and out == {}


def test_remap_renames_out_and_splits_qkv():
    sd, _ = P.fold_lora_alpha(_trainer_sd(n_layers=1))
    out, n_out, n_qkv = P._remap_fused_attention(sd)
    assert (n_out, n_qkv) == (2, 2), (n_out, n_qkv)   # out: A+B, qkv: A+B
    assert "diffusion_model.layers.0.attention.to_out.0.lora_A.weight" in out
    assert not [k for k in out if ".attention.out." in k or ".attention.qkv." in k]
    for name in ("to_q", "to_k", "to_v"):
        got_a = out[f"diffusion_model.layers.0.attention.{name}.lora_A.weight"]
        got_b = out[f"diffusion_model.layers.0.attention.{name}.lora_B.weight"]
        assert tuple(got_a.shape) == (RANK, DIM)
        assert tuple(got_b.shape) == (DIM, RANK)


def test_qkv_split_is_mathematically_exact():
    """The fused projection writes [q | k | v] along dim 0 (diffusers splits the BASE
    checkpoint with torch.chunk(..., 3, dim=0)): each third must get the same down weight
    and its own slice of the up weight."""
    sd = _trainer_sd(n_layers=1, alpha=float(RANK))   # alpha == rank -> scale 1.0
    a = sd["diffusion_model.layers.0.attention.qkv.lora_A.weight"]
    b = sd["diffusion_model.layers.0.attention.qkv.lora_B.weight"]
    fused = b @ a                                      # [3*DIM, DIM]
    folded, _ = P.fold_lora_alpha(sd)
    out, _, _ = P._remap_fused_attention(folded)
    for idx, name in enumerate(("to_q", "to_k", "to_v")):
        got = (out[f"diffusion_model.layers.0.attention.{name}.lora_B.weight"]
               @ out[f"diffusion_model.layers.0.attention.{name}.lora_A.weight"])
        assert torch.allclose(got, fused[idx * DIM:(idx + 1) * DIM], atol=1e-6), name


def test_needs_repair_only_for_the_faulty_layout():
    assert P._lora_needs_repair(_trainer_sd(n_layers=1)) is True
    healthy = {"transformer.layers.0.attention.to_q.lora_A.weight": torch.zeros(RANK, DIM),
               "transformer.layers.0.attention.to_q.lora_B.weight": torch.zeros(DIM, RANK)}
    assert P._lora_needs_repair(healthy) is False
    # The lora_down/lora_up form is the one diffusers' converter already handles.
    down_up = {"layers.0.attention.out.lora_down.weight": torch.zeros(RANK, DIM),
               "layers.0.attention.out.lora_up.weight": torch.zeros(DIM, RANK),
               "layers.0.attention.out.alpha": torch.tensor(4.0)}
    assert P._lora_needs_repair(down_up) is False


def test_lora_source_leaves_a_healthy_file_on_the_tested_path():
    """A file diffusers handles must KEEP the folder + weight_name route (offline mode
    refuses a full path): the repair must not change what already works."""
    d = tempfile.mkdtemp()
    p = os.path.join(d, "healthy.safetensors")
    save_file({"transformer.layers.0.attention.to_q.lora_A.weight": torch.zeros(RANK, DIM),
               "transformer.layers.0.attention.to_q.lora_B.weight": torch.zeros(DIM, RANK)}, p)
    src, kw = P._lora_source(p)
    assert src == d and kw == {"weight_name": "healthy.safetensors"}
    # ... while the faulty one comes back as a repaired dict.
    q = os.path.join(d, "faulty.safetensors")
    save_file(_trainer_sd(n_layers=1), q)
    src, kw = P._lora_source(q)
    assert isinstance(src, dict) and kw == {}
    assert not [k for k in src if k.endswith(".alpha")]


def test_unreadable_file_falls_back_to_the_path():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "not-a-safetensors.safetensors")
    with open(p, "wb") as f:
        f.write(b"garbage")
    src, kw = P._lora_source(p)                       # must not raise
    assert src == d and kw == {"weight_name": os.path.basename(p)}


def test_repaired_dict_passes_the_diffusers_converter():
    """The heart of it: the faulty layout used to raise ValueError('state_dict should be
    empty ... to_out.0.alpha') in diffusers' Z-Image converter, and the repair must leave
    the attention keys ON THE MODEL'S names."""
    from diffusers import ZImagePipeline
    sd = _trainer_sd(n_layers=2)
    try:
        ZImagePipeline.lora_state_dict(dict(sd))
    except ValueError as e:
        assert "should be empty" in str(e), e         # the bug, still reproducible
    else:
        raise AssertionError("diffusers no longer fails: this repair may be obsolete")
    folded, _ = P.fold_lora_alpha(sd)
    fixed, _, _ = P._remap_fused_attention(folded)
    out = ZImagePipeline.lora_state_dict(fixed)
    mods = {k.split(".lora_")[0] for k in out}
    assert any(m.endswith("attention.to_q") for m in mods), mods
    assert any(m.endswith("attention.to_out.0") for m in mods), mods
    assert not [m for m in mods if m.endswith(("attention.qkv", "attention.out"))], mods


def test_meta_params_detects_a_meta_module():
    assert P._meta_params(torch.nn.Linear(4, 4)) == []
    with torch.device("meta"):
        ghost = torch.nn.Linear(4, 4)
    found = P._meta_params(ghost)
    assert "weight" in found, found
    assert P._meta_params(None) == []


if __name__ == "__main__":
    for fn in (test_fold_alpha_scales_the_up_weight_and_removes_the_key,
               test_fold_alpha_keeps_the_product_identical,
               test_fold_alpha_drops_an_orphan_alpha,
               test_remap_renames_out_and_splits_qkv,
               test_qkv_split_is_mathematically_exact,
               test_needs_repair_only_for_the_faulty_layout,
               test_lora_source_leaves_a_healthy_file_on_the_tested_path,
               test_unreadable_file_falls_back_to_the_path,
               test_repaired_dict_passes_the_diffusers_converter,
               test_meta_params_detects_a_meta_module):
        fn()
        print(f"OK {fn.__name__}")
    print("All LoRA alpha/remap tests passed.")
