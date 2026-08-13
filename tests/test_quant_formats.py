"""Unit tests for the quantized-checkpoint support: header routing
(_safetensors_unsupported / _safetensors_dequant), the ComfyUI FP8/INT8 dequant
loader (_load_dequant_state_dict) and the GGUF guards (_gguf_arch /
_gguf_layout_unsupported). Synthetic files only (a few KB), no model download.

Run:  .venv/Scripts/python tests/test_quant_formats.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import cz_pipeline  # noqa: E402

TMP = tempfile.mkdtemp(prefix="cz_quant_test_")


def _st(name, tensors):
    p = os.path.join(TMP, name)
    save_file(tensors, p)
    return p


# Cles minimales "Z-Image" (marqueur feed_forward) au format ComfyUI.
_W = "model.diffusion_model.layers.0.feed_forward.w1.weight"


def test_bf16_passthrough():
    p = _st("bf16.safetensors", {_W: torch.randn(4, 4, dtype=torch.bfloat16)})
    assert cz_pipeline._safetensors_unsupported(p) is None
    assert cz_pipeline._safetensors_dequant(p) is None


def test_fp8_pure_detect_and_dequant():
    w = torch.randn(4, 4).to(torch.float8_e4m3fn)
    p = _st("fp8.safetensors", {_W: w})
    assert cz_pipeline._safetensors_dequant(p) == "FP8"
    sd = cz_pipeline._load_dequant_state_dict(p)
    assert sd[_W].dtype == cz_pipeline.DTYPE
    assert torch.allclose(sd[_W].float(), w.float().to(cz_pipeline.DTYPE).float())


def test_fp8_scaled_dequant_math():
    w = torch.randn(4, 4).to(torch.float8_e4m3fn)
    scale = torch.tensor(2.5, dtype=torch.float32)
    p = _st("fp8s.safetensors", {
        _W: w, _W + "_scale": scale,
        _W.replace(".weight", ".comfy_quant"): torch.zeros(27, dtype=torch.uint8),
    })
    assert cz_pipeline._safetensors_dequant(p) == "FP8 scaled"
    sd = cz_pipeline._load_dequant_state_dict(p)
    want = (w.to(torch.float32) * 2.5).to(cz_pipeline.DTYPE)
    assert torch.allclose(sd[_W].float(), want.float())
    # metadonnees consommees, jamais dans le dict final
    assert _W + "_scale" not in sd
    assert _W.replace(".weight", ".comfy_quant") not in sd


def test_int8_per_row_scale():
    w = torch.randint(-127, 127, (4, 3), dtype=torch.int8)
    scale = torch.rand(4, 1, dtype=torch.float32)
    p = _st("int8.safetensors", {_W: w, _W + "_scale": scale})
    assert cz_pipeline._safetensors_dequant(p) == "INT8 scaled"
    sd = cz_pipeline._load_dequant_state_dict(p)
    want = (w.to(torch.float32) * scale).to(cz_pipeline.DTYPE)
    assert torch.allclose(sd[_W].float(), want.float())


def test_aio_bundle_filtered():
    # bundle: transformer BF16 + "encodeur texte" FP8 -> seul le transformer reste
    p = _st("aio.safetensors", {
        _W: torch.randn(4, 4, dtype=torch.bfloat16),
        "text_encoders.qwen3_4b.layers.0.weight": torch.randn(4, 4).to(torch.float8_e4m3fn),
        "vae.decoder.weight": torch.randn(2, 2, dtype=torch.bfloat16),
    })
    assert cz_pipeline._safetensors_dequant(p) == "FP8"
    sd = cz_pipeline._load_dequant_state_dict(p)
    assert list(sd) == [_W]


def test_foreign_arch_rejected():
    w = torch.randn(4, 4).to(torch.float8_e4m3fn)
    p = _st("ernie.safetensors",
            {"model.diffusion_model.layers.0.mlp.gate_proj.weight": w})
    raised = False
    try:
        cz_pipeline._load_dequant_state_dict(p)
    except RuntimeError as e:
        raised = "Z-Image" in str(e)
    assert raised, "checkpoint quantifie d'une autre archi doit etre refuse clairement"


def test_lora_and_svdq_still_unsupported():
    lora = {f"lora_unet_a{i}.lora_down.weight": torch.zeros(2, 2) for i in range(4)}
    p = _st("lora.safetensors", lora)
    assert "LoRA" in (cz_pipeline._safetensors_unsupported(p) or "")
    p = _st("svdq.safetensors", {"blocks.0.qweight": torch.zeros(2, 2, dtype=torch.int8)})
    assert "SVDQuant" in (cz_pipeline._safetensors_unsupported(p) or "")


def _gguf(name, arch, tensor_name):
    import numpy as np
    from gguf import GGUFWriter
    p = os.path.join(TMP, name)
    w = GGUFWriter(p, arch)
    w.add_tensor(tensor_name, np.zeros((4, 32), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return p


def test_gguf_arch_and_layout():
    ok = _gguf("zimage.gguf", "lumina2", "layers.0.attention.qkv.weight")
    assert cz_pipeline._gguf_arch(ok) == "lumina2"
    assert cz_pipeline._gguf_layout_unsupported(ok) is None
    flux = _gguf("flux.gguf", "flux", "double_blocks.0.img_attn.qkv.weight")
    assert cz_pipeline._gguf_arch(flux) == "flux"
    sdcpp = _gguf("sdcpp.gguf", "lumina2", "blocks.0.attn.wq.weight")
    assert cz_pipeline._gguf_layout_unsupported(sdcpp) is not None


def test_gguf_listing_filter():
    d = os.path.join(TMP, "ckpts")
    os.makedirs(d, exist_ok=True)
    import shutil
    for src, dst in (("zimage.gguf", "good.gguf"), ("flux.gguf", "bad_arch.gguf"),
                     ("sdcpp.gguf", "bad_layout.gguf")):
        shutil.copy(os.path.join(TMP, src), os.path.join(d, dst))
    old = cz_pipeline.CHECKPOINTS_DIR
    try:
        cz_pipeline.CHECKPOINTS_DIR = d
        lst = cz_pipeline.list_checkpoints()
    finally:
        cz_pipeline.CHECKPOINTS_DIR = old
    assert "good.gguf" in lst and "bad_arch.gguf" not in lst and "bad_layout.gguf" not in lst


if __name__ == "__main__":
    for fn in (test_bf16_passthrough, test_fp8_pure_detect_and_dequant,
               test_fp8_scaled_dequant_math, test_int8_per_row_scale,
               test_aio_bundle_filtered, test_foreign_arch_rejected,
               test_lora_and_svdq_still_unsupported, test_gguf_arch_and_layout,
               test_gguf_listing_filter):
        fn()
        print(f"OK {fn.__name__}")
    print("All quant-format tests passed.")
