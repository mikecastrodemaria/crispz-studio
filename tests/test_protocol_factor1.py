"""upscale op: factor 1 = pure img2img (NO ESRGAN stage), and the default
ESRGAN model follows the factor's scale (never a 16x by accident).
Run: .venv/Scripts/python tests/test_protocol_factor1.py"""
import os
import sys
import types
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_protocol as cp  # noqa: E402


def test_pick_esrgan_follows_factor():
    models = ["16xPSNR.pth", "1x-SuperScale.safetensors", "2x-AnimeSharp.safetensors",
              "4x-UltraSharp.pth"]
    assert cp._pick_esrgan(models, 2.0) == "2x-AnimeSharp.safetensors"
    assert cp._pick_esrgan(models, 4.0) == "4x-UltraSharp.pth"
    assert cp._pick_esrgan(models, 3.0) == "4x-UltraSharp.pth"       # 3 -> 4x
    assert cp._pick_esrgan(["16xPSNR.pth"], 2.0) == "16xPSNR.pth"     # last resort
    assert cp._pick_esrgan([], 2.0) is None


def test_factor1_skips_esrgan():
    from PIL import Image
    calls = {}
    fake_pipe = types.SimpleNamespace(
        process_one=lambda img, model, factor, denoise, steps, prompt, seed, tile,
        overlap, **kw: (calls.update(model=model, factor=factor, kw=kw) or (img, {})))
    fake_esrgan = types.SimpleNamespace(
        list_esrgan_models=lambda: ["16xPSNR.pth", "4x-UltraSharp.pth"],
        ESRGAN_DIR="x")
    sys.modules["cz_pipeline"], old_p = fake_pipe, sys.modules.get("cz_pipeline")
    sys.modules["cz_esrgan"], old_e = fake_esrgan, sys.modules.get("cz_esrgan")
    d = tempfile.mkdtemp(prefix="cz_f1_")
    try:
        src = os.path.join(d, "in.png")
        Image.new("RGB", (64, 64), "white").save(src)
        res = cp.run_upscale({"input": src, "factor": 1.0, "denoise": 0.4,
                              "out_dir": d, "steps": 2})
        assert res["ok"] and res["esrgan_model"] is None
        assert calls["kw"]["do_esrgan"] is False and calls["model"] is None
        res = cp.run_upscale({"input": src, "factor": 4.0, "out_dir": d, "steps": 2})
        assert calls["kw"]["do_esrgan"] is True
        assert calls["model"] == "4x-UltraSharp.pth" and res["esrgan_model"] == calls["model"]
    finally:
        for k, v in (("cz_pipeline", old_p), ("cz_esrgan", old_e)):
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


if __name__ == "__main__":
    test_pick_esrgan_follows_factor()
    print("OK test_pick_esrgan_follows_factor")
    test_factor1_skips_esrgan()
    print("OK test_factor1_skips_esrgan")
    print("All 2 factor-1 tests passed.")
