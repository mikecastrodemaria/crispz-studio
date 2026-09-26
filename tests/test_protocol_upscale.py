"""Tests for the protocol's 'upscale' op (ESRGAN + refine through process_one).
Fake cz_pipeline / cz_esrgan, no GPU.
Run:  .venv/Scripts/python tests/test_protocol_upscale.py"""
import os
import sys
import types
import shutil
import tempfile
import contextlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

import cz_protocol as cp  # noqa: E402


class _FakePipe(types.ModuleType):
    def __init__(self):
        super().__init__("cz_pipeline")
        self.calls = []

    def process_one(self, img, model, factor, denoise, steps, prompt, seed,
                    tile, overlap, **kw):
        self.calls.append({"model": model, "factor": factor,
                           "denoise": denoise, "prompt": prompt})
        w, h = img.size
        return img.resize((int(w * factor), int(h * factor))), {"esrgan": 1.0}


class _FakeEsrgan(types.ModuleType):
    ESRGAN_DIR = "X:/models"
    MODELS = ["4x-clear.pth", "4x-soft.pth"]

    def __init__(self):
        super().__init__("cz_esrgan")

    def list_esrgan_models(self):
        return list(self.MODELS)


@contextlib.contextmanager
def _fakes(models=None):
    pipe, esr = _FakePipe(), _FakeEsrgan()
    if models is not None:
        esr.MODELS = models
    old = {k: sys.modules.get(k) for k in ("cz_pipeline", "cz_esrgan")}
    sys.modules["cz_pipeline"] = pipe
    sys.modules["cz_esrgan"] = esr
    try:
        yield pipe
    finally:
        for k, v in old.items():
            if v is not None:
                sys.modules[k] = v
            else:
                sys.modules.pop(k, None)


def test_upscale_runs_and_reports_size():
    d = tempfile.mkdtemp(prefix="cz_up_")
    try:
        src = os.path.join(d, "in.png")
        Image.new("RGB", (100, 80), "#333").save(src)
        with _fakes() as pipe:
            spec, w = cp.validate_spec(
                {"protocol": 1, "input": src, "factor": 2,
                 "prompt": "a face, portrait", "out_dir": d}, op="upscale")
            res = cp.run_upscale(spec, w)
        assert res["ok"] and res["size"] == [200, 160]
        assert os.path.isfile(res["images"][0])
        assert res["esrgan_model"] == "4x-clear.pth"
        assert pipe.calls[0]["factor"] == 2 and \
            pipe.calls[0]["prompt"] == "a face, portrait"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_upscale_validation():
    try:
        cp.validate_spec({"protocol": 1}, op="upscale")
    except cp.SpecError as e:
        assert "input" in str(e)
    else:
        raise AssertionError("missing input should raise")
    try:
        cp.validate_spec({"protocol": 1, "input": "X:/nope.png"},
                         op="upscale")
    except cp.SpecError as e:
        assert "not found" in str(e)
    else:
        raise AssertionError("missing file should raise")
    d = tempfile.mkdtemp(prefix="cz_upv_")
    try:
        src = os.path.join(d, "in.png")
        Image.new("RGB", (10, 10)).save(src)
        for bad in ({"factor": 12}, {"denoise": 3}):
            try:
                cp.validate_spec({"protocol": 1, "input": src, **bad},
                                 op="upscale")
            except cp.SpecError:
                pass
            else:
                raise AssertionError(f"{bad} should raise")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_unknown_esrgan_model_warns_and_falls_back():
    d = tempfile.mkdtemp(prefix="cz_upm_")
    try:
        src = os.path.join(d, "in.png")
        Image.new("RGB", (10, 10)).save(src)
        with _fakes() as _pipe:
            spec, w = cp.validate_spec(
                {"protocol": 1, "input": src, "model": "ghost.pth",
                 "out_dir": d}, op="upscale")
            res = cp.run_upscale(spec, w)
        assert res["ok"] and res["esrgan_model"] == "4x-clear.pth"
        assert any("ghost.pth" in x for x in res["warnings"])
        with _fakes(models=[]):
            spec, w = cp.validate_spec({"protocol": 1, "input": src,
                                        "out_dir": d}, op="upscale")
            try:
                cp.run_upscale(spec, w)
            except RuntimeError as e:
                assert "no ESRGAN model" in str(e)
            else:
                raise AssertionError("no model should raise")
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"All {len(tests)} upscale tests passed.")
