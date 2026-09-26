"""Tests for the protocol's 'edit' op (image + instruction -> image through omni).
Fake cz_pipeline (generate_omni); omni itself is simulated by monkeypatching
_omni_configured. Run:  .venv/Scripts/python tests/test_protocol_edit.py"""
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
    LORA_WEIGHT = 1.0

    def __init__(self):
        super().__init__("cz_pipeline")
        self.omni_calls = []

    def set_loras(self, slots):
        pass

    def generate_omni(self, refs, prompt, negative, w, h, steps, seed):
        self.omni_calls.append({"n_refs": len(refs), "prompt": prompt,
                                "size": (w, h)})
        return Image.new("RGB", (w, h), "#775533")

    def txt2img_run(self, *a, **k):
        raise AssertionError("edit must go through generate_omni, not txt2img")


@contextlib.contextmanager
def _fakes(omni=True):
    pipe = _FakePipe()
    old = sys.modules.get("cz_pipeline")
    old_omni = cp._omni_configured
    sys.modules["cz_pipeline"] = pipe
    cp._omni_configured = lambda: omni
    try:
        yield pipe
    finally:
        cp._omni_configured = old_omni
        if old is not None:
            sys.modules["cz_pipeline"] = old
        else:
            sys.modules.pop("cz_pipeline", None)


def test_edit_routes_through_omni_with_the_input_as_ref():
    d = tempfile.mkdtemp(prefix="cz_edit_")
    try:
        src = os.path.join(d, "panel.png")
        Image.new("RGB", (1000, 700), "#123").save(src)
        with _fakes(omni=True) as pipe:
            spec, w = cp.validate_spec(
                {"protocol": 1, "input": src, "prompt": "add rain",
                 "out_dir": d}, op="edit")
            assert spec["refs"] == [src]
            assert spec["width"] == 992 and spec["height"] == 672  # align 32
            res = cp.run_gen(spec, w)
        assert res["ok"] and res["refs_used"] == 1
        assert pipe.omni_calls[0]["prompt"] == "add rain"
        assert pipe.omni_calls[0]["n_refs"] == 1
        assert os.path.isfile(res["images"][0])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_edit_refused_without_omni_never_a_fresh_image():
    d = tempfile.mkdtemp(prefix="cz_edit2_")
    try:
        src = os.path.join(d, "panel.png")
        Image.new("RGB", (64, 64)).save(src)
        with _fakes(omni=False):
            try:
                cp.validate_spec({"protocol": 1, "input": src,
                                  "prompt": "add rain"}, op="edit")
            except cp.SpecError as e:
                assert e.code == 3 and "edit not supported" in str(e)
            else:
                raise AssertionError("edit without omni should be refused")
            assert cp.caps_dict()["supports"]["edit"] is False
        with _fakes(omni=True):
            assert cp.caps_dict()["supports"]["edit"] is True
            for bad in ({"protocol": 1, "input": src},          # no prompt
                        {"protocol": 1, "prompt": "x"}):         # no input
                try:
                    cp.validate_spec(bad, op="edit")
                except cp.SpecError as e:
                    assert e.code == 2
                else:
                    raise AssertionError(f"{bad} should raise")
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"All {len(tests)} edit tests passed.")
