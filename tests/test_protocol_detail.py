"""Tests for detail_faces/detail_hands in the CLI protocol (spec -> the
ADetailer pass on the instance side). Fake cz_pipeline + cz_detailer, no GPU.
Run:  .venv/Scripts/python tests/test_protocol_detail.py"""
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

    def set_loras(self, slots):
        pass

    def txt2img_run(self, prompt, w, h, steps, seed, negative=""):
        return Image.new("RGB", (w, h), "#123456"), {"txt2img": 1.0}


class _FakeDetailer(types.ModuleType):
    DETAILER_ENABLED = False

    def __init__(self):
        super().__init__("cz_detailer")
        self.calls = []

    def detail_faces(self, img, prompt, seed, steps):
        self.calls.append(("faces", prompt))
        return img, 2

    def detail_hands(self, img, prompt, seed, steps):
        self.calls.append(("hands", prompt))
        if getattr(self, "hands_boom", False):
            raise RuntimeError("no ultralytics")
        return img, 1


@contextlib.contextmanager
def _fakes():
    pipe, det = _FakePipe(), _FakeDetailer()
    old = {k: sys.modules.get(k) for k in ("cz_pipeline", "cz_detailer")}
    sys.modules["cz_pipeline"] = pipe
    sys.modules["cz_detailer"] = det
    try:
        yield pipe, det
    finally:
        for k, v in old.items():
            if v is not None:
                sys.modules[k] = v
            else:
                sys.modules.pop(k, None)


def _gen(spec_extra, det_enabled=False, hands_boom=False):
    d = tempfile.mkdtemp(prefix="cz_det_")
    try:
        with _fakes() as (_pipe, det):
            det.DETAILER_ENABLED = det_enabled
            det.hands_boom = hands_boom
            spec, warnings = cp.validate_spec(
                {"protocol": 1, "prompt": "a hero", "width": 64,
                 "height": 64, "out_dir": d, **spec_extra})
            res = cp.run_gen(spec, warnings)
            return res, det.calls
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_explicit_true_runs_the_face_pass():
    res, calls = _gen({"detail_faces": True})
    assert res["ok"] and res["faces_refined"] == 2
    assert calls == [("faces", res and "a hero")]
    assert "detail_faces" in res["timings"]


def test_absent_follows_the_tool_default():
    res, calls = _gen({})
    assert res["faces_refined"] == 0 and calls == []      # tool default OFF
    res, calls = _gen({}, det_enabled=True)
    assert res["faces_refined"] == 2                      # tool default ON


def test_explicit_false_beats_the_tool_default():
    res, calls = _gen({"detail_faces": False}, det_enabled=True)
    assert res["faces_refined"] == 0 and calls == []


def test_hands_never_implicit_and_fail_clean():
    res, calls = _gen({}, det_enabled=False)
    assert res["hands_refined"] == 0
    res, calls = _gen({"detail_hands": True})
    assert res["hands_refined"] == 1 and ("hands", "a hero") in calls
    res, _calls = _gen({"detail_hands": True}, hands_boom=True)
    assert res["ok"] and res["hands_refined"] == 0
    assert any("detailer skipped" in w for w in res["warnings"])


def test_caps_announce_detail_support():
    caps = cp.caps_dict()
    assert "detail_faces" in caps["supports"]
    assert "detail_hands" in caps["supports"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"All {len(tests)} detail tests passed.")
