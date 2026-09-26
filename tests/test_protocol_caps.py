"""Tests for the enriched caps (models/loras/model_loaded) of the family CLI protocol.
No torch: the model folders are tmp dirs pointed at by the env variables, and the
'loaded model' comes from a fake cz_pipeline injected in sys.modules.
Run:  .venv/Scripts/python tests/test_protocol_caps.py"""
import os
import sys
import types
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_protocol as cp  # noqa: E402


def test_caps_lists_models_and_loras_from_dirs():
    d = tempfile.mkdtemp(prefix="cz_caps_")
    try:
        os.makedirs(os.path.join(d, "ck", "sub"))
        os.makedirs(os.path.join(d, "lo", "_index"))
        for f in ("b.safetensors", os.path.join("sub", "a.gguf")):
            open(os.path.join(d, "ck", f), "w").close()
        open(os.path.join(d, "ck", "notes.txt"), "w").close()
        open(os.path.join(d, "lo", "ink.safetensors"), "w").close()
        open(os.path.join(d, "lo", "_index", "cache.safetensors"), "w").close()
        os.environ["CHECKPOINTS_DIR"] = os.path.join(d, "ck")
        os.environ["LORAS_DIR"] = os.path.join(d, "lo")
        caps = cp.caps_dict()
        assert caps["models"] == ["b.safetensors", "sub/a.gguf"]
        assert caps["loras"] == ["ink.safetensors"]     # _index ignored
        assert caps["model_loaded"] == ""               # cold czp
    finally:
        os.environ.pop("CHECKPOINTS_DIR", None)
        os.environ.pop("LORAS_DIR", None)
        shutil.rmtree(d, ignore_errors=True)


def test_caps_reports_the_loaded_model_on_the_instance_side():
    fake = types.ModuleType("cz_pipeline")
    # A full checkpoint path: caps must report the file name only.
    # os.path.join keeps the separator of the running OS - a backslash is
    # not a separator under Linux, so a path hardcoded Windows-style only
    # ever passed on Windows (the CI runs on Linux).
    fake.ZIMAGE_TRANSFORMER = os.path.join("D:" + os.sep, "models", "zit",
                                           "cool-model.safetensors")
    fake.BASE_REPO = "some/base"
    old = sys.modules.get("cz_pipeline")
    sys.modules["cz_pipeline"] = fake
    try:
        assert cp.caps_dict()["model_loaded"] == "cool-model.safetensors"
        fake.ZIMAGE_TRANSFORMER = None
        assert cp.caps_dict()["model_loaded"] == "some/base"
    finally:
        if old is not None:
            sys.modules["cz_pipeline"] = old
        else:
            sys.modules.pop("cz_pipeline", None)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"All {len(tests)} caps tests passed.")
