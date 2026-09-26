"""Protocol op 'inpaint': validation (input/mask/denoise), empty-mask refusal,
mask resized to the image, prompt optional, caps/ops announce it; the
pipeline is faked (no GPU). Run: .venv/Scripts/python tests/test_protocol_inpaint.py"""
import os
import sys
import types
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_protocol as cp  # noqa: E402


def _fake_pipeline(calls):
    def inpaint_run(bg, mask, prompt, steps, denoise, seed):
        calls.append({"size": bg.size, "mask": mask.size,
                      "white": mask.getbbox(), "prompt": prompt,
                      "steps": steps, "denoise": denoise, "seed": seed})
        return bg
    return types.SimpleNamespace(inpaint_run=inpaint_run, _LAST_SEED=None)


def test_caps_announce_inpaint():
    caps = cp.caps_dict()
    assert "inpaint" in caps["ops"]
    assert caps["supports"]["inpaint"] is cp.FAMILY_CAPS["inpaint"]
    assert caps["supports"]["img2img"] is cp.FAMILY_CAPS["img2img"]
    assert "inpaint" in cp.OPS and "mask" in cp.SPEC_FIELDS


def test_validate_and_run():
    from PIL import Image
    if not cp.FAMILY_CAPS["inpaint"]:
        # family without an inpaint pipeline: clean refusal, exit 3
        try:
            cp.validate_spec({"protocol": 1, "input": __file__, "mask": __file__},
                             op="inpaint")
            raise AssertionError("inpaint accepted")
        except cp.SpecError as e:
            assert e.code == 3 and "no inpaint pipeline" in str(e)
        try:
            cp.validate_spec({"protocol": 1, "input": __file__, "factor": 1.0},
                             op="upscale")
            raise AssertionError("variation accepted")
        except cp.SpecError as e:
            assert e.code == 3 and "img2img" in str(e)
        return
    d = tempfile.mkdtemp(prefix="cz_inp_")
    img, msk, empty, small = (os.path.join(d, n) for n in
                              ("in.png", "mask.png", "empty.png", "small.png"))
    Image.new("RGB", (128, 96), "white").save(img)
    m = Image.new("L", (128, 96), 0)
    m.paste(255, (10, 10, 50, 50))
    m.save(msk)
    Image.new("L", (128, 96), 0).save(empty)
    m.resize((64, 48)).save(small)

    for bad, why in (({}, "input"), ({"input": img}, "mask"),
                     ({"input": img, "mask": "nope.png"}, "not found"),
                     ({"input": img, "mask": msk, "denoise": 3}, "range")):
        try:
            cp.validate_spec({"protocol": 1, **bad}, op="inpaint")
            raise AssertionError(f"accepted {bad}")
        except cp.SpecError as e:
            assert why in str(e), (why, str(e))
    spec, warns = cp.validate_spec({"protocol": 1, "input": img, "mask": msk,
                                    "prompt": "", "denoise": 0.7, "seed": 7},
                                   op="inpaint")
    assert spec["denoise"] == 0.7 and spec["mask"] == msk and spec["seed"] == 7

    calls = []
    old = sys.modules.get("cz_pipeline")
    sys.modules["cz_pipeline"] = _fake_pipeline(calls)
    try:
        res = cp.run_inpaint(dict(spec, out_dir=d), warns)
        assert res["ok"] and res["op"] == "inpaint" and res["seed_used"] == 7
        assert os.path.isfile(res["images"][0])
        c = calls[-1]
        assert c["size"] == (128, 96) and c["denoise"] == 0.7 and c["steps"] > 0
        assert c["white"] == (10, 10, 50, 50) and c["prompt"] == ""
        # mask at another size -> resized with a warning
        spec2, w2 = cp.validate_spec({"protocol": 1, "input": img,
                                      "mask": small}, op="inpaint")
        res = cp.run_inpaint(dict(spec2, out_dir=d), w2)
        assert calls[-1]["mask"] == (128, 96)
        assert any("resized" in w for w in res["warnings"])
        assert res["denoise"] == float(cp.CONFIG.get("default_inpaint_strength", 0.9))
        # empty mask -> clean refusal
        spec3, _ = cp.validate_spec({"protocol": 1, "input": img,
                                     "mask": empty}, op="inpaint")
        try:
            cp.run_inpaint(dict(spec3, out_dir=d))
            raise AssertionError("empty mask accepted")
        except cp.SpecError as e:
            assert "empty" in str(e)
    finally:
        if old is None:
            sys.modules.pop("cz_pipeline", None)
        else:
            sys.modules["cz_pipeline"] = old


if __name__ == "__main__":
    test_caps_announce_inpaint()
    print("OK test_caps_announce_inpaint")
    test_validate_and_run()
    print("OK test_validate_and_run")
    print("All 2 inpaint protocol tests passed.")
