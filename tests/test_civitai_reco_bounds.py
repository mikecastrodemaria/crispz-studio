"""A CivitAI consensus must never push a value the Generate sliders refuse.

Seen on 2026-11-07 (crispz-moss, a MOSS build of Jib Mix; same code in the whole family): the sidecar's 'recommended'
carried steps 0 (CivitAI publishes 0, not null, for images without the value, and the
median kept them). Selecting the checkpoint put the steps slider at 0, below its minimum
of 2, and Gradio refused every click after that ("Value 0 is less than minimum value 2").
Two guards: analyze_settings ignores zeros in the median, and the UI keeps a sidecar's
values inside the sliders' bounds (said in the log), falling back to the model profile.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _imgs(*pairs):
    return [{"meta": {"steps": st, "cfgScale": cfg, "sampler": "Euler"}} for st, cfg in pairs]


def test_the_median_ignores_zero_steps_and_cfg():
    import cz_civitai as C
    # A majority of zeros used to drag the median to 0.
    reco = C.analyze_settings(_imgs((0, 0), (0, 0), (0, 0), (30, 4.0), (40, 5.0)))
    assert reco["steps"] == 40 and reco["guidance"] == 5.0, reco
    assert reco["n"] == 5                      # the images still count as published
    # Only zeros: no consensus at all on steps / CFG, the sampler majority stays.
    reco = C.analyze_settings(_imgs((0, 0), (0, 0), (0, 0)))
    assert "steps" not in reco and "guidance" not in reco, reco
    assert reco.get("sampler") == "Euler"
    print("OK test_the_median_ignores_zero_steps_and_cfg")


def test_a_sidecar_value_outside_the_sliders_never_reaches_them():
    import cz_ui as U
    lo, hi = U.GEN_STEPS_BOUNDS
    glo, ghi = U.GUIDANCE_BOUNDS
    said = []
    orig = U._log
    U._log = lambda msg, *a, **k: said.append(str(msg))
    try:
        assert U._sane_reco({"steps": 0, "guidance": 0}) == (None, None)
        assert U._sane_reco({"steps": -3, "guidance": -1.0}) == (None, None)
        assert U._sane_reco({"steps": hi + 500, "guidance": ghi + 90}) == (hi, ghi)
        assert U._sane_reco({"steps": lo, "guidance": 3.5}) == (lo, 3.5)
        assert U._sane_reco({"steps": "12", "guidance": "2.5"}) == (12, 2.5)
        assert U._sane_reco({"steps": "n/a"}) == (None, None)
        assert U._sane_reco({}) == (None, None)
        assert any("ignored" in m for m in said) and any("clamped" in m for m in said), said
    finally:
        U._log = orig
    # The whole path: a sidecar with steps 0 next to the checkpoint = the model profile
    # applies, the slider never sees 0.
    tmp = tempfile.mkdtemp()
    ck = os.path.join(tmp, "jibMixMoss_v2.safetensors")
    open(ck, "wb").close()
    with open(os.path.join(tmp, "jibMixMoss_v2.civitai.json"), "w", encoding="utf-8") as f:
        json.dump({"modelName": "Jib Mix MOSS", "recommended": {"n": 12, "steps": 0, "guidance": 0}}, f)
    if hasattr(U, "_profile_for_checkpoint"):
        st, g, _why = U._profile_for_checkpoint(ck)
        assert lo <= st <= hi and glo <= g <= ghi, (st, g)
    # The slider itself is built from the same bounds.
    import gradio as gr
    demo = U.build_ui()
    sl = [b for b in demo.blocks.values() if isinstance(b, gr.Slider)
          and str(getattr(b, "label", "")).startswith("Generation steps")]
    assert sl and (sl[0].minimum, sl[0].maximum) == (lo, hi), [(s.minimum, s.maximum) for s in sl]
    print("OK test_a_sidecar_value_outside_the_sliders_never_reaches_them")


if __name__ == "__main__":
    test_the_median_ignores_zero_steps_and_cfg()
    test_a_sidecar_value_outside_the_sliders_never_reaches_them()
    print("ALL OK")
