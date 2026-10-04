"""Unit tests for the numeric-control bounds (no UI is built).

Regression guard: loading an A1111/Civitai image's parameters wrote its CFG straight into
the slider. gradio validates a slider on PREPROCESS, not on write, so nothing failed at
import time -- the NEXT Generate died with

    gradio.exceptions.Error: 'Value 12 is greater than maximum value 8.0.'

and nothing in that traceback pointed back at the import. Every path that writes into a
bounded control now clamps through _clamp_ui, and the sliders are built from the same
table so the two cannot drift apart.

Run:  .venv/Scripts/python tests/test_ui_bounds.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_ui  # noqa: E402


def test_clamp_reports_only_when_it_changes_something():
    assert cz_ui._clamp_ui("guidance", 4.0) == (4.0, "")
    v, note = cz_ui._clamp_ui("guidance", 99.0)
    assert v == 20.0 and note == "99.0 -> 20.0"
    v, note = cz_ui._clamp_ui("gen_steps", 1)
    assert v == 2 and note == "1 -> 2"
    # An unknown key passes through: _clamp_ui must never invent a bound.
    assert cz_ui._clamp_ui("seed", 123456) == (123456, "")
    assert cz_ui._clamp_ui("guidance", None) == (None, "")


def test_clamp_keeps_the_type():
    v, _ = cz_ui._clamp_ui("gen_steps", 400)
    assert isinstance(v, int) and v == 40
    v, _ = cz_ui._clamp_ui("guidance", 12.0)
    assert isinstance(v, float) and v == 12.0      # 12 is INSIDE the range now


def test_cfg_12_is_a_legal_value():
    """The reported crash: CFG 12 is an ordinary Civitai recipe and the slider stopped at
    8. It must go through untouched, not be clamped to 8."""
    assert cz_ui._clamp_ui("guidance", 12.0) == (12.0, "")


def test_meta_apply_all_clamps_every_slider():
    """A realistic A1111 payload: CFG 12 (fits now), 60 steps and 2560x1440 (do not)."""
    out = cz_ui._ui_meta_apply_all({"prompt": "a cat", "seed": 42, "steps": 60,
                                    "guidance": 12.0, "size": "2560x1440"})
    report = out[0]["value"] if isinstance(out[0], dict) else str(out[0])
    seen = {}
    for upd in out[1:]:
        if isinstance(upd, dict) and "value" in upd:
            seen.setdefault(type(upd["value"]).__name__, []).append(upd["value"])
    assert 40 in seen.get("int", []), f"steps not clamped: {seen}"
    assert 2048 in seen.get("int", []), f"size not clamped: {seen}"
    assert 12.0 in seen.get("float", []), f"CFG 12 should pass: {seen}"
    assert "clamped" in report, report


def test_meta_apply_all_leaves_in_range_values_alone():
    out = cz_ui._ui_meta_apply_all({"steps": 8, "guidance": 4.0, "size": "1024x1536"})
    report = out[0]["value"] if isinstance(out[0], dict) else str(out[0])
    assert "clamped" not in report, report


def test_preset_load_clamps_out_of_range_values(tmp=None):
    """A preset written by hand, or saved by a build with another range."""
    import json
    import tempfile
    d = tempfile.mkdtemp()
    old_dir = cz_ui._PRESETS_DIR
    cz_ui._PRESETS_DIR = d
    try:
        with open(os.path.join(d, "wild.json"), "w", encoding="utf-8") as f:
            json.dump({"width": 9999, "height": 10, "steps": 500, "guidance": 99.0,
                       "image_number": 999}, f)
        scal = cz_ui._ui_preset_load("wild")[:len(cz_ui._PRESET_KEYS)]
        by_key = dict(zip(cz_ui._PRESET_KEYS, scal))
        assert by_key["width"]["value"] == 2048
        assert by_key["height"]["value"] == 256
        assert by_key["steps"]["value"] == 40          # _PRESET_KEYS 'steps' -> gen_steps
        assert by_key["guidance"]["value"] == 20.0
        assert by_key["image_number"]["value"] == 30
    finally:
        cz_ui._PRESETS_DIR = old_dir


def test_bounds_table_covers_the_sliders_built_from_it():
    """The point of the table: a slider must not be able to drift away from the clamp."""
    import re
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "cz_ui.py"), encoding="utf-8").read()
    for key in ("width", "height", "gen_steps", "guidance", "image_number"):
        assert re.search(r"gr\.Slider\(\*_UI_BOUNDS\[\"" + key + r"\"\]", src), key


if __name__ == "__main__":
    for fn in (test_clamp_reports_only_when_it_changes_something,
               test_clamp_keeps_the_type,
               test_cfg_12_is_a_legal_value,
               test_meta_apply_all_clamps_every_slider,
               test_meta_apply_all_leaves_in_range_values_alone,
               test_preset_load_clamps_out_of_range_values,
               test_bounds_table_covers_the_sliders_built_from_it):
        fn()
        print(f"OK {fn.__name__}")
    print("All UI bounds tests passed.")
