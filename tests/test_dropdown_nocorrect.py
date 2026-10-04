"""Unit tests for the browser-correction switches on the dropdown fields.

A Gradio dropdown is an `<input role="listbox">` holding the current value — here a file
name. The browser decorated it and dropped a popup over the open list.

`spellcheck="false"` alone was NOT enough: it only governs the red underline and the
right-click list. The floating bar of word candidates ("Latin / latine / latins") is a
SEPARATE feature, `writingsuggestions`, and measuring it on the running app showed the
input still reporting `writingSuggestions === "true"` while `spellcheck` was already false.

Run:  .venv/Scripts/python tests/test_dropdown_nocorrect.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cz_assets import CZ_JS  # noqa: E402

FLAT = re.sub(r"\s+", "", CZ_JS)


def test_both_features_are_switched_off():
    """spellcheck covers the underline, writingsuggestions covers the candidate bar.
    Dropping either one brings the popup back."""
    for attr in ("spellcheck", "writingsuggestions"):
        assert f"setAttribute('{attr}','false')" in FLAT, attr


def test_the_mobile_and_autofill_switches_are_there_too():
    for attr in ("autocorrect", "autocapitalize", "autocomplete"):
        assert f"setAttribute('{attr}','off')" in FLAT, attr


def test_only_the_dropdowns_are_targeted():
    """role=listbox matches the closed lists and nothing else in this UI: the prompts are
    <textarea>s and must keep both features, which is what you want on prose."""
    assert """querySelectorAll('input[role="listbox"]:not([spellcheck])')""" in FLAT


def test_it_is_re_applied_to_what_gradio_remounts():
    """Gradio mounts the dropdowns of a closed accordion late and rebuilds one whenever its
    choices change (a LoRA refresh), so a one-shot pass at load is not enough."""
    assert "MutationObserver" in CZ_JS
    # setTimeout and not requestAnimationFrame: rAF is paused while the tab is hidden.
    assert "requestAnimationFrame" not in CZ_JS.split("noCorrect")[-1]
    assert "setTimeout" in CZ_JS


def test_the_injected_js_stays_one_balanced_function():
    assert CZ_JS.strip().startswith("() => {") and CZ_JS.strip().endswith("}")
    assert CZ_JS.count("{") == CZ_JS.count("}")
    assert CZ_JS.count("(") == CZ_JS.count(")")


if __name__ == "__main__":
    for fn in (test_both_features_are_switched_off,
               test_the_mobile_and_autofill_switches_are_there_too,
               test_only_the_dropdowns_are_targeted,
               test_it_is_re_applied_to_what_gradio_remounts,
               test_the_injected_js_stays_one_balanced_function):
        fn()
        print(f"OK {fn.__name__}")
    print("All dropdown no-correct tests passed.")
