"""Describe : styles d'analyse, consigne v4 par defaut, nettoyage, plafonds Ollama, Caption
model Ollama avec repli BLIP, detection qui reprend le modele retenu.

Mesure du 2026-09-11 (Agents-A1-4B, muse-glimmer ; chaque description regeneree par klein) :
sans le medium en tete un dessin revenait en photo, un texte cite ligne par ligne revenait
melange, et un petit modele ecrivait "No text is visible." malgre la consigne.

Run:  .venv/Scripts/python tests/test_describe_styles.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

import cz_core as C
import cz_ollama as O


def test_the_default_style_is_the_measured_v4():
    t = C.describe_instruction(C.DEFAULT_DESCRIBE_STYLE, C.DEFAULT_DESCRIBE_LENGTH)
    for must in ("Begin with the medium and style", "single string in double quotes",
                 "about 180 words", "never mention what is absent", "State every detail as a fact",
                 "era when they are identifiable"):   # l'epoque : +0,11 de fidelite sur le portrait
        assert must in t, must
    assert "comma-separated" not in t and "{words}" not in t, t
    for s in C.DESCRIBE_STYLES:
        for n in C.DESCRIBE_LENGTHS:
            got = C.describe_instruction(s, n)
            assert "{words}" not in got and "Output only" in got, (s, n)
    assert "about 60 words" in C.describe_instruction(C.DEFAULT_DESCRIBE_STYLE, "Short")
    assert "at most 25 words" in C.describe_instruction(C.SHORT_CAPTION_STYLE, "Very long")
    # inconnu -> le defaut
    assert C.describe_instruction("nope", "nope") == t
    # la legende de dataset de Captionz, gardee comme style a part (labelling)
    ds = C.describe_instruction("Dataset paragraph", "Medium")
    assert "aspect ratio" in ds and "era if identifiable" in ds and "about 120 words" in ds, ds
    print("OK test_the_default_style_is_the_measured_v4")


def test_the_old_sample_instruction_is_not_a_customisation():
    """L'ancienne consigne en tags, recopiee de config-sample dans les config.txt, ne doit
    pas masquer les styles ; une vraie consigne personnelle, si."""
    old = dict(C.CONFIG)
    try:
        for v in (C.LEGACY_DESCRIBE_INSTRUCTION, "   ", "", None):
            C.CONFIG["ollama_describe_prompt"] = v
            assert C._instruction("ollama_describe_prompt", C.LEGACY_DESCRIBE_INSTRUCTION, None) is None, v
        C.CONFIG["ollama_describe_prompt"] = "My own instruction"
        assert C._instruction("ollama_describe_prompt", C.LEGACY_DESCRIBE_INSTRUCTION, None) == "My own instruction"
        C.CONFIG["ollama_improve_prompt"] = C.LEGACY_IMPROVE_INSTRUCTION
        got = C._instruction("ollama_improve_prompt", C.LEGACY_IMPROVE_INSTRUCTION, "NEW {prompt}")
        assert got == "NEW {prompt}", got
    finally:
        C.CONFIG.clear()
        C.CONFIG.update(old)
    assert "{prompt}" in C.IMPROVE_INSTRUCTION and "{descriptions}" in C.COMPOSE_INSTRUCTION
    print("OK test_the_old_sample_instruction_is_not_a_customisation")


CLEAN_CASES = [
    ("A cat on a mat. No text is visible. No people are visible.", "A cat on a mat."),
    ("A quiet street. There is no one around.", "A quiet street."),
    ("The scarf appears to be wool.", "The scarf is wool."),
    ("The lights seem to be on.", "The lights are on."),
    ("Soft light, likely from the left.", "Soft light from the left."),
    ("Candid photograph of a street.", "Candid photograph of a street."),
    # une seule phrase (liste de tags) : jamais videe
    ("cat, mat, no text", "cat, mat, no text"),
    ("", ""),
    (None, ""),
]


def test_clean_description():
    for raw, want in CLEAN_CASES:
        got = O.clean_description(raw)
        assert got == want, f"{raw!r} -> {got!r} (attendu {want!r})"
    print("OK test_clean_description")


def test_gen_opts_cap_context_and_length():
    p = O._ollama_gen_opts()
    assert p["think"] is False and p["stream"] is False, p
    assert p["options"]["num_ctx"] == O.OLLAMA_NUM_CTX > 0, p
    assert p["options"]["num_predict"] == O.OLLAMA_NUM_PREDICT > 0, p
    assert "temperature" not in p["options"], p
    assert O._ollama_gen_opts(0.3)["options"]["temperature"] == 0.3
    print("OK test_gen_opts_cap_context_and_length")


def test_describe_sends_the_chosen_style_and_cleans_the_answer():
    seen = []

    def fake_http(path, payload=None, base=None, timeout=8):
        seen.append(payload)
        return {"response": "<think>x</think>Candid photograph of a cat. No text is visible."}

    old = O._ollama_http
    O._ollama_http = fake_http
    try:
        out = O._ollama_describe(Image.new("RGB", (32, 32)), "m", style="Photo (technical)", length="Short")
    finally:
        O._ollama_http = old
    assert out == "Candid photograph of a cat.", out
    p = seen[0]
    assert p["prompt"] == C.describe_instruction("Photo (technical)", "Short"), p["prompt"]
    assert p["images"] and p["think"] is False, p
    assert p["options"]["temperature"] == float(O.OLLAMA_DESCRIBE_TEMPERATURE), p
    print("OK test_describe_sends_the_chosen_style_and_cleans_the_answer")


def test_describe_style_state():
    old = (O.DESCRIBE_STYLE, O.DESCRIBE_LENGTH)
    try:
        assert O.set_describe_style("Photo (technical)", "Short") == ("Photo (technical)", "Short")
        assert O.set_describe_style("bogus", "bogus") == ("Photo (technical)", "Short")
        assert "Prompt (prose)" in O.describe_style_choices()
    finally:
        O.DESCRIBE_STYLE, O.DESCRIBE_LENGTH = old
    print("OK test_describe_style_state")


class _Inputs(dict):
    def to(self, _dev):
        return self


class _Proc:
    def __call__(self, img, return_tensors=None):
        return _Inputs()

    def decode(self, ids, skip_special_tokens=True):
        return " blip caption "


class _Mdl:
    def generate(self, **kw):
        return [[1, 2]]


def test_caption_model_can_be_an_ollama_model_and_falls_back_to_blip():
    import cz_face as F
    img = Image.new("RGB", (32, 32))
    old = (F._CAPTION_MODEL, F._CAPTIONER, O._ollama_caption, F._load_captioner)
    try:
        assert F.set_caption_model("ollama:Some/Model-4B:latest") == "ollama:Some/Model-4B:latest"
        assert F.set_caption_model("ollama:") == "ollama:Some/Model-4B:latest"   # vide -> ignore
        O._ollama_caption = lambda image, model, base=None: f"caption by {model}"
        assert F._local_caption(img) == "caption by Some/Model-4B:latest"

        def boom(image, model, base=None):
            raise RuntimeError("connection refused")

        O._ollama_caption = boom
        F._load_captioner = lambda: ("blip", _Proc(), _Mdl())
        assert F._local_caption(img) == "blip caption"
        assert F.set_caption_model("BLIP-BASE") == "blip-base"
    finally:
        F._CAPTION_MODEL, F._CAPTIONER, O._ollama_caption, F._load_captioner = old
    print("OK test_caption_model_can_be_an_ollama_model_and_falls_back_to_blip")


def test_describe_falls_back_to_the_caption_model_when_ollama_fails():
    import cz_ui as U
    old = (U._ollama_describe, U._local_caption)
    try:
        def boom(*a, **k):
            raise RuntimeError("connection refused")

        U._ollama_describe = boom
        U._local_caption = lambda image: "a local caption"
        upd, status = U._ui_describe(Image.new("RGB", (16, 16)), "some-model", "http://x")
    finally:
        U._ollama_describe, U._local_caption = old
    assert upd.get("value") == "a local caption", upd
    assert "failed" in status and "caption model" in status, status
    print("OK test_describe_falls_back_to_the_caption_model_when_ollama_fails")


def test_detect_restores_the_remembered_vision_model():
    import cz_ui as U
    old = (U._ollama_vision_models, C._load_prefs_raw)
    try:
        U._ollama_vision_models = lambda base=None: ["a:latest", "b:latest"]
        C._load_prefs_raw = lambda: {"ollama_model": "b:latest"}
        dd, _status, cap = U._ui_detect_ollama("http://x")
        assert dd["value"] == "b:latest" and list(dd["choices"]) == ["a:latest", "b:latest"], dd
        assert "ollama:b:latest" in cap["choices"] and "blip-large" in cap["choices"], cap
        C._load_prefs_raw = lambda: {"ollama_model": "gone:latest"}
        dd, _status, _cap = U._ui_detect_ollama("http://x")
        assert dd["value"] == "a:latest", dd

        def down(base=None):
            raise OSError("refused")

        U._ollama_vision_models = down
        dd, status, cap = U._ui_detect_ollama("http://x")
        assert dd["value"] is None and "not reachable" in status and "blip-large" in cap["choices"]
    finally:
        U._ollama_vision_models, C._load_prefs_raw = old
    print("OK test_detect_restores_the_remembered_vision_model")


if __name__ == "__main__":
    test_the_default_style_is_the_measured_v4()
    test_the_old_sample_instruction_is_not_a_customisation()
    test_clean_description()
    test_gen_opts_cap_context_and_length()
    test_describe_sends_the_chosen_style_and_cleans_the_answer()
    test_describe_style_state()
    test_caption_model_can_be_an_ollama_model_and_falls_back_to_blip()
    test_describe_falls_back_to_the_caption_model_when_ollama_fails()
    test_detect_restores_the_remembered_vision_model()
    print("All describe-style tests passed.")
