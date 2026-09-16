"""{a|b|c} variant groups wired into crispz-studio (prompt_variants + cz_prompt).

Covers the family rules for the variant syntax:
  - rule 3: a prompt WITHOUT group gives exactly the same text, the same random
    draws, hence the same image as before (checked against the pre-variants
    _apply_wildcards, copied verbatim below);
  - groups are seed-bound, walk the batch index in order mode, and resolve BEFORE
    __wildcards__ (a placeholder in an unpicked option is never read);
  - the negative prompt is expanded too;
  - <lora:...> tags are read on the EXPANDED text (a LoRA in an unpicked option is
    never applied): UI Generate, CLI, protocol;
  - seed -1 is resolved to a concrete value where a prompt is expanded;
  - comics: {@A|@B} cites ONE character, the panel seed is fixed and stored.

No torch, no GPU: the pipeline calls are stubbed.

Run:  .venv/Scripts/python tests/test_variants_wiring.py
"""
import os
import re
import sys
import types
import random
import shutil
import tempfile
import contextlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

import cz_prompt as P  # noqa: E402


# ---------------------------------------------------------------- helpers ---
def _legacy_apply_wildcards(text, rng=None, index=None, wildcards_dir=None,
                            in_order_flag=False):
    """cz_prompt._apply_wildcards as it was BEFORE the variant syntax (verbatim
    logic), used as the reference for the no-regression rule."""
    if not text or "__" not in text:
        return text
    rng = rng or random
    in_order = in_order_flag and index is not None
    for _ in range(64):
        m = re.search(r"__([A-Za-z0-9_\-/]+)__", text)
        if not m:
            break
        name = m.group(1)
        path = os.path.join(wildcards_dir, name + ".txt")
        repl = ""
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    lines = [ln.strip() for ln in fh
                             if ln.strip() and not ln.lstrip().startswith("#")]
                if lines:
                    repl = lines[int(index) % len(lines)] if in_order else rng.choice(lines)
            except Exception:
                pass
        text = text[:m.start()] + repl + text[m.end():]
    return text


@contextlib.contextmanager
def _wildcards(files, in_order=False):
    """Temporary wildcards folder {name: [lines]} + in-order flag."""
    d = tempfile.mkdtemp(prefix="cz_wc_")
    for name, lines in files.items():
        with open(os.path.join(d, name + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    old_dir, old_order = P.WILDCARDS_DIR, P.READ_WILDCARDS_IN_ORDER
    P.WILDCARDS_DIR, P.READ_WILDCARDS_IN_ORDER = d, in_order
    try:
        yield d
    finally:
        P.WILDCARDS_DIR, P.READ_WILDCARDS_IN_ORDER = old_dir, old_order
        shutil.rmtree(d, ignore_errors=True)


class _RecordingRng(random.Random):
    """random.Random that records every choice() population."""
    def __init__(self, seed=0):
        super().__init__(seed)
        self.choices = []

    def choice(self, seq):
        self.choices.append(list(seq))
        return seq[0]


# ------------------------------------------------ rule 3: no seed regression ---
_PLAIN_PROMPTS = [
    "a cat on a sofa, cinematic",
    "portrait of __color__ haired woman, __place__ background",
    "{prompt} is a style placeholder, not a group",
    "stray braces { like this and } that",
    "<lora:ink:0.7> a knight, __color__ armor",
    "",
]


def test_prompt_without_group_is_unchanged_and_draws_the_same():
    with _wildcards({"color": ["red", "blue", "green", "black"],
                     "place": ["forest", "city", "beach"]}) as d:
        for in_order in (False, True):
            P.READ_WILDCARDS_IN_ORDER = in_order
            for text in _PLAIN_PROMPTS:
                for seed in range(25):
                    for index in (None, 0, 3):
                        a, b = random.Random(seed), random.Random(seed)
                        new = P._apply_wildcards(text, a, index=index)
                        old = _legacy_apply_wildcards(text, b, index=index,
                                                      wildcards_dir=d,
                                                      in_order_flag=in_order)
                        assert new == old, (text, seed, index, new, old)
                        assert a.getstate() == b.getstate(), \
                            f"extra random draw for {text!r}"


def test_expand_prompt_pair_on_plain_text_is_identity():
    assert P.expand_prompt_pair("a fox", "blurry", 7, index=2) == ("a fox", "blurry")


# ------------------------------------------------------------- expansion ---
def test_group_is_seed_bound():
    with _wildcards({}):
        picks = {P._apply_wildcards("{a|b|c|d|e|f}", P._seed_rng(s)) for s in range(30)}
        assert len(picks) > 1
        assert (P._apply_wildcards("x {a|b|c|d} y", P._seed_rng(11))
                == P._apply_wildcards("x {a|b|c|d} y", P._seed_rng(11)))


def test_in_order_walks_the_batch_index():
    with _wildcards({}, in_order=True):
        out = [P._apply_wildcards("{shy|sad|smile}", P._seed_rng(1), index=i)
               for i in range(4)]
        assert out == ["shy", "sad", "smile", "shy"]


def test_group_resolves_before_wildcard_placeholder():
    """{calm|__mood__}: picking 'calm' must never read wildcards/mood.txt."""
    with _wildcards({"mood": ["angry", "happy"]}):
        rng = _RecordingRng()
        out = P._apply_wildcards("a {calm|__mood__} face", rng)
        assert out == "a calm face"
        assert rng.choices == [["calm", "__mood__"]], rng.choices


def test_wildcard_line_may_hold_a_group_and_a_group_may_hold_a_wildcard():
    with _wildcards({"coat": ["{red|blue} coat"]}, in_order=True):
        assert P._apply_wildcards("__coat__", P._seed_rng(1), index=1) == "blue coat"
        assert P._apply_wildcards("{__coat__|nothing}", P._seed_rng(1), index=0) == "red coat"


def test_negative_is_expanded_with_its_own_rng():
    with _wildcards({}):
        p1, n1 = P.expand_prompt_pair("{a|b|c|d}", "{x|y|z|w}", 5)
        p2, n2 = P.expand_prompt_pair("a changed prompt", "{x|y|z|w}", 5)
        assert "{" not in p1 and "{" not in n1
        assert n1 == n2, "editing the positive must not change the negative picks"


def test_resolve_seed():
    assert P.resolve_seed(42) == 42
    assert P.resolve_seed(0) == 0
    for bad in (-1, "-1", None, "x"):
        s = P.resolve_seed(bad)
        assert isinstance(s, int) and s >= 0


# -------------------------------------------------------------- protocol ---
class _FakePipeline(types.ModuleType):
    LORA_WEIGHT = 1.0
    _LAST_SEED = -1

    def __init__(self):
        super().__init__("cz_pipeline")
        self.calls = {"loras": None, "gen": None, "inpaint": None}

    def set_loras(self, slots):
        self.calls["loras"] = slots

    def set_zimage_model(self, m):
        pass

    def txt2img_run(self, prompt, w, h, steps, seed, negative=""):
        self.calls["gen"] = (prompt, seed, negative)
        return Image.new("RGB", (w, h)), {"txt2img": 0.1}

    def inpaint_run(self, bg, mask, prompt, steps, denoise, seed):
        self.calls["inpaint"] = (prompt, seed)
        return bg.copy()


@contextlib.contextmanager
def _fake_pipeline():
    fake = _FakePipeline()
    old = sys.modules.get("cz_pipeline")
    sys.modules["cz_pipeline"] = fake
    try:
        yield fake
    finally:
        if old is not None:
            sys.modules["cz_pipeline"] = old
        else:
            sys.modules.pop("cz_pipeline", None)


def _run_protocol_gen(spec_in):
    import cz_protocol as cp
    d = tempfile.mkdtemp(prefix="cz_var_proto_")
    try:
        with _fake_pipeline() as fake:
            spec, w = cp.validate_spec(dict(spec_in, protocol=1, out_dir=d,
                                            width=64, height=64))
            res = cp.run_gen(spec, w)
        return res, fake
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_protocol_plain_prompt_is_unchanged():
    with _wildcards({}):
        res, fake = _run_protocol_gen({"prompt": "a cat <lora:ink:0.5>",
                                       "negative": "blurry", "seed": 42})
    assert fake.calls["gen"] == ("a cat", 42, "blurry")
    assert fake.calls["loras"] == [("ink", 0.5)]
    assert res["seed_used"] == 42


def test_protocol_validate_keeps_lora_tags_inside_groups_until_run_time():
    import cz_protocol as cp
    spec, _w = cp.validate_spec({"protocol": 1,
                                 "prompt": "a {cat <lora:catlora>|dog <lora:doglora>}"})
    assert spec["loras"] == [] and "<lora:" in spec["prompt"]


def test_protocol_expands_and_reads_only_the_picked_lora():
    with _wildcards({}, in_order=True):   # in order + index 0 -> first option
        res, fake = _run_protocol_gen({
            "prompt": "a {cat <lora:catlora:0.8>|dog <lora:doglora>}, park",
            "negative": "{blurry|noisy}", "seed": 7})
    prompt, seed, negative = fake.calls["gen"]
    assert (prompt, seed, negative) == ("a cat, park", 7, "blurry")
    assert fake.calls["loras"] == [("catlora", 0.8)]
    assert res["loras"] == ["catlora:0.8"]


def test_protocol_resolves_seed_before_expanding():
    with _wildcards({}):
        res, fake = _run_protocol_gen({"prompt": "{a|b|c|d|e|f} tree", "seed": -1})
    assert res["seed_used"] >= 0
    assert fake.calls["gen"][0] == P._apply_wildcards("{a|b|c|d|e|f} tree",
                                                      P._seed_rng(res["seed_used"]),
                                                      index=0)


# ---------------------------------------------------------- UI Generate ---
def _ui_txt2img(prompt, negative, n, seed):
    """Calls cz_ui._ui_generate in txt2img mode with a stubbed pipeline; returns
    the (prompt, seed, negative) of every image, in order."""
    import cz_ui
    import cz_detailer
    calls = []

    def fake_txt2img_run(fp, w, h, gen_steps, s, fn, **kw):
        calls.append((fp, s, fn))
        return Image.new("RGB", (32, 32)), {"txt2img": 0.0}

    real = cz_ui.txt2img_run
    det = (cz_detailer.DETAILER_ENABLED, cz_detailer.HAND_ENABLED)
    cz_ui.txt2img_run = fake_txt2img_run
    cz_detailer.DETAILER_ENABLED = cz_detailer.HAND_ENABLED = False
    try:
        cz_ui._ui_generate(
            prompt, negative, [], False, False, None, "Input image",
            None, None, None, None, False, None,
            64, 64, 4, n, seed, 0.0, "balanced",
            "esrgan.pth", False, False, False, 2.0, 0.3, 8,
            512, 64, 0, 64, "display", "out", "png", [],
            progress=lambda f, desc=None: None)
    finally:
        cz_ui.txt2img_run = real
        cz_detailer.DETAILER_ENABLED, cz_detailer.HAND_ENABLED = det
    return calls


def test_ui_generate_plain_prompt_is_unchanged():
    with _wildcards({}):
        calls = _ui_txt2img("a lighthouse at dusk", "blurry", 3, 1000)
    assert calls == [("a lighthouse at dusk", 1000, "blurry"),
                     ("a lighthouse at dusk", 1001, "blurry"),
                     ("a lighthouse at dusk", 1002, "blurry")]


def test_ui_generate_expands_per_image_positive_and_negative():
    with _wildcards({}, in_order=True):
        calls = _ui_txt2img("a {red|blue} car", "{blurry|noisy}", 3, 50)
    assert calls == [("a red car", 50, "blurry"), ("a blue car", 51, "noisy"),
                     ("a red car", 52, "blurry")]


def test_ui_generate_ignores_a_lora_in_an_unpicked_option():
    """A missing LoRA in the unpicked option must not stop the run (it would if the
    tags were still read on the raw prompt)."""
    with _wildcards({}, in_order=True):
        calls = _ui_txt2img("a {cat|dog <lora:__no_such_lora__>}", "", 1, 3)
    assert calls == [("a cat", 3, "")]


# ------------------------------------------------------------------- CLI ---
def test_cli_txt2img_resolves_seed_and_expands():
    import cz_cli
    got = {}

    def fake_txt2img_run(prompt, w, h, gen_steps, seed, negative, **kw):
        got.update(prompt=prompt, seed=seed, negative=negative)
        return Image.new("RGB", (32, 32)), {"txt2img": 0.0}

    real = cz_cli.txt2img_run
    cz_cli.txt2img_run = fake_txt2img_run
    try:
        with _wildcards({}, in_order=True):
            rc = cz_cli.cli_main(["--txt2img", "--prompt", "a {cat|dog} <lora:x>",
                                  "--negative", "{blurry|noisy}", "--seed", "-1",
                                  "--save-mode", "display", "--quiet"])
    finally:
        cz_cli.txt2img_run = real
    # <lora:x> is missing locally -> the CLI exits 2 BEFORE generating; retry
    # without it to check the expansion itself.
    assert rc == 2 and not got
    cz_cli.txt2img_run = fake_txt2img_run
    try:
        with _wildcards({}, in_order=True):
            rc = cz_cli.cli_main(["--txt2img", "--prompt", "a {cat|dog}",
                                  "--negative", "{blurry|noisy}", "--seed", "-1",
                                  "--save-mode", "display", "--quiet"])
    finally:
        cz_cli.txt2img_run = real
    assert rc in (0, None)
    assert got["prompt"] == "a cat" and got["negative"] == "blurry"
    assert got["seed"] >= 0


# ---------------------------------------------------------------- comics ---
def _comic_project(text, seed=-1):
    import cz_comic as cc
    p = cc.new_project("v", casting={
        "Lea": {"desc": "a woman, {red|green} coat", "refs": ["refs/lea.png"],
                "loras": ["lea.safetensors:0.8"], "negative": "beard"},
        "Sam": {"desc": "a man", "refs": ["refs/sam.png"],
                "loras": ["sam.safetensors"], "negative": "makeup"}},
        style={"prompt_suffix": "comic panel"})
    ch = cc.add_chapter(p, "c")
    page = cc.add_page(p, ch["id"], "splash", texts=[text])
    page["panels"][0]["seed"] = seed
    return p, page, page["panels"][0]


def test_comic_panel_without_group_is_unchanged():
    import cz_comic as cc
    p, page, panel = _comic_project("@Sam walks")
    r = cc.resolve_panel(p, page, panel)
    assert r["prompt"] == "a man walks, comic panel"
    assert r["seed"] == -1 and panel["seed"] == -1


def test_comic_group_cites_one_character_and_fixes_the_seed():
    import cz_comic as cc
    p, page, panel = _comic_project("{@Lea|@Sam} waits in the rain")
    r = cc.resolve_panel(p, page, panel)
    assert r["seed"] >= 0 and panel["seed"] == r["seed"], "seed must be stored"
    lea = r["refs"] == ["refs/lea.png"]
    sam = r["refs"] == ["refs/sam.png"]
    assert lea != sam, r["refs"]
    assert len(r["loras"]) == 1 and "{" not in r["prompt"]
    again = cc.resolve_panel(p, page, panel)
    assert again["prompt"] == r["prompt"] and again["refs"] == r["refs"]


def test_comic_detail_prompt_picks_the_same_option_as_the_render():
    import cz_comic as cc
    for seed in range(20):
        p, page, panel = _comic_project("@Lea runs", seed=seed)
        r = cc.resolve_panel(p, page, panel)
        coat = "red coat" if "red coat" in r["prompt"] else "green coat"
        assert coat in cc.detail_prompt(p, panel)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"All {len(tests)} variant wiring tests passed.")
