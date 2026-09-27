"""crispz-studio - Ollama integration (Describe / Improve / Vision Mix).

Pulled out of app.py. It calls Ollama's local HTTP API (/api/tags, /api/show,
/api/generate). It depends only on cz_core (config, log, b64). The UI handlers
(_ui_describe...) stay in app.py (the Gradio layer) and call these functions.

REASONING DISABLED. The "thinking" models (Qwen3, DeepSeek-R1, Kimi...) emit their
inner monologue, which ended up *in the image prompt*. Two defences, because
neither is enough on its own:
  1. `think: false` in the /api/generate payload (Ollama >= 0.9). A model that does
     not know the field answers 400 -> `_ollama_http` replays WITHOUT the field.
  2. `_strip_thinking()` on every answer: some models emit <think>...</think> tags
     in `response` anyway (a Modelfile template, an old Ollama), and the API can
     return a separate `thinking` field, which we ignore.

"""

import os
import re

import cz_core
import prompt_improve
from prompt_improve import OllamaError  # noqa: F401  (a re-export for the UI and the CLI)
from cz_core import (
    CONFIG, DESCRIBE_INSTRUCTION, IMPROVE_INSTRUCTION, COMPOSE_INSTRUCTION,
    DESCRIBE_STYLES, DESCRIBE_LENGTHS, DEFAULT_DESCRIBE_STYLE, DEFAULT_DESCRIBE_LENGTH,
    CUSTOM_STYLE, DESCRIBE_CUSTOM, SHORT_CAPTION_STYLE, describe_instruction,
    LEGACY_IMPROVE_INSTRUCTION, _prefs, _dbg, _pil_to_b64_jpeg,
)

# The Ollama URL (Describe image->prompt + Improve prompt). Settable, persisted.
# 127.0.0.1 by default, and a 'localhost' already configured is rewritten: under Windows,
# Python tries ::1 first and the call times out when Ollama only listens on IPv4.
OLLAMA_URL = prompt_improve.normalize_endpoint(
    os.environ.get("OLLAMA_URL") or _prefs.get("ollama_url")
    or CONFIG.get("ollama_url") or prompt_improve.DEFAULT_ENDPOINT)
# How long the Ollama model is kept in VRAM after a call (keep_alive). 0 =
# unloaded immediately -> frees the VRAM before the image generation.
OLLAMA_KEEP_ALIVE = CONFIG.get("ollama_keep_alive", 0)
# Forces Ollama onto the CPU (num_gpu=0) -> 0 VRAM shared with the model (slower).
OLLAMA_CPU = bool(CONFIG.get("ollama_cpu", False))
# The context and answer length sent on every call. Without num_ctx, Ollama takes the
# Modelfile's: 131,072 for Agents-A1-4B, which is 6.45 GB of VRAM instead of 3.33 GB at 8192
# (measured on 2026-09-11). Without num_predict, a model that loops never stops.
# 0 = leave Ollama's value.
OLLAMA_NUM_CTX = int(CONFIG.get("ollama_num_ctx", 8192) or 0)
OLLAMA_NUM_PREDICT = int(CONFIG.get("ollama_num_predict", 700) or 0)
# Describe's temperature: low = a faithful description (null = the model's own).
# Improve and Vision Mix's merging keep the model's.
OLLAMA_DESCRIBE_TEMPERATURE = CONFIG.get("ollama_describe_temperature", 0.3)


def describe_style_choices():
    """The Describe styles offered in Prompt AI (+ config.txt's own when there is one)."""
    return list(DESCRIBE_STYLES) + ([CUSTOM_STYLE] if DESCRIBE_CUSTOM else [])


def _initial_style():
    s = _prefs.get("describe_style")
    if s in describe_style_choices():
        return s
    return CUSTOM_STYLE if DESCRIBE_CUSTOM else DEFAULT_DESCRIBE_STYLE


# Describe's current style and length: the Prompt AI choice (preferences.json).
DESCRIBE_STYLE = _initial_style()
DESCRIBE_LENGTH = (_prefs.get("describe_length") if _prefs.get("describe_length") in DESCRIBE_LENGTHS
                   else DEFAULT_DESCRIBE_LENGTH)


def set_describe_style(style=None, length=None):
    """Changes Describe's style / length; an unknown value is ignored."""
    global DESCRIBE_STYLE, DESCRIBE_LENGTH
    if style in describe_style_choices():
        DESCRIBE_STYLE = style
    if length in DESCRIBE_LENGTHS:
        DESCRIBE_LENGTH = length
    return DESCRIBE_STYLE, DESCRIBE_LENGTH


# The reasoning tags of the "thinking" models. Non-greedy, case-insensitive,
# DOTALL: a block can run to dozens of lines.
_THINK_RE = re.compile(r"<\s*(think|thinking|reasoning)\s*>.*?<\s*/\s*\1\s*>",
                       re.IGNORECASE | re.DOTALL)
# A block opened and never closed (a truncation, a missing stop token): we cut up to
# the end of the opening and keep what follows.
_THINK_OPEN_RE = re.compile(r"^\s*<\s*(think|thinking|reasoning)\s*>", re.IGNORECASE)


def _strip_thinking(text):
    """Removes the inner monologue of a reasoning model.

    Without it, an image prompt ends up prefixed with "Okay, the user wants...".
    Handles the closed block, the block left open, and an orphan closing tag (the
    model started thinking before the first token was captured)."""
    t = text or ""
    t = _THINK_RE.sub("", t)
    if _THINK_OPEN_RE.match(t):
        # an opening with no closing -> nothing but reasoning is left
        return ""
    # an orphan closing: everything before it is reasoning
    m = re.search(r"<\s*/\s*(think|thinking|reasoning)\s*>", t, re.IGNORECASE)
    if m:
        t = t[m.end():]
    return t.strip()


# A deterministic clean-up of the descriptions. Despite the instruction, a small model
# (measured on Agents-A1-4B on 2026-09-11) still writes "No text is visible." or "appears to
# be": a stated absence can make the thing appear in the image, a hesitation says nothing.
_ABSENCE_RE = re.compile(r"(?i)\b(?:no|without any)\s+(?:visible\s+|other\s+|legible\s+)?"
                         r"(?:text|people|person|one|words|writing|signage|figures|humans)\b"
                         r"|\bnot visible\b|\b(?:is|are) absent\b")


def clean_description(text):
    """Removes the sentences that state an absence and the hesitant turns of phrase. A
    one-sentence answer (a tag list) is never emptied."""
    t = (text or "").strip()
    kept = [s for s in re.split(r"(?<=[.!?])\s+", t) if not _ABSENCE_RE.search(s)]
    out = " ".join(kept) if kept else t
    out = re.sub(r"(?i)\b(?:appears|seems) to be\b", "is", out)
    out = re.sub(r"(?i)\b(?:appear|seem) to be\b", "are", out)
    out = re.sub(r"(?i),?\s*\b(?:likely|possibly|probably|perhaps)\b,?", "", out)
    return re.sub(r"\s{2,}", " ", out).replace(" ,", ",").replace(" .", ".").strip()


def _ollama_gen_opts(temperature=None):
    """The options shared by the /api/generate calls: keep_alive, a capped context and
    answer length, the temperature when given, the CPU optionally.

    `think: false` turns off the reasoning of the models that support it. The others
    answer 400 -> _ollama_http replays without the field (see the module's docstring)."""
    p = {"stream": False, "keep_alive": OLLAMA_KEEP_ALIVE, "think": False}
    opts = {}
    if OLLAMA_NUM_CTX > 0:
        opts["num_ctx"] = OLLAMA_NUM_CTX
    if OLLAMA_NUM_PREDICT > 0:
        opts["num_predict"] = OLLAMA_NUM_PREDICT
    if temperature is not None:
        opts["temperature"] = float(temperature)
    if OLLAMA_CPU:
        opts["num_gpu"] = 0
    if opts:
        p["options"] = opts
    return p


def _ollama_http(path, payload=None, base=None, timeout=8):
    """The shared transport (Describe, Improve, Vision Mix) -> prompt_improve.http: the
    system proxy ignored (Ollama is local), `think` replayed without the field on an HTTP 400
    (a model with no reasoning refuses it), errors turned into an OllamaError with an
    actionable message."""
    return prompt_improve.http(path, payload, base=base or OLLAMA_URL, timeout=timeout)


def _ollama_vision_models(base=None):
    """The Ollama models really capable of vision. We trust the 'vision' capability
    reported by /api/show (Ollama's authoritative source). Should /api/show fail (an old
    version), we fall back on a clearly multimodal name. We do NOT trust the families
    (clip...), which give false positives (qwen3.6, say)."""
    _VISION_NAME = ("llava", "-vl", "vl:", "moondream", "minicpm-v", "bakllava",
                    "llama3.2-vision", "llama-3.2-vision")
    block = [b.lower() for b in (CONFIG.get("ollama_vision_blocklist") or []) if b]
    tags = _ollama_http("/api/tags", base=base, timeout=5)
    names = [m.get("name") for m in tags.get("models", []) if m.get("name")]
    vision = []
    for n in names:
        if any(b in n.lower() for b in block):   # excluded by the user (config)
            continue
        try:
            info = _ollama_http("/api/show", {"model": n}, base=base, timeout=8)
            caps = [c.lower() for c in (info.get("capabilities") or [])]
            if "vision" in caps:               # Ollama's truth -> we keep it
                vision.append(n)
            elif not info.get("capabilities"):  # the field is absent (an old Ollama)
                if any(k in n.lower() for k in _VISION_NAME):
                    vision.append(n)
        except Exception:
            if any(k in n.lower() for k in _VISION_NAME):
                vision.append(n)
    # The sorting: the real "known" vision models (llava, *-vl, moondream...) first, so
    # that the default choice is a reliable one.
    vision.sort(key=lambda n: 0 if any(k in n.lower() for k in _VISION_NAME) else 1)
    return vision


def _ollama_describe(image, model, base=None, style=None, length=None):
    """Describes the image as a text-to-image prompt through an Ollama vision model, in
    the style and the length chosen in Prompt AI (or the ones passed), then cleans the
    answer up."""
    style, length = style or DESCRIBE_STYLE, length or DESCRIBE_LENGTH
    _dbg(f"ollama describe: url={base or OLLAMA_URL} model={model} style={style} length={length}")
    b64 = _pil_to_b64_jpeg(image, max_side=1024)
    out = _ollama_http("/api/generate",
                       {"model": model, "prompt": describe_instruction(style, length),
                        "images": [b64], **_ollama_gen_opts(OLLAMA_DESCRIBE_TEMPERATURE)},
                       base=base, timeout=180)
    return clean_description(_strip_thinking(out.get("response")))


def _ollama_caption(image, model, base=None):
    """A one-sentence caption through an Ollama vision model: the Caption model
    "ollama:<name>" (Inpaint/Outpaint's Auto-describe, Describe's fallback)."""
    return _ollama_describe(image, model, base=base, style=SHORT_CAPTION_STYLE)


# ----------------------------------------------------------------------------
# Improve (prompt_improve, a module shared by the crispz family)
# ----------------------------------------------------------------------------
# The positive instruction is now the module's (shared by the family): the INPUT FORMAT
# note computed in the code takes over from "prose stays prose, a tag list stays a tag
# list". An instruction shipped by a previous version is not a customisation.
_SHIPPED_IMPROVE_INSTRUCTIONS = (LEGACY_IMPROVE_INSTRUCTION,)


def _improve_settings(config=None):
    """The `ollama_improve` block of config.txt, completed for compatibility:
    - an old CUSTOMISED `ollama_improve_prompt` instruction becomes the positive
      instruction (when the block gives none);
    - keep_alive missing -> `ollama_keep_alive` (0 by default: the model leaves the VRAM,
      which is shared with the image generation)."""
    config = CONFIG if config is None else config
    s = dict(config.get("ollama_improve") or {})
    legacy = str(config.get("ollama_improve_prompt") or "").strip()
    if (not s.get("positive_instruction") and legacy
            and legacy not in [x.strip() for x in _SHIPPED_IMPROVE_INSTRUCTIONS]):
        s["positive_instruction"] = legacy
    if s.get("keep_alive") in (None, ""):
        s["keep_alive"] = config.get("ollama_keep_alive", 0)
    return s


prompt_improve.configure(_improve_settings())
IMPROVE_ENABLED = bool((CONFIG.get("ollama_improve") or {}).get("enabled", True))


def _improve_base(base=None):
    """Improve's Ollama host: `ollama_improve.endpoint`, otherwise the UI's URL (the same
    host as Describe), otherwise OLLAMA_URL."""
    return prompt_improve._setting("endpoint", "") or base or OLLAMA_URL


def _improve_options():
    """The Ollama options specific to the tool (num_ctx, num_predict, a forced CPU),
    merged into the call. `think: false` is set by the module."""
    return dict(_ollama_gen_opts().get("options") or {})


def improve_prompt(text, kind="positive", model=None, base=None, directives=None):
    """Rewrites `text` ('positive' or 'negative'). Returns (the text, the model used).
    The model: the UI's, otherwise `ollama_improve.model`, otherwise the first installed one.
    Raises OllamaError (an actionable message): an empty text, Ollama stopped, no model..."""
    return prompt_improve.improve(text, kind=kind, model=model or None,
                                  base=_improve_base(base), directives=directives,
                                  options=_improve_options())


def improve_negative(text, model=None, base=None, directives=None):
    """Improve of the negative. Returns (the negative, the model|None, a warning|None).
    An empty box: we start from the standard negative (ollama_improve.default_negative) and
    the model extends it; Ollama unreachable -> the standard negative is inserted AS IT IS,
    with a warning that says why. A filled box: an Ollama error -> OllamaError."""
    start = (text or "").strip()
    if start:
        out, used = improve_prompt(start, "negative", model, base, directives)
        return out, used, None
    start = prompt_improve.default_negative()
    try:
        out, used = improve_prompt(start, "negative", model, base, directives)
        return out, used, None
    except OllamaError as e:
        return start, None, f"standard negative inserted as is ({e})"


def list_text_models(base=None):
    """Every Ollama model installed (Improve does not require vision)."""
    return prompt_improve.list_models(base=_improve_base(base))


def _ollama_improve(prompt_text, model, base=None):
    """Compat: a rewrite of the positive prompt (see improve_prompt)."""
    return improve_prompt(prompt_text, "positive", model, base)[0]


def _ollama_compose(captions, model, base=None):
    """'Fake Omni': merges several image descriptions into ONE single prompt."""
    listing = "\n".join(f"Image {i + 1}: {c}" for i, c in enumerate(captions) if c)
    instr = (COMPOSE_INSTRUCTION.replace("{descriptions}", listing)
             if "{descriptions}" in COMPOSE_INSTRUCTION
             else f"{COMPOSE_INSTRUCTION}\n\n{listing}")
    out = _ollama_http("/api/generate", {"model": model, "prompt": instr, **_ollama_gen_opts()},
                       base=base, timeout=120)
    return _strip_thinking(out.get("response"))
