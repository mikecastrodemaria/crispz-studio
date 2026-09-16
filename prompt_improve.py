"""custom-28 / custom-31 / custom-34: Improve a prompt via a local Ollama text model.

Rewrites a prompt richer while keeping its intent. The positive prompt is made more
vivid and detailed; the negative prompt is expanded and tidied into a fuller list of
defects to avoid, keeping every term already present. Optional user directives are
added for one call, the input format (tags or prose) is detected in code and stated to
the model, and the {a|b|c} / __wildcard__ syntax is protected.

Shared by the crispz family, adapted from Fooocus2026 (modules/ollama_improve.py and the
transport of modules/ollama_describe.py). Keep the copies identical across repos. The
constants and the pure functions are the reference ones; two things differ:
  - settings come from `configure(dict)` (the tool's `ollama_improve` config block)
    instead of being read from modules.config;
  - the HTTP transport lives here (standard library only): default endpoint
    http://127.0.0.1:11434 (`localhost` resolves to ::1 first on Windows Python and
    times out), system proxy variables ignored, `think` dropped and replayed on HTTP 400,
    errors turned into OllamaError with an actionable message.
Import-safe: no side effect, no tool import.
"""
import json
import re
import sys
import urllib.error
import urllib.request

from prompt_variants import uses_dynamic_syntax

DEFAULT_ENDPOINT = "http://127.0.0.1:11434"

# Config block `ollama_improve`, same keys in every tool. keep_alive 0: the model leaves
# the VRAM right after the answer, the GPU is shared with image generation.
DEFAULTS = {
    "enabled": True,
    "endpoint": "",            # '' = the tool's Ollama URL (same host as Describe)
    "model": "",               # '' = the model picked in the UI, else the first installed
    "timeout": 120,
    "temperature": 0.7,
    "keep_alive": 0,
    "positive_instruction": "",
    "negative_instruction": "",
    "default_negative": "",    # '' = DEFAULT_NEGATIVE
    "format": "auto",          # auto | tags | prose | off
}

_SETTINGS = {}


def configure(settings=None):
    """Sets the `ollama_improve` block used by every function of this module."""
    global _SETTINGS
    _SETTINGS = dict(settings or {})


def _setting(key, default):
    value = _SETTINGS.get(key)
    return default if value in (None, "") else value


class OllamaError(RuntimeError):
    """Message readable by the user (Ollama stopped, no model, empty answer...)."""


# ------------------------------------------------------------------ transport ---
# Ollama is local or on the LAN, never behind an HTTP proxy. urllib's default opener obeys
# HTTP_PROXY / HTTPS_PROXY (Pinokio, company networks): the request to the local server
# then went to the proxy and timed out.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def normalize_endpoint(url):
    """Endpoint without trailing slash; empty -> DEFAULT_ENDPOINT; host `localhost` ->
    127.0.0.1 (Windows Python tries ::1 first and times out when Ollama listens on IPv4)."""
    url = (url or "").strip().rstrip("/") or DEFAULT_ENDPOINT
    return re.sub(r"^(https?://)localhost(?=[:/]|$)", r"\g<1>127.0.0.1", url,
                  flags=re.IGNORECASE)


def endpoint():
    return normalize_endpoint(_setting("endpoint", DEFAULT_ENDPOINT))


def http(path, payload=None, base=None, timeout=8):
    """GET (payload None) or POST JSON to Ollama; returns the decoded JSON answer.
    Raises OllamaError with an actionable message."""
    b = normalize_endpoint(base or endpoint())

    def _call(pl):
        data = json.dumps(pl).encode("utf-8") if pl is not None else None
        req = urllib.request.Request(b + path, data=data,
                                     headers={"Content-Type": "application/json"} if data else {})
        with _OPENER.open(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    try:
        try:
            return _call(payload)
        except urllib.error.HTTPError as e:
            # a model without reasoning refuses `think` (400): replay without the field
            if e.code == 400 and isinstance(payload, dict) and "think" in payload:
                return _call({k: v for k, v in payload.items() if k != "think"})
            raise
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = json.loads(e.read().decode("utf-8", "replace")).get("error", "")
        except Exception:
            pass
        if e.code == 404 and isinstance(payload, dict) and payload.get("model"):
            raise OllamaError(f'model "{payload["model"]}" not found in Ollama '
                              f'(ollama pull {payload["model"]})')
        raise OllamaError(f"Ollama answered HTTP {e.code} {detail or e.reason}")
    except (urllib.error.URLError, OSError) as e:
        raise OllamaError(f"Ollama unreachable at {b} ({getattr(e, 'reason', e)}). Start "
                          f"Ollama or check the Ollama URL setting.")


# Reasoning tags of "thinking" models (non-greedy, DOTALL).
_THINK_RE = re.compile(r"<\s*(think|thinking|reasoning)\s*>.*?<\s*/\s*\1\s*>",
                       re.IGNORECASE | re.DOTALL)
_THINK_OPEN_RE = re.compile(r"^\s*<\s*(think|thinking|reasoning)\s*>", re.IGNORECASE)


def strip_thinking(text):
    """Removes a reasoning model's inner monologue: closed block, block opened and never
    closed (only reasoning is left), orphan closing tag."""
    t = _THINK_RE.sub("", text or "")
    if _THINK_OPEN_RE.match(t):
        return ""
    m = re.search(r"<\s*/\s*(think|thinking|reasoning)\s*>", t, re.IGNORECASE)
    if m:
        t = t[m.end():]
    return t.strip()


# --------------------------------------------------------------- instructions ---
IMPROVE_POSITIVE = (
    "You are an expert text-to-image prompt writer. Rewrite the following prompt to be more "
    "vivid and detailed while keeping the SAME subject and intent. Prefer concrete visual "
    "terms; keep the format of the input (a tag list stays a tag list, prose stays prose, "
    "see INPUT FORMAT when given); do not pad it with generic quality filler (masterpiece, "
    "best quality, 8k). Output ONLY the improved prompt, on one line, no preamble, no "
    "quotes, no explanation.\n\nPROMPT: {prompt}")

# custom-34: the input format is detected in code (small models guess it badly, above all
# with braces or wildcards in the text) and stated to the model. Setting `format`:
# 'auto' (default), 'tags', 'prose' or 'off'.
FORMAT_NOTES = {
    "tags": ("INPUT FORMAT: comma-separated tags. Answer in the SAME format: one line of "
             "comma-separated tags or short phrases (2 to 5 words each), most specific first; "
             "no full sentences, no narrative, no bullet points."),
    "prose": ("INPUT FORMAT: prose. Answer in the SAME format: one flowing paragraph of "
              "natural sentences, no comma-separated tag list, no bullet points, no "
              "keyword dump at the end."),
}
_FORMAT_CHOICES = ("auto", "tags", "prose", "off")
_SENTENCE_END_RE = re.compile(r"[.!?](?:\s|$)")
_STRIP_SYNTAX_RE = re.compile(r"\{[^{}]*\}|__[\w-]+__|<lora:[^>]*>|\([^()]*:\s*[\d.]+\)")


def detect_format(text):
    """'tags' or 'prose' for the positive prompt, from the text alone.

    Dynamic syntax ({a|b}, __wildcards__, <lora:...>, (word:1.2)) is blanked first so it
    never tips the balance. Sentence punctuation inside the text means prose; otherwise
    the mean length of the comma-separated fragments decides: up to 4 words a fragment is
    a tag list, longer fragments read as prose. A short single phrase ("a fox") is tags.
    """
    cleaned = _STRIP_SYNTAX_RE.sub(" x ", text or "").strip()
    if not cleaned:
        return "tags"
    inner = cleaned.rstrip(".!? ")
    if _SENTENCE_END_RE.search(inner):
        return "prose"
    fragments = [f.strip() for f in cleaned.split(",") if f.strip()]
    if not fragments:
        return "tags"
    mean_words = sum(len(f.split()) for f in fragments) / len(fragments)
    return "tags" if mean_words <= 4 else "prose"


def _format_note(text):
    """The INPUT FORMAT block for `text`, or '' (setting off, empty text)."""
    mode = str(_setting("format", "auto")).strip().lower()
    if mode not in _FORMAT_CHOICES:
        mode = "auto"
    if mode == "off" or not (text or "").strip():
        return ""
    fmt = detect_format(text) if mode == "auto" else mode
    return FORMAT_NOTES[fmt]


IMPROVE_NEGATIVE = (
    "You are an expert text-to-image prompt writer. The following is a NEGATIVE prompt: a "
    "comma-separated list of things that must NOT appear in the image. Expand and tidy it "
    "into a fuller comma-separated list of common defects and unwanted elements to avoid "
    "(anatomy errors, artifacts, low quality, watermarks, text...), KEEPING every term "
    "already present and removing duplicates. Output ONLY the negative prompt, on one line, "
    "no preamble, no quotes, no explanation.\n\nNEGATIVE PROMPT: {prompt}")

# custom-29: added to the instruction only when the text uses the dynamic syntax, so the
# model keeps {a|b|c} groups and __wildcard__ placeholders instead of expanding them.
SYNTAX_NOTE = (
    "The prompt uses dynamic syntax that is resolved later, at generation time, "
    "and must be kept EXACTLY as written: {a|b|c} is a variant group (one option is "
    "picked per image), {2$$a|b|c} picks two options, __name__ is a wildcard file "
    "placeholder. Keep every group and placeholder verbatim: do not expand, reorder, "
    "merge, translate or drop them. You may improve the text around them and the "
    "wording inside each option, as long as the braces, the | separators and the "
    "__names__ stay intact.")

_LABEL_RE = re.compile(r"\n\n(?:NEGATIVE )?PROMPT:")

# custom-31: what Improve negative starts from when the negative prompt is empty. A plain
# SDXL baseline, overridable with the `default_negative` setting.
DEFAULT_NEGATIVE = (
    "lowres, worst quality, low quality, jpeg artifacts, blurry, out of focus, "
    "bad anatomy, bad proportions, bad hands, missing fingers, extra digits, fewer digits, "
    "extra limbs, deformed, disfigured, mutated, cropped, cut off, "
    "text, watermark, signature, logo, username")

# custom-31: user directives, appended to the instruction for one call.
DIRECTIVES_HEAD = (
    "USER DIRECTIVES for this rewrite (apply them on top of the rules above; when they "
    "conflict with the rules above, the directives win):\n")


def _insert_before_label(tpl, block):
    """Inserts `block` right before the final PROMPT: / NEGATIVE PROMPT: label (else at the end)."""
    last = None
    for last in _LABEL_RE.finditer(tpl):
        pass
    if last is None:                        # custom instruction without the label: append
        return tpl + "\n\n" + block
    return tpl[:last.start()] + "\n\n" + block + tpl[last.start():]


def _instruction(kind, text="", directives=None):
    """Instruction for this `kind`: format note (positive only, custom-34), then the syntax
    note when `text` uses it (custom-29), then the user's directives (custom-31), all
    before the final label."""
    if kind == "negative":
        tpl = _setting("negative_instruction", IMPROVE_NEGATIVE)
    else:
        tpl = _setting("positive_instruction", IMPROVE_POSITIVE)
        note = _format_note(text)
        if note:
            tpl = _insert_before_label(tpl, note)
    if uses_dynamic_syntax(text):
        tpl = _insert_before_label(tpl, SYNTAX_NOTE)
    directives = (directives or "").strip()
    if directives:
        tpl = _insert_before_label(tpl, DIRECTIVES_HEAD + directives)
    return tpl


def default_negative():
    """The starting negative when the box is empty (setting `default_negative`)."""
    return str(_setting("default_negative", DEFAULT_NEGATIVE)).strip() or DEFAULT_NEGATIVE


# ---------------------------------------------------------------------- calls ---
def list_models(base=None):
    """Every installed Ollama model (rewriting text does not need vision)."""
    tags = http("/api/tags", base=base, timeout=5).get("models") or []
    return [m.get("name") for m in tags if m.get("name")]


def improve(text, kind="positive", model=None, base=None, timeout=None, temperature=None,
            directives=None, options=None):
    """Rewrites `text` (kind 'positive' or 'negative'). Returns (improved text, model used).
    `model`: the model picked in the UI; else the `model` setting; else the first installed.
    `directives`: the user's free instructions for this call (custom-31).
    `options`: extra Ollama options merged into the call (num_ctx, num_gpu...).
    Raises OllamaError with an actionable message (Ollama stopped, no model, empty...)."""
    text = (text or "").strip()
    if not text:
        raise OllamaError("nothing to improve: the prompt is empty")
    model = model or _setting("model", "")
    if not model:
        found = list_models(base)
        if not found:
            raise OllamaError('no model in Ollama: pull one first '
                              '(for example "ollama pull llama3.1:8b")')
        model = found[0]
    opts = dict(options or {})
    opts["temperature"] = float(temperature if temperature is not None
                                else _setting("temperature", 0.7))
    payload = {
        "model": model,
        "prompt": _instruction(kind, text, directives).replace("{prompt}", text),
        "stream": False,
        "think": False,
        "keep_alive": _setting("keep_alive", 0),
        "options": opts,
    }
    out = http("/api/generate", payload, base=base,
               timeout=int(timeout or _setting("timeout", 120)))
    result = strip_thinking(out.get("response") or "").strip().strip('"').strip()
    if not result:
        raise OllamaError(f'"{model}" returned an empty result (a reasoning model may have '
                          'spent its whole answer thinking)')
    if uses_dynamic_syntax(text) and not uses_dynamic_syntax(result):
        print(f'[Improve] "{model}" dropped the {{a|b|c}} / __wildcard__ syntax from the '
              'prompt; check the result before generating.', file=sys.stderr, flush=True)
    return result, model
