"""crispz-studio - prompt helpers: styles (Fooocus) + wildcards (__name__).

Pulled out of app.py. It depends only on cz_core (HERE/CONFIG/_prefs) + the stdlib. The
UI handlers (the wildcard manager, the style search) stay in app.py.

Note: WILDCARDS_DIR is reassignable at run time (set_wildcards_dir). Readers outside this
module use `cz_prompt.WILDCARDS_DIR` to see the up-to-date value.

"""

import os
import re
import json
import random

from cz_core import HERE, CONFIG, _prefs, _log
from prompt_variants import expand_variants, has_variants

_FALLBACK_STYLES = {
    "Fooocus Cinematic": {"prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
                          "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"},
    "SAI Photographic": {"prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
                         "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly"},
    "SAI Anime": {"prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime, highly detailed",
                  "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast"},
}


def _load_styles():
    """Loads the style library from styles/*.json (the Fooocus format:
    {name, prompt with {prompt}, negative_prompt}). Empty -> the fallback."""
    out = {}
    sdir = os.path.join(HERE, "styles")
    if os.path.isdir(sdir):
        for fn in sorted(os.listdir(sdir)):
            if not fn.lower().endswith(".json"):
                continue
            try:
                with open(os.path.join(sdir, fn), "r", encoding="utf-8") as f:
                    for s in (json.load(f) or []):
                        name = s.get("name")
                        if name:
                            out[name] = {"prompt": s.get("prompt"),
                                         "negative_prompt": s.get("negative_prompt", "")}
            except Exception:
                pass
    return out


STYLES = _load_styles() or _FALLBACK_STYLES

WILDCARDS_DIR = (os.environ.get("WILDCARDS_DIR") or _prefs.get("wildcards_dir")
                 or CONFIG.get("wildcards_dir") or os.path.join(HERE, "wildcards"))


def set_wildcards_dir(path):
    global WILDCARDS_DIR
    if path:
        WILDCARDS_DIR = path


# The LoRA tags in the prompt, the A1111/ComfyUI syntax: <lora:name> or
# <lora:name:weight>. The name can be a relative path ('chars/my_lora.safetensors'),
# with or without its extension. Those tags must NEVER reach the text encoder: they are
# extracted here and resolved/activated by cz_pipeline.consume_prompt_loras.
LORA_TAG_RE = re.compile(r"<\s*lora\s*:\s*([^:<>]+?)\s*(?::\s*([-+]?\d*\.?\d+)\s*)?>",
                         re.IGNORECASE)


def extract_lora_tags(text):
    """Extracts the <lora:name[:weight]> tags from a prompt. Returns (cleaned_text, tags)
    with tags = a list of (name, weight_or_None) in order of appearance (duplicate names
    deduplicated, the last occurrence wins — as in A1111). The cleaned text keeps neither
    the tags nor the double commas/spaces they leave behind."""
    if not text or "<" not in text:
        return text, []
    tags = {}
    for m in LORA_TAG_RE.finditer(text):
        name = m.group(1).strip()
        if not name:
            continue
        w = None
        if m.group(2) is not None:
            try:
                w = float(m.group(2))
            except ValueError:
                w = None
        tags[name] = w
    clean = LORA_TAG_RE.sub("", text)
    # Cleaning up what is left where the tag was: double spaces, a space before a
    # comma, consecutive commas ('a, <tag>, b' -> 'a, b').
    clean = re.sub(r"\s{2,}", " ", clean)
    clean = re.sub(r"\s+,", ",", clean)
    clean = re.sub(r"(?:,\s*){2,}", ", ", clean)
    clean = clean.strip(" ,")
    return clean, [(n, w) for n, w in tags.items()]


def strip_lora_tags(text):
    """Removes the remaining <lora:...> tags (injected by a wildcard, say) without
    activating them: a fragment of syntax must never go to the text encoder."""
    if not text or "<" not in text:
        return text
    clean, _tags = extract_lora_tags(text)
    return clean


def _seed_rng(seed):
    """A reproducible RNG when seed>=0 (the same wildcards/styles for the same seed)."""
    try:
        s = int(seed)
        return random.Random(s) if s >= 0 else random.Random()
    except Exception:
        return random.Random()


def list_wildcards():
    if not os.path.isdir(WILDCARDS_DIR):
        return []
    return sorted(f[:-4] for f in os.listdir(WILDCARDS_DIR) if f.lower().endswith(".txt"))


READ_WILDCARDS_IN_ORDER = bool(CONFIG.get("wildcards_in_order", False))


def set_wildcards_in_order(v):
    """Switches the wildcards' reading mode (random <-> in order)."""
    global READ_WILDCARDS_IN_ORDER
    READ_WILDCARDS_IN_ORDER = bool(v)
    return f"Wildcards: {'in order' if READ_WILDCARDS_IN_ORDER else 'random'}"


def _apply_wildcards(text, rng=None, index=None):
    """Expands the {a|b|c} variants (prompt_variants) and the __name__ (one line of
    wildcards/name.txt), nesting included.
    By default: a RANDOM draw (rng, reproducible by seed). With READ_WILDCARDS_IN_ORDER
    and an index given: the option (index % n) / the line (index % line_count) -> it walks
    the options along the batch, deterministically (Fooocus' 'read wildcards in order'
    style). Both syntaxes share that boolean and that rng.

    The order (identical to Fooocus2026's apply_wildcards): on every pass, ONE level of
    groups is expanded (the innermost one) BEFORE a __name__ is looked for. So a
    placeholder placed in an option that was not drawn is never expanded. When no
    placeholder is left but a nested group still is, we make another pass.
    A text with neither a group nor a placeholder makes NO draw at all: the seeds of the
    existing prompts give the same image again."""
    if not text or ("__" not in text and not has_variants(text)):
        return text
    raw = text
    rng = rng or random.Random()
    in_order = READ_WILDCARDS_IN_ORDER and index is not None
    idx = int(index) if index is not None else 0
    for _ in range(64):  # an anti-loop guard rail
        text = expand_variants(text, rng, index=idx, in_order=in_order, max_depth=1)
        m = re.search(r"__([A-Za-z0-9_\-/]+)__", text)
        if not m:
            if has_variants(text):
                continue
            break
        name = m.group(1)
        path = os.path.join(WILDCARDS_DIR, name + ".txt")
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
    if has_variants(raw):
        _log(f"{raw} -> {text}", mod="Variants")
    return text


def resolve_seed(seed):
    """A concrete seed: -1 (or invalid) -> a random draw. The {a|b|c} variants and the
    wildcards are tied to the seed; an unresolved -1 seed would make them irreproducible
    and absent from the metadata."""
    try:
        s = int(seed)
    except (TypeError, ValueError):
        s = -1
    return s if s >= 0 else random.randint(0, 2**31 - 1)


def expand_prompt_pair(prompt, negative, seed, index=None):
    """The {a|b|c} variants + wildcards of the positive AND the negative for ONE image.
    Each text has its own random.Random(seed): the same draws for the same seed, and
    changing the positive does not change the negative's draws."""
    return (_apply_wildcards(prompt, _seed_rng(seed), index=index),
            _apply_wildcards(negative, _seed_rng(seed), index=index))


def _pick_styles(selected, randomize):
    """When randomize: draws 1 style at random from the selection (or from ALL the
    styles when nothing is selected). Otherwise it returns the selection as it is."""
    if not randomize:
        return list(selected or [])
    pool = [s for s in (selected or []) if s in STYLES] or list(STYLES)
    return [random.choice(pool)] if pool else []


def _apply_styles(prompt, negative, style_names):
    """Applies the Fooocus styles: it chains the {prompt} templates and accumulates the
    negative_prompt. Returns (final_prompt, final_negative)."""
    cur = (prompt or "").strip()
    negs = [(negative or "").strip()] if (negative or "").strip() else []
    for n in (style_names or []):
        s = STYLES.get(n)
        if not s:
            continue
        tmpl = s.get("prompt")
        if tmpl and "{prompt}" in tmpl:
            cur = tmpl.replace("{prompt}", cur).strip()
        elif tmpl:
            cur = f"{cur}, {tmpl}".strip(" ,")
        neg = s.get("negative_prompt")
        if neg:
            negs.append(neg)
    return cur.strip(" ,"), ", ".join(negs)
