"""crispz-studio - prompt helpers: styles (Fooocus) + wildcards (__name__).

Extrait de app.py. Ne depend que de cz_core (HERE/CONFIG/_prefs) + stdlib. Les
handlers d'UI (gestionnaire de wildcards, recherche de styles) restent dans app.py.

Note: WILDCARDS_DIR est reassignable a l'execution (set_wildcards_dir). Les lecteurs
hors de ce module utilisent `cz_prompt.WILDCARDS_DIR` pour voir la valeur a jour.
"""

import os
import re
import json
import random

from cz_core import HERE, CONFIG, _prefs

_FALLBACK_STYLES = {
    "Fooocus Cinematic": {"prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
                          "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"},
    "SAI Photographic": {"prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
                         "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly"},
    "SAI Anime": {"prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime, highly detailed",
                  "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast"},
}


def _load_styles():
    """Charge la biblio de styles depuis styles/*.json (format Fooocus:
    {name, prompt avec {prompt}, negative_prompt}). Vide -> fallback."""
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


# Tags LoRA dans le prompt, syntaxe A1111/ComfyUI: <lora:nom> ou <lora:nom:poids>.
# Le nom peut etre un chemin relatif ('perso/ma_lora.safetensors'), avec ou sans
# extension. Ces tags ne doivent JAMAIS atteindre l'encodeur de texte: ils sont
# extraits ici et resolus/actives par cz_pipeline.consume_prompt_loras.
LORA_TAG_RE = re.compile(r"<\s*lora\s*:\s*([^:<>]+?)\s*(?::\s*([-+]?\d*\.?\d+)\s*)?>",
                         re.IGNORECASE)


def extract_lora_tags(text):
    """Extrait les tags <lora:nom[:poids]> d'un prompt. Renvoie (texte_nettoye, tags)
    avec tags = liste de (nom, poids_ou_None) dans l'ordre d'apparition (doublons de nom
    dedoublonnes, la derniere occurrence gagne — comme A1111). Le texte nettoye ne garde
    ni les tags ni les doubles virgules/espaces qu'ils laissent derriere eux."""
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
    # Nettoyage des restes la ou etait le tag: espaces doubles, espace avant virgule,
    # virgules consecutives ('a, <tag>, b' -> 'a, b').
    clean = re.sub(r"\s{2,}", " ", clean)
    clean = re.sub(r"\s+,", ",", clean)
    clean = re.sub(r"(?:,\s*){2,}", ", ", clean)
    clean = clean.strip(" ,")
    return clean, [(n, w) for n, w in tags.items()]


def strip_lora_tags(text):
    """Retire les tags <lora:...> restants (ex. injectes par un wildcard) sans les
    activer: un fragment de syntaxe ne doit jamais partir a l'encodeur de texte."""
    if not text or "<" not in text:
        return text
    clean, _tags = extract_lora_tags(text)
    return clean


def _seed_rng(seed):
    """RNG reproductible si seed>=0 (memes wildcards/styles pour une meme seed)."""
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
    """Bascule le mode de lecture des wildcards (aleatoire <-> dans l'ordre)."""
    global READ_WILDCARDS_IN_ORDER
    READ_WILDCARDS_IN_ORDER = bool(v)
    return f"Wildcards: {'in order' if READ_WILDCARDS_IN_ORDER else 'random'}"


def _apply_wildcards(text, rng=None, index=None):
    """Remplace les __nom__ par une ligne de wildcards/nom.txt (gere l'imbrication).
    Par defaut: ligne ALEATOIRE (rng, reproductible par seed). Si READ_WILDCARDS_IN_ORDER
    et index fourni: prend la ligne (index % nb_lignes) -> parcourt le fichier au fil du
    batch, de facon deterministe (facon Fooocus 'read wildcards in order')."""
    if not text or "__" not in text:
        return text
    import re
    rng = rng or random
    in_order = READ_WILDCARDS_IN_ORDER and index is not None
    for _ in range(64):  # garde-fou anti-boucle
        m = re.search(r"__([A-Za-z0-9_\-/]+)__", text)
        if not m:
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
    return text


def _pick_styles(selected, randomize):
    """Si randomize: tire 1 style au hasard dans la selection (ou dans TOUS les
    styles si rien n'est selectionne). Sinon renvoie la selection telle quelle."""
    if not randomize:
        return list(selected or [])
    pool = [s for s in (selected or []) if s in STYLES] or list(STYLES)
    return [random.choice(pool)] if pool else []


def _apply_styles(prompt, negative, style_names):
    """Applique les styles Fooocus: enchaine les templates {prompt} et cumule les
    negative_prompt. Renvoie (prompt_final, negative_final)."""
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
