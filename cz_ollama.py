"""crispz-studio - Ollama integration (Describe / Improve / Vision Mix).

Extrait de app.py. Appelle l'API HTTP locale d'Ollama (/api/tags, /api/show,
/api/generate). Ne depend que de cz_core (config, log, b64). Les handlers d'UI
(_ui_describe...) restent dans app.py (couche Gradio) et appellent ces fonctions.
"""

import os
import json
import urllib.request

import cz_core
from cz_core import (
    CONFIG, DESCRIBE_INSTRUCTION, IMPROVE_INSTRUCTION, COMPOSE_INSTRUCTION,
    _prefs, _dbg, _pil_to_b64_jpeg,
)

# URL Ollama (Describe image->prompt + Improve prompt). Configurable, persistee.
OLLAMA_URL = (os.environ.get("OLLAMA_URL") or _prefs.get("ollama_url")
              or CONFIG.get("ollama_url") or "http://localhost:11434")
# Duree de maintien du modele Ollama en VRAM apres un appel (keep_alive). 0 =
# decharge immediatement -> libere la VRAM avant la generation Z-Image.
OLLAMA_KEEP_ALIVE = CONFIG.get("ollama_keep_alive", 0)
# Force Ollama sur CPU (num_gpu=0) -> 0 VRAM partagee avec Z-Image (plus lent).
OLLAMA_CPU = bool(CONFIG.get("ollama_cpu", False))


def _ollama_gen_opts():
    """Options communes pour /api/generate (keep_alive + CPU optionnel)."""
    p = {"stream": False, "keep_alive": OLLAMA_KEEP_ALIVE}
    if OLLAMA_CPU:
        p["options"] = {"num_gpu": 0}
    return p


def _ollama_http(path, payload=None, base=None, timeout=8):
    b = (base or OLLAMA_URL).rstrip("/")
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(b + path, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _ollama_vision_models(base=None):
    """Modeles Ollama reellement capables de vision. On se fie a la capacite
    'vision' rapportee par /api/show (source autoritaire d'Ollama). Si /api/show
    echoue (vieille version), repli sur un nom clairement multimodal. On NE se fie
    PAS aux familles (clip...) qui donnent des faux positifs (ex. qwen3.6)."""
    _VISION_NAME = ("llava", "-vl", "vl:", "moondream", "minicpm-v", "bakllava",
                    "llama3.2-vision", "llama-3.2-vision")
    block = [b.lower() for b in (CONFIG.get("ollama_vision_blocklist") or []) if b]
    tags = _ollama_http("/api/tags", base=base, timeout=5)
    names = [m.get("name") for m in tags.get("models", []) if m.get("name")]
    vision = []
    for n in names:
        if any(b in n.lower() for b in block):   # exclu par l'utilisateur (config)
            continue
        try:
            info = _ollama_http("/api/show", {"model": n}, base=base, timeout=8)
            caps = [c.lower() for c in (info.get("capabilities") or [])]
            if "vision" in caps:               # verite Ollama -> on garde
                vision.append(n)
            elif not info.get("capabilities"):  # champ absent (vieux Ollama)
                if any(k in n.lower() for k in _VISION_NAME):
                    vision.append(n)
        except Exception:
            if any(k in n.lower() for k in _VISION_NAME):
                vision.append(n)
    # Tri: vrais modeles vision "connus" (llava, *-vl, moondream...) d'abord, pour
    # que le choix par defaut soit fiable.
    vision.sort(key=lambda n: 0 if any(k in n.lower() for k in _VISION_NAME) else 1)
    return vision


def _ollama_describe(image, model, base=None):
    """Decrit l'image en un prompt text-to-image via un modele vision Ollama."""
    _dbg(f"ollama describe: url={base or OLLAMA_URL} model={model}")
    b64 = _pil_to_b64_jpeg(image, max_side=1024)
    out = _ollama_http("/api/generate",
                       {"model": model, "prompt": DESCRIBE_INSTRUCTION, "images": [b64],
                        **_ollama_gen_opts()}, base=base, timeout=180)
    return (out.get("response") or "").strip()


def _ollama_improve(prompt_text, model, base=None):
    """Reecrit un prompt pour le rendre plus riche, via Ollama (modele texte/vision).
    Instruction editable dans config.txt (ollama_improve_prompt, {prompt} = le prompt)."""
    pt = prompt_text or ""
    instr = (IMPROVE_INSTRUCTION.replace("{prompt}", pt) if "{prompt}" in IMPROVE_INSTRUCTION
             else f"{IMPROVE_INSTRUCTION}\n\nPROMPT: {pt}")
    out = _ollama_http("/api/generate", {"model": model, "prompt": instr, **_ollama_gen_opts()},
                       base=base, timeout=120)
    return (out.get("response") or "").strip()


_IMPROVE_LOCAL_KEYWORDS = CONFIG.get(
    "improve_local_keywords",
    "highly detailed, sharp focus, professional photography, intricate details, "
    "natural lighting, high quality, 8k")


def _local_improve(prompt_text):
    """Amelioration LOCALE du prompt, SANS Ollama et sans modele (rule-based): ajoute les
    mots-cles de qualite (config 'improve_local_keywords') encore absents du prompt.
    Instantane, 100% offline. Fallback quand aucun modele Ollama n'est selectionne."""
    pt = (prompt_text or "").strip().rstrip(",").strip()
    low = pt.lower()
    adds = [k.strip() for k in _IMPROVE_LOCAL_KEYWORDS.split(",") if k.strip()]
    extra = [k for k in adds if k.lower() not in low]
    if not extra:
        return pt
    return (pt + (", " if pt else "") + ", ".join(extra)).strip()


def _ollama_compose(captions, model, base=None):
    """'Faux Omni': fusionne plusieurs descriptions d'images en UN seul prompt."""
    listing = "\n".join(f"Image {i + 1}: {c}" for i, c in enumerate(captions) if c)
    instr = (COMPOSE_INSTRUCTION.replace("{descriptions}", listing)
             if "{descriptions}" in COMPOSE_INSTRUCTION
             else f"{COMPOSE_INSTRUCTION}\n\n{listing}")
    out = _ollama_http("/api/generate", {"model": model, "prompt": instr, **_ollama_gen_opts()},
                       base=base, timeout=120)
    return (out.get("response") or "").strip()
