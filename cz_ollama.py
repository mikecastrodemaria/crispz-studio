"""crispz-studio - Ollama integration (Describe / Improve / Vision Mix).

Extrait de app.py. Appelle l'API HTTP locale d'Ollama (/api/tags, /api/show,
/api/generate). Ne depend que de cz_core (config, log, b64). Les handlers d'UI
(_ui_describe...) restent dans app.py (couche Gradio) et appellent ces fonctions.
"""

import os

import cz_core
import prompt_improve
from prompt_improve import OllamaError  # noqa: F401  (re-export pour l'UI et la CLI)
from cz_core import (
    CONFIG, DESCRIBE_INSTRUCTION, IMPROVE_INSTRUCTION, COMPOSE_INSTRUCTION,
    _prefs, _dbg, _pil_to_b64_jpeg,
)

# URL Ollama (Describe image->prompt + Improve prompt). Configurable, persistee.
# 127.0.0.1 par defaut, et un 'localhost' deja configure est reecrit: sous Windows,
# Python tente ::1 d'abord et l'appel expire quand Ollama n'ecoute qu'en IPv4.
OLLAMA_URL = prompt_improve.normalize_endpoint(
    os.environ.get("OLLAMA_URL") or _prefs.get("ollama_url")
    or CONFIG.get("ollama_url") or prompt_improve.DEFAULT_ENDPOINT)
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
    """Transport commun (Describe, Improve, Vision Mix) -> prompt_improve.http: proxy
    systeme ignore (Ollama est local), `think` rejoue sans le champ sur HTTP 400, erreurs
    en OllamaError au message actionnable."""
    return prompt_improve.http(path, payload, base=base or OLLAMA_URL, timeout=timeout)


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


# ----------------------------------------------------------------------------
# Improve (prompt_improve, module partage par la famille crispz)
# ----------------------------------------------------------------------------
# Consigne livree par les versions precedentes (config.txt / cz_core). Une valeur
# IDENTIQUE n'est pas une personnalisation: la consigne positive du module s'applique.
_SHIPPED_IMPROVE_INSTRUCTIONS = (
    "Rewrite the following text-to-image prompt to be more vivid and detailed while keeping "
    "the same subject and intent. Output ONLY the improved prompt (comma-separated), no "
    "preamble.\n\nPROMPT: {prompt}",
)


def _improve_settings(config=None):
    """Bloc `ollama_improve` de config.txt, complete pour la compatibilite:
    - une ancienne consigne `ollama_improve_prompt` PERSONNALISEE devient la consigne
      positive (si le bloc n'en donne pas);
    - keep_alive absent -> `ollama_keep_alive` (0 par defaut: le modele quitte la VRAM,
      partagee avec la generation d'image)."""
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
    """Hote Ollama d'Improve: `ollama_improve.endpoint`, sinon l'URL de l'UI (meme hote
    que Describe), sinon OLLAMA_URL."""
    return prompt_improve._setting("endpoint", "") or base or OLLAMA_URL


def _improve_options():
    """Options Ollama propres a l'outil (CPU force...), fusionnees dans l'appel."""
    return dict(_ollama_gen_opts().get("options") or {})


def improve_prompt(text, kind="positive", model=None, base=None, directives=None):
    """Reecrit `text` ('positive' ou 'negative'). Renvoie (texte, modele utilise).
    Modele: celui de l'UI, sinon `ollama_improve.model`, sinon le premier installe.
    Leve OllamaError (message actionnable): texte vide, Ollama arrete, aucun modele..."""
    return prompt_improve.improve(text, kind=kind, model=model or None,
                                  base=_improve_base(base), directives=directives,
                                  options=_improve_options())


def improve_negative(text, model=None, base=None, directives=None):
    """Improve du negatif. Renvoie (negatif, modele|None, avertissement|None).
    Case vide: on part du negatif standard (ollama_improve.default_negative) et le modele
    l'etend; Ollama injoignable -> le negatif standard est insere TEL QUEL, avec un
    avertissement qui dit pourquoi. Case remplie: erreur Ollama -> OllamaError."""
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
    """Tous les modeles Ollama installes (Improve n'exige pas la vision)."""
    return prompt_improve.list_models(base=_improve_base(base))


def _ollama_improve(prompt_text, model, base=None):
    """Compat: reecriture du prompt positif (cf. improve_prompt)."""
    return improve_prompt(prompt_text, "positive", model, base)[0]


def _ollama_compose(captions, model, base=None):
    """'Faux Omni': fusionne plusieurs descriptions d'images en UN seul prompt."""
    listing = "\n".join(f"Image {i + 1}: {c}" for i, c in enumerate(captions) if c)
    instr = (COMPOSE_INSTRUCTION.replace("{descriptions}", listing)
             if "{descriptions}" in COMPOSE_INSTRUCTION
             else f"{COMPOSE_INSTRUCTION}\n\n{listing}")
    out = _ollama_http("/api/generate", {"model": model, "prompt": instr, **_ollama_gen_opts()},
                       base=base, timeout=120)
    return (out.get("response") or "").strip()
