"""crispz-studio - core foundation (config, paths, logging, device).

Extrait de app.py. Aucune dependance sur le reste du projet (app.py et les autres
modules importent cz_core, jamais l'inverse). Contient:
  - chemins (HERE, PREFS_PATH, CONFIG_PATH...) et constantes par defaut (DEFAULT_*)
  - chargement de la config JSON (config.txt -> config-sample.txt) -> CONFIG
  - profils par modele (MODEL_PROFILES / profile_for_model)
  - instructions Ollama (DESCRIBE/IMPROVE/COMPOSE_INSTRUCTION)
  - preferences.json (_load_prefs_raw / _save_prefs_keys / _prefs)
  - DEVICE / DTYPE
  - logging (LOG_LEVEL / _log / _dbg / set_log_level)

Note: LOG_LEVEL est reassigne a l'execution (set_log_level). Les lecteurs hors de ce
module DOIVENT lire `cz_core.LOG_LEVEL` (pas `from cz_core import LOG_LEVEL`) pour voir
la valeur a jour. _log/_dbg lisent la valeur vive ici, donc les importer est sans risque.
"""

import os

# Force protobuf's pure-Python backend AVANT tout import de transformers/sentencepiece.
# Sinon le tokenizer (Qwen3 / T5 / sentencepiece) plante: "Descriptors cannot be created
# directly" (pb2 genere avec un vieux protoc, incompatible avec protobuf >=3.20 en C++).
# setdefault: ne surcharge pas un reglage explicite de l'utilisateur.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import sys
import json
import io
import base64

import torch
from PIL import Image

# Version de l'application (affichee dans le titre; entrees CHANGELOG.md par version).
APP_VERSION = "1.10.1"

HERE = os.path.dirname(os.path.abspath(__file__))
PREFS_PATH = os.path.join(HERE, "preferences.json")
CONFIG_PATH = os.path.join(HERE, "config.txt")
CONFIG_SAMPLE_PATH = os.path.join(HERE, "config-sample.txt")

# Defauts d'UI / CLI: reglages de reference (voir README)
DEFAULT_MODEL = "4x-ClearRealityV1_Soft.safetensors"
DEFAULT_FACTOR = 2.0
DEFAULT_DENOISE = 0.30
DEFAULT_STEPS = 12
DEFAULT_TILE = 760
DEFAULT_OVERLAP = 32
# Tiling de la passe diffusion Z-Image (4K+). 0 = image entiere (defaut).
DEFAULT_REFINE_TILE = 0
DEFAULT_REFINE_OVERLAP = 64
DEFAULT_SAVE_MODE = "display"        # display | local | alongside | custom
DEFAULT_OUTPUT_DIR = "out"
DEFAULT_OUTPUT_FORMAT = "png"        # png | webp | jpg
SUPPORTED_FORMATS = ("png", "webp", "jpg")
IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".avif", ".heic")
DEFAULT_BASE_REPO = "Tongyi-MAI/Z-Image-Turbo"
DEFAULT_ESRGAN_DIR = os.path.join(HERE, "upscale_models")


def _load_config():
    """Charge la config (JSON, facon Fooocus). Priorite: config.txt (local, gitignore)
    -> config-sample.txt (livre) -> {} (les valeurs codees servent de repli)."""
    for path in (CONFIG_PATH, CONFIG_SAMPLE_PATH):
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f) or {}
            except Exception:
                pass
    return {}


CONFIG = _load_config()

# (Token Hugging Face: applique plus bas, apres le chargement de preferences.json.)

# Defauts pilotes par config.txt (repli sur les constantes ci-dessus).
DEFAULT_FACTOR = float(CONFIG.get("default_factor", DEFAULT_FACTOR))
DEFAULT_DENOISE = float(CONFIG.get("default_denoise", DEFAULT_DENOISE))
DEFAULT_STEPS = int(CONFIG.get("default_refine_steps", DEFAULT_STEPS))
DEFAULT_TILE = int(CONFIG.get("default_tile", DEFAULT_TILE))
DEFAULT_OVERLAP = int(CONFIG.get("default_overlap", DEFAULT_OVERLAP))
DEFAULT_REFINE_TILE = int(CONFIG.get("default_refine_tile", DEFAULT_REFINE_TILE))
DEFAULT_REFINE_OVERLAP = int(CONFIG.get("default_refine_overlap", DEFAULT_REFINE_OVERLAP))
DEFAULT_SAVE_MODE = CONFIG.get("default_save_mode", DEFAULT_SAVE_MODE)
DEFAULT_OUTPUT_DIR = CONFIG.get("default_output_dir", DEFAULT_OUTPUT_DIR)
DEFAULT_OUTPUT_FORMAT = CONFIG.get("default_output_format", DEFAULT_OUTPUT_FORMAT)

# Profils par modele: substring du nom -> reglages recommandes (steps/guidance).
MODEL_PROFILES = CONFIG.get("model_profiles") or {
    "turbo": {"steps": 8, "guidance": 0.0},
    "juggernaut": {"steps": 28, "guidance": 6.0},
    "base": {"steps": 24, "guidance": 4.0},
}
DEFAULT_MODEL_PROFILE = CONFIG.get("default_model_profile") or {"steps": 8, "guidance": 0.0}


def profile_for_model(name):
    """Renvoie (steps, guidance) recommandes pour un modele d'apres son nom
    (matching de substring dans model_profiles), sinon le profil par defaut."""
    n = (name or "").lower()
    for key, prof in MODEL_PROFILES.items():
        if key.lower() in n:
            return int(prof.get("steps", DEFAULT_MODEL_PROFILE.get("steps", 8))), \
                float(prof.get("guidance", DEFAULT_MODEL_PROFILE.get("guidance", 0.0)))
    return int(DEFAULT_MODEL_PROFILE.get("steps", 8)), float(DEFAULT_MODEL_PROFILE.get("guidance", 0.0))


# Strings d'instruction Ollama (editable dans config.txt).
DESCRIBE_INSTRUCTION = CONFIG.get(
    "ollama_describe_prompt",
    "You are an expert text-to-image prompt writer. Look at the image and output ONE "
    "detailed prompt as comma-separated visual tags (subject, clothing, setting, lighting, "
    "style, quality). No preamble, no explanation, just the prompt.")
IMPROVE_INSTRUCTION = CONFIG.get(
    "ollama_improve_prompt",
    "Rewrite the following text-to-image prompt to be more vivid and detailed while keeping "
    "the same subject and intent. Output ONLY the improved prompt (comma-separated), no "
    "preamble.\n\nPROMPT: {prompt}")
COMPOSE_INSTRUCTION = CONFIG.get(
    "ollama_compose_prompt",
    "You are an expert text-to-image prompt writer. Below are descriptions of several "
    "reference images. Merge their key elements (subject, clothing, pose, setting, style) "
    "into ONE single coherent, detailed image prompt. Output ONLY the prompt (comma-"
    "separated), no preamble.\n\n{descriptions}")


def _load_prefs_raw():
    if not os.path.isfile(PREFS_PATH):
        return {}
    try:
        with open(PREFS_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_prefs_keys(updates):
    """Met a jour quelques cles dans preferences.json, garde le reste intact."""
    data = _load_prefs_raw()
    data.update(updates)
    with open(PREFS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _is_single_file(p):
    """Vrai si p est un fichier checkpoint (ex. .safetensors Civitai) plutot qu'un
    repo HF ou un dossier diffusers."""
    return bool(p) and os.path.isfile(p) and p.lower().endswith(
        (".safetensors", ".ckpt", ".pt", ".sft"))


_prefs = _load_prefs_raw()


# Token Hugging Face pour les repos GATED (ex. FLUX.1-Krea-dev). Resolution (1er non vide):
# env HF_TOKEN / HUGGING_FACE_HUB_TOKEN -> config.txt 'hf_token' -> preferences.json 'hf_token'.
# On pose les env vars pour que diffusers/huggingface_hub authentifient SANS 'huggingface-cli
# login'. config.txt ET preferences.json sont gitignores -> le token n'est jamais commit.
def _apply_hf_token(token):
    token = (token or "").strip()
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    return token


def set_hf_token(token):
    """Pose le token HF pour la session ET le persiste dans preferences.json (gitignore).
    Appele par l'UI (onglet Models). Renvoie le token applique (vide si efface)."""
    token = _apply_hf_token(token)
    try:
        _save_prefs_keys({"hf_token": token})
    except Exception:
        pass
    return token


def hf_token_is_set():
    """Vrai si un token HF est actif dans l'environnement courant."""
    return bool((os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip())


_apply_hf_token(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                or CONFIG.get("hf_token") or _prefs.get("hf_token") or "")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# ----------------------------------------------------------------------------
# Logging. 0 = quiet, 1 = info, 2 = debug. Source: env CRISPZ_LOG_LEVEL, sinon 1.
# ----------------------------------------------------------------------------
_LOG_NAMES = {"quiet": 0, "info": 1, "debug": 2, "0": 0, "1": 1, "2": 2}


def _parse_log_level(v, default=1):
    if v is None:
        return default
    return _LOG_NAMES.get(str(v).strip().lower(), default)


LOG_LEVEL = _parse_log_level(os.environ.get("CRISPZ_LOG_LEVEL") or CONFIG.get("log_level"), 1)
VERBOSE = True  # back-compat (non utilise pour le gating)


def set_log_level(level):
    """Regle le niveau de log (quiet/info/debug ou 0/1/2). Renvoie un libelle."""
    global LOG_LEVEL
    LOG_LEVEL = _parse_log_level(level, LOG_LEVEL)
    name = {0: "quiet", 1: "info", 2: "debug"}.get(LOG_LEVEL, str(LOG_LEVEL))
    return f"Log level: {name}"


def _log(msg, level=1, mod=None):
    """Log console. mod (optionnel) = prefixe de module, ex. _log('...', mod='queue')
    -> '[crispz][queue] ...'."""
    if LOG_LEVEL >= level:
        tag = f"[crispz][{mod}]" if mod else "[crispz]"
        print(f"{tag} {msg}", file=sys.stderr, flush=True)


def _dbg(msg):
    """Log niveau debug (visible seulement en LOG_LEVEL >= 2)."""
    if LOG_LEVEL >= 2:
        print(f"[crispz][dbg] {msg}", file=sys.stderr, flush=True)


def download_with_progress(url, dst, label=None, block=65536, timeout=30):
    """Telechargement ATOMIQUE (ecrit dst.tmp puis os.replace -> jamais de fichier
    tronque servi) avec progression reecrite sur une ligne:
    'fichier: 2.1/4.3 MB (48%)'. Leve en cas d'echec (tmp nettoye). Stdlib seulement."""
    import urllib.request
    label = label or os.path.basename(dst)
    tmp = dst + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "crispz-studio"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            while True:
                chunk = r.read(block)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if LOG_LEVEL >= 1:
                    if total:
                        sys.stderr.write(f"\r{label}: {got / 1e6:.1f}/{total / 1e6:.1f} MB "
                                         f"({100 * got // total}%)")
                    else:
                        sys.stderr.write(f"\r{label}: {got / 1e6:.1f} MB")
                    sys.stderr.flush()
        if LOG_LEVEL >= 1:
            sys.stderr.write("\n")
        os.replace(tmp, dst)
        return dst
    except Exception:
        if LOG_LEVEL >= 1:
            sys.stderr.write("\n")
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _pil_to_b64_jpeg(img, max_side=1600, quality=85):
    """Reduit + encode une image PIL en JPEG base64 (pour Ollama ou un <img> HTML)."""
    if img is None:
        return None
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        if w >= h:
            new_w = max_side
            new_h = int(h * max_side / w)
        else:
            new_h = max_side
            new_w = int(w * max_side / h)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")
