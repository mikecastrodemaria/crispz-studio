"""crispz-studio - core foundation (config, paths, logging, device).

Pulled out of app.py. No dependency on the rest of the project (app.py and the other
modules import cz_core, never the other way round). It holds:
  - the paths (HERE, PREFS_PATH, CONFIG_PATH...) and the default constants (DEFAULT_*)
  - the loading of the JSON config (config.txt -> config-sample.txt) -> CONFIG
  - the per-model profiles (MODEL_PROFILES / profile_for_model)
  - the Ollama instructions (DESCRIBE/IMPROVE/COMPOSE_INSTRUCTION)
  - preferences.json (_load_prefs_raw / _save_prefs_keys / _prefs)
  - DEVICE / DTYPE
  - logging (LOG_LEVEL / _log / _dbg / set_log_level)

Note: LOG_LEVEL is reassigned at run time (set_log_level). Readers outside this module
MUST read `cz_core.LOG_LEVEL` (not `from cz_core import LOG_LEVEL`) to see the up-to-date
value. _log/_dbg read the live value here, so importing them is safe.

"""

import os

# Force protobuf's pure-Python backend BEFORE any import of transformers/sentencepiece.
# Otherwise the tokenizer (Qwen3 / T5 / sentencepiece) crashes: "Descriptors cannot be created
# directly" (a pb2 generated with an old protoc, incompatible with protobuf >=3.20 in C++).
# setdefault: it does not override an explicit setting of the user's.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import sys
import json
import io
import base64

import torch
from PIL import Image

# The application's version (shown in the title; one CHANGELOG.md entry per version).
APP_VERSION = "1.17.0"

HERE = os.path.dirname(os.path.abspath(__file__))
PREFS_PATH = os.path.join(HERE, "preferences.json")
CONFIG_PATH = os.path.join(HERE, "config.txt")
CONFIG_SAMPLE_PATH = os.path.join(HERE, "config-sample.txt")

# The UI / CLI defaults: the reference settings (see the README)
DEFAULT_MODEL = "4x-ClearRealityV1_Soft.safetensors"
DEFAULT_FACTOR = 2.0
DEFAULT_DENOISE = 0.30
DEFAULT_STEPS = 12
DEFAULT_TILE = 760
DEFAULT_OVERLAP = 32
# Tiling of the Z-Image diffusion pass (4K+). 0 = the whole image (the default).
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
    """Loads the config (JSON, Fooocus-style). The priority: config.txt (local,
    gitignored) -> config-sample.txt (shipped) -> {} (the hardcoded values serve as the
    fallback)."""
    for path in (CONFIG_PATH, CONFIG_SAMPLE_PATH):
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f) or {}
            except Exception as e:
                # Never silent: a broken config.txt used to fall back to the
                # sample without a word (wrong offload, wrong models, hours
                # lost). Windows paths are the classic cause.
                print(f"[crispz] WARNING: {os.path.basename(path)} is not valid "
                      f"JSON ({e}) -> IGNORED, falling back to the next config. "
                      f"Hint: in JSON, write Windows paths with forward slashes "
                      f"(D:/models/x.gguf) or doubled backslashes (D:\\models).",
                      flush=True)
    return {}


CONFIG = _load_config()

# (The Hugging Face token: applied further down, after preferences.json is loaded.)

# The defaults driven by config.txt (falling back on the constants above).
DEFAULT_FACTOR = float(CONFIG.get("default_factor", DEFAULT_FACTOR))
DEFAULT_DENOISE = float(CONFIG.get("default_denoise", DEFAULT_DENOISE))
DEFAULT_STEPS = int(CONFIG.get("default_refine_steps", DEFAULT_STEPS))
DEFAULT_TILE = int(CONFIG.get("default_tile", DEFAULT_TILE))
DEFAULT_OVERLAP = int(CONFIG.get("default_overlap", DEFAULT_OVERLAP))
DEFAULT_REFINE_TILE = int(CONFIG.get("default_refine_tile", DEFAULT_REFINE_TILE))
DEFAULT_REFINE_OVERLAP = int(CONFIG.get("default_refine_overlap", DEFAULT_REFINE_OVERLAP))

# The choices offered by the "Diffusion tile" dropdown. 0 = Auto: the whole image below
# auto_refine_tile_above, and beyond that tiling at the size computed by
# cz_pipeline._pick_refine_tile. The fixed sizes stay available to force the issue.
REFINE_TILE_CHOICES = [("Auto", 0)] + [(str(t), t) for t in
                                       (512, 640, 768, 896, 1024, 1280, 1536, 2048)]
if DEFAULT_REFINE_TILE not in [v for _, v in REFINE_TILE_CHOICES]:
    REFINE_TILE_CHOICES.append((str(DEFAULT_REFINE_TILE), DEFAULT_REFINE_TILE))
    REFINE_TILE_CHOICES.sort(key=lambda c: c[1])
DEFAULT_SAVE_MODE = CONFIG.get("default_save_mode", DEFAULT_SAVE_MODE)
DEFAULT_OUTPUT_DIR = CONFIG.get("default_output_dir", DEFAULT_OUTPUT_DIR)
DEFAULT_OUTPUT_FORMAT = CONFIG.get("default_output_format", DEFAULT_OUTPUT_FORMAT)

# The per-model profiles: a substring of the name -> the recommended settings (steps/guidance).
MODEL_PROFILES = CONFIG.get("model_profiles") or {
    "turbo": {"steps": 8, "guidance": 0.0},
    "juggernaut": {"steps": 28, "guidance": 6.0},
    "base": {"steps": 24, "guidance": 4.0},
}
DEFAULT_MODEL_PROFILE = CONFIG.get("default_model_profile") or {"steps": 8, "guidance": 0.0}


def profile_for_model(name):
    """The (steps, guidance) recommended for a model from its name (a substring match in
    model_profiles), otherwise the default profile."""
    n = (name or "").lower()
    for key, prof in MODEL_PROFILES.items():
        if key.lower() in n:
            return int(prof.get("steps", DEFAULT_MODEL_PROFILE.get("steps", 8))), \
                float(prof.get("guidance", DEFAULT_MODEL_PROFILE.get("guidance", 0.0)))
    return int(DEFAULT_MODEL_PROFILE.get("steps", 8)), float(DEFAULT_MODEL_PROFILE.get("guidance", 0.0))


# The Ollama instruction strings (editable in config.txt). The examples from before 1.36
# -- the ones in config-sample.txt, copied as they are into most config.txt -- do not
# count as a customisation: otherwise the old tag-based instruction would hide the new
# ones for everyone who copied the example.
LEGACY_DESCRIBE_INSTRUCTION = (
    "You are an expert text-to-image prompt writer. Look at the image and output ONE "
    "detailed prompt as comma-separated visual tags (subject, clothing, setting, lighting, "
    "style, quality). No preamble, no explanation, just the prompt.")
LEGACY_IMPROVE_INSTRUCTION = (
    "Rewrite the following text-to-image prompt to be more vivid and detailed while keeping "
    "the same subject and intent. Output ONLY the improved prompt (comma-separated), no "
    "preamble.\n\nPROMPT: {prompt}")
LEGACY_COMPOSE_INSTRUCTION = (
    "You are an expert text-to-image prompt writer. Below are descriptions of several "
    "reference images. Merge their key elements (subject, clothing, pose, setting, style) "
    "into ONE single coherent, detailed image prompt. Output ONLY the prompt (comma-"
    "separated), no preamble.\n\n{descriptions}")


def _instruction(key, legacy, default):
    """The `key` instruction from config.txt; empty or identical to the old example -> `default`."""
    v = CONFIG.get(key)
    if not isinstance(v, str) or not v.strip() or v.strip() == legacy.strip():
        return default
    return v


# Describe: one analysis style = one instruction, {words} = the length. "Prompt (prose)" is
# the v4 measured on 2026-09-11 (Agents-A1-4B and muse-glimmer; 3 images with a known prompt,
# every description regenerated by klein 4B at the same seed): without the medium up front, a
# pencil portrait came back as a photo; a text quoted line by line came back with its lines
# shuffled; a stated absence ("No text is visible") or a hesitation ("appears to be") brings
# nothing to the prompt. On 2026-09-12, the same bench: the period (which Captionz' "Dataset
# paragraph" instruction asks for) lifts the portrait's fidelity from 0.54 to 0.65-0.66 on both
# models. The other styles follow the same rules.
_DESCRIBE_RULES = (
    "Describe only what is present: never mention what is absent. State every detail as a "
    "fact: no \"appears\", \"seems\", \"likely\", \"possibly\", \"as if\". No filler "
    "(masterpiece, best quality, 8k, stunning, beautiful). Do not start with \"This image\". ")
_DESCRIBE_TEXT_RULE = (
    "Quote text only when it is a main element of the image (a sign, a title or a label in "
    "the foreground): give it once, in reading order, as a single string in double quotes, "
    "exactly as written; skip small or background text entirely. ")
SHORT_CAPTION_STYLE = "Short caption"
DESCRIBE_STYLES = {
    "Prompt (prose)": (
        "Describe this image as a text-to-image prompt, in one flowing paragraph of about "
        "{words} words. Begin with the medium and style (for example: black-and-white ink "
        "illustration, candid photograph, 3D render, oil painting). Then: main subject(s) "
        "(count, age range, build, face, expression, hair); clothing and accessories "
        "(materials, colors, fit); pose and action; setting from foreground to background, "
        "with positions (left, right, center); camera (shot size, angle, lens, focus); "
        "lighting (sources, direction, softness, color temperature); color palette with "
        "precise color names; time of day, weather and era when they are identifiable; mood. "
        + _DESCRIBE_TEXT_RULE + _DESCRIBE_RULES + "Output only the paragraph."),
    "Prompt (tags)": (
        "Describe this image as a text-to-image prompt made of about {words} words of "
        "comma-separated visual tags, most important first: medium and style, subject, "
        "clothing, pose, setting, camera, lighting, colors, mood. "
        + _DESCRIBE_TEXT_RULE + _DESCRIBE_RULES + "Output only the tags."),
    "Photo (technical)": (
        "Describe this photograph as a text-to-image prompt, in one paragraph of about "
        "{words} words, for a photographer who must reproduce it. Begin with the kind of "
        "photograph (studio portrait, street, product, landscape...). Then: subject and pose; "
        "shot size and camera angle; lens focal length and aperture, depth of field and what "
        "is in focus; lighting setup (key, fill and rim lights, their direction, softness and "
        "color temperature); color grading, contrast, film grain or noise; setting and "
        "background. " + _DESCRIBE_TEXT_RULE + _DESCRIBE_RULES + "Output only the paragraph."),
    "Art & style": (
        "Describe this image as a text-to-image prompt, in one paragraph of about {words} "
        "words, focused on how it is made. Begin with the medium and technique (ink, pencil, "
        "watercolor, oil, digital painting, 3D render, pixel art...). Then: line work and "
        "brush strokes, shading and rendering, color palette with precise color names, level "
        "of detail, art movement or genre, composition; then the subject in a few words. "
        + _DESCRIBE_TEXT_RULE + _DESCRIBE_RULES + "Output only the paragraph."),
    "Composition & layout": (
        "Describe this image as a text-to-image prompt, in one paragraph of about {words} "
        "words, so that the same layout can be rebuilt. Begin with the medium and style. "
        "Then place every element: foreground, middle ground and background; left, center "
        "and right; relative sizes and distances; where the horizon and the vanishing point "
        "sit; framing, camera height and angle; empty space. "
        + _DESCRIBE_TEXT_RULE + _DESCRIBE_RULES + "Output only the paragraph."),
    "Character sheet": (
        "Describe the main character of this image as a text-to-image prompt, in one "
        "paragraph of about {words} words, so that the same character can be drawn again. "
        "Begin with the medium and style. Then: age range, build and height, face shape, "
        "eyes, nose, lips, skin tone, hair (color, length, texture, style); outfit from the "
        "inner layer to the outer one with materials, colors and fit; accessories; "
        "distinguishing marks; pose and expression. Keep the setting to one short sentence. "
        + _DESCRIBE_RULES + "Output only the paragraph."),
    "Text & typography": (
        "Describe this image as a text-to-image prompt, in one paragraph of about {words} "
        "words, for an image whose text matters. Begin with the medium and style. Quote every "
        "legible line of text exactly, in reading order, in double quotes; for each, give its "
        "place, size, font style (serif, sans-serif, script, hand-lettered...), color and "
        "material. Then describe the support (sign, poster, screen, label...) and the "
        "setting. Never guess blurry or partial text. " + _DESCRIBE_RULES
        + "Output only the paragraph."),
    "Dataset paragraph": (
        "Describe this image in one detailed paragraph of about {words} words: subjects and "
        "characters, objects, setting, era if identifiable, medium and technique (photo, "
        "painting, 3D render, illustration...), visual style and mood. Use concrete visual "
        "terms. End with the aspect ratio and orientation. " + _DESCRIBE_RULES
        + "Output only the paragraph."),
    SHORT_CAPTION_STYLE: (
        "Describe this image in one short sentence of at most {words} words: the medium, the "
        "main subject and the setting. " + _DESCRIBE_RULES + "Output only the sentence."),
}
DESCRIBE_LENGTHS = {"Short": 60, "Medium": 120, "Long": 180, "Very long": 300}
DEFAULT_DESCRIBE_STYLE, DEFAULT_DESCRIBE_LENGTH = "Prompt (prose)", "Long"
CUSTOM_STYLE = "Custom (config.txt)"
# The personal instruction from config.txt (the "Custom (config.txt)" style), None otherwise.
DESCRIBE_CUSTOM = _instruction("ollama_describe_prompt", LEGACY_DESCRIBE_INSTRUCTION, None)


def describe_instruction(style=None, length=None):
    """The instruction sent to the vision model for this style and this length (the
    defaults otherwise)."""
    if style == CUSTOM_STYLE and DESCRIBE_CUSTOM:
        return DESCRIBE_CUSTOM
    tpl = DESCRIBE_STYLES.get(style) or DESCRIBE_STYLES[DEFAULT_DESCRIBE_STYLE]
    words = 25 if style == SHORT_CAPTION_STYLE else DESCRIBE_LENGTHS.get(
        length, DESCRIBE_LENGTHS[DEFAULT_DESCRIBE_LENGTH])
    return tpl.replace("{words}", str(words))


# Describe's default instruction (compat: imported as it is by some old callers).
DESCRIBE_INSTRUCTION = describe_instruction(CUSTOM_STYLE if DESCRIBE_CUSTOM else DEFAULT_DESCRIBE_STYLE)
IMPROVE_INSTRUCTION = _instruction(
    "ollama_improve_prompt", LEGACY_IMPROVE_INSTRUCTION,
    "Rewrite the following text-to-image prompt to be more vivid and detailed while keeping "
    "the same subject, intent and every detail it already gives; add what is missing among "
    "medium and style, camera, lighting and color. Keep its form: prose stays prose, a tag "
    "list stays a tag list. No filler (masterpiece, best quality, 8k). Output ONLY the "
    "improved prompt, no preamble.\n\nPROMPT: {prompt}")
COMPOSE_INSTRUCTION = _instruction(
    "ollama_compose_prompt", LEGACY_COMPOSE_INSTRUCTION,
    "You are an expert text-to-image prompt writer. Below are descriptions of several "
    "reference images. Merge their key elements (subject, clothing, pose, setting, style) "
    "into ONE single coherent, detailed image prompt, written as one flowing paragraph that "
    "starts with the medium and style. Output ONLY the prompt, no preamble.\n\n{descriptions}")


def _load_prefs_raw():
    if not os.path.isfile(PREFS_PATH):
        return {}
    try:
        with open(PREFS_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_prefs_keys(updates):
    """Updates a few keys in preferences.json, leaves the rest intact."""
    data = _load_prefs_raw()
    data.update(updates)
    with open(PREFS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _is_single_file(p):
    """True when p is a checkpoint file (a Civitai .safetensors, a quantised .gguf...)
    rather than an HF repo or a diffusers folder."""
    return bool(p) and os.path.isfile(p) and p.lower().endswith(
        (".safetensors", ".ckpt", ".pt", ".sft", ".gguf"))


_prefs = _load_prefs_raw()


# The Hugging Face token for the GATED repos (e.g. FLUX.1-Krea-dev). The resolution (the
# 1st non-empty one): the HF_TOKEN / HUGGING_FACE_HUB_TOKEN env -> config.txt 'hf_token' ->
# preferences.json 'hf_token'. We set the env vars so that diffusers/huggingface_hub
# authenticate WITHOUT 'huggingface-cli login'. config.txt AND preferences.json are
# gitignored -> the token is never committed.
def _apply_hf_token(token):
    token = (token or "").strip()
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    return token


def set_hf_token(token):
    """Sets the HF token for the session AND persists it in preferences.json
    (gitignored). Called by the UI (the Models tab). Returns the token applied (empty when
    cleared)."""
    token = _apply_hf_token(token)
    try:
        _save_prefs_keys({"hf_token": token})
    except Exception:
        pass
    return token


def hf_token_is_set():
    """True when an HF token is active in the current environment."""
    return bool((os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip())


_apply_hf_token(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                or CONFIG.get("hf_token") or _prefs.get("hf_token") or "")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# ----------------------------------------------------------------------------
# Logging. 0 = quiet, 1 = info, 2 = debug. The source: the CRISPZ_LOG_LEVEL env, otherwise 1.
# ----------------------------------------------------------------------------
_LOG_NAMES = {"quiet": 0, "info": 1, "debug": 2, "0": 0, "1": 1, "2": 2}


def _parse_log_level(v, default=1):
    if v is None:
        return default
    return _LOG_NAMES.get(str(v).strip().lower(), default)


LOG_LEVEL = _parse_log_level(os.environ.get("CRISPZ_LOG_LEVEL") or CONFIG.get("log_level"), 1)
VERBOSE = True  # back-compat (not used for the gating)


def set_log_level(level):
    """Sets the log level (quiet/info/debug or 0/1/2). Returns a label."""
    global LOG_LEVEL
    LOG_LEVEL = _parse_log_level(level, LOG_LEVEL)
    name = {0: "quiet", 1: "info", 2: "debug"}.get(LOG_LEVEL, str(LOG_LEVEL))
    return f"Log level: {name}"


def _log(msg, level=1, mod=None):
    """Console log. mod (optional) = the module prefix, e.g. _log('...', mod='queue')
    -> '[crispz][queue] ...'."""
    if LOG_LEVEL >= level:
        tag = f"[crispz][{mod}]" if mod else "[crispz]"
        print(f"{tag} {msg}", file=sys.stderr, flush=True)


def _dbg(msg):
    """A debug-level log (visible only at LOG_LEVEL >= 2)."""
    if LOG_LEVEL >= 2:
        print(f"[crispz][dbg] {msg}", file=sys.stderr, flush=True)


def download_with_progress(url, dst, label=None, block=65536, timeout=30):
    """An ATOMIC download (it writes dst.tmp then os.replace -> never a truncated file
    served) with the progress rewritten on a single line:
    'file: 2.1/4.3 MB (48%)'. Raises on failure (the tmp is cleaned up). Stdlib only."""
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
    """Downscales + encodes a PIL image into base64 JPEG (for Ollama or an HTML <img>)."""
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
