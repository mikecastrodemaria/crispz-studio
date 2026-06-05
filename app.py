"""
crispz-studio - Z-Image txt2img + upscaler/detailer (standalone, sans ComfyUI)

Fork "studio" de crispz. Ajoute:
  0. txt2img (ZImagePipeline) -> generation depuis un prompt, upscale optionnel.
  1. Real-ESRGAN (charge via spandrel) -> agrandissement reel des pixels, avec tiling.
  2. Z-Image Turbo en img2img (diffusers, BF16) -> passe de raffinement a bas denoise
     qui reinjecte du detail sans changer la composition.
  + modeles single-file .safetensors (Civitai) via from_single_file (transformer),
    VAE + encodeur Qwen3 tires du repo de base. Partage VRAM txt2img/img2img (from_pipe).

Pre-requis cote machine (RTX 5090, PyTorch 2.7 / CUDA 12.8 deja installes):
  pip install -r requirements.txt
  (ne pas reinstaller torch, garder ton build cu128)

Lancer:
  python app.py
"""

import os
import sys
import gc
import io
import base64
import glob
import time
import uuid
import datetime
import numpy as np
import torch
from PIL import Image
import gradio as gr

# Defauts d'UI / CLI: reglages de reference (voir README)
DEFAULT_MODEL = "4x-ClearRealityV1_Soft.safetensors"
DEFAULT_FACTOR = 2.0
DEFAULT_DENOISE = 0.30
DEFAULT_STEPS = 12
DEFAULT_TILE = 760
DEFAULT_OVERLAP = 32
# Tiling de la passe diffusion Z-Image (4K+). 0 = image entiere (defaut, pas de
# regression). >0 = decoupe en tuiles de cette taille (arrondie a un multiple de 16).
DEFAULT_REFINE_TILE = 0
DEFAULT_REFINE_OVERLAP = 64
DEFAULT_SAVE_MODE = "display"        # display | local | alongside | custom
DEFAULT_OUTPUT_DIR = "out"
DEFAULT_OUTPUT_FORMAT = "png"        # png | webp | jpg
SUPPORTED_FORMATS = ("png", "webp", "jpg")
IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

# Presets "cas d'usage" -> reglages auto. Seules les cles presentes sont appliquees,
# le reste est laisse tel quel. Utilise par l'UI (_apply_preset) et la CLI (--preset).
PRESETS = {
    "Custom": {},
    "Photo (balanced)":    {"factor": 2.0, "denoise": 0.30, "steps": 12, "refine_tile": 0, "cpu_offload": "none"},
    "Subtle (clean-up)":   {"factor": 2.0, "denoise": 0.12, "steps": 16, "refine_tile": 0},
    "Detailed (creative)": {"factor": 2.0, "denoise": 0.40, "steps": 16},
    "Portrait (faces)":    {"factor": 2.0, "denoise": 0.22, "steps": 14},
    "4K (tiled)":          {"factor": 4.0, "denoise": 0.30, "steps": 12, "refine_tile": 1024, "refine_overlap": 64, "cpu_offload": "model"},
    "Low VRAM (8-12GB)":   {"denoise": 0.30, "steps": 12, "tile": 512, "refine_tile": 1024, "refine_overlap": 64, "cpu_offload": "sequential"},
}
# param interne -> flag CLI, pour appliquer un preset sans ecraser un flag explicite.
PRESET_FLAGMAP = {
    "factor": "--factor", "denoise": "--denoise", "steps": "--steps", "tile": "--tile",
    "overlap": "--overlap", "refine_tile": "--refine-tile", "refine_overlap": "--refine-overlap",
    "cpu_offload": "--cpu-offload",
}

# ----------------------------------------------------------------------------
# Config (persistance dans preferences.json a cote de app.py)
# Ordre de priorite pour ESRGAN_DIR et BASE_REPO:
#   1) variable d'environnement (ESRGAN_DIR / ZIMAGE_MODEL)
#   2) preferences.json
#   3) defaut: ./upscale_models  et  Tongyi-MAI/Z-Image-Turbo
# ----------------------------------------------------------------------------
import json

HERE = os.path.dirname(os.path.abspath(__file__))
PREFS_PATH = os.path.join(HERE, "preferences.json")
DEFAULT_BASE_REPO = "Tongyi-MAI/Z-Image-Turbo"
DEFAULT_ESRGAN_DIR = os.path.join(HERE, "upscale_models")


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
_zmodel = os.environ.get("ZIMAGE_MODEL") or _prefs.get("zimage_model") or DEFAULT_BASE_REPO
# Transformer single-file optionnel (poids du transformer seul, ex. Civitai). Le VAE
# et l'encodeur Qwen3 restent tires du repo de base (BASE_REPO).
ZIMAGE_TRANSFORMER = os.environ.get("ZIMAGE_TRANSFORMER") or _prefs.get("zimage_transformer") or None
if _is_single_file(_zmodel):
    # Un fichier passe comme "modele" = le transformer; base par defaut pour le reste.
    ZIMAGE_TRANSFORMER = _zmodel
    BASE_REPO = DEFAULT_BASE_REPO
else:
    BASE_REPO = _zmodel

ESRGAN_DIR = os.environ.get("ESRGAN_DIR") or _prefs.get("esrgan_dir") or DEFAULT_ESRGAN_DIR
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# Caches process-wide. Un pipeline "base" (txt2img ZImagePipeline) detient les
# composants; img2img / inpaint en derivent via from_pipe -> poids partages, pas de
# VRAM en double. Clef de cache = (BASE_REPO, ZIMAGE_TRANSFORMER, OFFLOAD_MODE).
_BASE_PIPE = None
_DERIVED = {}
_LOADED_KEY = None
_ESRGAN_CACHE = {}

# Palier 2 (cohabitation VRAM, brief plugin Fooocus): offload CPU de la passe
# diffusion. none = tout en VRAM (defaut). model = decharge par sous-module
# (bon compromis). sequential = plus agressif, plus lent. N'est PAS de la quantif:
# les poids restent BF16, ils transitent juste RAM <-> GPU. Requiert accelerate.
OFFLOAD_MODE = "none"
OFFLOAD_CHOICES = ("none", "model", "sequential")

# CFG. Z-Image *Turbo* = distille -> guidance 0 (defaut). Z-Image *Base* (non Turbo,
# ex. checkpoint Civitai "Z-Image Base") a besoin d'une vraie guidance (~3.5-5) et de
# plus de steps (~20-28). Reglable par run (CLI --guidance, sliders UI).
GUIDANCE = 0.0


def set_guidance(g):
    global GUIDANCE
    GUIDANCE = float(g)


# Logs d'etape sur stderr (chargement modeles, etages, tuiles). Coupes par --quiet.
# stderr donc ne pollue pas le stdout de --print-output.
VERBOSE = True


def _log(msg):
    if VERBOSE:
        print(f"[crispz] {msg}", file=sys.stderr, flush=True)


# Hook de progression UI (gradio gr.Progress). None hors UI (CLI/serveur). Permet
# d'afficher l'avancement des etapes (ESRGAN, refine, tuiles i/N) dans l'interface.
_PROGRESS = None


def _progress(frac, desc=""):
    if _PROGRESS is not None:
        try:
            _PROGRESS(min(1.0, max(0.0, float(frac))), desc)
        except Exception:
            pass


def set_esrgan_dir(path):
    """Change le dossier ESRGAN. Invalide le cache (les noms peuvent collisionner entre dossiers)."""
    global ESRGAN_DIR, _ESRGAN_CACHE
    if path and path != ESRGAN_DIR:
        ESRGAN_DIR = path
        _ESRGAN_CACHE = {}


def set_zimage_model(repo_or_path):
    """Change le modele Z-Image. Un repo HF / dossier diffusers -> BASE_REPO.
    Un fichier single-file (.safetensors Civitai) -> transformer override.
    Invalide le pipe si change."""
    global BASE_REPO, ZIMAGE_TRANSFORMER
    if not repo_or_path:
        return
    if _is_single_file(repo_or_path):
        if repo_or_path != ZIMAGE_TRANSFORMER:
            ZIMAGE_TRANSFORMER = repo_or_path
            free_vram()
            _log("Z-Image transformer (single-file) changed -> will reload")
    elif repo_or_path != BASE_REPO:
        BASE_REPO = repo_or_path
        free_vram()
        _log("Z-Image base repo changed -> will reload")


def set_zimage_transformer(path):
    """Definit (ou enleve avec '' / None) le transformer single-file."""
    global ZIMAGE_TRANSFORMER
    path = path or None
    if path != ZIMAGE_TRANSFORMER:
        ZIMAGE_TRANSFORMER = path
        free_vram()
        _log(f"Z-Image transformer -> {path or '(repo de base)'} -> will reload")


def set_offload_mode(mode):
    """Change le mode d'offload CPU. Invalide le pipe (hooks poses au chargement)."""
    global OFFLOAD_MODE
    mode = mode if mode in OFFLOAD_CHOICES else "none"
    if mode != OFFLOAD_MODE:
        OFFLOAD_MODE = mode
        free_vram()
        _log(f"offload -> {OFFLOAD_MODE}: pipeline invalidated -> will reload")


def free_vram():
    """Libere le pipeline de base + les pipelines derives et rend la VRAM
    (palier 3: unload sur inactivite ou endpoint /unload). Rechargement paresseux."""
    global _BASE_PIPE, _DERIVED, _LOADED_KEY
    _BASE_PIPE = None
    _DERIVED = {}
    _LOADED_KEY = None
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def apply_preset_to_args(args, raw_argv):
    """Applique un preset aux champs de args qui n'ont PAS ete passes explicitement
    en CLI (un flag explicite gagne toujours sur le preset)."""
    preset = PRESETS.get(getattr(args, "preset", None) or "Custom") or {}
    raw = list(raw_argv or [])
    for key, val in preset.items():
        flag = PRESET_FLAGMAP[key]
        if not any(tok == flag or tok.startswith(flag + "=") for tok in raw):
            setattr(args, key, val)


# ----------------------------------------------------------------------------
# Etage 1 : Real-ESRGAN via spandrel
# ----------------------------------------------------------------------------
def list_esrgan_models():
    if not os.path.isdir(ESRGAN_DIR):
        return []
    return sorted(
        f for f in os.listdir(ESRGAN_DIR)
        if f.lower().endswith((".pth", ".safetensors"))
    )


def load_esrgan(model_name):
    if model_name in _ESRGAN_CACHE:
        return _ESRGAN_CACHE[model_name]
    from spandrel import ModelLoader, ImageModelDescriptor
    _log(f"loading ESRGAN model: {model_name} ...")
    path = os.path.join(ESRGAN_DIR, model_name)
    model = ModelLoader().load_from_file(path)
    if not isinstance(model, ImageModelDescriptor):
        raise ValueError(f"{model_name} is not a usable image SR model.")
    model = model.to(DEVICE).eval()
    _ESRGAN_CACHE[model_name] = model
    return model


def _pil_to_tensor(img):
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


def _tensor_to_pil(t):
    arr = t.clamp(0, 1).squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    return Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8))


def esrgan_upscale(img, model, tile, overlap):
    """Upscale ESRGAN avec tiling overlap-add et feather lineaire pour eviter les coutures."""
    scale = model.scale
    t = _pil_to_tensor(img)
    _, _, h, w = t.shape

    if tile <= 0 or (h <= tile and w <= tile):
        with torch.no_grad():
            out = model(t)
        return _tensor_to_pil(out)

    out_h, out_w = h * scale, w * scale
    acc = torch.zeros(1, 3, out_h, out_w, device=DEVICE)
    weight = torch.zeros(1, 1, out_h, out_w, device=DEVICE)
    step = tile - overlap

    for y in range(0, h, step):
        for x in range(0, w, step):
            y2, x2 = min(y + tile, h), min(x + tile, w)
            y1, x1 = max(y2 - tile, 0), max(x2 - tile, 0)
            patch = t[:, :, y1:y2, x1:x2]
            with torch.no_grad():
                up = model(patch)
            ph, pw = up.shape[2], up.shape[3]
            # masque feather: rampe lineaire sur la zone d'overlap
            mask = torch.ones(1, 1, ph, pw, device=DEVICE)
            f = overlap * scale
            if f > 0:
                ramp = torch.linspace(0, 1, int(f), device=DEVICE)
                if x1 > 0:
                    mask[:, :, :, :int(f)] *= ramp.view(1, 1, 1, -1)
                if x2 < w:
                    mask[:, :, :, -int(f):] *= ramp.flip(0).view(1, 1, 1, -1)
                if y1 > 0:
                    mask[:, :, :int(f), :] *= ramp.view(1, 1, -1, 1)
                if y2 < h:
                    mask[:, :, -int(f):, :] *= ramp.flip(0).view(1, 1, -1, 1)
            oy, ox = y1 * scale, x1 * scale
            acc[:, :, oy:oy + ph, ox:ox + pw] += up * mask
            weight[:, :, oy:oy + ph, ox:ox + pw] += mask

    out = acc / weight.clamp(min=1e-6)
    return _tensor_to_pil(out)


# ----------------------------------------------------------------------------
# Z-Image (diffusers, BF16) : un pipeline "base" txt2img qui detient les composants,
# img2img / inpaint derives via from_pipe (poids partages, pas de VRAM en double).
# ----------------------------------------------------------------------------
def _ensure_base():
    """Charge (si besoin) le pipeline de base txt2img. Gere le transformer
    single-file (Civitai) et l'offload. Cache par (repo, transformer, offload)."""
    global _BASE_PIPE, _DERIVED, _LOADED_KEY
    key = (BASE_REPO, ZIMAGE_TRANSFORMER, OFFLOAD_MODE)
    if _BASE_PIPE is not None and _LOADED_KEY == key:
        return _BASE_PIPE
    if _BASE_PIPE is not None:
        free_vram()
    from diffusers import ZImagePipeline, ZImageTransformer2DModel
    t0 = time.time()
    kwargs = {}
    if ZIMAGE_TRANSFORMER:
        _log(f"loading Z-Image transformer (single-file): {ZIMAGE_TRANSFORMER} ...")
        kwargs["transformer"] = ZImageTransformer2DModel.from_single_file(
            ZIMAGE_TRANSFORMER, torch_dtype=DTYPE)
    _log(f"loading Z-Image base: {BASE_REPO} (offload={OFFLOAD_MODE}, dtype=bf16) ... "
         "first time downloads from HF, then cached")
    pipe = ZImagePipeline.from_pretrained(BASE_REPO, torch_dtype=DTYPE, **kwargs)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    # enable_*_cpu_offload gere lui-meme le device -> ne PAS faire .to(cuda) alors.
    if DEVICE == "cuda" and OFFLOAD_MODE == "model":
        pipe.enable_model_cpu_offload()
    elif DEVICE == "cuda" and OFFLOAD_MODE == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe = pipe.to(DEVICE)
    _BASE_PIPE = pipe
    _DERIVED = {"txt2img": pipe}
    _LOADED_KEY = key
    _log(f"Z-Image base ready in {time.time() - t0:.1f}s")
    return pipe


def get_pipe(kind="img2img"):
    """Renvoie le pipeline demande (txt2img / img2img / inpaint), derive du base
    via from_pipe (composants partages, aucune VRAM en double)."""
    base = _ensure_base()
    if kind in _DERIVED:
        return _DERIVED[kind]
    from diffusers import ZImageImg2ImgPipeline, ZImageInpaintPipeline
    cls = {"img2img": ZImageImg2ImgPipeline, "inpaint": ZImageInpaintPipeline}.get(kind)
    if cls is None:
        return base
    _log(f"deriving {kind} pipeline (shared weights, no extra VRAM)")
    p = cls.from_pipe(base)
    _DERIVED[kind] = p
    return p


def load_pipe():
    """Compat: pipeline img2img (etage de raffinement)."""
    return get_pipe("img2img")


def generate(prompt, width, height, steps, seed, negative_prompt=""):
    """txt2img Z-Image: genere une image depuis un prompt.
    Turbo -> GUIDANCE 0. Base -> GUIDANCE ~3.5-5 + plus de steps."""
    pipe = get_pipe("txt2img")
    w = round_to_multiple(int(width))
    h = round_to_multiple(int(height))
    _log(f"txt2img: {w}x{h}, {int(steps)} steps, guidance {GUIDANCE:.1f} ...")
    _progress(0.1, f"Generating {w}x{h} ({int(steps)} steps)...")
    t0 = time.time()
    img = pipe(
        prompt=prompt or "",
        negative_prompt=(negative_prompt or None),
        width=w, height=h,
        num_inference_steps=int(steps),
        guidance_scale=GUIDANCE,
        generator=_make_generator(seed),
    ).images[0]
    _log(f"txt2img done in {time.time() - t0:.1f}s")
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return img


def round_to_multiple(x, m=16):
    return max(m, int(round(x / m) * m))


def _make_generator(seed):
    return torch.Generator(DEVICE).manual_seed(int(seed)) if int(seed) >= 0 else None


def _refine_whole(pipe, image, denoise, steps, prompt, seed):
    """Passe Z-Image img2img sur l'image entiere."""
    return pipe(
        prompt=prompt or "",
        image=image,
        strength=float(denoise),
        num_inference_steps=int(steps),
        guidance_scale=GUIDANCE,
        generator=_make_generator(seed),
    ).images[0]


def _feather_mask_np(th, tw, overlap, left, right, top, bottom):
    """Masque (th, tw, 1) a rampe lineaire sur les bords qui jouxtent une autre tuile."""
    mask = np.ones((th, tw, 1), dtype=np.float32)
    f = int(overlap)
    if f > 0:
        ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
        if left:
            mask[:, :f, 0] *= ramp[np.newaxis, :]
        if right:
            mask[:, tw - f:, 0] *= ramp[::-1][np.newaxis, :]
        if top:
            mask[:f, :, 0] *= ramp[:, np.newaxis]
        if bottom:
            mask[th - f:, :, 0] *= ramp[::-1][:, np.newaxis]
    return mask


def _refine_tiled(pipe, image, denoise, steps, prompt, seed, tile, overlap):
    """Passe Z-Image en tuiles avec recomposition feather (facon Ultimate SD Upscale).
    Plafonne le pic VRAM (une tuile a la fois) et permet le 4K+ sans coutures.
    Memes rampe lineaire + overlap-add que esrgan_upscale, mais a scale 1 sur PIL."""
    w, h = image.size
    tile = round_to_multiple(tile)                       # multiple de 16 pour le VAE
    overlap = max(0, min(int(overlap), tile - 16))
    if w <= tile and h <= tile:
        return _refine_whole(pipe, image, denoise, steps, prompt, seed)

    acc = np.zeros((h, w, 3), dtype=np.float32)
    weight = np.zeros((h, w, 1), dtype=np.float32)
    step = max(16, tile - overlap)
    ys = list(range(0, h, step))
    xs = list(range(0, w, step))
    total = len(ys) * len(xs)
    _log(f"refine: tiled {w}x{h}, tile {tile} overlap {overlap} -> {len(xs)}x{len(ys)} = {total} tiles")
    i = 0
    for y in ys:
        for x in xs:
            i += 1
            x2, y2 = min(x + tile, w), min(y + tile, h)
            x1, y1 = max(x2 - tile, 0), max(y2 - tile, 0)
            cw, ch = x2 - x1, y2 - y1
            _log(f"  tile {i}/{total}")
            _progress(0.45 + 0.5 * (i - 1) / max(1, total), f"Refine tile {i}/{total}")
            crop = image.crop((x1, y1, x2, y2))
            out = _refine_whole(pipe, crop, denoise, steps, prompt, seed)
            if out.size != (cw, ch):
                out = out.resize((cw, ch), Image.LANCZOS)
            out_arr = np.asarray(out.convert("RGB"), dtype=np.float32) / 255.0
            mask = _feather_mask_np(ch, cw, overlap,
                                    left=x1 > 0, right=x2 < w, top=y1 > 0, bottom=y2 < h)
            acc[y1:y2, x1:x2, :] += out_arr * mask
            weight[y1:y2, x1:x2, :] += mask

    out = acc / np.clip(weight, 1e-6, None)
    return Image.fromarray((out * 255.0 + 0.5).astype(np.uint8))


# ----------------------------------------------------------------------------
# Orchestration : process_one, save, batch, run (UI/CLI commun)
# ----------------------------------------------------------------------------
def process_one(image, esrgan_model, factor, denoise, steps, prompt, seed, tile, overlap,
                refine_tile=DEFAULT_REFINE_TILE, refine_overlap=DEFAULT_REFINE_OVERLAP,
                do_esrgan=True):
    """Pipeline sur une PIL Image, renvoie (image, timings_dict).
    do_esrgan=False -> img2img pur (saute l'etage ESRGAN, refine sur l'image native)."""
    timings = {}
    image = image.convert("RGB")
    w0, h0 = image.size

    # Etage 1 : ESRGAN (saute si do_esrgan=False -> img2img pur)
    if do_esrgan and esrgan_model:
        t0 = time.time()
        _progress(0.15, f"ESRGAN upscale {w0}x{h0}...")
        model = load_esrgan(esrgan_model)
        _log(f"stage 1/2 ESRGAN upscale: {w0}x{h0} (tile {int(tile)}) ...")
        upscaled = esrgan_upscale(image, model, int(tile), int(overlap))
        target_w = round_to_multiple(w0 * factor)
        target_h = round_to_multiple(h0 * factor)
        upscaled = upscaled.resize((target_w, target_h), Image.LANCZOS)
        timings["esrgan"] = time.time() - t0
        _log(f"stage 1/2 done in {timings['esrgan']:.1f}s -> {target_w}x{target_h}")
    else:
        upscaled = image
        target_w, target_h = w0, h0
        timings["esrgan"] = 0.0
        _log(f"ESRGAN skipped (img2img only) on {w0}x{h0}")

    if denoise <= 0.001:
        timings["refine"] = 0.0
        _log("refine skipped (denoise = 0)")
        return upscaled, timings

    # Etage 2 : Z-Image img2img (image entiere, ou tuiles si refine_tile > 0)
    t0 = time.time()
    pipe = load_pipe()
    if int(refine_tile) > 0:
        refined = _refine_tiled(pipe, upscaled, denoise, steps, prompt, seed,
                                int(refine_tile), int(refine_overlap))
    else:
        _log(f"stage 2/2 Z-Image refine: whole image {target_w}x{target_h}, "
             f"denoise {float(denoise):.2f}, {int(steps)} steps ...")
        _progress(0.5, f"Z-Image refine {target_w}x{target_h}...")
        refined = _refine_whole(pipe, upscaled, denoise, steps, prompt, seed)
    timings["refine"] = time.time() - t0

    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    _progress(1.0, "Done")
    _log(f"stage 2/2 done in {timings['refine']:.1f}s | total "
         f"{timings['esrgan'] + timings['refine']:.1f}s")
    return refined, timings


def txt2img_run(prompt, width, height, gen_steps, seed, negative_prompt="",
                upscale=False, esrgan_model=None, factor=2.0, denoise=0.30, steps=12,
                tile=DEFAULT_TILE, overlap=DEFAULT_OVERLAP,
                refine_tile=DEFAULT_REFINE_TILE, refine_overlap=DEFAULT_REFINE_OVERLAP):
    """Genere une image (txt2img Z-Image) puis, si upscale=True, la passe dans le
    pipeline ESRGAN + refine. Renvoie (image, timings_dict)."""
    timings = {"txt2img": 0.0, "esrgan": 0.0, "refine": 0.0}
    t0 = time.time()
    base = generate(prompt, width, height, gen_steps, seed, negative_prompt)
    timings["txt2img"] = time.time() - t0
    if not upscale:
        return base, timings
    result, t = process_one(base, esrgan_model, factor, denoise, steps, prompt, seed,
                            tile, overlap, refine_tile=refine_tile, refine_overlap=refine_overlap)
    timings["esrgan"] = t.get("esrgan", 0.0)
    timings["refine"] = t.get("refine", 0.0)
    return result, timings


def build_output_path(source_path, save_mode, output_dir, output_format):
    """Decide ou ecrire l'image upscale. Renvoie un chemin absolu ou None (display only).
    - display    : pas de sauvegarde (None)
    - local      : output_dir relatif au projet
    - alongside  : dans le dossier de source_path
    - custom     : output_dir tel quel (peut etre absolu)
    """
    if save_mode == "display":
        return None

    ext = output_format.lower().lstrip(".")
    if ext not in SUPPORTED_FORMATS:
        ext = "png"

    if source_path:
        base = os.path.splitext(os.path.basename(source_path))[0]
    else:
        base = "image"
    fname = f"{base}_upscaled.{ext}"

    if save_mode == "alongside":
        if not source_path:
            raise ValueError("save_mode=alongside requires a source path (CLI or batch folder).")
        return os.path.join(os.path.dirname(os.path.abspath(source_path)), fname)

    if save_mode == "custom":
        target_dir = output_dir or DEFAULT_OUTPUT_DIR
    else:  # local
        target_dir = output_dir or DEFAULT_OUTPUT_DIR
        if not os.path.isabs(target_dir):
            target_dir = os.path.join(HERE, target_dir)

    os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, fname)


def save_image(img, dst_path, output_format):
    """Sauve avec le bon format Pillow."""
    fmt = output_format.lower().lstrip(".")
    if fmt in ("jpg", "jpeg"):
        img.convert("RGB").save(dst_path, "JPEG", quality=95)
    elif fmt == "webp":
        img.save(dst_path, "WEBP", quality=95, method=6)
    else:
        img.save(dst_path, "PNG")


def _list_folder_images(folder):
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(IMG_EXTS)
    )


def _format_timings(t, src_path=None, dst_path=None):
    total = t.get("esrgan", 0.0) + t.get("refine", 0.0)
    parts = []
    if src_path:
        parts.append(f"Source: `{src_path}`")
    parts.append(f"ESRGAN: **{t.get('esrgan', 0.0):.1f}s**  |  Z-Image refine: **{t.get('refine', 0.0):.1f}s**  |  Total: **{total:.1f}s**")
    if dst_path:
        parts.append(f"Saved: `{dst_path}`")
    return "  \n".join(parts)


def _reset_vram_peak():
    """Remet a zero le compteur de pic VRAM avant un traitement."""
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _report_vram():
    """Affiche le pic VRAM du run sur stderr. No-op hors CUDA.

    Format stable et parsable: la ligne commence par '[VRAM]'.
    alloue  = pic des tensors PyTorch (max_memory_allocated).
    reserve = pic du cache allocateur PyTorch (max_memory_reserved), plus proche
              de ce que nvidia-smi voit pour ce process.
    """
    if DEVICE != "cuda":
        print("[VRAM] pas de GPU CUDA, mesure ignoree.", file=sys.stderr)
        return
    alloc = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.max_memory_reserved() / 1024**3
    print(f"[VRAM] pic alloue: {alloc:.2f} Go | pic reserve: {reserved:.2f} Go",
          file=sys.stderr)


def run(image, source_folder, esrgan_model, factor, denoise, steps, prompt, seed,
        tile, overlap, save_mode=DEFAULT_SAVE_MODE, output_dir=DEFAULT_OUTPUT_DIR,
        output_format=DEFAULT_OUTPUT_FORMAT, time_log_path=None, print_output=False,
        refine_tile=DEFAULT_REFINE_TILE, refine_overlap=DEFAULT_REFINE_OVERLAP,
        do_esrgan=True):
    """Point d'entree commun UI / CLI.
    Renvoie (last_result_PIL, last_source_PIL, report_markdown).
    - Si source_folder est un dossier existant -> batch sur ses images.
    - Sinon, image est utilisee (PIL ou chemin str).
    - print_output: imprime le chemin absolu de chaque image sauvee sur stdout.
    - refine_tile > 0: passe Z-Image en tuiles (4K+, plafonne le pic VRAM).
    - do_esrgan=False: img2img pur (pas d'ESRGAN, juste le refine Z-Image).
    """
    if do_esrgan and not esrgan_model:
        raise gr.Error(f"No ESRGAN model found in {ESRGAN_DIR}.")

    # Mode batch
    if source_folder and os.path.isdir(source_folder):
        paths = _list_folder_images(source_folder)
        if not paths:
            raise gr.Error(f"No image in {source_folder}")
        last_result = last_source = None
        lines = [f"### Batch: {len(paths)} image(s) from `{source_folder}`"]
        t_batch = time.time()
        for p in paths:
            try:
                src = Image.open(p)
                result, t = process_one(src, esrgan_model, factor, denoise, steps,
                                        prompt, seed, tile, overlap,
                                        refine_tile=refine_tile, refine_overlap=refine_overlap,
                                        do_esrgan=do_esrgan)
                dst = build_output_path(p, save_mode, output_dir, output_format)
                if dst:
                    save_image(result, dst, output_format)
                    if print_output:
                        print(os.path.abspath(dst))
                _append_time_log(time_log_path, p, dst, t, save_mode, output_format)
                lines.append(f"- `{os.path.basename(p)}` {result.size[0]}x{result.size[1]} "
                             f"esrgan {t['esrgan']:.1f}s + refine {t['refine']:.1f}s"
                             + (f" -> `{dst}`" if dst else " (display)"))
                last_result, last_source = result, src.convert("RGB")
            except Exception as e:
                lines.append(f"- `{os.path.basename(p)}` FAILED: {e}")
        lines.append(f"**Batch total: {time.time()-t_batch:.1f}s**")
        return last_result, last_source, "  \n".join(lines)

    # Mode image unique
    if image is None:
        raise gr.Error("Load an image (or specify a source folder for batch mode).")
    if isinstance(image, str):
        source_path = image
        src_img = Image.open(source_path)
    else:
        source_path = None
        src_img = image

    result, t = process_one(src_img, esrgan_model, factor, denoise, steps,
                            prompt, seed, tile, overlap,
                            refine_tile=refine_tile, refine_overlap=refine_overlap,
                            do_esrgan=do_esrgan)
    dst = None
    try:
        dst = build_output_path(source_path, save_mode, output_dir, output_format)
    except ValueError as e:
        dst = None
        save_warning = f"  \n[WARN] {e}"
    else:
        save_warning = ""
    if dst:
        save_image(result, dst, output_format)
        if print_output:
            print(os.path.abspath(dst))
    _append_time_log(time_log_path, source_path, dst, t, save_mode, output_format)
    report = _format_timings(t, src_path=source_path, dst_path=dst) + save_warning
    return result, src_img.convert("RGB"), report


def _append_time_log(path, src, dst, t, save_mode, output_format):
    if not path:
        return
    try:
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        line = (f"{ts}\t{src or ''}\t{dst or ''}\t"
                f"esrgan={t.get('esrgan', 0):.2f}s\trefine={t.get('refine', 0):.2f}s\t"
                f"mode={save_mode}\tfmt={output_format}\n")
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        print(f"[AVERT] time-log echec: {e}", file=sys.stderr)


# ----------------------------------------------------------------------------
# UI Gradio
# ----------------------------------------------------------------------------
def _refresh_models(new_dir):
    """Change ESRGAN_DIR puis renvoie une mise a jour du Dropdown."""
    set_esrgan_dir(new_dir)
    models = list_esrgan_models()
    value = models[0] if models else None
    return gr.update(choices=models, value=value), f"{len(models)} model(s) found in {ESRGAN_DIR}"


def _apply_zimage(repo):
    set_zimage_model(repo)
    return f"Z-Image: {BASE_REPO} (will be (re)loaded on next run)"


def _save_paths_to_prefs(esrgan_dir, zimage_model):
    set_esrgan_dir(esrgan_dir)
    set_zimage_model(zimage_model)
    _save_prefs_keys({"esrgan_dir": ESRGAN_DIR, "zimage_model": BASE_REPO})
    return f"Saved to {PREFS_PATH}: esrgan_dir={ESRGAN_DIR}, zimage_model={BASE_REPO}"


# Ordre des composants mis a jour par le dropdown de presets (doit matcher l'UI).
_PRESET_UI_ORDER = ("factor", "denoise", "steps", "tile", "overlap",
                    "refine_tile", "refine_overlap", "cpu_offload")


def _apply_preset(name):
    """UI: renvoie les updates des controles pour le preset choisi (ordre _PRESET_UI_ORDER).
    Custom ou cle absente = pas de changement sur ce controle."""
    p = PRESETS.get(name, {})
    return [gr.update(value=p[k]) if k in p else gr.update() for k in _PRESET_UI_ORDER]


def _pil_to_b64_jpeg(img, max_side=1600, quality=85):
    """Reduit + encode en JPEG base64 pour embarquer en HTML sans saturer la page."""
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


def _make_compare_html(src_img, result_img):
    """Comparateur avant/apres standalone: 2 <img> superposees, slider range pilote un clip-path."""
    if src_img is None or result_img is None:
        return "<div style='padding:1em;color:#888'>No result to compare.</div>"
    src_b64 = _pil_to_b64_jpeg(src_img)
    res_b64 = _pil_to_b64_jpeg(result_img)
    uid = uuid.uuid4().hex[:8]
    return f"""
<div style="position:relative; max-width:100%; user-select:none;">
  <img src="data:image/jpeg;base64,{src_b64}" style="display:block; width:100%; height:auto;" alt="source" />
  <img id="cmp-top-{uid}" src="data:image/jpeg;base64,{res_b64}"
       style="position:absolute; top:0; left:0; display:block; width:100%; height:100%;
              clip-path: inset(0 50% 0 0); -webkit-clip-path: inset(0 50% 0 0);" alt="resultat" />
  <div id="cmp-bar-{uid}" style="position:absolute; top:0; left:50%; width:2px; height:100%;
       background:#fff; box-shadow:0 0 4px rgba(0,0,0,0.5); pointer-events:none;"></div>
  <input type="range" min="0" max="100" value="50"
         oninput="
           var v=this.value;
           document.getElementById('cmp-top-{uid}').style.clipPath='inset(0 '+(100-v)+'% 0 0)';
           document.getElementById('cmp-top-{uid}').style.webkitClipPath='inset(0 '+(100-v)+'% 0 0)';
           document.getElementById('cmp-bar-{uid}').style.left=v+'%';
         "
         style="position:absolute; bottom:10px; left:5%; width:90%; height:14px; cursor:ew-resize;" />
  <div style="position:absolute; top:8px; left:8px; padding:2px 8px; background:rgba(0,0,0,0.6); color:#fff;
              font-size:12px; border-radius:4px; pointer-events:none;">BEFORE</div>
  <div style="position:absolute; top:8px; right:8px; padding:2px 8px; background:rgba(0,0,0,0.6); color:#fff;
              font-size:12px; border-radius:4px; pointer-events:none;">AFTER</div>
</div>
"""


def _ui_run(image, source_folder, esrgan_model, factor, denoise, steps, prompt, seed,
            tile, overlap, offload_mode, refine_tile, refine_overlap, do_esrgan, guidance,
            save_mode, output_dir, output_format):
    """Adaptateur UI: appelle run() et renvoie (result_image, html_slider, report_markdown)."""
    set_offload_mode(offload_mode)
    set_guidance(guidance)
    last_result, last_source, report = run(
        image, source_folder, esrgan_model, factor, denoise, steps, prompt, seed,
        tile, overlap, save_mode=save_mode, output_dir=output_dir,
        output_format=output_format, refine_tile=refine_tile, refine_overlap=refine_overlap,
        do_esrgan=bool(do_esrgan),
    )
    html = _make_compare_html(last_source, last_result)
    return last_result, html, report


def _ui_txt2img(prompt, negative, width, height, gen_steps, seed, guidance, upscale,
                esrgan_model, factor, denoise, offload_mode):
    """Adaptateur UI txt2img: genere puis (optionnel) upscale. Renvoie (image, report)."""
    set_offload_mode(offload_mode)
    set_guidance(guidance)
    result, t = txt2img_run(prompt, width, height, gen_steps, seed, negative,
                            upscale=bool(upscale), esrgan_model=esrgan_model,
                            factor=factor, denoise=denoise, steps=DEFAULT_STEPS)
    rep = f"txt2img **{result.size[0]}x{result.size[1]}** in **{t['txt2img']:.1f}s**"
    if upscale:
        rep += (f"  \nESRGAN **{t['esrgan']:.1f}s** + refine **{t['refine']:.1f}s** "
                f"-> **{result.size[0]}x{result.size[1]}**")
    return result, rep


def _ui_generate(prompt, negative, use_input, input_image,
                 width, height, gen_steps, seed, guidance, offload_mode,
                 esrgan_model, do_esrgan, factor, denoise, refine_steps,
                 tile, overlap, refine_tile, refine_overlap,
                 save_mode, output_dir, output_format,
                 progress=gr.Progress(track_tqdm=True)):
    """Bouton Generate unifie facon Fooocus: si une image d'entree est fournie ->
    img2img / upscale, sinon txt2img. Renvoie (image, report_markdown).
    Affiche l'avancement des etapes (ESRGAN, refine, tuiles i/N) via gr.Progress."""
    global _PROGRESS
    _PROGRESS = lambda f, d: progress(f, desc=d)
    progress(0.0, desc="Starting...")
    try:
        set_offload_mode(offload_mode)
        set_guidance(guidance)
        if use_input and input_image is not None:
            last_result, last_source, report = run(
                input_image, None, esrgan_model, factor, denoise, refine_steps, prompt, seed,
                tile, overlap, save_mode=save_mode, output_dir=output_dir,
                output_format=output_format, refine_tile=refine_tile, refine_overlap=refine_overlap,
                do_esrgan=bool(do_esrgan))
            return last_result, report
        # txt2img
        result, t = txt2img_run(prompt, width, height, gen_steps, seed, negative,
                                upscale=False, steps=refine_steps)
        progress(1.0, desc="Done")
        dst = None
        if save_mode != "display":
            try:
                dst = build_output_path(None, save_mode, output_dir, output_format)
                if dst:
                    save_image(result, dst, output_format)
            except Exception as e:
                dst = f"(save error: {e})"
        rep = f"txt2img **{result.size[0]}x{result.size[1]}** in **{t['txt2img']:.1f}s**"
        if dst:
            rep += f"  \nSaved: `{dst}`"
        return result, rep
    finally:
        _PROGRESS = None


FOOOCUS_CSS = """
.gradio-container { max-width: 1280px !important; margin: auto !important; }
#cz_result { min-height: 520px; }
#cz_prompt textarea { font-size: 1.05rem; }
#cz_generate { height: 88px; font-size: 1.1rem; }
"""


def build_ui():
    models = list_esrgan_models()
    default_model = DEFAULT_MODEL if DEFAULT_MODEL in models else (models[0] if models else None)

    with gr.Blocks(title="crispz-studio", theme=gr.themes.Default(), css=FOOOCUS_CSS) as demo:
        with gr.Row():
            # ----- Colonne principale (grand apercu + barre de prompt en bas) -----
            with gr.Column(scale=3):
                out = gr.Image(type="pil", label="Result", elem_id="cz_result",
                               height=560, show_download_button=True)
                report = gr.Markdown(value="*Ready. Type a prompt and press Generate.*")

                use_input = gr.Checkbox(value=False, label="Input Image (img2img / upscale)")
                with gr.Group(visible=False) as input_group:
                    with gr.Row():
                        inp = gr.Image(type="pil", label="Source image", height=240)
                        with gr.Column():
                            do_esrgan_cb = gr.Checkbox(value=True, label="ESRGAN upscale",
                                                       info="Uncheck = img2img only (no enlargement).")
                            preset = gr.Dropdown(list(PRESETS), value="Custom", label="Use case preset")

                with gr.Row():
                    prompt = gr.Textbox(show_label=False, placeholder="Type your prompt here...",
                                        elem_id="cz_prompt", lines=2, scale=4, container=False)
                    btn = gr.Button("Generate", variant="primary", elem_id="cz_generate", scale=1, min_width=120)
                advanced_cb = gr.Checkbox(value=False, label="Advanced")

            # ----- Colonne Advanced (a droite, masquee par defaut) -----
            with gr.Column(scale=2, visible=False) as advanced_col:
                with gr.Tabs():
                    with gr.Tab("Setting"):
                        negative = gr.Textbox(label="Negative prompt", lines=2,
                                              placeholder="(needs guidance > 0)")
                        with gr.Row():
                            width = gr.Slider(256, 2048, value=1024, step=16, label="Width")
                            height = gr.Slider(256, 2048, value=1024, step=16, label="Height")
                        gen_steps = gr.Slider(2, 40, value=8, step=1, label="Generation steps (txt2img)")
                        guidance = gr.Slider(0.0, 8.0, value=0.0, step=0.5, label="CFG guidance",
                                             info="0 = Z-Image Turbo. Z-Image Base: ~3.5-5.")
                        seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)

                    with gr.Tab("Upscale / img2img"):
                        esrgan = gr.Dropdown(models, value=default_model, label="ESRGAN model")
                        factor = gr.Slider(1.0, 4.0, value=DEFAULT_FACTOR, step=0.5, label="Upscale factor")
                        denoise = gr.Slider(0.0, 0.8, value=DEFAULT_DENOISE, step=0.01,
                                            label="Refine denoise (strength)")
                        refine_steps = gr.Slider(4, 30, value=DEFAULT_STEPS, step=1, label="Refine steps")
                        with gr.Accordion("ESRGAN tiling (VRAM)", open=False):
                            tile = gr.Slider(0, 1024, value=DEFAULT_TILE, step=8, label="Tile (0 = off)")
                            overlap = gr.Slider(0, 128, value=DEFAULT_OVERLAP, step=8, label="Overlap")
                        with gr.Accordion("Z-Image tiling (4K+)", open=False):
                            refine_tile = gr.Slider(0, 2048, value=DEFAULT_REFINE_TILE, step=16,
                                                    label="Diffusion tile (0 = whole image)")
                            refine_overlap = gr.Slider(0, 256, value=DEFAULT_REFINE_OVERLAP, step=16,
                                                       label="Diffusion tile overlap")

                    with gr.Tab("Model"):
                        zimage_model_tb = gr.Textbox(
                            value=BASE_REPO,
                            label="Z-Image (HF repo, diffusers folder, or .safetensors file)",
                            info="A .safetensors (Civitai) = transformer; VAE+encoder from base repo.")
                        esrgan_dir_tb = gr.Textbox(value=ESRGAN_DIR, label="ESRGAN_DIR (.pth/.safetensors folder)")
                        offload = gr.Dropdown(choices=list(OFFLOAD_CHOICES), value="none",
                                              label="CPU offload (VRAM)",
                                              info="none | model (~half) | sequential (~9GB, slower)")
                        with gr.Row():
                            refresh_btn = gr.Button("Refresh ESRGAN", size="sm")
                            apply_zimage_btn = gr.Button("Apply Z-Image", size="sm", variant="primary")
                            save_paths_btn = gr.Button("Save paths", size="sm")
                        paths_status = gr.Markdown("")

                    with gr.Tab("Save"):
                        save_mode = gr.Radio(choices=["display", "local", "alongside", "custom"],
                                             value=DEFAULT_SAVE_MODE, label="Save mode")
                        output_dir = gr.Textbox(value=DEFAULT_OUTPUT_DIR, label="Output folder")
                        output_format = gr.Dropdown(choices=list(SUPPORTED_FORMATS),
                                                    value=DEFAULT_OUTPUT_FORMAT, label="Output format")

        # Toggles facon Fooocus
        advanced_cb.change(lambda v: gr.update(visible=bool(v)), advanced_cb, advanced_col)
        use_input.change(lambda v: gr.update(visible=bool(v)), use_input, input_group)

        # Actions
        refresh_btn.click(_refresh_models, [esrgan_dir_tb], [esrgan, paths_status])
        apply_zimage_btn.click(_apply_zimage, [zimage_model_tb], [paths_status])
        save_paths_btn.click(_save_paths_to_prefs, [esrgan_dir_tb, zimage_model_tb], [paths_status])
        preset.change(_apply_preset, [preset],
                      [factor, denoise, refine_steps, tile, overlap, refine_tile, refine_overlap, offload])
        btn.click(
            _ui_generate,
            inputs=[prompt, negative, use_input, inp, width, height, gen_steps, seed, guidance, offload,
                    esrgan, do_esrgan_cb, factor, denoise, refine_steps,
                    tile, overlap, refine_tile, refine_overlap, save_mode, output_dir, output_format],
            outputs=[out, report],
        )
    return demo


# ----------------------------------------------------------------------------
# Palier 3 : serveur HTTP persistant (FastAPI), load paresseux + unload sur idle
# ----------------------------------------------------------------------------
def serve_main(host="127.0.0.1", port=7861, idle_timeout=300):
    """Petit serveur HTTP. Le modele Z-Image se charge au premier /upscale et reste
    chaud (plus de rechargement entre appels -> temps stables). Apres idle_timeout
    secondes sans requete, la VRAM est rendue (utile pour cohabiter avec Fooocus).
    Endpoints: GET /health, GET /models, POST /upscale, POST /unload."""
    try:
        import threading
        import uvicorn
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except Exception as e:
        print("[serve] FastAPI/uvicorn required: pip install fastapi uvicorn", file=sys.stderr)
        print(f"[serve] detail: {e}", file=sys.stderr)
        return 1

    os.makedirs(ESRGAN_DIR, exist_ok=True)
    app = FastAPI(title="crispz")
    lock = threading.Lock()
    state = {"last": time.time()}

    class UpscaleReq(BaseModel):
        input: str
        model: str = DEFAULT_MODEL
        factor: float = DEFAULT_FACTOR
        denoise: float = DEFAULT_DENOISE
        steps: int = DEFAULT_STEPS
        prompt: str = ""
        seed: int = -1
        tile: int = DEFAULT_TILE
        overlap: int = DEFAULT_OVERLAP
        refine_tile: int = DEFAULT_REFINE_TILE
        refine_overlap: int = DEFAULT_REFINE_OVERLAP
        cpu_offload: str = "none"
        preset: str = "Custom"
        save_mode: str = "local"
        output_dir: str = DEFAULT_OUTPUT_DIR
        output_format: str = DEFAULT_OUTPUT_FORMAT

    @app.get("/health")
    def health():
        return {"status": "ok", "device": DEVICE, "pipe_loaded": _BASE_PIPE is not None,
                "offload": OFFLOAD_MODE, "idle_timeout": idle_timeout}

    @app.get("/models")
    def models():
        return {"esrgan_dir": ESRGAN_DIR, "models": list_esrgan_models()}

    @app.post("/unload")
    def unload():
        with lock:
            free_vram()
        return {"status": "unloaded"}

    @app.post("/upscale")
    def upscale(req: UpscaleReq):
        if not os.path.isfile(req.input):
            raise HTTPException(status_code=400, detail=f"input not found: {req.input}")
        avail = list_esrgan_models()
        if not avail:
            raise HTTPException(status_code=400, detail=f"no ESRGAN model in {ESRGAN_DIR}")
        # preset (s'il est fourni) sert de base; sinon les champs de la requete.
        p = PRESETS.get(req.preset or "Custom") or {}
        def pick(name, val):
            return p.get(name, val)
        model = req.model if req.model in avail else avail[0]
        with lock:
            state["last"] = time.time()
            set_offload_mode(pick("cpu_offload", req.cpu_offload))
            img = Image.open(req.input)
            result, t = process_one(
                img, model, pick("factor", req.factor), pick("denoise", req.denoise),
                pick("steps", req.steps), req.prompt, req.seed,
                pick("tile", req.tile), pick("overlap", req.overlap),
                refine_tile=pick("refine_tile", req.refine_tile),
                refine_overlap=pick("refine_overlap", req.refine_overlap),
            )
            dst = build_output_path(req.input, req.save_mode, req.output_dir, req.output_format)
            if dst:
                save_image(result, dst, req.output_format)
            state["last"] = time.time()
        return {"output": os.path.abspath(dst) if dst else None,
                "size": list(result.size),
                "esrgan_s": round(t.get("esrgan", 0.0), 2),
                "refine_s": round(t.get("refine", 0.0), 2),
                "total_s": round(t.get("esrgan", 0.0) + t.get("refine", 0.0), 2)}

    def _idle_watch():
        period = min(30, max(5, idle_timeout // 4)) if idle_timeout > 0 else 30
        while True:
            time.sleep(period)
            if idle_timeout > 0 and _BASE_PIPE is not None and (time.time() - state["last"]) > idle_timeout:
                with lock:
                    if _BASE_PIPE is not None and (time.time() - state["last"]) > idle_timeout:
                        free_vram()
                        print(f"[serve] model unloaded after {idle_timeout}s idle", file=sys.stderr)

    if idle_timeout and idle_timeout > 0:
        threading.Thread(target=_idle_watch, daemon=True).start()
    print(f"[serve] crispz on http://{host}:{port}  (idle unload: {idle_timeout}s)", file=sys.stderr)
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


# ----------------------------------------------------------------------------
# CLI (mode batch / scripting)
# ----------------------------------------------------------------------------
def cli_main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="crispz-studio CLI (txt2img + upscale). No args: launches the Gradio UI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cli", action="store_true", help="Force CLI mode (otherwise: launches the UI)")
    # Sources : fichier, glob, dossier
    parser.add_argument("-i", "--input", help="Image, glob (in/*.png) or source FOLDER for batch")
    parser.add_argument("--input-folder", help="Explicit alias for the batch folder (otherwise -i works too)")
    # Sortie
    parser.add_argument("-o", "--output",
                        help="Output file (single mode, overrides auto naming). "
                             "If a folder: equivalent to --save-mode local --output-dir <that folder>.")
    parser.add_argument("--save-mode", choices=["display", "local", "alongside", "custom"],
                        default=DEFAULT_SAVE_MODE,
                        help="display=no save | local=output_dir relative to project | "
                             "alongside=same folder as the source | custom=output_dir as-is")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Output folder for --save-mode local/custom")
    parser.add_argument("--output-format", choices=list(SUPPORTED_FORMATS),
                        default=DEFAULT_OUTPUT_FORMAT, help="Output format (png/webp/jpg)")
    # Pipeline
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL,
                        help="ESRGAN model (file in ESRGAN_DIR). Fallback: first found.")
    parser.add_argument("--factor", type=float, default=DEFAULT_FACTOR, help="Net upscale factor")
    parser.add_argument("--denoise", type=float, default=DEFAULT_DENOISE, help="Z-Image strength (0 = ESRGAN only)")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Diffusion steps")
    parser.add_argument("--prompt", default="", help="Optional prompt")
    parser.add_argument("--seed", type=int, default=-1, help="Seed (-1 = random)")
    parser.add_argument("--tile", type=int, default=DEFAULT_TILE, help="ESRGAN tile size (0 = disabled)")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP, help="ESRGAN tiling overlap")
    parser.add_argument("--refine-tile", type=int, default=DEFAULT_REFINE_TILE,
                        help="Z-Image diffusion tile size (0 = whole image). >0 tiles the "
                             "refine pass: caps VRAM and enables 4K+ without seams. Try 1024-1280.")
    parser.add_argument("--refine-overlap", type=int, default=DEFAULT_REFINE_OVERLAP,
                        help="Overlap (feather) of the Z-Image diffusion tiles")
    parser.add_argument("--cpu-offload", choices=list(OFFLOAD_CHOICES), default="none",
                        help="CPU offload of the diffusion pass (VRAM). none=all in VRAM | "
                             "model=offload per submodule (good tradeoff) | "
                             "sequential=more aggressive, slower. Requires accelerate.")
    parser.add_argument("--guidance", type=float, default=0.0,
                        help="CFG guidance scale. 0 for Z-Image Turbo (default). "
                             "Z-Image Base needs ~3.5-5 (and ~20+ steps).")
    parser.add_argument("--no-esrgan", action="store_true",
                        help="img2img only: skip the ESRGAN upscale, just run the Z-Image refine "
                             "on the input at native size (no enlargement).")
    parser.add_argument("--preset", choices=list(PRESETS), default="Custom",
                        help="Use-case preset (auto settings). Explicit flags override it.")
    # Text -> Image (txt2img)
    parser.add_argument("--txt2img", action="store_true",
                        help="Generate an image from --prompt (Z-Image txt2img) instead of "
                             "reading -i. Add --upscale to also run ESRGAN + refine.")
    parser.add_argument("--gen-width", type=int, default=1024, help="txt2img width (mult. of 16)")
    parser.add_argument("--gen-height", type=int, default=1024, help="txt2img height (mult. of 16)")
    parser.add_argument("--gen-steps", type=int, default=8, help="txt2img steps (Z-Image Turbo)")
    parser.add_argument("--negative", default="", help="Negative prompt (txt2img)")
    parser.add_argument("--upscale", action="store_true",
                        help="In --txt2img: run the ESRGAN + refine upscale on the generated image")
    # Server (stage 3)
    parser.add_argument("--serve", action="store_true",
                        help="Run a persistent HTTP server (lazy model load + idle unload) "
                             "instead of the UI/one-shot. Requires fastapi + uvicorn.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (--serve)")
    parser.add_argument("--port", type=int, default=7861, help="Server port (--serve)")
    parser.add_argument("--idle-timeout", type=int, default=300,
                        help="Seconds of inactivity before the server frees VRAM (0 = never)")
    # Chemins config / Z-Image
    parser.add_argument("--esrgan-dir", help="Override ESRGAN_DIR for this run")
    parser.add_argument("--zimage-model",
                        help="Override Z-Image: HF repo, diffusers folder, OR a single-file "
                             ".safetensors (Civitai) used as the transformer (VAE+encoder from base).")
    parser.add_argument("--zimage-transformer",
                        help="Single-file .safetensors transformer override (Civitai), keeping "
                             "the VAE + Qwen3 encoder from --zimage-model / the base repo.")
    parser.add_argument("--save-paths", action="store_true",
                        help="Save --esrgan-dir and --zimage-model to preferences.json")
    # Reports
    parser.add_argument("--list-models", action="store_true", help="List ESRGAN models then exit")
    parser.add_argument("--time-log", default=None,
                        help="If set, append the time of each run to this file (TSV)")
    parser.add_argument("--quiet", action="store_true", help="Reduce stdout verbosity")
    parser.add_argument("--report-vram", action="store_true",
                        help="Print the run VRAM peak on stderr (line '[VRAM] ...'). "
                             "Used to size coexistence with Fooocus.")
    parser.add_argument("--print-output", action="store_true",
                        help="Print ONLY the absolute output path on stdout (one per saved "
                             "image), nothing else. For external integration (Fooocus). "
                             "Implies a silent stdout; the VRAM peak stays on stderr.")
    args = parser.parse_args(argv)
    apply_preset_to_args(args, argv if argv is not None else sys.argv[1:])

    global VERBOSE
    VERBOSE = not args.quiet

    if args.esrgan_dir:
        set_esrgan_dir(args.esrgan_dir)
    if args.zimage_model:
        set_zimage_model(args.zimage_model)
    if args.zimage_transformer:
        set_zimage_transformer(args.zimage_transformer)
    set_offload_mode(args.cpu_offload)
    set_guidance(args.guidance)

    if args.serve:
        return serve_main(args.host, args.port, args.idle_timeout)

    # Mode txt2img (Text -> Image, + upscale optionnel)
    if args.txt2img:
        if not args.prompt:
            parser.error("--txt2img requires --prompt")
        os.makedirs(ESRGAN_DIR, exist_ok=True)
        model_name = None
        if args.upscale:
            avail = list_esrgan_models()
            if not avail:
                parser.error(f"--upscale needs an ESRGAN model in {ESRGAN_DIR}")
            model_name = args.model if args.model in avail else avail[0]
        if args.report_vram:
            _reset_vram_peak()
        result, t = txt2img_run(
            args.prompt, args.gen_width, args.gen_height, args.gen_steps, args.seed,
            args.negative, upscale=args.upscale, esrgan_model=model_name,
            factor=args.factor, denoise=args.denoise, steps=args.steps,
            tile=args.tile, overlap=args.overlap,
            refine_tile=args.refine_tile, refine_overlap=args.refine_overlap)
        # Sortie : -o fichier, sinon output_dir (sauf save-mode display)
        dst = None
        if args.output and not (os.path.isdir(args.output) or args.output.endswith(("/", "\\"))):
            dst = args.output
            os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
        elif args.save_mode != "display":
            out_dir = args.output or args.output_dir
            if not os.path.isabs(out_dir):
                out_dir = os.path.join(HERE, out_dir)
            os.makedirs(out_dir, exist_ok=True)
            ext = args.output_format if args.output_format in SUPPORTED_FORMATS else "png"
            dst = os.path.join(out_dir, f"txt2img.{ext}")
        if dst:
            save_image(result, dst, args.output_format)
            if args.print_output:
                print(os.path.abspath(dst))
        quiet = args.quiet or args.print_output
        if not quiet:
            parts = [f"txt2img {result.size[0]}x{result.size[1]} in {t['txt2img']:.1f}s"]
            if args.upscale:
                parts.append(f"esrgan {t['esrgan']:.1f}s + refine {t['refine']:.1f}s")
            if dst:
                parts.append(f"-> {dst}")
            print("  |  ".join(parts))
        if args.report_vram:
            _report_vram()
        return 0

    if args.save_paths:
        _save_prefs_keys({"esrgan_dir": ESRGAN_DIR, "zimage_model": BASE_REPO})
        print(f"Saved to {PREFS_PATH}: esrgan_dir={ESRGAN_DIR}, zimage_model={BASE_REPO}")
        if not args.input and not args.input_folder:
            return 0

    os.makedirs(ESRGAN_DIR, exist_ok=True)
    models = list_esrgan_models()

    if args.list_models:
        if not models:
            print(f"No model in {ESRGAN_DIR}")
        else:
            for m in models:
                print(m)
        return 0

    # Pas de --cli et pas d'entree -> UI
    if not args.cli and not args.input and not args.input_folder:
        build_ui().launch()
        return 0

    if not models and not args.no_esrgan:
        parser.error(f"No ESRGAN model in {ESRGAN_DIR} (or use --no-esrgan for img2img only)")

    model_name = (args.model if args.model in models else (models[0] if models else None))

    if args.report_vram:
        _reset_vram_peak()

    # Resoudre les entrees : dossier > glob > fichier unique
    source_folder = args.input_folder
    if not source_folder and args.input and os.path.isdir(args.input):
        source_folder = args.input
        args.input = None

    # --output (compat) : si c'est un dossier, equivalent a --save-mode local --output-dir <dossier>
    save_mode = args.save_mode
    output_dir = args.output_dir
    explicit_output_file = None
    if args.output:
        if os.path.isdir(args.output) or args.output.endswith(("/", "\\")):
            save_mode = "custom" if os.path.isabs(args.output) else "local"
            output_dir = args.output
        else:
            explicit_output_file = args.output
            save_mode = "custom"

    # --print-output: stdout reserve aux chemins de sortie (contrat machine).
    # Le pic VRAM, lui, reste sur stderr et n'est donc pas pollue.
    quiet = args.quiet or args.print_output

    # Mode batch dossier
    if source_folder:
        last_result, last_source, report = run(
            None, source_folder, model_name, args.factor, args.denoise, args.steps,
            args.prompt, args.seed, args.tile, args.overlap,
            save_mode=save_mode, output_dir=output_dir,
            output_format=args.output_format, time_log_path=args.time_log,
            print_output=args.print_output,
            refine_tile=args.refine_tile, refine_overlap=args.refine_overlap,
            do_esrgan=not args.no_esrgan,
        )
        if not quiet:
            print(report)
        if args.report_vram:
            _report_vram()
        return 0

    # Mode unique : glob possible
    paths = sorted(glob.glob(args.input)) if any(c in args.input for c in "*?[") else [args.input]
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        parser.error(f"No file matches {args.input}")

    # Si plusieurs fichiers via glob, on les passe un par un
    for p in paths:
        if not quiet:
            print(f"-> {p}")
        img = Image.open(p)
        # explicit_output_file ne s'applique qu'au premier fichier
        if explicit_output_file and len(paths) == 1:
            result, t = process_one(img, model_name, args.factor, args.denoise, args.steps,
                                    args.prompt, args.seed, args.tile, args.overlap,
                                    refine_tile=args.refine_tile, refine_overlap=args.refine_overlap,
                                    do_esrgan=not args.no_esrgan)
            os.makedirs(os.path.dirname(os.path.abspath(explicit_output_file)) or ".", exist_ok=True)
            save_image(result, explicit_output_file, args.output_format)
            if args.print_output:
                print(os.path.abspath(explicit_output_file))
            _append_time_log(args.time_log, p, explicit_output_file, t, "custom", args.output_format)
            if not quiet:
                print(_format_timings(t, src_path=p, dst_path=explicit_output_file))
        else:
            # mode standard: build_output_path applique le save_mode
            last_result, last_source, report = run(
                p, None, model_name, args.factor, args.denoise, args.steps,
                args.prompt, args.seed, args.tile, args.overlap,
                save_mode=save_mode, output_dir=output_dir,
                output_format=args.output_format, time_log_path=args.time_log,
                print_output=args.print_output,
                refine_tile=args.refine_tile, refine_overlap=args.refine_overlap,
                do_esrgan=not args.no_esrgan,
            )
            if not quiet:
                print(report)
    if args.report_vram:
        _report_vram()
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
