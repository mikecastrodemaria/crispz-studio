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
import random
import threading
import warnings

# Masque les DeprecationWarning Gradio "pass theme/css/js to launch() instead":
# launch() ne les accepte PAS encore en Gradio 5.x (avertissement anticipe Gradio 6),
# donc on les garde dans Blocks(...) et on coupe juste le bruit au demarrage.
warnings.filterwarnings(
    "ignore", category=DeprecationWarning,
    message=r"The '(theme|css|js)' parameter in the Blocks constructor will be removed")
import base64
import glob
import time
import uuid
import datetime
import numpy as np
import torch
from PIL import Image, ImageChops
import gradio as gr

# Support AVIF/HEIC en entree (les .avif sinon: PIL.UnidentifiedImageError).
try:
    import pillow_avif  # noqa: F401  enregistre l'ouvreur AVIF dans PIL
except Exception:
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except Exception:
        pass


def _disable_brotli():
    """Neutralise le brotli_middleware de Gradio (bug h11 'Content-Length' a l'envoi
    de gros resultats). Patch sur le symbole importe par gradio.routes."""
    class _Passthrough:
        def __init__(self, app, *a, **k):
            self.app = app

        async def __call__(self, scope, receive, send):
            await self.app(scope, receive, send)
    import importlib
    for modname in ("gradio.routes", "gradio.brotli_middleware"):
        try:
            m = importlib.import_module(modname)
            if hasattr(m, "BrotliMiddleware"):
                m.BrotliMiddleware = _Passthrough
        except Exception:
            pass

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
IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".avif", ".heic")

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

# Ratios d'aspect facon Fooocus. label -> (width, height) en multiples de 16.
ASPECT_RATIOS = {
    "1024 x 1024  (1:1)":  (1024, 1024),
    "1152 x 896  (9:7)":   (1152, 896),
    "896 x 1152  (7:9)":   (896, 1152),
    "1216 x 832  (3:2)":   (1216, 832),
    "832 x 1216  (2:3)":   (832, 1216),
    "1344 x 768  (16:9)":  (1344, 768),
    "768 x 1344  (9:16)":  (768, 1344),
    "1536 x 640  (21:9)":  (1536, 640),
}
# Performance facon Fooocus -> (gen_steps, guidance) pour le modele charge.
PERFORMANCE = {
    "Turbo (8 steps)":    (8, 0.0),
    "Quality (20 steps)": (20, 0.0),
    "Base CFG (28 steps)": (28, 4.0),
}
# Styles. Format Fooocus: nom -> {"prompt": template avec {prompt} (ou None),
# "negative_prompt": str}. La vraie biblio est chargee depuis styles/*.json (cf.
# _load_styles plus bas). Ceci n'est qu'un fallback si le dossier est absent.
_FALLBACK_STYLES = {
    "Fooocus Cinematic": {"prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
                          "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"},
    "SAI Photographic": {"prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
                         "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly"},
    "SAI Anime": {"prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime, highly detailed",
                  "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast"},
}


def _style_sample(name):
    """Chemin de la vignette d'un style (styles/samples/<nom>.jpg) ou None."""
    try:
        fn = name.lower().replace(" ", "_").replace("-", "_") + ".jpg"
        p = os.path.join(HERE, "styles", "samples", fn)
        return p if os.path.isfile(p) else None
    except Exception:
        return None


def _filter_styles(query, selected):
    """Filtre la liste des styles par recherche. Conserve les styles deja coches."""
    q = (query or "").strip().lower()
    matches = [n for n in STYLES if q in n.lower()] if q else list(STYLES)
    selected = [s for s in (selected or []) if s in STYLES]
    # choices = resultats + styles coches (pour ne pas perdre la selection)
    choices = list(dict.fromkeys(matches + selected))
    return gr.update(choices=choices, value=selected)


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


def _apply_wildcards(text, rng=None):
    """Remplace les __nom__ par une ligne aleatoire de wildcards/nom.txt (gere
    l'imbrication: une ligne peut contenir d'autres __wildcards__)."""
    if not text or "__" not in text:
        return text
    import re
    rng = rng or random
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
                    repl = rng.choice(lines)
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


# ----------------------------------------------------------------------------
# Ollama (Describe image -> prompt, Improve prompt). + fallback local BLIP.
# ----------------------------------------------------------------------------
def _ollama_http(path, payload=None, base=None, timeout=8):
    import urllib.request
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


def _ollama_compose(captions, model, base=None):
    """'Faux Omni': fusionne plusieurs descriptions d'images en UN seul prompt."""
    listing = "\n".join(f"Image {i + 1}: {c}" for i, c in enumerate(captions) if c)
    instr = (COMPOSE_INSTRUCTION.replace("{descriptions}", listing)
             if "{descriptions}" in COMPOSE_INSTRUCTION
             else f"{COMPOSE_INSTRUCTION}\n\n{listing}")
    out = _ollama_http("/api/generate", {"model": model, "prompt": instr, **_ollama_gen_opts()},
                       base=base, timeout=120)
    return (out.get("response") or "").strip()


_BLIP = None


def _local_caption(image):
    """Fallback local (sans Ollama): petit captioneur BLIP via transformers (lazy)."""
    global _BLIP
    if _BLIP is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        _log("loading local captioner BLIP base (first time downloads ~1GB)...")
        proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        mdl = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base").to(DEVICE)
        _BLIP = (proc, mdl)
    proc, mdl = _BLIP
    inputs = proc(image.convert("RGB"), return_tensors="pt").to(DEVICE)
    out = mdl.generate(**inputs, max_new_tokens=50)
    return proc.decode(out[0], skip_special_tokens=True).strip()


# ----------------------------------------------------------------------------
# FaceSwap (post-process, optionnel). InsightFace + modele inswapper. Active
# seulement si insightface/onnxruntime sont installes ET faceswap_model_path
# pointe sur un inswapper (.onnx). Sinon -> message clair (feature gated).
# ----------------------------------------------------------------------------
_FACE_APP = None
_FACE_SWAPPER = None


def _resolve_faceswap_model():
    """Trouve le modele inswapper: faceswap_model_path, sinon recherche dans des
    emplacements usuels, sinon telechargement si faceswap_model_url est defini."""
    cfg = (os.environ.get("FACESWAP_MODEL") or CONFIG.get("faceswap_model_path") or "").strip()
    cands = [cfg] if cfg else []
    search_dirs = [os.path.join(HERE, "faceswap"), os.path.join(HERE, "models"),
                   CHECKPOINTS_DIR, os.path.join(os.path.expanduser("~"), ".insightface", "models")]
    for d in search_dirs:
        cands += [os.path.join(d, "inswapper_128.onnx"),
                  os.path.join(d, "inswapper_128_fp16.onnx")]
    for p in cands:
        if p and os.path.isfile(p):
            return p
    # Telechargement optionnel (URL fournie par l'utilisateur dans config.txt).
    url = (CONFIG.get("faceswap_model_url") or "").strip()
    if url:
        import urllib.request
        dst_dir = os.path.join(HERE, "faceswap")
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, "inswapper_128.onnx")
        _log(f"downloading inswapper model from {url} ...")
        urllib.request.urlretrieve(url, dst)
        return dst
    return None


def _faceswap(target_img, source_img):
    """Remplace le(s) visage(s) de target_img par celui de source_img."""
    global _FACE_APP, _FACE_SWAPPER
    try:
        import insightface
        from insightface.app import FaceAnalysis
    except Exception:
        raise RuntimeError("insightface not installed (pip install insightface onnxruntime-gpu).")
    model_path = _resolve_faceswap_model()
    if not model_path:
        raise RuntimeError(
            "inswapper model not found. Put 'inswapper_128.onnx' in the 'faceswap' folder "
            "(next to app.py), or set 'faceswap_model_path' in config.txt, or set "
            "'faceswap_model_url' to download it once.")
    if _FACE_APP is None:
        _log("loading insightface buffalo_l (face detection) ...")
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=0 if DEVICE == "cuda" else -1, det_size=(640, 640))
        _FACE_APP = app
    if _FACE_SWAPPER is None:
        _log(f"loading inswapper: {model_path}")
        _FACE_SWAPPER = insightface.model_zoo.get_model(model_path)
    tgt = np.asarray(target_img.convert("RGB"))[:, :, ::-1].copy()  # RGB -> BGR
    src = np.asarray(source_img.convert("RGB"))[:, :, ::-1].copy()
    src_faces = _FACE_APP.get(src)
    if not src_faces:
        raise RuntimeError("No face found in the source image.")
    src_face = max(src_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    tgt_faces = _FACE_APP.get(tgt)
    if not tgt_faces:
        raise RuntimeError("No face found in the generated image.")
    res = tgt
    for f in tgt_faces:
        res = _FACE_SWAPPER.get(res, f, src_face, paste_back=True)
    out_img = Image.fromarray(res[:, :, ::-1])  # BGR -> RGB
    # Restauration optionnelle du visage (GFPGAN) -> nettete (inswapper sort en 128px).
    if FACESWAP_RESTORE:
        out_img = _face_restore(out_img, tgt_faces, FACESWAP_RESTORE_BLEND)
    return out_img


# FFHQ 5-point template (alignement attendu par GFPGAN), normalise -> x512.
_FFHQ_512 = np.array([
    [0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904],
    [0.39308822, 0.72541100], [0.61150205, 0.72490465]], dtype=np.float32) * 512.0
_FACE_RESTORE_SESSION = None


def _resolve_face_restore_model():
    """Trouve le modele GFPGAN (.onnx): config, emplacements usuels, sinon download."""
    cfg = (CONFIG.get("faceswap_restore_path") or "").strip()
    cands = [cfg] if cfg else []
    for d in (os.path.join(HERE, "faceswap"), os.path.join(HERE, "models")):
        cands += [os.path.join(d, "gfpgan_1.4.onnx")]
    for p in cands:
        if p and os.path.isfile(p):
            return p
    url = (CONFIG.get("faceswap_restore_url") or "").strip()
    if url:
        import urllib.request
        dst_dir = os.path.join(HERE, "faceswap")
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, "gfpgan_1.4.onnx")
        _log(f"downloading GFPGAN restore model from {url} ...")
        urllib.request.urlretrieve(url, dst)
        return dst
    return None


def _get_face_restore_session():
    global _FACE_RESTORE_SESSION
    if _FACE_RESTORE_SESSION is not None:
        return _FACE_RESTORE_SESSION
    path = _resolve_face_restore_model()
    if not path:
        _log("GFPGAN restore model not found -> skip restore "
             "(set faceswap_restore_url/path or drop gfpgan_1.4.onnx in faceswap/).")
        return None
    import onnxruntime as ort
    # On evite TensorRT (nvinfer_*.dll souvent absent) -> CUDA puis CPU.
    provs = [p for p in ort.get_available_providers() if p != "TensorrtExecutionProvider"]
    _log(f"loading GFPGAN restore: {path} (providers={provs})")
    _FACE_RESTORE_SESSION = ort.InferenceSession(path, providers=provs)
    return _FACE_RESTORE_SESSION


def _face_restore(image, faces, blend=0.8):
    """Restaure (GFPGAN ONNX) chaque visage detecte: aligne en 512 (template FFHQ),
    debruite/affine, recolle avec un masque adouci. Renvoie l'image PIL."""
    sess = _get_face_restore_session()
    if sess is None:
        return image
    import cv2
    arr = np.asarray(image.convert("RGB"))[:, :, ::-1].astype(np.uint8).copy()  # BGR
    h, w = arr.shape[:2]
    iname = sess.get_inputs()[0].name
    for f in faces:
        try:
            M, _ = cv2.estimateAffinePartial2D(f.kps.astype(np.float32), _FFHQ_512,
                                               method=cv2.LMEDS)
            if M is None:
                continue
            aligned = cv2.warpAffine(arr, M, (512, 512), borderMode=cv2.BORDER_REPLICATE)
            blob = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            blob = ((blob - 0.5) / 0.5).transpose(2, 0, 1)[None].astype(np.float32)
            out = sess.run(None, {iname: blob})[0][0]
            out = np.clip(out.transpose(1, 2, 0) * 0.5 + 0.5, 0, 1)
            out = cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            IM = cv2.invertAffineTransform(M)
            back = cv2.warpAffine(out, IM, (w, h))
            # Masque ELLIPTIQUE adouci dans l'espace du crop 512 (s'estompe AVANT
            # les bords) -> recollage sans bord carre visible.
            m512 = np.zeros((512, 512), np.uint8)
            cv2.ellipse(m512, (256, 256), (256 - 28, 256 - 28), 0, 0, 360, 255, -1)
            m512 = cv2.GaussianBlur(m512, (0, 0), 24)
            mask = cv2.warpAffine(m512, IM, (w, h)).astype(np.float32) / 255.0
            mask = (mask * float(blend))[:, :, None]
            arr = (back * mask + arr * (1 - mask)).astype(np.uint8)
        except Exception as e:
            _log(f"face restore (one face) skipped: {e}")
    return Image.fromarray(arr[:, :, ::-1])


def set_faceswap_restore(enabled, blend):
    """Active/desactive la restauration GFPGAN apres le swap + son intensite."""
    global FACESWAP_RESTORE, FACESWAP_RESTORE_BLEND
    FACESWAP_RESTORE = bool(enabled)
    FACESWAP_RESTORE_BLEND = float(blend)
    return f"Face restore (GFPGAN): {'on' if enabled else 'off'} (blend {blend})"


def _remove_bg(image):
    """Detoure le sujet (fond transparent). Local via rembg (telecharge u2net au
    1er usage). Renvoie une image RGBA."""
    try:
        from rembg import remove
    except Exception:
        raise RuntimeError("rembg not installed. pip install rembg (or requirements-faceswap.txt).")
    return remove(image.convert("RGBA"))

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


CONFIG_PATH = os.path.join(HERE, "config.txt")
CONFIG_SAMPLE_PATH = os.path.join(HERE, "config-sample.txt")


def _load_config():
    """Charge la config (JSON, facon Fooocus): defauts + strings d'instruction
    Ollama. Priorite: config.txt (local, gitignore) -> config-sample.txt (livre) ->
    {} (les valeurs codees servent de repli)."""
    for path in (CONFIG_PATH, CONFIG_SAMPLE_PATH):
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f) or {}
            except Exception:
                pass
    return {}


CONFIG = _load_config()

# Defauts pilotes par config.txt (repli sur les constantes deja definies plus haut).
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

# Presets Performance editables via config.txt (performance_presets: nom -> [steps, guidance]).
if isinstance(CONFIG.get("performance_presets"), dict) and CONFIG["performance_presets"]:
    try:
        PERFORMANCE = {k: (int(v[0]), float(v[1])) for k, v in CONFIG["performance_presets"].items()}
    except Exception:
        pass

# Profils par modele: substring du nom -> reglages recommandes (steps/guidance).
# Quand on choisit un modele, les sliders s'ajustent automatiquement (contextuel).
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
# Dossiers de modeles Z-Image (comme ESRGAN_DIR): checkpoints single-file a switcher
# + LoRA a appliquer. Path config (checkpoints_dir / loras_dir) ou defaut local.
CHECKPOINTS_DIR = (os.environ.get("CHECKPOINTS_DIR") or _prefs.get("checkpoints_dir")
                   or CONFIG.get("checkpoints_dir") or os.path.join(HERE, "checkpoints"))
LORAS_DIR = (os.environ.get("LORAS_DIR") or _prefs.get("loras_dir")
             or CONFIG.get("loras_dir") or os.path.join(HERE, "loras"))
WILDCARDS_DIR = (os.environ.get("WILDCARDS_DIR") or _prefs.get("wildcards_dir")
                 or CONFIG.get("wildcards_dir") or os.path.join(HERE, "wildcards"))
# LoRA active (chemin .safetensors) + poids. Inclus dans la clef de cache du pipe.
# LoRA actives: liste de (chemin, poids). Plusieurs LoRA combinables (multi-slots).
LORAS = []
LORA_WEIGHT = float(CONFIG.get("default_lora_weight", 1.0))  # poids par defaut des slots
# Modele Omni/Edit (multi-reference). Reglable via config.txt ou l'UI.
OMNI_MODEL = (os.environ.get("ZIMAGE_OMNI_MODEL") or CONFIG.get("zimage_omni_model") or "").strip()
# Ollama (Describe image->prompt + Improve prompt). URL configurable, persistee.
OLLAMA_URL = (os.environ.get("OLLAMA_URL") or _prefs.get("ollama_url")
              or CONFIG.get("ollama_url") or "http://localhost:11434")
# Duree de maintien du modele Ollama en VRAM apres un appel (keep_alive). 0 =
# decharge immediatement -> libere la VRAM avant la generation Z-Image (evite la
# concurrence VRAM sur un seul GPU). Peut etre un nombre (s) ou "30s"/"5m"/-1.
OLLAMA_KEEP_ALIVE = CONFIG.get("ollama_keep_alive", 0)
# Force Ollama sur CPU (num_gpu=0) -> 0 VRAM partagee avec Z-Image (plus lent).
OLLAMA_CPU = bool(CONFIG.get("ollama_cpu", False))


def _ollama_gen_opts():
    """Options communes pour /api/generate (keep_alive + CPU optionnel)."""
    p = {"stream": False, "keep_alive": OLLAMA_KEEP_ALIVE}
    if OLLAMA_CPU:
        p["options"] = {"num_gpu": 0}
    return p
# FaceSwap: restauration GFPGAN post-swap (nettete du visage). Reglable via l'UI.
FACESWAP_RESTORE = bool(CONFIG.get("faceswap_restore", False))
FACESWAP_RESTORE_BLEND = float(CONFIG.get("faceswap_restore_blend", 0.8))
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


# Niveau de log sur stderr. 0 = quiet, 1 = info (etapes), 2 = debug (params, etat
# pipe, timings detailles -> aide au dev). Source: env CRISPZ_LOG_LEVEL, sinon 1.
# stderr donc ne pollue pas le stdout de --print-output.
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


def _log(msg, level=1):
    if LOG_LEVEL >= level:
        print(f"[crispz] {msg}", file=sys.stderr, flush=True)


def _dbg(msg):
    """Log niveau debug (visible seulement en LOG_LEVEL >= 2)."""
    if LOG_LEVEL >= 2:
        print(f"[crispz][dbg] {msg}", file=sys.stderr, flush=True)


# Hook de progression UI (gradio gr.Progress). None hors UI (CLI/serveur). Permet
# d'afficher l'avancement des etapes (ESRGAN, refine, tuiles i/N) dans l'interface.
_PROGRESS = None


def _progress(frac, desc=""):
    if _PROGRESS is not None:
        try:
            _PROGRESS(min(1.0, max(0.0, float(frac))), desc)
        except Exception:
            pass


# Stop "facon Fooocus": flag global + interruption des pipelines diffusers.
_STOP = False


def request_stop():
    """Demande l'arret: stoppe la boucle de debruitage en cours (pipe._interrupt) et
    les boucles batch/tuiles (_STOP). Quasi-immediat (s'arrete au pas suivant)."""
    global _STOP
    _STOP = True
    n = 0
    for p in [_BASE_PIPE] + list(_DERIVED.values()):
        if p is not None:
            try:
                p._interrupt = True
                n += 1
            except Exception:
                pass
    _log(f"STOP requested (interrupt set on {n} pipeline(s))")
    return "Stopping..."


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


def _safetensors_is_fp8(path):
    """Vrai si le .safetensors contient des tenseurs FP8 (F8_E4M3/E5M2) -> ne charge
    pas dans diffusers. Lit juste l'en-tete (rapide)."""
    try:
        import struct
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(min(n, 2_000_000)).decode("utf-8", "ignore"))
        for k, v in hdr.items():
            if k != "__metadata__" and isinstance(v, dict):
                if str(v.get("dtype", "")).upper().startswith("F8"):
                    return True
    except Exception:
        pass
    return False


def list_checkpoints():
    """Modeles Z-Image single-file (.safetensors) du dossier checkpoints. Exclut les
    checkpoints FP8 (non charges par diffusers; prendre la version BF16/FP16)."""
    if not os.path.isdir(CHECKPOINTS_DIR):
        return []
    out = []
    for f in sorted(os.listdir(CHECKPOINTS_DIR)):
        if not f.lower().endswith((".safetensors", ".ckpt", ".pt", ".sft")):
            continue
        if f.lower().endswith(".safetensors") and _safetensors_is_fp8(os.path.join(CHECKPOINTS_DIR, f)):
            _log(f"checkpoint skipped (FP8, not supported by diffusers): {f}")
            continue
        out.append(f)
    return out


def list_loras():
    """LoRA (.safetensors) du dossier loras."""
    if not os.path.isdir(LORAS_DIR):
        return []
    return sorted(f for f in os.listdir(LORAS_DIR)
                  if f.lower().endswith((".safetensors", ".ckpt", ".pt")))


def set_checkpoints_dir(path):
    global CHECKPOINTS_DIR
    if path:
        CHECKPOINTS_DIR = path


def set_loras_dir(path):
    global LORAS_DIR
    if path:
        LORAS_DIR = path


def set_wildcards_dir(path):
    global WILDCARDS_DIR
    if path:
        WILDCARDS_DIR = path


def _read_safetensors_metadata(path):
    """Lit le header JSON (__metadata__) d'un .safetensors SANS charger les poids."""
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = f.read(n)
    return (json.loads(header.decode("utf-8")) or {}).get("__metadata__", {}) or {}


def lora_keywords(path):
    """Extrait les mots-cles / trigger words d'une LoRA depuis ses metadonnees:
    champs trigger explicites + top tags d'entrainement (ss_tag_frequency)."""
    if not path or not os.path.isfile(path):
        return ""
    try:
        meta = _read_safetensors_metadata(path)
    except Exception as e:
        _dbg(f"lora metadata read failed: {e}")
        return ""
    words = []
    for k in ("ss_trigger_words", "modelspec.trigger_phrase", "trigger_words",
              "activation text", "ss_activation_text"):
        v = meta.get(k)
        if v:
            words.append(v if isinstance(v, str) else ", ".join(map(str, v)))
    tf = meta.get("ss_tag_frequency")
    if tf:
        try:
            d = json.loads(tf) if isinstance(tf, str) else tf
            counts = {}
            for ds in d.values():
                for tag, c in ds.items():
                    counts[tag] = counts.get(tag, 0) + int(c)
            words.extend(sorted(counts, key=counts.get, reverse=True)[:15])
        except Exception:
            pass
    seen, out = set(), []
    for w in words:
        for part in str(w).split(","):
            part = part.strip()
            if part and part.lower() not in seen:
                seen.add(part.lower())
                out.append(part)
    return ", ".join(out)


def set_loras(slots):
    """Definit les LoRA actives. slots = liste de (nom_ou_None, poids). Resout les
    noms en chemins, ignore les None. Invalide le pipe si la combinaison change."""
    global LORAS
    new = []
    for name, weight in slots:
        if name and name not in ("None", "none", ""):
            p = name if os.path.isabs(name) else os.path.join(LORAS_DIR, name)
            new.append((p, float(weight)))
    if new != LORAS:
        LORAS = new
        free_vram()
        _log("LoRAs -> " + (", ".join(f"{os.path.basename(p)}@{w}" for p, w in new) or "(none)")
             + " -> will reload")


def set_omni_model(repo):
    """Definit le modele Omni/Edit (repo HF ou dossier). Invalide le pipe omni."""
    global OMNI_MODEL
    repo = (repo or "").strip()
    if repo != OMNI_MODEL:
        OMNI_MODEL = repo
        _DERIVED.pop("omni", None)
        _log(f"Omni model -> {repo or '(none)'}")


def check_omni_available():
    """Teste l'existence des repos Omni/Edit sur Hugging Face (API publique)."""
    import urllib.request
    found = []
    for repo in ("Tongyi-MAI/Z-Image-Omni-Base", "Tongyi-MAI/Z-Image-Edit"):
        try:
            req = urllib.request.Request("https://huggingface.co/api/models/" + repo,
                                         headers={"User-Agent": "crispz-studio"})
            with urllib.request.urlopen(req, timeout=8) as r:
                if r.status == 200:
                    found.append(repo)
        except Exception:
            pass
    if found:
        return ("**Omni model available!** " + ", ".join(f"`{r}`" for r in found)
                + " - set it in config.txt `zimage_omni_model` (or Models tab).")
    return ("Not released yet. Z-Image-Omni-Base / Z-Image-Edit are still 'coming "
            "soon'. The Omni tab will work once they ship.")


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
    key = (BASE_REPO, ZIMAGE_TRANSFORMER, OFFLOAD_MODE, tuple(LORAS))
    _dbg(f"_ensure_base key={key} cached={_LOADED_KEY}")
    if _BASE_PIPE is not None and _LOADED_KEY == key:
        _dbg("base pipeline: reusing cached (no reload)")
        return _BASE_PIPE
    if _BASE_PIPE is not None:
        _dbg("base pipeline: key changed -> free + reload")
        free_vram()
    from diffusers import ZImagePipeline, ZImageTransformer2DModel
    t0 = time.time()
    kwargs = {}
    if ZIMAGE_TRANSFORMER:
        if _is_single_file(ZIMAGE_TRANSFORMER):
            _log(f"loading Z-Image transformer (single-file): {ZIMAGE_TRANSFORMER} ...")
            kwargs["transformer"] = ZImageTransformer2DModel.from_single_file(
                ZIMAGE_TRANSFORMER, torch_dtype=DTYPE)
        else:
            # repo HF / dossier diffusers -> charge le sous-dossier 'transformer'
            # (utile pour les modeles comme Juggernaut-Z dont le tokenizer est
            # incomplet: on garde VAE + encodeur + tokenizer du repo de base).
            _log(f"loading Z-Image transformer (repo subfolder): {ZIMAGE_TRANSFORMER} ...")
            kwargs["transformer"] = ZImageTransformer2DModel.from_pretrained(
                ZIMAGE_TRANSFORMER, subfolder="transformer", torch_dtype=DTYPE)
    _log(f"loading Z-Image base: {BASE_REPO} (offload={OFFLOAD_MODE}, dtype=bf16) ... "
         "first time downloads from HF, then cached")
    pipe = ZImagePipeline.from_pretrained(BASE_REPO, torch_dtype=DTYPE, **kwargs)
    # LoRA Z-Image (sur le transformer du base -> partage par les pipes derives).
    if LORAS:
        try:
            names, weights = [], []
            for i, (p, w) in enumerate(LORAS):
                if os.path.isfile(p):
                    an = f"cz_lora_{i}"
                    _log(f"applying LoRA: {os.path.basename(p)} (weight {w})")
                    pipe.load_lora_weights(p, adapter_name=an)
                    names.append(an)
                    weights.append(float(w))
            if names:
                pipe.set_adapters(names, weights)
        except Exception as e:
            _log(f"LoRA load failed ({e}); continuing without LoRA")
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
    """Renvoie le pipeline demande. txt2img/img2img/inpaint derivent du base via
    from_pipe (poids partages). Omni a besoin de composants en plus (SigLIP) ->
    charge separement depuis un modele Omni dedie (CONFIG['zimage_omni_model'])."""
    base = _ensure_base()
    if kind in _DERIVED:
        _dbg(f"get_pipe('{kind}'): reuse derived")
        return _DERIVED[kind]
    if kind == "omni":
        return _load_omni()
    from diffusers import ZImageImg2ImgPipeline, ZImageInpaintPipeline
    cls = {"img2img": ZImageImg2ImgPipeline, "inpaint": ZImageInpaintPipeline}.get(kind)
    if cls is None:
        return base
    _log(f"deriving {kind} pipeline (shared weights, no extra VRAM)")
    p = cls.from_pipe(base)
    _DERIVED[kind] = p
    return p


def _load_omni():
    """Charge le pipeline Omni (multi-reference). Necessite un modele Z-Image
    Omni/Edit (avec encodeur SigLIP) -> CONFIG['zimage_omni_model'] ou env
    ZIMAGE_OMNI_MODEL. Pipeline separe (ne partage pas avec le base)."""
    global _DERIVED
    from diffusers import ZImageOmniPipeline
    repo = (OMNI_MODEL or os.environ.get("ZIMAGE_OMNI_MODEL")
            or CONFIG.get("zimage_omni_model") or "").strip()
    if not repo:
        raise RuntimeError(
            "Omni needs a dedicated Z-Image Omni/Edit model (with a SigLIP encoder that "
            "the Turbo/Base text-to-image models do not ship). As of now Tongyi has only "
            "released Z-Image-Turbo and Z-Image-Base; 'Z-Image-Omni-Base' and 'Z-Image-Edit' "
            "are still 'coming soon'. Once published, set 'zimage_omni_model' in config.txt "
            "to its HF repo id (likely 'Tongyi-MAI/Z-Image-Omni-Base' or 'Tongyi-MAI/"
            "Z-Image-Edit') or a local diffusers folder.")
    _log(f"loading Z-Image Omni: {repo} (offload={OFFLOAD_MODE}) ...")
    t0 = time.time()
    pipe = ZImageOmniPipeline.from_pretrained(repo, torch_dtype=DTYPE)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    if DEVICE == "cuda" and OFFLOAD_MODE == "model":
        pipe.enable_model_cpu_offload()
    elif DEVICE == "cuda" and OFFLOAD_MODE == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe = pipe.to(DEVICE)
    _DERIVED["omni"] = pipe
    _log(f"Z-Image Omni ready in {time.time() - t0:.1f}s")
    return pipe


def generate_omni(refs, prompt, negative, width, height, steps, seed):
    """Omni multi-reference: compose une image a partir de plusieurs images de
    reference + un prompt (ex. personne + vetement). ZImageOmniPipeline natif."""
    refs = [r for r in (refs or []) if r is not None]
    if not refs:
        raise ValueError("Omni needs at least one reference image.")
    pipe = get_pipe("omni")
    w = round_to_multiple(int(width))
    h = round_to_multiple(int(height))
    _log(f"omni: {len(refs)} ref(s) -> {w}x{h}, {int(steps)} steps, guidance {GUIDANCE:.1f} ...")
    _progress(0.1, f"Omni compose ({len(refs)} refs)...")
    t0 = time.time()
    out = pipe(
        image=[r.convert("RGB") for r in refs],
        prompt=prompt or "",
        negative_prompt=(negative or None),
        width=w, height=h,
        num_inference_steps=int(steps),
        guidance_scale=GUIDANCE,
        generator=_make_generator(seed),
    ).images[0]
    _log(f"omni done in {time.time() - t0:.1f}s")
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


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
    _dbg(f"txt2img seed={seed} dtype=bf16 device={DEVICE} offload={OFFLOAD_MODE} "
         f"transformer={'single-file' if ZIMAGE_TRANSFORMER else 'repo'}")
    if DEVICE == "cuda":
        _dbg(f"VRAM before: alloc={torch.cuda.memory_allocated()/1024**3:.2f} Go")
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
    if DEVICE == "cuda":
        _dbg(f"VRAM peak: alloc={torch.cuda.max_memory_allocated()/1024**3:.2f} Go | "
             f"reserved={torch.cuda.max_memory_reserved()/1024**3:.2f} Go")
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return img


def round_to_multiple(x, m=16):
    return max(m, int(round(x / m) * m))


def _reframe_canvas(image, ratio_w, ratio_h, overlap=8):
    """Place l'image dans un canevas plus grand au ratio cible (expansion sur 1 axe),
    + un masque (blanc = a remplir, noir = a garder, avec un petit overlap)."""
    from PIL import ImageDraw
    image = image.convert("RGB")
    w, h = image.size
    r = ratio_w / ratio_h
    # Alignement sur 32 (patch 2 x VAE 16): evite les erreurs de conv (no engine).
    if w / h < r:  # trop etroit -> elargir
        nw, nh = round_to_multiple(int(round(h * r)), 32), round_to_multiple(h, 32)
    else:          # trop large -> agrandir en hauteur
        nw, nh = round_to_multiple(w, 32), round_to_multiple(int(round(w / r)), 32)
    nw, nh = max(nw, round_to_multiple(w, 32)), max(nh, round_to_multiple(h, 32))
    ox, oy = (nw - w) // 2, (nh - h) // 2
    canvas = Image.new("RGB", (nw, nh), (127, 127, 127))
    canvas.paste(image, (ox, oy))
    mask = Image.new("L", (nw, nh), 255)
    ImageDraw.Draw(mask).rectangle(
        [ox + overlap, oy + overlap, ox + w - overlap, oy + h - overlap], fill=0)
    return canvas, mask, nw, nh


def inpaint_run(background, mask, prompt, steps, denoise, seed):
    """Inpaint: regenere la zone blanche du masque selon le prompt
    (ZImageInpaintPipeline). background + mask = PIL (L: blanc = a changer)."""
    bg = background.convert("RGB")
    w = round_to_multiple(bg.width, 32)
    h = round_to_multiple(bg.height, 32)
    if (w, h) != bg.size:
        bg = bg.resize((w, h), Image.LANCZOS)
        mask = mask.resize((w, h), Image.NEAREST)
    pipe = get_pipe("inpaint")
    _log(f"inpaint: {w}x{h}, {int(steps)} steps, strength {float(denoise):.2f}, "
         f"guidance {GUIDANCE:.1f} ...")
    _progress(0.1, "Inpainting...")
    t0 = time.time()
    out = pipe(prompt=prompt or "", image=bg, mask_image=mask, strength=float(denoise),
               num_inference_steps=int(steps), guidance_scale=GUIDANCE,
               generator=_make_generator(seed)).images[0]
    _log(f"inpaint done in {time.time() - t0:.1f}s")
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


def _editor_to_image_mask(editor_value):
    """Extrait (image, masque) d'un gr.ImageEditor. Masque = zone peinte (diff
    composite/background), blanc = a regenerer."""
    if not editor_value:
        return None, None
    bg = editor_value.get("background")
    comp = editor_value.get("composite")
    if bg is None:
        return None, None
    bg = bg.convert("RGB")
    mask = Image.new("L", bg.size, 0)
    if comp is not None:
        diff = ImageChops.difference(comp.convert("RGB"), bg).convert("L")
        mask = diff.point(lambda p: 255 if p > 8 else 0)
    # fallback: alpha des layers peints
    for ly in (editor_value.get("layers") or []):
        if ly is not None:
            a = ly.convert("RGBA").split()[-1]
            mask = ImageChops.lighter(mask, a.point(lambda p: 255 if p > 8 else 0))
    return bg, mask


def _editor_img(v):
    """Extrait l'image PIL d'un gr.ImageEditor (dict {background,composite,layers})
    ou renvoie le PIL tel quel (retro-compat). Renvoie l'image recadree."""
    if isinstance(v, dict):
        return v.get("composite") or v.get("background")
    return v


def _crop_input(label, height=280):
    """Entree image avec recadrage (crop) facon Fooocus, sans pinceau ni calques."""
    return gr.ImageEditor(type="pil", label=label, height=height,
                          sources=["upload", "clipboard"], brush=False, eraser=False,
                          layers=False, transforms=["crop"])


def outpaint(image, ratio_w, ratio_h, prompt, steps, seed):
    """Reframe / outpainting: agrandit l'image au ratio cible et fait remplir les
    bords par Z-Image (ZImageInpaintPipeline)."""
    canvas, mask, nw, nh = _reframe_canvas(image, ratio_w, ratio_h)
    pipe = get_pipe("inpaint")
    _log(f"outpaint: {image.size[0]}x{image.size[1]} -> {nw}x{nh}, {int(steps)} steps, "
         f"guidance {GUIDANCE:.1f} ...")
    _progress(0.1, f"Outpaint -> {nw}x{nh}...")
    t0 = time.time()
    out = pipe(prompt=prompt or "", image=canvas, mask_image=mask, strength=1.0,
               num_inference_steps=int(steps), guidance_scale=GUIDANCE,
               generator=_make_generator(seed)).images[0]
    if out.size != (nw, nh):
        out = out.resize((nw, nh), Image.LANCZOS)
    _log(f"outpaint done in {time.time() - t0:.1f}s")
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


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
            if _STOP:
                _log("refine tiled: stop requested")
                break
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
    _dbg(f"process_one in={w0}x{h0} factor={factor} denoise={denoise} steps={int(steps)} "
         f"do_esrgan={do_esrgan} esrgan={esrgan_model} refine_tile={int(refine_tile)}")

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


def _now_stamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def _unique_path(path):
    """Evite l'ecrasement: ajoute _2, _3... si le fichier existe deja."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 2
    while os.path.exists(f"{base}_{i}{ext}"):
        i += 1
    return f"{base}_{i}{ext}"


def _format_filename(tag, seed, w, h, index=0):
    """Nom de fichier depuis CONFIG['filename_pattern']. Placeholders: {date} {tag}
    {seed} {w} {h} {index} {name}. Defaut: date + tag + seed + dimensions + index."""
    pat = CONFIG.get("filename_pattern", "{date}_{tag}_seed{seed}_{w}x{h}{index}")
    seed_s = str(int(seed)) if (seed is not None and int(seed) >= 0) else "rand"
    idx_s = f"_{int(index)}" if index else ""
    try:
        name = pat.format(date=_now_stamp(), tag=(tag or "image"), seed=seed_s,
                          w=(w or 0), h=(h or 0), index=idx_s, name=(tag or "image"))
    except Exception:
        name = f"{_now_stamp()}_{tag or 'image'}_seed{seed_s}{idx_s}"
    name = "".join(c for c in name if c.isalnum() or c in "-_.").strip("_")
    return name or "image"


def build_output_path(source_path, save_mode, output_dir, output_format,
                      tag=None, seed=None, size=None, index=0):
    """Chemin de sortie (ou None si display). Le nom suit CONFIG['filename_pattern']
    (date + seed + tag + dimensions + index) et est rendu UNIQUE (pas d'ecrasement).
    tag = 'upscaled' / 'txt2img' / 'img2img' (+ nom source si fourni)."""
    if save_mode == "display":
        return None
    ext = output_format.lower().lstrip(".")
    if ext not in SUPPORTED_FORMATS:
        ext = "png"
    if not tag:
        srcbase = os.path.splitext(os.path.basename(source_path))[0] if source_path else "image"
        tag = f"{srcbase}_upscaled"
    w = size[0] if size else 0
    h = size[1] if size else 0
    fname = f"{_format_filename(tag, seed, w, h, index)}.{ext}"

    if save_mode == "alongside":
        if not source_path:
            raise ValueError("save_mode=alongside requires a source path (CLI or batch folder).")
        target_dir = os.path.dirname(os.path.abspath(source_path))
    elif save_mode == "custom":
        target_dir = output_dir or DEFAULT_OUTPUT_DIR
    else:  # local
        target_dir = output_dir or DEFAULT_OUTPUT_DIR
        if not os.path.isabs(target_dir):
            target_dir = os.path.join(HERE, target_dir)
    # Sous-dossier par date (facon Fooocus) pour local/custom, si active (defaut oui).
    if save_mode in ("local", "custom") and CONFIG.get("date_subfolders", True):
        target_dir = os.path.join(target_dir, datetime.datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(target_dir, exist_ok=True)
    return _unique_path(os.path.join(target_dir, fname))


def _exif_bytes(meta):
    """EXIF (ImageDescription=0x010e) contenant le JSON des metadonnees, pour jpg/webp."""
    try:
        exif = Image.Exif()
        exif[0x010E] = json.dumps(meta, ensure_ascii=False)  # ImageDescription
        return exif.tobytes()
    except Exception:
        return None


def save_image(img, dst_path, output_format, meta=None):
    """Sauve avec le bon format Pillow. Si meta (dict): embarque dans le PNG (chunk
    'crispz'), en EXIF (ImageDescription) pour jpg/webp, ET ecrit un sidecar .json."""
    fmt = output_format.lower().lstrip(".")
    if fmt in ("jpg", "jpeg"):
        kw = {"quality": 95}
        eb = _exif_bytes(meta) if meta else None
        if eb:
            kw["exif"] = eb
        img.convert("RGB").save(dst_path, "JPEG", **kw)
    elif fmt == "webp":
        kw = {"quality": 95, "method": 6}
        eb = _exif_bytes(meta) if meta else None
        if eb:
            kw["exif"] = eb
        img.save(dst_path, "WEBP", **kw)
    else:
        pnginfo = None
        if meta:
            try:
                from PIL import PngImagePlugin
                pnginfo = PngImagePlugin.PngInfo()
                pnginfo.add_text("crispz", json.dumps(meta, ensure_ascii=False))
            except Exception:
                pnginfo = None
        img.save(dst_path, "PNG", pnginfo=pnginfo)
    if meta:
        try:
            with open(dst_path + ".json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        except Exception as e:
            _dbg(f"sidecar json failed: {e}")


def _gen_meta(mode, prompt, negative="", seed=None, steps=None, guidance=None,
              size=None, model=None, extra=None):
    """Construit le dict de metadonnees de generation (pour sidecar/PNG)."""
    m = {"app": "crispz-studio", "mode": mode, "prompt": prompt or "",
         "negative": negative or "", "date": _now_stamp()}
    if seed is not None and int(seed) >= 0:
        m["seed"] = int(seed)
    if steps is not None:
        m["steps"] = int(steps)
    if guidance is not None:
        m["guidance"] = float(guidance)
    if size:
        m["size"] = f"{size[0]}x{size[1]}"
    m["model"] = model or (ZIMAGE_TRANSFORMER or BASE_REPO)
    if LORAS:
        m["loras"] = [f"{os.path.basename(p)}@{w}" for p, w in LORAS]
    if extra:
        m.update(extra)
    return m


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
                _srcbase = os.path.splitext(os.path.basename(p))[0]
                _tag = f"{_srcbase}_" + ("upscaled" if do_esrgan else "img2img")
                dst = build_output_path(p, save_mode, output_dir, output_format,
                                        tag=_tag, seed=seed, size=result.size)
                if dst:
                    save_image(result, dst, output_format, meta=_gen_meta(
                        "upscale" if do_esrgan else "img2img", prompt, seed=seed,
                        steps=steps, guidance=GUIDANCE, size=result.size,
                        extra={"source": os.path.basename(p), "factor": factor,
                               "denoise": denoise, "esrgan": esrgan_model if do_esrgan else None}))
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
    _srcbase = os.path.splitext(os.path.basename(source_path))[0] if source_path else None
    _tag = (f"{_srcbase}_" if _srcbase else "") + ("upscaled" if do_esrgan else "img2img")
    try:
        dst = build_output_path(source_path, save_mode, output_dir, output_format,
                                tag=_tag, seed=seed, size=result.size)
    except ValueError as e:
        dst = None
        save_warning = f"  \n[WARN] {e}"
    else:
        save_warning = ""
    if dst:
        save_image(result, dst, output_format, meta=_gen_meta(
            "upscale" if do_esrgan else "img2img", prompt, seed=seed, steps=steps,
            guidance=GUIDANCE, size=result.size,
            extra={"factor": factor, "denoise": denoise,
                   "esrgan": esrgan_model if do_esrgan else None}))
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


def _refresh_checkpoints(new_dir):
    """Change le dossier checkpoints + liste les modeles + persiste."""
    set_checkpoints_dir(new_dir)
    try:
        _save_prefs_keys({"checkpoints_dir": CHECKPOINTS_DIR})
    except Exception:
        pass
    cks = list_checkpoints()
    return gr.update(choices=["(base repo)"] + cks), f"{len(cks)} checkpoint(s) in {CHECKPOINTS_DIR} (saved)."


def _apply_checkpoint(name):
    """Selectionne un checkpoint single-file comme transformer (ou revient au base repo).
    Ajuste aussi steps/guidance selon le profil du modele (contextuel)."""
    if not name or name == "(base repo)":
        set_zimage_transformer("")
        st, g = profile_for_model(BASE_REPO)
        return ("Z-Image: base repo transformer (single-file cleared).",
                gr.update(value=st), gr.update(value=g))
    path = name if os.path.isabs(name) else os.path.join(CHECKPOINTS_DIR, name)
    set_zimage_transformer(path)
    st, g = profile_for_model(os.path.basename(path))
    return (f"Z-Image transformer: {os.path.basename(path)} -> auto steps={st}, CFG={g} "
            f"(reload on next run).", gr.update(value=st), gr.update(value=g))


def _apply_transformer_repo(repo):
    """Definit le transformer depuis un repo HF / dossier diffusers OU un .safetensors.
    Ajuste steps/guidance selon le profil du modele."""
    repo = (repo or "").strip()
    set_zimage_transformer(repo)
    if not repo:
        return "Transformer: from base repo.", gr.update(), gr.update()
    st, g = profile_for_model(repo)
    return (f"Transformer override: {repo} -> auto steps={st}, CFG={g} "
            f"(keeps base VAE/encoder; reload on next run).",
            gr.update(value=st), gr.update(value=g))


def _wild_sanitize(name):
    return "".join(c for c in (name or "").strip() if c.isalnum() or c in "_-")[:64]


def _ui_wild_refresh(new_dir):
    """Change le dossier + rafraichit le dropdown de tous les wildcards + persiste."""
    set_wildcards_dir(new_dir)
    try:
        _save_prefs_keys({"wildcards_dir": WILDCARDS_DIR})
    except Exception:
        pass
    w = list_wildcards()
    return gr.update(choices=["None"] + w, value="None"), \
        f"{len(w)} wildcard file(s) in {WILDCARDS_DIR} (saved)."


def _ui_wild_load(name):
    """Charge le contenu du wildcard selectionne dans l'editeur."""
    if not name or name == "None":
        return "", ""
    p = os.path.join(WILDCARDS_DIR, name + ".txt")
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        nlines = len([ln for ln in txt.splitlines() if ln.strip() and not ln.lstrip().startswith("#")])
        return txt, f"{name}: {nlines} option(s). Use __{name}__ in the prompt."
    except Exception as e:
        return "", f"Cannot read {name}: {e}"


def _ui_wild_insert(name, prompt_text):
    """Insere __name__ a la fin du prompt."""
    if not name or name == "None":
        return gr.update(), "Pick a wildcard file first."
    tok = f"__{name}__"
    base = (prompt_text or "").rstrip()
    new = (base + (" " if base else "") + tok)
    return gr.update(value=new), f"Inserted {tok}."


def _ui_wild_save(name, content):
    """Sauve le contenu de l'editeur dans le wildcard selectionne."""
    n = _wild_sanitize(name if name and name != "None" else "")
    if not n:
        return "Pick a wildcard file (or use Create new)."
    try:
        os.makedirs(WILDCARDS_DIR, exist_ok=True)
        with open(os.path.join(WILDCARDS_DIR, n + ".txt"), "w", encoding="utf-8") as f:
            f.write(content or "")
        return f"Saved {n}.txt."
    except Exception as e:
        return f"Save failed: {e}"


def _ui_wild_create(newname, content):
    """Cree un nouveau wildcard + rafraichit le dropdown."""
    n = _wild_sanitize(newname)
    if not n:
        return gr.update(), "Enter a valid name (letters/digits/_/-).", newname
    try:
        os.makedirs(WILDCARDS_DIR, exist_ok=True)
        with open(os.path.join(WILDCARDS_DIR, n + ".txt"), "w", encoding="utf-8") as f:
            f.write(content or "")
        return gr.update(choices=["None"] + list_wildcards(), value=n), f"Created {n}.txt.", ""
    except Exception as e:
        return gr.update(), f"Create failed: {e}", newname


def _refresh_loras(new_dir):
    """Change le dossier loras + liste les LoRA (met a jour les 3 slots) + persiste."""
    set_loras_dir(new_dir)
    try:
        _save_prefs_keys({"loras_dir": LORAS_DIR})   # persiste -> survit au reboot
    except Exception:
        pass
    lr = ["None"] + list_loras()
    return (gr.update(choices=lr), gr.update(choices=lr), gr.update(choices=lr),
            f"{len(lr) - 1} LoRA(s) in {LORAS_DIR} (saved).")


def _apply_loras(n1, w1, n2, w2, n3, w3):
    """Applique la combinaison des 3 slots LoRA."""
    set_loras([(n1, w1), (n2, w2), (n3, w3)])
    if not LORAS:
        return "LoRA: none."
    return "LoRA: " + ", ".join(f"{os.path.basename(p)}@{w}" for p, w in LORAS) + " (reload on next run)."


def _path_for_lora(name):
    if not name or name in ("None", "none", ""):
        return None
    return name if os.path.isabs(name) else os.path.join(LORAS_DIR, name)


def _ui_loras_apply(n1, w1, n2, w2, n3, w3):
    """Applique les slots + agrege les mots-cles des LoRA selectionnees."""
    status = _apply_loras(n1, w1, n2, w2, n3, w3)
    kws = []
    for n in (n1, n2, n3):
        p = _path_for_lora(n)
        if p:
            k = lora_keywords(p)
            if k:
                kws.append(k)
    return status, ", ".join(kws)


def _ui_loras_keywords(n1, n2, n3):
    """Recupere les mots-cles de toutes les LoRA selectionnees (bouton)."""
    kws = []
    for n in (n1, n2, n3):
        p = _path_for_lora(n)
        if p:
            k = lora_keywords(p)
            if k:
                kws.append(k)
    merged = ", ".join(kws)
    return merged, (f"{len(merged.split(','))} keyword(s)." if merged
                    else "No keywords in the selected LoRA(s).")


def _ui_kw_to_prompt(prompt_text, keywords):
    """Ajoute les mots-cles a la fin du prompt courant."""
    kw = (keywords or "").strip().strip(",").strip()
    if not kw:
        return gr.update()
    base = (prompt_text or "").strip()
    if base and not base.endswith(","):
        base += ", "
    return gr.update(value=base + kw)


def _ui_check_omni():
    return check_omni_available()


def _save_paths_to_prefs(esrgan_dir, zimage_model, checkpoints_dir=None, loras_dir=None,
                         wildcards_dir=None):
    """Persiste les chemins dans preferences.json (local) -> charges au prochain boot."""
    set_esrgan_dir(esrgan_dir)
    set_zimage_model(zimage_model)
    if checkpoints_dir:
        set_checkpoints_dir(checkpoints_dir)
    if loras_dir:
        set_loras_dir(loras_dir)
    if wildcards_dir:
        set_wildcards_dir(wildcards_dir)
    _save_prefs_keys({"esrgan_dir": ESRGAN_DIR, "zimage_model": BASE_REPO,
                      "checkpoints_dir": CHECKPOINTS_DIR, "loras_dir": LORAS_DIR,
                      "wildcards_dir": WILDCARDS_DIR})
    return (f"Saved to {PREFS_PATH}: esrgan_dir, zimage_model, checkpoints_dir, "
            f"loras_dir, wildcards_dir={WILDCARDS_DIR}")


# Ordre des composants mis a jour par le dropdown de presets (doit matcher l'UI).
_PRESET_UI_ORDER = ("factor", "denoise", "steps", "tile", "overlap",
                    "refine_tile", "refine_overlap", "cpu_offload")


def _apply_preset(name):
    """UI: renvoie les updates des controles pour le preset choisi (ordre _PRESET_UI_ORDER).
    Custom ou cle absente = pas de changement sur ce controle."""
    p = PRESETS.get(name, {})
    return [gr.update(value=p[k]) if k in p else gr.update() for k in _PRESET_UI_ORDER]


def _set_aspect(name):
    """UI: applique un ratio d'aspect -> (width, height)."""
    w, h = ASPECT_RATIOS.get(name, (1024, 1024))
    return w, h


def _set_performance(name):
    """UI: applique un preset Performance -> (gen_steps, guidance)."""
    steps, g = PERFORMANCE.get(name, (8, 0.0))
    return steps, g


def _ui_detect_ollama(url):
    """Detecte Ollama et liste UNIQUEMENT les modeles vision pour le Describe."""
    try:
        models = _ollama_vision_models(base=url)
    except Exception:
        return (gr.update(choices=[], value=None),
                "Ollama not reachable. Describe will use the local BLIP fallback. "
                "Improve needs Ollama.")
    if not models:
        return (gr.update(choices=[], value=None),
                "Ollama OK but no VISION model. Pull one, e.g. `ollama pull llava` or `moondream`.")
    return gr.update(choices=models, value=models[0]), f"Ollama OK - {len(models)} vision model(s)."


def _ui_describe(image, model, url):
    """Decrit l'image -> remplit le prompt. Ollama si modele choisi, sinon BLIP local."""
    image = _editor_img(image)
    if image is None:
        return gr.update(), "Drop an image to describe first."
    if model:
        try:
            return gr.update(value=_ollama_describe(image, model, base=url)), f"Described via {model}."
        except Exception as e:
            return gr.update(), f"Ollama describe failed: {e}"
    try:
        return gr.update(value=_local_caption(image)), "Described via local BLIP (no Ollama model)."
    except Exception as e:
        return gr.update(), f"No Ollama model selected and local captioner failed: {e}"


def _ui_improve(prompt_text, model, url):
    """Ameliore le prompt courant via Ollama (meme modele)."""
    if not (prompt_text or "").strip():
        return gr.update(), "Type a prompt first."
    if not model:
        return gr.update(), "Select a model in Advanced > Prompt AI (click Detect Ollama)."
    try:
        return gr.update(value=_ollama_improve(prompt_text, model, base=url)), f"Improved via {model}."
    except Exception as e:
        return gr.update(), f"Improve failed: {e}"


def _ui_compose(r1, r2, r3, r4, model, url):
    """'Faux Omni': decrit chaque image de reference (vision) puis fusionne en UN
    prompt via le LLM. Remplit la zone de prompt."""
    refs = [im for im in (_editor_img(r) for r in [r1, r2, r3, r4]) if im is not None]
    if not refs:
        return gr.update(), "Add at least one reference image."
    if not model:
        return gr.update(), "Select an Ollama vision model in Advanced > Prompt AI (Detect)."
    try:
        caps = [_ollama_describe(im, model, base=url) for im in refs]
    except Exception as e:
        return gr.update(), f"Describe failed: {e}"
    try:
        merged = _ollama_compose(caps, model, base=url)
    except Exception as e:
        return gr.update(), f"Compose failed: {e}"
    return gr.update(value=merged), f"Composed one prompt from {len(refs)} image(s) via {model}."


def _ui_remove_bg(image, history, save_mode, output_dir):
    """Remove background -> resultat (PNG transparent) dans la galerie + historique."""
    image = _editor_img(image)
    if image is None:
        return [], "Drop an image first.", history, history
    try:
        res = _remove_bg(image)
    except Exception as e:
        return [], f"Remove BG failed: {e}", history, history
    if save_mode != "display":
        try:
            dst = build_output_path(None, save_mode, output_dir, "png", tag="nobg", size=res.size)
            if dst:
                save_image(res, dst, "png")
        except Exception as e:
            _dbg(f"save nobg failed: {e}")
    new_hist = ([res] + list(history or []))[:200]
    return [res], "Background removed (transparent PNG).", new_hist, new_hist


def _ui_reframe(image, ratio, steps, prompt, guidance, offload_mode, seed,
                save_mode, output_dir, output_format, history,
                progress=gr.Progress(track_tqdm=True)):
    """Reframe / outpaint -> resultat dans la galerie + historique."""
    image = _editor_img(image)
    if image is None:
        return [], "Drop an image first.", history, history
    global _PROGRESS
    _PROGRESS = lambda f, d: progress(f, desc=d)
    try:
        set_offload_mode(offload_mode)
        set_guidance(guidance)
        try:
            rw, rh = [int(x) for x in str(ratio).split(":")]
        except Exception:
            rw, rh = 16, 9
        try:
            res = outpaint(image, rw, rh, prompt, steps, seed)
        except Exception as e:
            _log(f"outpaint error: {e}")
            return [], f"Reframe failed: {e}", history, history
        if save_mode != "display":
            try:
                dst = build_output_path(None, save_mode, output_dir, output_format,
                                        tag="reframe", seed=seed, size=res.size)
                if dst:
                    save_image(res, dst, output_format, meta=_gen_meta(
                        "reframe", prompt, seed=seed, steps=steps, guidance=GUIDANCE,
                        size=res.size, extra={"ratio": ratio}))
            except Exception as e:
                _dbg(f"save reframe failed: {e}")
        new_hist = ([res] + list(history or []))[:200]
        return [res], f"Reframed to {res.size[0]}x{res.size[1]} ({ratio}).", new_hist, new_hist
    finally:
        _PROGRESS = None


def _ui_inpaint(editor_value, prompt, negative, styles, guidance, offload_mode, steps, denoise,
                seed, save_mode, output_dir, output_format, history,
                progress=gr.Progress(track_tqdm=True)):
    """Inpaint depuis l'editeur (image + masque peint) -> galerie + historique."""
    global _PROGRESS
    _PROGRESS = lambda f, d: progress(f, desc=d)
    try:
        bg, mask = _editor_to_image_mask(editor_value)
        if bg is None:
            return [], "Load an image in the Inpaint editor.", history, history
        if mask is None or mask.getbbox() is None:
            return [], "Paint the area to change (the mask is empty).", history, history
        set_offload_mode(offload_mode)
        set_guidance(guidance)
        full_prompt, _ = _apply_styles(prompt, negative, styles)
        try:
            res = inpaint_run(bg, mask, full_prompt, steps, denoise, seed)
        except Exception as e:
            _log(f"inpaint error: {e}")
            return [], f"Inpaint failed: {e}", history, history
        if save_mode != "display":
            try:
                dst = build_output_path(None, save_mode, output_dir, output_format,
                                        tag="inpaint", seed=seed, size=res.size)
                if dst:
                    save_image(res, dst, output_format, meta=_gen_meta(
                        "inpaint", full_prompt, seed=seed, steps=steps, guidance=GUIDANCE,
                        size=res.size, extra={"strength": denoise}))
            except Exception as e:
                _dbg(f"save inpaint failed: {e}")
        new_hist = ([res] + list(history or []))[:200]
        return [res], f"Inpaint -> {res.size[0]}x{res.size[1]}", new_hist, new_hist
    finally:
        _PROGRESS = None


def _ui_clear_history():
    """Vide l'historique de session (state + galerie)."""
    return [], []


def _list_output_files(output_dir, limit=300):
    """Liste les images du dossier de sortie, recursif (sous-dossiers date),
    plus recentes en tete. Ignore _index (artefacts Asset Browser)."""
    d = output_dir or DEFAULT_OUTPUT_DIR
    if not os.path.isabs(d):
        d = os.path.join(HERE, d)
    if not os.path.isdir(d):
        return []
    files = []
    for root, dirs, fs in os.walk(d):
        dirs[:] = [x for x in dirs if x != "_index"]
        for f in fs:
            if f.lower().endswith(IMG_EXTS):
                files.append(os.path.join(root, f))
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[:limit]


def _ui_load_outputs(output_dir):
    """Charge les images du dossier de sortie dans l'historique de session."""
    files = _list_output_files(output_dir, 200)
    return files, files


# ---- Galerie avancee (panneau dedie: meta, suppression, flou) ----
def _gallery_filtered(output_dir, sort="Newest", filt=""):
    files = _list_output_files(output_dir, 4000)
    if filt:
        f = filt.lower()
        files = [p for p in files if f in os.path.basename(p).lower()]
    if sort == "Oldest":
        files.sort(key=os.path.getmtime)
    elif sort == "Name":
        files.sort(key=lambda p: os.path.basename(p).lower())
    else:  # Newest
        files.sort(key=os.path.getmtime, reverse=True)
    return files[:300]


def _gallery_load(output_dir, sort="Newest", filt=""):
    files = _gallery_filtered(output_dir, sort, filt)
    return files, files, f"{len(files)} image(s)."


def _read_image_meta(path):
    """Lit les metadonnees: sidecar '<fichier>.json', sinon chunk PNG 'crispz'."""
    sc = path + ".json"
    if os.path.isfile(sc):
        try:
            with open(sc, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    try:
        with Image.open(path) as im:
            txt = (im.info or {}).get("crispz")          # PNG tEXt
            if txt:
                return json.loads(txt)
            desc = im.getexif().get(0x010E)              # EXIF ImageDescription (jpg/webp)
            if desc:
                return json.loads(desc)
    except Exception:
        pass
    return {}


def _gallery_selected(paths, evt: gr.SelectData):
    """Affiche les metadonnees de l'image selectionnee + son chemin."""
    if not paths or evt is None or evt.index is None or evt.index >= len(paths):
        return "*No selection.*", ""
    path = paths[evt.index]
    info = [f"**File:** `{os.path.basename(path)}`"]
    meta = _read_image_meta(path)
    if meta.get("prompt"):
        info.append(f"**Prompt:** {meta['prompt']}")
    if meta.get("negative"):
        info.append(f"**Negative:** {meta['negative']}")
    line2 = []
    for k in ("mode", "seed", "steps", "guidance", "size"):
        if meta.get(k) is not None:
            line2.append(f"{k}={meta[k]}")
    if meta.get("model"):
        line2.append(f"model={os.path.basename(str(meta['model']))}")
    if meta.get("loras"):
        line2.append("loras=" + ",".join(meta["loras"]))
    if line2:
        info.append("**Params:** " + "  ".join(line2))
    if not meta:  # pas de sidecar -> infos minimales (taille reelle)
        try:
            with Image.open(path) as im:
                info.append(f"**Size:** {im.size[0]}x{im.size[1]}")
        except Exception:
            pass
    try:
        st = os.stat(path)
        info.append(f"**Weight:** {st.st_size / 1024:.0f} KB  ·  "
                    f"**Modified:** {datetime.datetime.fromtimestamp(st.st_mtime):%Y-%m-%d %H:%M}")
    except Exception:
        pass
    info.append(f"**Path:** `{path}`")
    return "  \n".join(info), path


def _gallery_delete(path, output_dir, sort="Newest", filt=""):
    """Supprime le fichier selectionne (+ sidecar) puis recharge la galerie."""
    msg = "Nothing to delete."
    if path and os.path.isfile(path):
        try:
            os.remove(path)
            if os.path.isfile(path + ".json"):   # supprime aussi le sidecar
                os.remove(path + ".json")
            msg = f"Deleted {os.path.basename(path)}."
        except Exception as e:
            msg = f"Delete failed: {e}"
    files = _gallery_filtered(output_dir, sort, filt)
    return files, files, msg, "*Select an image to see its info.*", ""


# ----------------------------------------------------------------------------
# Asset Browser (facon Fooocus2026): SPA statique deposee dans le dossier de
# sortie, ouverte via un lien file=, alimentee par un manifest JSON + thumbnails.
# Memes options (enabled, generate_thumbnails, thumbnail_size/quality, blur).
# ----------------------------------------------------------------------------
_AB_DEFAULTS = {"enabled": False, "generate_thumbnails": True,
                "thumbnail_size": 256, "thumbnail_quality": 85, "blur_thumbnails": False}


def _ab_get(key):
    cfg = CONFIG.get("asset_browser") or {}
    return cfg.get(key, _AB_DEFAULTS.get(key))


def _ab_resolve_dir(output_dir):
    d = output_dir or DEFAULT_OUTPUT_DIR
    return d if os.path.isabs(d) else os.path.join(HERE, d)


def _ab_make_thumb(src, dst, size, quality):
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        side = min(w, h)
        im = im.crop(((w - side) // 2, (h - side) // 2, (w - side) // 2 + side, (h - side) // 2 + side))
        im = im.resize((int(size), int(size)), Image.LANCZOS)
        im.save(dst, "JPEG", quality=int(quality), optimize=True)


ASSET_BROWSER_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>crispz-studio - Asset Browser</title>
<style>
:root{--bg:#0b1018;--panel:#1a2233;--line:#2a3346;--fg:#e6ebf2;--mut:#8b98ad}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font-family:system-ui,Segoe UI,Roboto,sans-serif}
header{position:sticky;top:0;z-index:5;background:#0b1018ee;backdrop-filter:blur(6px);
padding:10px 14px;display:flex;gap:12px;align-items:center;border-bottom:1px solid var(--line);flex-wrap:wrap}
header h1{font-size:15px;margin:0;font-weight:600}
input,button{background:#141b29;color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:7px 10px;font-size:13px}
button{cursor:pointer}#count{color:var(--mut);font-size:12px}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px;padding:12px}
.cell{position:relative;aspect-ratio:1;border-radius:8px;overflow:hidden;background:#11182a;cursor:zoom-in;border:1px solid var(--line)}
.cell img{width:100%;height:100%;object-fit:cover;display:block;transition:.15s}
.cell:hover img{transform:scale(1.04)}
.cell .cap{position:absolute;left:0;right:0;bottom:0;font-size:10px;padding:3px 5px;color:#cfd8e6;
background:linear-gradient(transparent,#000b);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
body.blur .cell img{filter:blur(14px)}body.blur .cell:hover img{filter:none}
#lb{position:fixed;inset:0;background:#000d;z-index:10;display:none;grid-template-columns:1fr 340px}
#lb.open{display:grid}
#lbimg{display:flex;align-items:center;justify-content:center;padding:16px;min-width:0}
#lbimg img{max-width:100%;max-height:96vh;object-fit:contain}
#side{background:var(--panel);border-left:1px solid var(--line);padding:16px;overflow:auto;font-size:13px}
#side h3{margin:.2em 0 .4em;font-size:13px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
#side .v{margin:0 0 10px;word-break:break-word;white-space:pre-wrap}
#side button{margin:4px 6px 4px 0}
.nav{position:fixed;top:50%;transform:translateY(-50%);font-size:40px;color:#fff;cursor:pointer;
user-select:none;padding:0 14px;opacity:.7}.nav:hover{opacity:1}#prev{left:0}#next{right:344px}
#close{position:fixed;top:10px;right:352px;font-size:30px;color:#fff;cursor:pointer;z-index:11}
</style></head><body>
<header><h1>🖼️ crispz-studio</h1>
<input id="q" placeholder="Search prompt / filename..." style="flex:1;min-width:160px">
<button id="blurbtn">Blur</button><span id="count"></span></header>
<div id="grid"></div>
<div id="lb"><span id="close">&times;</span><span class="nav" id="prev">&#10094;</span>
<span class="nav" id="next">&#10095;</span><div id="lbimg"><img id="big"></div>
<div id="side"></div></div>
<script>
let DATA=[],VIEW=[],cur=0;
const grid=document.getElementById('grid'),lb=document.getElementById('lb'),big=document.getElementById('big'),
side=document.getElementById('side'),q=document.getElementById('q'),cnt=document.getElementById('count');
function esc(s){return (s==null?'':String(s)).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function render(){grid.innerHTML='';VIEW.forEach((e,i)=>{const c=document.createElement('div');c.className='cell';
c.innerHTML='<img loading="lazy" src="'+encodeURI(e.thumb)+'" onerror="this.onerror=null;this.src=\''+encodeURI(e.file)+'\'">'+
'<div class="cap">'+esc(e.file)+'</div>';
c.onclick=()=>open(i);grid.appendChild(c);});cnt.textContent=VIEW.length+' / '+DATA.length;}
function filter(){const s=q.value.toLowerCase().trim();VIEW=!s?DATA.slice():DATA.filter(e=>
(e.file+' '+(e.prompt||'')+' '+(e.mode||'')+' '+(e.seed||'')).toLowerCase().includes(s));render();}
function open(i){cur=i;const e=VIEW[i];big.src=encodeURI(e.file);
let h='<h3>Prompt</h3><div class="v">'+esc(e.prompt||'(none)')+'</div>';
if(e.negative)h+='<h3>Negative</h3><div class="v">'+esc(e.negative)+'</div>';
h+='<h3>Info</h3><div class="v">';
['mode','seed','steps','guidance','size','model','date'].forEach(k=>{if(e[k]!=null&&e[k]!=='')h+=k+': '+esc(e[k])+'\n';});
if(e.loras&&e.loras.length)h+='loras: '+esc(e.loras.join(', '))+'\n';
h+='file: '+esc(e.file)+'</div>';
h+='<button onclick="cp(\''+'prompt'+'\')">Copy prompt</button>';
h+='<button onclick="cp(\''+'all'+'\')">Copy all</button>';
h+='<a href="'+encodeURI(e.file)+'" download style="margin-left:6px;color:#9fb3d6">Download</a>';
h+='<button onclick="delAsset()" style="margin-left:6px;background:#5a2230;border-color:#7a2e40">Delete</button>';
side.innerHTML=h;lb.classList.add('open');}
function cp(what){const e=VIEW[cur];let t=e.prompt||'';if(what==='all')t=JSON.stringify(e,null,2);
navigator.clipboard.writeText(t).catch(()=>{});}
async function delAsset(){const e=VIEW[cur];if(!e||!confirm('Delete '+e.file+' ?'))return;
try{const r=await fetch('/gradio_api/call/delete_asset',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({data:[e.file]})});const j=await r.json();const eid=j.event_id||j.hash;
if(eid){await fetch('/gradio_api/call/delete_asset/'+eid);}
DATA=DATA.filter(x=>x.file!==e.file);close();filter();}catch(err){alert('Delete failed: '+err);}}
function close(){lb.classList.remove('open');}
document.getElementById('close').onclick=close;
document.getElementById('prev').onclick=()=>open((cur-1+VIEW.length)%VIEW.length);
document.getElementById('next').onclick=()=>open((cur+1)%VIEW.length);
lb.onclick=ev=>{if(ev.target===lb||ev.target===big.parentNode)close();};
document.addEventListener('keydown',ev=>{if(!lb.classList.contains('open'))return;
if(ev.key==='Escape')close();if(ev.key==='ArrowLeft')document.getElementById('prev').click();
if(ev.key==='ArrowRight')document.getElementById('next').click();});
q.oninput=filter;
document.getElementById('blurbtn').onclick=()=>document.body.classList.toggle('blur');
fetch('_index/manifest.json?t='+Date.now()).then(r=>r.json()).then(m=>{
DATA=m.images||[];if(m.blur)document.body.classList.add('blur');filter();})
.catch(e=>{grid.innerHTML='<p style="padding:20px;color:#8b98ad">No manifest. Click Reindex in crispz-studio.</p>';});
</script></body></html>
"""


def _ab_scan(d):
    """(relpath, fullpath) de toutes les images sous d (recursif), _index ignore.
    Plus recentes en tete."""
    out = []
    for root, dirs, files in os.walk(d):
        dirs[:] = [x for x in dirs if x != "_index"]
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                fp = os.path.join(root, f)
                out.append((os.path.relpath(fp, d).replace("\\", "/"), fp))
    out.sort(key=lambda t: os.path.getmtime(t[1]), reverse=True)
    return out


def _ab_gen_thumbs(jobs, size, quality):
    """Genere une liste de thumbnails (utilise en tache de fond)."""
    for src, tp in jobs:
        try:
            os.makedirs(os.path.dirname(tp), exist_ok=True)
            if not (os.path.isfile(tp) and os.path.getmtime(tp) >= os.path.getmtime(src)):
                _ab_make_thumb(src, tp, size, quality)
        except Exception:
            pass
    _log(f"asset-browser: {len(jobs)} thumbnail(s) generated (background)")


def ab_reindex(output_dir, thumb_size=256, quality=85, blur=False, gen_thumbs=True,
               background_thumbs=False):
    """Ecrit index.html + _index/manifest.json (+ thumbnails). Recursif (sous-dossiers
    date). background_thumbs=True -> ouverture immediate, miniatures en tache de fond
    (l'image complete sert de fallback en attendant)."""
    d = _ab_resolve_dir(output_dir)
    os.makedirs(d, exist_ok=True)
    idx_dir = os.path.join(d, "_index")
    os.makedirs(os.path.join(idx_dir, "thumbs"), exist_ok=True)
    with open(os.path.join(d, "index.html"), "w", encoding="utf-8") as f:
        f.write(ASSET_BROWSER_HTML)
    entries, jobs = [], []
    for rel, p in _ab_scan(d):
        thumb_rel = rel  # fallback = image complete
        trel = "_index/thumbs/" + os.path.splitext(rel)[0] + ".jpg"
        tp = os.path.join(d, trel)
        if os.path.isfile(tp) and os.path.getmtime(tp) >= os.path.getmtime(p):
            thumb_rel = trel
        elif gen_thumbs:
            if background_thumbs:
                jobs.append((p, tp))
            else:
                try:
                    os.makedirs(os.path.dirname(tp), exist_ok=True)
                    _ab_make_thumb(p, tp, thumb_size, quality)
                    thumb_rel = trel
                except Exception as e:
                    _dbg(f"ab thumb failed {rel}: {e}")
        meta = _read_image_meta(p)
        sub = os.path.dirname(rel)
        try:
            date = sub if (len(sub) == 10 and sub[4] == "-") else \
                datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M")
        except Exception:
            date = sub
        entries.append({
            "file": rel, "thumb": thumb_rel, "date": date, "day": sub or "(root)",
            "prompt": meta.get("prompt", ""), "negative": meta.get("negative", ""),
            "seed": meta.get("seed"), "steps": meta.get("steps"),
            "guidance": meta.get("guidance"), "size": meta.get("size"), "mode": meta.get("mode"),
            "model": (os.path.basename(str(meta["model"])) if meta.get("model") else ""),
            "loras": meta.get("loras"),
        })
    manifest = {"count": len(entries), "blur": bool(blur), "thumb_size": int(thumb_size),
                "pending_thumbs": len(jobs),
                "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "images": entries}
    with open(os.path.join(idx_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)
    if jobs and background_thumbs:
        threading.Thread(target=_ab_gen_thumbs, args=(jobs, int(thumb_size), int(quality)),
                         daemon=True).start()
    return len(entries), os.path.join(d, "index.html"), len(jobs)


def _ui_ab_reindex(output_dir, thumb_size, quality, blur, gen_thumbs):
    """Bouton FORCE: regenere TOUTES les miniatures (synchrone) + lien."""
    try:
        n, idx, _ = ab_reindex(output_dir, thumb_size, quality, blur, gen_thumbs,
                               background_thumbs=False)
    except Exception as e:
        return "", f"Asset Browser reindex failed: {e}"
    url = "/gradio_api/file=" + os.path.abspath(idx).replace("\\", "/")
    link = (f'<a href="{url}" target="_blank" style="display:inline-block;padding:8px 14px;'
            f'background:#3b4356;color:#fff;border-radius:6px;text-decoration:none;">'
            f'\U0001F5BC️ Open Asset Browser ({n} images)</a>')
    return link, f"Rebuilt all thumbnails for {n} image(s)."


def _ui_gallery_open(output_dir):
    """Ouverture RAPIDE: manifest immediat, miniatures generees en tache de fond."""
    try:
        n, idx, pending = ab_reindex(output_dir, _ab_get("thumbnail_size"),
                                     _ab_get("thumbnail_quality"), bool(_ab_get("blur_thumbnails")),
                                     bool(_ab_get("generate_thumbnails")), background_thumbs=True)
    except Exception as e:
        return f"Gallery build failed: {e}", ""
    url = "/gradio_api/file=" + os.path.abspath(idx).replace("\\", "/")
    extra = f" Generating {pending} thumbnail(s) in background (full images shown meanwhile)." if pending else ""
    return f"Opening {n} image(s) in a new tab...{extra}", url


def delete_asset(rel, output_dir=None):
    """Supprime une image du dossier de sortie (+ sidecar + thumbnail). 'rel' est le
    chemin relatif fourni par l'Asset Browser. Verifie que ca reste DANS le dossier."""
    d = os.path.abspath(_ab_resolve_dir(output_dir or DEFAULT_OUTPUT_DIR))
    target = os.path.abspath(os.path.join(d, rel or ""))
    if not target.startswith(d + os.sep) or not os.path.isfile(target):
        return "not found"
    try:
        os.remove(target)
        for extra in (target + ".json",
                      os.path.join(d, "_index", "thumbs", os.path.splitext(rel)[0] + ".jpg")):
            if os.path.isfile(extra):
                os.remove(extra)
        _log(f"asset deleted: {rel}")
        return "deleted"
    except Exception as e:
        return f"error: {e}"


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


def _ui_generate(prompt, negative, styles, style_random, use_input, input_image,
                 input_mode, ref1, ref2, ref3, ref4, faceswap_enable, faceswap_src,
                 width, height, gen_steps, image_number, seed, guidance, offload_mode,
                 esrgan_model, do_esrgan, factor, denoise, refine_steps,
                 tile, overlap, refine_tile, refine_overlap,
                 save_mode, output_dir, output_format, history,
                 progress=gr.Progress(track_tqdm=True)):
    """Bouton Generate unifie facon Fooocus. Renvoie 4 sorties:
    (images du run, report, history_state, history_gallery). L'historique accumule
    les rendus de la session (plus recents en tete, cap 200)."""
    global _PROGRESS, _STOP
    _STOP = False
    _PROGRESS = lambda f, d: progress(f, desc=d)
    progress(0.0, desc="Starting...")
    # Les entrees image sont des gr.ImageEditor (crop) -> extraire le PIL recadre.
    input_image = _editor_img(input_image)
    faceswap_src = _editor_img(faceswap_src)
    ref1, ref2, ref3, ref4 = (_editor_img(ref1), _editor_img(ref2),
                              _editor_img(ref3), _editor_img(ref4))

    def _done(imgs, rep):
        # FaceSwap post-process (optionnel, gated). S'applique a tous les modes.
        if faceswap_enable and faceswap_src is not None and imgs:
            try:
                imgs = [_faceswap(im, faceswap_src) for im in imgs]
                rep += " + faceswap"
                if save_mode != "display":
                    for k, im in enumerate(imgs):
                        dst = build_output_path(None, save_mode, output_dir, output_format,
                                                tag="faceswap", seed=seed, size=im.size,
                                                index=(k + 1 if len(imgs) > 1 else 0))
                        if dst:
                            save_image(im, dst, output_format)
            except Exception as e:
                _log(f"faceswap error: {e}")
                rep += f"  \n[faceswap skipped: {e}]"
        new_hist = (list(imgs) + list(history or []))[:200]
        return imgs, rep, new_hist, new_hist

    try:
        set_offload_mode(offload_mode)
        set_guidance(guidance)
        base_prompt = _apply_wildcards(prompt, _seed_rng(seed))   # __name__ -> random line
        full_prompt, full_negative = _apply_styles(base_prompt, negative,
                                                   _pick_styles(styles, style_random))
        mode = "img2img/upscale" if (use_input and input_image is not None) else "txt2img"
        _log(f"Generate ({mode})")
        _dbg(f"params: mode={mode} use_input={use_input} has_img={input_image is not None} "
             f"size={int(width)}x{int(height)} gen_steps={int(gen_steps)} n={int(image_number)} "
             f"seed={int(seed)} guidance={float(guidance)} offload={offload_mode} styles={styles}")
        _dbg(f"prompt='{(full_prompt or '')[:160]}' | negative='{(negative or '')[:80]}'")
        # --- Omni multi-reference (compo a partir de plusieurs images) ---
        # Garde-fou: on ne route en Omni que si un modele Omni est configure. Sinon
        # (UI obsolete dans le navigateur, mode reste sur Omni) on retombe en
        # txt2img/img2img au lieu d'echouer.
        omni_ready = bool((OMNI_MODEL or "").strip())
        if use_input and input_mode == "Reference (Omni)" and omni_ready:
            refs = [r for r in [ref1, ref2, ref3, ref4] if r is not None]
            _dbg(f"omni: {len(refs)} ref(s), size={int(width)}x{int(height)}")
            if not refs:
                return _done([], "Omni: add at least one reference image.")
            try:
                img = generate_omni(refs, full_prompt, full_negative, width, height, gen_steps, seed)
            except Exception as e:
                _log(f"omni error: {e}")
                return _done([], f"Omni error: {e}")
            if save_mode != "display":
                try:
                    dst = build_output_path(None, save_mode, output_dir, output_format,
                                            tag="omni", seed=seed, size=img.size)
                    if dst:
                        save_image(img, dst, output_format, meta=_gen_meta(
                            "omni", full_prompt, full_negative, seed, gen_steps, GUIDANCE,
                            img.size, extra={"refs": len(refs)}))
                        _dbg(f"saved: {dst}")
                except Exception as e:
                    _dbg(f"save failed: {e}")
            return _done([img], f"omni - **{img.size[0]}x{img.size[1]}** from {len(refs)} ref(s)")
        if use_input and input_image is not None:
            _dbg(f"img2img: esrgan={esrgan_model} do_esrgan={do_esrgan} factor={factor} "
                 f"denoise={denoise} refine_steps={int(refine_steps)} tile={int(tile)} "
                 f"refine_tile={int(refine_tile)} model={BASE_REPO} transformer={ZIMAGE_TRANSFORMER}")
            last_result, last_source, report = run(
                input_image, None, esrgan_model, factor, denoise, refine_steps, full_prompt, seed,
                tile, overlap, save_mode=save_mode, output_dir=output_dir,
                output_format=output_format, refine_tile=refine_tile, refine_overlap=refine_overlap,
                do_esrgan=bool(do_esrgan))
            return _done([last_result], report)
        # txt2img (batch image_number)
        n = max(1, int(image_number))
        images, total_t = [], 0.0
        for i in range(n):
            if _STOP:
                _log(f"stop requested after {i}/{n} image(s)")
                break
            s = (int(seed) + i) if int(seed) >= 0 else -1
            progress(i / n, desc=f"Image {i + 1}/{n}")
            # Wildcards (__name__) + style aleatoire, par image (seed -> reproductible)
            chosen = _pick_styles(styles, style_random)
            p_i = _apply_wildcards(prompt, _seed_rng(s))
            fp, fn = _apply_styles(p_i, negative, chosen)
            if style_random:
                _log(f"random style #{i + 1}: {chosen}")
            img, t = txt2img_run(fp, width, height, gen_steps, s, fn,
                                 upscale=False, steps=refine_steps)
            images.append(img)
            total_t += t["txt2img"]
            if save_mode != "display":
                try:
                    dst = build_output_path(None, save_mode, output_dir, output_format,
                                            tag="txt2img", seed=s, size=img.size,
                                            index=(i + 1 if n > 1 else 0))
                    if dst:
                        save_image(img, dst, output_format, meta=_gen_meta(
                            "txt2img", fp, fn, s, gen_steps, GUIDANCE,
                            img.size, extra={"styles": chosen} if chosen else None))
                        _dbg(f"saved: {dst}")
                except Exception as e:
                    _dbg(f"save failed: {e}")
        progress(1.0, desc="Done")
        if not images:
            return _done([], "Stopped before any image.")
        suffix = " (stopped)" if _STOP else ""
        rep = (f"txt2img x{len(images)} - **{images[0].size[0]}x{images[0].size[1]}** "
               f"in **{total_t:.1f}s**{suffix}")
        return _done(images, rep)
    finally:
        _PROGRESS = None


# JS injecte au chargement: force le theme sombre, preview de style au survol,
# et lightbox plein ecran au clic sur le rendu. __MAP__ = {nom_style: url_vignette}.
CZ_JS = """
() => {
  const u = new URL(window.location.href);
  if (u.searchParams.get('__theme') !== 'dark') {
    u.searchParams.set('__theme', 'dark'); window.location.replace(u.toString()); return;
  }
  const SAMPLES = __MAP__;

  // --- Preview de style au survol ---
  let tip = null;
  const ensureTip = () => {
    if (!tip) { tip = document.createElement('div'); tip.className = 'cz-style-preview';
      tip.style.display = 'none'; tip.innerHTML = '<img>'; document.body.appendChild(tip); }
    return tip;
  };
  document.addEventListener('mouseover', (e) => {
    const lbl = e.target.closest && e.target.closest('#cz_styles label');
    if (!lbl) return;
    const name = (lbl.innerText || '').trim();
    const url = SAMPLES[name];
    if (!url) return;
    const t = ensureTip(); const im = t.querySelector('img');
    im.onerror = () => { t.style.display = 'none'; };
    im.src = url; t.style.display = 'block';
  });
  document.addEventListener('mousemove', (e) => {
    if (tip && tip.style.display === 'block') {
      let x = e.clientX + 18, y = e.clientY + 18;
      if (x + 240 > window.innerWidth) x = e.clientX - 240;
      if (y + 240 > window.innerHeight) y = e.clientY - 240;
      tip.style.left = x + 'px'; tip.style.top = y + 'px';
    }
  });
  document.addEventListener('mouseout', (e) => {
    const lbl = e.target.closest && e.target.closest('#cz_styles label');
    if (lbl && tip) tip.style.display = 'none';
  });

  // Le plein ecran + les fleches sont gerees nativement par la galerie Gradio
  // (preview / fullscreen). Pas de lightbox custom (evite le doublon au clic).
}
"""

FOOOCUS_CSS = """
.gradio-container { max-width: 100% !important; width: 100% !important; padding: 0 1rem !important; }
.dark, :root {
  --body-background-fill: #0b1018;
  --background-fill-primary: #0b1018;
  --background-fill-secondary: #11182400;
  --block-background-fill: #1a2233;
  --block-border-color: #2a3346;
  --border-color-primary: #2a3346;
  --input-background-fill: #141b29;
}
/* Rendu homothetique: image entierement visible (contain), centree, jamais plus
   grande que la zone -> pas de scroll, pas de cover. La galerie se dimensionne a
   l'image (plafond 78vh), donc plus de bande vide ni d'image coupee. */
#cz_result { min-height: 60vh !important; }
#cz_result .grid-wrap, #cz_result .grid-container { min-height: 58vh !important; max-height: 82vh !important; }
#cz_result .empty, #cz_result .image-container { min-height: 56vh !important; }
#cz_result img {
  object-fit: contain !important;
  max-width: 100% !important;
  max-height: 78vh !important;
  width: auto !important;
  height: auto !important;
  margin-left: auto !important;
  margin-right: auto !important;
  cursor: zoom-in;
}
#cz_result .thumbnail-item, #cz_result .thumbnail-item img, #cz_result button img {
  object-fit: contain !important; }
#cz_prompt textarea, #cz_neg textarea { font-size: 1.04rem; }
#cz_generate { min-height: 96px !important; height: 100% !important; font-size: 1.12rem; font-weight: 600;
  background: linear-gradient(180deg,#5a6376,#3b4356) !important; color: #fff !important;
  border: 1px solid #5d6884 !important; box-shadow: none !important; }
#cz_generate:hover { background: linear-gradient(180deg,#69738a,#454e63) !important; }
/* Bloc styles: scroller interne */
#cz_styles { max-height: 340px; overflow-y: auto; padding-right: 6px; }
/* Preview de style au survol */
.cz-style-preview { position: fixed; z-index: 10000; pointer-events: none;
  border: 1px solid #2a3346; border-radius: 8px; overflow: hidden;
  box-shadow: 0 6px 24px rgba(0,0,0,.6); background: #0b1018; }
.cz-style-preview img { display: block; width: 110px; height: auto; }
/* Galerie avancee: flou NSFW optionnel */
#cz_gallery.cz-blur img { filter: blur(18px); transition: filter .15s; }
#cz_gallery.cz-blur img:hover { filter: none; }
/* Lightbox plein ecran */
.cz-lightbox { position: fixed; inset: 0; background: rgba(0,0,0,.93); z-index: 10001;
  display: flex; align-items: center; justify-content: center; cursor: zoom-out; }
.cz-lightbox img { max-width: 95vw; max-height: 95vh; object-fit: contain; }
.cz-lightbox .cz-close { position: fixed; top: 14px; right: 26px; color: #fff;
  font-size: 44px; line-height: 1; cursor: pointer; font-weight: 300; }
"""


def build_ui():
    models = list_esrgan_models()
    default_model = DEFAULT_MODEL if DEFAULT_MODEL in models else (models[0] if models else None)

    # Map nom_style -> URL de vignette (servie par Gradio) pour le preview au survol.
    _sample_urls = {}
    for n in STYLES:
        p = _style_sample(n)
        if p:
            _sample_urls[n] = "/gradio_api/file=" + os.path.abspath(p).replace("\\", "/")
    js_full = CZ_JS.replace("__MAP__", json.dumps(_sample_urls))
    # Omni (multi-reference) propose seulement si un modele Omni/Edit est configure.
    omni_on = bool((OMNI_MODEL or "").strip())

    with gr.Blocks(title="crispz-studio", theme=gr.themes.Default(), css=FOOOCUS_CSS, js=js_full) as demo:
        # La galerie du dossier de sortie s'ouvre dans un nouvel onglet (Asset Browser),
        # via le bouton sous l'apercu. Pas de panneau galerie inline.
        gallery_url = gr.Textbox(visible=False)
        # Endpoint API (appele par l'Asset Browser pour supprimer une image)
        del_in = gr.Textbox(visible=False)
        del_out = gr.Textbox(visible=False)
        del_btn = gr.Button(visible=False)
        del_btn.click(delete_asset, del_in, del_out, api_name="delete_asset")

        with gr.Row():
            # ===== Colonne principale (apercu en haut, prompt + Generate, negative, input) =====
            with gr.Column(scale=3):
                out = gr.Gallery(label="Result", elem_id="cz_result", columns=2,
                                 object_fit="contain", preview=True, allow_preview=True,
                                 show_fullscreen_button=True, show_download_button=True)
                report = gr.Markdown(value="*Ready. Type a prompt and press Generate.*")

                history = gr.State([])
                with gr.Accordion("History (this session)", open=False):
                    gr.Markdown("*Renders made in this session (in-memory, not the disk folder). "
                                "Use 'Load output folder' to pull saved files in.*")
                    history_gallery = gr.Gallery(label=None, height=240, columns=6,
                                                 object_fit="cover", show_download_button=True)
                    with gr.Row():
                        load_out_btn = gr.Button("Load output folder", size="sm")
                        clear_hist_btn = gr.Button("Clear history", size="sm")
                        gallery_btn = gr.Button("\U0001F5BC️ Asset Browser", size="sm",
                                                variant="primary")
                gallery_status = gr.Markdown("")

                with gr.Row():
                    prompt = gr.Textbox(show_label=False, value=CONFIG.get("default_prompt", ""),
                                        placeholder="Type your prompt here...",
                                        elem_id="cz_prompt", lines=2, scale=4, container=False)
                    with gr.Column(scale=1, min_width=150):
                        btn = gr.Button("Generate", elem_id="cz_generate", min_width=150)
                        stop_btn = gr.Button("Stop", variant="stop", size="sm", min_width=150)

                negative = gr.Textbox(show_label=False, value=CONFIG.get("default_negative_prompt", ""),
                                      elem_id="cz_neg", lines=1, container=False,
                                      placeholder="Negative prompt - what you do NOT want (needs guidance > 0)")

                with gr.Row():
                    improve_btn = gr.Button("Improve prompt (Ollama)", size="sm", scale=0, min_width=200)
                    improve_status = gr.Markdown("")

                with gr.Row():
                    use_input = gr.Checkbox(value=False, label="Input Image", min_width=160)
                    advanced_cb = gr.Checkbox(value=False, label="Advanced", min_width=160)

                with gr.Group(visible=False) as input_group:
                    input_mode = gr.Radio(
                        ["Upscale / img2img"] + (["Reference (Omni)"] if omni_on else []),
                        value="Upscale / img2img", label="Input mode", visible=omni_on,
                        info="Reference (Omni) = compose from several reference images + prompt.")
                    with gr.Tabs():
                        with gr.Tab("Upscale or img2img"):
                            with gr.Row():
                                inp = _crop_input("Drop image here / click to upload", 300)
                                with gr.Column():
                                    do_esrgan_cb = gr.Checkbox(value=True, label="ESRGAN upscale",
                                                               info="Uncheck = img2img only (no enlargement).")
                                    preset = gr.Dropdown(list(PRESETS), value="Custom", label="Use case preset")
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

                        with gr.Tab("Describe"):
                            describe_img = _crop_input("Image to describe", 280)
                            describe_btn = gr.Button("Describe -> prompt", variant="primary", size="sm")
                            describe_status = gr.Markdown(
                                "*Uses the Ollama vision model selected in Advanced > Prompt AI "
                                "(or the local BLIP fallback if Ollama is off).*")

                        with gr.Tab("Vision Mix"):
                            gr.Markdown("*Vision Mix: a vision model looks at your reference images "
                                        "and an LLM blends them into ONE text prompt (e.g. a person + "
                                        "an outfit + a setting), then Z-Image generates from it. "
                                        "Needs Ollama with a vision model (Advanced > Prompt AI). "
                                        "It mixes ideas/style, not exact pixels.*")
                            with gr.Row():
                                cref1 = _crop_input("Ref 1", 180)
                                cref2 = _crop_input("Ref 2", 180)
                            with gr.Row():
                                cref3 = _crop_input("Ref 3", 180)
                                cref4 = _crop_input("Ref 4", 180)
                            with gr.Row():
                                compose_btn = gr.Button("Vision Mix -> prompt", size="sm")
                                vmix_gen_btn = gr.Button("Vision Mix & Generate", variant="primary",
                                                         size="sm")
                            compose_status = gr.Markdown("")

                        with gr.Tab("Remove BG"):
                            rembg_img = _crop_input("Image", 280)
                            rembg_btn = gr.Button("Remove background", variant="primary", size="sm")
                            rembg_status = gr.Markdown("*Local (rembg). Output = transparent PNG. "
                                                       "First use downloads the u2net model.*")

                        with gr.Tab("Reframe (outpaint)"):
                            gr.Markdown("*Expand the image to a new aspect ratio; Z-Image fills "
                                        "the new borders (inpaint). The prompt guides the fill.*")
                            reframe_img = _crop_input("Image", 260)
                            with gr.Row():
                                reframe_ratio = gr.Dropdown(
                                    ["16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "1:1", "21:9"],
                                    value="16:9", label="Target ratio")
                                reframe_steps = gr.Slider(4, 30, value=12, step=1, label="Fill steps")
                            reframe_btn = gr.Button("Reframe / Outpaint", variant="primary", size="sm")
                            reframe_status = gr.Markdown("")

                        with gr.Tab("Inpaint"):
                            gr.Markdown("*Paint the area to change (white brush), describe what "
                                        "should appear in the prompt, then Inpaint.*")
                            inpaint_editor = gr.ImageEditor(
                                type="pil", label="Image + mask (paint the area)",
                                brush=gr.Brush(colors=["#ffffff"], color_mode="fixed"),
                                layers=False, transforms=[], height=380)
                            with gr.Row():
                                inpaint_steps = gr.Slider(4, 40, value=20, step=1, label="Steps")
                                inpaint_denoise = gr.Slider(0.3, 1.0, value=0.85, step=0.05,
                                                            label="Inpaint strength")
                            inpaint_btn = gr.Button("Inpaint", variant="primary", size="sm")
                            inpaint_status = gr.Markdown("")

                        with gr.Tab("Reference (Omni)", visible=omni_on):
                            gr.Markdown("*Compose from up to 4 reference images + a prompt. "
                                        "Set **Input mode = Reference (Omni)** above. "
                                        "Uses width/height/steps/guidance from Settings.*")
                            with gr.Row():
                                omni_check_btn2 = gr.Button("Check Omni model availability", size="sm")
                                omni_status2 = gr.Markdown("")
                            with gr.Row():
                                ref1 = _crop_input("Ref 1", 220)
                                ref2 = _crop_input("Ref 2", 220)
                            with gr.Row():
                                ref3 = _crop_input("Ref 3", 220)
                                ref4 = _crop_input("Ref 4", 220)

                        with gr.Tab("Face Swap"):
                            gr.Markdown("*Post-process: replace the face in the result with this "
                                        "source face. Works on any mode (txt2img / img2img / omni). "
                                        "Needs `insightface` + `onnxruntime-gpu` installed and "
                                        "`faceswap_model_path` (inswapper .onnx) set in config.txt.*")
                            faceswap_src = _crop_input("Source face", 240)
                            faceswap_enable = gr.Checkbox(value=False, label="Apply face swap to result")
                            faceswap_restore_cb = gr.Checkbox(
                                value=FACESWAP_RESTORE,
                                label="Restore face (GFPGAN) - fixes the soft 128px swap",
                                info="Sharpens the swapped face. Downloads gfpgan_1.4.onnx on first "
                                     "use (faceswap_restore_url in config.txt).")
                            faceswap_restore_blend = gr.Slider(0.0, 1.0, value=float(FACESWAP_RESTORE_BLEND),
                                                               step=0.05, label="Restore strength")
                            faceswap_restore_status = gr.Markdown("")

            # ===== Colonne Advanced (a droite, masquee par defaut comme Fooocus) =====
            with gr.Column(scale=2, visible=False) as advanced_col:
                with gr.Tabs():
                    with gr.Tab("Settings"):
                        performance = gr.Radio(list(PERFORMANCE),
                                               value=CONFIG.get("default_performance", "Turbo (8 steps)"),
                                               label="Performance",
                                               info="Sets steps + guidance. Turbo = your Turbo model; "
                                                    "Base CFG = for a Z-Image Base checkpoint.")
                        aspect = gr.Dropdown(list(ASPECT_RATIOS),
                                             value=CONFIG.get("default_aspect_ratio", "1024 x 1024  (1:1)"),
                                             label="Aspect ratio")
                        with gr.Row():
                            width = gr.Slider(256, 2048, value=int(CONFIG.get("default_width", 1024)),
                                              step=16, label="Width")
                            height = gr.Slider(256, 2048, value=int(CONFIG.get("default_height", 1024)),
                                               step=16, label="Height")
                        gen_steps = gr.Slider(2, 40, value=int(CONFIG.get("default_gen_steps", 8)),
                                              step=1, label="Generation steps (txt2img)")
                        guidance = gr.Slider(0.0, 8.0, value=float(CONFIG.get("default_guidance", 0.0)),
                                             step=0.5, label="CFG guidance",
                                             info="0 = Z-Image Turbo. Z-Image Base: ~3.5-5.")
                        image_number = gr.Slider(1, 30, value=int(CONFIG.get("default_image_number", 1)),
                                                 step=1, label="Image number (batch)")
                        seed = gr.Number(value=int(CONFIG.get("default_seed", -1)),
                                         label="Seed (-1 = random)", precision=0)

                    with gr.Tab("Styles"):
                        style_search = gr.Textbox(show_label=False, container=False,
                                                  placeholder="Search styles... (e.g. anime, cinematic, sai)")
                        style_random = gr.Checkbox(
                            value=False, label="Random style each image",
                            info="Each render picks a random style from the selected ones "
                                 "(or from ALL styles if none selected).")
                        gr.Markdown("*Hover a style to preview it.*")
                        styles = gr.CheckboxGroup(list(STYLES), value=CONFIG.get("default_styles", []),
                                                  label="Styles (combinable)", elem_id="cz_styles")

                    with gr.Tab("Prompt AI"):
                        ollama_url = gr.Textbox(value=OLLAMA_URL, label="Ollama URL",
                                                info="Local LLM server. Used for Describe (vision) "
                                                     "and Improve prompt.")
                        detect_btn = gr.Button("Detect Ollama (vision models)", size="sm", variant="primary")
                        ollama_model = gr.Dropdown([], label="Vision model (Describe / Improve)",
                                                   interactive=True)
                        ollama_status = gr.Markdown("*Click Detect. If Ollama is off, Describe falls "
                                                    "back to a local BLIP captioner.*")
                        gr.Markdown("---")
                        log_level_dd = gr.Dropdown(["quiet", "info", "debug"],
                                                   value={0: "quiet", 1: "info", 2: "debug"}.get(LOG_LEVEL, "info"),
                                                   label="Console log level (dev)",
                                                   info="debug = full params, pipe state, VRAM in the .bat console.")
                        log_level_status = gr.Markdown("")

                    with gr.Tab("Models"):
                        zimage_model_tb = gr.Textbox(
                            value=BASE_REPO,
                            label="Z-Image base (HF repo, diffusers folder, or .safetensors file)",
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

                        gr.Markdown("### Checkpoints (switch model, like ESRGAN)")
                        ckpt_dir_tb = gr.Textbox(value=CHECKPOINTS_DIR, label="Checkpoints folder")
                        with gr.Row():
                            ckpt_dd = gr.Dropdown(choices=["(base repo)"] + list_checkpoints(),
                                                  value="(base repo)", label="Z-Image checkpoint", scale=3)
                            ckpt_refresh_btn = gr.Button("Refresh", size="sm", scale=1)
                        ckpt_status = gr.Markdown("")
                        with gr.Row():
                            transformer_tb = gr.Textbox(
                                value="", scale=3,
                                label="Transformer override (HF repo / diffusers folder)",
                                placeholder="e.g. RunDiffusion/Juggernaut-Z-Image",
                                info="For community models with an incomplete tokenizer (Juggernaut-Z): "
                                     "loads only the transformer, keeps base VAE/encoder. Set base = Turbo.")
                            transformer_apply_btn = gr.Button("Apply", size="sm", scale=1, variant="primary")

                        gr.Markdown("### LoRA (up to 3, combinable)")
                        lora_dir_tb = gr.Textbox(value=LORAS_DIR, label="LoRA folder")
                        _lchoices = ["None"] + list_loras()
                        with gr.Row():
                            lora_dd1 = gr.Dropdown(choices=_lchoices, value="None", label="LoRA 1", scale=3)
                            lw1 = gr.Slider(0.0, 2.0, value=float(LORA_WEIGHT), step=0.05,
                                            label="Weight 1", scale=2)
                        with gr.Row():
                            lora_dd2 = gr.Dropdown(choices=_lchoices, value="None", label="LoRA 2", scale=3)
                            lw2 = gr.Slider(0.0, 2.0, value=float(LORA_WEIGHT), step=0.05,
                                            label="Weight 2", scale=2)
                        with gr.Row():
                            lora_dd3 = gr.Dropdown(choices=_lchoices, value="None", label="LoRA 3", scale=3)
                            lw3 = gr.Slider(0.0, 2.0, value=float(LORA_WEIGHT), step=0.05,
                                            label="Weight 3", scale=2)
                        lora_refresh_btn = gr.Button("Refresh LoRA list", size="sm")
                        lora_keywords_tb = gr.Textbox(label="Keywords / trigger words", lines=2,
                                                      placeholder="Auto-filled from the selected LoRA(s).")
                        with gr.Row():
                            lora_kw_btn = gr.Button("Get keywords", size="sm")
                            lora_kw_to_prompt_btn = gr.Button("Add to prompt", size="sm", variant="primary")
                        lora_status = gr.Markdown("")

                        gr.Markdown("### \U0001F3B2 Wildcards")
                        gr.Markdown("*`__name__` in the prompt -> a random line from name.txt "
                                    "(nested, reproducible per seed). Pick a file to view/edit it, "
                                    "Insert to add it to the prompt, or Create a new one.*")
                        wild_dir_tb = gr.Textbox(value=WILDCARDS_DIR, label="Wildcards folder")
                        with gr.Row():
                            wild_dd = gr.Dropdown(["None"] + list_wildcards(), value="None",
                                                  label="Wildcard file", scale=3)
                            wild_refresh_btn = gr.Button("Refresh", size="sm", scale=1, min_width=90)
                            wild_insert_btn = gr.Button("Insert __name__", size="sm", scale=1,
                                                        variant="primary", min_width=140)
                        wild_editor = gr.Textbox(label="Contents (one option per line)", lines=8,
                                                 placeholder="Select a file above to view/edit, "
                                                             "or type lines for a new one.")
                        with gr.Row():
                            wild_save_btn = gr.Button("Save", size="sm")
                            wild_new_name = gr.Textbox(show_label=False, scale=2, container=False,
                                                       placeholder="new_wildcard_name (no extension)")
                            wild_create_btn = gr.Button("Create new", size="sm", variant="primary")
                        wild_status = gr.Markdown("")

                        gr.Markdown("### Omni / Edit model (multi-reference)")
                        gr.Markdown("*The Reference (Omni) tab stays hidden until a model is set "
                                    "here (then restart). Z-Image-Omni-Base / Z-Image-Edit are not "
                                    "released yet. For a reference image now, use img2img.*")
                        omni_model_tb = gr.Textbox(value=CONFIG.get("zimage_omni_model", ""),
                                                   label="Omni model (HF repo or local folder)",
                                                   info="Needs SigLIP. Set it then restart to enable "
                                                        "the Reference (Omni) tab.")
                        omni_check_btn = gr.Button("Check Omni availability (Hugging Face)", size="sm")
                        omni_status = gr.Markdown("")

                        gr.Markdown("### \U0001F5BC️ Asset Browser")
                        gr.Markdown("*Standalone gallery page (thumbnails + metadata + lightbox) "
                                    "built into the output folder, opened in a new tab. Reindex to "
                                    "(re)build it from your saved images.*")
                        with gr.Row():
                            ab_thumb_size = gr.Slider(96, 512, value=int(_ab_get("thumbnail_size")),
                                                      step=32, label="Thumbnail size")
                            ab_quality = gr.Slider(40, 100, value=int(_ab_get("thumbnail_quality")),
                                                   step=5, label="Thumbnail quality")
                        with gr.Row():
                            ab_blur = gr.Checkbox(value=bool(_ab_get("blur_thumbnails")),
                                                  label="Blur thumbnails (NSFW)")
                            ab_gen_thumbs = gr.Checkbox(value=bool(_ab_get("generate_thumbnails")),
                                                        label="Generate thumbnails")
                        ab_reindex_btn = gr.Button("Rebuild ALL thumbnails (force) + open link",
                                                   variant="primary", size="sm")
                        ab_open_link = gr.HTML("")
                        ab_status = gr.Markdown("")

                    with gr.Tab("Save"):
                        save_mode = gr.Radio(choices=["display", "local", "alongside", "custom"],
                                             value=DEFAULT_SAVE_MODE, label="Save mode")
                        output_dir = gr.Textbox(value=DEFAULT_OUTPUT_DIR, label="Output folder")
                        output_format = gr.Dropdown(choices=list(SUPPORTED_FORMATS),
                                                    value=DEFAULT_OUTPUT_FORMAT, label="Output format")

        # Toggles facon Fooocus
        advanced_cb.change(lambda v: gr.update(visible=bool(v)), advanced_cb, advanced_col)
        use_input.change(lambda v: gr.update(visible=bool(v)), use_input, input_group)
        aspect.change(_set_aspect, [aspect], [width, height])
        performance.change(_set_performance, [performance], [gen_steps, guidance])
        style_search.change(_filter_styles, [style_search, styles], [styles])

        # Actions
        refresh_btn.click(_refresh_models, [esrgan_dir_tb], [esrgan, paths_status])
        apply_zimage_btn.click(_apply_zimage, [zimage_model_tb], [paths_status])
        save_paths_btn.click(_save_paths_to_prefs,
                             [esrgan_dir_tb, zimage_model_tb, ckpt_dir_tb, lora_dir_tb, wild_dir_tb],
                             [paths_status])
        wild_refresh_btn.click(_ui_wild_refresh, [wild_dir_tb], [wild_dd, wild_status])
        wild_dd.change(_ui_wild_load, [wild_dd], [wild_editor, wild_status])
        wild_insert_btn.click(_ui_wild_insert, [wild_dd, prompt], [prompt, wild_status])
        wild_save_btn.click(_ui_wild_save, [wild_dd, wild_editor], [wild_status])
        wild_create_btn.click(_ui_wild_create, [wild_new_name, wild_editor],
                              [wild_dd, wild_status, wild_new_name])
        ckpt_refresh_btn.click(_refresh_checkpoints, [ckpt_dir_tb], [ckpt_dd, ckpt_status])
        ckpt_dd.change(_apply_checkpoint, [ckpt_dd], [ckpt_status, gen_steps, guidance])
        transformer_apply_btn.click(_apply_transformer_repo, [transformer_tb],
                                    [ckpt_status, gen_steps, guidance])
        lora_refresh_btn.click(_refresh_loras, [lora_dir_tb],
                               [lora_dd1, lora_dd2, lora_dd3, lora_status])
        _lora_slots = [lora_dd1, lw1, lora_dd2, lw2, lora_dd3, lw3]
        for _c in (lora_dd1, lora_dd2, lora_dd3):
            _c.change(_ui_loras_apply, _lora_slots, [lora_status, lora_keywords_tb])
        for _c in (lw1, lw2, lw3):
            _c.change(_apply_loras, _lora_slots, [lora_status])
        lora_kw_btn.click(_ui_loras_keywords, [lora_dd1, lora_dd2, lora_dd3],
                          [lora_keywords_tb, lora_status])
        lora_kw_to_prompt_btn.click(_ui_kw_to_prompt, [prompt, lora_keywords_tb], [prompt])
        omni_model_tb.change(lambda r: (set_omni_model(r), f"Omni model set: {r or '(none)'}")[1],
                             [omni_model_tb], [omni_status])
        omni_check_btn.click(_ui_check_omni, None, [omni_status])
        omni_check_btn2.click(_ui_check_omni, None, [omni_status2])
        ab_reindex_btn.click(_ui_ab_reindex,
                             [output_dir, ab_thumb_size, ab_quality, ab_blur, ab_gen_thumbs],
                             [ab_open_link, ab_status])
        log_level_dd.change(set_log_level, [log_level_dd], [log_level_status])
        detect_btn.click(_ui_detect_ollama, [ollama_url], [ollama_model, ollama_status])
        describe_btn.click(_ui_describe, [describe_img, ollama_model, ollama_url], [prompt, describe_status])
        improve_btn.click(_ui_improve, [prompt, ollama_model, ollama_url], [prompt, improve_status])
        compose_btn.click(_ui_compose, [cref1, cref2, cref3, cref4, ollama_model, ollama_url],
                          [prompt, compose_status])
        rembg_btn.click(_ui_remove_bg, [rembg_img, history, save_mode, output_dir],
                        [out, report, history, history_gallery])
        faceswap_restore_cb.change(set_faceswap_restore,
                                   [faceswap_restore_cb, faceswap_restore_blend],
                                   [faceswap_restore_status])
        faceswap_restore_blend.change(set_faceswap_restore,
                                      [faceswap_restore_cb, faceswap_restore_blend],
                                      [faceswap_restore_status])
        reframe_btn.click(_ui_reframe,
                          [reframe_img, reframe_ratio, reframe_steps, prompt, guidance, offload,
                           seed, save_mode, output_dir, output_format, history],
                          [out, report, history, history_gallery])
        inpaint_btn.click(_ui_inpaint,
                          [inpaint_editor, prompt, negative, styles, guidance, offload,
                           inpaint_steps, inpaint_denoise, seed, save_mode, output_dir,
                           output_format, history],
                          [out, report, history, history_gallery])
        # Stop facon Fooocus: tourne en parallele du Generate (thread separe) et pose
        # le flag d'arret + interrompt la boucle de debruitage en cours.
        stop_btn.click(request_stop, None, [report])
        clear_hist_btn.click(_ui_clear_history, None, [history, history_gallery])
        load_out_btn.click(_ui_load_outputs, [output_dir], [history, history_gallery])
        # Galerie du dossier de sortie -> Asset Browser dans un nouvel onglet
        gallery_btn.click(_ui_gallery_open, [output_dir], [gallery_status, gallery_url]).then(
            None, [gallery_url], None, js="(u) => { if (u) window.open(u, '_blank'); }")
        preset.change(_apply_preset, [preset],
                      [factor, denoise, refine_steps, tile, overlap, refine_tile, refine_overlap, offload])
        _gen_inputs = [prompt, negative, styles, style_random, use_input, inp, input_mode,
                       ref1, ref2, ref3, ref4,
                       faceswap_enable, faceswap_src,
                       width, height, gen_steps, image_number,
                       seed, guidance, offload, esrgan, do_esrgan_cb, factor, denoise, refine_steps,
                       tile, overlap, refine_tile, refine_overlap, save_mode, output_dir, output_format,
                       history]
        _gen_outputs = [out, report, history, history_gallery]
        btn.click(_ui_generate, inputs=_gen_inputs, outputs=_gen_outputs)
        # Vision Mix & Generate: fusionne les refs en un prompt, puis genere (txt2img).
        vmix_gen_btn.click(
            _ui_compose, [cref1, cref2, cref3, cref4, ollama_model, ollama_url],
            [prompt, compose_status]
        ).then(_ui_generate, inputs=_gen_inputs, outputs=_gen_outputs)
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
            _srcbase = os.path.splitext(os.path.basename(req.input))[0]
            dst = build_output_path(req.input, req.save_mode, req.output_dir, req.output_format,
                                    tag=f"{_srcbase}_upscaled", seed=req.seed, size=result.size)
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
    # Nouvelles features en CLI (mêmes que l'UI)
    parser.add_argument("--lora", action="append", default=[], metavar="NAME[:WEIGHT]",
                        help="Apply a LoRA (file in the loras dir, or a path), optional :weight "
                             "(default 1.0). Repeatable, e.g. --lora a.safetensors:0.8 --lora b")
    parser.add_argument("--loras-dir", help="Override the LoRA folder (for --lora names)")
    parser.add_argument("--remove-bg", action="store_true",
                        help="Remove the background of -i (rembg) -> transparent PNG, then exit.")
    parser.add_argument("--reframe", metavar="W:H",
                        help="Outpaint -i to a target aspect ratio (e.g. 16:9 / 9:16), then exit. "
                             "--prompt guides the fill; --gen-steps sets the steps.")
    parser.add_argument("--faceswap-src", metavar="PATH",
                        help="Post-process: swap the face in the (txt2img/reframe) result with this "
                             "source face. Needs insightface + an inswapper model.")
    parser.add_argument("--vision-mix", nargs="+", metavar="IMG",
                        help="Describe these reference images with an Ollama vision model and merge "
                             "them into ONE prompt, then run txt2img from it.")
    parser.add_argument("--ollama-model", help="Ollama vision model for --vision-mix "
                                               "(default: first detected vision model)")
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
    parser.add_argument("--log-level", choices=["quiet", "info", "debug"], default=None,
                        help="Console log level on stderr. debug = full params/state (dev). "
                             "Default from env CRISPZ_LOG_LEVEL or 'info'.")
    parser.add_argument("--report-vram", action="store_true",
                        help="Print the run VRAM peak on stderr (line '[VRAM] ...'). "
                             "Used to size coexistence with Fooocus.")
    parser.add_argument("--print-output", action="store_true",
                        help="Print ONLY the absolute output path on stdout (one per saved "
                             "image), nothing else. For external integration (Fooocus). "
                             "Implies a silent stdout; the VRAM peak stays on stderr.")
    args = parser.parse_args(argv)
    apply_preset_to_args(args, argv if argv is not None else sys.argv[1:])

    global LOG_LEVEL
    if args.log_level:
        LOG_LEVEL = _parse_log_level(args.log_level)
    elif args.quiet:
        LOG_LEVEL = 0
    _log(f"log level = {LOG_LEVEL} (0=quiet 1=info 2=debug)")

    if args.esrgan_dir:
        set_esrgan_dir(args.esrgan_dir)
    if args.zimage_model:
        set_zimage_model(args.zimage_model)
    if args.zimage_transformer:
        set_zimage_transformer(args.zimage_transformer)
    set_offload_mode(args.cpu_offload)
    set_guidance(args.guidance)

    # LoRA(s) en CLI: --lora NAME[:WEIGHT] (repetable)
    if args.loras_dir:
        set_loras_dir(args.loras_dir)
    if args.lora:
        slots = []
        for spec in args.lora:
            head, _, tail = spec.rpartition(":")
            try:
                slots.append((head, float(tail)))   # NAME:WEIGHT
            except ValueError:
                slots.append((spec, LORA_WEIGHT))    # NAME (poids par defaut)
        set_loras(slots)

    def _maybe_faceswap(img):
        if args.faceswap_src and os.path.isfile(args.faceswap_src):
            try:
                return _faceswap(img, Image.open(args.faceswap_src))
            except Exception as e:
                _log(f"faceswap skipped: {e}")
        return img

    # --remove-bg : detoure -i puis termine
    if args.remove_bg:
        if not args.input or not os.path.isfile(args.input):
            parser.error("--remove-bg requires -i <image>")
        res = _remove_bg(Image.open(args.input))
        sm = args.save_mode if args.save_mode != "display" else "local"
        base = os.path.splitext(os.path.basename(args.input))[0]
        dst = args.output if (args.output and not os.path.isdir(args.output)) else \
            build_output_path(args.input, sm, args.output_dir, "png", tag=f"{base}_nobg")
        save_image(res, dst, "png")
        print(os.path.abspath(dst))
        return 0

    # --reframe W:H : outpaint -i puis termine
    if args.reframe:
        if not args.input or not os.path.isfile(args.input):
            parser.error("--reframe requires -i <image>")
        try:
            rw, rh = [int(x) for x in str(args.reframe).split(":")]
        except Exception:
            parser.error("--reframe expects W:H, e.g. 16:9")
        res = _maybe_faceswap(outpaint(Image.open(args.input), rw, rh, args.prompt,
                                       args.gen_steps, args.seed))
        sm = args.save_mode if args.save_mode != "display" else "local"
        base = os.path.splitext(os.path.basename(args.input))[0]
        dst = args.output if (args.output and not os.path.isdir(args.output)) else \
            build_output_path(args.input, sm, args.output_dir, args.output_format,
                              tag=f"{base}_reframe", seed=args.seed, size=res.size)
        save_image(res, dst, args.output_format)
        print(os.path.abspath(dst))
        return 0

    # --vision-mix IMG... : decrit + fusionne en un prompt, puis txt2img
    if args.vision_mix:
        imgs = [Image.open(p) for p in args.vision_mix if os.path.isfile(p)]
        if not imgs:
            parser.error("--vision-mix: no valid image path")
        vmodel = args.ollama_model or (_ollama_vision_models() or [None])[0]
        if not vmodel:
            parser.error("--vision-mix needs an Ollama vision model (none detected)")
        caps = [_ollama_describe(im, vmodel) for im in imgs]
        args.prompt = _ollama_compose(caps, vmodel)
        args.txt2img = True
        if not (args.quiet or args.print_output):
            print(f"[vision-mix] {vmodel} -> {args.prompt}", file=sys.stderr)

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
        result = _maybe_faceswap(result)
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
        _disable_brotli()  # evite le bug h11 'Content-Length' a l'envoi des resultats
        build_ui().launch(allowed_paths=[os.path.join(HERE, "styles", "samples"),
                                         _ab_resolve_dir(DEFAULT_OUTPUT_DIR)])
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
