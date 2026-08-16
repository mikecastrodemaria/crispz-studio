"""crispz-studio - auto face detailer (facon ADetailer / Fooocus "Enhance").

Apres un rendu, detecte les visages (insightface buffalo_l, deja charge pour le Face
Swap) et repasse CHAQUE visage en img2img a haute resolution :
  crop elargi (+60%) -> agrandi au sweet spot du modele (~832 px) -> refine Z-Image
  (denoise modere, meme seed/prompt) -> reduit -> recolle avec un masque elliptique
  feather (comme le collage GFPGAN du Face Swap: pas de bord carre).

Active par la case "🔧 Detail faces" sous le bouton Generate (flag module, pas de
nouvel input dans _gen_inputs -> la queue et la grille X/Y/Z ne bougent pas), ou par
config 'face_detailer'. Reglages: 'face_detailer_denoise' (0.35), 'face_detailer_max_faces'.
"""

import numpy as np
from PIL import Image

from cz_core import CONFIG, _log, _dbg

DETAILER_ENABLED = bool(CONFIG.get("face_detailer", False))
DETAILER_DENOISE = float(CONFIG.get("face_detailer_denoise", 0.35))
_MAX_FACES = max(1, int(CONFIG.get("face_detailer_max_faces", 4)))
_TARGET = 832      # cote de travail du crop (sweet spot Z-Image, /32)
_MARGIN = 0.6      # expansion de la bbox visage (contexte: cheveux, cou)
_MIN_FACE = 28     # px: en-dessous, trop petit pour gagner quoi que ce soit

# --- Detailer MAINS (meme mecanique, autre detecteur) -------------------------
# Les mains sont le point faible de tous les modeles de diffusion. Meme circuit que
# les visages: crop elargi -> refine haute-res -> recollage feather. Le detecteur est
# un YOLOv8 mains (ultralytics), dependance OPTIONNELLE: absente -> la feature se
# desactive avec un message clair, le reste de l'app est intact.
HAND_ENABLED = bool(CONFIG.get("hand_detailer", False))
HAND_DENOISE = float(CONFIG.get("hand_detailer_denoise", 0.4))
_MAX_HANDS = max(1, int(CONFIG.get("hand_detailer_max_hands", 4)))
_MIN_HAND = 24
_HAND_MARGIN = float(CONFIG.get("hand_detailer_margin", 0.35))
# depot HF du modele (Bingsu/adetailer): 'hand_yolov8n.pt' (6 Mo, rapide) ou
# 'hand_yolov8s.pt' (plus precis). Un chemin local absolu marche aussi.
_HAND_MODEL = str(CONFIG.get("hand_detailer_model", "hand_yolov8n.pt")).strip()
_HAND_CONF = float(CONFIG.get("hand_detailer_conf", 0.3))
_hand_model = None


def set_enabled(v):
    global DETAILER_ENABLED
    DETAILER_ENABLED = bool(v)


def set_hands_enabled(v):
    global HAND_ENABLED
    HAND_ENABLED = bool(v)


def set_denoise(v):
    global DETAILER_DENOISE
    try:
        DETAILER_DENOISE = min(0.7, max(0.1, float(v)))
    except (TypeError, ValueError):
        pass
    return f"Face detailer denoise: {DETAILER_DENOISE}"


def set_hand_denoise(v):
    global HAND_DENOISE
    try:
        HAND_DENOISE = min(0.7, max(0.1, float(v)))
    except (TypeError, ValueError):
        pass
    return f"Hand detailer denoise: {HAND_DENOISE}"


def _ensure_hand_model():
    """Charge (une fois) le detecteur de mains YOLOv8. Leve avec un message actionnable
    si ultralytics manque -- c'est une dependance optionnelle (requirements-extra.txt)."""
    global _hand_model
    if _hand_model is not None:
        return _hand_model
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError(
            "hand detailer needs 'ultralytics' (optional): "
            "pip install -r requirements-extra.txt")
    import os
    path = _HAND_MODEL
    if not os.path.isabs(path) and not os.path.isfile(path):
        # nom court -> modele publie par Bingsu/adetailer (telecharge une fois, ~6 Mo)
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("Bingsu/adetailer", _HAND_MODEL)
    _log(f"loading hand detector: {os.path.basename(path)}")
    _hand_model = YOLO(path)
    return _hand_model


def detect_hands(image):
    """Bboxes [x1,y1,x2,y2] des mains detectees (liste vide si aucune)."""
    model = _ensure_hand_model()
    res = model.predict(source=np.asarray(image.convert("RGB")), conf=_HAND_CONF,
                        verbose=False)
    out = []
    for r in res:
        for b in getattr(r, "boxes", []):
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            out.append([x1, y1, x2, y2])
    return out


def _expand_box(b, W, H, margin=_MARGIN):
    """Bbox visage -> crop carre elargi, borne a l'image."""
    x1, y1, x2, y2 = b
    side = max(x2 - x1, y2 - y1) * (1.0 + margin)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    nx1, ny1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    nx2, ny2 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
    return nx1, ny1, nx2, ny2


def _feather_mask(w, h):
    """Masque elliptique 0..1 adouci (jamais de bord carre au recollage)."""
    import cv2
    yy, xx = np.ogrid[:h, :w]
    rx, ry = max(1.0, w * 0.46), max(1.0, h * 0.46)
    m = ((((xx - w / 2.0) / rx) ** 2 + ((yy - h / 2.0) / ry) ** 2) <= 1.0).astype(np.float32)
    return cv2.GaussianBlur(m, (0, 0), max(3.0, min(w, h) * 0.06))


def _detail_regions(image, boxes, prompt, seed, steps, denoise, kind,
                    margin=_MARGIN, min_size=_MIN_FACE, max_n=4, progress=None):
    """Coeur commun visages/mains: pour chaque bbox, crop elargi -> agrandi au sweet
    spot -> refine img2img -> recolle avec un masque elliptique feather.
    Renvoie (image, nb_zones_traitees). Ne leve jamais."""
    import cz_pipeline
    if not boxes:
        _dbg(f"detailer: no {kind} found")
        return image, 0
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)[:max_n]
    out = image.convert("RGB")
    pipe = cz_pipeline.get_pipe("img2img")
    done = 0
    for i, b in enumerate(boxes):
        if (b[2] - b[0]) < min_size or (b[3] - b[1]) < min_size:
            continue
        x1, y1, x2, y2 = _expand_box(b, out.width, out.height, margin)
        cw, ch = x2 - x1, y2 - y1
        if cw <= 0 or ch <= 0:
            continue
        if cw >= out.width * 0.9 and ch >= out.height * 0.9:
            continue   # gros plan: la zone EST l'image, rien a gagner
        if progress:
            try:
                progress(f"{kind} {i + 1}/{len(boxes)}")
            except Exception:
                pass
        crop = out.crop((x1, y1, x2, y2))
        scale = _TARGET / max(cw, ch)
        work = (crop.resize((max(32, int(cw * scale)), max(32, int(ch * scale))), Image.LANCZOS)
                if scale > 1.0 else crop)
        try:
            ref = cz_pipeline._refine_whole(pipe, work, denoise, int(steps), prompt or "", seed)
        except Exception as e:
            _log(f"detailer: refine failed on {kind} {i + 1} ({e})")
            continue
        ref = ref.resize((cw, ch), Image.LANCZOS)
        m = _feather_mask(cw, ch)[..., None]
        base = np.asarray(crop, np.float32)
        blend = (np.asarray(ref, np.float32) * m + base * (1.0 - m)).clip(0, 255).astype(np.uint8)
        out.paste(Image.fromarray(blend), (x1, y1))
        done += 1
    if done:
        _log(f"detailer: refined {done} {kind}(s) (denoise {denoise}, steps {steps})")
    return out, done


def detail_faces(image, prompt, seed, steps=12, denoise=None, progress=None):
    """Retouche chaque visage de l'image (jusqu'a face_detailer_max_faces, du plus grand
    au plus petit). Renvoie (image, nb_visages_traites). Ne leve jamais: en cas de pepin
    (detection indisponible...), renvoie l'image telle quelle."""
    import cz_face
    try:
        boxes = cz_face.detect_faces(image)
    except Exception as e:
        _log(f"detailer: face detection unavailable ({e})")
        return image, 0
    return _detail_regions(image, boxes, prompt, seed, steps,
                           DETAILER_DENOISE if denoise is None else float(denoise),
                           "face", _MARGIN, _MIN_FACE, _MAX_FACES, progress)


def detail_hands(image, prompt, seed, steps=12, denoise=None, progress=None):
    """Retouche chaque main detectee (YOLOv8). Marge plus SERREE que pour un visage:
    elargir trop ferait re-generer l'avant-bras et le decor autour. Renvoie
    (image, nb_mains_traitees); ne leve jamais (ultralytics absent -> message + no-op)."""
    try:
        boxes = detect_hands(image)
    except Exception as e:
        _log(f"detailer: hand detection unavailable ({e})")
        return image, 0
    return _detail_regions(image, boxes, prompt, seed, steps,
                           HAND_DENOISE if denoise is None else float(denoise),
                           "hand", _HAND_MARGIN, _MIN_HAND, _MAX_HANDS, progress)
