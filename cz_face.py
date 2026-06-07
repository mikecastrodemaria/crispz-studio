"""crispz-studio - FaceSwap (InsightFace/inswapper) + restauration GFPGAN, caption
local BLIP (fallback Ollama) et detourage rembg.

Extrait de app.py. Calcul "feuille" optionnel (features gated): ne depend que de
cz_core (config/paths/log/device) + numpy/PIL; insightface/onnxruntime/cv2/rembg/
transformers sont importes paresseusement et echouent proprement si absents.

L'etat mutable (caches de modeles + reglages restore) vit ici. Le dossier des
checkpoints (encore dans app.py jusqu'au step 7) est passe en parametre a
_faceswap/_resolve_faceswap_model plutot qu'importe (pas de dependance vers app).
"""

import os

import numpy as np
from PIL import Image

from cz_core import CONFIG, HERE, DEVICE, _log

# FaceSwap: restauration GFPGAN post-swap (nettete du visage). Reglable via l'UI.
FACESWAP_RESTORE = bool(CONFIG.get("faceswap_restore", False))
FACESWAP_RESTORE_BLEND = float(CONFIG.get("faceswap_restore_blend", 0.8))


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


def _resolve_faceswap_model(checkpoints_dir=None):
    """Trouve le modele inswapper: faceswap_model_path, sinon recherche dans des
    emplacements usuels, sinon telechargement si faceswap_model_url est defini."""
    cfg = (os.environ.get("FACESWAP_MODEL") or CONFIG.get("faceswap_model_path") or "").strip()
    cands = [cfg] if cfg else []
    search_dirs = [os.path.join(HERE, "faceswap"), os.path.join(HERE, "models")]
    if checkpoints_dir:
        search_dirs.append(checkpoints_dir)
    search_dirs.append(os.path.join(os.path.expanduser("~"), ".insightface", "models"))
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


def _faceswap(target_img, source_img, checkpoints_dir=None):
    """Remplace le(s) visage(s) de target_img par celui de source_img."""
    global _FACE_APP, _FACE_SWAPPER
    try:
        import insightface
        from insightface.app import FaceAnalysis
    except Exception:
        raise RuntimeError("insightface not installed (pip install insightface onnxruntime-gpu).")
    model_path = _resolve_faceswap_model(checkpoints_dir)
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
