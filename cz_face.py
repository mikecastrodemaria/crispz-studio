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

from cz_core import CONFIG, HERE, DEVICE, _log, _prefs, download_with_progress

# FaceSwap: reglages de qualite du post-traitement. Tous reglables via l'UI.
# - restore   : re-synthese du visage a 512 (inswapper ne sort qu'en 128 -> flou).
# - occlusion : masque XSeg, empeche de repeindre par-dessus ce qui passe DEVANT le
#               visage (main, aliment, micro). C'est le defaut d'insightface, qui
#               recolle via un simple rectangle (cf. _swap_one).
# - regions   : segmentation faciale, limite le swap a la peau/yeux/nez/bouche.
# - color     : harmonisation colorimetrique visage genere <-> visage d'origine.
FACESWAP_RESTORE = bool(CONFIG.get("faceswap_restore", True))
FACESWAP_RESTORE_BLEND = float(CONFIG.get("faceswap_restore_blend", 0.8))
FACESWAP_RESTORE_MODEL = str(CONFIG.get("faceswap_restore_model", "codeformer")).lower().strip()
# CodeFormer: 0 = qualite max (plus generatif), 1 = fidelite max a l'entree. Sur un
# swap 128px (degradation forte) l'article recommande ~0.5-0.7.
FACESWAP_RESTORE_FIDELITY = float(CONFIG.get("faceswap_restore_fidelity", 0.7))
FACESWAP_OCCLUSION = bool(CONFIG.get("faceswap_occlusion", True))
FACESWAP_REGIONS = bool(CONFIG.get("faceswap_regions", True))
FACESWAP_COLOR_MATCH = bool(CONFIG.get("faceswap_color_match", True))


_CAPTIONER = None  # (kind, processor, model), charge paresseusement

# Captioner local (auto-describe, SANS Ollama). Configurable via config.txt:
#   "caption_model": "blip-large" (defaut) | "blip-base"
# - blip-large : meme API que blip-base, captions plus riches (~1.9 GB).
# (Florence-2 a ete retire: son code distant est incompatible avec transformers >= ~4.5x
#  exige par Z-Image -> chargeait mais plantait a la generation.)
_CAPTION_REPOS = {
    "blip-base":  "Salesforce/blip-image-captioning-base",
    "blip-large": "Salesforce/blip-image-captioning-large",
}


_CAPTION_MODEL = None  # override UI (None = lire config.txt)


def _current_caption_kind():
    """Type de captioner courant: override UI (session) sinon preferences.json (persiste)
    sinon config.txt, sinon blip-large. Toute valeur inconnue (ex. 'florence2' retire)
    retombe sur blip-large."""
    if _CAPTION_MODEL in _CAPTION_REPOS:
        return _CAPTION_MODEL
    kind = str(_prefs.get("caption_model") or CONFIG.get("caption_model", "blip-large")).lower().strip()
    return kind if kind in _CAPTION_REPOS else "blip-large"


def set_caption_model(kind):
    """Change le captioner local (UI). Invalide le cache -> recharge au prochain usage."""
    global _CAPTION_MODEL, _CAPTIONER
    k = str(kind or "").lower().strip()
    if k in _CAPTION_REPOS and k != _current_caption_kind():
        _CAPTION_MODEL = k
        _CAPTIONER = None
        _log(f"caption model -> {k} (will load on next use)")
    elif k in _CAPTION_REPOS:
        _CAPTION_MODEL = k
    return _current_caption_kind()


def _load_captioner():
    """Charge (une fois) le captioner courant (UI/config). Renvoie (kind, proc, mdl)."""
    global _CAPTIONER
    if _CAPTIONER is not None:
        return _CAPTIONER
    kind = _current_caption_kind()
    repo = _CAPTION_REPOS.get(kind, _CAPTION_REPOS["blip-large"])
    from transformers import BlipProcessor, BlipForConditionalGeneration
    _log(f"loading local captioner BLIP ({repo}); first time downloads ~1-2GB...")
    proc = BlipProcessor.from_pretrained(repo)
    mdl = BlipForConditionalGeneration.from_pretrained(repo).to(DEVICE)
    _CAPTIONER = ("blip", proc, mdl)
    return _CAPTIONER


def _local_caption(image):
    """Caption local SANS Ollama (BLIP). Modele configurable (config.txt 'caption_model':
    blip-large par defaut / blip-base). Charge paresseusement; renvoie une phrase."""
    _kind, proc, mdl = _load_captioner()
    img = image.convert("RGB")
    inputs = proc(img, return_tensors="pt").to(DEVICE)
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
        dst_dir = os.path.join(HERE, "faceswap")
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, "inswapper_128.onnx")
        _log(f"downloading inswapper model from {url} ...")
        download_with_progress(url, dst, timeout=120)   # atomique + progression
        return dst
    return None


def _faceswap(target_img, source_img, checkpoints_dir=None):
    """Remplace le(s) visage(s) de target_img par celui de source_img.

    Pipeline par visage: swap inswapper (128px) -> harmonisation couleur ->
    restauration a 512 (CodeFormer/GFPGAN) -> recollage via un masque d'OCCLUSION
    calcule sur l'image d'origine.

    Ce masque est la difference essentielle avec le recollage natif d'insightface
    (`paste_back=True`), qui utilise un rectangle plein: tout objet situe devant le
    visage (main, aliment, micro, mecheveux) y est repeint par les pixels generes.
    C'est la cause des visages "casses" sur les scenes ou quelque chose touche la
    bouche. Ici on passe donc par `paste_back=False` et on compose nous-memes.
    """
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
    provs = _onnx_providers()
    if _FACE_APP is None:
        _log(f"loading insightface buffalo_l (face detection); providers={provs} ...")
        app = FaceAnalysis(name="buffalo_l", providers=provs) if provs else FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=0 if DEVICE == "cuda" else -1, det_size=(640, 640))
        _FACE_APP = app
    if _FACE_SWAPPER is None:
        _log(f"loading inswapper: {model_path}")
        _FACE_SWAPPER = (insightface.model_zoo.get_model(model_path, providers=provs) if provs
                         else insightface.model_zoo.get_model(model_path))
    tgt = np.asarray(target_img.convert("RGB"))[:, :, ::-1].copy()  # RGB -> BGR
    src = np.asarray(source_img.convert("RGB"))[:, :, ::-1].copy()
    src_faces = _FACE_APP.get(src)
    if not src_faces:
        raise RuntimeError("No face found in the source image.")
    src_face = max(src_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    tgt_faces = _FACE_APP.get(tgt)
    if not tgt_faces:
        raise RuntimeError("No face found in the generated image.")
    res = tgt.copy()
    for f in tgt_faces:
        # `tgt` (original) est passe a part: c'est la seule image ou l'occlusion est
        # encore observable une fois les visages precedents deja remplaces.
        res = _swap_one(res, tgt, f, src_face)
    return Image.fromarray(res[:, :, ::-1])  # BGR -> RGB


def _swap_one(res, orig, face, src_face):
    """Swappe UN visage dans `res` (BGR uint8) et renvoie l'image composee."""
    import cv2
    h, w = res.shape[:2]
    fake, M = _FACE_SWAPPER.get(res, face, src_face, paste_back=False)  # crop 128 + affine
    if FACESWAP_COLOR_MATCH:
        aligned = cv2.warpAffine(res, M, fake.shape[:2][::-1], borderValue=0.0)
        fake = _color_match(fake, aligned)
    IM = cv2.invertAffineTransform(M)
    fake_full = cv2.warpAffine(fake, IM, (w, h), borderValue=0.0)
    mask = _box_mask(fake.shape[0], IM, (h, w))
    visible, M512 = _visible_face_mask(orig, face)
    if visible is not None:
        mask = mask * visible
    m3 = mask[:, :, None]
    out = (fake_full.astype(np.float32) * m3
           + res.astype(np.float32) * (1.0 - m3)).astype(np.uint8)
    if FACESWAP_RESTORE and M512 is not None:
        out = _restore_one(out, M512, mask, FACESWAP_RESTORE_BLEND)
    return out


def _box_mask(crop_size, IM, shape):
    """Masque "boite" d'insightface: le carre aligne, erode puis floute, ramene dans
    l'espace image. On le conserve comme garde-fou sur les bords du crop, mais c'est
    le SEUL masque qu'utilise insightface -- d'ou les artefacts qu'on corrige via
    _visible_face_mask."""
    import cv2
    h, w = shape
    box = cv2.warpAffine(np.full((crop_size, crop_size), 255.0, np.float32), IM, (w, h),
                         borderValue=0.0)
    box[box > 20] = 255
    ys, xs = np.where(box == 255)
    if len(ys) == 0:
        return np.zeros((h, w), np.float32)
    size = int(np.sqrt(max(int(ys.max() - ys.min()), 1) * max(int(xs.max() - xs.min()), 1)))
    box = cv2.erode(box, np.ones((max(size // 10, 10),) * 2, np.uint8), iterations=1)
    k = max(size // 20, 5)
    box = cv2.GaussianBlur(box, (2 * k + 1, 2 * k + 1), 0)
    return box / 255.0


# FFHQ 5-point template (alignement attendu par GFPGAN/CodeFormer), normalise -> x512.
_FFHQ_512 = np.array([
    [0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904],
    [0.39308822, 0.72541100], [0.61150205, 0.72490465]], dtype=np.float32) * 512.0


def _ffhq_matrix(face):
    """Transformation affine vers le crop FFHQ 512 (repere commun a la restauration
    et aux masques). None si les 5 points ne permettent pas de l'estimer."""
    import cv2
    M, _ = cv2.estimateAffinePartial2D(face.kps.astype(np.float32), _FFHQ_512,
                                       method=cv2.LMEDS)
    return M


# ----------------------------------------------------------------------------
# Modeles auxiliaires ONNX (restauration + masques), meme source que gfpgan_1.4
# (facefusion/models-3.0.0). Resolution commune: chemin config -> dossiers usuels
# -> telechargement via URL config. Absent = fonction desactivee proprement.
# ----------------------------------------------------------------------------
_FACEFUSION_HF = "https://huggingface.co/facefusion/models-3.0.0/resolve/main/"

_AUX_MODELS = {   # cle -> (fichier, cle config chemin, cle config URL)
    "gfpgan":     ("gfpgan_1.4.onnx",        "faceswap_restore_path",    "faceswap_restore_url"),
    "codeformer": ("codeformer.onnx",        "faceswap_codeformer_path", "faceswap_codeformer_url"),
    "occluder":   ("dfl_xseg.onnx",          "faceswap_occluder_path",   "faceswap_occluder_url"),
    "parser":     ("bisenet_resnet_34.onnx", "faceswap_parser_path",     "faceswap_parser_url"),
}

_AUX_SESSIONS = {}
_AUX_MISSING = set()   # modeles introuvables: on n'insiste pas (ni retry ni re-log)


def _resolve_aux_model(key):
    fname, path_key, url_key = _AUX_MODELS[key]
    cfg = (CONFIG.get(path_key) or "").strip()
    cands = [cfg] if cfg else []
    for d in (os.path.join(HERE, "faceswap"), os.path.join(HERE, "models")):
        cands.append(os.path.join(d, fname))
    for p in cands:
        if p and os.path.isfile(p):
            return p
    url = (CONFIG.get(url_key) or (_FACEFUSION_HF + fname)).strip()
    if not url:
        return None
    dst_dir = os.path.join(HERE, "faceswap")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, fname)
    _log(f"downloading {key} model ({fname}) from {url} ...")
    download_with_progress(url, dst, timeout=120)   # atomique + progression
    return dst


def _aux_session(key):
    """Session ONNX (mise en cache) d'un modele auxiliaire. Renvoie None si le modele
    est indisponible: chaque appelant doit alors degrader proprement, jamais crasher."""
    if key in _AUX_SESSIONS:
        return _AUX_SESSIONS[key]
    if key in _AUX_MISSING:
        return None
    try:
        path = _resolve_aux_model(key)
        if not path:
            raise RuntimeError("model not found and no URL configured")
        import onnxruntime as ort
        provs = _onnx_providers()   # CUDA puis CPU, sans TensorRT
        _log(f"loading {key}: {path} (providers={provs})")
        sess = (ort.InferenceSession(path, providers=provs) if provs
                else ort.InferenceSession(path))
    except Exception as e:
        _log(f"{key} model unavailable -> feature skipped ({e})")
        _AUX_MISSING.add(key)
        return None
    _AUX_SESSIONS[key] = sess
    return sess


def _soften(mask):
    """Adoucit un masque: flou puis re-etalement de [0.5,1] sur [0,1]. Donne un bord
    progressif mais franc (evite a la fois le lisere dur et le halo diffus)."""
    import cv2
    return (cv2.GaussianBlur(mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2.0


def _occlusion_mask(crop_bgr):
    """Masque XSeg (DeepFaceLab): 1 = peau du visage visible, 0 = quelque chose passe
    DEVANT (main, aliment, micro, cheveux, lunettes). C'est ce masque qui empeche le
    swap de repeindre un objet tenu devant la bouche. None si le modele est absent."""
    sess = _aux_session("occluder")
    if sess is None:
        return None
    import cv2
    inp = sess.get_inputs()[0]
    shape = list(inp.shape)
    nchw = len(shape) == 4 and shape[1] == 3            # NCHW vs NHWC selon l'export
    dim = shape[2] if nchw else shape[1]
    size = int(dim) if isinstance(dim, int) else 256
    blob = cv2.resize(crop_bgr, (size, size)).astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[None] if nchw else blob[None]
    out = np.squeeze(sess.run(None, {inp.name: blob})[0]).astype(np.float32)
    if out.ndim == 3:                                   # (C,H,W) ou (H,W,C) -> 1er plan
        out = out[0] if out.shape[0] < out.shape[-1] else out[..., 0]
    return _soften(cv2.resize(out, crop_bgr.shape[:2][::-1]))


# BiSeNet / CelebAMask-HQ: on garde peau, sourcils, yeux, lunettes, nez, bouche,
# levres. Exclut cheveux (17), chapeau (18), cou (14/15), vetements (16), fond (0).
_PARSER_REGIONS = (1, 2, 3, 4, 5, 6, 10, 11, 12, 13)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _region_mask(crop_bgr):
    """Segmentation faciale (BiSeNet): limite le swap aux regions du visage, pour
    qu'il ne deborde ni sur la chevelure ni sur le cou ni sur l'arriere-plan."""
    sess = _aux_session("parser")
    if sess is None:
        return None
    import cv2
    inp = sess.get_inputs()[0]
    blob = cv2.resize(crop_bgr, (512, 512))[:, :, ::-1].astype(np.float32) / 255.0
    blob = ((blob - _IMAGENET_MEAN) / _IMAGENET_STD).transpose(2, 0, 1)[None].astype(np.float32)
    out = np.squeeze(sess.run(None, {inp.name: blob})[0])       # (19, 512, 512) logits
    m = np.isin(out.argmax(0), _PARSER_REGIONS).astype(np.float32)
    return _soften(cv2.resize(m, crop_bgr.shape[:2][::-1]))


def _visible_face_mask(orig_bgr, face):
    """Masque image-espace des pixels du visage REELLEMENT visibles, calcule sur le
    crop FFHQ 512 de l'image d'origine. Renvoie (masque HxW float32 | None, M512)."""
    import cv2
    M = _ffhq_matrix(face)
    if M is None:
        return None, None
    crop = cv2.warpAffine(orig_bgr, M, (512, 512), borderMode=cv2.BORDER_REPLICATE)
    parts = []
    if FACESWAP_OCCLUSION:
        m = _occlusion_mask(crop)
        if m is not None:
            parts.append(m)
    if FACESWAP_REGIONS:
        m = _region_mask(crop)
        if m is not None:
            parts.append(m)
    if not parts:
        return None, M
    m512 = parts[0]
    for extra in parts[1:]:
        m512 = m512 * extra
    h, w = orig_bgr.shape[:2]
    full = cv2.warpAffine(m512, cv2.invertAffineTransform(M), (w, h))
    return np.clip(full, 0, 1), M


def _color_match(src_bgr, ref_bgr):
    """Aligne la colorimetrie du visage genere sur celle du visage d'origine
    (moyenne/ecart-type par canal en LAB): corrige les ecarts de teint et
    d'exposition entre la photo source et l'image cible."""
    import cv2
    s = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    r = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    for c in range(3):
        ss = float(s[:, :, c].std())
        if ss > 1e-5:
            s[:, :, c] = ((s[:, :, c] - s[:, :, c].mean()) * (float(r[:, :, c].std()) / ss)
                          + r[:, :, c].mean())
    return cv2.cvtColor(np.clip(s, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def _restore_kind():
    return FACESWAP_RESTORE_MODEL if FACESWAP_RESTORE_MODEL in ("codeformer", "gfpgan") else "codeformer"


def _restore_crop(crop512_bgr):
    """Passe un crop FFHQ 512 dans l'enhancer (CodeFormer ou GFPGAN). Meme
    pre/post-traitement pour les deux; CodeFormer prend en plus une entree 'weight'
    (fidelite). Renvoie None si aucun modele n'est disponible."""
    key = _restore_kind()
    sess = _aux_session(key)
    if sess is None and key == "codeformer":
        key, sess = "gfpgan", _aux_session("gfpgan")   # repli si CodeFormer indispo
    if sess is None:
        return None
    import cv2
    blob = cv2.cvtColor(crop512_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = ((blob - 0.5) / 0.5).transpose(2, 0, 1)[None].astype(np.float32)
    feed = {}
    for i in sess.get_inputs():
        feed[i.name] = (np.array([FACESWAP_RESTORE_FIDELITY], dtype=np.double)
                        if i.name == "weight" else blob)
    out = sess.run(None, feed)[0][0]
    out = np.clip(out.transpose(1, 2, 0) * 0.5 + 0.5, 0, 1)
    return cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _restore_one(img_bgr, M512, mask, blend):
    """Restaure le visage aligne par M512 et le recolle en respectant `mask`: la
    restauration non plus ne doit pas repasser par-dessus une occlusion."""
    import cv2
    try:
        h, w = img_bgr.shape[:2]
        crop = cv2.warpAffine(img_bgr, M512, (512, 512), borderMode=cv2.BORDER_REPLICATE)
        rest = _restore_crop(crop)
        if rest is None:
            return img_bgr
        IM = cv2.invertAffineTransform(M512)
        back = cv2.warpAffine(rest, IM, (w, h))
        # Masque elliptique adouci dans l'espace du crop (s'estompe AVANT les bords)
        # -> pas de bord carre visible, puis intersection avec le masque d'occlusion.
        ell = np.zeros((512, 512), np.uint8)
        cv2.ellipse(ell, (256, 256), (256 - 28, 256 - 28), 0, 0, 360, 255, -1)
        ell = cv2.GaussianBlur(ell, (0, 0), 24)
        m = cv2.warpAffine(ell, IM, (w, h)).astype(np.float32) / 255.0
        m = (np.minimum(m, mask) * float(blend))[:, :, None]
        return (back * m + img_bgr * (1 - m)).astype(np.uint8)
    except Exception as e:
        _log(f"face restore (one face) skipped: {e}")
        return img_bgr


def set_faceswap_restore(enabled, blend):
    """Active/desactive la restauration du visage apres le swap + son intensite."""
    global FACESWAP_RESTORE, FACESWAP_RESTORE_BLEND
    FACESWAP_RESTORE = bool(enabled)
    FACESWAP_RESTORE_BLEND = float(blend)
    return f"Face restore ({_restore_kind()}): {'on' if enabled else 'off'} (blend {blend})"


def set_faceswap_quality(occlusion, regions, color_match, model, fidelity):
    """Reglages de qualite du recollage (UI). `occlusion` est le plus important:
    sans lui, tout objet devant le visage est repeint par le swap."""
    global FACESWAP_OCCLUSION, FACESWAP_REGIONS, FACESWAP_COLOR_MATCH
    global FACESWAP_RESTORE_MODEL, FACESWAP_RESTORE_FIDELITY
    FACESWAP_OCCLUSION = bool(occlusion)
    FACESWAP_REGIONS = bool(regions)
    FACESWAP_COLOR_MATCH = bool(color_match)
    m = str(model or "").lower().strip()
    if m in ("codeformer", "gfpgan"):
        FACESWAP_RESTORE_MODEL = m
    FACESWAP_RESTORE_FIDELITY = float(fidelity)
    bits = [f"occlusion {'on' if FACESWAP_OCCLUSION else 'off'}",
            f"regions {'on' if FACESWAP_REGIONS else 'off'}",
            f"color match {'on' if FACESWAP_COLOR_MATCH else 'off'}",
            f"restore {_restore_kind()} (fidelity {FACESWAP_RESTORE_FIDELITY:.2f})"]
    return "Face swap quality: " + ", ".join(bits)


_REMBG_SESSION = None


def _onnx_providers():
    """Providers ONNX disponibles SANS TensorRT (souvent absent -> erreur 'nvinfer_*.dll
    missing' puis chute sur CPU lent). Garde CUDA (GPU) puis CPU. None si onnxruntime
    indisponible (les appelants retombent alors sur le defaut)."""
    try:
        import onnxruntime as ort
        return [p for p in ort.get_available_providers() if p != "TensorrtExecutionProvider"]
    except Exception:
        return None


def _remove_bg(image):
    """Detoure le sujet (fond transparent). Local via rembg (telecharge u2net au
    1er usage). Renvoie une image RGBA. La session ONNX est forcee sur CUDA+CPU
    (TensorRT exclu): evite l'erreur 'nvinfer_10.dll missing' + le fallback CPU lent."""
    global _REMBG_SESSION
    try:
        from rembg import remove, new_session
    except Exception:
        raise RuntimeError("rembg not installed. pip install rembg (or requirements-faceswap.txt).")
    if _REMBG_SESSION is None:
        provs = _onnx_providers()
        try:
            _REMBG_SESSION = new_session("u2net", providers=provs) if provs else new_session("u2net")
            _log(f"rembg session (providers={provs})")
        except Exception as e:
            _log(f"rembg custom session failed ({e}); using default session")
            _REMBG_SESSION = new_session("u2net")
    return remove(image.convert("RGBA"), session=_REMBG_SESSION)
