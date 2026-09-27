"""crispz-studio - FaceSwap (InsightFace/inswapper) + GFPGAN restoration, local BLIP
captioning (with an Ollama fallback) and rembg background removal.

Pulled out of app.py. An optional "leaf" module (gated features): it depends only on
cz_core (config/paths/log/device) + numpy/PIL; insightface/onnxruntime/cv2/rembg/
transformers are imported lazily and fail cleanly when absent.

The mutable state (the model caches + the restore settings) lives here. The checkpoints
folder (still in app.py until step 7) is passed as a parameter to
_faceswap/_resolve_faceswap_model rather than imported (no dependency towards app).

"""

import os
import warnings

import numpy as np
from PIL import Image

from cz_core import CONFIG, HERE, DEVICE, _log, _prefs, download_with_progress

# insightface calls np.linalg.lstsq without rcond (the affine alignment of the
# faces): numpy emits a FutureWarning FOR EVERY face processed - pure library
# noise, with no effect on the result. A TARGETED filter on that module (never a
# global silencing of the FutureWarnings).
warnings.filterwarnings("ignore", category=FutureWarning,
                        module=r"insightface\.utils\.transform")

# FaceSwap: the quality settings of the post-processing. All of them settable from
# the UI.
# - restore   : re-synthesises the face at 512 (inswapper only outputs 128 -> blurry).
# - occlusion : an XSeg mask, keeps us from repainting over whatever passes IN FRONT of
#               the face (a hand, food, a microphone). That is insightface's default,
#               which pastes back through a plain rectangle (see _swap_one).
# - regions   : facial segmentation, limits the swap to the skin/eyes/nose/mouth.
# - color     : colour harmonisation between the generated face and the original one.
FACESWAP_RESTORE = bool(CONFIG.get("faceswap_restore", True))
FACESWAP_RESTORE_BLEND = float(CONFIG.get("faceswap_restore_blend", 0.8))
FACESWAP_RESTORE_MODEL = str(CONFIG.get("faceswap_restore_model", "codeformer")).lower().strip()
# CodeFormer: 0 = max quality (more generative), 1 = max fidelity to the input. On a
# 128px swap (a heavy degradation) the paper recommends ~0.5-0.7.
FACESWAP_RESTORE_FIDELITY = float(CONFIG.get("faceswap_restore_fidelity", 0.7))
FACESWAP_OCCLUSION = bool(CONFIG.get("faceswap_occlusion", True))
FACESWAP_REGIONS = bool(CONFIG.get("faceswap_regions", True))
FACESWAP_COLOR_MATCH = bool(CONFIG.get("faceswap_color_match", True))


_CAPTIONER = None  # (kind, processor, model), loaded lazily

# The local captioner (auto-describe, WITHOUT Ollama). Settable through config.txt:
#   "caption_model": "blip-large" (the default) | "blip-base"
# - blip-large : the same API as blip-base, richer captions (~1.9 GB).
# (Florence-2 was dropped: its remote code is incompatible with the transformers >= ~4.5x
#  Z-Image requires -> it loaded, then crashed at generation time.)
_CAPTION_REPOS = {
    "blip-base":  "Salesforce/blip-image-captioning-base",
    "blip-large": "Salesforce/blip-image-captioning-large",
}


_CAPTION_MODEL = None  # the UI override (None = read config.txt)
# A Caption model can also be an Ollama vision model: "ollama:<name>" (the name as
# Ollama lists it, case included). BLIP stays the fallback should Ollama fail.
OLLAMA_CAPTION_PREFIX = "ollama:"


def _valid_caption_kind(k):
    return k in _CAPTION_REPOS or (k.startswith(OLLAMA_CAPTION_PREFIX)
                                   and len(k) > len(OLLAMA_CAPTION_PREFIX))


def _norm_caption_kind(kind):
    k = str(kind or "").strip()
    return k.lower() if k.lower() in _CAPTION_REPOS else k


def _current_caption_kind():
    """The current captioner kind: the UI override (the session), otherwise preferences.json
    (persisted), otherwise config.txt, otherwise blip-large. Any unknown value ('florence2',
    which was dropped, say) falls back on blip-large."""
    if _CAPTION_MODEL and _valid_caption_kind(_CAPTION_MODEL):
        return _CAPTION_MODEL
    kind = _norm_caption_kind(_prefs.get("caption_model") or CONFIG.get("caption_model", "blip-large"))
    return kind if _valid_caption_kind(kind) else "blip-large"


def set_caption_model(kind):
    """Changes the captioner (UI). Invalidates the BLIP cache -> reloaded on the next use."""
    global _CAPTION_MODEL, _CAPTIONER
    k = _norm_caption_kind(kind)
    if _valid_caption_kind(k) and k != _current_caption_kind():
        _CAPTION_MODEL = k
        _CAPTIONER = None
        _log(f"caption model -> {k} (will load on next use)")
    elif _valid_caption_kind(k):
        _CAPTION_MODEL = k
    return _current_caption_kind()


def _load_captioner():
    """Loads (once) the current captioner (UI/config). Returns (kind, proc, mdl)."""
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
    """A one-sentence caption for Auto-describe and for Describe's fallback. The Caption
    model "ollama:<name>" goes through Ollama; should it fail (off, the model missing, an
    empty answer), BLIP takes over so as not to block the render. Otherwise local BLIP
    (blip-large by default / blip-base), loaded lazily."""
    kind = _current_caption_kind()
    if kind.startswith(OLLAMA_CAPTION_PREFIX):
        model = kind[len(OLLAMA_CAPTION_PREFIX):]
        try:
            from cz_ollama import _ollama_caption
            cap = _ollama_caption(image, model)
            if cap:
                return cap
            _log(f"caption via Ollama ({model}): empty answer, falling back to BLIP")
        except Exception as e:
            _log(f"caption via Ollama ({model}) failed, falling back to BLIP: {e}")
    _kind, proc, mdl = _load_captioner()
    img = image.convert("RGB")
    inputs = proc(img, return_tensors="pt").to(DEVICE)
    out = mdl.generate(**inputs, max_new_tokens=50)
    return proc.decode(out[0], skip_special_tokens=True).strip()


# ----------------------------------------------------------------------------
# FaceSwap (a post-process, optional). InsightFace + the inswapper model. Active
# only when insightface/onnxruntime are installed AND faceswap_model_path points
# at an inswapper (.onnx). Otherwise -> a clear message (a gated feature).
# ----------------------------------------------------------------------------
_FACE_APP = None
_FACE_SWAPPER = None


def _ensure_face_detector():
    """Loads (once) the insightface buffalo_l face detector. Detection ONLY: it does
    not require the inswapper (used by the auto detailer, not just by the swap)."""
    global _FACE_APP
    if _FACE_APP is not None:
        return _FACE_APP
    try:
        from insightface.app import FaceAnalysis
    except Exception:
        raise RuntimeError("insightface not installed (pip install insightface onnxruntime-gpu).")
    provs = _onnx_providers()
    _log(f"loading insightface buffalo_l (face detection); providers={provs} ...")
    app = FaceAnalysis(name="buffalo_l", providers=provs) if provs else FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0 if DEVICE == "cuda" else -1, det_size=(640, 640))
    _FACE_APP = app
    return app


def detect_faces(image):
    """The bboxes of the faces [(x1, y1, x2, y2), ...] of a PIL image (floats, in
    insightface's native order). An empty list when there is no face."""
    app = _ensure_face_detector()
    arr = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()   # RGB -> BGR
    return [tuple(float(v) for v in f.bbox) for f in app.get(arr)]


def detect_faces_full(image):
    """The faces with the position of the MOUTH: [{'box': (x1,y1,x2,y2),
    'mouth': (x,y) | None}]. mouth = the middle of the two mouth corners of the 5
    insightface keypoints (kps[3]/kps[4]). Used by the comic lettering: a bubble's tail
    aims at the speaker's mouth, never at an arbitrary point."""
    app = _ensure_face_detector()
    arr = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()   # RGB -> BGR
    out = []
    for f in app.get(arr):
        mouth = None
        kps = getattr(f, "kps", None)
        if kps is not None and len(kps) >= 5:
            mouth = (float((kps[3][0] + kps[4][0]) / 2.0),
                     float((kps[3][1] + kps[4][1]) / 2.0))
        emb = getattr(f, "normed_embedding", None)
        out.append({"box": tuple(float(v) for v in f.bbox), "mouth": mouth,
                    "embedding": (emb.tolist() if emb is not None else None)})
    return out


_REF_EMB_CACHE = {}


def ref_embedding(path):
    """The embedding of the BIGGEST face of a reference image (a list of floats,
    L2-normalised by insightface), None when there is no face. Cached by (path, mtime).
    Serves the comic lettering: pairing 'who speaks' with 'which face' by comparing the
    faces of a panel with the cast's reference portraits."""
    try:
        key = (os.path.abspath(path), os.path.getmtime(path))
    except OSError:
        return None
    if key in _REF_EMB_CACHE:
        return _REF_EMB_CACHE[key]
    emb = None
    try:
        with Image.open(path) as im:
            faces = detect_faces_full(im)
        faces = [f for f in faces if f.get("embedding")]
        if faces:
            def _area(f):
                x1, y1, x2, y2 = f["box"]
                return (x2 - x1) * (y2 - y1)
            emb = max(faces, key=_area)["embedding"]
    except Exception as e:
        _log(f"ref_embedding({os.path.basename(path)}) failed: {e}")
    _REF_EMB_CACHE[key] = emb
    return emb


def _resolve_faceswap_model(checkpoints_dir=None):
    """Finds the inswapper model: faceswap_model_path, otherwise a search through the
    usual locations, otherwise a download when faceswap_model_url is set."""
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
    # An optional download (a URL supplied by the user in config.txt).
    url = (CONFIG.get("faceswap_model_url") or "").strip()
    if url:
        dst_dir = os.path.join(HERE, "faceswap")
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, "inswapper_128.onnx")
        _log(f"downloading inswapper model from {url} ...")
        download_with_progress(url, dst, timeout=120)   # atomic + progress
        return dst
    return None


def _faceswap(target_img, source_img, checkpoints_dir=None):
    """Replaces the face(s) of target_img with the one from source_img.

    The pipeline per face: an inswapper swap (128px) -> colour harmonisation ->
    restoration at 512 (CodeFormer/GFPGAN) -> pasting back through an OCCLUSION mask
    computed on the original image.

    That mask is the essential difference from insightface's native pasting
    (`paste_back=True`), which uses a solid rectangle: any object in front of the face
    (a hand, food, a microphone, a lock of hair) gets repainted there by the generated
    pixels. That is the cause of the "broken" faces on the scenes where something
    touches the mouth. So we go through `paste_back=False` and compose it ourselves.
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
    _ensure_face_detector()
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
        # `tgt` (the original) is passed separately: it is the only image where the
        # occlusion is still observable once the previous faces have been replaced.
        res = _swap_one(res, tgt, f, src_face)
    return Image.fromarray(res[:, :, ::-1])  # BGR -> RGB


def _swap_one(res, orig, face, src_face):
    """Swaps ONE face in `res` (BGR uint8) and returns the composed image."""
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
    """Insightface's "box" mask: the aligned square, eroded then blurred, brought back
    into image space. We keep it as a guard rail on the crop's edges, but it is the ONLY
    mask insightface uses -- hence the artefacts we correct through
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


# The FFHQ 5-point template (the alignment GFPGAN/CodeFormer expect), normalised -> x512.
_FFHQ_512 = np.array([
    [0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904],
    [0.39308822, 0.72541100], [0.61150205, 0.72490465]], dtype=np.float32) * 512.0


def _ffhq_matrix(face):
    """The affine transform to the FFHQ 512 crop (the frame shared by the restoration
    and the masks). None when the 5 points do not allow estimating it."""
    import cv2
    M, _ = cv2.estimateAffinePartial2D(face.kps.astype(np.float32), _FFHQ_512,
                                       method=cv2.LMEDS)
    return M


# ----------------------------------------------------------------------------
# The auxiliary ONNX models (restoration + masks), from the same source as gfpgan_1.4
# (facefusion/models-3.0.0). A shared resolution: the config path -> the usual folders
# -> a download through the config URL. Absent = the function cleanly disabled.
# ----------------------------------------------------------------------------
_FACEFUSION_HF = "https://huggingface.co/facefusion/models-3.0.0/resolve/main/"

_AUX_MODELS = {   # key -> (file, the path config key, the URL config key)
    "gfpgan":     ("gfpgan_1.4.onnx",        "faceswap_restore_path",    "faceswap_restore_url"),
    "codeformer": ("codeformer.onnx",        "faceswap_codeformer_path", "faceswap_codeformer_url"),
    "occluder":   ("dfl_xseg.onnx",          "faceswap_occluder_path",   "faceswap_occluder_url"),
    "parser":     ("bisenet_resnet_34.onnx", "faceswap_parser_path",     "faceswap_parser_url"),
}

_AUX_SESSIONS = {}
_AUX_MISSING = set()   # models not found: we do not insist (no retry, no re-log)


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
    download_with_progress(url, dst, timeout=120)   # atomic + progress
    return dst


def _aux_session(key):
    """The (cached) ONNX session of an auxiliary model. Returns None when the model is
    unavailable: every caller must then degrade cleanly, never crash."""
    if key in _AUX_SESSIONS:
        return _AUX_SESSIONS[key]
    if key in _AUX_MISSING:
        return None
    try:
        path = _resolve_aux_model(key)
        if not path:
            raise RuntimeError("model not found and no URL configured")
        import onnxruntime as ort
        provs = _onnx_providers()   # CUDA then CPU, without TensorRT
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
    """Softens a mask: a blur then a re-spreading of [0.5,1] over [0,1]. Gives a
    gradual but clean edge (avoids both the hard seam and the diffuse halo)."""
    import cv2
    return (cv2.GaussianBlur(mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2.0


def _occlusion_mask(crop_bgr):
    """The XSeg mask (DeepFaceLab): 1 = visible facial skin, 0 = something passes IN
    FRONT (a hand, food, a microphone, hair, glasses). That mask is what keeps the swap
    from repainting an object held in front of the mouth. None when the model is
    absent."""
    sess = _aux_session("occluder")
    if sess is None:
        return None
    import cv2
    inp = sess.get_inputs()[0]
    shape = list(inp.shape)
    nchw = len(shape) == 4 and shape[1] == 3            # NCHW vs NHWC depending on the export
    dim = shape[2] if nchw else shape[1]
    size = int(dim) if isinstance(dim, int) else 256
    blob = cv2.resize(crop_bgr, (size, size)).astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[None] if nchw else blob[None]
    out = np.squeeze(sess.run(None, {inp.name: blob})[0]).astype(np.float32)
    if out.ndim == 3:                                   # (C,H,W) or (H,W,C) -> the 1st plane
        out = out[0] if out.shape[0] < out.shape[-1] else out[..., 0]
    return _soften(cv2.resize(out, crop_bgr.shape[:2][::-1]))


# BiSeNet / CelebAMask-HQ: we keep skin, eyebrows, eyes, glasses, nose, mouth,
# lips. Excludes hair (17), hat (18), neck (14/15), clothes (16), background (0).
_PARSER_REGIONS = (1, 2, 3, 4, 5, 6, 10, 11, 12, 13)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _region_mask(crop_bgr):
    """Facial segmentation (BiSeNet): limits the swap to the regions of the face, so
    that it spills neither onto the hair nor the neck nor the background."""
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
    """The image-space mask of the face pixels REALLY visible, computed on the FFHQ 512
    crop of the original image. Returns (an HxW float32 mask | None, M512)."""
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
    """Aligns the colours of the generated face with those of the original face (the
    per-channel mean/standard deviation in LAB): corrects the differences in skin tone
    and exposure between the source photo and the target image."""
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
    """Runs an FFHQ 512 crop through the enhancer (CodeFormer or GFPGAN). The same
    pre/post-processing for both; CodeFormer also takes a 'weight' input (fidelity).
    Returns None when no model is available."""
    key = _restore_kind()
    sess = _aux_session(key)
    if sess is None and key == "codeformer":
        key, sess = "gfpgan", _aux_session("gfpgan")   # a fallback when CodeFormer is unavailable
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
    """Restores the face aligned by M512 and pastes it back respecting `mask`: the
    restoration must not go over an occlusion either."""
    import cv2
    try:
        h, w = img_bgr.shape[:2]
        crop = cv2.warpAffine(img_bgr, M512, (512, 512), borderMode=cv2.BORDER_REPLICATE)
        rest = _restore_crop(crop)
        if rest is None:
            return img_bgr
        IM = cv2.invertAffineTransform(M512)
        back = cv2.warpAffine(rest, IM, (w, h))
        # A softened elliptical mask in the crop's space (it fades out BEFORE the
        # edges) -> no visible square edge, then an intersection with the occlusion mask.
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
    """Enables/disables the restoration of the face after the swap + its strength."""
    global FACESWAP_RESTORE, FACESWAP_RESTORE_BLEND
    FACESWAP_RESTORE = bool(enabled)
    FACESWAP_RESTORE_BLEND = float(blend)
    return f"Face restore ({_restore_kind()}): {'on' if enabled else 'off'} (blend {blend})"


def set_faceswap_quality(occlusion, regions, color_match, model, fidelity):
    """The quality settings of the pasting back (UI). `occlusion` is the most
    important one: without it, any object in front of the face is repainted by the swap."""
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
    """The ONNX providers available WITHOUT TensorRT (often absent -> an 'nvinfer_*.dll
    missing' error then a fall back to the slow CPU). Keeps CUDA (GPU) then CPU. None when
    onnxruntime is unavailable (the callers then fall back to the default)."""
    try:
        import onnxruntime as ort
        return [p for p in ort.get_available_providers() if p != "TensorrtExecutionProvider"]
    except Exception:
        return None


def _remove_bg(image):
    """Cuts the subject out (a transparent background). Local, through rembg (it
    downloads u2net on the 1st use). Returns an RGBA image. The ONNX session is forced
    onto CUDA+CPU (TensorRT excluded): avoids the 'nvinfer_10.dll missing' error + the
    slow CPU fallback."""
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
