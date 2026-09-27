"""crispz-studio - the auto face detailer (in the style of ADetailer / Fooocus "Enhance").

After a render, it detects the faces (insightface buffalo_l, already loaded for the Face
Swap) and puts EVERY face through img2img at high resolution:
  a widened crop (+60%) -> enlarged to the model's sweet spot (~832 px) -> a Z-Image refine
  (a moderate denoise, the same seed/prompt) -> shrunk back -> pasted back with a feathered
  elliptical mask (like the Face Swap's GFPGAN pasting: no square edge).

Enabled by the "🔧 Detail faces" box under the Generate button (a module flag, no new input
in _gen_inputs -> the queue and the X/Y/Z grid do not move), or by the 'face_detailer'
config. Settings: 'face_detailer_denoise' (0.35), 'face_detailer_max_faces'.

"""

import numpy as np
from PIL import Image

from cz_core import CONFIG, _log, _dbg

DETAILER_ENABLED = bool(CONFIG.get("face_detailer", False))
DETAILER_DENOISE = float(CONFIG.get("face_detailer_denoise", 0.35))
_MAX_FACES = max(1, int(CONFIG.get("face_detailer_max_faces", 4)))
# The prompt passed to the refine of EVERY face crop. EMPTY by default, as for the
# hands and the tiles (refine_tile_prompt): the SCENE prompt makes it paint the scene
# inside the crop -- seen in the act: a 'CRISPZ STUDIO sign' prompt wrote the text ON
# the cheeks of the refined face. Empty, the img2img only sharpens the source face.
_FACE_PROMPT = str(CONFIG.get("face_detailer_prompt", ""))
_TARGET = 832      # the crop's working side (the Z-Image sweet spot, /32)
_MARGIN = 0.6      # the expansion of the face bbox (context: hair, neck)
_MIN_FACE = 28     # px: below that, too small to gain anything

# --- The HANDS detailer (the same mechanics, another detector) ------------------
# The hands are the weak point of every diffusion model. The same circuit as the faces:
# a widened crop -> a high-res refine -> a feathered pasting. The detector is a hands
# YOLOv8 (ultralytics), an OPTIONAL dependency: absent -> the feature disables itself
# with a clear message, the rest of the app is intact.
HAND_ENABLED = bool(CONFIG.get("hand_detailer", False))
HAND_DENOISE = float(CONFIG.get("hand_detailer_denoise", 0.4))
_MAX_HANDS = max(1, int(CONFIG.get("hand_detailer_max_hands", 4)))
_MIN_HAND = 24
_HAND_MARGIN = float(CONFIG.get("hand_detailer_margin", 0.35))
# the model's HF repo (Bingsu/adetailer): 'hand_yolov8n.pt' (6 MB, fast) or
# 'hand_yolov8s.pt' (more precise). An absolute local path works too.
_HAND_MODEL = str(CONFIG.get("hand_detailer_model", "hand_yolov8n.pt")).strip()
_HAND_CONF = float(CONFIG.get("hand_detailer_conf", 0.3))
# The device of the hands DETECTOR. CPU BY DEFAULT, and this is not a comfort option:
# the ultralytics predict() on the GPU poisons the process' CUDA/torch state, and ALL the
# diffusions that follow come out as a mosaic until the restart. Proven on 2026-08-17
# (sidecars to back it up, the detail_*_run flags): a clean base render -> a hands pass
# H:1 rendered clean -> the NEXT render destroyed, reproduced every time, even after a
# complete revert of everything else. The YOLOv8n weighs 6 MB: the CPU detection costs
# ~0.1 s per image. 'cuda' is still accepted, to re-test it the day ultralytics/torch
# settle the conflict -- with one's eyes open.
_HAND_DEVICE = str(CONFIG.get("hand_detailer_device", "cpu")).strip().lower() or "cpu"
# The prompt passed to the refine of EVERY hand crop. EMPTY by default, and that
# matters: with the SCENE prompt, the model repaints the subject INSIDE the crop (seen in
# the act: a mini-face embedded between the thumb and the index finger at denoise 0.4).
# The same principle as refine_tile_prompt for the tile refine: a global prompt on a local
# crop makes it recompose the scene; empty, the img2img only sharpens what the source
# image holds. Settable for whoever wants to guide it ("detailed hand, natural fingers...").
_HAND_PROMPT = str(CONFIG.get("hand_detailer_prompt", ""))
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


def _resolve_hand_pt():
    """The local path of the YOLO .pt (downloaded once from Bingsu/adetailer)."""
    import os
    path = _HAND_MODEL
    if not os.path.isabs(path) and not os.path.isfile(path):
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("Bingsu/adetailer", _HAND_MODEL)
    return path


def _ensure_hand_onnx():
    """The path of the hands detector in the ONNX format, exported ONCE into cache/.

    WHY ONNX + A SUBPROCESS, and not ultralytics inside the app: loading the YOLO
    model (torch) into the diffusion process CORRUPTS THE WEIGHTS of the shared
    components during the offload transfers -- proven by checksum on 2026-08-17 on the
    GGUF/offload 'model' path: sum|weights| of the text encoder 7.2239e7 stable in a
    clean process, 7.3413e7 then a continuous drift (7.3456, 7.3686) as soon as
    YOLO(path) lived in memory, EVEN WITHOUT predict, EVEN on device cpu. The renders
    that follow come out as a mosaic then as NaN. So the export runs in a subprocess
    (ultralytics lives and dies there), and at run time the app uses ONLY onnxruntime
    -- insightface's stack, which coexists without incident."""
    import os
    import shutil
    import subprocess
    import sys
    from cz_core import HERE
    pt = _resolve_hand_pt()
    stem = os.path.splitext(os.path.basename(pt))[0]
    cache_dir = os.path.join(HERE, "cache")
    onnx_path = os.path.join(cache_dir, stem + ".onnx")
    if os.path.isfile(onnx_path):
        return onnx_path
    os.makedirs(cache_dir, exist_ok=True)
    # ultralytics writes the .onnx next to the .pt -> so we export on a COPY in
    # cache/ (the HF cache is not a place to write to).
    pt_copy = os.path.join(cache_dir, stem + ".pt")
    shutil.copyfile(pt, pt_copy)
    _log(f"exporting hand detector to ONNX (once, in a subprocess): {stem}.pt ...")
    code = ("import sys\n"
            "from ultralytics import YOLO\n"
            "YOLO(sys.argv[1]).export(format='onnx', imgsz=640, dynamic=False, "
            "device='cpu')\n")
    try:
        r = subprocess.run([sys.executable, "-c", code, pt_copy],
                           capture_output=True, text=True, timeout=300)
        if r.returncode != 0 or not os.path.isfile(onnx_path):
            tail = (r.stderr or r.stdout or "").strip()[-400:]
            raise RuntimeError(
                f"ONNX export of {stem}.pt failed (needs 'ultralytics' + 'onnx', "
                f"see requirements-extra.txt): {tail}")
    finally:
        try:
            os.remove(pt_copy)
        except OSError:
            pass
    _log(f"hand detector ready: {os.path.basename(onnx_path)}")
    return onnx_path


def _ensure_hand_session():
    """The onnxruntime session (once). The provider according to hand_detailer_device."""
    global _hand_model
    if _hand_model is not None:
        return _hand_model
    try:
        import onnxruntime
    except ImportError:
        raise RuntimeError(
            "hand detailer needs 'onnxruntime' (already required by Face Swap): "
            "pip install onnxruntime-gpu")
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if _HAND_DEVICE == "cuda" else ["CPUExecutionProvider"])
    _hand_model = onnxruntime.InferenceSession(_ensure_hand_onnx(),
                                               providers=providers)
    return _hand_model


def _letterbox(arr, size=640, pad=114):
    """Resizes keeping the ratio + centred padding (the YOLO protocol).
    Returns (a size x size image, scale, pad_x, pad_y)."""
    h, w = arr.shape[:2]
    s = min(size / w, size / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    from PIL import Image as _Image
    resized = np.asarray(_Image.fromarray(arr).resize((nw, nh), _Image.BILINEAR))
    out = np.full((size, size, 3), pad, dtype=np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    out[py:py + nh, px:px + nw] = resized
    return out, s, px, py


def _nms(boxes, scores, iou_thr=0.45):
    """A greedy numpy NMS. boxes: (N,4) xyxy."""
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        a = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        b = ((boxes[order[1:], 2] - boxes[order[1:], 0])
             * (boxes[order[1:], 3] - boxes[order[1:], 1]))
        iou = inter / (a + b - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def detect_hands(image):
    """The bboxes [x1,y1,x2,y2] of the hands detected (an empty list when there is none).

    Pure onnxruntime inference (no ultralytics in this process, see
    _ensure_hand_onnx): a 640 letterbox -> the ONNX session -> a YOLOv8 decode + NMS."""
    sess = _ensure_hand_session()
    rgb = np.asarray(image.convert("RGB"))
    inp, s, px, py = _letterbox(rgb)
    x = inp.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    if out.shape[0] < out.shape[1]:            # (4+nc, N) -> (N, 4+nc)
        out = out.T
    scores = out[:, 4:].max(axis=1)
    m = scores >= _HAND_CONF
    if not m.any():
        return []
    cx, cy, w, h = (out[m, 0], out[m, 1], out[m, 2], out[m, 3])
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    scores = scores[m]
    keep = _nms(boxes, scores)
    W, H = image.size
    res = []
    for i in keep:
        x1 = min(max((boxes[i, 0] - px) / s, 0), W)
        y1 = min(max((boxes[i, 1] - py) / s, 0), H)
        x2 = min(max((boxes[i, 2] - px) / s, 0), W)
        y2 = min(max((boxes[i, 3] - py) / s, 0), H)
        if x2 > x1 and y2 > y1:
            res.append([float(x1), float(y1), float(x2), float(y2)])
    return res


def _expand_box(b, W, H, margin=_MARGIN):
    """A face bbox -> a widened square crop, bounded by the image."""
    x1, y1, x2, y2 = b
    side = max(x2 - x1, y2 - y1) * (1.0 + margin)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    nx1, ny1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    nx2, ny2 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
    return nx1, ny1, nx2, ny2


def _feather_mask(w, h):
    """A softened 0..1 elliptical mask (never a square edge at pasting time)."""
    import cv2
    yy, xx = np.ogrid[:h, :w]
    rx, ry = max(1.0, w * 0.46), max(1.0, h * 0.46)
    m = ((((xx - w / 2.0) / rx) ** 2 + ((yy - h / 2.0) / ry) ** 2) <= 1.0).astype(np.float32)
    return cv2.GaussianBlur(m, (0, 0), max(3.0, min(w, h) * 0.06))


def _detail_regions(image, boxes, prompt, seed, steps, denoise, kind,
                    margin=_MARGIN, min_size=_MIN_FACE, max_n=4, progress=None):
    """The core shared by faces and hands: for every bbox, a widened crop -> enlarged to
    the sweet spot -> an img2img refine -> pasted back with a feathered elliptical mask.
    Returns (image, the number of areas processed). It never raises."""
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
            continue   # a close-up: the area IS the image, nothing to gain
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
            # After the upscale, torch's cache could hold the whole card: the pass
            # failed on "CUDA error: out of memory". We empty it and retry once.
            ref = cz_pipeline.retry_on_oom(f"detailer {kind} {i + 1}", cz_pipeline._refine_whole,
                                           pipe, work, denoise, int(steps), prompt or "", seed)
        except Exception as e:
            _log(f"detailer: refine failed on {kind} {i + 1} ({e})")
            if cz_pipeline.is_oom(e):
                # Still full after the emptying: the areas that follow would fail the same way.
                _log(f"detailer: still out of VRAM, remaining {kind}(s) skipped")
                break
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
    """Retouches every face of the image (up to face_detailer_max_faces, from the
    biggest to the smallest). Returns (image, the number of faces processed). It never
    raises: on a hitch (the detection unavailable...), it returns the image as it is."""
    import cz_face
    try:
        boxes = cz_face.detect_faces(image)
    except Exception as e:
        _log(f"detailer: face detection unavailable ({e})")
        return image, 0
    # The scene prompt is IGNORED for the refine of the crops (see _FACE_PROMPT: the
    # prompt's text/scenery ends up painted on the face otherwise).
    return _detail_regions(image, boxes, _FACE_PROMPT, seed, steps,
                           DETAILER_DENOISE if denoise is None else float(denoise),
                           "face", _MARGIN, _MIN_FACE, _MAX_FACES, progress)


def detail_hands(image, prompt, seed, steps=12, denoise=None, progress=None):
    """Retouches every hand detected (YOLOv8). A TIGHTER margin than for a face:
    widening too much would re-generate the forearm and the scenery around it. Returns
    (image, the number of hands processed); it never raises (ultralytics absent -> a message
    + a no-op).

    The scene prompt received is IGNORED for the refine of the crops: it makes it paint the
    subject into the hand (a mini-face between thumb and index finger, seen in the act). We
    refine with _HAND_PROMPT (empty by default = local detail only, see the comment)."""
    try:
        boxes = detect_hands(image)
    except Exception as e:
        _log(f"detailer: hand detection unavailable ({e})")
        return image, 0
    return _detail_regions(image, boxes, _HAND_PROMPT, seed, steps,
                           HAND_DENOISE if denoise is None else float(denoise),
                           "hand", _HAND_MARGIN, _MIN_HAND, _MAX_HANDS, progress)
