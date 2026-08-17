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
# Prompt passe au refine de CHAQUE crop de visage. VIDE par defaut, comme pour les
# mains et les tuiles (refine_tile_prompt): le prompt de SCENE fait peindre la scene
# dans le crop -- constate: un prompt 'pancarte CRISPZ STUDIO' a ecrit le texte SUR
# les joues du visage refine. Vide, l'img2img n'affine que le visage source.
_FACE_PROMPT = str(CONFIG.get("face_detailer_prompt", ""))
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
# Peripherique du DETECTEUR de mains. CPU PAR DEFAUT, et ce n'est pas une option de
# confort: le predict() ultralytics sur le GPU empoisonne l'etat CUDA/torch du process,
# et TOUTES les diffusions suivantes sortent en mosaique jusqu'au redemarrage. Prouve
# le 2026-08-17 (sidecars a l'appui, drapeaux detail_*_run): rendu base propre ->
# passe mains H:1 rendue propre -> rendu SUIVANT detruit, reproduit a chaque fois,
# meme apres revert complet du reste. Le YOLOv8n fait 6 Mo: la detection CPU coute
# ~0.1 s par image. 'cuda' reste accepte pour re-tester le jour ou ultralytics/torch
# regle le conflit -- en connaissance de cause.
_HAND_DEVICE = str(CONFIG.get("hand_detailer_device", "cpu")).strip().lower() or "cpu"
# Prompt passe au refine de CHAQUE crop de main. VIDE par defaut, et c'est important:
# avec le prompt de SCENE, le modele repeint le sujet DANS le crop (constate: un
# mini-visage incruste entre le pouce et l'index a denoise 0.4). Meme principe que
# refine_tile_prompt pour le refine tuile: un prompt global sur un crop local fait
# recomposer la scene; vide, l'img2img n'affine que ce que l'image source contient.
# Configurable pour qui veut guider ("detailed hand, natural fingers...").
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
    """Chemin local du .pt YOLO (telecharge une fois depuis Bingsu/adetailer)."""
    import os
    path = _HAND_MODEL
    if not os.path.isabs(path) and not os.path.isfile(path):
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("Bingsu/adetailer", _HAND_MODEL)
    return path


def _ensure_hand_onnx():
    """Chemin du detecteur de mains au format ONNX, exporte UNE fois dans cache/.

    POURQUOI ONNX + SOUS-PROCESS, et pas ultralytics dans l'app: charger le modele
    YOLO (torch) dans le process de diffusion CORROMPT LES POIDS des composants
    partages pendant les transferts d'offload -- prouve au checksum le 2026-08-17 sur
    le chemin GGUF/offload 'model': somme|poids| de l'encodeur de texte 7.2239e7
    stable en process propre, 7.3413e7 puis derive continue (7.3456, 7.3686) des que
    YOLO(path) residait en memoire, MEME SANS predict, MEME en device cpu. Les rendus
    suivants sortent en mosaique puis en NaN. L'export tourne donc dans un
    sous-process (ultralytics y vit et y meurt), et l'app n'utilise a l'execution
    QUE onnxruntime -- la stack d'insightface, qui coexiste sans incident."""
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
    # ultralytics ecrit le .onnx a cote du .pt -> on exporte sur une COPIE dans
    # cache/ (le cache HF n'est pas un endroit ou ecrire).
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
    """Session onnxruntime (une fois). Provider selon hand_detailer_device."""
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
    """Redimensionne en gardant le ratio + padding centre (protocole YOLO).
    Renvoie (image size x size, scale, pad_x, pad_y)."""
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
    """NMS glouton numpy. boxes: (N,4) xyxy."""
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
    """Bboxes [x1,y1,x2,y2] des mains detectees (liste vide si aucune).

    Inference onnxruntime pure (pas d'ultralytics dans ce process, cf.
    _ensure_hand_onnx): letterbox 640 -> session ONNX -> decode YOLOv8 + NMS."""
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
    # Prompt de scene IGNORE pour le refine des crops (cf. _FACE_PROMPT: le texte/decor
    # du prompt finit peint sur le visage sinon).
    return _detail_regions(image, boxes, _FACE_PROMPT, seed, steps,
                           DETAILER_DENOISE if denoise is None else float(denoise),
                           "face", _MARGIN, _MIN_FACE, _MAX_FACES, progress)


def detail_hands(image, prompt, seed, steps=12, denoise=None, progress=None):
    """Retouche chaque main detectee (YOLOv8). Marge plus SERREE que pour un visage:
    elargir trop ferait re-generer l'avant-bras et le decor autour. Renvoie
    (image, nb_mains_traitees); ne leve jamais (ultralytics absent -> message + no-op).

    Le prompt de scene recu est IGNORE pour le refine des crops: il fait peindre le
    sujet dans la main (mini-visage entre pouce et index, constate). On refine avec
    _HAND_PROMPT (vide par defaut = detail local seulement, cf. commentaire)."""
    try:
        boxes = detect_hands(image)
    except Exception as e:
        _log(f"detailer: hand detection unavailable ({e})")
        return image, 0
    return _detail_regions(image, boxes, _HAND_PROMPT, seed, steps,
                           HAND_DENOISE if denoise is None else float(denoise),
                           "hand", _HAND_MARGIN, _MIN_HAND, _MAX_HANDS, progress)
