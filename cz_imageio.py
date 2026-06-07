"""crispz-studio - image saving, metadata and output filenames.

Extrait de app.py. I/O pure: ne depend que de cz_core (config/paths/log) + PIL.
_gen_meta (qui construit le dict de metadonnees a partir de l'etat modele) reste
dans app.py et passe le dict a save_image().
"""

import os
import json
import datetime

from PIL import Image

from cz_core import (
    CONFIG, SUPPORTED_FORMATS, DEFAULT_OUTPUT_DIR, HERE, IMG_EXTS, _dbg,
)


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
