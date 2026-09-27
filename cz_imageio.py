"""crispz-studio - image saving, metadata and output filenames.

Pulled out of app.py. Pure I/O: it depends only on cz_core (config/paths/log) + PIL.
_gen_meta (which builds the metadata dict from the model state) stays in app.py and
passes the dict to save_image().

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
    """Avoids overwriting: it adds _2, _3... when the file already exists."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 2
    while os.path.exists(f"{base}_{i}{ext}"):
        i += 1
    return f"{base}_{i}{ext}"


def _format_filename(tag, seed, w, h, index=0):
    """The file name from CONFIG['filename_pattern']. The placeholders: {date} {tag}
    {seed} {w} {h} {index} {name}. The default: date + tag + seed + dimensions + index."""
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
    """The output path (or None when display). The name follows CONFIG['filename_pattern']
    (date + seed + tag + dimensions + index) and is made UNIQUE (no overwriting).
    tag = 'upscaled' / 'txt2img' / 'img2img' (+ the source name when one is given)."""
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
    # A subfolder per date (Fooocus-style) for local/custom, when enabled (yes by default).
    if save_mode in ("local", "custom") and CONFIG.get("date_subfolders", True):
        target_dir = os.path.join(target_dir, datetime.datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(target_dir, exist_ok=True)
    return _unique_path(os.path.join(target_dir, fname))


def _exif_bytes(meta):
    """The EXIF (ImageDescription=0x010e) holding the metadata JSON, for jpg/webp."""
    try:
        exif = Image.Exif()
        exif[0x010E] = json.dumps(meta, ensure_ascii=False)  # ImageDescription
        return exif.tobytes()
    except Exception:
        return None


# The metadata scheme (settable in the UI's Advanced / the 'metadata_scheme' config):
#   "crispz" (the default) = a 'crispz' PNG chunk (json) + a .json sidecar.
#   "a1111"                = the same + a 'parameters' PNG chunk (A1111 text) -> read by Civitai.
METADATA_SCHEME = (CONFIG.get("metadata_scheme") or "crispz").lower()


def set_metadata_scheme(v):
    global METADATA_SCHEME
    if v:
        METADATA_SCHEME = str(v).strip().lower().split()[0]  # "a1111 (plain text)" -> "a1111"
    return f"Metadata scheme: {METADATA_SCHEME}"


def _a1111_parameters(meta):
    """Formats a metadata dict the Automatic1111 / Civitai way (the 'parameters' chunk):
        <prompt>\\nNegative prompt: <neg>\\nSteps: N, Sampler: X, CFG scale: Y, Seed: Z, Size: WxH, Model: M"""
    if not meta:
        return ""
    out = [str(meta.get("prompt") or "").strip()]
    if meta.get("negative"):
        out.append(f"Negative prompt: {meta['negative']}")
    parts = []
    if meta.get("steps") is not None:
        parts.append(f"Steps: {meta['steps']}")
    if meta.get("sampler"):
        parts.append(f"Sampler: {meta['sampler']}")
    if meta.get("guidance") is not None:
        parts.append(f"CFG scale: {meta['guidance']}")
    if meta.get("seed") is not None:
        parts.append(f"Seed: {meta['seed']}")
    size = meta.get("size")
    if size:
        if isinstance(size, (list, tuple)) and len(size) == 2:
            parts.append(f"Size: {int(size[0])}x{int(size[1])}")
        else:
            parts.append(f"Size: {size}")
    if meta.get("model"):
        parts.append(f"Model: {os.path.basename(str(meta['model']))}")
    # A replacement text encoder: the same prompt, the same seed, another encoder = another image.
    if meta.get("text_encoder"):
        parts.append(f"Text encoder: {meta['text_encoder']}")
    if parts:
        out.append(", ".join(parts))
    return "\n".join(out)


def save_image(img, dst_path, output_format, meta=None):
    """Saves with the right Pillow format. When meta (a dict): embedded in the PNG (the
    'crispz' chunk, + an A1111 'parameters' chunk when metadata_scheme=a1111), in the EXIF
    (ImageDescription) for jpg/webp, AND written as a .json sidecar.
    When provenance_watermark=on (and trustmark is installed): an invisible TrustMark
    watermark applied to the pixels BEFORE the encoding (see cz_provenance)."""
    try:
        import cz_provenance
        if cz_provenance.wm_enabled():
            img = cz_provenance.wm_apply(img)
    except Exception as e:
        _dbg(f"provenance hook skipped: {e}")
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
                if METADATA_SCHEME == "a1111":
                    pnginfo.add_text("parameters", _a1111_parameters(meta))
            except Exception:
                pnginfo = None
        img.save(dst_path, "PNG", pnginfo=pnginfo)
    if meta:
        try:
            with open(dst_path + ".json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        except Exception as e:
            _dbg(f"sidecar json failed: {e}")
    # The Asset Browser's incremental indexing (Fooocus-style): the thumbnail and the
    # day's manifest are updated HERE, at saving time -> no need to rescan the folder at
    # opening time any more. A late import: cz_assetbrowser imports cz_imageio.
    try:
        import cz_assetbrowser
        cz_assetbrowser.on_image_saved(dst_path, meta=meta)
    except Exception as e:
        _dbg(f"asset-browser incremental index skipped: {e}")


def _list_output_files(output_dir, limit=300):
    """Lists the images of the output folder, recursively (the date subfolders), the most
    recent first. Ignores _index (the Asset Browser's artefacts)."""
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
    """Reads the metadata: the '<file>.json' sidecar, otherwise the 'crispz' PNG chunk."""
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
