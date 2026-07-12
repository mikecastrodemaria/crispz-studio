"""crispz-studio - Asset Browser (standalone SPA in the output folder).

Extrait de app.py. Ecrit index.html (SPA) + _index/manifest.json + miniatures dans
le dossier de sortie, scanne recursivement (sous-dossiers date), et supprime une
image (delete_asset, appele via l'API Gradio par la SPA). Depend de cz_core,
cz_imageio (_read_image_meta) et cz_assets (ASSET_BROWSER_HTML). Les boutons UI
(_ui_ab_reindex/_ui_gallery_open) restent dans app.py.
"""

import os
import json
import datetime
import threading

from PIL import Image

from cz_core import CONFIG, DEFAULT_OUTPUT_DIR, HERE, IMG_EXTS, _log, _dbg
from cz_imageio import _read_image_meta
from cz_assets import ASSET_BROWSER_HTML

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
                thumb_rel = trel   # vignette a venir -> la SPA montre un placeholder puis
                                   # charge la vraie vignette (pas l'image complete, lourde)
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
            "loras": meta.get("loras"), "styles": meta.get("styles"), "sampler": meta.get("sampler", ""),
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


def ab_open_fast(output_dir, thumb_size=256, quality=85, blur=False, gen_thumbs=True):
    """Ouverture INSTANTANEE: ecrit seulement index.html (immediat) et lance la
    (re)construction complete du manifest + miniatures en tache de fond. Renvoie le
    chemin de index.html sans attendre l'indexation. La SPA charge le manifest existant
    tout de suite (s'il y en a un) et re-essaie/rafraichit pendant que l'index se
    reconstruit -> pas de latence au clic (comme Fooocus)."""
    d = _ab_resolve_dir(output_dir)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "index.html"), "w", encoding="utf-8") as f:
        f.write(ASSET_BROWSER_HTML)
    threading.Thread(
        target=lambda: ab_reindex(output_dir, thumb_size, quality, blur, gen_thumbs,
                                  background_thumbs=True),
        daemon=True).start()
    return os.path.join(d, "index.html")


def _find_preview(safepath):
    """Cherche une image de preview a cote d'un .safetensors (conventions Civitai)."""
    base = os.path.splitext(safepath)[0]
    for ext in (".preview.png", ".preview.jpg", ".preview.jpeg", ".preview.webp",
                ".png", ".jpg", ".jpeg", ".webp"):
        if os.path.isfile(base + ext):
            return base + ext
    return None


def _scan_catalog(model_dir, out_dir, kind):
    """Scanne un dossier de modeles (.safetensors): nom, taille, preview eventuelle,
    trigger words (LoRA). Genere les miniatures des previews en tache de fond.
    Renvoie la liste d'entrees pour <kind>.json."""
    if not model_dir or not os.path.isdir(model_dir):
        return []
    try:
        from cz_pipeline import lora_keywords
    except Exception:
        def lora_keywords(_p):
            return ""
    entries, jobs = [], []
    for root, dirs, files in os.walk(model_dir):
        dirs[:] = [x for x in dirs if x not in ("_index", ".cache", "recipes")]
        for f in files:
            if not f.lower().endswith(".safetensors"):
                continue
            fp = os.path.join(root, f)
            rel = os.path.relpath(fp, model_dir).replace("\\", "/")
            sub = os.path.dirname(rel)
            try:
                size_mb = os.path.getsize(fp) / 1e6
            except Exception:
                size_mb = 0
            prev = _find_preview(fp)
            thumb, img = "", ""
            if prev:
                trel = "_index/thumbs/" + kind + "/" + os.path.splitext(rel)[0] + ".jpg"
                jobs.append((prev, os.path.join(out_dir, trel)))
                thumb = trel
                img = "/gradio_api/file=" + os.path.abspath(prev).replace("\\", "/")
            trig = ""
            if kind == "loras":
                try:
                    trig = lora_keywords(fp) or ""
                except Exception:
                    trig = ""
            entries.append({
                "file": rel, "name": os.path.splitext(os.path.basename(f))[0],
                "thumb": thumb, "img": img, "day": sub or "(root)",
                "mode": kind[:-1], "size": f"{size_mb:.0f} MB", "prompt": trig,
            })
    entries.sort(key=lambda e: e["file"].lower())
    if jobs:
        threading.Thread(target=_ab_gen_thumbs, args=(jobs, 256, 85), daemon=True).start()
    return entries


def ab_build_catalog(output_dir, loras_dir, checkpoints_dir):
    """Ecrit _index/loras.json et _index/models.json dans le dossier de sortie (pour les
    onglets LoRAs / Models de l'Asset Browser)."""
    d = _ab_resolve_dir(output_dir)
    idx = os.path.join(d, "_index")
    os.makedirs(idx, exist_ok=True)
    for kind, mdir in (("loras", loras_dir), ("models", checkpoints_dir)):
        try:
            items = _scan_catalog(mdir, d, kind)
        except Exception as e:
            _dbg(f"catalog {kind} failed: {e}")
            items = []
        manifest = {"count": len(items), "kind": kind,
                    "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "images": items}
        with open(os.path.join(idx, kind + ".json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False)
        _log(f"asset-browser catalog: {kind} = {len(items)} item(s)")
    return True


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
