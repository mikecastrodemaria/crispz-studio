"""crispz-studio - Asset Browser (standalone SPA in the output folder).

Pulled out of app.py. It writes index.html (the SPA) + _index/manifest.json + the
thumbnails into the output folder, scans recursively (the date subfolders), and deletes an
image (delete_asset, called through the Gradio API by the SPA). It depends on cz_core,
cz_imageio (_read_image_meta) and cz_assets (ASSET_BROWSER_HTML). The UI buttons
(_ui_ab_reindex/_ui_gallery_open) stay in app.py.

"""

import os
import json
import time
import hashlib
import datetime
import threading

from PIL import Image

from cz_core import CONFIG, DEFAULT_OUTPUT_DIR, HERE, IMG_EXTS, _log, _dbg, _prefs
from cz_imageio import _read_image_meta
from cz_assets import ASSET_BROWSER_HTML

_AB_DEFAULTS = {"enabled": False, "generate_thumbnails": True,
                "thumbnail_size": 256, "thumbnail_quality": 85, "blur_thumbnails": False,
                "cache_dir": ""}


def _ab_get(key):
    """An Asset Browser setting. The priority: preferences.json ('ab_<key>', set by the
    UI) > config.txt (asset_browser.<key>) > the default."""
    v = _prefs.get("ab_" + key)
    if v not in (None, ""):
        return v
    cfg = CONFIG.get("asset_browser") or {}
    return cfg.get(key, _AB_DEFAULTS.get(key))


def _batch_enabled():
    cfg = CONFIG.get("civitai_batch")
    return bool(cfg.get("enabled", True)) if isinstance(cfg, dict) else True


def _render_spa():
    """The SPA with the batch button's flag injected (zero cost when disabled: the
    'Fetch all missing' button is not even rendered)."""
    return ASSET_BROWSER_HTML.replace("__CZ_BATCH__", "1" if _batch_enabled() else "")


def _ab_resolve_dir(output_dir):
    d = output_dir or DEFAULT_OUTPUT_DIR
    return d if os.path.isabs(d) else os.path.join(HERE, d)


def _thumbs_root(d):
    """(the thumbnails' folder on disk, the URL prefix) for an output folder.

    The default: '<app>/cache/crispz-thumbs/<slug>' — the app's folder (gitignored) is
    generally on a fast disk, whereas the output folder can be a slow HDD/NAS. Served by
    ABSOLUTE URL (/gradio_api/file=...). The slug depends on the output folder: two output
    folders do not share their cache.
    cache_dir is customisable (UI Save > Asset Browser / the asset_browser.cache_dir
    config); the special value 'output' restores the old behaviour: '<output>/_index/thumbs',
    served RELATIVE, next to the images."""
    cache = str(_ab_get("cache_dir") or "").strip()
    if cache.lower() == "output":
        return os.path.join(d, "_index", "thumbs"), "_index/thumbs/"
    if not cache:
        cache = os.path.join(HERE, "cache")
    slug = hashlib.sha1(os.path.abspath(d).lower().encode("utf-8")).hexdigest()[:12]
    root = os.path.join(cache, "crispz-thumbs", slug)
    return root, "/gradio_api/file=" + os.path.abspath(root).replace("\\", "/") + "/"


def _thumb_paths(d, key):
    """(the path on disk, the URL) of the thumbnail 'key' (e.g. '2026-08-03/img.jpg', 'loras/x.jpg')."""
    root, pfx = _thumbs_root(d)
    return os.path.join(root, key.replace("/", os.sep)), pfx + key


def _replace_retry(tmp, dst, attempts=10):
    """os.replace with retries. On Windows it fails when the destination is open in the
    thread that serves it (Python does not open in FILE_SHARE_DELETE); an HTTP request
    lasts a few ms, so we retry with a capped backoff (~1 s in total). Should it fail
    anyway, we let it through: the caller counts a failure and the thumbnail will be
    regenerated on the next pass, which is better than rewriting dst in place and
    reintroducing the race."""
    for attempt in range(attempts):
        try:
            os.replace(tmp, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(min(0.2, 0.02 * (attempt + 1)))


def _write_atomic_text(path, text):
    """Writes a text file served by the SPA without ever exposing it half written
    (the same reason as _ab_make_thumb: manifest.json and index.html are re-read by the
    browser while the background indexing rewrites them)."""
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        _replace_retry(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _write_text_if_changed(path, text):
    """Like _write_atomic_text, but it does NOT rewrite when the content is already
    identical. Avoids the slow write (an HDD + an antivirus scan on every write) of
    index.html on EVERY opening of the Asset Browser: the SPA is a constant, we only write
    it after a code update. Reading + comparing = fast (~10 KB)."""
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                if f.read() == text:
                    return False
    except Exception:
        pass
    _write_atomic_text(path, text)
    return True


def _ab_make_thumb(src, dst, size, quality):
    """An ATOMIC write: a temporary file then os.replace().

    The SPA serves these thumbnails while the workers generate them. A direct
    im.save(dst) truncates dst to 0 then grows it: an HTTP request that falls into
    that window reads a size (Content-Length through os.stat) then sends more bytes
    -> h11 "Too much data for declared Content-Length", and the browser receives a
    broken thumbnail. With os.replace, a reader sees either the complete old
    version or the new one, never a file being written. A corollary: no more
    truncated thumbnail carrying a fresh mtime that the following passes would take
    for "up to date"."""
    tmp = f"{dst}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            side = min(w, h)
            im = im.crop(((w - side) // 2, (h - side) // 2, (w - side) // 2 + side, (h - side) // 2 + side))
            im = im.resize((int(size), int(size)), Image.LANCZOS)
            im.save(tmp, "JPEG", quality=int(quality), optimize=True)
        _replace_retry(tmp, dst)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _ab_scan(d):
    """The (relpath, fullpath) of every image under d (recursive), _index ignored.
    The most recent first."""
    out = []
    for root, dirs, files in os.walk(d):
        dirs[:] = [x for x in dirs if x != "_index"]
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                fp = os.path.join(root, f)
                out.append((os.path.relpath(fp, d).replace("\\", "/"), fp))
    out.sort(key=lambda t: os.path.getmtime(t[1]), reverse=True)
    return out


def _thumb_workers():
    """The number of threads for generating the thumbnails. PIL releases the GIL while
    decoding/resizing -> the threads really do speed it up. The
    asset_browser.thumb_workers config; the default is min(8, cpu)."""
    cfg = CONFIG.get("asset_browser") or {}
    try:
        n = int(cfg.get("thumb_workers") or 0)
    except (TypeError, ValueError):
        n = 0
    if n < 1:
        n = min(8, os.cpu_count() or 4)
    return max(1, n)


def _ab_gen_thumbs(jobs, size, quality, force=False, progress=None, workers=None):
    """Generates a list of thumbnails IN PARALLEL (used in the background and by the
    'Rebuild thumbnails' button).

    force=False -> skips a thumbnail that is already up to date (newer than the source).
    force=True  -> regenerates everything (corrupted thumbnails / a size change).
    progress(done, total, name) is called after every file.
    Returns {total, made, skipped, failed}."""
    total = len(jobs)
    res = {"total": total, "made": 0, "skipped": 0, "failed": 0}
    if not total:
        return res
    lock = threading.Lock()
    done = [0]

    def _one(job):
        src, tp = job
        out = "failed"
        try:
            if (not force and os.path.isfile(tp)
                    and os.path.getmtime(tp) >= os.path.getmtime(src)):
                out = "skipped"
            else:
                os.makedirs(os.path.dirname(tp), exist_ok=True)
                _ab_make_thumb(src, tp, size, quality)
                out = "made"
        except Exception as e:
            _dbg(f"thumb failed {src}: {e}")
        with lock:
            res[out] += 1
            done[0] += 1
            d = done[0]
        if progress:
            try:
                progress(d, total, os.path.basename(src))
            except Exception:
                pass

    n = workers or _thumb_workers()
    if n > 1 and total > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n) as ex:
            list(ex.map(_one, jobs))
    else:
        for j in jobs:
            _one(j)
    _log(f"asset-browser: thumbnails {res['made']} generated, {res['skipped']} up-to-date, "
         f"{res['failed']} failed ({n} worker(s))")
    return res


_META_CACHE_FILE = "meta_cache.json"
DAYS_INDEX_FILE = "days.json"
DAY_MANIFEST_FILE = "manifest.json"


def _day_of(rel):
    """The day of an image from its subfolder ('2026-07-27/x.png' -> '2026-07-27').
    The root -> '(root)'."""
    sub = os.path.dirname(rel)
    return sub or "(root)"


def _day_dir(out_dir, day):
    return out_dir if day == "(root)" else os.path.join(out_dir, day)


def _write_day_manifests(out_dir, entries, blur, thumb_size):
    """Writes one manifest PER DAY (in that day's folder, Fooocus-style) + the
    _index/days.json index. The UI then opens instantly: it reads days.json (a few KB) and
    only loads the manifest of the day being displayed, instead of a global ~9 MB manifest
    holding 9000+ images."""
    by_day = {}
    for e in entries:
        by_day.setdefault(e.get("day") or "(root)", []).append(e)
    days = []
    for day, imgs in by_day.items():
        payload = {"date": day, "count": len(imgs), "blur": bool(blur),
                   "thumb_size": int(thumb_size),
                   "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                   "images": imgs}
        target_dir = _day_dir(out_dir, day)
        try:
            os.makedirs(target_dir, exist_ok=True)
            _write_atomic_text(os.path.join(target_dir, DAY_MANIFEST_FILE),
                               json.dumps(payload, ensure_ascii=False))
            days.append({"date": day, "count": len(imgs)})
        except Exception as e:
            _dbg(f"day manifest failed for {day}: {e}")
    days.sort(key=lambda x: x["date"], reverse=True)
    idx = {"generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
           "today": datetime.date.today().isoformat(),
           "blur": bool(blur), "thumb_size": int(thumb_size),
           "total": sum(d["count"] for d in days), "days": days}
    _write_atomic_text(os.path.join(out_dir, "_index", DAYS_INDEX_FILE),
                       json.dumps(idx, ensure_ascii=False))
    return days


# Serialises the incremental updates: two images saved in parallel would make a
# concurrent read-modify-write on the same day manifest (a lost entry).
_INCR_LOCK = threading.Lock()


def _entry_for(rel, thumb_rel, path, meta):
    """A manifest entry for an image. ONE single definition, shared by the full
    reindexing and by the incremental hook -> the two paths cannot diverge on the
    format."""
    meta = meta or {}
    sub = os.path.dirname(rel)
    try:
        date = sub if (len(sub) == 10 and sub[4] == "-") else \
            datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
    except Exception:
        date = sub
    return {
        "file": rel, "thumb": thumb_rel, "date": date, "day": sub or "(root)",
        "prompt": meta.get("prompt", ""), "negative": meta.get("negative", ""),
        "seed": meta.get("seed"), "steps": meta.get("steps"),
        "guidance": meta.get("guidance"), "size": meta.get("size"), "mode": meta.get("mode"),
        "model": (os.path.basename(str(meta["model"])) if meta.get("model") else ""),
        "loras": meta.get("loras"), "styles": meta.get("styles"),
        "sampler": meta.get("sampler", ""),
    }


def _load_meta_cache(idx_dir):
    """The image metadata cache: rel -> {mtime, size, meta}. Re-reading the PNG tags
    costs ~25 ms/image (measured: 229 s for 9278 images) and it is redone on EVERY opening
    although 99% of the files have not moved. Defensive: an unreadable cache is ignored (we
    start over), never a blocking error."""
    p = os.path.join(idx_dir, _META_CACHE_FILE)
    try:
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("files"), dict):
                return data["files"]
    except Exception as e:
        _dbg(f"meta cache unreadable, rebuilding: {e}")
    return {}


def _save_meta_cache(idx_dir, files):
    try:
        _write_atomic_text(os.path.join(idx_dir, _META_CACHE_FILE),
                           json.dumps({"files": files}, ensure_ascii=False))
    except Exception as e:
        _dbg(f"meta cache write failed: {e}")


def _meta_cached(cache, rel, path):
    """The metadata of `path`, from the cache when the file has not changed
    (mtime+size), otherwise re-read and cached. Returns (meta, from_cache)."""
    try:
        st = os.stat(path)
        sig = [int(st.st_mtime), int(st.st_size)]
    except Exception:
        sig = None
    hit = cache.get(rel)
    if sig and isinstance(hit, dict) and hit.get("sig") == sig and isinstance(hit.get("meta"), dict):
        return hit["meta"], True
    meta = _read_image_meta(path) or {}
    if sig:
        cache[rel] = {"sig": sig, "meta": meta}
    return meta, False


def ab_reindex(output_dir, thumb_size=256, quality=85, blur=False, gen_thumbs=True,
               background_thumbs=False):
    """Writes index.html + _index/manifest.json (+ the thumbnails). Recursive (the date
    subfolders). background_thumbs=True -> an immediate opening, the thumbnails in the
    background (the full image serves as the fallback in the meantime)."""
    d = _ab_resolve_dir(output_dir)
    os.makedirs(d, exist_ok=True)
    idx_dir = os.path.join(d, "_index")
    os.makedirs(_thumbs_root(d)[0], exist_ok=True)
    os.makedirs(idx_dir, exist_ok=True)
    _write_text_if_changed(os.path.join(d, "index.html"), _render_spa())
    meta_cache = _load_meta_cache(idx_dir)
    fresh_cache, hits, reads = {}, 0, 0
    entries, jobs = [], []
    t_idx = time.time()
    for rel, p in _ab_scan(d):
        thumb_rel = rel  # the fallback = the full image
        tp, trel = _thumb_paths(d, os.path.splitext(rel)[0] + ".jpg")
        if os.path.isfile(tp) and os.path.getmtime(tp) >= os.path.getmtime(p):
            thumb_rel = trel
        elif gen_thumbs:
            if background_thumbs:
                jobs.append((p, tp))
                thumb_rel = trel   # the thumbnail is on its way -> the SPA shows a placeholder then
                                   # loads the real thumbnail (not the full image, which is heavy)
            else:
                try:
                    os.makedirs(os.path.dirname(tp), exist_ok=True)
                    _ab_make_thumb(p, tp, thumb_size, quality)
                    thumb_rel = trel
                except Exception as e:
                    _dbg(f"ab thumb failed {rel}: {e}")
        meta, cached = _meta_cached(meta_cache, rel, p)
        # We only keep the files still present -> the cache does not swell forever
        # when images are deleted.
        if rel in meta_cache:
            fresh_cache[rel] = meta_cache[rel]
        hits += 1 if cached else 0
        reads += 0 if cached else 1
        entries.append(_entry_for(rel, thumb_rel, p, meta))
    manifest = {"count": len(entries), "blur": bool(blur), "thumb_size": int(thumb_size),
                "pending_thumbs": len(jobs),
                "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "images": entries}
    _write_atomic_text(os.path.join(idx_dir, "manifest.json"),
                       json.dumps(manifest, ensure_ascii=False))
    # A per-day index (an instant opening) ON TOP OF the global manifest, which is still
    # written for the global search and for backward compatibility.
    _write_day_manifests(d, entries, blur, thumb_size)
    _save_meta_cache(idx_dir, fresh_cache)
    _log(f"asset-browser: indexed {len(entries)} image(s) in {time.time() - t_idx:.1f}s "
         f"({hits} from meta cache, {reads} read)")
    if jobs and background_thumbs:
        threading.Thread(target=_ab_gen_thumbs, args=(jobs, int(thumb_size), int(quality)),
                         daemon=True).start()
    return len(entries), os.path.join(d, "index.html"), len(jobs)


def ab_open_fast(output_dir, thumb_size=256, quality=85, blur=False, gen_thumbs=True):
    """An INSTANT opening: it writes index.html only (immediately) and launches the full
    (re)building of the manifest + the thumbnails in the background. Returns the path of
    index.html without waiting for the indexing. The SPA loads the existing manifest right
    away (when there is one) and retries/refreshes while the index is rebuilt -> no
    latency on the click (as in Fooocus)."""
    d = _ab_resolve_dir(output_dir)
    os.makedirs(d, exist_ok=True)
    _write_text_if_changed(os.path.join(d, "index.html"), _render_spa())
    # An immediate STUB manifest when none exists -> the SPA loads right away (never
    # "No manifest" again); the real manifest (the background indexing) arrives through the polling.
    idx_dir = os.path.join(d, "_index")
    os.makedirs(idx_dir, exist_ok=True)
    mpath = os.path.join(idx_dir, "manifest.json")
    if not os.path.isfile(mpath):
        try:
            _write_atomic_text(mpath, json.dumps(
                {"count": 0, "building": True, "blur": bool(blur),
                 "generated": "", "images": []}))
        except Exception as e:
            _dbg(f"stub manifest failed: {e}")
    threading.Thread(
        target=lambda: ab_reindex(output_dir, thumb_size, quality, blur, gen_thumbs,
                                  background_thumbs=True),
        daemon=True).start()
    return os.path.join(d, "index.html")


def on_image_saved(image_path, output_dir=None, meta=None):
    """An incremental hook (Fooocus' on_image_logged style): indexes ONE image at the
    moment it is saved -> a thumbnail + an addition to its day's manifest + a refresh of
    days.json. The Asset Browser thus stays up to date without ever rescanning the folder.

    Always silent: an error here must NEVER break a generation."""
    if not _ab_get("enabled"):
        return False
    try:
        d = _ab_resolve_dir(output_dir or DEFAULT_OUTPUT_DIR)
        ap = os.path.abspath(image_path)
        if not os.path.isfile(ap) or not ap.lower().endswith(IMG_EXTS):
            return False
        rel = os.path.relpath(ap, d).replace("\\", "/")
        if rel.startswith(".."):
            return False                       # an image outside the output folder
        day = _day_of(rel)
        size = int(_ab_get("thumbnail_size") or 256)
        quality = int(_ab_get("thumbnail_quality") or 85)
        with _INCR_LOCK:
            # 1) the thumbnail
            tp, trel = _thumb_paths(d, os.path.splitext(rel)[0] + ".jpg")
            thumb_rel = rel
            if _ab_get("generate_thumbnails"):
                try:
                    os.makedirs(os.path.dirname(tp), exist_ok=True)
                    _ab_make_thumb(ap, tp, size, quality)
                    thumb_rel = trel
                except Exception as e:
                    _dbg(f"incr thumb failed {rel}: {e}")
            elif os.path.isfile(tp):
                thumb_rel = trel
            # 2) the entry (the meta supplied by the caller -> zero disk re-read)
            m = meta if isinstance(meta, dict) else (_read_image_meta(ap) or {})
            entry = _entry_for(rel, thumb_rel, ap, m)
            # 3) the day's manifest: replaces the existing entry, the most recent first
            dd = _day_dir(d, day)
            mp = os.path.join(dd, DAY_MANIFEST_FILE)
            man = {"date": day, "images": []}
            try:
                if os.path.isfile(mp):
                    with open(mp, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    if isinstance(loaded, dict) and isinstance(loaded.get("images"), list):
                        man = loaded
            except Exception as e:
                _dbg(f"day manifest unreadable ({day}), recreated: {e}")
            imgs = [x for x in man.get("images", []) if x.get("file") != rel]
            imgs.insert(0, entry)
            man.update({"date": day, "count": len(imgs), "images": imgs,
                        "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")})
            os.makedirs(dd, exist_ok=True)
            _write_atomic_text(mp, json.dumps(man, ensure_ascii=False))
            # 4) days.json (the day's count) — no rescan, we read the existing index
            _bump_days_index(d, day, len(imgs))
        return True
    except Exception as e:
        _dbg(f"on_image_saved failed for {image_path}: {e}")
        return False


def _bump_days_index(out_dir, day, count):
    """Updates a day's count in _index/days.json without rescanning the folder."""
    idx_dir = os.path.join(out_dir, "_index")
    p = os.path.join(idx_dir, DAYS_INDEX_FILE)
    idx = {"days": []}
    try:
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict) and isinstance(loaded.get("days"), list):
                idx = loaded
    except Exception as e:
        _dbg(f"days.json unreadable, recreated: {e}")
    days = [x for x in idx.get("days", []) if x.get("date") != day]
    days.append({"date": day, "count": int(count)})
    days.sort(key=lambda x: str(x.get("date")), reverse=True)
    idx.update({"days": days, "total": sum(int(x.get("count") or 0) for x in days),
                "today": datetime.date.today().isoformat(),
                "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")})
    os.makedirs(idx_dir, exist_ok=True)
    _write_atomic_text(p, json.dumps(idx, ensure_ascii=False))


def _find_preview(safepath):
    """Looks for a preview image next to a .safetensors (the Civitai conventions)."""
    base = os.path.splitext(safepath)[0]
    for ext in (".preview.png", ".preview.jpg", ".preview.jpeg", ".preview.webp",
                ".png", ".jpg", ".jpeg", ".webp"):
        if os.path.isfile(base + ext):
            return base + ext
    return None


# The LoRAs / Models tabs: the SAME folders and the SAME extensions as the rest of the
# app. Historically the Asset Browser only scanned the MAIN folder and only recognised
# .safetensors -- so a library kept in the "extra" folder (the case of the installs that
# keep the models on another disk) showed an EMPTY Models tab, and the GGUFs never
# appeared at all.
_CATALOG_EXTS = {"models": (".safetensors", ".gguf", ".ckpt", ".pt", ".sft"),
                 "loras": (".safetensors", ".ckpt", ".pt")}


def _catalog_dirs(dirs):
    """Normalises into a list of existing folders, without duplicates, order preserved
    (the main one first: at equal names, it is the one that wins)."""
    if not dirs:
        return []
    if isinstance(dirs, str):
        dirs = [dirs]
    out = []
    for d in dirs:
        d = (d or "").strip()
        if d and os.path.isdir(d) and d not in out:
            out.append(d)
    return out


def _scan_catalog(model_dirs, out_dir, kind):
    """Scans the model folder(s): the name, the size, a possible preview, the trigger
    words (LoRA). Generates the previews' thumbnails in the background. Returns the list of
    entries for <kind>.json. `model_dirs` accepts a folder or a list."""
    model_dirs = _catalog_dirs(model_dirs)
    if not model_dirs:
        return []
    exts = _CATALOG_EXTS.get(kind, _CATALOG_EXTS["loras"])
    try:
        from cz_pipeline import lora_keywords
    except Exception:
        def lora_keywords(_p):
            return ""
    entries, jobs, seen = [], [], set()
    for model_dir in model_dirs:
      for root, dirs, files in os.walk(model_dir):
        dirs[:] = [x for x in dirs if x not in ("_index", ".cache", "recipes")]
        for f in files:
            if not f.lower().endswith(exts):
                continue
            fp = os.path.join(root, f)
            rel = os.path.relpath(fp, model_dir).replace("\\", "/")
            if rel.lower() in seen:      # the same name: the main folder wins
                continue
            seen.add(rel.lower())
            sub = os.path.dirname(rel)
            try:
                size_mb = os.path.getsize(fp) / 1e6
            except Exception:
                size_mb = 0
            prev = _find_preview(fp)
            thumb, img = "", ""
            if prev:
                tp, trel = _thumb_paths(out_dir, kind + "/" + os.path.splitext(rel)[0] + ".jpg")
                jobs.append((prev, tp))
                thumb = trel
                img = "/gradio_api/file=" + os.path.abspath(prev).replace("\\", "/")
            # The CivitAI sidecar (<stem>.civitai.json): trigger words + examples + a link.
            try:
                import cz_civitai
                civ = cz_civitai.load_civitai_sidecar(fp)
            except Exception:
                civ = {}
            trig = ", ".join(civ.get("trainedWords") or [])
            if not trig and kind == "loras":
                try:
                    trig = lora_keywords(fp) or ""
                except Exception:
                    trig = ""
            entries.append({
                "file": rel, "name": os.path.splitext(os.path.basename(f))[0],
                "thumb": thumb, "img": img, "day": sub or "(root)",
                "mode": kind[:-1], "size": f"{size_mb:.0f} MB", "prompt": trig,
                "examples": [{"url": e.get("url"), "prompt": e.get("prompt") or "",
                              "width": e.get("width"), "height": e.get("height"),
                              "has_prompt": bool((e.get("prompt") or "").strip())}
                             for e in (civ.get("examples") or []) if e.get("url")][:8],
                "civitai": civ.get("url") or "",
                "reco": civ.get("recommended") or {},
                "update": bool(civ.get("update_available")),
                "latest": civ.get("latest_versionName") or "",
            })
    entries.sort(key=lambda e: e["file"].lower())
    if jobs:
        threading.Thread(target=_ab_gen_thumbs, args=(jobs, 256, 85), daemon=True).start()
    return entries


def _thumb_jobs_for(kind, output_dir, loras_dir=None, checkpoints_dir=None, size=256):
    """The list of (source, destination) thumbnails of an Asset Browser tab.
    kind: 'outputs' | 'loras' | 'models'."""
    d = _ab_resolve_dir(output_dir)
    jobs = []
    if kind == "outputs":
        for rel, p in _ab_scan(d):
            jobs.append((p, _thumb_paths(d, os.path.splitext(rel)[0] + ".jpg")[0]))
        return jobs
    mdirs = _catalog_dirs(loras_dir if kind == "loras" else checkpoints_dir)
    exts = _CATALOG_EXTS.get(kind, _CATALOG_EXTS["loras"])
    seen = set()
    for mdir in mdirs:
      for root, dirs, files in os.walk(mdir):
        dirs[:] = [x for x in dirs if x not in ("_index", ".cache", "recipes")]
        for f in files:
            if not f.lower().endswith(exts):
                continue
            fp = os.path.join(root, f)
            prev = _find_preview(fp)      # no preview -> nothing to make a thumbnail of
            if not prev:
                continue
            rel = os.path.relpath(fp, mdir).replace("\\", "/")
            if rel.lower() in seen:
                continue
            seen.add(rel.lower())
            jobs.append((prev, _thumb_paths(d, kind + "/" + os.path.splitext(rel)[0] + ".jpg")[0]))
    return jobs


def rebuild_thumbs(kind, output_dir, loras_dir=None, checkpoints_dir=None, force=True,
                   progress=None):
    """(Re)generates ALL the thumbnails of a tab, in parallel. force=True regenerates
    even the ones already up to date (corrupted thumbnails, a changed size). Returns
    _ab_gen_thumbs' summary (+ 'kind')."""
    size = int(_ab_get("thumbnail_size") or 256)
    quality = int(_ab_get("thumbnail_quality") or 85)
    jobs = _thumb_jobs_for(kind, output_dir, loras_dir, checkpoints_dir, size)
    _log(f"asset-browser: rebuilding {len(jobs)} {kind} thumbnail(s) (force={force})")
    res = _ab_gen_thumbs(jobs, size, quality, force=force, progress=progress)
    res["kind"] = kind
    return res


def ab_build_catalog(output_dir, loras_dir, checkpoints_dir):
    """Writes _index/loras.json and _index/models.json into the output folder (for the
    Asset Browser's LoRAs / Models tabs)."""
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
        _write_atomic_text(os.path.join(idx, kind + ".json"),
                           json.dumps(manifest, ensure_ascii=False))
        _log(f"asset-browser catalog: {kind} = {len(items)} item(s)")
    return True


def delete_asset(rel, output_dir=None):
    """Deletes an image from the output folder (+ the sidecar + the thumbnail). 'rel' is
    the relative path supplied by the Asset Browser. Checks that it stays INSIDE the
    folder."""
    d = os.path.abspath(_ab_resolve_dir(output_dir or DEFAULT_OUTPUT_DIR))
    target = os.path.abspath(os.path.join(d, rel or ""))
    if not target.startswith(d + os.sep) or not os.path.isfile(target):
        return "not found"
    try:
        os.remove(target)
        for extra in (target + ".json",
                      _thumb_paths(d, os.path.splitext(rel)[0] + ".jpg")[0]):
            if os.path.isfile(extra):
                os.remove(extra)
        _log(f"asset deleted: {rel}")
        return "deleted"
    except Exception as e:
        return f"error: {e}"
