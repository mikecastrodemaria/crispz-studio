"""crispz-studio - Asset Browser (standalone SPA in the output folder).

Extrait de app.py. Ecrit index.html (SPA) + _index/manifest.json + miniatures dans
le dossier de sortie, scanne recursivement (sous-dossiers date), et supprime une
image (delete_asset, appele via l'API Gradio par la SPA). Depend de cz_core,
cz_imageio (_read_image_meta) et cz_assets (ASSET_BROWSER_HTML). Les boutons UI
(_ui_ab_reindex/_ui_gallery_open) restent dans app.py.
"""

import os
import json
import time
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


def _batch_enabled():
    cfg = CONFIG.get("civitai_batch")
    return bool(cfg.get("enabled", True)) if isinstance(cfg, dict) else True


def _render_spa():
    """SPA avec le drapeau du bouton batch injecte (zero cout si desactive: le bouton
    'Fetch all missing' n'est meme pas rendu)."""
    return ASSET_BROWSER_HTML.replace("__CZ_BATCH__", "1" if _batch_enabled() else "")


def _ab_resolve_dir(output_dir):
    d = output_dir or DEFAULT_OUTPUT_DIR
    return d if os.path.isabs(d) else os.path.join(HERE, d)


def _replace_retry(tmp, dst, attempts=10):
    """os.replace avec retentatives. Sur Windows il echoue si la destination est
    ouverte par le thread qui la sert (Python n'ouvre pas en FILE_SHARE_DELETE) ;
    une requete HTTP dure quelques ms, on retente avec un backoff plafonne
    (~1 s au total). Si ca echoue quand meme, on laisse remonter : l'appelant
    compte un echec et la miniature sera regeneree a la prochaine passe, ce qui
    vaut mieux que de reecrire dst en direct et de reintroduire la course."""
    for attempt in range(attempts):
        try:
            os.replace(tmp, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(min(0.2, 0.02 * (attempt + 1)))


def _write_atomic_text(path, text):
    """Ecrit un fichier texte servi par la SPA sans jamais l'exposer a moitie ecrit
    (meme raison que _ab_make_thumb : manifest.json et index.html sont relus par le
    navigateur pendant que l'indexation en tache de fond les reecrit)."""
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


def _ab_make_thumb(src, dst, size, quality):
    """Ecriture ATOMIQUE : fichier temporaire puis os.replace().

    La SPA sert ces miniatures pendant que les workers les generent. Un
    im.save(dst) direct tronque dst a 0 puis le fait grossir : une requete HTTP
    qui tombe dans cette fenetre lit une taille (Content-Length via os.stat) puis
    envoie plus d'octets -> h11 "Too much data for declared Content-Length", et
    le navigateur recoit une vignette cassee. Avec os.replace, un lecteur voit
    soit l'ancienne version complete, soit la nouvelle, jamais un fichier en
    cours d'ecriture. Corollaire : plus de miniature tronquee avec un mtime frais
    que les passes suivantes prendraient pour "a jour"."""
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


def _thumb_workers():
    """Nb de threads pour la generation de miniatures. PIL relache le GIL pendant le
    decodage/redimensionnement -> les threads accelerent vraiment. Config
    asset_browser.thumb_workers; defaut min(8, cpu)."""
    cfg = CONFIG.get("asset_browser") or {}
    try:
        n = int(cfg.get("thumb_workers") or 0)
    except (TypeError, ValueError):
        n = 0
    if n < 1:
        n = min(8, os.cpu_count() or 4)
    return max(1, n)


def _ab_gen_thumbs(jobs, size, quality, force=False, progress=None, workers=None):
    """Genere une liste de miniatures EN PARALLELE (utilise en tache de fond et par le
    bouton 'Rebuild thumbnails').

    force=False -> saute une miniature deja a jour (plus recente que la source).
    force=True  -> regenere tout (miniatures corrompues / changement de taille).
    progress(done, total, name) est appele apres chaque fichier.
    Renvoie {total, made, skipped, failed}."""
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


def ab_reindex(output_dir, thumb_size=256, quality=85, blur=False, gen_thumbs=True,
               background_thumbs=False):
    """Ecrit index.html + _index/manifest.json (+ thumbnails). Recursif (sous-dossiers
    date). background_thumbs=True -> ouverture immediate, miniatures en tache de fond
    (l'image complete sert de fallback en attendant)."""
    d = _ab_resolve_dir(output_dir)
    os.makedirs(d, exist_ok=True)
    idx_dir = os.path.join(d, "_index")
    os.makedirs(os.path.join(idx_dir, "thumbs"), exist_ok=True)
    _write_atomic_text(os.path.join(d, "index.html"), _render_spa())
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
    _write_atomic_text(os.path.join(idx_dir, "manifest.json"),
                       json.dumps(manifest, ensure_ascii=False))
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
    _write_atomic_text(os.path.join(d, "index.html"), _render_spa())
    # Manifest STUB immediat si aucun n'existe -> la SPA charge tout de suite (plus jamais
    # "No manifest") ; le vrai manifest (indexation en tache de fond) arrive via le polling.
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
            # CivitAI sidecar (<stem>.civitai.json): trigger words + exemples + lien.
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
                "update": bool(civ.get("update_available")),
                "latest": civ.get("latest_versionName") or "",
            })
    entries.sort(key=lambda e: e["file"].lower())
    if jobs:
        threading.Thread(target=_ab_gen_thumbs, args=(jobs, 256, 85), daemon=True).start()
    return entries


def _thumb_jobs_for(kind, output_dir, loras_dir=None, checkpoints_dir=None, size=256):
    """Liste des (source, destination) de miniatures d'un onglet de l'Asset Browser.
    kind: 'outputs' | 'loras' | 'models'."""
    d = _ab_resolve_dir(output_dir)
    jobs = []
    if kind == "outputs":
        for rel, p in _ab_scan(d):
            jobs.append((p, os.path.join(d, "_index/thumbs/" + os.path.splitext(rel)[0] + ".jpg")))
        return jobs
    mdir = loras_dir if kind == "loras" else checkpoints_dir
    if not mdir or not os.path.isdir(mdir):
        return jobs
    for root, dirs, files in os.walk(mdir):
        dirs[:] = [x for x in dirs if x not in ("_index", ".cache", "recipes")]
        for f in files:
            if not f.lower().endswith(".safetensors"):
                continue
            fp = os.path.join(root, f)
            prev = _find_preview(fp)      # pas de preview -> rien a miniaturiser
            if not prev:
                continue
            rel = os.path.relpath(fp, mdir).replace("\\", "/")
            trel = "_index/thumbs/" + kind + "/" + os.path.splitext(rel)[0] + ".jpg"
            jobs.append((prev, os.path.join(d, trel)))
    return jobs


def rebuild_thumbs(kind, output_dir, loras_dir=None, checkpoints_dir=None, force=True,
                   progress=None):
    """(Re)genere TOUTES les miniatures d'un onglet, en parallele. force=True regenere
    meme celles deja a jour (miniatures corrompues, taille changee). Renvoie le resume
    de _ab_gen_thumbs (+ 'kind')."""
    size = int(_ab_get("thumbnail_size") or 256)
    quality = int(_ab_get("thumbnail_quality") or 85)
    jobs = _thumb_jobs_for(kind, output_dir, loras_dir, checkpoints_dir, size)
    _log(f"asset-browser: rebuilding {len(jobs)} {kind} thumbnail(s) (force={force})")
    res = _ab_gen_thumbs(jobs, size, quality, force=force, progress=progress)
    res["kind"] = kind
    return res


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
        _write_atomic_text(os.path.join(idx, kind + ".json"),
                           json.dumps(manifest, ensure_ascii=False))
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
