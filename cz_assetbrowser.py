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
import hashlib
import datetime
import threading

from PIL import Image

from cz_core import CONFIG, DEFAULT_OUTPUT_DIR, HERE, IMG_EXTS, _log, _dbg
from cz_imageio import _read_image_meta
from cz_assets import ASSET_BROWSER_HTML

_AB_DEFAULTS = {"enabled": False, "generate_thumbnails": True,
                "thumbnail_size": 256, "thumbnail_quality": 85, "blur_thumbnails": False,
                "cache_dir": ""}


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


def _thumbs_root(d):
    """(dossier disque des miniatures, prefixe d'URL) pour un dossier de sortie.

    Defaut: '<sortie>/_index/thumbs', servi en RELATIF par la SPA.
    Si asset_browser.cache_dir est defini (ex. un SSD alors que les images sont sur un
    disque lent), les miniatures vont dans '<cache>/crispz-thumbs/<slug>' et sont servies
    en URL ABSOLUE (/gradio_api/file=...). Le slug depend du dossier de sortie: deux
    dossiers de sortie ne partagent pas leur cache."""
    cache = str(_ab_get("cache_dir") or "").strip()
    if not cache:
        return os.path.join(d, "_index", "thumbs"), "_index/thumbs/"
    slug = hashlib.sha1(os.path.abspath(d).lower().encode("utf-8")).hexdigest()[:12]
    root = os.path.join(cache, "crispz-thumbs", slug)
    return root, "/gradio_api/file=" + os.path.abspath(root).replace("\\", "/") + "/"


def _thumb_paths(d, key):
    """(chemin disque, URL) de la miniature 'key' (ex. '2026-08-03/img.jpg', 'loras/x.jpg')."""
    root, pfx = _thumbs_root(d)
    return os.path.join(root, key.replace("/", os.sep)), pfx + key


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


def _write_text_if_changed(path, text):
    """Comme _write_atomic_text, mais NE reecrit pas si le contenu est deja identique.
    Evite le write lent (HDD + scan antivirus a chaque write) de index.html a CHAQUE
    ouverture de l'Asset Browser : la SPA est un constante, on ne l'ecrit qu'apres une
    mise a jour du code. Lecture+comparaison = rapide (~10 Ko)."""
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


_META_CACHE_FILE = "meta_cache.json"
DAYS_INDEX_FILE = "days.json"
DAY_MANIFEST_FILE = "manifest.json"


def _day_of(rel):
    """Jour d'une image d'apres son sous-dossier ('2026-07-27/x.png' -> '2026-07-27').
    Racine -> '(root)'."""
    sub = os.path.dirname(rel)
    return sub or "(root)"


def _day_dir(out_dir, day):
    return out_dir if day == "(root)" else os.path.join(out_dir, day)


def _write_day_manifests(out_dir, entries, blur, thumb_size):
    """Ecrit un manifest PAR JOUR (dans le dossier du jour, facon Fooocus) + l'index
    _index/days.json. L'UI ouvre alors instantanement: elle lit days.json (quelques Ko)
    et ne charge que le manifest du jour affiche, au lieu d'un manifest global de ~9 Mo
    contenant 9000+ images."""
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


# Serialise les mises a jour incrementales: deux images sauvees en parallele feraient
# un read-modify-write concurrent sur le meme manifest de jour (perte d'entree).
_INCR_LOCK = threading.Lock()


def _entry_for(rel, thumb_rel, path, meta):
    """Entree de manifest pour une image. UNE seule definition, partagee par la
    reindexation complete et le hook incremental -> les deux chemins ne peuvent pas
    diverger sur le format."""
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
    """Cache des metadonnees d'images: rel -> {mtime, size, meta}. Relire les tags PNG
    coute ~25 ms/image (mesure: 229 s pour 9278 images) et c'est refait a CHAQUE
    ouverture alors que 99% des fichiers n'ont pas bouge. Defensif: un cache illisible
    est ignore (on repart de zero), jamais d'erreur bloquante."""
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
    """Metadonnees de `path`, depuis le cache si le fichier n'a pas change (mtime+taille),
    sinon relues et mises en cache. Renvoie (meta, from_cache)."""
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
    """Ecrit index.html + _index/manifest.json (+ thumbnails). Recursif (sous-dossiers
    date). background_thumbs=True -> ouverture immediate, miniatures en tache de fond
    (l'image complete sert de fallback en attendant)."""
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
        thumb_rel = rel  # fallback = image complete
        tp, trel = _thumb_paths(d, os.path.splitext(rel)[0] + ".jpg")
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
        meta, cached = _meta_cached(meta_cache, rel, p)
        # On ne garde que les fichiers encore presents -> le cache ne gonfle pas
        # indefiniment quand des images sont supprimees.
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
    # Index par jour (ouverture instantanee) EN PLUS du manifest global, qui reste ecrit
    # pour la recherche globale et la compatibilite descendante.
    _write_day_manifests(d, entries, blur, thumb_size)
    _save_meta_cache(idx_dir, fresh_cache)
    _log(f"asset-browser: indexed {len(entries)} image(s) in {time.time() - t_idx:.1f}s "
         f"({hits} from meta cache, {reads} read)")
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
    _write_text_if_changed(os.path.join(d, "index.html"), _render_spa())
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


def on_image_saved(image_path, output_dir=None, meta=None):
    """Hook incremental (facon Fooocus on_image_logged): indexe UNE image au moment ou
    elle est sauvegardee -> miniature + ajout au manifest de son jour + refresh de
    days.json. L'Asset Browser reste ainsi a jour sans jamais rescanner le dossier.

    Toujours silencieux: une erreur ici ne doit JAMAIS casser une generation."""
    if not _ab_get("enabled"):
        return False
    try:
        d = _ab_resolve_dir(output_dir or DEFAULT_OUTPUT_DIR)
        ap = os.path.abspath(image_path)
        if not os.path.isfile(ap) or not ap.lower().endswith(IMG_EXTS):
            return False
        rel = os.path.relpath(ap, d).replace("\\", "/")
        if rel.startswith(".."):
            return False                       # image hors du dossier de sortie
        day = _day_of(rel)
        size = int(_ab_get("thumbnail_size") or 256)
        quality = int(_ab_get("thumbnail_quality") or 85)
        with _INCR_LOCK:
            # 1) miniature
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
            # 2) entree (meta fournie par l'appelant -> zero relecture disque)
            m = meta if isinstance(meta, dict) else (_read_image_meta(ap) or {})
            entry = _entry_for(rel, thumb_rel, ap, m)
            # 3) manifest du jour: remplace l'entree existante, plus recent en tete
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
            # 4) days.json (compte du jour) — pas de rescan, on lit l'index existant
            _bump_days_index(d, day, len(imgs))
        return True
    except Exception as e:
        _dbg(f"on_image_saved failed for {image_path}: {e}")
        return False


def _bump_days_index(out_dir, day, count):
    """Met a jour le compte d'un jour dans _index/days.json sans rescanner le dossier."""
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
                tp, trel = _thumb_paths(out_dir, kind + "/" + os.path.splitext(rel)[0] + ".jpg")
                jobs.append((prev, tp))
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
                "reco": civ.get("recommended") or {},
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
            jobs.append((p, _thumb_paths(d, os.path.splitext(rel)[0] + ".jpg")[0]))
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
            jobs.append((prev, _thumb_paths(d, kind + "/" + os.path.splitext(rel)[0] + ".jpg")[0]))
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
                      _thumb_paths(d, os.path.splitext(rel)[0] + ".jpg")[0]):
            if os.path.isfile(extra):
                os.remove(extra)
        _log(f"asset deleted: {rel}")
        return "deleted"
    except Exception as e:
        return f"error: {e}"
