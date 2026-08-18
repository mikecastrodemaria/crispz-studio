"""crispz-studio - CivitAI enrichment for the Asset Browser (previews / trigger words /
examples), inspired by Fooocus2026's civitai_api + model_indexer.

Flow (per .safetensors):
  1. Get its SHA256 (from the sibling '<stem>.metadata.json' if present -> no hashing of
     multi-GB files; otherwise compute it once).
  2. GET /model-versions/by-hash/<sha> -> trainedWords + modelVersionId + names.
  3. GET /images?modelVersionId=... -> top images (url + generation meta).
  4. Download the first image -> save '<stem>.preview.png' (the convention our Asset
     Browser already scans) and write '<stem>.civitai.json' (trainedWords + examples).

Network is only hit when the user explicitly triggers a fetch (button in the Asset
Browser). An optional CivitAI API key (config 'civitai_api_key') is passed as a token.
"""

import os
import io
import re
import json
import hashlib
import urllib.request
import urllib.parse
import urllib.error

from cz_core import _log, _dbg, CONFIG, _prefs

CIVITAI_API = "https://civitai.com/api/v1"
_UA = "crispz-studio/asset-browser"

# Cle API CivitAI (optionnelle: previews gated/NSFW + anti rate-limit). Source: UI
# (preferences.json) -> config.txt. Reglable a chaud via set_api_key().
API_KEY = (str(_prefs.get("civitai_api_key") or CONFIG.get("civitai_api_key") or "").strip() or None)


def set_api_key(k):
    global API_KEY
    API_KEY = (str(k or "").strip() or None)


def _api_get(endpoint, params=None, api_key=None, timeout=20):
    """GET sur l'API CivitAI. api_key=None -> on retombe sur la cle GLOBALE (UI/prefs/
    config): sinon les appels internes (versions, images) partaient anonymes et rataient
    les contenus gates/NSFW."""
    params = dict(params or {})
    key = api_key or API_KEY
    if key:
        params["token"] = key
    url = CIVITAI_API + endpoint
    if params:
        url += "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # Visible par defaut: 401/403 (cle absente/invalide) et 429 (rate limit) sont
        # exactement ce qu'on veut voir en batch, pas noyer dans le debug.
        body = ""
        try:
            body = e.read().decode("utf-8", errors="ignore")[:160]
        except Exception:
            pass
        _log(f"civitai GET {endpoint} -> HTTP {e.code} {e.reason}"
             + (f" | {body}" if body else "")
             + ("  (no API key set: gated/NSFW content is hidden)" if not key and e.code in (401, 403) else ""))
        return None
    except Exception as e:
        _dbg(f"civitai GET {endpoint} failed: {e}")
        return None


def _sidecar_sha256(safepath):
    """SHA256 (64 hex) lu depuis '<stem>.metadata.json' si present, sinon None."""
    mp = os.path.splitext(safepath)[0] + ".metadata.json"
    try:
        if os.path.isfile(mp):
            with open(mp, encoding="utf-8") as f:
                h = str((json.load(f) or {}).get("sha256") or "").strip()
            if len(h) == 64:
                return h.lower()
    except Exception:
        pass
    return None


def _compute_sha256(safepath, progress=None):
    """SHA256 en streaming. Rapporte un % REEL via progress('hash', frac, texte) — c'est
    la seule phase potentiellement longue (fichiers multi-Go sans sidecar)."""
    h = hashlib.sha256()
    try:
        total = os.path.getsize(safepath)
    except Exception:
        total = 0
    done = 0
    with open(safepath, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            done += len(chunk)
            if progress and total:
                pct = done / total
                progress("hash", pct, f"Hashing model file… {int(pct * 100)}%")
    return h.hexdigest()


def _safe_size(p):
    try:
        return os.path.getsize(p)
    except Exception:
        return -1


def _cached_sha256(safepath):
    """SHA256 mis en cache par nos soins dans '<stem>.civitai.json'. Invalide si la taille
    du fichier a change (modele re-telecharge / autre version) -> recalcul."""
    sc = load_civitai_sidecar(safepath)
    sha = str(sc.get("sha256") or "").strip().lower()
    if len(sha) != 64:
        return None
    try:
        if int(sc.get("sha256_size") or -1) != os.path.getsize(safepath):
            _dbg(f"sha256 cache stale (size changed): {os.path.basename(safepath)}")
            return None
    except Exception:
        return None
    return sha


def _cache_sha256(safepath, sha):
    """Persiste le SHA256 dans '<stem>.civitai.json' (fusion, on ne perd rien d'existant).
    Sans ca, chaque passe re-lisait TOUT le fichier (des centaines de Go sur une grosse
    bibliotheque) juste pour retrouver le meme hash. Ecriture atomique (tmp + replace)."""
    p = os.path.splitext(safepath)[0] + ".civitai.json"
    try:
        sc = load_civitai_sidecar(safepath)
        sc["sha256"] = sha
        sc["sha256_size"] = os.path.getsize(safepath)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(sc, f, ensure_ascii=False, indent=2)
        os.replace(tmp, p)
    except Exception as e:
        _dbg(f"sha256 cache write failed {safepath}: {e}")


def model_sha256(safepath, allow_compute=True, progress=None):
    """SHA256 du modele. Ordre: sidecar '<stem>.metadata.json' (convention externe) ->
    notre cache '<stem>.civitai.json' -> calcul (puis mise en cache)."""
    sha = _sidecar_sha256(safepath) or _cached_sha256(safepath)
    if sha:
        return sha
    if allow_compute:
        try:
            sha = _compute_sha256(safepath, progress=progress)
            if sha:
                _cache_sha256(safepath, sha)   # meme si le modele est inconnu de CivitAI
            return sha
        except Exception as e:
            _dbg(f"sha256 compute failed {safepath}: {e}")
    return None


def get_version_by_hash(sha, api_key=None):
    data = _api_get(f"/model-versions/by-hash/{sha}", api_key=api_key)
    if not data or "id" not in data:
        return None
    triggers = [str(w).strip() for w in (data.get("trainedWords") or []) if str(w).strip()]
    return {
        "modelId": data.get("modelId"),
        "versionId": data.get("id"),
        "modelName": (data.get("model") or {}).get("name") or data.get("name") or "Unknown",
        "baseModel": data.get("baseModel") or "",
        "trainedWords": triggers,
        # Images vitrine de la version: contrairement a l'endpoint /images, celles-ci
        # portent un 'meta' REMPLI (prompt, steps, cfg...) + les drapeaux hasMeta /
        # hasPositivePrompt. Deja dans cette reponse -> zero requete supplementaire.
        "images": data.get("images") or [],
    }


def _norm_base(s):
    """'Z-Image', 'Z Image', 'zimage' -> 'zimage'. Les libelles de modele de base CivitAI
    varient en casse/espaces/tirets d'une version a l'autre -> comparaison tolerante."""
    return re.sub(r"[^a-z0-9]+", "", str(s or "").lower())


def get_latest_version(model_id, api_key=None, base_model=None, current_version_id=None):
    """Derniere version publiee d'un modele CivitAI: {id, name, baseModel} ou None.
    GET /models/<id> -> modelVersions[0] est la plus recente (l'API les trie du plus recent
    au plus ancien).

    base_model (ex. 'Z-Image') restreint la recherche aux versions du MEME modele de base.
    Beaucoup de pages CivitAI publient la suite d'un LoRA pour une AUTRE base (Krea2, Flux,
    SDXL...): ce n'est pas une mise a jour de notre fichier, qui ne tournerait pas dessus.
    Aucune version de la meme base -> None (pas d'update). Si l'API ne renseigne le
    baseModel nulle part, on ne filtre pas: l'info est indisponible, pas contradictoire.
    base_model inconnue (vieux sidecar) -> deduite de current_version_id dans la reponse."""
    if not model_id:
        return None
    data = _api_get(f"/models/{model_id}", api_key=api_key)
    vers = [v for v in ((data or {}).get("modelVersions") or []) if isinstance(v, dict)]
    want = _norm_base(base_model)
    if not want and current_version_id is not None:
        want = _norm_base(next((v.get("baseModel") for v in vers
                                if v.get("id") == current_version_id), None))
    if want and any(_norm_base(v.get("baseModel")) for v in vers):
        vers = [v for v in vers if _norm_base(v.get("baseModel")) == want]
    if not vers:
        return None
    v = vers[0]
    return {"id": v.get("id"), "name": str(v.get("name") or "").strip(),
            "baseModel": str(v.get("baseModel") or "").strip()}


def _update_fields(model_id, current_version_id, api_key=None, base_model=None):
    """Compare la version locale a la derniere sur CivitAI *pour le meme modele de base*
    (base_model, cf. get_latest_version). Renvoie un dict a fusionner dans le sidecar:
    {update_available, latest_versionId, latest_versionName}. Silencieux en cas d'echec
    (network/inconnu) -> pas de faux positif."""
    try:
        latest = get_latest_version(model_id, api_key, base_model=base_model,
                                    current_version_id=current_version_id)
    except Exception as e:
        _dbg(f"latest-version check failed for model {model_id}: {e}")
        latest = None
    if not latest or latest.get("id") is None or current_version_id is None:
        return {"update_available": False, "latest_versionId": None, "latest_versionName": ""}
    newer = latest["id"] != current_version_id
    return {"update_available": bool(newer), "latest_versionId": latest["id"],
            "latest_versionName": latest.get("name") or ""}


def get_top_images(version_id, api_key=None, limit=8):
    """Images communautaires d'une version (FALLBACK). Attention: cet endpoint renvoie
    'meta': null (CivitAI ne publie plus les parametres de generation ici) -> pas de
    prompt. Les images de get_version_by_hash()['images'] sont a preferer."""
    data = _api_get("/images", {"modelVersionId": version_id, "sort": "Most Reactions",
                                "limit": int(limit)}, api_key=api_key)
    return (data or {}).get("items") or []


def _examples_from(imgs, limit=8):
    """Normalise des images CivitAI en exemples {url, prompt, width, height, has_prompt}.
    'meta' peut etre None (parametres non publies) -> prompt vide + has_prompt=False, ce
    qui permet a l'UI de dire 'non publie' au lieu de laisser croire a un bug."""
    out = []
    for it in imgs[:limit]:
        if not isinstance(it, dict) or not it.get("url"):
            continue
        meta = it.get("meta") or {}
        prompt = str(meta.get("prompt") or "").strip()
        out.append({
            "url": it["url"], "prompt": prompt[:2000],
            "width": it.get("width"), "height": it.get("height"),
            "has_prompt": bool(prompt),
        })
    return out


def analyze_settings(imgs, min_meta=2):
    """Consensus des reglages communautaires (technique Fooocus2026): a partir des 'meta'
    des images d'exemple (sampler, cfgScale, steps, Size), renvoie
      {steps, guidance, sampler, size, n} (mediane pour steps/CFG, majorite pour le reste)
    ou {} si moins de min_meta images publient leurs parametres."""
    samplers, cfgs, steps, sizes = [], [], [], []
    for it in imgs or []:
        meta = (it or {}).get("meta") or {}
        if not isinstance(meta, dict) or not meta:
            continue
        s = str(meta.get("sampler") or "").strip()
        if s:
            samplers.append(s)
        try:
            if meta.get("cfgScale") is not None:
                cfgs.append(float(meta["cfgScale"]))
        except (TypeError, ValueError):
            pass
        try:
            if meta.get("steps") is not None:
                steps.append(int(meta["steps"]))
        except (TypeError, ValueError):
            pass
        sz = str(meta.get("Size") or meta.get("size") or "").strip()
        if sz and "x" in sz:
            sizes.append(sz)
    n = max(len(cfgs), len(steps), len(samplers))
    if n < min_meta:
        return {}

    def _median(vals):
        v = sorted(vals)
        return v[len(v) // 2] if v else None

    def _majority(vals):
        return max(set(vals), key=vals.count) if vals else None

    out = {"n": n}
    if steps:
        out["steps"] = int(_median(steps))
    if cfgs:
        out["guidance"] = round(float(_median(cfgs)), 1)
    if samplers:
        out["sampler"] = _majority(samplers)
    if sizes:
        out["size"] = _majority(sizes)
    return out


def map_sampler_name(name):
    """Mappe un nom de sampler CivitAI/A1111 vers (sampler crispz, schedule crispz).
    Conservateur: renvoie (None, None) pour les familles sans equivalent (DPM++ etc.),
    l'appelant garde alors le sampler courant et n'applique que steps/CFG."""
    n = str(name or "").strip().lower()
    if not n:
        return None, None
    sched = None
    if "karras" in n:
        sched = "karras"
    elif "exponential" in n:
        sched = "exponential"
    elif "beta" in n:
        sched = "beta"
    elif "simple" in n or "normal" in n or "sgm" in n:
        sched = "sgm_uniform"
    samp = None
    if n.startswith("euler"):
        samp = "euler"          # 'Euler a' -> euler (le plus proche chez Z-Image)
    elif "unipc" in n or n.startswith("uni"):
        samp = "unipc"
    elif "lcm" in n:
        samp = "lcm"
    return samp, sched


def _download(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def search_loras(query, limit=10, api_key=None, types="LORA", base_model=None):
    """Recherche CivitAI par NOM: GET /models?query=...&types=LORA. Renvoie une liste de
    candidats PLATS, une entree par VERSION du modele (les versions d'une meme page
    CivitAI visent souvent des bases differentes: Z-Image, Flux, SDXL...). Champs:
      {modelId, modelName, creator, nsfw, versionId, versionName, baseModel,
       fileName, sizeKB, downloadUrl, sha256, previewUrl, url}
    base_model (ex. 'Z-Image') remonte les versions de cette base EN TETE sans exclure
    les autres (tri stable) — l'appelant filtre s'il veut du strict. [] si echec reseau
    ou aucun resultat (jamais d'exception: l'UI affiche 'no result')."""
    q = str(query or "").strip()
    if not q:
        return []
    data = _api_get("/models", {"query": q, "types": types, "limit": int(limit),
                                "sort": "Highest Rated"}, api_key=api_key)
    out = []
    for m in (data or {}).get("items") or []:
        if not isinstance(m, dict) or m.get("id") is None:
            continue
        for v in m.get("modelVersions") or []:
            if not isinstance(v, dict) or v.get("id") is None:
                continue
            files = [f for f in (v.get("files") or []) if isinstance(f, dict)]
            f = next((x for x in files if x.get("primary")), files[0] if files else {})
            out.append({
                "modelId": m.get("id"),
                "modelName": str(m.get("name") or "").strip(),
                "creator": str((m.get("creator") or {}).get("username") or "").strip(),
                "nsfw": bool(m.get("nsfw")),
                "versionId": v.get("id"),
                "versionName": str(v.get("name") or "").strip(),
                "baseModel": str(v.get("baseModel") or "").strip(),
                "fileName": str(f.get("name") or "").strip(),
                "sizeKB": float(f.get("sizeKB") or 0),
                "downloadUrl": str(f.get("downloadUrl") or v.get("downloadUrl") or "").strip(),
                "sha256": str((f.get("hashes") or {}).get("SHA256") or "").strip().lower(),
                "previewUrl": next((i.get("url") for i in (v.get("images") or [])
                                    if isinstance(i, dict) and i.get("url")), ""),
                "url": f"https://civitai.com/models/{m.get('id')}",
            })
    if base_model:
        want = _norm_base(base_model)
        out.sort(key=lambda e: 0 if _norm_base(e["baseModel"]) == want else 1)
    return out


def download_model_file(cand, dest_dir, api_key=None, progress=None):
    """Telecharge le fichier d'un candidat search_loras() dans dest_dir (stream 1 Mo +
    progress('download', frac, texte) avec % REEL si la taille est connue). Le SHA256 est
    calcule PENDANT le streaming et compare a celui annonce par CivitAI: mismatch ->
    fichier supprime + echec propre (pas de LoRA corrompue silencieuse). Ecrit vers un
    '.part' puis renomme (jamais de fichier partiel visible), met le hash en cache dans
    '<stem>.civitai.json' et enrichit preview + trigger words (best effort).
    Renvoie {success, message, path}. Jamais d'exception vers l'appelant."""
    def _p(frac, text):
        if progress:
            try:
                progress("download", frac, text)
            except Exception:
                pass
    try:
        cand = cand or {}
        url = str(cand.get("downloadUrl") or "").strip()
        if not url and cand.get("versionId"):
            url = f"https://civitai.com/api/download/models/{cand['versionId']}"
        if not url:
            return {"success": False, "message": "no download URL for this version", "path": ""}
        key = api_key or API_KEY
        if key:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode({"token": key})
        fname = os.path.basename(str(cand.get("fileName") or "").strip().replace("\\", "/"))
        if not fname:
            fname = f"civitai_{cand.get('versionId') or 'model'}.safetensors"
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, fname)
        if os.path.isfile(dest):
            return {"success": True, "message": f"{fname} already exists (not overwritten)",
                    "path": dest}
        expected = str(cand.get("sha256") or "").strip().lower()
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        h = hashlib.sha256()
        done = 0
        tmp = dest + ".part"
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                total = int(r.headers.get("Content-Length") or 0) \
                    or int(float(cand.get("sizeKB") or 0) * 1024)
                with open(tmp, "wb") as f:
                    for chunk in iter(lambda: r.read(1 << 20), b""):
                        f.write(chunk)
                        h.update(chunk)
                        done += len(chunk)
                        if total:
                            frac = min(1.0, done / total)
                            _p(frac, f"Downloading {fname}… {int(frac * 100)}% "
                                     f"({done / 1024**2:.0f} MB)")
                        else:
                            _p(None, f"Downloading {fname}… {done / 1024**2:.0f} MB")
        except urllib.error.HTTPError as e:
            _try_remove(tmp)
            hint = (" (this file may require a CivitAI API key — set one in Advanced)"
                    if e.code in (401, 403) and not key else "")
            return {"success": False, "message": f"download failed: HTTP {e.code} {e.reason}{hint}",
                    "path": ""}
        except Exception as e:
            _try_remove(tmp)
            return {"success": False, "message": f"download failed: {e}", "path": ""}
        sha = h.hexdigest().lower()
        if expected and len(expected) == 64 and sha != expected:
            _try_remove(tmp)
            return {"success": False,
                    "message": f"SHA256 mismatch for {fname} (corrupted download, file removed)",
                    "path": ""}
        os.replace(tmp, dest)
        _cache_sha256(dest, sha)
        _log(f"civitai download: {fname} ({done / 1024**2:.0f} MB) -> {dest_dir}")
        # Enrichissement (preview + trigger words): le hash est deja en cache -> aucune
        # relecture du fichier. Best effort: un echec reseau ne gache pas le download.
        try:
            _p(None, "Fetching preview + trigger words…")
            fetch_civitai_for_model(dest, api_key=api_key, check_update=False)
        except Exception as e:
            _dbg(f"post-download enrich failed: {e}")
        return {"success": True, "message": f"{fname} downloaded ({done / 1024**2:.0f} MB, "
                                            f"SHA256 {'verified' if expected else 'recorded'})",
                "path": dest}
    except Exception as e:
        _dbg(f"download_model_file failed: {e}")
        return {"success": False, "message": f"download failed: {e}", "path": ""}


def _try_remove(path):
    try:
        if os.path.isfile(path):
            os.remove(path)
    except Exception:
        pass


def has_preview(safepath):
    stem = os.path.splitext(safepath)[0]
    return any(os.path.isfile(stem + e) for e in
               (".preview.png", ".preview.jpg", ".preview.jpeg", ".preview.webp"))


def load_civitai_sidecar(safepath):
    """Renvoie le dict '<stem>.civitai.json' (trainedWords + examples) ou {}."""
    p = os.path.splitext(safepath)[0] + ".civitai.json"
    try:
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f) or {}
    except Exception:
        pass
    return {}


def fetch_civitai_for_model(safepath, api_key=None, overwrite=False, progress=None,
                            check_update=True):
    """Enrichit un .safetensors depuis CivitAI: ecrit '<stem>.preview.png' (si absent) et
    '<stem>.civitai.json' (trainedWords + examples + drapeau nouvelle version). Renvoie
    {success, message, triggers, update_available}.

    progress(phase, frac, text) est appele a chaque etape (phase: hash|query|images|
    download). frac est un % reel pour 'hash' seulement (sinon None -> barre indeterminee)."""
    def _p(phase, frac, text):
        if progress:
            try:
                progress(phase, frac, text)
            except Exception:
                pass
    if not safepath or not os.path.isfile(safepath):
        return {"success": False, "message": "model file not found"}
    api_key = api_key or API_KEY
    stem = os.path.splitext(safepath)[0]
    if has_preview(safepath) and not overwrite:
        # On rafraichit quand meme les infos (triggers/examples), sans re-telecharger.
        want_preview = False
    else:
        want_preview = True
    _p("hash", None, "Reading model hash…")
    sha = model_sha256(safepath, progress=progress)
    if not sha:
        return {"success": False, "message": "no SHA256 (metadata.json missing + hashing failed)"}
    _p("query", None, "Querying CivitAI…")
    ver = get_version_by_hash(sha, api_key)
    if not ver:
        return {"success": False, "message": "not found on CivitAI (unknown hash)"}
    _p("images", None, "Fetching example images…")
    # Source 1 (gratuite, AVEC les prompts): les images de la reponse by-hash.
    imgs = ver.get("images") or []
    if not imgs and ver.get("versionId"):
        # Source 2 (fallback): endpoint /images -- images communautaires, sans prompt.
        imgs = get_top_images(ver["versionId"], api_key, limit=8)
    saved_preview = False
    if want_preview:
        url = next((it.get("url") for it in imgs if isinstance(it, dict) and it.get("url")), None)
        if url:
            try:
                from PIL import Image
                _p("download", None, "Downloading preview…")
                im = Image.open(io.BytesIO(_download(url))).convert("RGB")
                im.save(stem + ".preview.png", "PNG", optimize=True)
                saved_preview = True
            except Exception as e:
                _dbg(f"civitai preview save failed: {e}")
    examples = _examples_from(imgs)
    # Fusion (et non remplacement): le sidecar porte aussi notre cache de hash
    # (sha256/sha256_size) -- l'ecraser reprovoquerait un re-hash complet au run suivant.
    sidecar = load_civitai_sidecar(safepath)
    sidecar.update({
        "modelName": ver.get("modelName"), "modelId": ver.get("modelId"),
        "versionId": ver.get("versionId"), "baseModel": ver.get("baseModel"),
        "trainedWords": ver.get("trainedWords") or [], "examples": examples,
        "recommended": analyze_settings(imgs),
        "url": f"https://civitai.com/models/{ver.get('modelId')}" if ver.get("modelId") else "",
    })
    sidecar.setdefault("sha256", sha)
    sidecar.setdefault("sha256_size", _safe_size(safepath))
    upd = {"update_available": False, "latest_versionId": None, "latest_versionName": ""}
    if check_update:
        _p("update", None, "Checking for a newer version…")
        upd = _update_fields(ver.get("modelId"), ver.get("versionId"), api_key,
                             base_model=ver.get("baseModel"))
    sidecar.update(upd)
    try:
        tmp = stem + ".civitai.json.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(sidecar, f, ensure_ascii=False, indent=2)
        os.replace(tmp, stem + ".civitai.json")
    except Exception as e:
        _dbg(f"civitai.json write failed: {e}")
    n_prompt = sum(1 for e in examples if e.get("has_prompt"))
    msg = f"CivitAI: {ver.get('modelName')} — {len(examples)} example(s)"
    if examples:
        msg += f" ({n_prompt} with prompt)"
    if saved_preview:
        msg += " + preview"
    if upd.get("update_available"):
        msg += f" ⚠ newer version: {upd.get('latest_versionName') or '?'}"
    _log(f"civitai fetch: {os.path.basename(safepath)} -> {msg}")
    return {"success": True, "message": msg, "triggers": ver.get("trainedWords") or [],
            "update_available": bool(upd.get("update_available"))}


def refresh_update_flag(safepath, api_key=None):
    """Rafraichit UNIQUEMENT le drapeau 'nouvelle version' d'un modele deja enrichi (lit le
    sidecar existant, compare a CivitAI, reecrit). Pas de re-telechargement de preview.
    Renvoie {success, update_available}. Utilise par le batch pour les fichiers deja faits."""
    sc = load_civitai_sidecar(safepath)
    if not sc or sc.get("modelId") is None or sc.get("versionId") is None:
        return {"success": False, "update_available": False}
    upd = _update_fields(sc.get("modelId"), sc.get("versionId"), api_key,
                         base_model=sc.get("baseModel"))
    sc.update(upd)
    try:
        p = os.path.splitext(safepath)[0] + ".civitai.json"
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            json.dump(sc, f, ensure_ascii=False, indent=2)
        os.replace(p + ".tmp", p)
    except Exception as e:
        _dbg(f"civitai.json update-flag write failed: {e}")
        return {"success": False, "update_available": bool(upd.get("update_available"))}
    return {"success": True, "update_available": bool(upd.get("update_available"))}
