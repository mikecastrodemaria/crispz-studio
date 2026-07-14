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
import json
import hashlib
import urllib.request
import urllib.parse

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
    params = dict(params or {})
    if api_key:
        params["token"] = api_key
    url = CIVITAI_API + endpoint
    if params:
        url += "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
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


def model_sha256(safepath, allow_compute=True, progress=None):
    sha = _sidecar_sha256(safepath)
    if sha:
        return sha
    if allow_compute:
        try:
            return _compute_sha256(safepath, progress=progress)
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
    }


def get_latest_version(model_id, api_key=None):
    """Derniere version publiee d'un modele CivitAI: {id, name} ou None. GET /models/<id>
    -> modelVersions[0] est la plus recente (l'API les trie du plus recent au plus ancien)."""
    if not model_id:
        return None
    data = _api_get(f"/models/{model_id}", api_key=api_key)
    vers = (data or {}).get("modelVersions") or []
    if not vers or not isinstance(vers[0], dict):
        return None
    v = vers[0]
    return {"id": v.get("id"), "name": str(v.get("name") or "").strip()}


def _update_fields(model_id, current_version_id, api_key=None):
    """Compare la version locale a la derniere sur CivitAI. Renvoie un dict a fusionner
    dans le sidecar: {update_available, latest_versionId, latest_versionName}. Silencieux
    en cas d'echec (network/inconnu) -> pas de faux positif."""
    try:
        latest = get_latest_version(model_id, api_key)
    except Exception as e:
        _dbg(f"latest-version check failed for model {model_id}: {e}")
        latest = None
    if not latest or latest.get("id") is None or current_version_id is None:
        return {"update_available": False, "latest_versionId": None, "latest_versionName": ""}
    newer = latest["id"] != current_version_id
    return {"update_available": bool(newer), "latest_versionId": latest["id"],
            "latest_versionName": latest.get("name") or ""}


def get_top_images(version_id, api_key=None, limit=8):
    data = _api_get("/images", {"modelVersionId": version_id, "sort": "Most Reactions",
                                "limit": int(limit)}, api_key=api_key)
    return (data or {}).get("items") or []


def _download(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


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
    imgs = get_top_images(ver["versionId"], api_key, limit=8) if ver.get("versionId") else []
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
    examples = []
    for it in imgs[:8]:
        if isinstance(it, dict) and it.get("url"):
            meta = it.get("meta") or {}
            examples.append({"url": it["url"], "prompt": str(meta.get("prompt") or "")[:500],
                             "width": it.get("width"), "height": it.get("height")})
    sidecar = {
        "modelName": ver.get("modelName"), "modelId": ver.get("modelId"),
        "versionId": ver.get("versionId"), "baseModel": ver.get("baseModel"),
        "trainedWords": ver.get("trainedWords") or [], "examples": examples,
        "url": f"https://civitai.com/models/{ver.get('modelId')}" if ver.get("modelId") else "",
    }
    upd = {"update_available": False, "latest_versionId": None, "latest_versionName": ""}
    if check_update:
        _p("update", None, "Checking for a newer version…")
        upd = _update_fields(ver.get("modelId"), ver.get("versionId"), api_key)
    sidecar.update(upd)
    try:
        with open(stem + ".civitai.json", "w", encoding="utf-8") as f:
            json.dump(sidecar, f, ensure_ascii=False, indent=2)
    except Exception as e:
        _dbg(f"civitai.json write failed: {e}")
    msg = f"CivitAI: {ver.get('modelName')} — {len(examples)} example(s)"
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
    upd = _update_fields(sc.get("modelId"), sc.get("versionId"), api_key)
    sc.update(upd)
    try:
        with open(os.path.splitext(safepath)[0] + ".civitai.json", "w", encoding="utf-8") as f:
            json.dump(sc, f, ensure_ascii=False, indent=2)
    except Exception as e:
        _dbg(f"civitai.json update-flag write failed: {e}")
        return {"success": False, "update_available": bool(upd.get("update_available"))}
    return {"success": True, "update_available": bool(upd.get("update_available"))}
