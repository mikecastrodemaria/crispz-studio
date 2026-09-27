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

# The CivitAI API key (optional: gated/NSFW previews + anti rate-limit). The source: UI
# (preferences.json) -> config.txt. Settable hot through set_api_key().
API_KEY = (str(_prefs.get("civitai_api_key") or CONFIG.get("civitai_api_key") or "").strip() or None)


def set_api_key(k):
    global API_KEY
    API_KEY = (str(k or "").strip() or None)


def _api_get(endpoint, params=None, api_key=None, timeout=20):
    """A GET on the CivitAI API. api_key=None -> we fall back on the GLOBAL key (UI/prefs/
    config): otherwise the internal calls (versions, images) went out anonymous and missed
    the gated/NSFW contents."""
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
        # Visible by default: 401/403 (a missing/invalid key) and 429 (the rate limit) are
        # exactly what one wants to see in a batch, not drowned in the debug output.
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
    """The SHA256 (64 hex) read from '<stem>.metadata.json' when present, otherwise None."""
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
    """A streaming SHA256. It reports a REAL % through progress('hash', frac, text) — that
    is the only potentially long phase (multi-GB files with no sidecar)."""
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
    """A SHA256 cached by us in '<stem>.civitai.json'. Invalidated when the file's size
    has changed (a model re-downloaded / another version) -> recomputed."""
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
    """Persists the SHA256 in '<stem>.civitai.json' (a merge, nothing existing is lost).
    Without it, every pass re-read the WHOLE file (hundreds of GB on a big library) just to
    find the same hash again. An atomic write (tmp + replace)."""
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
    """The model's SHA256. The order: the '<stem>.metadata.json' sidecar (an external
    convention) -> our '<stem>.civitai.json' cache -> a computation (then cached)."""
    sha = _sidecar_sha256(safepath) or _cached_sha256(safepath)
    if sha:
        return sha
    if allow_compute:
        try:
            sha = _compute_sha256(safepath, progress=progress)
            if sha:
                _cache_sha256(safepath, sha)   # even when the model is unknown to CivitAI
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
        # The version's showcase images: unlike the /images endpoint, these carry a
        # FILLED 'meta' (prompt, steps, cfg...) + the hasMeta / hasPositivePrompt flags.
        # Already in this answer -> zero extra request.
        "images": data.get("images") or [],
    }


def _norm_base(s):
    """'Z-Image', 'Z Image', 'zimage' -> 'zimage'. The CivitAI base model labels vary in
    case/spaces/dashes from one version to the next -> a tolerant comparison."""
    return re.sub(r"[^a-z0-9]+", "", str(s or "").lower())


def get_latest_version(model_id, api_key=None, base_model=None, current_version_id=None):
    """The latest published version of a CivitAI model: {id, name, baseModel} or None.
    GET /models/<id> -> modelVersions[0] is the most recent one (the API sorts them from the
    most recent to the oldest).

    base_model ('Z-Image', say) restricts the search to the versions of the SAME base model.
    Many CivitAI pages publish a LoRA's sequel for ANOTHER base (Krea2, Flux, SDXL...):
    that is not an update of our file, which would not run on it.
    No version of the same base -> None (no update). When the API states the baseModel
    nowhere, we do not filter: the information is unavailable, not contradictory.
    An unknown base_model (an old sidecar) -> deduced from current_version_id in the answer."""
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
    """Compares the local version with the latest one on CivitAI *for the same base model*
    (base_model, see get_latest_version). Returns a dict to merge into the sidecar:
    {update_available, latest_versionId, latest_versionName}. Silent on a failure
    (network/unknown) -> no false positive."""
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
    """A version's community images (a FALLBACK). Careful: that endpoint returns
    'meta': null (CivitAI does not publish the generation parameters there any more) -> no
    prompt. The images from get_version_by_hash()['images'] are to be preferred."""
    data = _api_get("/images", {"modelVersionId": version_id, "sort": "Most Reactions",
                                "limit": int(limit)}, api_key=api_key)
    return (data or {}).get("items") or []


def _examples_from(imgs, limit=8):
    """Normalises CivitAI images into examples {url, prompt, width, height, has_prompt}.
    'meta' can be None (the parameters are not published) -> an empty prompt +
    has_prompt=False, which lets the UI say 'not published' instead of letting one believe
    in a bug."""
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
    """The consensus of the community settings (the Fooocus2026 technique): from the 'meta'
    of the example images (sampler, cfgScale, steps, Size), it returns
      {steps, guidance, sampler, size, n} (the median for steps/CFG, the majority for the rest)
    or {} when fewer than min_meta images publish their parameters."""
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
    """Maps a CivitAI/A1111 sampler name to (a crispz sampler, a crispz schedule).
    Conservative: it returns (None, None) for the families with no equivalent (DPM++ etc.),
    and the caller then keeps the current sampler and only applies steps/CFG."""
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
        samp = "euler"          # 'Euler a' -> euler (the closest thing on Z-Image)
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
    """A CivitAI search by NAME: GET /models?query=...&types=LORA. It returns a FLAT list
    of candidates, one entry per model VERSION (the versions of a single CivitAI page often
    target different bases: Z-Image, Flux, SDXL...). The fields:
      {modelId, modelName, creator, nsfw, versionId, versionName, baseModel,
       fileName, sizeKB, downloadUrl, sha256, previewUrl, url}
    base_model ('Z-Image', say) brings the versions of that base UP FRONT without excluding
    the others (a stable sort) — the caller filters when it wants strictness. [] on a
    network failure or no result (never an exception: the UI displays 'no result')."""
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
    """Downloads the file of a search_loras() candidate into dest_dir (a 1 MB stream +
    progress('download', frac, text) with a REAL % when the size is known). The SHA256 is
    computed DURING the streaming and compared with the one CivitAI announced: a mismatch
    -> the file is deleted + a clean failure (no silently corrupted LoRA). It writes to a
    '.part' then renames (never a partial file visible), caches the hash in
    '<stem>.civitai.json' and enriches the preview + trigger words (best effort).
    Returns {success, message, path}. Never an exception towards the caller."""
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
        # The enrichment (the preview + the trigger words): the hash is cached already
        # -> no re-reading of the file. Best effort: a network failure does not spoil the
        # download.
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
    """Enriches a .safetensors from CivitAI: it writes '<stem>.preview.png' (when absent)
    and '<stem>.civitai.json' (trainedWords + examples + the new-version flag). Returns
    {success, message, triggers, update_available}.

    progress(phase, frac, text) is called at every step (phase: hash|query|images|
    download). frac is a real % for 'hash' only (None otherwise -> an indeterminate bar)."""
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
        # We refresh the info (triggers/examples) anyway, without re-downloading.
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
    # Source 1 (free, WITH the prompts): the images of the by-hash answer.
    imgs = ver.get("images") or []
    if not imgs and ver.get("versionId"):
        # Source 2 (a fallback): the /images endpoint -- community images, with no prompt.
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
    # A merge (and not a replacement): the sidecar also carries our hash cache
    # (sha256/sha256_size) -- overwriting it would trigger a full re-hash on the next run.
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
    """Refreshes ONLY the 'new version' flag of a model already enriched (it reads the
    existing sidecar, compares with CivitAI, rewrites). No preview is downloaded again.
    Returns {success, update_available}. Used by the batch for the files already done."""
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
