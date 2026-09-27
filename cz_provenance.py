"""crispz - AI provenance (EU AI Act art. 50): reading and marking.

Two bricks, both OPTIONAL (clean degradation, the rembg pattern):
  - c2pa-python : READS the embedded C2PA / Content Credentials manifests
    (Firefly, ChatGPT/DALL-E, Gemini... sign their outputs that way).
  - trustmark   : an invisible pixel-level watermark (Adobe, open source).
    Written at saving time (when provenance_watermark=on) + decoded on demand
    in PNG Info. A useful payload of ~9 ASCII characters (ECC active).

Everything runs on the CPU (device='cpu' forced): the GPU is reserved for the renders.
The TrustMark model (~40 MB, downloaded on the 1st use into site-packages)
loads in ~4 s then encodes/decodes in ~0.1 s per image.

The ABSENCE of a mark proves nothing (an image from another tool, stripped
metadata, a watermark removed): the UI must never display "authentic".

"""

import importlib.util
import json
import os

from cz_core import CONFIG, _dbg

C2PA_AVAILABLE = importlib.util.find_spec("c2pa") is not None
TRUSTMARK_AVAILABLE = importlib.util.find_spec("trustmark") is not None

# TrustMark Q + ECC: ~68 useful bits -> 9 ASCII characters max (truncated beyond).
WM_MAX_CHARS = 9

_TM = None  # singleton TrustMark (init ~4s, lazy)


def _tm():
    global _TM
    if _TM is None:
        from trustmark import TrustMark
        _TM = TrustMark(verbose=False, model_type="Q", device="cpu",
                        loadRemover=False)
    return _TM


def wm_id():
    """The identifier embedded in the watermark (the provenance_wm_id config),
    truncated to WM_MAX_CHARS ASCII characters."""
    ident = str(CONFIG.get("provenance_wm_id", "crispzAI") or "crispzAI")
    ident = ident.encode("ascii", "ignore").decode("ascii")[:WM_MAX_CHARS]
    return ident or "crispzAI"


def wm_enabled():
    return TRUSTMARK_AVAILABLE and str(
        CONFIG.get("provenance_watermark", "off")).lower() in ("on", "true", "1", "yes")


def wm_apply(img):
    """Applies the invisible watermark (RGB, the same size). Returns the image
    unchanged when trustmark is absent or on an error (saving must never fail
    because of the provenance)."""
    if not TRUSTMARK_AVAILABLE:
        return img
    try:
        alpha = img.getchannel("A") if img.mode == "RGBA" else None
        out = _tm().encode(img.convert("RGB"), wm_id())
        if alpha is not None:
            out.putalpha(alpha)
        return out
    except Exception as e:
        _dbg(f"provenance watermark skipped: {e}")
        return img


def wm_read(path_or_img):
    """Decodes the TrustMark watermark. Returns (present: bool, secret: str).
    (False, '') when trustmark is absent, the image is unreadable or there is no
    watermark."""
    if not TRUSTMARK_AVAILABLE:
        return False, ""
    try:
        from PIL import Image
        img = path_or_img
        if isinstance(path_or_img, str):
            with Image.open(path_or_img) as im:
                img = im.convert("RGB")
        secret, present, _schema = _tm().decode(img)
        return bool(present), (secret or "")
    except Exception as e:
        _dbg(f"provenance watermark decode failed: {e}")
        return False, ""


def read_c2pa(path):
    """Reads the embedded C2PA manifest. Returns a dict {generator, issuer,
    when, state} or None (no manifest / c2pa absent / an unhandled format)."""
    if not C2PA_AVAILABLE or not path or not os.path.isfile(path):
        return None
    try:
        import c2pa
        with c2pa.Reader(path) as r:
            data = json.loads(r.json())
            state = ""
            try:
                state = str(r.get_validation_state() or "")
            except Exception:
                pass
            active = data.get("manifests", {}).get(data.get("active_manifest", ""), {})
            sig = active.get("signature_info") or {}
            return {
                "generator": active.get("claim_generator", ""),
                "issuer": sig.get("issuer", ""),
                "when": sig.get("time", ""),
                "title": active.get("title", ""),
                "state": state,
            }
    except Exception:
        return None  # no manifest (the normal case) or it could not be read


def provenance_markdown(path, check_wm=False):
    """The 'Provenance' section for PNG Info (markdown). check_wm=True adds the
    TrustMark decoding (~4s on the 1st call, ~0.1s afterwards)."""
    lines = []
    c2 = read_c2pa(path)
    if c2:
        who = c2["issuer"] or c2["generator"] or "unknown"
        state = (c2["state"] or "").lower()
        if state == "valid":
            lines.append(f"✅ **C2PA manifest** — signed by **{who}**"
                         + (f" ({c2['when']})" if c2["when"] else "") + ", signature valid")
        elif state:
            lines.append(f"⚠️ **C2PA manifest** — {who}, state: **{state}** "
                         "(file may have been modified after signing)")
        else:
            lines.append(f"ℹ️ **C2PA manifest** found — {who}")
    elif C2PA_AVAILABLE:
        lines.append("No C2PA manifest.")
    else:
        lines.append("*C2PA check unavailable — `pip install c2pa-python`.*")
    if check_wm:
        if TRUSTMARK_AVAILABLE:
            present, secret = wm_read(path)
            if present:
                lines.append(f"✅ **Invisible watermark** (TrustMark) detected: `{secret}`")
            else:
                lines.append("No TrustMark watermark detected.")
        else:
            lines.append("*Watermark check unavailable — `pip install trustmark`.*")
    lines.append("*Absence of marks proves nothing: it never means "
                 "\"not AI\" or \"authentic\".*")
    return "**Provenance** — " + "  \n".join(lines)
