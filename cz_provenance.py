"""crispz - provenance IA (EU AI Act art. 50) : lecture et marquage.

Deux briques, toutes deux OPTIONNELLES (degradation propre, pattern rembg):
  - c2pa-python : LECTURE des manifestes C2PA / Content Credentials embarques
    (Firefly, ChatGPT/DALL-E, Gemini... signent leurs sorties ainsi).
  - trustmark   : watermark invisible pixel-level (Adobe, open source).
    Ecriture a la sauvegarde (si provenance_watermark=on) + decodage a la
    demande dans PNG Info. Payload utile ~9 caracteres ASCII (ECC actif).

Tout tourne sur CPU (device='cpu' force) : le GPU est reserve aux renders.
Le modele TrustMark (~40 MB, telecharge au 1er usage dans site-packages)
charge en ~4 s puis encode/decode en ~0.1 s par image.

L'ABSENCE de marque ne prouve rien (image d'un autre outil, metadonnees
strippees, watermark retire) : l'UI ne doit jamais afficher "authentique".
"""

import importlib.util
import json
import os

from cz_core import CONFIG, _dbg

C2PA_AVAILABLE = importlib.util.find_spec("c2pa") is not None
TRUSTMARK_AVAILABLE = importlib.util.find_spec("trustmark") is not None

# TrustMark Q + ECC: ~68 bits utiles -> 9 caracteres ASCII max (tronque au-dela).
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
    """Identifiant embarque dans le watermark (config provenance_wm_id),
    tronque a WM_MAX_CHARS caracteres ASCII."""
    ident = str(CONFIG.get("provenance_wm_id", "crispzAI") or "crispzAI")
    ident = ident.encode("ascii", "ignore").decode("ascii")[:WM_MAX_CHARS]
    return ident or "crispzAI"


def wm_enabled():
    return TRUSTMARK_AVAILABLE and str(
        CONFIG.get("provenance_watermark", "off")).lower() in ("on", "true", "1", "yes")


def wm_apply(img):
    """Applique le watermark invisible (RGB, meme taille). Renvoie l'image
    inchangee si trustmark absent ou en cas d'erreur (la sauvegarde ne doit
    jamais echouer a cause de la provenance)."""
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
    """Decode le watermark TrustMark. Renvoie (present: bool, secret: str).
    (False, '') si trustmark absent, image illisible ou pas de watermark."""
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
    """Lit le manifeste C2PA embarque. Renvoie un dict {generator, issuer,
    when, state} ou None (pas de manifeste / c2pa absent / format non gere)."""
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
        return None  # pas de manifeste (cas normal) ou lecture impossible


def provenance_markdown(path, check_wm=False):
    """Section 'Provenance' pour PNG Info (markdown). check_wm=True ajoute le
    decodage TrustMark (~4s au 1er appel, ~0.1s ensuite)."""
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
