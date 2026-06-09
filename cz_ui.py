"""crispz-studio - interface Gradio (build_ui) + handlers UI + orchestration commune
(run / _editor_* / presets / timing) extraite de app.py (step 8).

Ce module branche tous les cz_* (core/pipeline/esrgan/face/prompt/ollama/imageio/
assets/assetbrowser). Il NE doit jamais importer app ni cz_cli (regle anti-circulaire:
l'UI importe le reste, le reste n'importe pas l'UI). La CLI/serveur vivent dans
cz_cli.py et importent ce module.
"""

import os
import sys
import gc
import io
import random
import threading
import warnings

# Masque les DeprecationWarning Gradio "pass theme/css/js to launch() instead":
# launch() ne les accepte PAS encore en Gradio 5.x (avertissement anticipe Gradio 6),
# donc on les garde dans Blocks(...) et on coupe juste le bruit au demarrage.
warnings.filterwarnings(
    "ignore", category=DeprecationWarning,
    message=r"The '(theme|css|js)' parameter in the Blocks constructor will be removed")
import base64
import glob
import time
import uuid
import datetime
import numpy as np
import torch
from PIL import Image, ImageChops
import gradio as gr

# Support AVIF/HEIC en entree (les .avif sinon: PIL.UnidentifiedImageError).
try:
    import pillow_avif  # noqa: F401  enregistre l'ouvreur AVIF dans PIL
except Exception:
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except Exception:
        pass


# _disable_brotli -> cz_cli.py (utilise seulement au lancement de l'UI).

# Fondation (config, chemins, defauts, logging, device) -> cz_core.py
import cz_core
from cz_core import (  # noqa: E402,F401
    HERE, PREFS_PATH, CONFIG_PATH, CONFIG_SAMPLE_PATH,
    DEFAULT_MODEL, DEFAULT_FACTOR, DEFAULT_DENOISE, DEFAULT_STEPS, DEFAULT_TILE,
    DEFAULT_OVERLAP, DEFAULT_REFINE_TILE, DEFAULT_REFINE_OVERLAP, DEFAULT_SAVE_MODE,
    DEFAULT_OUTPUT_DIR, DEFAULT_OUTPUT_FORMAT, SUPPORTED_FORMATS, IMG_EXTS,
    DEFAULT_BASE_REPO, DEFAULT_ESRGAN_DIR,
    CONFIG, MODEL_PROFILES, DEFAULT_MODEL_PROFILE, profile_for_model,
    DESCRIBE_INSTRUCTION, IMPROVE_INSTRUCTION, COMPOSE_INSTRUCTION,
    DEVICE, DTYPE, VERBOSE,
    _load_config, _load_prefs_raw, _save_prefs_keys, _prefs, _is_single_file,
    _parse_log_level, _LOG_NAMES, set_log_level, _log, _dbg, _pil_to_b64_jpeg,
)

# Presets "cas d'usage" -> reglages auto. Seules les cles presentes sont appliquees,
# le reste est laisse tel quel. Utilise par l'UI (_apply_preset) et la CLI (--preset).
PRESETS = {
    "Custom": {},
    # Reglages facon ComfyUI Ultimate SD Upscale (x2, tuiles 1024, denoise bas, offload
    # none sur grosse carte). Le tiling plafonne deja la VRAM -> PAS d'offload (qui ralentit).
    "Benchmark (fast)":    {"factor": 2.0, "denoise": 0.12, "steps": 12, "tile": 1024, "overlap": 32,
                            "refine_tile": 1024, "refine_overlap": 64, "cpu_offload": "none"},
    "Photo (balanced)":    {"factor": 2.0, "denoise": 0.30, "steps": 12, "refine_tile": 0, "cpu_offload": "none"},
    "Subtle (clean-up)":   {"factor": 2.0, "denoise": 0.12, "steps": 16, "refine_tile": 0},
    "Detailed (creative)": {"factor": 2.0, "denoise": 0.40, "steps": 16},
    "Portrait (faces)":    {"factor": 2.0, "denoise": 0.22, "steps": 14},
    # 4K: le tiling plafonne la VRAM -> offload none (l'offload ne sert plus a rien et
    # ralentit 5-10x sur grosse carte). Mettre offload sequential SEULEMENT si <16 Go.
    "4K (tiled)":          {"factor": 4.0, "denoise": 0.20, "steps": 12, "tile": 1024, "overlap": 32,
                            "refine_tile": 1024, "refine_overlap": 64, "cpu_offload": "none"},
    "Low VRAM (8-12GB)":   {"denoise": 0.30, "steps": 12, "tile": 512, "refine_tile": 1024, "refine_overlap": 64, "cpu_offload": "sequential"},
}
# param interne -> flag CLI, pour appliquer un preset sans ecraser un flag explicite.
PRESET_FLAGMAP = {
    "factor": "--factor", "denoise": "--denoise", "steps": "--steps", "tile": "--tile",
    "overlap": "--overlap", "refine_tile": "--refine-tile", "refine_overlap": "--refine-overlap",
    "cpu_offload": "--cpu-offload",
}

# Ratios d'aspect facon Fooocus. label -> (width, height) en multiples de 16.
ASPECT_RATIOS = {
    "1024 x 1024  (1:1)":  (1024, 1024),
    "1152 x 896  (9:7)":   (1152, 896),
    "896 x 1152  (7:9)":   (896, 1152),
    "1216 x 832  (3:2)":   (1216, 832),
    "832 x 1216  (2:3)":   (832, 1216),
    "1344 x 768  (16:9)":  (1344, 768),
    "768 x 1344  (9:16)":  (768, 1344),
    "1536 x 640  (21:9)":  (1536, 640),
}
# Performance facon Fooocus -> (gen_steps, guidance) pour le modele charge.
PERFORMANCE = {
    "Turbo (8 steps)":    (8, 0.0),
    "Quality (20 steps)": (20, 0.0),
    "Base CFG (28 steps)": (28, 4.0),
}
# Styles. Format Fooocus: nom -> {"prompt": template avec {prompt} (ou None),
# "negative_prompt": str}. La vraie biblio est chargee depuis styles/*.json (cf.
# _load_styles plus bas). Ceci n'est qu'un fallback si le dossier est absent.
# Styles (Fooocus) + wildcards (__name__) -> cz_prompt.py. Les handlers UI restent ici.
import cz_prompt
from cz_prompt import (  # noqa: E402,F401
    STYLES, _seed_rng, list_wildcards, _apply_wildcards, _pick_styles, _apply_styles,
    set_wildcards_dir,
)

# Real-ESRGAN (spandrel) + upscale tuile/overlap-add -> cz_esrgan.py. L'etat mutable
# (ESRGAN_DIR + cache) vit dans le module; app lit le dossier via cz_esrgan.ESRGAN_DIR.
import cz_esrgan
from cz_esrgan import (  # noqa: E402,F401
    set_esrgan_dir, list_esrgan_models, load_esrgan, esrgan_upscale,
)


def _style_sample(name):
    """Chemin de la vignette d'un style (styles/samples/<nom>.jpg) ou None."""
    try:
        fn = name.lower().replace(" ", "_").replace("-", "_") + ".jpg"
        p = os.path.join(HERE, "styles", "samples", fn)
        return p if os.path.isfile(p) else None
    except Exception:
        return None


def _filter_styles(query, selected):
    """Filtre la liste des styles par recherche. Conserve les styles deja coches."""
    q = (query or "").strip().lower()
    matches = [n for n in STYLES if q in n.lower()] if q else list(STYLES)
    selected = [s for s in (selected or []) if s in STYLES]
    # choices = resultats + styles coches (pour ne pas perdre la selection)
    choices = list(dict.fromkeys(matches + selected))
    return gr.update(choices=choices, value=selected)


# _seed_rng / list_wildcards / _apply_wildcards / _pick_styles / _apply_styles -> cz_prompt.py


# ----------------------------------------------------------------------------
# Ollama (Describe image -> prompt, Improve prompt). + fallback local BLIP.
# ----------------------------------------------------------------------------
# Fonctions Ollama -> cz_ollama.py (les handlers UI _ui_* restent ici, plus bas).
from cz_ollama import (  # noqa: E402,F401
    OLLAMA_URL, OLLAMA_KEEP_ALIVE, OLLAMA_CPU, _ollama_gen_opts, _ollama_http,
    _ollama_vision_models, _ollama_describe, _ollama_improve, _ollama_compose,
)


# Caption local BLIP (fallback Ollama), FaceSwap (InsightFace/inswapper) + restore
# GFPGAN, detourage rembg -> cz_face.py. L'etat mutable (caches modeles + reglages
# restore) vit dans le module.
import cz_face
from cz_face import (  # noqa: E402,F401
    _local_caption, _remove_bg, set_faceswap_restore,
)


def _faceswap(target_img, source_img):
    """Wrapper: passe le dossier checkpoints courant (cz_pipeline) a cz_face._faceswap,
    qui l'ajoute aux emplacements de recherche du modele inswapper."""
    return cz_face._faceswap(target_img, source_img, cz_pipeline.CHECKPOINTS_DIR)


def _dl_path(pil, path):
    """Pour la galerie: renvoie le CHEMIN du fichier sauve (-> telechargement avec le
    vrai nom unique au lieu de 'image') s'il existe SOUS le dossier de sortie autorise
    par Gradio; sinon l'image PIL (apercu). Garde-fou: jamais d'apercu casse si le
    fichier est hors allowed_paths (dossier de sortie non-defaut)."""
    if path:
        try:
            ap = os.path.abspath(path)
            root = os.path.abspath(_ab_resolve_dir(DEFAULT_OUTPUT_DIR))
            if os.path.isfile(ap) and (ap == root or ap.startswith(root + os.sep)):
                return ap
        except Exception:
            pass
    return pil

# ----------------------------------------------------------------------------
# Config (persistance dans preferences.json a cote de app.py)
# Ordre de priorite pour ESRGAN_DIR et BASE_REPO:
#   1) variable d'environnement (ESRGAN_DIR / ZIMAGE_MODEL)
#   2) preferences.json
#   3) defaut: ./upscale_models  et  Tongyi-MAI/Z-Image-Turbo
# ----------------------------------------------------------------------------
import json  # noqa: F811 (utilise par _load_styles ci-dessous)


# _load_styles / STYLES -> cz_prompt.py (importes en tete).


# CONFIG + defauts pilotes par config.txt -> cz_core.py (importes en tete).

# Presets Performance editables via config.txt (performance_presets: nom -> [steps, guidance]).
if isinstance(CONFIG.get("performance_presets"), dict) and CONFIG["performance_presets"]:
    try:
        PERFORMANCE = {k: (int(v[0]), float(v[1])) for k, v in CONFIG["performance_presets"].items()}
    except Exception:
        pass

# MODEL_PROFILES / profile_for_model -> cz_core.py (importes en tete).


# DESCRIBE/IMPROVE/COMPOSE_INSTRUCTION -> cz_core.py (importes en tete).


# _load_prefs_raw / _save_prefs_keys / _is_single_file / _prefs -> cz_core.py.
# Coeur Z-Image -> cz_pipeline.py: modele courant (BASE_REPO/ZIMAGE_TRANSFORMER),
# dossiers checkpoints/loras, LoRA actives, Omni, caches pipe, offload, guidance,
# stop/progress + generation/orchestration. app lit l'etat via cz_pipeline.NAME et
# pose cz_pipeline._PROGRESS / cz_pipeline._STOP depuis les handlers UI.
import cz_pipeline
from cz_pipeline import (  # noqa: E402,F401
    set_guidance, request_stop, set_zimage_model, set_zimage_transformer,
    list_checkpoints, list_loras, set_checkpoints_dir, set_loras_dir, lora_keywords,
    set_omni_model, check_omni_available, set_offload_mode, free_vram, set_loras,
    set_sampler, SAMPLER_CHOICES, set_schedule, SCHEDULE_CHOICES,
    generate, generate_omni, inpaint_run, outpaint, txt2img_run, process_one,
    round_to_multiple, _reframe_canvas, _gen_meta,
)

# Etat mutable lu en live depuis cz_pipeline.* / cz_face.* (LORAS, FACESWAP_RESTORE,
# CHECKPOINTS_DIR, GUIDANCE, _PROGRESS, _STOP, ...). app.py expose ces noms en proxy
# (__getattr__) pour le smoke; ici on lit toujours cz_pipeline.NAME / cz_face.NAME.


# Logging (LOG_LEVEL / _log / _dbg / set_log_level) -> cz_core.py (importes en tete).
# Note: les lectures directes de LOG_LEVEL hors de cz_core utilisent cz_core.LOG_LEVEL.


# Progress/stop, setters (model/transformer/checkpoints/loras/omni/offload/guidance),
# list_checkpoints/list_loras, lora_keywords, check_omni_available, free_vram +
# generation/orchestration -> cz_pipeline.py (importes en tete). L'UI pose
# cz_pipeline._PROGRESS / cz_pipeline._STOP et lit cz_pipeline.NAME pour l'etat.


def apply_preset_to_args(args, raw_argv):
    """Applique un preset aux champs de args qui n'ont PAS ete passes explicitement
    en CLI (un flag explicite gagne toujours sur le preset)."""
    preset = PRESETS.get(getattr(args, "preset", None) or "Custom") or {}
    raw = list(raw_argv or [])
    for key, val in preset.items():
        flag = PRESET_FLAGMAP[key]
        if not any(tok == flag or tok.startswith(flag + "=") for tok in raw):
            setattr(args, key, val)


# ----------------------------------------------------------------------------
# Etage 1 : Real-ESRGAN via spandrel
# ----------------------------------------------------------------------------
# list_esrgan_models / load_esrgan / _pil_to_tensor / _tensor_to_pil /
# esrgan_upscale -> cz_esrgan.py (importes en tete). ESRGAN_DIR y est lu.


# ----------------------------------------------------------------------------
# Z-Image (diffusers, BF16) -> cz_pipeline.py: _ensure_base / get_pipe / _load_omni /
# generate / generate_omni / inpaint_run / outpaint / process_one / txt2img_run +
# round_to_multiple / _reframe_canvas / _make_generator / _refine_* / _gen_meta.
# (importes en tete). _editor_to_image_mask / _editor_img / _crop_input restent ici
# (helpers gr.ImageEditor, pas d'etat pipeline).
# ----------------------------------------------------------------------------
def _editor_to_image_mask(editor_value):
    """Extrait (image, masque) d'un gr.ImageEditor. Masque = zone peinte (diff
    composite/background), blanc = a regenerer."""
    if not editor_value:
        return None, None
    bg = editor_value.get("background")
    comp = editor_value.get("composite")
    if bg is None:
        return None, None
    bg = bg.convert("RGB")
    mask = Image.new("L", bg.size, 0)
    if comp is not None:
        diff = ImageChops.difference(comp.convert("RGB"), bg).convert("L")
        mask = diff.point(lambda p: 255 if p > 8 else 0)
    # fallback: alpha des layers peints
    for ly in (editor_value.get("layers") or []):
        if ly is not None:
            a = ly.convert("RGBA").split()[-1]
            mask = ImageChops.lighter(mask, a.point(lambda p: 255 if p > 8 else 0))
    return bg, mask


def _editor_img(v):
    """Extrait l'image PIL d'un gr.ImageEditor (dict {background,composite,layers})
    ou renvoie le PIL tel quel (retro-compat). Renvoie l'image recadree."""
    if isinstance(v, dict):
        return v.get("composite") or v.get("background")
    return v


def _crop_input(label, height=280):
    """Entree image avec recadrage (crop) facon Fooocus, sans pinceau ni calques."""
    return gr.ImageEditor(type="pil", label=label, height=height,
                          sources=["upload", "clipboard"], brush=False, eraser=False,
                          layers=False, transforms=["crop"])


# outpaint / _make_generator / _refine_whole / _feather_mask_np / _refine_tiled /
# process_one / txt2img_run -> cz_pipeline.py (importes en tete).


# I/O image (noms, sauvegarde, metadonnees) -> cz_imageio.py (_gen_meta -> cz_pipeline).
from cz_imageio import (  # noqa: E402,F401
    _now_stamp, _unique_path, _format_filename, build_output_path, _exif_bytes,
    save_image, _list_output_files, _read_image_meta,
)


# _gen_meta -> cz_pipeline.py (importe en tete; lit le modele/LoRA courants).


def _list_folder_images(folder):
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(IMG_EXTS)
    )


def _format_timings(t, src_path=None, dst_path=None):
    total = t.get("esrgan", 0.0) + t.get("refine", 0.0)
    parts = []
    if src_path:
        parts.append(f"Source: `{src_path}`")
    parts.append(f"ESRGAN: **{t.get('esrgan', 0.0):.1f}s**  |  Z-Image refine: **{t.get('refine', 0.0):.1f}s**  |  Total: **{total:.1f}s**")
    if dst_path:
        parts.append(f"Saved: `{dst_path}`")
    return "  \n".join(parts)


def _reset_vram_peak():
    """Remet a zero le compteur de pic VRAM avant un traitement."""
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _report_vram():
    """Affiche le pic VRAM du run sur stderr. No-op hors CUDA.

    Format stable et parsable: la ligne commence par '[VRAM]'.
    alloue  = pic des tensors PyTorch (max_memory_allocated).
    reserve = pic du cache allocateur PyTorch (max_memory_reserved), plus proche
              de ce que nvidia-smi voit pour ce process.
    """
    if DEVICE != "cuda":
        print("[VRAM] pas de GPU CUDA, mesure ignoree.", file=sys.stderr)
        return
    alloc = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.max_memory_reserved() / 1024**3
    print(f"[VRAM] pic alloue: {alloc:.2f} Go | pic reserve: {reserved:.2f} Go",
          file=sys.stderr)


# Dernier fichier sauve par run() (img2img/upscale). run() garde son retour 3-tuple
# (utilise par le CLId) -> on expose le chemin ici pour que l'UI propose le vrai nom au
# telechargement (sinon Gradio nomme l'apercu PIL "image").
_LAST_RUN_DST = None


def run(image, source_folder, esrgan_model, factor, denoise, steps, prompt, seed,
        tile, overlap, save_mode=DEFAULT_SAVE_MODE, output_dir=DEFAULT_OUTPUT_DIR,
        output_format=DEFAULT_OUTPUT_FORMAT, time_log_path=None, print_output=False,
        refine_tile=DEFAULT_REFINE_TILE, refine_overlap=DEFAULT_REFINE_OVERLAP,
        do_esrgan=True, refine_first=False, styles=None):
    """Point d'entree commun UI / CLI.
    Renvoie (last_result_PIL, last_source_PIL, report_markdown).
    - Si source_folder est un dossier existant -> batch sur ses images.
    - Sinon, image est utilisee (PIL ou chemin str).
    - print_output: imprime le chemin absolu de chaque image sauvee sur stdout.
    - refine_tile > 0: passe Z-Image en tuiles (4K+, plafonne le pic VRAM).
    - do_esrgan=False: img2img pur (pas d'ESRGAN, juste le refine Z-Image).
    - refine_first=True: refine PUIS ESRGAN (diffusion a la resolution native = rapide).
    """
    global _LAST_RUN_DST
    _LAST_RUN_DST = None
    if do_esrgan and not esrgan_model:
        raise gr.Error(f"No ESRGAN model found in {cz_esrgan.ESRGAN_DIR}.")

    # Mode batch
    if source_folder and os.path.isdir(source_folder):
        paths = _list_folder_images(source_folder)
        if not paths:
            raise gr.Error(f"No image in {source_folder}")
        last_result = last_source = None
        lines = [f"### Batch: {len(paths)} image(s) from `{source_folder}`"]
        t_batch = time.time()
        for p in paths:
            try:
                src = Image.open(p)
                result, t = process_one(src, esrgan_model, factor, denoise, steps,
                                        prompt, seed, tile, overlap,
                                        refine_tile=refine_tile, refine_overlap=refine_overlap,
                                        do_esrgan=do_esrgan, refine_first=refine_first)
                _srcbase = os.path.splitext(os.path.basename(p))[0]
                _tag = f"{_srcbase}_" + ("upscaled" if do_esrgan else "img2img")
                dst = build_output_path(p, save_mode, output_dir, output_format,
                                        tag=_tag, seed=seed, size=result.size)
                if dst:
                    save_image(result, dst, output_format, meta=_gen_meta(
                        "upscale" if do_esrgan else "img2img", prompt, seed=seed,
                        steps=steps, guidance=cz_pipeline.GUIDANCE, size=result.size,
                        styles=styles,
                        extra={"source": os.path.basename(p), "factor": factor,
                               "denoise": denoise, "esrgan": esrgan_model if do_esrgan else None}))
                    if print_output:
                        print(os.path.abspath(dst))
                _LAST_RUN_DST = dst
                _append_time_log(time_log_path, p, dst, t, save_mode, output_format)
                lines.append(f"- `{os.path.basename(p)}` {result.size[0]}x{result.size[1]} "
                             f"esrgan {t['esrgan']:.1f}s + refine {t['refine']:.1f}s"
                             + (f" -> `{dst}`" if dst else " (display)"))
                last_result, last_source = result, src.convert("RGB")
            except Exception as e:
                lines.append(f"- `{os.path.basename(p)}` FAILED: {e}")
        lines.append(f"**Batch total: {time.time()-t_batch:.1f}s**")
        return last_result, last_source, "  \n".join(lines)

    # Mode image unique
    if image is None:
        raise gr.Error("Load an image (or specify a source folder for batch mode).")
    if isinstance(image, str):
        source_path = image
        src_img = Image.open(source_path)
    else:
        source_path = None
        src_img = image

    result, t = process_one(src_img, esrgan_model, factor, denoise, steps,
                            prompt, seed, tile, overlap,
                            refine_tile=refine_tile, refine_overlap=refine_overlap,
                            do_esrgan=do_esrgan, refine_first=refine_first)
    dst = None
    _srcbase = os.path.splitext(os.path.basename(source_path))[0] if source_path else None
    _tag = (f"{_srcbase}_" if _srcbase else "") + ("upscaled" if do_esrgan else "img2img")
    try:
        dst = build_output_path(source_path, save_mode, output_dir, output_format,
                                tag=_tag, seed=seed, size=result.size)
    except ValueError as e:
        dst = None
        save_warning = f"  \n[WARN] {e}"
    else:
        save_warning = ""
    if dst:
        save_image(result, dst, output_format, meta=_gen_meta(
            "upscale" if do_esrgan else "img2img", prompt, seed=seed, steps=steps,
            guidance=cz_pipeline.GUIDANCE, size=result.size, styles=styles,
            extra={"factor": factor, "denoise": denoise,
                   "esrgan": esrgan_model if do_esrgan else None}))
        if print_output:
            print(os.path.abspath(dst))
    _LAST_RUN_DST = dst
    _append_time_log(time_log_path, source_path, dst, t, save_mode, output_format)
    report = _format_timings(t, src_path=source_path, dst_path=dst) + save_warning
    return result, src_img.convert("RGB"), report


def _append_time_log(path, src, dst, t, save_mode, output_format):
    if not path:
        return
    try:
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        line = (f"{ts}\t{src or ''}\t{dst or ''}\t"
                f"esrgan={t.get('esrgan', 0):.2f}s\trefine={t.get('refine', 0):.2f}s\t"
                f"mode={save_mode}\tfmt={output_format}\n")
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        print(f"[AVERT] time-log echec: {e}", file=sys.stderr)


# ----------------------------------------------------------------------------
# UI Gradio
# ----------------------------------------------------------------------------
def _refresh_models(new_dir):
    """Change ESRGAN_DIR puis renvoie une mise a jour du Dropdown."""
    set_esrgan_dir(new_dir)
    models = list_esrgan_models()
    value = models[0] if models else None
    return gr.update(choices=models, value=value), f"{len(models)} model(s) found in {cz_esrgan.ESRGAN_DIR}"


def _apply_zimage(repo):
    set_zimage_model(repo)
    return f"Z-Image: {cz_pipeline.BASE_REPO} (will be (re)loaded on next run)"


def _refresh_checkpoints(new_dir):
    """Change le dossier checkpoints + liste les modeles + persiste."""
    set_checkpoints_dir(new_dir)
    try:
        _save_prefs_keys({"checkpoints_dir": cz_pipeline.CHECKPOINTS_DIR})
    except Exception:
        pass
    cks = list_checkpoints()
    return gr.update(choices=["(base repo)"] + cks), f"{len(cks)} checkpoint(s) in {cz_pipeline.CHECKPOINTS_DIR} (saved)."


def _apply_checkpoint(name):
    """Selectionne un checkpoint single-file comme transformer (ou revient au base repo).
    Ajuste aussi steps/guidance selon le profil du modele (contextuel)."""
    if not name or name == "(base repo)":
        set_zimage_transformer("")
        st, g = profile_for_model(cz_pipeline.BASE_REPO)
        return ("Z-Image: base repo transformer (single-file cleared).",
                gr.update(value=st), gr.update(value=g))
    path = name if os.path.isabs(name) else os.path.join(cz_pipeline.CHECKPOINTS_DIR, name)
    set_zimage_transformer(path)
    st, g = profile_for_model(os.path.basename(path))
    return (f"Z-Image transformer: {os.path.basename(path)} -> auto steps={st}, CFG={g} "
            f"(reload on next run).", gr.update(value=st), gr.update(value=g))


def _apply_transformer_repo(repo):
    """Definit le transformer depuis un repo HF / dossier diffusers OU un .safetensors.
    Ajuste steps/guidance selon le profil du modele."""
    repo = (repo or "").strip()
    set_zimage_transformer(repo)
    if not repo:
        return "Transformer: from base repo.", gr.update(), gr.update()
    st, g = profile_for_model(repo)
    return (f"Transformer override: {repo} -> auto steps={st}, CFG={g} "
            f"(keeps base VAE/encoder; reload on next run).",
            gr.update(value=st), gr.update(value=g))


def _wild_sanitize(name):
    return "".join(c for c in (name or "").strip() if c.isalnum() or c in "_-")[:64]


def _ui_wild_refresh(new_dir):
    """Change le dossier + rafraichit le dropdown de tous les wildcards + persiste."""
    set_wildcards_dir(new_dir)
    try:
        _save_prefs_keys({"wildcards_dir": cz_prompt.WILDCARDS_DIR})
    except Exception:
        pass
    w = list_wildcards()
    return gr.update(choices=["None"] + w, value="None"), \
        f"{len(w)} wildcard file(s) in {cz_prompt.WILDCARDS_DIR} (saved)."


def _ui_wild_load(name):
    """Charge le contenu du wildcard selectionne dans l'editeur."""
    if not name or name == "None":
        return "", ""
    p = os.path.join(cz_prompt.WILDCARDS_DIR, name + ".txt")
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        nlines = len([ln for ln in txt.splitlines() if ln.strip() and not ln.lstrip().startswith("#")])
        return txt, f"{name}: {nlines} option(s). Use __{name}__ in the prompt."
    except Exception as e:
        return "", f"Cannot read {name}: {e}"


def _ui_wild_insert(name, prompt_text):
    """Insere __name__ a la fin du prompt."""
    if not name or name == "None":
        return gr.update(), "Pick a wildcard file first."
    tok = f"__{name}__"
    base = (prompt_text or "").rstrip()
    new = (base + (" " if base else "") + tok)
    return gr.update(value=new), f"Inserted {tok}."


def _ui_wild_save(name, content):
    """Sauve le contenu de l'editeur dans le wildcard selectionne."""
    n = _wild_sanitize(name if name and name != "None" else "")
    if not n:
        return "Pick a wildcard file (or use Create new)."
    try:
        os.makedirs(cz_prompt.WILDCARDS_DIR, exist_ok=True)
        with open(os.path.join(cz_prompt.WILDCARDS_DIR, n + ".txt"), "w", encoding="utf-8") as f:
            f.write(content or "")
        return f"Saved {n}.txt."
    except Exception as e:
        return f"Save failed: {e}"


def _ui_wild_create(newname, content):
    """Cree un nouveau wildcard + rafraichit le dropdown."""
    n = _wild_sanitize(newname)
    if not n:
        return gr.update(), "Enter a valid name (letters/digits/_/-).", newname
    try:
        os.makedirs(cz_prompt.WILDCARDS_DIR, exist_ok=True)
        with open(os.path.join(cz_prompt.WILDCARDS_DIR, n + ".txt"), "w", encoding="utf-8") as f:
            f.write(content or "")
        return gr.update(choices=["None"] + list_wildcards(), value=n), f"Created {n}.txt.", ""
    except Exception as e:
        return gr.update(), f"Create failed: {e}", newname


def _refresh_loras(new_dir):
    """Change le dossier loras + liste les LoRA (met a jour les 3 slots) + persiste."""
    set_loras_dir(new_dir)
    try:
        _save_prefs_keys({"loras_dir": cz_pipeline.LORAS_DIR})   # persiste -> survit au reboot
    except Exception:
        pass
    lr = ["None"] + list_loras()
    return (gr.update(choices=lr), gr.update(choices=lr), gr.update(choices=lr),
            f"{len(lr) - 1} LoRA(s) in {cz_pipeline.LORAS_DIR} (saved).")


def _apply_loras(n1, w1, n2, w2, n3, w3):
    """Applique la combinaison des 3 slots LoRA."""
    set_loras([(n1, w1), (n2, w2), (n3, w3)])
    if not cz_pipeline.LORAS:
        return "LoRA: none."
    return "LoRA: " + ", ".join(f"{os.path.basename(p)}@{w}" for p, w in cz_pipeline.LORAS) + " (reload on next run)."


def _path_for_lora(name):
    if not name or name in ("None", "none", ""):
        return None
    return name if os.path.isabs(name) else os.path.join(cz_pipeline.LORAS_DIR, name)


def _ui_loras_apply(n1, w1, n2, w2, n3, w3):
    """Applique les slots + agrege les mots-cles des LoRA selectionnees."""
    status = _apply_loras(n1, w1, n2, w2, n3, w3)
    kws = []
    for n in (n1, n2, n3):
        p = _path_for_lora(n)
        if p:
            k = lora_keywords(p)
            if k:
                kws.append(k)
    return status, ", ".join(kws)


def _ui_loras_keywords(n1, n2, n3):
    """Recupere les mots-cles de toutes les LoRA selectionnees (bouton)."""
    kws = []
    for n in (n1, n2, n3):
        p = _path_for_lora(n)
        if p:
            k = lora_keywords(p)
            if k:
                kws.append(k)
    merged = ", ".join(kws)
    return merged, (f"{len(merged.split(','))} keyword(s)." if merged
                    else "No keywords in the selected LoRA(s).")


def _ui_kw_to_prompt(prompt_text, keywords):
    """Ajoute les mots-cles a la fin du prompt courant."""
    kw = (keywords or "").strip().strip(",").strip()
    if not kw:
        return gr.update()
    base = (prompt_text or "").strip()
    if base and not base.endswith(","):
        base += ", "
    return gr.update(value=base + kw)


def _ui_check_omni():
    return check_omni_available()


def _save_paths_to_prefs(esrgan_dir, zimage_model, checkpoints_dir=None, loras_dir=None,
                         wildcards_dir=None):
    """Persiste les chemins dans preferences.json (local) -> charges au prochain boot."""
    set_esrgan_dir(esrgan_dir)
    set_zimage_model(zimage_model)
    if checkpoints_dir:
        set_checkpoints_dir(checkpoints_dir)
    if loras_dir:
        set_loras_dir(loras_dir)
    if wildcards_dir:
        set_wildcards_dir(wildcards_dir)
    _save_prefs_keys({"esrgan_dir": cz_esrgan.ESRGAN_DIR, "zimage_model": cz_pipeline.BASE_REPO,
                      "checkpoints_dir": cz_pipeline.CHECKPOINTS_DIR, "loras_dir": cz_pipeline.LORAS_DIR,
                      "wildcards_dir": cz_prompt.WILDCARDS_DIR})
    return (f"Saved to {PREFS_PATH}: esrgan_dir, zimage_model, checkpoints_dir, "
            f"loras_dir, wildcards_dir={cz_prompt.WILDCARDS_DIR}")


# Ordre des composants mis a jour par le dropdown de presets (doit matcher l'UI).
_PRESET_UI_ORDER = ("factor", "denoise", "steps", "tile", "overlap",
                    "refine_tile", "refine_overlap", "cpu_offload")


def _apply_preset(name):
    """UI: renvoie les updates des controles pour le preset choisi (ordre _PRESET_UI_ORDER).
    Custom ou cle absente = pas de changement sur ce controle."""
    p = PRESETS.get(name, {})
    return [gr.update(value=p[k]) if k in p else gr.update() for k in _PRESET_UI_ORDER]


def _set_aspect(name):
    """UI: applique un ratio d'aspect -> (width, height)."""
    w, h = ASPECT_RATIOS.get(name, (1024, 1024))
    return w, h


def _set_performance(name):
    """UI: applique un preset Performance -> (gen_steps, guidance)."""
    steps, g = PERFORMANCE.get(name, (8, 0.0))
    return steps, g


def _ui_detect_ollama(url):
    """Detecte Ollama et liste UNIQUEMENT les modeles vision pour le Describe."""
    try:
        models = _ollama_vision_models(base=url)
    except Exception:
        return (gr.update(choices=[], value=None),
                "Ollama not reachable. Describe will use the local BLIP fallback. "
                "Improve needs Ollama.")
    if not models:
        return (gr.update(choices=[], value=None),
                "Ollama OK but no VISION model. Pull one, e.g. `ollama pull llava` or `moondream`.")
    return gr.update(choices=models, value=models[0]), f"Ollama OK - {len(models)} vision model(s)."


def _ui_describe(image, model, url):
    """Decrit l'image -> remplit le prompt. Ollama si modele choisi, sinon BLIP local."""
    image = _editor_img(image)
    if image is None:
        return gr.update(), "Drop an image to describe first."
    if model:
        try:
            return gr.update(value=_ollama_describe(image, model, base=url)), f"Described via {model}."
        except Exception as e:
            return gr.update(), f"Ollama describe failed: {e}"
    try:
        return gr.update(value=_local_caption(image)), "Described via local BLIP (no Ollama model)."
    except Exception as e:
        return gr.update(), f"No Ollama model selected and local captioner failed: {e}"


def _ui_improve(prompt_text, model, url):
    """Ameliore le prompt courant via Ollama (meme modele)."""
    if not (prompt_text or "").strip():
        return gr.update(), "Type a prompt first."
    if not model:
        return gr.update(), "Select a model in Advanced > Prompt AI (click Detect Ollama)."
    try:
        return gr.update(value=_ollama_improve(prompt_text, model, base=url)), f"Improved via {model}."
    except Exception as e:
        return gr.update(), f"Improve failed: {e}"


def _ui_compose(r1, r2, r3, r4, model, url):
    """'Faux Omni': decrit chaque image de reference (vision) puis fusionne en UN
    prompt via le LLM. Remplit la zone de prompt."""
    refs = [im for im in (_editor_img(r) for r in [r1, r2, r3, r4]) if im is not None]
    if not refs:
        return gr.update(), "Add at least one reference image."
    if not model:
        return gr.update(), "Select an Ollama vision model in Advanced > Prompt AI (Detect)."
    try:
        caps = [_ollama_describe(im, model, base=url) for im in refs]
    except Exception as e:
        return gr.update(), f"Describe failed: {e}"
    try:
        merged = _ollama_compose(caps, model, base=url)
    except Exception as e:
        return gr.update(), f"Compose failed: {e}"
    return gr.update(value=merged), f"Composed one prompt from {len(refs)} image(s) via {model}."


def _ui_remove_bg(image, history, save_mode, output_dir):
    """Remove background -> resultat (PNG transparent) dans la galerie + historique."""
    image = _editor_img(image)
    if image is None:
        return [], "Drop an image first.", history, history
    try:
        res = _remove_bg(image)
    except Exception as e:
        return [], f"Remove BG failed: {e}", history, history
    if save_mode != "display":
        try:
            dst = build_output_path(None, save_mode, output_dir, "png", tag="nobg", size=res.size)
            if dst:
                save_image(res, dst, "png")
        except Exception as e:
            _dbg(f"save nobg failed: {e}")
    new_hist = ([res] + list(history or []))[:200]
    return [res], "Background removed (transparent PNG).", new_hist, new_hist


def _ui_reframe(image, ratio, steps, prompt, guidance, offload_mode, seed,
                save_mode, output_dir, output_format, history,
                progress=gr.Progress(track_tqdm=True)):
    """Reframe / outpaint -> resultat dans la galerie + historique."""
    image = _editor_img(image)
    if image is None:
        return [], "Drop an image first.", history, history
    cz_pipeline._PROGRESS = lambda f, d: progress(f, desc=d)
    try:
        set_offload_mode(offload_mode)
        set_guidance(guidance)
        try:
            rw, rh = [int(x) for x in str(ratio).split(":")]
        except Exception:
            rw, rh = 16, 9
        try:
            res = outpaint(image, rw, rh, prompt, steps, seed)
        except Exception as e:
            _log(f"outpaint error: {e}")
            return [], f"Reframe failed: {e}", history, history
        dst = None
        if save_mode != "display":
            try:
                dst = build_output_path(None, save_mode, output_dir, output_format,
                                        tag="reframe", seed=seed, size=res.size)
                if dst:
                    save_image(res, dst, output_format, meta=_gen_meta(
                        "reframe", prompt, seed=seed, steps=steps, guidance=cz_pipeline.GUIDANCE,
                        size=res.size, extra={"ratio": ratio}))
            except Exception as e:
                dst = None
                _dbg(f"save reframe failed: {e}")
        item = _dl_path(res, dst)   # telechargement avec le vrai nom de fichier si sauve
        new_hist = ([item] + list(history or []))[:200]
        return [item], f"Reframed to {res.size[0]}x{res.size[1]} ({ratio}).", new_hist, new_hist
    finally:
        cz_pipeline._PROGRESS = None


def _ui_inpaint(editor_value, prompt, negative, styles, guidance, offload_mode, steps, denoise,
                seed, save_mode, output_dir, output_format, history,
                progress=gr.Progress(track_tqdm=True)):
    """Inpaint depuis l'editeur (image + masque peint) -> galerie + historique."""
    cz_pipeline._PROGRESS = lambda f, d: progress(f, desc=d)
    try:
        bg, mask = _editor_to_image_mask(editor_value)
        if bg is None:
            return [], "Load an image in the Inpaint editor.", history, history
        if mask is None or mask.getbbox() is None:
            return [], "Paint the area to change (the mask is empty).", history, history
        set_offload_mode(offload_mode)
        set_guidance(guidance)
        full_prompt, _ = _apply_styles(prompt, negative, styles)
        try:
            res = inpaint_run(bg, mask, full_prompt, steps, denoise, seed)
        except Exception as e:
            _log(f"inpaint error: {e}")
            return [], f"Inpaint failed: {e}", history, history
        dst = None
        if save_mode != "display":
            try:
                dst = build_output_path(None, save_mode, output_dir, output_format,
                                        tag="inpaint", seed=seed, size=res.size)
                if dst:
                    save_image(res, dst, output_format, meta=_gen_meta(
                        "inpaint", full_prompt, seed=seed, steps=steps, guidance=cz_pipeline.GUIDANCE,
                        size=res.size, styles=styles, extra={"strength": denoise}))
            except Exception as e:
                dst = None
                _dbg(f"save inpaint failed: {e}")
        item = _dl_path(res, dst)   # telechargement avec le vrai nom de fichier si sauve
        new_hist = ([item] + list(history or []))[:200]
        return [item], f"Inpaint -> {res.size[0]}x{res.size[1]}", new_hist, new_hist
    finally:
        cz_pipeline._PROGRESS = None


def _ui_clear_history():
    """Vide l'historique de session (state + galerie)."""
    return [], []


def _ui_load_outputs(output_dir):
    """Charge les images du dossier de sortie dans l'historique de session."""
    files = _list_output_files(output_dir, 200)
    return files, files


# ---- Galerie avancee (panneau dedie: meta, suppression, flou) ----
def _gallery_filtered(output_dir, sort="Newest", filt=""):
    files = _list_output_files(output_dir, 4000)
    if filt:
        f = filt.lower()
        files = [p for p in files if f in os.path.basename(p).lower()]
    if sort == "Oldest":
        files.sort(key=os.path.getmtime)
    elif sort == "Name":
        files.sort(key=lambda p: os.path.basename(p).lower())
    else:  # Newest
        files.sort(key=os.path.getmtime, reverse=True)
    return files[:300]


def _gallery_load(output_dir, sort="Newest", filt=""):
    files = _gallery_filtered(output_dir, sort, filt)
    return files, files, f"{len(files)} image(s)."


def _gallery_selected(paths, evt: gr.SelectData):
    """Affiche les metadonnees de l'image selectionnee + son chemin."""
    if not paths or evt is None or evt.index is None or evt.index >= len(paths):
        return "*No selection.*", ""
    path = paths[evt.index]
    info = [f"**File:** `{os.path.basename(path)}`"]
    meta = _read_image_meta(path)
    if meta.get("prompt"):
        info.append(f"**Prompt:** {meta['prompt']}")
    if meta.get("negative"):
        info.append(f"**Negative:** {meta['negative']}")
    line2 = []
    for k in ("mode", "seed", "steps", "guidance", "size"):
        if meta.get(k) is not None:
            line2.append(f"{k}={meta[k]}")
    if meta.get("model"):
        line2.append(f"model={os.path.basename(str(meta['model']))}")
    if meta.get("loras"):
        line2.append("loras=" + ",".join(meta["loras"]))
    if line2:
        info.append("**Params:** " + "  ".join(line2))
    if not meta:  # pas de sidecar -> infos minimales (taille reelle)
        try:
            with Image.open(path) as im:
                info.append(f"**Size:** {im.size[0]}x{im.size[1]}")
        except Exception:
            pass
    try:
        st = os.stat(path)
        info.append(f"**Weight:** {st.st_size / 1024:.0f} KB  Â·  "
                    f"**Modified:** {datetime.datetime.fromtimestamp(st.st_mtime):%Y-%m-%d %H:%M}")
    except Exception:
        pass
    info.append(f"**Path:** `{path}`")
    return "  \n".join(info), path


def _gallery_delete(path, output_dir, sort="Newest", filt=""):
    """Supprime le fichier selectionne (+ sidecar) puis recharge la galerie."""
    msg = "Nothing to delete."
    if path and os.path.isfile(path):
        try:
            os.remove(path)
            if os.path.isfile(path + ".json"):   # supprime aussi le sidecar
                os.remove(path + ".json")
            msg = f"Deleted {os.path.basename(path)}."
        except Exception as e:
            msg = f"Delete failed: {e}"
    files = _gallery_filtered(output_dir, sort, filt)
    return files, files, msg, "*Select an image to see its info.*", ""


# ----------------------------------------------------------------------------
# Asset Browser (facon Fooocus2026): SPA statique deposee dans le dossier de
# sortie, ouverte via un lien file=, alimentee par un manifest JSON + thumbnails.
# Memes options (enabled, generate_thumbnails, thumbnail_size/quality, blur).
# ----------------------------------------------------------------------------
# Asset Browser (SPA + reindex + thumbnails + delete) -> cz_assetbrowser.py.
from cz_assetbrowser import _ab_get, _ab_resolve_dir, ab_reindex, delete_asset  # noqa: E402,F401


# Assets statiques (SPA Asset Browser, JS d'interface, CSS) -> cz_assets.py
from cz_assets import ASSET_BROWSER_HTML, CZ_JS, FOOOCUS_CSS  # noqa: E402


def _ui_ab_reindex(output_dir, thumb_size, quality, blur, gen_thumbs):
    """Bouton FORCE: regenere TOUTES les miniatures (synchrone) + lien."""
    try:
        n, idx, _ = ab_reindex(output_dir, thumb_size, quality, blur, gen_thumbs,
                               background_thumbs=False)
    except Exception as e:
        return "", f"Asset Browser reindex failed: {e}"
    url = "/gradio_api/file=" + os.path.abspath(idx).replace("\\", "/")
    link = (f'<a href="{url}" target="_blank" style="display:inline-block;padding:8px 14px;'
            f'background:#3b4356;color:#fff;border-radius:6px;text-decoration:none;">'
            f'\U0001F5BCï¸ Open Asset Browser ({n} images)</a>')
    return link, f"Rebuilt all thumbnails for {n} image(s)."


def _ui_gallery_open(output_dir):
    """Ouverture RAPIDE: manifest immediat, miniatures generees en tache de fond."""
    try:
        n, idx, pending = ab_reindex(output_dir, _ab_get("thumbnail_size"),
                                     _ab_get("thumbnail_quality"), bool(_ab_get("blur_thumbnails")),
                                     bool(_ab_get("generate_thumbnails")), background_thumbs=True)
    except Exception as e:
        return f"Gallery build failed: {e}", ""
    url = "/gradio_api/file=" + os.path.abspath(idx).replace("\\", "/")
    extra = f" Generating {pending} thumbnail(s) in background (full images shown meanwhile)." if pending else ""
    return f"Opening {n} image(s) in a new tab...{extra}", url


# delete_asset -> cz_assetbrowser.py (importe en tete; expose via api_name dans build_ui).
# _pil_to_b64_jpeg -> cz_core.py (importe en tete).


def _make_compare_html(src_img, result_img):
    """Comparateur avant/apres standalone: 2 <img> superposees, slider range pilote un clip-path."""
    if src_img is None or result_img is None:
        return "<div style='padding:1em;color:#888'>No result to compare.</div>"
    src_b64 = _pil_to_b64_jpeg(src_img)
    res_b64 = _pil_to_b64_jpeg(result_img)
    uid = uuid.uuid4().hex[:8]
    return f"""
<div style="position:relative; max-width:100%; user-select:none;">
  <img src="data:image/jpeg;base64,{src_b64}" style="display:block; width:100%; height:auto;" alt="source" />
  <img id="cmp-top-{uid}" src="data:image/jpeg;base64,{res_b64}"
       style="position:absolute; top:0; left:0; display:block; width:100%; height:100%;
              clip-path: inset(0 50% 0 0); -webkit-clip-path: inset(0 50% 0 0);" alt="resultat" />
  <div id="cmp-bar-{uid}" style="position:absolute; top:0; left:50%; width:2px; height:100%;
       background:#fff; box-shadow:0 0 4px rgba(0,0,0,0.5); pointer-events:none;"></div>
  <input type="range" min="0" max="100" value="50"
         oninput="
           var v=this.value;
           document.getElementById('cmp-top-{uid}').style.clipPath='inset(0 '+(100-v)+'% 0 0)';
           document.getElementById('cmp-top-{uid}').style.webkitClipPath='inset(0 '+(100-v)+'% 0 0)';
           document.getElementById('cmp-bar-{uid}').style.left=v+'%';
         "
         style="position:absolute; bottom:10px; left:5%; width:90%; height:14px; cursor:ew-resize;" />
  <div style="position:absolute; top:8px; left:8px; padding:2px 8px; background:rgba(0,0,0,0.6); color:#fff;
              font-size:12px; border-radius:4px; pointer-events:none;">BEFORE</div>
  <div style="position:absolute; top:8px; right:8px; padding:2px 8px; background:rgba(0,0,0,0.6); color:#fff;
              font-size:12px; border-radius:4px; pointer-events:none;">AFTER</div>
</div>
"""


def _ui_run(image, source_folder, esrgan_model, factor, denoise, steps, prompt, seed,
            tile, overlap, offload_mode, refine_tile, refine_overlap, do_esrgan, guidance,
            save_mode, output_dir, output_format):
    """Adaptateur UI: appelle run() et renvoie (result_image, html_slider, report_markdown)."""
    set_offload_mode(offload_mode)
    set_guidance(guidance)
    last_result, last_source, report = run(
        image, source_folder, esrgan_model, factor, denoise, steps, prompt, seed,
        tile, overlap, save_mode=save_mode, output_dir=output_dir,
        output_format=output_format, refine_tile=refine_tile, refine_overlap=refine_overlap,
        do_esrgan=bool(do_esrgan),
    )
    html = _make_compare_html(last_source, last_result)
    return last_result, html, report


def _ui_txt2img(prompt, negative, width, height, gen_steps, seed, guidance, upscale,
                esrgan_model, factor, denoise, offload_mode):
    """Adaptateur UI txt2img: genere puis (optionnel) upscale. Renvoie (image, report)."""
    set_offload_mode(offload_mode)
    set_guidance(guidance)
    result, t = txt2img_run(prompt, width, height, gen_steps, seed, negative,
                            upscale=bool(upscale), esrgan_model=esrgan_model,
                            factor=factor, denoise=denoise, steps=DEFAULT_STEPS)
    rep = f"txt2img **{result.size[0]}x{result.size[1]}** in **{t['txt2img']:.1f}s**"
    if upscale:
        rep += (f"  \nESRGAN **{t['esrgan']:.1f}s** + refine **{t['refine']:.1f}s** "
                f"-> **{result.size[0]}x{result.size[1]}**")
    return result, rep


def _ui_generate(prompt, negative, styles, style_random, use_input, input_image,
                 input_mode, ref1, ref2, ref3, ref4, faceswap_enable, faceswap_src,
                 width, height, gen_steps, image_number, seed, guidance, offload_mode,
                 esrgan_model, do_esrgan, do_refine, refine_first, factor, denoise, refine_steps,
                 tile, overlap, refine_tile, refine_overlap,
                 save_mode, output_dir, output_format, history,
                 progress=gr.Progress(track_tqdm=True)):
    """Bouton Generate unifie facon Fooocus. Renvoie 4 sorties:
    (images du run, report, history_state, history_gallery). L'historique accumule
    les rendus de la session (plus recents en tete, cap 200)."""
    cz_pipeline._STOP = False
    cz_pipeline._PROGRESS = lambda f, d: progress(f, desc=d)
    progress(0.0, desc="Starting...")
    # Les entrees image sont des gr.ImageEditor (crop) -> extraire le PIL recadre.
    input_image = _editor_img(input_image)
    faceswap_src = _editor_img(faceswap_src)
    ref1, ref2, ref3, ref4 = (_editor_img(ref1), _editor_img(ref2),
                              _editor_img(ref3), _editor_img(ref4))

    def _done(imgs, rep, paths=None):
        # FaceSwap post-process (optionnel, gated). S'applique a tous les modes.
        if faceswap_enable and faceswap_src is not None and imgs:
            try:
                imgs = [_faceswap(im, faceswap_src) for im in imgs]
                rep += " + faceswap"
                paths = [None] * len(imgs)   # nouvelles images -> nouveaux chemins
                if save_mode != "display":
                    for k, im in enumerate(imgs):
                        dst = build_output_path(None, save_mode, output_dir, output_format,
                                                tag="faceswap", seed=seed, size=im.size,
                                                index=(k + 1 if len(imgs) > 1 else 0))
                        if dst:
                            save_image(im, dst, output_format)
                            paths[k] = dst
            except Exception as e:
                _log(f"faceswap error: {e}")
                rep += f"  \n[faceswap skipped: {e}]"
        # Galerie + historique: chemin du fichier (vrai nom au telechargement) si dispo,
        # sinon l'image PIL (apercu). _dl_path garde l'apercu intact dans tous les cas.
        paths = paths or [None] * len(imgs)
        gallery = [_dl_path(imgs[i], paths[i] if i < len(paths) else None) for i in range(len(imgs))]
        new_hist = (list(gallery) + list(history or []))[:200]
        return gallery, rep, new_hist, new_hist

    try:
        set_offload_mode(offload_mode)
        set_guidance(guidance)
        base_prompt = _apply_wildcards(prompt, _seed_rng(seed))   # __name__ -> random line
        picked_styles = _pick_styles(styles, style_random)        # noms de styles -> meta
        full_prompt, full_negative = _apply_styles(base_prompt, negative, picked_styles)
        mode = "img2img/upscale" if (use_input and input_image is not None) else "txt2img"
        _log(f"Generate ({mode})")
        _dbg(f"params: mode={mode} use_input={use_input} has_img={input_image is not None} "
             f"size={int(width)}x{int(height)} gen_steps={int(gen_steps)} n={int(image_number)} "
             f"seed={int(seed)} guidance={float(guidance)} offload={offload_mode} styles={styles}")
        _dbg(f"prompt='{(full_prompt or '')[:160]}' | negative='{(negative or '')[:80]}'")
        # --- Omni multi-reference (compo a partir de plusieurs images) ---
        # Garde-fou: on ne route en Omni que si un modele Omni est configure. Sinon
        # (UI obsolete dans le navigateur, mode reste sur Omni) on retombe en
        # txt2img/img2img au lieu d'echouer.
        omni_ready = bool((cz_pipeline.OMNI_MODEL or "").strip())
        if use_input and input_mode == "Reference (Omni)" and omni_ready:
            refs = [r for r in [ref1, ref2, ref3, ref4] if r is not None]
            _dbg(f"omni: {len(refs)} ref(s), size={int(width)}x{int(height)}")
            if not refs:
                return _done([], "Omni: add at least one reference image.")
            try:
                img = generate_omni(refs, full_prompt, full_negative, width, height, gen_steps, seed)
            except Exception as e:
                _log(f"omni error: {e}")
                return _done([], f"Omni error: {e}")
            omni_dst = None
            if save_mode != "display":
                try:
                    omni_dst = build_output_path(None, save_mode, output_dir, output_format,
                                                 tag="omni", seed=seed, size=img.size)
                    if omni_dst:
                        save_image(img, omni_dst, output_format, meta=_gen_meta(
                            "omni", full_prompt, full_negative, seed, gen_steps, cz_pipeline.GUIDANCE,
                            img.size, styles=picked_styles, extra={"refs": len(refs)}))
                        _dbg(f"saved: {omni_dst}")
                except Exception as e:
                    _dbg(f"save failed: {e}")
            return _done([img], f"omni - **{img.size[0]}x{img.size[1]}** from {len(refs)} ref(s)", [omni_dst])
        if use_input and input_image is not None:
            # Refine (img2img) decoche -> denoise 0 = saute la passe de diffusion (lente).
            eff_denoise = float(denoise) if do_refine else 0.0
            _dbg(f"img2img: esrgan={esrgan_model} do_esrgan={do_esrgan} do_refine={do_refine} "
                 f"factor={factor} denoise={eff_denoise} refine_steps={int(refine_steps)} tile={int(tile)} "
                 f"refine_tile={int(refine_tile)} model={cz_pipeline.BASE_REPO} transformer={cz_pipeline.ZIMAGE_TRANSFORMER}")
            try:
                last_result, last_source, report = run(
                    input_image, None, esrgan_model, factor, eff_denoise, refine_steps, full_prompt, seed,
                    tile, overlap, save_mode=save_mode, output_dir=output_dir,
                    output_format=output_format, refine_tile=refine_tile, refine_overlap=refine_overlap,
                    do_esrgan=bool(do_esrgan), refine_first=bool(refine_first), styles=picked_styles)
            except Exception as e:
                _log(f"img2img/upscale error: {e}")
                msg = f"Upscale/img2img failed: {e}"
                if "CUDA" in str(e) or "out of memory" in str(e).lower():
                    msg += ("  \n**VRAM saturee** (autre app GPU comme ComfyUI encore chargee ? "
                            "spill -> timeout Windows TDR). Ferme les autres apps GPU, **redemarre "
                            "crispz-studio** (le contexte CUDA est mort), baisse refine_tile / factor.")
                return _done([], msg)
            # _LAST_RUN_DST = fichier reellement sauve par run() -> vrai nom au download.
            return _done([last_result], report, [_LAST_RUN_DST])
        # txt2img (batch image_number)
        n = max(1, int(image_number))
        images, img_paths, total_t = [], [], 0.0
        for i in range(n):
            if cz_pipeline._STOP:
                _log(f"stop requested after {i}/{n} image(s)")
                break
            s = (int(seed) + i) if int(seed) >= 0 else -1
            progress(i / n, desc=f"Image {i + 1}/{n}")
            # Wildcards (__name__) + style aleatoire, par image (seed -> reproductible)
            chosen = _pick_styles(styles, style_random)
            p_i = _apply_wildcards(prompt, _seed_rng(s))
            fp, fn = _apply_styles(p_i, negative, chosen)
            if style_random:
                _log(f"random style #{i + 1}: {chosen}")
            img, t = txt2img_run(fp, width, height, gen_steps, s, fn,
                                 upscale=False, steps=refine_steps)
            images.append(img)
            total_t += t["txt2img"]
            dst = None
            if save_mode != "display":
                try:
                    dst = build_output_path(None, save_mode, output_dir, output_format,
                                            tag="txt2img", seed=s, size=img.size,
                                            index=(i + 1 if n > 1 else 0))
                    if dst:
                        save_image(img, dst, output_format, meta=_gen_meta(
                            "txt2img", fp, fn, s, gen_steps, cz_pipeline.GUIDANCE,
                            img.size, styles=chosen))
                        _dbg(f"saved: {dst}")
                except Exception as e:
                    dst = None
                    _dbg(f"save failed: {e}")
            img_paths.append(dst)
        progress(1.0, desc="Done")
        if not images:
            return _done([], "Stopped before any image.")
        suffix = " (stopped)" if cz_pipeline._STOP else ""
        rep = (f"txt2img x{len(images)} - **{images[0].size[0]}x{images[0].size[1]}** "
               f"in **{total_t:.1f}s**{suffix}")
        return _done(images, rep, img_paths)
    finally:
        cz_pipeline._PROGRESS = None


def _pick_download(evt: gr.SelectData):
    """Clic sur une image du resultat -> bouton Download pointant sur le VRAI fichier
    (avec son nom de disque), au lieu du 'image' generique de la galerie Gradio."""
    try:
        v = evt.value
        path = None
        if isinstance(v, dict):
            img = v.get("image")
            if isinstance(img, dict):
                path = img.get("path") or img.get("name")
            path = path or v.get("path") or v.get("name")
        elif isinstance(v, (list, tuple)) and v:
            path = v[0]
        elif isinstance(v, str):
            path = v
        if path and os.path.isfile(str(path)):
            return gr.DownloadButton(value=str(path), visible=True)
    except Exception:
        pass
    return gr.DownloadButton(visible=False)


# JS injecte au chargement: force le theme sombre, preview de style au survol,
# et lightbox plein ecran au clic sur le rendu. __MAP__ = {nom_style: url_vignette}.
def build_ui():
    models = list_esrgan_models()
    default_model = DEFAULT_MODEL if DEFAULT_MODEL in models else (models[0] if models else None)

    # Map nom_style -> URL de vignette (servie par Gradio) pour le preview au survol.
    _sample_urls = {}
    for n in STYLES:
        p = _style_sample(n)
        if p:
            _sample_urls[n] = "/gradio_api/file=" + os.path.abspath(p).replace("\\", "/")
    js_full = CZ_JS.replace("__MAP__", json.dumps(_sample_urls))
    # Omni (multi-reference) propose seulement si un modele Omni/Edit est configure.
    omni_on = bool((cz_pipeline.OMNI_MODEL or "").strip())

    with gr.Blocks(title="crispz-studio", theme=gr.themes.Default(), css=FOOOCUS_CSS, js=js_full) as demo:
        # La galerie du dossier de sortie s'ouvre dans un nouvel onglet (Asset Browser),
        # via le bouton sous l'apercu. Pas de panneau galerie inline.
        gallery_url = gr.Textbox(visible=False)
        # Endpoint API (appele par l'Asset Browser pour supprimer une image)
        del_in = gr.Textbox(visible=False)
        del_out = gr.Textbox(visible=False)
        del_btn = gr.Button(visible=False)
        del_btn.click(delete_asset, del_in, del_out, api_name="delete_asset")

        with gr.Row():
            # ===== Colonne principale (apercu en haut, prompt + Generate, negative, input) =====
            with gr.Column(scale=3):
                out = gr.Gallery(label="Result", elem_id="cz_result", columns=2,
                                 object_fit="contain", preview=True, allow_preview=True,
                                 show_fullscreen_button=True, show_download_button=True)
                # Le download natif de la galerie nomme le fichier "image" (limite Gradio).
                # Ce bouton telecharge le VRAI fichier (vrai nom) de l'image cliquee.
                result_dl = gr.DownloadButton("⬇ Download (real filename)", size="sm", visible=False)
                report = gr.Markdown(value="*Ready. Type a prompt and press Generate.*")

                history = gr.State([])
                with gr.Accordion("History (this session)", open=False):
                    gr.Markdown("*Renders made in this session (in-memory, not the disk folder). "
                                "Use 'Load output folder' to pull saved files in.*")
                    history_gallery = gr.Gallery(label=None, height=240, columns=6,
                                                 object_fit="cover", show_download_button=True)
                    with gr.Row():
                        load_out_btn = gr.Button("Load output folder", size="sm")
                        clear_hist_btn = gr.Button("Clear history", size="sm")
                        gallery_btn = gr.Button("\U0001F5BCï¸ Asset Browser", size="sm",
                                                variant="primary")
                gallery_status = gr.Markdown("")

                with gr.Row():
                    prompt = gr.Textbox(show_label=False, value=CONFIG.get("default_prompt", ""),
                                        placeholder="Type your prompt here...",
                                        elem_id="cz_prompt", lines=2, scale=4, container=False)
                    with gr.Column(scale=1, min_width=150):
                        btn = gr.Button("Generate", elem_id="cz_generate", min_width=150)
                        stop_btn = gr.Button("Stop", variant="stop", size="sm", min_width=150)

                negative = gr.Textbox(show_label=False, value=CONFIG.get("default_negative_prompt", ""),
                                      elem_id="cz_neg", lines=1, container=False,
                                      placeholder="Negative prompt - what you do NOT want (needs guidance > 0)")

                with gr.Row():
                    improve_btn = gr.Button("Improve prompt (Ollama)", size="sm", scale=0, min_width=200)
                    improve_status = gr.Markdown("")

                with gr.Row():
                    use_input = gr.Checkbox(value=False, label="Input Image", min_width=160)
                    advanced_cb = gr.Checkbox(value=False, label="Advanced", min_width=160)

                with gr.Group(visible=False) as input_group:
                    input_mode = gr.Radio(
                        ["Upscale / img2img"] + (["Reference (Omni)"] if omni_on else []),
                        value="Upscale / img2img", label="Input mode", visible=omni_on,
                        info="Reference (Omni) = compose from several reference images + prompt.")
                    with gr.Tabs():
                        with gr.Tab("Upscale or img2img"):
                            with gr.Row():
                                inp = _crop_input("Drop image here / click to upload", 300)
                                with gr.Column():
                                    with gr.Row():
                                        do_esrgan_cb = gr.Checkbox(value=True, label="ESRGAN upscale",
                                                                   info="Uncheck = img2img only (no enlargement).")
                                        do_refine_cb = gr.Checkbox(value=bool(CONFIG.get("default_refine", True)),
                                                                   label="Refine (img2img)",
                                                                   info="Uncheck = ESRGAN upscale only (skip the slow diffusion pass).")
                                    refine_first_cb = gr.Checkbox(value=bool(CONFIG.get("default_refine_first", False)),
                                                                  label="Refine before upscale (faster)",
                                                                  info="Refine at native res THEN ESRGAN enlarge "
                                                                       "(~4-16x faster refine; a touch less high-res detail).")
                                    preset = gr.Dropdown(list(PRESETS), value="Custom", label="Use case preset")
                                    esrgan = gr.Dropdown(models, value=default_model, label="ESRGAN model")
                                    factor = gr.Slider(1.0, 4.0, value=DEFAULT_FACTOR, step=0.5, label="Upscale factor")
                                    denoise = gr.Slider(0.0, 0.8, value=DEFAULT_DENOISE, step=0.01,
                                                        label="Refine denoise (strength) - 0 = skip refine (ESRGAN only)")
                                    refine_steps = gr.Slider(4, 30, value=DEFAULT_STEPS, step=1,
                                                             label="Refine steps (runs at upscaled res -> higher = slower)")
                            with gr.Accordion("ESRGAN tiling (VRAM)", open=False):
                                tile = gr.Slider(0, 1024, value=DEFAULT_TILE, step=8, label="Tile (0 = off)")
                                overlap = gr.Slider(0, 128, value=DEFAULT_OVERLAP, step=8, label="Overlap")
                            with gr.Accordion("Z-Image tiling (4K+)", open=False):
                                refine_tile = gr.Slider(0, 2048, value=DEFAULT_REFINE_TILE, step=16,
                                                        label="Diffusion tile (0 = whole image)")
                                refine_overlap = gr.Slider(0, 256, value=DEFAULT_REFINE_OVERLAP, step=16,
                                                           label="Diffusion tile overlap")

                        with gr.Tab("Describe"):
                            describe_img = _crop_input("Image to describe", 280)
                            describe_btn = gr.Button("Describe -> prompt", variant="primary", size="sm")
                            describe_status = gr.Markdown(
                                "*Uses the Ollama vision model selected in Advanced > Prompt AI "
                                "(or the local BLIP fallback if Ollama is off).*")

                        with gr.Tab("Vision Mix"):
                            gr.Markdown("*Vision Mix: a vision model looks at your reference images "
                                        "and an LLM blends them into ONE text prompt (e.g. a person + "
                                        "an outfit + a setting), then Z-Image generates from it. "
                                        "Needs Ollama with a vision model (Advanced > Prompt AI). "
                                        "It mixes ideas/style, not exact pixels.*")
                            with gr.Row():
                                cref1 = _crop_input("Ref 1", 180)
                                cref2 = _crop_input("Ref 2", 180)
                            with gr.Row():
                                cref3 = _crop_input("Ref 3", 180)
                                cref4 = _crop_input("Ref 4", 180)
                            with gr.Row():
                                compose_btn = gr.Button("Vision Mix -> prompt", size="sm")
                                vmix_gen_btn = gr.Button("Vision Mix & Generate", variant="primary",
                                                         size="sm")
                            compose_status = gr.Markdown("")

                        with gr.Tab("Remove BG"):
                            rembg_img = _crop_input("Image", 280)
                            rembg_btn = gr.Button("Remove background", variant="primary", size="sm")
                            rembg_status = gr.Markdown("*Local (rembg). Output = transparent PNG. "
                                                       "First use downloads the u2net model.*")

                        with gr.Tab("Reframe (outpaint)"):
                            gr.Markdown("*Expand the image to a new aspect ratio; Z-Image fills "
                                        "the new borders (inpaint). The prompt guides the fill.*")
                            reframe_img = _crop_input("Image", 260)
                            with gr.Row():
                                reframe_ratio = gr.Dropdown(
                                    ["16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "1:1", "21:9"],
                                    value="16:9", label="Target ratio")
                                reframe_steps = gr.Slider(4, 30, value=12, step=1, label="Fill steps")
                            reframe_btn = gr.Button("Reframe / Outpaint", variant="primary", size="sm")
                            reframe_status = gr.Markdown("")

                        with gr.Tab("Inpaint"):
                            gr.Markdown("*Paint the area to change (white brush), describe what "
                                        "should appear in the prompt, then Inpaint.*")
                            inpaint_editor = gr.ImageEditor(
                                type="pil", label="Image + mask (paint the area)",
                                brush=gr.Brush(colors=["#ffffff"], color_mode="fixed"),
                                layers=False, transforms=[], height=380)
                            with gr.Row():
                                inpaint_steps = gr.Slider(4, 40, value=20, step=1, label="Steps")
                                inpaint_denoise = gr.Slider(0.3, 1.0, value=0.85, step=0.05,
                                                            label="Inpaint strength")
                            inpaint_btn = gr.Button("Inpaint", variant="primary", size="sm")
                            inpaint_status = gr.Markdown("")

                        with gr.Tab("Reference (Omni)", visible=omni_on):
                            gr.Markdown("*Compose from up to 4 reference images + a prompt. "
                                        "Set **Input mode = Reference (Omni)** above. "
                                        "Uses width/height/steps/guidance from Settings.*")
                            with gr.Row():
                                omni_check_btn2 = gr.Button("Check Omni model availability", size="sm")
                                omni_status2 = gr.Markdown("")
                            with gr.Row():
                                ref1 = _crop_input("Ref 1", 220)
                                ref2 = _crop_input("Ref 2", 220)
                            with gr.Row():
                                ref3 = _crop_input("Ref 3", 220)
                                ref4 = _crop_input("Ref 4", 220)

                        with gr.Tab("Face Swap"):
                            gr.Markdown("*Post-process: replace the face in the result with this "
                                        "source face. Works on any mode (txt2img / img2img / omni). "
                                        "Needs `insightface` + `onnxruntime-gpu` installed and "
                                        "`faceswap_model_path` (inswapper .onnx) set in config.txt.*")
                            faceswap_src = _crop_input("Source face", 240)
                            faceswap_enable = gr.Checkbox(value=False, label="Apply face swap to result")
                            faceswap_restore_cb = gr.Checkbox(
                                value=cz_face.FACESWAP_RESTORE,
                                label="Restore face (GFPGAN) - fixes the soft 128px swap",
                                info="Sharpens the swapped face. Downloads gfpgan_1.4.onnx on first "
                                     "use (faceswap_restore_url in config.txt).")
                            faceswap_restore_blend = gr.Slider(0.0, 1.0, value=float(cz_face.FACESWAP_RESTORE_BLEND),
                                                               step=0.05, label="Restore strength")
                            faceswap_restore_status = gr.Markdown("")

            # ===== Colonne Advanced (a droite, masquee par defaut comme Fooocus) =====
            with gr.Column(scale=2, visible=False) as advanced_col:
                with gr.Tabs():
                    with gr.Tab("Settings"):
                        performance = gr.Radio(list(PERFORMANCE),
                                               value=CONFIG.get("default_performance", "Turbo (8 steps)"),
                                               label="Performance",
                                               info="Sets steps + guidance. Turbo = your Turbo model; "
                                                    "Base CFG = for a Z-Image Base checkpoint.")
                        aspect = gr.Dropdown(list(ASPECT_RATIOS),
                                             value=CONFIG.get("default_aspect_ratio", "1024 x 1024  (1:1)"),
                                             label="Aspect ratio")
                        with gr.Row():
                            width = gr.Slider(256, 2048, value=int(CONFIG.get("default_width", 1024)),
                                              step=16, label="Width")
                            height = gr.Slider(256, 2048, value=int(CONFIG.get("default_height", 1024)),
                                               step=16, label="Height")
                        gen_steps = gr.Slider(2, 40, value=int(CONFIG.get("default_gen_steps", 8)),
                                              step=1, label="Generation steps (txt2img)")
                        with gr.Row():
                            guidance = gr.Slider(0.0, 8.0, value=float(CONFIG.get("default_guidance", 0.0)),
                                                 step=0.5, label="CFG guidance", scale=2,
                                                 info="0 = Z-Image Turbo. Z-Image Base: ~3.5-5.")
                            sampler_dd = gr.Dropdown(
                                list(SAMPLER_CHOICES),
                                value=(CONFIG.get("default_sampler") or "euler").strip().lower()
                                if (CONFIG.get("default_sampler") or "euler").strip().lower() in SAMPLER_CHOICES
                                else "euler",
                                label="Sampler", scale=1,
                                info="euler = native flow. unipc = UniPC. (DPM++/DPM2a impossible: "
                                     "Z-Image forces custom sigmas.)")
                            schedule_dd = gr.Dropdown(
                                list(SCHEDULE_CHOICES),
                                value=(CONFIG.get("default_schedule") or "sgm_uniform").strip().lower()
                                if (CONFIG.get("default_schedule") or "sgm_uniform").strip().lower() in SCHEDULE_CHOICES
                                else "sgm_uniform",
                                label="Schedule", scale=1,
                                info="sigma schedule (ComfyUI-style). sgm_uniform = native Z-Image. "
                                     "beta/karras/exponential remap the sigmas.")
                        image_number = gr.Slider(1, 30, value=int(CONFIG.get("default_image_number", 1)),
                                                 step=1, label="Image number (batch)")
                        seed = gr.Number(value=int(CONFIG.get("default_seed", -1)),
                                         label="Seed (-1 = random)", precision=0)

                    with gr.Tab("Styles"):
                        style_search = gr.Textbox(show_label=False, container=False,
                                                  placeholder="Search styles... (e.g. anime, cinematic, sai)")
                        style_random = gr.Checkbox(
                            value=False, label="Random style each image",
                            info="Each render picks a random style from the selected ones "
                                 "(or from ALL styles if none selected).")
                        gr.Markdown("*Hover a style to preview it.*")
                        styles = gr.CheckboxGroup(list(STYLES), value=CONFIG.get("default_styles", []),
                                                  label="Styles (combinable)", elem_id="cz_styles")

                    with gr.Tab("Prompt AI"):
                        ollama_url = gr.Textbox(value=OLLAMA_URL, label="Ollama URL",
                                                info="Local LLM server. Used for Describe (vision) "
                                                     "and Improve prompt.")
                        detect_btn = gr.Button("Detect Ollama (vision models)", size="sm", variant="primary")
                        ollama_model = gr.Dropdown([], label="Vision model (Describe / Improve)",
                                                   interactive=True)
                        ollama_status = gr.Markdown("*Click Detect. If Ollama is off, Describe falls "
                                                    "back to a local BLIP captioner.*")
                        gr.Markdown("---")
                        log_level_dd = gr.Dropdown(["quiet", "info", "debug"],
                                                   value={0: "quiet", 1: "info", 2: "debug"}.get(cz_core.LOG_LEVEL, "info"),
                                                   label="Console log level (dev)",
                                                   info="debug = full params, pipe state, VRAM in the .bat console.")
                        log_level_status = gr.Markdown("")

                    with gr.Tab("Models"):
                        zimage_model_tb = gr.Textbox(
                            value=cz_pipeline.BASE_REPO,
                            label="Z-Image base (HF repo, diffusers folder, or .safetensors file)",
                            info="A .safetensors (Civitai) = transformer; VAE+encoder from base repo.")
                        esrgan_dir_tb = gr.Textbox(value=cz_esrgan.ESRGAN_DIR, label="ESRGAN_DIR (.pth/.safetensors folder)")
                        offload = gr.Dropdown(choices=list(cz_pipeline.OFFLOAD_CHOICES), value="none",
                                              label="CPU offload (VRAM)",
                                              info="none | model (~half) | sequential (~9GB, slower)")
                        with gr.Row():
                            refresh_btn = gr.Button("Refresh ESRGAN", size="sm")
                            apply_zimage_btn = gr.Button("Apply Z-Image", size="sm", variant="primary")
                            save_paths_btn = gr.Button("Save paths", size="sm")
                        paths_status = gr.Markdown("")

                        gr.Markdown("### Checkpoints (switch model, like ESRGAN)")
                        ckpt_dir_tb = gr.Textbox(value=cz_pipeline.CHECKPOINTS_DIR, label="Checkpoints folder")
                        with gr.Row():
                            ckpt_dd = gr.Dropdown(choices=["(base repo)"] + list_checkpoints(),
                                                  value="(base repo)", label="Z-Image checkpoint", scale=3)
                            ckpt_refresh_btn = gr.Button("Refresh", size="sm", scale=1)
                        ckpt_status = gr.Markdown("")
                        with gr.Row():
                            transformer_tb = gr.Textbox(
                                value="", scale=3,
                                label="Transformer override (HF repo / diffusers folder)",
                                placeholder="e.g. RunDiffusion/Juggernaut-Z-Image",
                                info="For community models with an incomplete tokenizer (Juggernaut-Z): "
                                     "loads only the transformer, keeps base VAE/encoder. Set base = Turbo.")
                            transformer_apply_btn = gr.Button("Apply", size="sm", scale=1, variant="primary")

                        gr.Markdown("### LoRA (up to 3, combinable)")
                        lora_dir_tb = gr.Textbox(value=cz_pipeline.LORAS_DIR, label="LoRA folder")
                        _lchoices = ["None"] + list_loras()
                        with gr.Row():
                            lora_dd1 = gr.Dropdown(choices=_lchoices, value="None", label="LoRA 1", scale=3)
                            lw1 = gr.Slider(0.0, 2.0, value=float(cz_pipeline.LORA_WEIGHT), step=0.05,
                                            label="Weight 1", scale=2)
                        with gr.Row():
                            lora_dd2 = gr.Dropdown(choices=_lchoices, value="None", label="LoRA 2", scale=3)
                            lw2 = gr.Slider(0.0, 2.0, value=float(cz_pipeline.LORA_WEIGHT), step=0.05,
                                            label="Weight 2", scale=2)
                        with gr.Row():
                            lora_dd3 = gr.Dropdown(choices=_lchoices, value="None", label="LoRA 3", scale=3)
                            lw3 = gr.Slider(0.0, 2.0, value=float(cz_pipeline.LORA_WEIGHT), step=0.05,
                                            label="Weight 3", scale=2)
                        lora_refresh_btn = gr.Button("Refresh LoRA list", size="sm")
                        lora_keywords_tb = gr.Textbox(label="Keywords / trigger words", lines=2,
                                                      placeholder="Auto-filled from the selected LoRA(s).")
                        with gr.Row():
                            lora_kw_btn = gr.Button("Get keywords", size="sm")
                            lora_kw_to_prompt_btn = gr.Button("Add to prompt", size="sm", variant="primary")
                        lora_status = gr.Markdown("")

                        gr.Markdown("### \U0001F3B2 Wildcards")
                        gr.Markdown("*`__name__` in the prompt -> a random line from name.txt "
                                    "(nested, reproducible per seed). Pick a file to view/edit it, "
                                    "Insert to add it to the prompt, or Create a new one.*")
                        wild_dir_tb = gr.Textbox(value=cz_prompt.WILDCARDS_DIR, label="Wildcards folder")
                        with gr.Row():
                            wild_dd = gr.Dropdown(["None"] + list_wildcards(), value="None",
                                                  label="Wildcard file", scale=3)
                            wild_refresh_btn = gr.Button("Refresh", size="sm", scale=1, min_width=90)
                            wild_insert_btn = gr.Button("Insert __name__", size="sm", scale=1,
                                                        variant="primary", min_width=140)
                        wild_editor = gr.Textbox(label="Contents (one option per line)", lines=8,
                                                 placeholder="Select a file above to view/edit, "
                                                             "or type lines for a new one.")
                        with gr.Row():
                            wild_save_btn = gr.Button("Save", size="sm")
                            wild_new_name = gr.Textbox(show_label=False, scale=2, container=False,
                                                       placeholder="new_wildcard_name (no extension)")
                            wild_create_btn = gr.Button("Create new", size="sm", variant="primary")
                        wild_status = gr.Markdown("")

                        gr.Markdown("### Omni / Edit model (multi-reference)")
                        gr.Markdown("*The Reference (Omni) tab stays hidden until a model is set "
                                    "here (then restart). Z-Image-Omni-Base / Z-Image-Edit are not "
                                    "released yet. For a reference image now, use img2img.*")
                        omni_model_tb = gr.Textbox(value=CONFIG.get("zimage_omni_model", ""),
                                                   label="Omni model (HF repo or local folder)",
                                                   info="Needs SigLIP. Set it then restart to enable "
                                                        "the Reference (Omni) tab.")
                        omni_check_btn = gr.Button("Check Omni availability (Hugging Face)", size="sm")
                        omni_status = gr.Markdown("")

                        gr.Markdown("### \U0001F5BCï¸ Asset Browser")
                        gr.Markdown("*Standalone gallery page (thumbnails + metadata + lightbox) "
                                    "built into the output folder, opened in a new tab. Reindex to "
                                    "(re)build it from your saved images.*")
                        with gr.Row():
                            ab_thumb_size = gr.Slider(96, 512, value=int(_ab_get("thumbnail_size")),
                                                      step=32, label="Thumbnail size")
                            ab_quality = gr.Slider(40, 100, value=int(_ab_get("thumbnail_quality")),
                                                   step=5, label="Thumbnail quality")
                        with gr.Row():
                            ab_blur = gr.Checkbox(value=bool(_ab_get("blur_thumbnails")),
                                                  label="Blur thumbnails (NSFW)")
                            ab_gen_thumbs = gr.Checkbox(value=bool(_ab_get("generate_thumbnails")),
                                                        label="Generate thumbnails")
                        ab_reindex_btn = gr.Button("Rebuild ALL thumbnails (force) + open link",
                                                   variant="primary", size="sm")
                        ab_open_link = gr.HTML("")
                        ab_status = gr.Markdown("")

                    with gr.Tab("Save"):
                        save_mode = gr.Radio(choices=["display", "local", "alongside", "custom"],
                                             value=DEFAULT_SAVE_MODE, label="Save mode")
                        output_dir = gr.Textbox(value=DEFAULT_OUTPUT_DIR, label="Output folder")
                        output_format = gr.Dropdown(choices=list(SUPPORTED_FORMATS),
                                                    value=DEFAULT_OUTPUT_FORMAT, label="Output format")

        # Toggles facon Fooocus
        advanced_cb.change(lambda v: gr.update(visible=bool(v)), advanced_cb, advanced_col)
        use_input.change(lambda v: gr.update(visible=bool(v)), use_input, input_group)
        aspect.change(_set_aspect, [aspect], [width, height])
        performance.change(_set_performance, [performance], [gen_steps, guidance])
        style_search.change(_filter_styles, [style_search, styles], [styles])

        # Actions
        refresh_btn.click(_refresh_models, [esrgan_dir_tb], [esrgan, paths_status])
        apply_zimage_btn.click(_apply_zimage, [zimage_model_tb], [paths_status])
        save_paths_btn.click(_save_paths_to_prefs,
                             [esrgan_dir_tb, zimage_model_tb, ckpt_dir_tb, lora_dir_tb, wild_dir_tb],
                             [paths_status])
        wild_refresh_btn.click(_ui_wild_refresh, [wild_dir_tb], [wild_dd, wild_status])
        wild_dd.change(_ui_wild_load, [wild_dd], [wild_editor, wild_status])
        wild_insert_btn.click(_ui_wild_insert, [wild_dd, prompt], [prompt, wild_status])
        wild_save_btn.click(_ui_wild_save, [wild_dd, wild_editor], [wild_status])
        wild_create_btn.click(_ui_wild_create, [wild_new_name, wild_editor],
                              [wild_dd, wild_status, wild_new_name])
        ckpt_refresh_btn.click(_refresh_checkpoints, [ckpt_dir_tb], [ckpt_dd, ckpt_status])
        ckpt_dd.change(_apply_checkpoint, [ckpt_dd], [ckpt_status, gen_steps, guidance])
        transformer_apply_btn.click(_apply_transformer_repo, [transformer_tb],
                                    [ckpt_status, gen_steps, guidance])
        lora_refresh_btn.click(_refresh_loras, [lora_dir_tb],
                               [lora_dd1, lora_dd2, lora_dd3, lora_status])
        _lora_slots = [lora_dd1, lw1, lora_dd2, lw2, lora_dd3, lw3]
        for _c in (lora_dd1, lora_dd2, lora_dd3):
            _c.change(_ui_loras_apply, _lora_slots, [lora_status, lora_keywords_tb])
        for _c in (lw1, lw2, lw3):
            _c.change(_apply_loras, _lora_slots, [lora_status])
        lora_kw_btn.click(_ui_loras_keywords, [lora_dd1, lora_dd2, lora_dd3],
                          [lora_keywords_tb, lora_status])
        lora_kw_to_prompt_btn.click(_ui_kw_to_prompt, [prompt, lora_keywords_tb], [prompt])
        omni_model_tb.change(lambda r: (set_omni_model(r), f"Omni model set: {r or '(none)'}")[1],
                             [omni_model_tb], [omni_status])
        omni_check_btn.click(_ui_check_omni, None, [omni_status])
        omni_check_btn2.click(_ui_check_omni, None, [omni_status2])
        ab_reindex_btn.click(_ui_ab_reindex,
                             [output_dir, ab_thumb_size, ab_quality, ab_blur, ab_gen_thumbs],
                             [ab_open_link, ab_status])
        log_level_dd.change(set_log_level, [log_level_dd], [log_level_status])
        detect_btn.click(_ui_detect_ollama, [ollama_url], [ollama_model, ollama_status])
        describe_btn.click(_ui_describe, [describe_img, ollama_model, ollama_url], [prompt, describe_status])
        improve_btn.click(_ui_improve, [prompt, ollama_model, ollama_url], [prompt, improve_status])
        compose_btn.click(_ui_compose, [cref1, cref2, cref3, cref4, ollama_model, ollama_url],
                          [prompt, compose_status])
        rembg_btn.click(_ui_remove_bg, [rembg_img, history, save_mode, output_dir],
                        [out, report, history, history_gallery])
        faceswap_restore_cb.change(set_faceswap_restore,
                                   [faceswap_restore_cb, faceswap_restore_blend],
                                   [faceswap_restore_status])
        faceswap_restore_blend.change(set_faceswap_restore,
                                      [faceswap_restore_cb, faceswap_restore_blend],
                                      [faceswap_restore_status])
        reframe_btn.click(_ui_reframe,
                          [reframe_img, reframe_ratio, reframe_steps, prompt, guidance, offload,
                           seed, save_mode, output_dir, output_format, history],
                          [out, report, history, history_gallery])
        inpaint_btn.click(_ui_inpaint,
                          [inpaint_editor, prompt, negative, styles, guidance, offload,
                           inpaint_steps, inpaint_denoise, seed, save_mode, output_dir,
                           output_format, history],
                          [out, report, history, history_gallery])
        # Stop facon Fooocus: tourne en parallele du Generate (thread separe) et pose
        # le flag d'arret + interrompt la boucle de debruitage en cours.
        stop_btn.click(request_stop, None, [report])
        clear_hist_btn.click(_ui_clear_history, None, [history, history_gallery])
        load_out_btn.click(_ui_load_outputs, [output_dir], [history, history_gallery])
        # Galerie du dossier de sortie -> Asset Browser dans un nouvel onglet
        gallery_btn.click(_ui_gallery_open, [output_dir], [gallery_status, gallery_url]).then(
            None, [gallery_url], None, js="(u) => { if (u) window.open(u, '_blank'); }")
        preset.change(_apply_preset, [preset],
                      [factor, denoise, refine_steps, tile, overlap, refine_tile, refine_overlap, offload])
        # Sampler (euler/unipc) + schedule (sgm_uniform/beta/karras/exp): applique le
        # scheduler choisi aux pipes en cache (pas de rechargement).
        sampler_dd.change(set_sampler, [sampler_dd], None)
        schedule_dd.change(set_schedule, [schedule_dd], None)
        _gen_inputs = [prompt, negative, styles, style_random, use_input, inp, input_mode,
                       ref1, ref2, ref3, ref4,
                       faceswap_enable, faceswap_src,
                       width, height, gen_steps, image_number,
                       seed, guidance, offload, esrgan, do_esrgan_cb, do_refine_cb, refine_first_cb, factor, denoise, refine_steps,
                       tile, overlap, refine_tile, refine_overlap, save_mode, output_dir, output_format,
                       history]
        _gen_outputs = [out, report, history, history_gallery]
        btn.click(_ui_generate, inputs=_gen_inputs, outputs=_gen_outputs)
        # Clic sur une image du resultat -> bouton Download avec le vrai nom de fichier.
        out.select(_pick_download, None, [result_dl])
        # Vision Mix & Generate: fusionne les refs en un prompt, puis genere (txt2img).
        vmix_gen_btn.click(
            _ui_compose, [cref1, cref2, cref3, cref4, ollama_model, ollama_url],
            [prompt, compose_status]
        ).then(_ui_generate, inputs=_gen_inputs, outputs=_gen_outputs)
    return demo

