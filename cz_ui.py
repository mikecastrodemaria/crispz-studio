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
import csv
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
    APP_VERSION,
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
    set_hf_token, hf_token_is_set,
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
    set_wildcards_dir, set_wildcards_in_order,
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
    _local_improve,
)


# Caption local BLIP (fallback Ollama), FaceSwap (InsightFace/inswapper) + restore
# GFPGAN, detourage rembg -> cz_face.py. L'etat mutable (caches modeles + reglages
# restore) vit dans le module.
import cz_face
from cz_face import (  # noqa: E402,F401
    _local_caption, _remove_bg, set_faceswap_restore,
    set_caption_model, _current_caption_kind,
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
import cz_civitai
from cz_pipeline import (  # noqa: E402,F401
    set_guidance, request_stop, set_zimage_model, set_zimage_transformer,
    list_checkpoints, list_loras, set_checkpoints_dir, set_checkpoints_extra_dir,
    resolve_checkpoint, set_loras_dir, lora_keywords,
    set_omni_model, check_omni_available, set_offload_mode, free_vram, set_loras,
    set_sampler, SAMPLER_CHOICES, set_schedule, SCHEDULE_CHOICES, set_force_ratio,
    generate, generate_omni, inpaint_run, outpaint, outpaint_directions, reframe,
    txt2img_run, process_one, round_to_multiple, _reframe_canvas, _gen_meta,
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
    save_image, _list_output_files, _read_image_meta, set_metadata_scheme,
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
                                        do_esrgan=do_esrgan, refine_first=refine_first,
                                        apply_force_ratio=True)
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
                            do_esrgan=do_esrgan, refine_first=refine_first,
                            apply_force_ratio=True)
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


# Repos de base officiels Z-Image, proposes directement dans le dropdown
# "Z-Image checkpoint" (selectionner = swap complet du BASE_REPO).
ZIMAGE_BASE_REPOS = ["Tongyi-MAI/Z-Image-Turbo", "Tongyi-MAI/Z-Image"]
# Preset Performance par defaut pour chaque repo de base officiel: Turbo (distille,
# guidance 0) vs Base (a besoin d'une vraie CFG + plus de steps). Le nom du repo de base
# ("...Z-Image") ne contient pas "base", donc on mappe explicitement plutot que par
# substring. steps/guidance sont ensuite tires du preset lui-meme (source unique).
ZIMAGE_BASE_PERFORMANCE = {
    "Tongyi-MAI/Z-Image-Turbo": "Turbo (8 steps)",
    "Tongyi-MAI/Z-Image": "Base CFG (28 steps)",
}


def _refresh_checkpoints(new_dir, extra_dir=""):
    """Change le(s) dossier(s) checkpoints (principal + extra) + liste les modeles fusionnes
    + persiste."""
    set_checkpoints_dir(new_dir)
    set_checkpoints_extra_dir(extra_dir)
    try:
        _save_prefs_keys({"checkpoints_dir": cz_pipeline.CHECKPOINTS_DIR,
                          "checkpoints_extra_dir": cz_pipeline.CHECKPOINTS_EXTRA_DIR})
    except Exception:
        pass
    cks = list_checkpoints()
    n_new = _ensure_model_presets(cks)  # preset basique pour tout nouveau modele local
    locs = " + ".join(d for d in (cz_pipeline.CHECKPOINTS_DIR, cz_pipeline.CHECKPOINTS_EXTRA_DIR) if d)
    msg = f"{len(cks)} checkpoint(s) in {locs} (saved)."
    if n_new:
        msg += f" +{n_new} preset(s) auto-created."
    # 3e sortie: rafraichit le menu Presets (nouveaux modeles -> nouveaux presets).
    return (gr.update(choices=ZIMAGE_BASE_REPOS + cks), msg, gr.update(choices=list_presets()))


def _performance_label_for(steps, guidance):
    """Nom du preset Performance correspondant a (steps, guidance), sinon None.
    Data-driven: respecte les presets surcharges via config.txt (performance_presets)."""
    for name, (s, gg) in PERFORMANCE.items():
        if int(s) == int(steps) and abs(float(gg) - float(guidance)) < 1e-3:
            return name
    return None


def _perf_update(steps, guidance):
    """gr.update pour le radio Performance correspondant a (steps, guidance), sinon no-op
    (laisse le choix courant si aucun preset ne matche exactement)."""
    label = _performance_label_for(steps, guidance)
    return gr.update(value=label) if label else gr.update()


def _apply_checkpoint(name):
    """Selectionne soit un repo de base officiel Z-Image (swap complet du BASE_REPO),
    soit un checkpoint single-file local (transformer override, VAE/encoder du base repo).
    Ajuste aussi steps/guidance ET le preset Performance selon le profil du modele."""
    if not name:
        return (gr.update(), gr.update(), gr.update(), gr.update())
    if name in ZIMAGE_BASE_REPOS:
        # Repo de base complet -> on enleve tout transformer single-file puis on swap le base.
        set_zimage_transformer("")
        set_zimage_model(name)
        perf = ZIMAGE_BASE_PERFORMANCE.get(name)
        if perf and perf in PERFORMANCE:
            st, g = PERFORMANCE[perf]
            perf_upd = gr.update(value=perf)
        else:
            st, g = profile_for_model(name)
            perf_upd = _perf_update(st, g)
        return (f"Z-Image base: {name} -> {perf or 'auto'} (steps={st}, CFG={g}, reload on next run).",
                gr.update(value=st), gr.update(value=g), perf_upd)
    path = resolve_checkpoint(name)
    set_zimage_transformer(path)
    st, g = profile_for_model(os.path.basename(path))
    return (f"Z-Image transformer: {os.path.basename(path)} -> auto steps={st}, CFG={g} "
            f"(reload on next run).", gr.update(value=st), gr.update(value=g), _perf_update(st, g))


def _apply_transformer_repo(repo):
    """Definit le transformer depuis un repo HF / dossier diffusers OU un .safetensors.
    Ajuste steps/guidance ET le preset Performance selon le profil du modele.
    Champ VIDE = no-op: on ne remet PAS a zero (sinon ce bouton effacerait le checkpoint
    choisi juste au-dessus). Pour revenir au base repo pur, choisir un repo officiel dans
    'Z-Image checkpoint'."""
    repo = (repo or "").strip()
    if not repo:
        return ("Transformer override is empty — no change. Pick a model in 'Z-Image "
                "checkpoint' above (that also clears any override).",
                gr.update(), gr.update(), gr.update())
    set_zimage_transformer(repo)
    st, g = profile_for_model(repo)
    return (f"Transformer override: {repo} -> auto steps={st}, CFG={g} "
            f"(keeps base VAE/encoder; reload on next run).",
            gr.update(value=st), gr.update(value=g), _perf_update(st, g))


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


# Nombre de slots LoRA affiches (configurable via config 'lora_slots', 1..10, defaut 3).
MAX_LORA_SLOTS = 10
LORA_SLOTS = max(1, min(MAX_LORA_SLOTS, int(_prefs.get("lora_slots", CONFIG.get("lora_slots", 3)))))


def _ui_set_lora_slots(n):
    """Regle le nombre de slots LoRA VISIBLES (Advanced) + persiste (preferences.json).
    Les slots au-dela restent presents mais caches (valeur 'None' -> ignores)."""
    n = max(1, min(MAX_LORA_SLOTS, int(n)))
    try:
        _save_prefs_keys({"lora_slots": n})
    except Exception:
        pass
    return [gr.update(visible=(i < n)) for i in range(MAX_LORA_SLOTS)]


def _refresh_loras(new_dir):
    """Change le dossier loras + rafraichit TOUS les slots (N configurable) + persiste."""
    set_loras_dir(new_dir)
    try:
        _save_prefs_keys({"loras_dir": cz_pipeline.LORAS_DIR})   # persiste -> survit au reboot
    except Exception:
        pass
    lr = ["None"] + list_loras()
    status = f"{len(lr) - 1} LoRA(s) in {cz_pipeline.LORAS_DIR} (saved)."
    return tuple(gr.update(choices=lr) for _ in range(MAX_LORA_SLOTS)) + (status,)


def _apply_loras(*vals):
    """Applique la combinaison des slots LoRA. vals = (name1, weight1, name2, weight2, ...)."""
    pairs = [(vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)]
    set_loras(pairs)
    if not cz_pipeline.LORAS:
        return "LoRA: none."
    return "LoRA: " + ", ".join(f"{os.path.basename(p)}@{w}" for p, w in cz_pipeline.LORAS) + " (reload on next run)."


def _path_for_lora(name):
    if not name or name in ("None", "none", ""):
        return None
    return name if os.path.isabs(name) else os.path.join(cz_pipeline.LORAS_DIR, name)


def _lora_keywords_for(names):
    """Agrege les mots-cles (trigger words) des LoRA selectionnees."""
    kws = []
    for n in names:
        p = _path_for_lora(n)
        if p:
            k = lora_keywords(p)
            if k:
                kws.append(k)
    return ", ".join(kws)


def _ui_loras_apply(*vals):
    """Applique les slots (N) + agrege les mots-cles des LoRA selectionnees."""
    status = _apply_loras(*vals)
    return status, _lora_keywords_for(vals[0::2])


def _ui_loras_keywords(*names):
    """Recupere les mots-cles de toutes les LoRA selectionnees (bouton)."""
    merged = _lora_keywords_for(names)
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


def _save_hf_token(token):
    """Onglet Models: pose + persiste le token HF (preferences.json, gitignore). Vide le
    champ apres coup (le token n'est jamais re-affiche)."""
    token = (token or "").strip()
    if not token:
        return gr.update(), ("No change. Enter a token (it won't be shown), or it stays as-is."
                             if hf_token_is_set() else "Enter a Hugging Face token to save.")
    set_hf_token(token)
    return gr.update(value=""), ("✅ HF token saved (preferences.json, gitignored) and applied. "
                                 "Gated downloads (e.g. FLUX.1-Krea-dev) will now authenticate.")


def _save_civitai_key(token):
    """Advanced: pose + persiste la cle CivitAI (preferences.json). Utilisee par
    'Fetch from CivitAI' dans l'Asset Browser. Vide le champ apres coup."""
    token = (token or "").strip()
    if not token:
        return gr.update(), ("✅ A CivitAI key is set." if cz_civitai.API_KEY
                             else "Enter a CivitAI API key to save.")
    cz_civitai.set_api_key(token)
    try:
        _save_prefs_keys({"civitai_api_key": token})
    except Exception:
        pass
    return gr.update(value=""), "✅ CivitAI key saved (preferences.json) and applied."


def _save_paths_to_prefs(esrgan_dir, checkpoints_dir=None, checkpoints_extra_dir=None,
                         loras_dir=None, wildcards_dir=None):
    """Persiste les chemins dans preferences.json (local) -> charges au prochain boot.
    Le base repo Z-Image courant (choisi via le dropdown) est persiste tel quel."""
    set_esrgan_dir(esrgan_dir)
    if checkpoints_dir:
        set_checkpoints_dir(checkpoints_dir)
    if checkpoints_extra_dir is not None:
        set_checkpoints_extra_dir(checkpoints_extra_dir)
    if loras_dir:
        set_loras_dir(loras_dir)
    if wildcards_dir:
        set_wildcards_dir(wildcards_dir)
    _save_prefs_keys({"esrgan_dir": cz_esrgan.ESRGAN_DIR, "zimage_model": cz_pipeline.BASE_REPO,
                      "checkpoints_dir": cz_pipeline.CHECKPOINTS_DIR,
                      "checkpoints_extra_dir": cz_pipeline.CHECKPOINTS_EXTRA_DIR,
                      "loras_dir": cz_pipeline.LORAS_DIR,
                      "wildcards_dir": cz_prompt.WILDCARDS_DIR})
    return (f"Saved to {PREFS_PATH}: esrgan_dir, zimage_model, checkpoints_dir, "
            f"checkpoints_extra_dir, loras_dir, wildcards_dir={cz_prompt.WILDCARDS_DIR}")


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


def _ui_set_force_ratio(on, aspect_name):
    """UI: (dés)active le ratio force pour Upscale/img2img. ON -> crop l'entree au ratio
    de l'Aspect ratio choisi ; OFF -> ratio natif preserve. Pose l'etat dans cz_pipeline."""
    set_force_ratio(aspect_name if on else "")


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
                "Ollama not reachable. Describe will use the local captioner fallback. "
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
        return gr.update(value=_local_caption(image)), "Described via local captioner (no Ollama model)."
    except Exception as e:
        return gr.update(), f"No Ollama model selected and local captioner failed: {e}"


def _ui_improve(prompt_text, model, url):
    """Ameliore le prompt courant. Ollama si un modele est choisi, sinon fallback LOCAL
    (rule-based, sans Ollama): ajoute des mots-cles de qualite."""
    if not (prompt_text or "").strip():
        return gr.update(), "Type a prompt first."
    if not model:
        return gr.update(value=_local_improve(prompt_text)), \
            "Improved locally (quality tags, no Ollama). Pick a model in Advanced > Prompt AI for a full rewrite."
    try:
        return gr.update(value=_ollama_improve(prompt_text, model, base=url)), f"Improved via {model}."
    except Exception as e:
        return gr.update(value=_local_improve(prompt_text)), f"Ollama failed ({e}); improved locally instead."


def _ui_set_caption_model(kind):
    """Change le captioner local + persiste le choix dans preferences.json."""
    k = set_caption_model(kind)
    try:
        _save_prefs_keys({"caption_model": k})
    except Exception as e:
        _dbg(f"save caption_model pref failed: {e}")
    return f"Caption model set to **{k}** (saved; loads on next use)."


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


def _ui_edit(mode, editor_value, dirs, ratio, fit, auto_describe, harmonize, harmonize_denoise,
             prompt, negative, styles, guidance, offload_mode, steps, strength, seed, save_mode,
             output_dir, output_format, history, progress=gr.Progress(track_tqdm=True)):
    """Onglet unifie Inpaint / Outpaint / Reframe. Le `mode` choisit l'operation; l'image
    (editeur), le prompt, les `steps` (du modele) et `strength` sont partages.
      - Brush       : inpaint de la zone peinte (inpaint_run)
      - Expand sides: outpaint directionnel L/R/T/B (outpaint_directions)
      - Reframe     : recadrage au ratio, ~1 MP, Contain/Cover (reframe)
    auto_describe (outpaint/reframe): decrit l'image centrale via BLIP local (sans Ollama)
    pour guider le remplissage des bords de facon coherente."""
    cz_pipeline._PROGRESS = lambda f, d: progress(f, desc=d)
    try:
        bg, mask = _editor_to_image_mask(editor_value)
        if bg is None:
            return [], "Load an image first.", history, history
        set_offload_mode(offload_mode)
        set_guidance(guidance)
        m = str(mode).lower()
        eff_prompt = prompt or ""
        # Auto-describe (captioner local, pas d'Ollama): utile pour outpaint/reframe -> le
        # modele "voit" le sujet du centre et prolonge la scene au lieu d'inventer.
        # Implicite si le prompt est VIDE; force aussi si la case est cochee (prefixe).
        if not m.startswith("brush") and (auto_describe or not eff_prompt.strip()):
            try:
                progress(0.05, desc="Describing center (caption model; first use downloads it)...")
                cap = _local_caption(bg)
                eff_prompt = (cap + (", " + eff_prompt if eff_prompt.strip() else "")).strip()
                _log(f"auto-describe: {cap}")
            except Exception as e:
                _log(f"auto-describe failed (continuing without): {e}")
        full_prompt, _ = _apply_styles(eff_prompt, negative, styles)
        painted = mask is not None and mask.getbbox() is not None
        try:
            if m.startswith("reframe"):
                try:
                    rw, rh = [int(x) for x in str(ratio).split(":")]
                except Exception:
                    rw, rh = 16, 9
                fit_mode = "cover" if str(fit).lower().startswith("cover") else "contain"
                res = reframe(bg, rw, rh, fit_mode, full_prompt, steps, seed, strength=strength)
                tag, info = "reframe", f"Reframe {ratio} ({fit_mode})"
            elif m.startswith("expand"):
                d = [x.lower() for x in (dirs or [])]
                if not d:
                    return [], "Pick at least one side (or Center).", history, history
                res = outpaint_directions(bg, mask if painted else None, d,
                                          full_prompt, steps, seed, strength=strength)
                tag, info = "outpaint", f"Outpaint {'+'.join(d)}"
            else:  # Brush -> inpaint
                if not painted:
                    return [], "Paint the area to change (brush).", history, history
                res = inpaint_run(bg, mask, full_prompt, steps, strength, seed)
                tag, info = "inpaint", "Inpaint"
        except Exception as e:
            _log(f"{mode} error: {e}")
            return [], f"{mode} failed: {e}", history, history
        # Harmonize: passe img2img legere (refine Z-Image, sans ESRGAN) sur TOUTE l'image
        # finale -> unifie grain/lumiere/raccord et efface l'effet "zone ajoutee". Low
        # denoise (~0.2) pour ne pas reinventer le sujet.
        if harmonize and float(harmonize_denoise) > 0.001:
            try:
                progress(0.9, desc="Harmonizing (img2img refine)...")
                hd = float(harmonize_denoise)
                # img2img: steps effectifs = base x denoise -> on releve la base pour
                # garder ~8 steps reels meme a bas denoise (sinon 1-2 steps = inutile).
                h_steps = min(40, max(int(steps), int(round(8.0 / max(hd, 0.05)))))
                res, _ = process_one(res, None, 1.0, hd, h_steps, full_prompt, seed, 512, 64,
                                     refine_tile=0, refine_overlap=64, do_esrgan=False)
                info += " + harmonize"
            except Exception as e:
                _log(f"harmonize failed (keeping edit): {e}")
        dst = None
        if save_mode != "display":
            try:
                dst = build_output_path(None, save_mode, output_dir, output_format,
                                        tag=tag, seed=seed, size=res.size)
                if dst:
                    save_image(res, dst, output_format, meta=_gen_meta(
                        tag, full_prompt, seed=seed, steps=steps, guidance=cz_pipeline.GUIDANCE,
                        size=res.size, styles=styles))
            except Exception as e:
                dst = None
                _dbg(f"save {tag} failed: {e}")
        item = _dl_path(res, dst)   # telechargement avec le vrai nom de fichier si sauve
        new_hist = ([item] + list(history or []))[:200]
        return [item], f"{info} -> {res.size[0]}x{res.size[1]}", new_hist, new_hist
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
from cz_assetbrowser import (_ab_get, _ab_resolve_dir, ab_reindex, ab_open_fast,  # noqa: E402,F401
                             ab_build_catalog, delete_asset)


# Assets statiques (SPA Asset Browser, JS d'interface, CSS) -> cz_assets.py
from cz_assets import ASSET_BROWSER_HTML, CZ_JS, FOOOCUS_CSS  # noqa: E402


# Reference vers le Blocks en cours (renseignee par build_ui), pour autoriser a la
# volee des dossiers de sortie choisis dans l'UI. Gradio fige `allowed_paths` au
# lancement, mais relit `blocks.allowed_paths` a CHAQUE requete de fichier -> on peut
# y ajouter le dossier courant au moment d'ouvrir l'Asset Browser (sans redemarrage).
_DEMO = None


def _allow_runtime_path(output_dir):
    """Ajoute le dossier de sortie resolu aux `allowed_paths` du Blocks en cours,
    afin que Gradio accepte de servir index.html / miniatures s'il a ete change dans
    l'UI apres le lancement. Sans effet si le Blocks n'est pas encore lance."""
    try:
        d = os.path.abspath(_ab_resolve_dir(output_dir))
        paths = getattr(_DEMO, "allowed_paths", None)
        if paths is not None and d not in paths:
            paths.append(d)
            _dbg(f"asset-browser: allowed runtime path {d}")
    except Exception as e:
        _dbg(f"allow runtime path failed: {e}")


def _ui_ab_reindex(output_dir, thumb_size, quality, blur, gen_thumbs):
    """Bouton FORCE: regenere TOUTES les miniatures (synchrone) + lien."""
    _allow_runtime_path(output_dir)
    try:
        n, idx, _ = ab_reindex(output_dir, thumb_size, quality, blur, gen_thumbs,
                               background_thumbs=False)
    except Exception as e:
        return "", f"Asset Browser reindex failed: {e}"
    url = "/gradio_api/file=" + os.path.abspath(idx).replace("\\", "/")
    link = (f'<a href="{url}" target="_blank" style="display:inline-block;padding:8px 14px;'
            f'background:#3b4356;color:#fff;border-radius:6px;text-decoration:none;">'
            f'\U0001F5BC️ Open Asset Browser ({n} images)</a>')
    return link, f"Rebuilt all thumbnails for {n} image(s)."


def _ui_gallery_open(output_dir):
    """Ouverture INSTANTANEE: ecrit index.html et ouvre l'onglet tout de suite, puis
    (re)indexe (manifest + miniatures) en tache de fond. La SPA charge le manifest
    existant immediatement et se rafraichit quand le nouvel index est pret."""
    _allow_runtime_path(output_dir)
    try:
        idx = ab_open_fast(output_dir, _ab_get("thumbnail_size"), _ab_get("thumbnail_quality"),
                           bool(_ab_get("blur_thumbnails")), bool(_ab_get("generate_thumbnails")))
    except Exception as e:
        return f"Gallery open failed: {e}", ""
    # Catalogue LoRAs / Models (onglets de l'Asset Browser) construit en tache de fond.
    try:
        threading.Thread(target=ab_build_catalog,
                         args=(output_dir, cz_pipeline.LORAS_DIR, cz_pipeline.CHECKPOINTS_DIR),
                         daemon=True).start()
    except Exception as e:
        _dbg(f"catalog build spawn failed: {e}")
    url = "/gradio_api/file=" + os.path.abspath(idx).replace("\\", "/")
    return "Opening Asset Browser in a new tab (indexing in background)...", url


def _asset_focus_url(kind, name):
    """Ouvre l'Asset Browser directement sur l'onglet 'loras' ou 'models', centre sur
    'name' (fiche = preview + trigger words + exemples) quand c'est un fichier local.
    Utilise par les icones a cote des dropdowns LoRA / checkpoint. Renvoie (status, url)."""
    import urllib.parse
    name = (name or "").strip()
    out_dir = DEFAULT_OUTPUT_DIR
    _allow_runtime_path(out_dir)
    try:
        idx = ab_open_fast(out_dir, _ab_get("thumbnail_size"), _ab_get("thumbnail_quality"),
                           bool(_ab_get("blur_thumbnails")), bool(_ab_get("generate_thumbnails")))
    except Exception as e:
        return f"Asset Browser open failed: {e}", ""
    # Catalogue construit SYNCHRONE ici (rapide: pas de hashing) pour que la cible soit
    # presente dans loras.json/models.json au moment ou la SPA se focalise dessus.
    try:
        ab_build_catalog(out_dir, cz_pipeline.LORAS_DIR, cz_pipeline.CHECKPOINTS_DIR)
    except Exception as e:
        _dbg(f"catalog build (focus) failed: {e}")
    focus = ""
    if kind == "loras":
        focus = name if name and name != "None" else ""            # list_loras -> chemin relatif
    else:  # models: repo HF de base = pas de fichier local -> pas de focus (onglet seul)
        if name and name not in ZIMAGE_BASE_REPOS:
            focus = os.path.basename(name)                          # catalogue models indexe par nom de fichier
    url = "/gradio_api/file=" + os.path.abspath(idx).replace("\\", "/") + "?src=" + kind
    if focus:
        url += "&focus=" + urllib.parse.quote(focus)
    tgt = f" → {focus}" if focus else ""
    return (f"Opening Asset Browser ({kind}{tgt})…", url)


# Registre des jobs CivitAI en cours (cle = chemin absolu du .safetensors). Chaque etat:
# {phase, frac (0..1 ou null), text, done, ok, message}. Ecrit par le thread de fetch,
# lu par _api_civitai_progress (polling depuis l'Asset Browser).
_CIVITAI_JOBS = {}
_CIVITAI_LOCK = threading.Lock()


def _civitai_job_set(key, **fields):
    with _CIVITAI_LOCK:
        st = _CIVITAI_JOBS.get(key) or {}
        st.update(fields)
        _CIVITAI_JOBS[key] = st


def _api_civitai_fetch(rel, kind):
    """API (Asset Browser): demarre l'enrichissement CivitAI d'un modele EN ARRIERE-PLAN
    et renvoie immediatement la cle du job. Le client interroge ensuite civitai_progress.
    Un thread execute le fetch (preview + trigger words + exemples), met a jour l'etat a
    chaque phase, puis reconstruit le catalogue LoRAs/Models."""
    try:
        import cz_civitai
        mdir = cz_pipeline.LORAS_DIR if kind == "loras" else cz_pipeline.CHECKPOINTS_DIR
        path = os.path.join(mdir, rel or "")
        key = os.path.abspath(path)
        _civitai_job_set(key, phase="start", frac=None, text="Starting…",
                         done=False, ok=False, message="")

        def _progress(phase, frac, text):
            _civitai_job_set(key, phase=phase, frac=frac, text=text, done=False)

        def _work():
            try:
                res = cz_civitai.fetch_civitai_for_model(path, progress=_progress)
                try:
                    ab_build_catalog(DEFAULT_OUTPUT_DIR, cz_pipeline.LORAS_DIR,
                                     cz_pipeline.CHECKPOINTS_DIR)
                except Exception as e:
                    _dbg(f"catalog rebuild after civitai fetch failed: {e}")
                _civitai_job_set(key, phase="done", frac=1.0, done=True,
                                 ok=bool(res.get("success")),
                                 text=res.get("message", "done"),
                                 message=res.get("message", "done"))
            except Exception as e:
                _civitai_job_set(key, phase="error", frac=None, done=True, ok=False,
                                 text=f"error: {e}", message=f"error: {e}")

        threading.Thread(target=_work, daemon=True).start()
        return key
    except Exception as e:
        return f"error: {e}"


def _api_civitai_progress(key):
    """API (Asset Browser): etat courant d'un job CivitAI (JSON). done=true quand fini.
    Sert aussi bien les jobs par-modele que les jobs batch (memes cles de registre)."""
    with _CIVITAI_LOCK:
        st = _CIVITAI_JOBS.get(key)
        st = dict(st) if st else {"phase": "unknown", "frac": None, "text": "",
                                  "done": True, "ok": False, "message": "no such job"}
    return json.dumps(st)


def _api_civitai_fetch_all(kind):
    """API (Asset Browser, bouton 'Fetch all missing'): enrichit EN ARRIERE-PLAN tous les
    modeles manquants du dossier LoRAs ou checkpoints (meme coeur que le script .bat/.sh
    cz_civitai_batch). Renvoie une cle de job batch a interroger via civitai_progress."""
    try:
        import cz_civitai_batch
        import cz_civitai
        kind = (kind or "").strip() or "loras"
        key = "__batch__:" + kind
        _civitai_job_set(key, phase="start", i=0, n=0, text="Starting…",
                         done=False, ok=False, summary=None)

        def _progress(i, n, name, phase, text):
            _civitai_job_set(key, phase=phase, i=i, n=n, text=text, done=False)

        def _work():
            try:
                api_key = getattr(cz_civitai, "API_KEY", None)
                summary = cz_civitai_batch.run(
                    kind=kind, api_key=api_key, progress=_progress,
                    loras_dir=cz_pipeline.LORAS_DIR,           # dossiers LIVE (modifiables dans l'UI)
                    checkpoints_dir=cz_pipeline.CHECKPOINTS_DIR)
                try:
                    ab_build_catalog(DEFAULT_OUTPUT_DIR, cz_pipeline.LORAS_DIR,
                                     cz_pipeline.CHECKPOINTS_DIR)
                except Exception as e:
                    _dbg(f"catalog rebuild after batch failed: {e}")
                _civitai_job_set(
                    key, phase="done", done=True, ok=True, summary=summary,
                    text=(f"enriched {summary['enriched']}, updated {summary['updated']}, "
                          f"skipped {summary['skipped']}, failed {summary['failed']}"))
            except Exception as e:
                _civitai_job_set(key, phase="error", done=True, ok=False,
                                 text=f"error: {e}", summary=None)

        threading.Thread(target=_work, daemon=True).start()
        return key
    except Exception as e:
        return f"error: {e}"


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
                 auto_upscale=False,
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
        base_prompt = _apply_wildcards(prompt, _seed_rng(seed), index=0)  # __name__ -> line
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
        # Resout un seed -1 (random) en une valeur CONCRETE -> reproductible, memorisee
        # (bouton "Reuse last seed") et ecrite correctement dans les metadonnees.
        base_seed = int(seed) if int(seed) >= 0 else random.randint(0, 2**31 - 1)
        cz_pipeline._LAST_SEED = base_seed
        images, img_paths, total_t = [], [], 0.0
        for i in range(n):
            if cz_pipeline._STOP:
                _log(f"stop requested after {i}/{n} image(s)")
                break
            s = base_seed if cz_pipeline._NO_SEED_INCREMENT else base_seed + i
            progress(i / n, desc=f"Image {i + 1}/{n}")
            # Wildcards (__name__) + style aleatoire, par image (seed -> reproductible)
            chosen = _pick_styles(styles, style_random)
            p_i = _apply_wildcards(prompt, _seed_rng(s), index=i)
            fp, fn = _apply_styles(p_i, negative, chosen)
            if style_random:
                _log(f"random style #{i + 1}: {chosen}")
            img, t = txt2img_run(fp, width, height, gen_steps, s, fn,
                                 upscale=False, steps=refine_steps)
            total_t += t["txt2img"]
            tag, gmode = "txt2img", "txt2img"
            base_img = img   # image txt2img avant un eventuel upscale
            # Chainage optionnel: upscale (ESRGAN + refine) sur l'image generee, sans
            # action manuelle. Reutilise le meme pipeline que l'onglet Upscale/img2img.
            if auto_upscale:
                progress((i + 0.5) / n, desc=f"Upscaling {i + 1}/{n}")
                eff_denoise = float(denoise) if do_refine else 0.0
                img, ut = process_one(
                    base_img, esrgan_model, factor, eff_denoise, refine_steps, fp, s,
                    tile, overlap, refine_tile=refine_tile, refine_overlap=refine_overlap,
                    do_esrgan=bool(do_esrgan), refine_first=bool(refine_first))
                total_t += ut.get("esrgan", 0.0) + ut.get("refine", 0.0)
                tag, gmode = "upscaled", "txt2img+upscale"
                # Option: sauver AUSSI l'image txt2img d'origine (avant l'upscale).
                if cz_pipeline._SAVE_PRE_UPSCALE and save_mode != "display":
                    try:
                        pre_dst = build_output_path(None, save_mode, output_dir, output_format,
                                                    tag="txt2img", seed=s, size=base_img.size,
                                                    index=(i + 1 if n > 1 else 0))
                        if pre_dst:
                            save_image(base_img, pre_dst, output_format, meta=_gen_meta(
                                "txt2img", fp, fn, s, gen_steps, cz_pipeline.GUIDANCE,
                                base_img.size, styles=chosen))
                            _dbg(f"saved pre-upscale: {pre_dst}")
                    except Exception as e:
                        _dbg(f"pre-upscale save failed: {e}")
            images.append(img)
            dst = None
            if save_mode != "display":
                try:
                    dst = build_output_path(None, save_mode, output_dir, output_format,
                                            tag=tag, seed=s, size=img.size,
                                            index=(i + 1 if n > 1 else 0))
                    if dst:
                        save_image(img, dst, output_format, meta=_gen_meta(
                            gmode, fp, fn, s, gen_steps, cz_pipeline.GUIDANCE,
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
        _label = "txt2img+upscale" if auto_upscale else "txt2img"
        rep = (f"{_label} x{len(images)} - **{images[0].size[0]}x{images[0].size[1]}** "
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


# ---- Job queue: snapshots complets de reglages empiles, executes en serie ----
# Config: bloc "job_queue" de config.txt. enabled=false -> AUCUN composant cree,
# aucun handler cable (contrat zero-cout quand off).
_JQ_CFG = CONFIG.get("job_queue") if isinstance(CONFIG.get("job_queue"), dict) else {}
JOB_QUEUE_ENABLED = bool(_JQ_CFG.get("enabled", True))

# Indices dans _gen_inputs (a garder synchro avec la liste dans build_ui).
_Q_HISTORY_IDX = 34   # l'historique de session est injecte LIVE au run, pas du snapshot
_Q_IDX = {"prompt": 0, "use_input": 4, "width": 13, "height": 14,
          "gen_steps": 15, "image_number": 16, "seed": 17}


def _q_model_state():
    """Snapshot de l'etat modele GLOBAL (hors _gen_inputs): checkpoint/transformer,
    LoRA actives, sampler/schedule. Rend chaque job autonome et reproductible."""
    return {"base_repo": cz_pipeline.BASE_REPO,
            "transformer": cz_pipeline.ZIMAGE_TRANSFORMER,
            "loras": list(cz_pipeline.LORAS),
            "sampler": cz_pipeline.SAMPLER,
            "schedule": cz_pipeline.SCHEDULE}


def _q_restore_model_state(ms):
    """Restaure l'etat modele d'un job via les setters existants -> free_vram() se
    declenche automatiquement si (et seulement si) le modele change entre 2 jobs."""
    if ms.get("base_repo"):
        set_zimage_model(ms["base_repo"])
    set_zimage_transformer(ms.get("transformer") or "")
    set_loras([(p, w) for p, w in (ms.get("loras") or [])])
    set_sampler(ms.get("sampler") or "euler")
    set_schedule(ms.get("schedule") or "sgm_uniform")


def _q_label(vals, ms):
    """Etiquette lisible d'un job (parametres cles) depuis le snapshot."""
    mode = "img2img" if vals[_Q_IDX["use_input"]] else "txt2img"
    model = os.path.basename(str(ms.get("transformer") or ms.get("base_repo") or "?"))
    n = max(1, int(vals[_Q_IDX["image_number"]] or 1))
    seed = int(vals[_Q_IDX["seed"]] if vals[_Q_IDX["seed"]] is not None else -1)
    p = str(vals[_Q_IDX["prompt"]] or "").strip().replace("\n", " ")
    lbl = (f"{mode} · {model} · {int(vals[_Q_IDX['width']])}x{int(vals[_Q_IDX['height']])} · "
           f"{int(vals[_Q_IDX['gen_steps']])} steps · seed {seed} · x{n}")
    if p:
        lbl += f" · “{p[:40]}{'…' if len(p) > 40 else ''}”"
    return lbl


def _q_move(items, sel, delta):
    """Deplace l'element sel de delta (liste pure). Renvoie (items, nouvelle selection)."""
    items = list(items or [])
    if sel is None or not (0 <= int(sel) < len(items)):
        return items, None
    i, j = int(sel), int(sel) + int(delta)
    if not (0 <= j < len(items)):
        return items, i
    items[i], items[j] = items[j], items[i]
    return items, j


def _q_remove(items, sel):
    """Supprime l'element sel (liste pure). Renvoie (items, selection ajustee)."""
    items = list(items or [])
    if sel is None or not (0 <= int(sel) < len(items)):
        return items, None
    items.pop(int(sel))
    return items, (min(int(sel), len(items) - 1) if items else None)


def _q_render(items, sel=None):
    """Updates UI d'apres la file: (dropdown selection, markdown liste, label bouton)."""
    choices = [(f"#{i + 1} {it['label']}", i) for i, it in enumerate(items)]
    val = int(sel) if (sel is not None and 0 <= int(sel) < len(items)) else None
    md = "\n".join(f"{i + 1}. {it['label']}" for i, it in enumerate(items)) or "*Queue empty.*"
    return (gr.update(choices=choices, value=val), md,
            gr.update(value=f"+ Queue ({len(items)})"))


def _ui_queue_add(*args):
    """'+ Queue': fige les 36 valeurs courantes + l'etat modele global, empile."""
    *vals, items = args
    job = {"vals": list(vals), "ms": _q_model_state()}
    job["label"] = _q_label(job["vals"], job["ms"])
    items = list(items or []) + [job]
    _log(f"job added ({len(items)} queued): {job['label']}", mod="queue")
    return (items, *_q_render(items, len(items) - 1))


def _ui_queue_move(items, sel, delta):
    items, sel = _q_move(items, sel, delta)
    return (items, *_q_render(items, sel))


def _ui_queue_remove(items, sel):
    items, sel = _q_remove(items, sel)
    return (items, *_q_render(items, sel))


def _ui_queue_clear(items):
    _log("queue cleared", mod="queue")
    return ([], *_q_render([]))


def _ui_queue_run(items, history, progress=gr.Progress(track_tqdm=True)):
    """Execute la file en serie. Chaque job restaure son etat modele (purge VRAM auto au
    changement) puis rejoue _ui_generate. Stop = interrompt le job courant et met la
    file en PAUSE: les jobs restants demeurent empiles."""
    items = list(items or [])
    if not items:
        return ([], *_q_render([]), gr.update(), "*Queue empty.*", history, history)
    total, done, gallery_all, rep = len(items), 0, [], ""
    touched_gids = set()
    while items:
        job = items[0]
        _log(f"running job {done + 1}/{total}: {job['label']}", mod="queue")
        try:
            _q_restore_model_state(job["ms"])
            vals = list(job["vals"])
            vals[_Q_HISTORY_IDX] = history
            g, rep, history, _hg = _ui_generate(*vals, progress=progress)
            gallery_all.extend(list(g or []))
            # Cellule d'une grille X/Y/Z: memorise la 1re image (reduite) pour la planche.
            xj = job.get("xyz")
            if xj and g:
                try:
                    im = _as_pil(g[0])
                    if im is not None:
                        im = im.copy()
                        im.thumbnail((XYZ_THUMB, XYZ_THUMB), Image.LANCZOS)
                        meta = _XYZ_PENDING.get(xj["gid"])
                        if meta is not None:
                            meta.setdefault("cells", {})[(xj["ix"], xj["iy"], xj["iz"])] = im
                            touched_gids.add(xj["gid"])
                except Exception as e:
                    _log(f"cell capture failed ({e})", mod="xyz")
        except Exception as e:
            _log(f"job failed ({e}); continuing with next", mod="queue")
            rep = f"Job failed: {e}"
        done += 1
        items.pop(0)
        if cz_pipeline._STOP:
            _log(f"stopped; queue PAUSED, {len(items)} job(s) remaining", mod="queue")
            break
    # Assemblage des planches X/Y/Z touchees pendant ce run (cellules cumulees a travers
    # pause/reprise). gid libere quand plus aucun job de cette grille n'est en file.
    for gid in sorted(touched_gids):
        meta = _XYZ_PENDING.get(gid)
        if not meta:
            continue
        try:
            for p in _xyz_assemble(meta, meta.get("cells", {}), thumb=XYZ_THUMB):
                gallery_all.append(_dl_path(Image.open(p), p))
        except Exception as e:
            _log(f"assembly failed ({e})", mod="xyz")
        if not any((j.get("xyz") or {}).get("gid") == gid for j in items):
            _XYZ_PENDING.pop(gid, None)
    if items and cz_pipeline._STOP:
        status = f"Queue paused after {done}/{total} job(s) — {len(items)} remaining (Run queue to resume)."
    else:
        status = f"Queue done: {done} job(s)."
    return (items, *_q_render(items), gallery_all, f"{status}  \n{rep}", history, history)


# ---- X/Y/Z grid: combos de parametres -> jobs de la Job queue + planche annotee ----
# Config: bloc "xyz_grid" de config.txt. Necessite job_queue (reutilise snapshots,
# runner et pause Stop). enabled=false -> aucun composant, zero cout.
_XYZ_CFG = CONFIG.get("xyz_grid") if isinstance(CONFIG.get("xyz_grid"), dict) else {}
XYZ_FEATURE_ENABLED = bool(_XYZ_CFG.get("enabled", True))          # feature (UI + CLI)
XYZ_ENABLED = XYZ_FEATURE_ENABLED and JOB_QUEUE_ENABLED            # panneau UI (via la file)
XYZ_MAX_JOBS = int(_XYZ_CFG.get("max_jobs", 100))
XYZ_THUMB = int(_XYZ_CFG.get("thumb", 512))

# Table des axes: kind=val -> _gen_inputs[idx] cote UI, param abstrait cote CLI
# (cz_cli --xyz); kind=ms -> etat modele; kinds speciaux geres dans _xyz_apply.
# choices=callable -> liste fermee evaluee au build.
_XYZ_AXES = {
    "(none)":       None,
    "Checkpoint":   {"kind": "checkpoint"},
    "Sampler":      {"kind": "ms", "key": "sampler", "choices": lambda: list(SAMPLER_CHOICES)},
    "Schedule":     {"kind": "ms", "key": "schedule", "choices": lambda: list(SCHEDULE_CHOICES)},
    "Steps":        {"kind": "val", "idx": 15, "cast": int, "param": "gen_steps"},
    "Guidance":     {"kind": "val", "idx": 18, "cast": float, "param": "guidance"},
    "Seed":         {"kind": "val", "idx": 17, "cast": int, "param": "seed"},
    "ESRGAN model": {"kind": "val", "idx": 20, "cast": str, "param": "esrgan",
                     "choices": lambda: list_esrgan_models()},
    "Factor":       {"kind": "val", "idx": 24, "cast": float, "param": "factor"},
    "Denoise":      {"kind": "val", "idx": 25, "cast": float, "param": "denoise"},
    "Tile":         {"kind": "val", "idx": 27, "cast": int, "param": "tile"},
    "Refine tile":  {"kind": "val", "idx": 29, "cast": int, "param": "refine_tile"},
    "LoRA weight":  {"kind": "lora_weight"},
    "Performance":  {"kind": "performance"},
    "Prompt S/R":   {"kind": "sr"},
}


# Autosuggest des champs de valeurs (sous-cle "suggest" du bloc xyz_grid).
XYZ_SUGGEST = bool(_XYZ_CFG.get("suggest", True))

# Valeurs de calibrage classiques proposees pour les axes numeriques.
_XYZ_CALIB = {
    "Steps": "4, 8, 12, 20, 28",
    "Guidance": "0, 2, 3.5, 5",
    "Seed": "-1, 42, 1234",
    "Factor": "1.5, 2, 3, 4",
    "Denoise": "0.2, 0.3, 0.4",
    "Tile": "384, 512, 768",
    "Refine tile": "0, 768, 1024",
    "LoRA weight": "0.4, 0.7, 1.0",
}


def _xyz_csv_join(values):
    """Joint des valeurs en CSV re-parsable par _xyz_parse_values: guillemete celles
    qui contiennent virgule ou guillemet (double les guillemets internes)."""
    out = []
    for v in map(str, values):
        if "," in v or '"' in v:
            out.append('"' + v.replace('"', '""') + '"')
        else:
            out.append(v)
    return ", ".join(out)


def _xyz_suggestions(axis):
    """Suggestions contextuelles d'un axe: (texte inserable, placeholder). Listes
    fermees de l'app pour les choix finis, calibrage pour le numerique, aide pour S/R."""
    spec = _XYZ_AXES.get(axis)
    if not spec:
        return "", "pick an axis first"
    kind = spec.get("kind")
    if kind == "sr":
        return "", 'search term, replacement1, "replacement, with comma", ...'
    if kind == "checkpoint":
        names = ZIMAGE_BASE_REPOS + list_checkpoints()
        return _xyz_csv_join(names), f"e.g. {_xyz_csv_join(names[:2])}"
    if kind == "performance":
        names = list(PERFORMANCE)
        return _xyz_csv_join(names), f"e.g. {_xyz_csv_join(names[:2])}"
    choices = spec.get("choices")
    if choices:
        try:
            names = list(choices())
        except Exception:
            names = []
        return _xyz_csv_join(names), (f"e.g. {_xyz_csv_join(names[:3])}" if names else "no items found")
    calib = _XYZ_CALIB.get(axis, "")
    return calib, (f"e.g. {calib}" if calib else "comma-separated values")


def _ui_xyz_axis_changed(axis):
    """Change d'axe -> placeholder contextualise du champ valeurs."""
    _fill, ph = _xyz_suggestions(axis)
    return gr.update(placeholder=ph)


def _ui_xyz_fill(axis, current):
    """Bouton suggest: insere la liste complete (choix fermes / calibrage) si le champ
    est vide, sinon ne touche pas a la saisie de l'utilisateur."""
    fill, _ph = _xyz_suggestions(axis)
    if not fill or (current or "").strip():
        return gr.update()
    return gr.update(value=fill)


def _xyz_parse_values(s):
    """Parse un champ de valeurs CSV; les guillemets protegent les virgules
    (csv stdlib). Renvoie la liste des valeurs non vides."""
    if not (s or "").strip():
        return []
    row = next(csv.reader([s], skipinitialspace=True))
    return [v.strip() for v in row if v.strip()]


def _xyz_match(value, choices):
    """Resout `value` dans une liste fermee: exact insensible a la casse, sinon
    sous-chaine unique. Renvoie (choix, None) ou (None, message d'erreur)."""
    v = str(value).lower().strip()
    exact = [c for c in choices if str(c).lower() == v]
    if exact:
        return exact[0], None
    part = [c for c in choices if v in str(c).lower()]
    if len(part) == 1:
        return part[0], None
    if not part:
        return None, f"'{value}' not found (choices: {', '.join(map(str, choices))[:120]})"
    return None, f"'{value}' is ambiguous ({', '.join(map(str, part))[:120]})"


def _xyz_validate_axis(name, raw_values, base_vals, base_ms):
    """Valide/normalise les valeurs d'un axe AVANT le build. Renvoie (values, None)
    ou (None, message d'erreur)."""
    spec = _XYZ_AXES.get(name)
    if not spec:
        return None, f"unknown axis '{name}'"
    if not raw_values:
        return None, f"{name}: no values"
    kind = spec.get("kind")
    if kind == "sr":
        term = raw_values[0]
        if len(raw_values) < 2:
            return None, "Prompt S/R needs the search term + at least one replacement"
        if term not in str(base_vals[_Q_IDX["prompt"]] or ""):
            return None, f"Prompt S/R: '{term}' not found in the prompt"
        return list(raw_values), None
    if kind == "lora_weight":
        if not base_ms.get("loras"):
            return None, "LoRA weight: no active LoRA (pick one in Models first)"
        try:
            return [float(v) for v in raw_values], None
        except ValueError as e:
            return None, f"LoRA weight: {e}"
    if kind == "performance":
        out = []
        for v in raw_values:
            m, err = _xyz_match(v, list(PERFORMANCE))
            if err:
                return None, f"Performance: {err}"
            out.append(m)
        return out, None
    if kind == "checkpoint":
        choices = ZIMAGE_BASE_REPOS + list_checkpoints()
        out = []
        for v in raw_values:
            m, err = _xyz_match(v, choices)
            if err:
                return None, f"Checkpoint: {err}"
            out.append(m)
        return out, None
    choices = spec.get("choices")
    if choices:
        out = []
        for v in raw_values:
            m, err = _xyz_match(v, choices())
            if err:
                return None, f"{name}: {err}"
            out.append(m)
        return out, None
    cast = spec.get("cast", str)
    try:
        return [cast(v) for v in raw_values], None
    except ValueError:
        return None, f"{name}: non-numeric value in {raw_values}"


def _xyz_apply(name, value, vals, ms):
    """Applique la valeur d'un axe a un snapshot (vals, ms) — mutation en place."""
    spec = _XYZ_AXES[name]
    kind = spec.get("kind")
    if kind == "val":
        vals[spec["idx"]] = value
    elif kind == "ms":
        ms[spec["key"]] = value
    elif kind == "checkpoint":
        if value in ZIMAGE_BASE_REPOS:
            ms["base_repo"], ms["transformer"] = value, None
        else:
            ms["transformer"] = resolve_checkpoint(value)
    elif kind == "lora_weight":
        ms["loras"] = [(p, float(value)) for p, _w in (ms.get("loras") or [])]
    elif kind == "performance":
        st, g = PERFORMANCE[value]
        vals[_Q_IDX["gen_steps"]], vals[18] = int(st), float(g)
    elif kind == "sr":
        term = spec["_term"]
        if str(value) != term:
            vals[_Q_IDX["prompt"]] = str(vals[_Q_IDX["prompt"]] or "").replace(term, str(value))


def _xyz_build_jobs(axes, base_vals, base_ms):
    """Construit les jobs du produit croise. axes = liste ordonnee [(nom, values)] pour
    X (requis), Y, Z (optionnels). Renvoie (jobs, meta) — meta decrit la planche."""
    gid = time.strftime("%Y%m%d_%H%M%S")
    (xn, xv) = axes[0]
    (yn, yv) = axes[1] if len(axes) > 1 else (None, [None])
    (zn, zv) = axes[2] if len(axes) > 2 else (None, [None])
    # Prompt S/R: memorise le terme cherche (1re valeur) pour _xyz_apply.
    for n, v in axes:
        if _XYZ_AXES[n].get("kind") == "sr":
            _XYZ_AXES[n]["_term"] = str(v[0])
    jobs = []
    for iz, z in enumerate(zv):
        for iy, y in enumerate(yv):
            for ix, x in enumerate(xv):
                vals, ms = list(base_vals), dict(base_ms, loras=list(base_ms.get("loras") or []))
                _xyz_apply(xn, x, vals, ms)
                if yn is not None:
                    _xyz_apply(yn, y, vals, ms)
                if zn is not None:
                    _xyz_apply(zn, z, vals, ms)
                parts = [f"{xn}={x}"] + ([f"{yn}={y}"] if yn else []) + ([f"{zn}={z}"] if zn else [])
                jobs.append({"vals": vals, "ms": ms, "label": "xyz · " + " · ".join(parts),
                             "xyz": {"gid": gid, "ix": ix, "iy": iy, "iz": iz}})
    meta = {"gid": gid, "x": (xn, [str(v) for v in xv]),
            "y": (yn, [str(v) for v in yv]) if yn else None,
            "z": (zn, [str(v) for v in zv]) if zn else None,
            "out_dir": str(base_vals[32] or DEFAULT_OUTPUT_DIR)}
    return jobs, meta


def _as_pil(item):
    """PIL depuis un item de galerie (_dl_path: chemin str ou PIL)."""
    if isinstance(item, str):
        return Image.open(item).convert("RGB")
    if isinstance(item, (tuple, list)) and item:
        return _as_pil(item[0])
    return item.convert("RGB") if item is not None else None


def _xyz_font(size):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def _xyz_assemble(meta, cells, thumb=512):
    """Assemble les planches annotees (une par valeur de Z): X en colonnes, Y en lignes,
    vignettes letterbox `thumb`px, marges pour les libelles. cells = {(ix,iy,iz): PIL}.
    Renvoie la liste des chemins sauves."""
    from PIL import ImageDraw
    xn, xv = meta["x"]
    yn, yv = meta["y"] if meta["y"] else (None, [""])
    zn, zv = meta["z"] if meta["z"] else (None, [""])
    left = 200 if yn else 20
    top_hdr, top_lbl = 44, 40
    bg, fg, line = (17, 24, 42), (223, 230, 242), (60, 72, 100)
    f_lbl, f_hdr = _xyz_font(22), _xyz_font(26)
    out_root = _ab_resolve_dir(meta["out_dir"])
    sheet_dir = os.path.join(out_root, f"xyz_{meta['gid']}")
    os.makedirs(sheet_dir, exist_ok=True)
    saved = []
    for iz, z in enumerate(zv):
        w = left + len(xv) * (thumb + 8) + 20
        h = top_hdr + top_lbl + len(yv) * (thumb + 8) + 20
        sheet = Image.new("RGB", (w, h), bg)
        d = ImageDraw.Draw(sheet)
        hdr = f"X: {xn}" + (f"  ·  Y: {yn}" if yn else "") + (f"  ·  Z: {zn} = {z}" if zn else "")
        d.text((left, 10), hdr, fill=fg, font=f_hdr)
        for ix, x in enumerate(xv):
            cx = left + ix * (thumb + 8)
            d.text((cx + 6, top_hdr + 8), str(x)[:40], fill=fg, font=f_lbl)
        for iy, y in enumerate(yv):
            cy = top_hdr + top_lbl + iy * (thumb + 8)
            if yn:
                d.text((10, cy + thumb // 2 - 12), str(y)[:22], fill=fg, font=f_lbl)
            for ix in range(len(xv)):
                cx = left + ix * (thumb + 8)
                img = cells.get((ix, iy, iz))
                if img is None:
                    d.rectangle([cx, cy, cx + thumb, cy + thumb], outline=line, width=2)
                    d.text((cx + thumb // 2 - 8, cy + thumb // 2 - 12), "—", fill=line, font=f_hdr)
                    continue
                t = img.copy()
                t.thumbnail((thumb, thumb), Image.LANCZOS)
                sheet.paste(t, (cx + (thumb - t.width) // 2, cy + (thumb - t.height) // 2))
        suffix = f"_{zn}-{z}" if zn else ""
        safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(suffix))[:60]
        dst = os.path.join(sheet_dir, f"sheet{safe or ''}.png")
        sheet.save(dst)
        saved.append(dst)
        _log(f"sheet saved: {dst}", mod="xyz")
    return saved


def _ui_xyz_build(*args):
    """'Build grid -> queue': valide les axes, produit les combos, empile les jobs."""
    *gen_vals, xa, xv, ya, yv, za, zv, items = args
    items = list(items or [])
    axes_in = [(a, v) for a, v in ((xa, xv), (ya, yv), (za, zv)) if a and a != "(none)"]
    if not axes_in:
        return (items, *_q_render(items), "Pick at least the X axis.")
    names = [a for a, _v in axes_in]
    if len(set(names)) != len(names):
        return (items, *_q_render(items), "Each axis must vary a different parameter.")
    base_ms = _q_model_state()
    axes = []
    for name, raw in axes_in:
        values, err = _xyz_validate_axis(name, _xyz_parse_values(raw), list(gen_vals), base_ms)
        if err:
            return (items, *_q_render(items), f"❌ {err}")
        axes.append((name, values))
    total = 1
    for _n, v in axes:
        total *= len(v)
    if total > XYZ_MAX_JOBS:
        return (items, *_q_render(items),
                f"❌ {total} combos > max_jobs ({XYZ_MAX_JOBS}). Reduce the value lists "
                f"(or raise xyz_grid.max_jobs in config.txt).")
    jobs, meta = _xyz_build_jobs(axes, list(gen_vals), base_ms)
    _XYZ_PENDING[meta["gid"]] = meta
    items += jobs
    _log(f"grid built: {total} job(s) queued (gid {meta['gid']})", mod="xyz")
    return (items, *_q_render(items, len(items) - 1),
            f"Built **{total}** job(s) ({' × '.join(str(len(v)) for _n, v in axes)}). "
            f"Press **Run queue** to execute; the annotated sheet(s) will be saved in "
            f"`xyz_{meta['gid']}/` and shown in the gallery.")


# Planches en attente d'assemblage: gid -> meta (rempli au build, consomme au run).
_XYZ_PENDING = {}


# ---- Tag autocomplete (prompts): CSV tags/ + assets locaux, dropdown sous le caret ----
# Config: bloc "tag_autocomplete". enabled=false -> pas d'import cz_tags, pas de
# telechargement, pas de JS injecte (contrat zero-cout quand off).
_TAC_CFG = CONFIG.get("tag_autocomplete") if isinstance(CONFIG.get("tag_autocomplete"), dict) else {}
TAGAC_ENABLED = bool(_TAC_CFG.get("enabled", True))
TAGAC_SOURCES = _TAC_CFG.get("sources", [
    "https://raw.githubusercontent.com/DominikDoom/a1111-sd-webui-tagcomplete/main/tags/danbooru.csv",
])
TAGAC_MAX = int(_TAC_CFG.get("max_results", 8))


def _tagac_head():
    """Prepare le <script> d'autocomplete (ou None): telecharge les sources une fois
    (atomique + progression), construit le payload client (URLs des CSV + wildcards
    locaux). Tout echec -> warning et feature simplement absente (le boot continue)."""
    if not TAGAC_ENABLED:
        return None
    try:
        from cz_tags import ensure_tag_sources, list_tag_files
        from cz_assets import TAG_AC_JS
        ensure_tag_sources(TAGAC_SOURCES)
        files = list_tag_files()
        urls = ["/gradio_api/file=" + os.path.abspath(p).replace("\\", "/") for p in files]
        local = [f"__{w}__" for w in list_wildcards()]
        if not urls and not local:
            _log("no tag source available; autocomplete inactive", mod="tagac")
            return None
        js = (TAG_AC_JS.replace("__SRC__", json.dumps(urls))
              .replace("__LOCAL__", json.dumps(local))
              .replace("__MAX__", str(TAGAC_MAX)))
        _log(f"{len(urls)} CSV source(s) + {len(local)} local asset(s)", mod="tagac")
        return "<script>" + js + "</script>"
    except Exception as e:
        _log(f"init failed ({e}); autocomplete disabled", mod="tagac")
        return None


# JS injecte au chargement: force le theme sombre, preview de style au survol,
# et lightbox plein ecran au clic sur le rendu. __MAP__ = {nom_style: url_vignette}.
def _parse_a1111_params(text):
    """Parse le format A1111/Civitai (chunk PNG 'parameters'):
        <prompt>\\nNegative prompt: <neg>\\nSteps: N, Sampler: ..., Seed: N, Size: WxH, Model: ...
    Renvoie un dict {prompt, negative, seed, steps, guidance, sampler, size, model}."""
    def _int(v):
        try:
            return int(str(v).strip())
        except Exception:
            return None

    def _float(v):
        try:
            return float(str(v).strip())
        except Exception:
            return None

    out = {}
    t = (text or "").replace("\r", "")
    neg_i = t.find("Negative prompt:")
    steps_i = t.find("\nSteps:")
    if neg_i >= 0:
        out["prompt"] = t[:neg_i].strip()
        rest = t[neg_i + len("Negative prompt:"):]
        s2 = rest.find("\nSteps:")
        out["negative"] = (rest[:s2] if s2 >= 0 else rest).strip()
        params = rest[s2:].strip() if s2 >= 0 else ""
    else:
        out["prompt"] = (t[:steps_i] if steps_i >= 0 else t).strip()
        out["negative"] = ""
        params = t[steps_i:].strip() if steps_i >= 0 else ""
    for part in params.split(","):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        k, v = k.strip().lower(), v.strip()
        if k == "seed":
            out["seed"] = _int(v)
        elif k == "steps":
            out["steps"] = _int(v)
        elif k in ("cfg scale", "guidance", "cfg"):
            out["guidance"] = _float(v)
        elif k == "sampler":
            out["sampler"] = v
        elif k == "size":
            out["size"] = v
        elif k == "model":
            out["model"] = v
    return out


def _ui_read_meta(path):
    """PNG Info: lit le prompt + les parametres embarques d'une image (crispz JSON,
    A1111/Civitai 'parameters', ComfyUI, ou EXIF). Renvoie (markdown, dict parse)."""
    empty = "*Upload an image to read its embedded prompt & parameters.*"
    if not path or not os.path.isfile(path):
        return empty, {}
    meta = dict(_read_image_meta(path) or {})   # sidecar + chunk 'crispz' + EXIF
    scheme = "crispz" if meta.get("prompt") else None
    if not meta.get("prompt"):
        try:
            with Image.open(path) as im:
                info = im.info or {}
        except Exception:
            info = {}
        if info.get("parameters"):              # A1111 / Civitai
            meta.update(_parse_a1111_params(info["parameters"]))
            scheme = "a1111"
        elif info.get("prompt"):                # ComfyUI (workflow json, brut)
            meta["prompt"] = str(info.get("prompt"))[:2000]
            scheme = "comfyui"
    if not meta.get("prompt") and not meta.get("negative"):
        return "*No embedded prompt/metadata found in this image.*", {}
    lines = [f"*Detected scheme: **{scheme or 'unknown'}***"]
    if meta.get("prompt"):
        lines.append(f"**Prompt**\n\n{meta['prompt']}")
    if meta.get("negative"):
        lines.append(f"**Negative**\n\n{meta['negative']}")
    kv = [f"{k}: {meta[k]}" for k in ("seed", "steps", "guidance", "sampler", "size", "model", "mode")
          if meta.get(k) not in (None, "")]
    if kv:
        lines.append("**Params** — " + "  ·  ".join(kv))
    return "\n\n".join(lines), meta


# ============================ Presets (facon Fooocus) =========================
# Un preset = un bundle de reglages (prompt, styles, taille, steps/CFG, sampler,
# checkpoint, transformer, LoRAs) sauve en JSON dans presets/. Charger / creer / mettre
# a jour / supprimer depuis l'onglet Settings.
_PRESETS_DIR = os.path.join(HERE, "presets")
_PRESET_KEYS = ["prompt", "negative", "styles", "width", "height", "steps", "guidance",
                "sampler", "schedule", "image_number", "checkpoint", "transformer"]


def _preset_sanitize(name):
    n = "".join(c for c in (name or "") if c.isalnum() or c in " -_").strip()
    return n or "preset"


def list_presets():
    try:
        return sorted(f[:-5] for f in os.listdir(_PRESETS_DIR) if f.lower().endswith(".json"))
    except Exception:
        return []


def _load_preset_file(name):
    try:
        with open(os.path.join(_PRESETS_DIR, _preset_sanitize(name) + ".json"), encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _ensure_model_presets(checkpoints):
    """Cree un preset 'basique' presets/<stem>.json pour chaque checkpoint LOCAL
    chargeable (issu de list_checkpoints) qui n'en a pas encore. steps/CFG deduits par
    profile_for_model (comme la selection d'un modele), le reste = defauts de config.
    Ne modifie JAMAIS un preset existant. Renvoie le nombre de presets crees.
    Appele au demarrage et a chaque refresh/filtrage des checkpoints."""
    created = 0
    try:
        existing = set(list_presets())
    except Exception:
        existing = set()
    for f in checkpoints or []:
        # Ignore les repos de base HF (Tongyi-MAI/...) : ce ne sont pas des fichiers locaux.
        if not isinstance(f, str) or f in ZIMAGE_BASE_REPOS or "/" in f or "\\" in f:
            continue
        name = _preset_sanitize(os.path.splitext(f)[0])
        if name in existing:
            continue
        steps, g = profile_for_model(f)
        data = {
            "prompt": "",
            "negative": CONFIG.get("default_negative_prompt", "") or "",
            "styles": list(CONFIG.get("default_styles", []) or []),
            "width": int(CONFIG.get("default_width", 1024)),
            "height": int(CONFIG.get("default_height", 1024)),
            "steps": steps, "guidance": g,
            "sampler": (CONFIG.get("default_sampler") or "euler").strip().lower(),
            "schedule": (CONFIG.get("default_schedule") or "sgm_uniform").strip().lower(),
            "image_number": int(CONFIG.get("default_image_number", 1)),
            "checkpoint": f, "transformer": "", "loras": [],
        }
        try:
            os.makedirs(_PRESETS_DIR, exist_ok=True)
            with open(os.path.join(_PRESETS_DIR, name + ".json"), "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
            existing.add(name)
            created += 1
        except Exception as e:
            _dbg(f"auto-preset failed for {f}: {e}")
    if created:
        _log(f"presets: auto-created {created} basic model preset(s)")
    return created


def _ui_preset_save(name, *vals):
    """Sauve l'etat courant sous 'name'. vals = scalaires (_PRESET_KEYS) + lora_dds + lora_lws."""
    name = _preset_sanitize(name)
    nk = len(_PRESET_KEYS)
    scalars, lora_vals = vals[:nk], vals[nk:]
    half = len(lora_vals) // 2
    dds, lws = lora_vals[:half], lora_vals[half:]
    data = {k: v for k, v in zip(_PRESET_KEYS, scalars)}
    data["loras"] = [[dds[i], float(lws[i])] for i in range(half)
                     if dds[i] and dds[i] not in ("None", "none", "")]
    try:
        os.makedirs(_PRESETS_DIR, exist_ok=True)
        with open(os.path.join(_PRESETS_DIR, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        return gr.update(), f"Save failed: {e}"
    return gr.update(choices=list_presets(), value=name), f"Preset '{name}' saved."


def _ui_preset_load(name):
    """Renvoie les gr.update pour tous les composants (scalaires + 10 LoRA dd + 10 poids)."""
    data = _load_preset_file(name)
    scal = [gr.update(value=data[k]) if k in data else gr.update() for k in _PRESET_KEYS]
    loras = data.get("loras", []) or []
    dd_up = [gr.update(value=(loras[i][0] if i < len(loras) else "None")) for i in range(MAX_LORA_SLOTS)]
    w_up = [gr.update(value=(float(loras[i][1]) if i < len(loras) else float(cz_pipeline.LORA_WEIGHT)))
            for i in range(MAX_LORA_SLOTS)]
    return scal + dd_up + w_up


def _ui_preset_delete(name):
    try:
        p = os.path.join(_PRESETS_DIR, _preset_sanitize(name) + ".json")
        if os.path.isfile(p):
            os.remove(p)
    except Exception as e:
        return gr.update(), f"Delete failed: {e}"
    return gr.update(choices=list_presets(), value=None), f"Deleted '{_preset_sanitize(name)}'."


def _ui_apply_ckpt_silent(name):
    """Applique le checkpoint SANS toucher steps/guidance (le preset les a deja poses)."""
    try:
        return _apply_checkpoint(name)[0]
    except Exception as e:
        return f"Checkpoint apply failed: {e}"


def _ui_apply_transformer_silent(repo):
    if not (repo or "").strip():
        return gr.update()
    try:
        return _apply_transformer_repo(repo)[0]
    except Exception as e:
        return f"Transformer apply failed: {e}"


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

    with gr.Blocks(title=f"crispz-studio {APP_VERSION}", theme=gr.themes.Default(), css=FOOOCUS_CSS,
                   js=js_full, head=_tagac_head()) as demo:
        # La galerie du dossier de sortie s'ouvre dans un nouvel onglet (Asset Browser),
        # via le bouton sous l'apercu. Pas de panneau galerie inline.
        gallery_url = gr.Textbox(visible=False)
        # Endpoint API (appele par l'Asset Browser pour supprimer une image)
        del_in = gr.Textbox(visible=False)
        del_out = gr.Textbox(visible=False)
        del_btn = gr.Button(visible=False)
        del_btn.click(delete_asset, del_in, del_out, api_name="delete_asset")
        # Endpoint API CivitAI (Asset Browser -> preview/trigger words/exemples d'un modele)
        cf_rel = gr.Textbox(visible=False)
        cf_kind = gr.Textbox(visible=False)
        cf_out = gr.Textbox(visible=False)
        cf_btn = gr.Button(visible=False)
        cf_btn.click(_api_civitai_fetch, [cf_rel, cf_kind], cf_out, api_name="civitai_fetch")
        # Endpoint de progression (polling par l'Asset Browser pendant le fetch CivitAI)
        cp_in = gr.Textbox(visible=False)
        cp_out = gr.Textbox(visible=False)
        cp_btn = gr.Button(visible=False)
        cp_btn.click(_api_civitai_progress, cp_in, cp_out, api_name="civitai_progress")
        # Endpoint batch (bouton 'Fetch all missing' de l'Asset Browser)
        cfa_in = gr.Textbox(visible=False)
        cfa_out = gr.Textbox(visible=False)
        cfa_btn = gr.Button(visible=False)
        cfa_btn.click(_api_civitai_fetch_all, cfa_in, cfa_out, api_name="civitai_fetch_all")

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
                        gallery_btn = gr.Button("\U0001F5BC️ Asset Browser", size="sm",
                                                variant="primary")
                gallery_status = gr.Markdown("")

                with gr.Row(equal_height=True):
                    prompt = gr.Textbox(show_label=False, value=CONFIG.get("default_prompt", ""),
                                        placeholder="Type your prompt here...",
                                        elem_id="cz_prompt", lines=2, scale=4, container=False)
                    btn = gr.Button("Generate", elem_id="cz_generate", scale=1, min_width=150)

                with gr.Row(equal_height=True):
                    negative = gr.Textbox(show_label=False, value=CONFIG.get("default_negative_prompt", ""),
                                          elem_id="cz_neg", lines=1, container=False, scale=4,
                                          placeholder="Negative prompt - what you do NOT want (needs guidance > 0)")
                    stop_btn = gr.Button("Stop", variant="stop", scale=1, min_width=150)

                with gr.Row():
                    # Chainage txt2img -> upscale (gauche) + Improve prompt (droite), alignes.
                    auto_upscale_cb = gr.Checkbox(
                        value=bool(CONFIG.get("default_auto_upscale", False)), scale=4,
                        label="Upscale after generate — chain each txt2img image through the "
                              "Upscale pipeline (ESRGAN + refine), no manual step")
                    improve_btn = gr.Button("Improve prompt", scale=1, min_width=150)
                improve_status = gr.Markdown("")

                if JOB_QUEUE_ENABLED:
                    queue_state = gr.State([])
                    with gr.Accordion("Job queue", open=False):
                        gr.Markdown("*'+ Queue' snapshots ALL current settings (incl. model, "
                                    "LoRAs, sampler). 'Run queue' executes jobs in order; "
                                    "**Stop** pauses the queue — remaining jobs are kept.*")
                        with gr.Row():
                            queue_add_btn = gr.Button("+ Queue (0)", size="sm", scale=1, min_width=140)
                            queue_run_btn = gr.Button("Run queue", variant="primary", size="sm",
                                                      scale=1, min_width=140)
                        queue_md = gr.Markdown("*Queue empty.*")
                        with gr.Row():
                            queue_sel = gr.Dropdown([], label="Selected job", scale=4)
                            queue_up_btn = gr.Button("Up", size="sm", scale=0, min_width=60)
                            queue_down_btn = gr.Button("Down", size="sm", scale=0, min_width=70)
                            queue_rm_btn = gr.Button("Remove", size="sm", scale=0, min_width=90)
                            queue_clear_btn = gr.Button("Clear", size="sm", scale=0, min_width=70)

                if XYZ_ENABLED:
                    with gr.Accordion("X/Y/Z grid", open=False):
                        gr.Markdown("*Vary 1–3 parameters (comma-separated values; use quotes to "
                                    "protect commas). Each combo becomes a queued job; when it has "
                                    "run, an annotated contact sheet is saved (one per Z value) and "
                                    "shown in the gallery. Prompt S/R: first value = search term, "
                                    "then its replacements.*")
                        _ax = [k for k in _XYZ_AXES]
                        _xyz_rows = []
                        for _axis_lbl, _default in (("X", "Steps"), ("Y", "(none)"), ("Z", "(none)")):
                            with gr.Row():
                                _dd = gr.Dropdown(_ax, value=_default, label=f"{_axis_lbl} axis",
                                                  scale=1)
                                _tb = gr.Textbox(label=f"{_axis_lbl} values", scale=3,
                                                 placeholder=_xyz_suggestions(_default)[1])
                                _sg = (gr.Button("⤵ suggest", size="sm", scale=0, min_width=90)
                                       if XYZ_SUGGEST else None)
                                _xyz_rows.append((_dd, _tb, _sg))
                        (xyz_xa, xyz_xv, xyz_xs), (xyz_ya, xyz_yv, xyz_ys), (xyz_za, xyz_zv, xyz_zs) = _xyz_rows
                        xyz_build_btn = gr.Button("Build grid → queue", variant="primary", size="sm")
                        xyz_status = gr.Markdown("")

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
                            with gr.Accordion("\U0001F4C4 Read prompt / metadata from an image (PNG Info)",
                                              open=False):
                                gr.Markdown("*Drop a PNG/JPG to read its embedded prompt & parameters "
                                            "(crispz, A1111/Civitai, or EXIF), then send them to the fields.*")
                                meta_reader = gr.Image(type="filepath", sources=["upload", "clipboard"],
                                                       height=150,
                                                       label="Image to read")
                                input_meta_md = gr.Markdown(
                                    "*Upload an image above to read its prompt & parameters.*")
                                meta_state = gr.State({})
                                with gr.Row():
                                    meta_to_prompt_btn = gr.Button("→ Send prompt", size="sm",
                                                                   variant="primary")
                                    meta_to_seed_btn = gr.Button("→ Send seed", size="sm")
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
                                "(or the local captioner if Ollama is off).*")

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
                            compose_status = gr.Markdown(
                                "*Select an Ollama vision model in Advanced > Prompt AI (click Detect), "
                                "then add reference images and run Vision Mix.*",
                                container=True, min_height=48)

                        with gr.Tab("Remove BG"):
                            rembg_img = _crop_input("Image", 280)
                            rembg_btn = gr.Button("Remove background", variant="primary", size="sm")
                            rembg_status = gr.Markdown("*Local (rembg). Output = transparent PNG. "
                                                       "First use downloads the u2net model.*")

                        with gr.Tab("Inpaint / Outpaint"):
                            gr.Markdown("*One editor for everything: **Brush** = inpaint the painted "
                                        "area · **Expand sides** = outpaint Left/Right/Top/Bottom · "
                                        "**Reframe** = new aspect ratio. Steps follow the model "
                                        "(Performance). Describe the result in the prompt.*")
                            edit_mode = gr.Radio(
                                ["Brush (inpaint)", "Expand sides (outpaint)", "Reframe (ratio)"],
                                value="Brush (inpaint)", label="Mode")
                            inpaint_editor = gr.ImageEditor(
                                type="pil", label="Image + mask (paint for Brush mode; click the "
                                                  "brush icon to change its size)",
                                brush=gr.Brush(default_size=25,
                                               colors=["#ffffff", "#ff3b3b", "#3b82f6"],
                                               default_color="#ffffff"),
                                layers=False, transforms=[], height=420)
                            with gr.Group(visible=False) as expand_group:
                                inpaint_outpaint = gr.CheckboxGroup(
                                    choices=["Left", "Right", "Top", "Bottom"], value=[],
                                    label="Sides to expand (~30% each; blurred-edge fill + blend)")
                                center_btn = gr.Button("Center (all sides)", size="sm")
                            with gr.Group(visible=False) as reframe_group:
                                with gr.Row():
                                    reframe_ratio = gr.Dropdown(
                                        ["16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "1:1", "21:9"],
                                        value="16:9", label="Target ratio")
                                    reframe_fit = gr.Radio(
                                        ["Contain (outpaint)", "Cover (crop)"],
                                        value="Contain (outpaint)", label="Fit")
                            edit_autodescribe = gr.Checkbox(
                                value=False, label="Auto-describe center (local model, no Ollama)",
                                info="Outpaint/Reframe: runs automatically when the prompt is EMPTY. "
                                     "Check it to also prepend a description when you DO have a "
                                     "prompt. Local model (BLIP), set in Prompt AI > Caption model.")
                            edit_strength = gr.Slider(
                                0.3, 1.0, value=0.85, step=0.05, label="Strength",
                                info="Inpaint/outpaint denoise. Outpaint & reframe: ~0.8 keeps the "
                                     "edge colors (best blend), ~1.0 = more new detail.")
                            with gr.Row():
                                edit_harmonize = gr.Checkbox(
                                    value=False, label="Harmonize (final img2img pass)",
                                    info="Light img2img refine over the whole result -> unifies "
                                         "grain/light and removes the 'added zone' look.")
                                edit_harmonize_denoise = gr.Slider(
                                    0.05, 0.5, value=0.2, step=0.05, label="Harmonize denoise")
                            edit_btn = gr.Button("Generate", variant="primary",
                                                 elem_id="cz_edit_generate")
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
                        with gr.Accordion("⭐ Presets", open=False):
                            gr.Markdown("*A preset bundles prompt, styles, size, steps/CFG, "
                                        "sampler, checkpoint, transformer + LoRAs. Load applies "
                                        "them (incl. the model). Create/Update save the current state.*")
                            # Auto-cree un preset basique par modele local avant de peupler
                            # le menu (les modeles FP8/INT8-INT4 sont deja exclus par la liste).
                            _ensure_model_presets(list_checkpoints())
                            with gr.Row():
                                preset_dd = gr.Dropdown(list_presets(), label="Preset", scale=3)
                                preset_refresh_btn = gr.Button("↻", size="sm", scale=0, min_width=44)
                                preset_load_btn = gr.Button("Load", size="sm", variant="primary", scale=1)
                            with gr.Row():
                                preset_name_tb = gr.Textbox(show_label=False, scale=2, container=False,
                                                            placeholder="new preset name")
                                preset_save_btn = gr.Button("Save as new", size="sm", scale=1)
                                preset_update_btn = gr.Button("Update selected", size="sm", scale=1)
                                preset_delete_btn = gr.Button("Delete", size="sm", variant="stop",
                                                              scale=0, min_width=80)
                            preset_status = gr.Markdown("")
                        performance = gr.Radio(list(PERFORMANCE),
                                               value=CONFIG.get("default_performance", "Turbo (8 steps)"),
                                               label="Performance",
                                               info="Sets steps + guidance. Turbo = your Turbo model; "
                                                    "Base CFG = for a Z-Image Base checkpoint.")
                        aspect = gr.Dropdown(list(ASPECT_RATIOS),
                                             value=CONFIG.get("default_aspect_ratio", "1024 x 1024  (1:1)"),
                                             label="Aspect ratio")
                        force_ratio_cb = gr.Checkbox(
                            value=bool(cz_pipeline.FORCE_RATIO),
                            label="Force aspect ratio on Upscale/img2img (crop input to fit)",
                            info="When ON, the loaded image is centre-cropped to the Aspect ratio "
                                 "above before Upscale/img2img (Fooocus-style). OFF = keep the "
                                 "input's native ratio.")
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
                        with gr.Row():
                            reuse_seed_btn = gr.Button("♻️ Reuse last seed", size="sm",
                                                       scale=1, min_width=140)
                            no_seed_inc_cb = gr.Checkbox(
                                value=False, scale=1, label="Fix seed (no +1 per image)",
                                info="Batch reuses the same seed for every image (no increment).")

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
                        gr.Markdown("### Local captioner (no Ollama)")
                        caption_model_dd = gr.Dropdown(
                            ["blip-large", "blip-base"],
                            value=_current_caption_kind(), label="Caption model",
                            info="Used by Auto-describe (Inpaint/Outpaint) and the Describe "
                                 "fallback. blip-large = richer captions. Loads on next use.")
                        caption_model_status = gr.Markdown("")
                        gr.Markdown("### Ollama (optional)")
                        ollama_url = gr.Textbox(value=OLLAMA_URL, label="Ollama URL",
                                                info="Local LLM server. Used for Describe (vision) "
                                                     "and Improve prompt.")
                        detect_btn = gr.Button("Detect Ollama (vision models)", size="sm", variant="primary")
                        ollama_model = gr.Dropdown([], label="Vision model (Describe / Improve)",
                                                   interactive=True)
                        ollama_status = gr.Markdown("*Click Detect. If Ollama is off, Describe falls "
                                                    "back to a local captioner.*")
                        gr.Markdown("---")
                        log_level_dd = gr.Dropdown(["quiet", "info", "debug"],
                                                   value={0: "quiet", 1: "info", 2: "debug"}.get(cz_core.LOG_LEVEL, "info"),
                                                   label="Console log level (dev)",
                                                   info="debug = full params, pipe state, VRAM in the .bat console.")
                        log_level_status = gr.Markdown("")

                    with gr.Tab("Models"):
                        offload = gr.Dropdown(choices=list(cz_pipeline.OFFLOAD_CHOICES), value="none",
                                              label="CPU offload (VRAM)",
                                              info="How much of the model to move to CPU RAM to save VRAM. Details below.")
                        with gr.Accordion("ℹ️  What is CPU offload?", open=False):
                            gr.Markdown(
                                "**CPU offload** moves part of the model weights from VRAM (GPU) to RAM (CPU) "
                                "between steps so large models fit on low-VRAM cards. It is **not** quantization "
                                "— weights stay BF16, they just shuttle between RAM and GPU.\n\n"
                                "- **none** — everything stays in VRAM. **Fastest.** Use it when you have enough "
                                "VRAM (e.g. RTX 5090 / 24GB+ → keep `none`).\n"
                                "- **model** — offload per submodule. ~half the VRAM, small slowdown. Good balance "
                                "on 12–16GB cards.\n"
                                "- **sequential** — aggressive, module-by-module. Runs in ~9GB but much slower "
                                "(5–10×). For small cards (8–12GB).")

                        with gr.Accordion("\U0001F4E6 Checkpoints (switch model)", open=True):
                            ckpt_dir_tb = gr.Textbox(value=cz_pipeline.CHECKPOINTS_DIR, label="Checkpoints folder")
                            ckpt_extra_dir_tb = gr.Textbox(
                                value=cz_pipeline.CHECKPOINTS_EXTRA_DIR,
                                label="Extra checkpoints folder (optional)",
                                placeholder="e.g. D:\\models\\Z-Image",
                                info="Merged into the single 'Z-Image checkpoint' list above. Leave empty to disable.")
                            esrgan_dir_tb = gr.Textbox(value=cz_esrgan.ESRGAN_DIR,
                                                       label="ESRGAN_DIR (.pth/.safetensors folder)")
                            with gr.Row():
                                refresh_btn = gr.Button("Refresh ESRGAN", size="sm")
                                save_paths_btn = gr.Button("Save paths", size="sm")
                            paths_status = gr.Markdown("")
                            with gr.Row():
                                _ckpt_choices = ZIMAGE_BASE_REPOS + list_checkpoints()
                                _ckpt_value = cz_pipeline.BASE_REPO if cz_pipeline.BASE_REPO in _ckpt_choices else ZIMAGE_BASE_REPOS[0]
                                ckpt_dd = gr.Dropdown(choices=_ckpt_choices,
                                                      value=_ckpt_value, label="Z-Image checkpoint", scale=3)
                                ckpt_open_btn = gr.Button("\U0001F5BC️", size="sm", scale=0, min_width=44,
                                                          elem_id="cz_ckpt_open")
                                ckpt_refresh_btn = gr.Button("Refresh", size="sm", scale=1)
                            ckpt_status = gr.Markdown("")
                            with gr.Row():
                                transformer_tb = gr.Textbox(
                                    value="", scale=3,
                                    label="Transformer override (HF repo / diffusers folder)",
                                    placeholder="e.g. RunDiffusion/Juggernaut-Z-Image",
                                    info="For community models with an incomplete tokenizer (Juggernaut-Z): "
                                         "loads only the transformer, keeps base VAE/encoder. Set base = Turbo.")
                                transformer_apply_btn = gr.Button("Apply override", size="sm", scale=1,
                                                                  variant="secondary")

                        with gr.Accordion("\U0001F9E9 LoRA (combinable)", open=False):
                            lora_dir_tb = gr.Textbox(value=cz_pipeline.LORAS_DIR, label="LoRA folder")
                            gr.Markdown("*Number of slots is set in Advanced > Generation "
                                        "(LoRA slots), or config `lora_slots`.*")
                            _lchoices = ["None"] + list_loras()
                            lora_dds, lora_lws, lora_rows, lora_open_btns = [], [], [], []
                            for _i in range(MAX_LORA_SLOTS):
                                with gr.Row(visible=(_i < LORA_SLOTS)) as _row:
                                    _dd = gr.Dropdown(choices=_lchoices, value="None",
                                                      label=f"LoRA {_i + 1}", scale=3)
                                    _ob = gr.Button("\U0001F5BC️", size="sm", scale=0, min_width=44)
                                    _lw = gr.Slider(0.0, 2.0, value=float(cz_pipeline.LORA_WEIGHT),
                                                    step=0.05, label=f"Weight {_i + 1}", scale=2)
                                lora_dds.append(_dd)
                                lora_lws.append(_lw)
                                lora_rows.append(_row)
                                lora_open_btns.append(_ob)
                            lora_refresh_btn = gr.Button("Refresh LoRA list", size="sm")
                            lora_keywords_tb = gr.Textbox(label="Keywords / trigger words", lines=2,
                                                          placeholder="Auto-filled from the selected LoRA(s).")
                            with gr.Row():
                                lora_kw_btn = gr.Button("Get keywords", size="sm")
                                lora_kw_to_prompt_btn = gr.Button("Add to prompt", size="sm", variant="primary")
                            lora_status = gr.Markdown("")

                        with gr.Accordion("\U0001F3B2 Wildcards", open=False):
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

                        with gr.Accordion("\U0001F5BC️ Omni / Edit (multi-reference)", open=False):
                            gr.Markdown("*The Reference (Omni) tab stays hidden until a model is set "
                                        "here (then restart). Z-Image-Omni-Base / Z-Image-Edit are not "
                                        "released yet. For a reference image now, use img2img.*")
                            omni_model_tb = gr.Textbox(value=CONFIG.get("zimage_omni_model", ""),
                                                       label="Omni model (HF repo or local folder)",
                                                       info="Needs SigLIP. Set it then restart to enable "
                                                            "the Reference (Omni) tab.")
                            omni_check_btn = gr.Button("Check Omni availability (Hugging Face)", size="sm")
                            omni_status = gr.Markdown("")

                    with gr.Tab("Save"):
                        save_mode = gr.Radio(choices=["display", "local", "alongside", "custom"],
                                             value=DEFAULT_SAVE_MODE, label="Save mode")
                        output_dir = gr.Textbox(value=DEFAULT_OUTPUT_DIR, label="Output folder")
                        output_format = gr.Dropdown(choices=list(SUPPORTED_FORMATS),
                                                    value=DEFAULT_OUTPUT_FORMAT, label="Output format")

                        gr.Markdown("### \U0001F5BC️ Asset Browser")
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
                        with gr.Row():
                            ab_open_btn = gr.Button("\U0001F5BC️ Open Asset Browser",
                                                    variant="primary", size="sm")
                            ab_reindex_btn = gr.Button("Rebuild ALL thumbnails (force)", size="sm")
                        ab_open_link = gr.HTML("")
                        ab_status = gr.Markdown("")

                    with gr.Tab("Advanced"):
                        gr.Markdown("### Hugging Face access (gated models)")
                        with gr.Row():
                            hf_token_tb = gr.Textbox(
                                value="", type="password", scale=3, label="HF token",
                                placeholder="hf_... (for gated models, e.g. FLUX.1-Krea-dev)",
                                info="Saved to preferences.json (gitignored) and applied immediately. "
                                     "Leave empty to keep the current token.")
                            hf_token_save_btn = gr.Button("Save token", size="sm", scale=1, variant="primary")
                        hf_token_status = gr.Markdown(
                            "✅ A Hugging Face token is currently set."
                            if hf_token_is_set() else
                            "No HF token set (only needed for gated models).")

                        gr.Markdown("### CivitAI access (previews / trigger words)")
                        with gr.Row():
                            civitai_key_tb = gr.Textbox(
                                value="", type="password", scale=3, label="CivitAI API key",
                                placeholder="CivitAI token (optional)",
                                info="Saved to preferences.json (gitignored). Used by 'Fetch from "
                                     "CivitAI' in the Asset Browser. Public models work without a key; "
                                     "a key unlocks gated/NSFW previews and avoids rate limits.")
                            civitai_key_save_btn = gr.Button("Save key", size="sm", scale=1,
                                                             variant="primary")
                        civitai_key_status = gr.Markdown(
                            "✅ A CivitAI key is set." if cz_civitai.API_KEY
                            else "No CivitAI key set (public models still work).")

                        gr.Markdown("### Metadata")
                        meta_scheme_dd = gr.Dropdown(
                            choices=[("crispz (json)", "crispz"),
                                     ("a1111 (plain text — Civitai-compatible)", "a1111")],
                            value=(CONFIG.get("metadata_scheme") or "crispz").lower(),
                            label="Metadata scheme (PNG output)",
                            info="a1111 adds a 'parameters' chunk to PNGs so Civitai reads the "
                                 "prompt/seed/params. crispz metadata (chunk + sidecar) is kept in both.")
                        meta_scheme_status = gr.Markdown("")

                        gr.Markdown("### Generation")
                        wildcards_order_cb = gr.Checkbox(
                            value=bool(CONFIG.get("wildcards_in_order", False)),
                            label="Read wildcards in order",
                            info="Batch: each image takes the NEXT line of the wildcard file "
                                 "(deterministic) instead of a random one.")
                        wild_order_status = gr.Markdown("")
                        save_pre_upscale_cb = gr.Checkbox(
                            value=bool(CONFIG.get("save_pre_upscale", False)),
                            label="Also save pre-upscale image",
                            info="In txt2img + auto-upscale, also save the original txt2img image "
                                 "(before ESRGAN/refine), tagged 'txt2img'.")
                        lora_slots_num = gr.Slider(1, MAX_LORA_SLOTS, value=LORA_SLOTS, step=1,
                                                   label="LoRA slots (Models > LoRA)",
                                                   info="How many LoRA slots to show. Applied live and "
                                                        "persisted (preferences.json).")

        # Toggles facon Fooocus
        advanced_cb.change(lambda v: gr.update(visible=bool(v)), advanced_cb, advanced_col)
        use_input.change(lambda v: gr.update(visible=bool(v)), use_input, input_group)
        # PNG Info: lire les meta d'une image + les envoyer aux champs
        meta_reader.change(_ui_read_meta, [meta_reader], [input_meta_md, meta_state])
        meta_to_prompt_btn.click(
            lambda m: (gr.update(value=(m or {}).get("prompt", "")),
                       gr.update(value=(m or {}).get("negative", ""))),
            [meta_state], [prompt, negative])
        meta_to_seed_btn.click(
            lambda m: gr.update(value=int((m or {}).get("seed"))) if (m or {}).get("seed") is not None
            else gr.update(),
            [meta_state], [seed])
        aspect.change(_set_aspect, [aspect], [width, height]) \
              .then(_ui_set_force_ratio, [force_ratio_cb, aspect], None)
        force_ratio_cb.change(_ui_set_force_ratio, [force_ratio_cb, aspect], None)
        performance.change(_set_performance, [performance], [gen_steps, guidance])
        style_search.change(_filter_styles, [style_search, styles], [styles])

        # Actions
        hf_token_save_btn.click(_save_hf_token, [hf_token_tb], [hf_token_tb, hf_token_status])
        civitai_key_save_btn.click(_save_civitai_key, [civitai_key_tb],
                                   [civitai_key_tb, civitai_key_status])
        meta_scheme_dd.change(set_metadata_scheme, [meta_scheme_dd], [meta_scheme_status])
        wildcards_order_cb.change(set_wildcards_in_order, [wildcards_order_cb], [wild_order_status])
        save_pre_upscale_cb.change(cz_pipeline.set_save_pre_upscale, [save_pre_upscale_cb], None)
        lora_slots_num.change(_ui_set_lora_slots, [lora_slots_num], lora_rows)
        refresh_btn.click(_refresh_models, [esrgan_dir_tb], [esrgan, paths_status])
        save_paths_btn.click(_save_paths_to_prefs,
                             [esrgan_dir_tb, ckpt_dir_tb, ckpt_extra_dir_tb, lora_dir_tb, wild_dir_tb],
                             [paths_status])
        wild_refresh_btn.click(_ui_wild_refresh, [wild_dir_tb], [wild_dd, wild_status])
        wild_dd.change(_ui_wild_load, [wild_dd], [wild_editor, wild_status])
        wild_insert_btn.click(_ui_wild_insert, [wild_dd, prompt], [prompt, wild_status])
        wild_save_btn.click(_ui_wild_save, [wild_dd, wild_editor], [wild_status])
        wild_create_btn.click(_ui_wild_create, [wild_new_name, wild_editor],
                              [wild_dd, wild_status, wild_new_name])
        ckpt_refresh_btn.click(_refresh_checkpoints, [ckpt_dir_tb, ckpt_extra_dir_tb],
                               [ckpt_dd, ckpt_status, preset_dd])
        ckpt_dd.change(_apply_checkpoint, [ckpt_dd], [ckpt_status, gen_steps, guidance, performance])
        transformer_apply_btn.click(_apply_transformer_repo, [transformer_tb],
                                    [ckpt_status, gen_steps, guidance, performance])
        lora_refresh_btn.click(_refresh_loras, [lora_dir_tb], lora_dds + [lora_status])
        # slots entrelaces: dd1, lw1, dd2, lw2, ... (attendu par _apply_loras/_ui_loras_apply)
        _lora_slots = [c for _pair in zip(lora_dds, lora_lws) for c in _pair]
        for _c in lora_dds:
            _c.change(_ui_loras_apply, _lora_slots, [lora_status, lora_keywords_tb])
        for _c in lora_lws:
            _c.change(_apply_loras, _lora_slots, [lora_status])
        lora_kw_btn.click(_ui_loras_keywords, lora_dds,
                          [lora_keywords_tb, lora_status])
        lora_kw_to_prompt_btn.click(_ui_kw_to_prompt, [prompt, lora_keywords_tb], [prompt])
        # ----- Presets (Settings) -----
        _preset_scalars = [prompt, negative, styles, width, height, gen_steps, guidance,
                           sampler_dd, schedule_dd, image_number, ckpt_dd, transformer_tb]
        _preset_io = _preset_scalars + lora_dds + lora_lws
        preset_refresh_btn.click(lambda: gr.update(choices=list_presets()), None, [preset_dd])
        preset_load_btn.click(_ui_preset_load, [preset_dd], _preset_io) \
            .then(_ui_apply_ckpt_silent, [ckpt_dd], [ckpt_status]) \
            .then(_ui_apply_transformer_silent, [transformer_tb], [ckpt_status]) \
            .then(_apply_loras, _lora_slots, [lora_status]) \
            .then(set_sampler, [sampler_dd], None) \
            .then(set_schedule, [schedule_dd], None)
        preset_save_btn.click(_ui_preset_save, [preset_name_tb] + _preset_io, [preset_dd, preset_status])
        preset_update_btn.click(_ui_preset_save, [preset_dd] + _preset_io, [preset_dd, preset_status])
        preset_delete_btn.click(_ui_preset_delete, [preset_dd], [preset_dd, preset_status])
        omni_model_tb.change(lambda r: (set_omni_model(r), f"Omni model set: {r or '(none)'}")[1],
                             [omni_model_tb], [omni_status])
        omni_check_btn.click(_ui_check_omni, None, [omni_status])
        omni_check_btn2.click(_ui_check_omni, None, [omni_status2])
        ab_reindex_btn.click(_ui_ab_reindex,
                             [output_dir, ab_thumb_size, ab_quality, ab_blur, ab_gen_thumbs],
                             [ab_open_link, ab_status])
        # Open Asset Browser in a new tab (fast: manifest now, thumbnails in background).
        ab_open_btn.click(_ui_gallery_open, [output_dir], [ab_status, gallery_url]).then(
            None, [gallery_url], None, js="(u) => { if (u) window.open(u, '_blank'); }")
        # Icones 🖼️: ouvrir l'Asset Browser centre sur le checkpoint / la LoRA selectionne(e).
        _open_js = "(u) => { if (u) window.open(u, '_blank'); }"
        ckpt_open_btn.click(lambda n: _asset_focus_url("models", n), [ckpt_dd],
                            [ckpt_status, gallery_url]).then(None, [gallery_url], None, js=_open_js)
        for _dd, _ob in zip(lora_dds, lora_open_btns):
            _ob.click(lambda n: _asset_focus_url("loras", n), [_dd],
                      [lora_status, gallery_url]).then(None, [gallery_url], None, js=_open_js)
        log_level_dd.change(set_log_level, [log_level_dd], [log_level_status])
        caption_model_dd.change(_ui_set_caption_model, [caption_model_dd], [caption_model_status])
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
        # Onglet unifie Inpaint / Outpaint / Reframe: le mode affiche les bons controles.
        edit_mode.change(
            lambda mo: (gr.update(visible="expand" in mo.lower()),
                        gr.update(visible="reframe" in mo.lower())),
            [edit_mode], [expand_group, reframe_group])
        center_btn.click(lambda: ["Left", "Right", "Top", "Bottom"], None, [inpaint_outpaint])
        edit_btn.click(_ui_edit,
                       [edit_mode, inpaint_editor, inpaint_outpaint, reframe_ratio, reframe_fit,
                        edit_autodescribe, edit_harmonize, edit_harmonize_denoise,
                        prompt, negative, styles, guidance, offload, gen_steps,
                        edit_strength, seed, save_mode, output_dir, output_format, history],
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
        # Seed (facon Fooocus): reutiliser le seed concret du dernier rendu + fixer le seed.
        reuse_seed_btn.click(lambda: gr.update(value=int(cz_pipeline._LAST_SEED)), None, [seed])
        no_seed_inc_cb.change(cz_pipeline.set_no_seed_increment, [no_seed_inc_cb], None)
        _gen_inputs = [prompt, negative, styles, style_random, use_input, inp, input_mode,
                       ref1, ref2, ref3, ref4,
                       faceswap_enable, faceswap_src,
                       width, height, gen_steps, image_number,
                       seed, guidance, offload, esrgan, do_esrgan_cb, do_refine_cb, refine_first_cb, factor, denoise, refine_steps,
                       tile, overlap, refine_tile, refine_overlap, save_mode, output_dir, output_format,
                       history, auto_upscale_cb]
        _gen_outputs = [out, report, history, history_gallery]
        btn.click(_ui_generate, inputs=_gen_inputs, outputs=_gen_outputs)
        if JOB_QUEUE_ENABLED:
            _q_panel = [queue_state, queue_sel, queue_md, queue_add_btn]
            queue_add_btn.click(_ui_queue_add, [*_gen_inputs, queue_state], _q_panel)
            queue_up_btn.click(lambda it, s: _ui_queue_move(it, s, -1),
                               [queue_state, queue_sel], _q_panel)
            queue_down_btn.click(lambda it, s: _ui_queue_move(it, s, +1),
                                 [queue_state, queue_sel], _q_panel)
            queue_rm_btn.click(_ui_queue_remove, [queue_state, queue_sel], _q_panel)
            queue_clear_btn.click(_ui_queue_clear, [queue_state], _q_panel)
            queue_run_btn.click(_ui_queue_run, [queue_state, history],
                                [*_q_panel, out, report, history, history_gallery])
        if XYZ_ENABLED:
            xyz_build_btn.click(_ui_xyz_build,
                                [*_gen_inputs, xyz_xa, xyz_xv, xyz_ya, xyz_yv, xyz_za, xyz_zv,
                                 queue_state],
                                [*_q_panel, xyz_status])
            if XYZ_SUGGEST:
                for _dd, _tb, _sg in ((xyz_xa, xyz_xv, xyz_xs), (xyz_ya, xyz_yv, xyz_ys),
                                      (xyz_za, xyz_zv, xyz_zs)):
                    _dd.change(_ui_xyz_axis_changed, [_dd], [_tb])
                    _sg.click(_ui_xyz_fill, [_dd, _tb], [_tb])
        # Clic sur une image du resultat -> bouton Download avec le vrai nom de fichier.
        out.select(_pick_download, None, [result_dl])
        # Vision Mix & Generate: fusionne les refs en un prompt, puis genere (txt2img).
        vmix_gen_btn.click(
            _ui_compose, [cref1, cref2, cref3, cref4, ollama_model, ollama_url],
            [prompt, compose_status]
        ).then(_ui_generate, inputs=_gen_inputs, outputs=_gen_outputs)
    global _DEMO
    _DEMO = demo  # pour autoriser a la volee les dossiers de sortie changes dans l'UI
    return demo

