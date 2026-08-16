"""crispz-studio - coeur Z-Image (diffusers, BF16): chargement des pipelines
(txt2img / img2img / inpaint / omni), LoRA / checkpoints / transformer, generation
et orchestration (generate / txt2img_run / process_one / outpaint / inpaint) + l'etat
mutable runtime (modele courant, caches pipe, offload, guidance, stop/progress).

Extrait de app.py en UN seul module (step 7): les nombreuses fonctions partagent ces
globaux par reference nue, donc elles vivent ensemble ici. app lit l'etat courant via
cz_pipeline.NAME (BASE_REPO, ZIMAGE_TRANSFORMER, CHECKPOINTS_DIR, LORAS_DIR, LORAS,
OMNI_MODEL, OFFLOAD_MODE, GUIDANCE, _PROGRESS, _STOP, _BASE_PIPE, ...) et pose
cz_pipeline._PROGRESS / cz_pipeline._STOP depuis les handlers UI.
Ne depend que de cz_core / cz_esrgan / cz_imageio (jamais de app ni de gradio).
"""

import os
import sys
import gc
import time
import json
import hashlib
import threading

import numpy as np
import torch
from PIL import Image

import cz_core
from cz_core import (
    CONFIG, HERE, DEVICE, DTYPE, DEFAULT_BASE_REPO,
    DEFAULT_TILE, DEFAULT_OVERLAP, DEFAULT_REFINE_TILE, DEFAULT_REFINE_OVERLAP,
    _prefs, _is_single_file, _log, _dbg,
)
from cz_esrgan import load_esrgan, esrgan_upscale
from cz_imageio import _now_stamp

# Vitesse: autorise TF32 (matmul/cudnn) sur GPU. Gain gratuit sur Ampere+ pour les
# operations fp32 residuelles; les poids restent BF16. Sans effet hors CUDA.
if DEVICE == "cuda":
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


# Modele Z-Image courant. Un repo HF / dossier diffusers -> BASE_REPO. Un fichier
# single-file (.safetensors Civitai) passe comme "modele" -> transformer override
# (le VAE et l'encodeur Qwen3 restent tires du repo de base).
_zmodel = os.environ.get("ZIMAGE_MODEL") or _prefs.get("zimage_model") or DEFAULT_BASE_REPO
ZIMAGE_TRANSFORMER = os.environ.get("ZIMAGE_TRANSFORMER") or _prefs.get("zimage_transformer") or None
if _is_single_file(_zmodel):
    ZIMAGE_TRANSFORMER = _zmodel
    BASE_REPO = DEFAULT_BASE_REPO
else:
    BASE_REPO = _zmodel

# Dossiers de modeles Z-Image: checkpoints single-file a switcher + LoRA a appliquer.
CHECKPOINTS_DIR = (os.environ.get("CHECKPOINTS_DIR") or _prefs.get("checkpoints_dir")
                   or CONFIG.get("checkpoints_dir") or os.path.join(HERE, "checkpoints"))
# Dossier checkpoints supplementaire (optionnel) -> fusionne avec CHECKPOINTS_DIR dans
# la meme liste de checkpoints. Vide par defaut; configurable via UI / prefs / config / env.
CHECKPOINTS_EXTRA_DIR = (os.environ.get("CHECKPOINTS_EXTRA_DIR") or _prefs.get("checkpoints_extra_dir")
                         or CONFIG.get("checkpoints_extra_dir") or "").strip()
LORAS_DIR = (os.environ.get("LORAS_DIR") or _prefs.get("loras_dir")
             or CONFIG.get("loras_dir") or os.path.join(HERE, "loras"))
# LoRA actives: liste de (chemin, poids). Plusieurs LoRA combinables (multi-slots).
LORAS = []
LORA_WEIGHT = float(CONFIG.get("default_lora_weight", 1.0))  # poids par defaut des slots


def _lora_weight_range():
    """Bornes des curseurs de poids LoRA (config 'lora_weight_min'/'lora_weight_max').
    Defaut -2..2: les poids NEGATIFS sont valides et utiles (ils inversent l'effet de la
    LoRA -- ex. un slider 'skinny' a -1 pousse vers l'oppose). Defensif: valeurs illisibles
    ou min >= max -> on retombe sur le defaut."""
    try:
        lo = float(CONFIG.get("lora_weight_min", -2.0))
        hi = float(CONFIG.get("lora_weight_max", 2.0))
    except (TypeError, ValueError):
        _log("lora_weight_min/max: not a number, using -2..2")
        return -2.0, 2.0
    if lo >= hi:
        _log(f"lora_weight_min ({lo}) >= lora_weight_max ({hi}), using -2..2")
        return -2.0, 2.0
    return lo, hi


LORA_WEIGHT_MIN, LORA_WEIGHT_MAX = _lora_weight_range()
# Le poids par defaut doit rester dans les bornes (sinon le curseur naitrait hors plage).
LORA_WEIGHT = min(LORA_WEIGHT_MAX, max(LORA_WEIGHT_MIN, LORA_WEIGHT))
# Modele Omni/Edit (multi-reference). Reglable via config.txt ou l'UI.
OMNI_MODEL = (os.environ.get("ZIMAGE_OMNI_MODEL") or CONFIG.get("zimage_omni_model") or "").strip()

# Caches process-wide. Un pipeline "base" (txt2img ZImagePipeline) detient les
# composants; img2img / inpaint en derivent via from_pipe -> poids partages, pas de
# VRAM en double. Clef de cache = (BASE_REPO, ZIMAGE_TRANSFORMER, OFFLOAD_MODE, LORAS).
_BASE_PIPE = None
_DERIVED = {}
_LOADED_KEY = None
# LoRA reellement posees sur _BASE_PIPE (liste de (chemin, poids)). Sert a echanger les
# LoRA a chaud sans recharger le modele: si ca diverge de LORAS, _apply_loras resynchronise.
_APPLIED_LORAS = []

# Palier 2 (cohabitation VRAM): offload CPU de la passe diffusion. none = tout en VRAM
# (defaut). model = decharge par sous-module (bon compromis). sequential = plus agressif,
# plus lent. N'est PAS de la quantif: les poids restent BF16, ils transitent RAM <-> GPU.
OFFLOAD_MODE = "none"
OFFLOAD_CHOICES = ("none", "model", "sequential")

# CFG. Z-Image *Turbo* = distille -> guidance 0 (defaut). Z-Image *Base* (non Turbo) a
# besoin d'une vraie guidance (~3.5-5) et de plus de steps (~20-28). Reglable par run.
GUIDANCE = 0.0

# Force ratio (facon Fooocus) pour upscale/img2img: si defini, l'image d'ENTREE est
# recadree au centre a ce ratio avant traitement (crop to fit). Vide = ratio natif preserve
# (defaut). Format: 'W:H' ou 'WxH' (ex. '13:19', '832x1216'). Pilotable par l'UI (case a
# cocher + dropdown Aspect ratio) via set_force_ratio, ou par config.txt 'force_upscale_ratio'.
FORCE_RATIO = (os.environ.get("CZ_FORCE_RATIO") or CONFIG.get("force_upscale_ratio") or "").strip()
# Comment atteindre le ratio force: 'crop' = recadrage centre (perd les bords, defaut),
# 'extend' = etend l'image au ratio par outpaint (ne perd rien, ajoute des bandes
# generees par Z-Image). UI (radio) via set_force_ratio_mode, config 'force_ratio_mode'.
FORCE_RATIO_MODE = (os.environ.get("CZ_FORCE_RATIO_MODE")
                    or CONFIG.get("force_ratio_mode") or "crop").strip().lower()
# Passe d'harmonisation du mode extend: apres l'outpaint des bandes, une passe img2img
# LEGERE sur l'image etendue ENTIERE fond les raccords (exposition/texture au niveau
# des jointures, sans re-composer l'image a ce denoise). 0 = desactive.
try:
    EXTEND_DENOISE = float(CONFIG.get("force_ratio_extend_denoise", 0.22) or 0.0)
except Exception:
    EXTEND_DENOISE = 0.22

# Sampler / scheduler. Le pipeline Z-Image impose un schedule `sigmas` custom: seuls
# les schedulers dont set_timesteps accepte `sigmas` fonctionnent. En pratique -> Euler
# flow-matching (natif, defaut), UniPC (multistep) et LCM flow-matching (interessant sur
# les modeles distilles/Turbo: peu de steps, guidance ~0-1).
# Les DPM++ 2M / DPM2a / DPM++ SDE (dpmpp_sde) de diffusers ne prennent PAS de sigmas
# custom -> incompatibles (DPMSolverSDEScheduler exige en plus torchsde). Non exposes.
SAMPLER_CHOICES = ("euler", "unipc", "lcm")
SAMPLER = (os.environ.get("ZIMAGE_SAMPLER") or CONFIG.get("default_sampler") or "euler").strip().lower()
if SAMPLER not in SAMPLER_CHOICES:
    SAMPLER = "euler"

# Schedule de sigmas (= le "scheduler" facon ComfyUI). sgm_uniform = natif Z-Image
# (linspace + dynamic shift). beta/karras/exponential = re-mapping des sigmas applique
# PAR-DESSUS le schedule du pipeline (FlowMatchEuler/UniPC: use_*_sigmas). beta -> scipy.
SCHEDULE_CHOICES = ("sgm_uniform", "beta", "karras", "exponential")
# 'simple' (ComfyUI) designe EXACTEMENT le schedule natif expose ici sous 'sgm_uniform':
# les sigmas que le pipeline Z-Image impose sont linspace(1, 1/n, n)
# (get_default_z_image_sigmas), ce que ComfyUI appelle 'simple' sur un modele flow. Accepte
# en entree partout (config/env/CLI/XYZ) pour recopier une recette CivitAI au mot pres,
# mais normalise vers le nom canonique: metadonnees et presets ne portent qu'un seul nom.
_SCHEDULE_ALIASES = {"simple": "sgm_uniform"}
SCHEDULE_INPUTS = SCHEDULE_CHOICES + tuple(_SCHEDULE_ALIASES)   # listes ouvertes (CLI/XYZ)


def _norm_schedule(name, default="sgm_uniform"):
    """Nom de schedule -> nom canonique (alias resolus). Inconnu -> `default`."""
    n = (name or "").strip().lower()
    n = _SCHEDULE_ALIASES.get(n, n)
    return n if n in SCHEDULE_CHOICES else default


SCHEDULE = _norm_schedule(os.environ.get("ZIMAGE_SCHEDULE") or CONFIG.get("default_schedule"))
_SCHEDULE_FLAG = {"beta": "use_beta_sigmas", "karras": "use_karras_sigmas",
                  "exponential": "use_exponential_sigmas"}  # sgm_uniform -> aucun flag (natif)
# Config natif du scheduler du modele (capture au 1er chargement) -> base de construction
# des autres samplers (conserve shift/flow params quel que soit le sampler courant).
_BASE_SCHED_CONFIG = None

# Hook de progression UI (gradio gr.Progress). None hors UI (CLI/serveur). Pose par
# les handlers via cz_pipeline._PROGRESS = ...
_PROGRESS = None
# Stop "facon Fooocus": flag global + interruption des pipelines diffusers. Pose par
# les handlers via cz_pipeline._STOP = ... et par request_stop().
_STOP = False

# Verrou GPU: serialise TOUTES les generations. Gradio ne serialise pas les events de
# LISTENERS differents (Generate manuel vs Run queue vs detaileur): deux threads peuvent
# alors appeler le MEME pipeline partage et stepper le MEME scheduler -> son index
# depasse la fin ("IndexError: index 31 is out of bounds for dimension 0 with size 31",
# scheduling_flow_match_euler_discrete.step). RLock: les imbrications d'un meme thread
# (txt2img_run -> generate, process_one -> _refine_whole) restent libres.
_GPU_LOCK = threading.RLock()


def _gpu_serial(fn):
    """Decorateur: execute fn sous _GPU_LOCK (une seule generation GPU a la fois)."""
    import functools

    @functools.wraps(fn)
    def _locked(*args, **kwargs):
        with _GPU_LOCK:
            return fn(*args, **kwargs)
    return _locked

# Gestion du seed (facon Fooocus):
#  _LAST_SEED         = seed CONCRET du dernier rendu (un -1 aleatoire est resolu en
#                       valeur reelle) -> bouton "Reuse last seed" + metadonnees justes.
#  _NO_SEED_INCREMENT = True -> tout un batch utilise le meme seed (pas de +i par image).
_LAST_SEED = -1
_NO_SEED_INCREMENT = False
# True -> en txt2img+upscale, sauve AUSSI l'image txt2img d'origine (avant l'upscale).
_SAVE_PRE_UPSCALE = bool(CONFIG.get("save_pre_upscale", False))


def set_no_seed_increment(v):
    global _NO_SEED_INCREMENT
    _NO_SEED_INCREMENT = bool(v)


def set_save_pre_upscale(v):
    global _SAVE_PRE_UPSCALE
    _SAVE_PRE_UPSCALE = bool(v)


def set_guidance(g):
    global GUIDANCE
    GUIDANCE = float(g)


def _scheduler_accepts_sigmas(sched):
    """Le pipeline Z-Image appelle set_timesteps(..., sigmas=<schedule custom>). Un
    scheduler dont set_timesteps n'accepte pas `sigmas` plante a la generation."""
    import inspect
    try:
        return "sigmas" in inspect.signature(sched.set_timesteps).parameters
    except Exception:
        return False


def _build_scheduler(sampler, schedule, config):
    """Construit le scheduler choisi (sampler x schedule) depuis le config natif du modele.
    schedule (sgm_uniform/beta/karras/exponential) = remapping des sigmas (use_*_sigmas)."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    kw = {}
    flag = _SCHEDULE_FLAG.get((schedule or "").lower())
    if flag:
        kw[flag] = True
    name = (sampler or "euler").lower()
    if name == "unipc":
        from diffusers import UniPCMultistepScheduler
        try:
            return UniPCMultistepScheduler.from_config(config, use_flow_sigmas=True, **kw)
        except Exception:
            return UniPCMultistepScheduler.from_config(config, **kw)
    if name == "lcm":
        # LCM flow-matching: accepte les sigmas custom du pipeline ET les flags de
        # schedule. Repli sur Euler si la version de diffusers ne l'expose pas.
        try:
            from diffusers import FlowMatchLCMScheduler
            return FlowMatchLCMScheduler.from_config(config, **kw)
        except Exception as e:
            _log(f"sampler 'lcm' unavailable ({e}); falling back to euler")
    return FlowMatchEulerDiscreteScheduler.from_config(config, **kw)


def _apply_sampler(pipe):
    """Pose le scheduler courant (SAMPLER x SCHEDULE) sur un pipe. Verifie la compatibilite
    (sigmas custom) et retombe sur Euler/sgm_uniform si KO -> jamais de crash a la generation."""
    if _BASE_SCHED_CONFIG is None:
        return
    from diffusers import FlowMatchEulerDiscreteScheduler
    try:
        sched = _build_scheduler(SAMPLER, SCHEDULE, _BASE_SCHED_CONFIG)
        if not _scheduler_accepts_sigmas(sched):
            raise ValueError(f"{type(sched).__name__} n'accepte pas les sigmas custom de Z-Image")
        pipe.scheduler = sched
        _dbg(f"sampler applied: {SAMPLER}/{SCHEDULE} -> {type(pipe.scheduler).__name__}")
    except Exception as e:
        _log(f"sampler '{SAMPLER}/{SCHEDULE}' incompatible ({e}); fallback Euler/sgm_uniform")
        try:
            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(_BASE_SCHED_CONFIG)
        except Exception:
            pass


def _reapply_sampler_all():
    """Re-applique le scheduler courant a tous les pipes en cache (base + derives)."""
    for p in [_BASE_PIPE] + list(_DERIVED.values()):
        if p is not None:
            _apply_sampler(p)


def set_sampler(name):
    """Change le sampler (euler/unipc) et le re-applique aux pipes en cache (pas de
    rechargement). Pas d'effet sur le pipe Omni (scheduler propre)."""
    global SAMPLER
    name = (name or "euler").strip().lower()
    if name not in SAMPLER_CHOICES:
        name = "euler"
    if name != SAMPLER:
        SAMPLER = name
        _reapply_sampler_all()
        _log(f"sampler -> {SAMPLER}")
    return f"Sampler: {SAMPLER} / {SCHEDULE}"


def set_schedule(name):
    """Change le schedule de sigmas (sgm_uniform/beta/karras/exponential, alias 'simple'
    = sgm_uniform) et le re-applique aux pipes en cache."""
    global SCHEDULE
    name = _norm_schedule(name)
    if name != SCHEDULE:
        SCHEDULE = name
        _reapply_sampler_all()
        _log(f"schedule -> {SCHEDULE}")
    return f"Sampler: {SAMPLER} / {SCHEDULE}"


def _progress(frac, desc=""):
    if _PROGRESS is not None:
        try:
            _PROGRESS(min(1.0, max(0.0, float(frac))), desc)
        except Exception:
            pass


# ---- Feedback de chargement des modeles (terminal + UI) ----
# from_pretrained est bloquant et silencieux (le 1er chargement telecharge depuis HF ->
# plusieurs minutes). On execute le chargement dans un thread et on rafraichit toutes les
# ~2s une ligne terminal + la barre Gradio (temps ecoule + VRAM allouee). Config bloc
# "load_progress"; enabled=false -> chargement direct (aucun thread, zero cout).
_LOAD_CFG = CONFIG.get("load_progress") if isinstance(CONFIG.get("load_progress"), dict) else {}
LOAD_PROGRESS_ENABLED = bool(_LOAD_CFG.get("enabled", True))
_LOAD_TARGET_GB = float(_LOAD_CFG.get("target_vram_gb", 14.0))
_LOAD_HEARTBEAT = float(_LOAD_CFG.get("heartbeat_s", 2.0))


def _fmt_load(label, elapsed, vram_gb):
    """Texte de progression de chargement (pur, testable). VRAM > 0 -> phase chargement
    en memoire; sinon phase download/lecture disque."""
    if vram_gb > 0.05:
        return f"{label}... {elapsed:.0f}s | {vram_gb:.1f} GB in VRAM"
    return f"{label}... {elapsed:.0f}s (downloading / reading, first run only)"


def _load_pct(elapsed, vram_gb, target_gb=None):
    """% honnete: base sur la VRAM allouee / cible une fois le chargement en memoire
    commence (plafonne 0.95); pendant le download (VRAM~0) petite barre temporelle."""
    target_gb = target_gb or _LOAD_TARGET_GB
    if vram_gb <= 0.05:
        return min(0.12, elapsed / 600.0)
    return min(0.95, vram_gb / max(1.0, float(target_gb)))


def _load_monitor(label, fn):
    """Execute fn() (chargement bloquant) dans un thread et rafraichit terminal + UI
    (temps + VRAM) toutes les ~2s. Renvoie le resultat de fn (releve son exception)."""
    if not LOAD_PROGRESS_ENABLED:
        return fn()
    box = {}

    def _work():
        try:
            box["v"] = fn()
        except BaseException as e:   # noqa: BLE001 - on re-leve dans le thread principal
            box["e"] = e

    th = threading.Thread(target=_work, daemon=True)
    t0 = time.time()
    th.start()
    while True:
        th.join(timeout=_LOAD_HEARTBEAT)
        el = time.time() - t0
        vram = (torch.cuda.memory_allocated() / 1024 ** 3) if DEVICE == "cuda" else 0.0
        line = _fmt_load(label, el, vram)
        if cz_core.LOG_LEVEL >= 1:
            sys.stderr.write("\r[crispz][load] " + line + "        ")
            sys.stderr.flush()
        _progress(_load_pct(el, vram), "Loading " + line)
        if not th.is_alive():
            break
    if cz_core.LOG_LEVEL >= 1:
        sys.stderr.write("\n")
        sys.stderr.flush()
    if "e" in box:
        raise box["e"]
    return box.get("v")


def request_stop():
    """Demande l'arret: stoppe la boucle de debruitage en cours (pipe._interrupt) et
    les boucles batch/tuiles (_STOP). Quasi-immediat (s'arrete au pas suivant)."""
    global _STOP
    _STOP = True
    n = 0
    for p in [_BASE_PIPE] + list(_DERIVED.values()):
        if p is not None:
            try:
                p._interrupt = True
                n += 1
            except Exception:
                pass
    _log(f"STOP requested (interrupt set on {n} pipeline(s))")
    return "Stopping..."


def set_zimage_model(repo_or_path):
    """Change le modele Z-Image. Un repo HF / dossier diffusers -> BASE_REPO.
    Un fichier single-file (.safetensors Civitai) -> transformer override.
    Invalide le pipe si change."""
    global BASE_REPO, ZIMAGE_TRANSFORMER
    if not repo_or_path:
        return
    if _is_single_file(repo_or_path):
        # Changement de transformer seul: PAS de free_vram -> _ensure_base echangera
        # uniquement le transformer (VAE + encodeur Qwen3 gardes en VRAM).
        if repo_or_path != ZIMAGE_TRANSFORMER:
            ZIMAGE_TRANSFORMER = repo_or_path
            _log("Z-Image transformer (single-file) changed -> transformer swap on next run")
    elif repo_or_path != BASE_REPO:
        # Le repo de base change: VAE/encodeur/tokenizer changent aussi -> reload complet.
        BASE_REPO = repo_or_path
        free_vram()
        _log("Z-Image base repo changed -> will reload")


def set_zimage_transformer(path):
    """Definit (ou enleve avec '' / None) le transformer single-file.

    NE libere PAS le pipeline: a repo de base identique, _ensure_base ne rechargera que
    le transformer (_swap_transformer) et gardera VAE + encodeur Qwen3 en VRAM."""
    global ZIMAGE_TRANSFORMER
    path = path or None
    if path != ZIMAGE_TRANSFORMER:
        ZIMAGE_TRANSFORMER = path
        _log(f"Z-Image transformer -> {path or '(repo de base)'} "
             "-> transformer swap on next run (base components kept)")


_HDR_CACHE = {}          # (chemin, taille, mtime) -> en-tete JSON deja parse


def _file_key(path):
    """Identite stable et pas chere d'un fichier: (chemin absolu, taille, mtime)."""
    st = os.stat(path)
    return (os.path.abspath(path), st.st_size, int(st.st_mtime))


def _safetensors_header(path):
    """En-tete JSON d'un .safetensors (noms/dtypes/shapes des tenseurs, JAMAIS les
    poids) -- lecture de quelques centaines de Ko au plus, meme sur un fichier de 12 Go.
    Memoise par (chemin, taille, mtime): le listing, la detection de format et le
    loader lisent le meme en-tete, inutile de retaper le disque (HDD) a chaque fois."""
    import struct
    try:
        key = _file_key(path)
    except OSError:
        key = None
    if key is not None and key in _HDR_CACHE:
        return _HDR_CACHE[key]
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(min(n, 10_000_000)).decode("utf-8", "ignore"))
    if key is not None:
        if len(_HDR_CACHE) > 512:        # borne memoire (dossiers de modeles enormes)
            _HDR_CACHE.clear()
        _HDR_CACHE[key] = hdr
    return hdr


def _safetensors_unsupported(path):
    """Renvoie une raison (str) si le .safetensors n'est PAS chargeable, sinon None.
    Lit juste l'en-tete (rapide). Deux cas restent non supportes:
      - fichier LoRA range dans le dossier checkpoints (cles kohya/peft)
      - SVDQuant / Nunchaku (tenseurs nommes '*.qweight'): poids pre-quantifies INT4
        qui exigent le runtime nunchaku (kernels dedies), pas dequantifiables ici.
    Les FP8 / INT8 'scaled' facon ComfyUI ne sont PLUS rejetes: ils passent par le
    loader dequant (_safetensors_dequant + _load_dequant_state_dict)."""
    try:
        hdr = _safetensors_header(path)
        has_qweight = False
        lora_keys = 0
        for k, v in hdr.items():
            if k == "__metadata__" or not isinstance(v, dict):
                continue
            if k.endswith(".qweight"):
                has_qweight = True
            if (".lora_down." in k or ".lora_up." in k or ".lora_A." in k
                    or ".lora_B." in k or k.startswith(("lora_unet_", "lora_te"))):
                lora_keys += 1
        # Fichier LoRA range dans le dossier checkpoints (erreur classique): le charger
        # comme transformer envoie diffusers chercher une config par defaut (SD1.5) ->
        # 404 'stable-diffusion-v1-5 does not appear to have a file named config.json'.
        if lora_keys >= 4:
            return "LoRA file, not a checkpoint - move it to the LoRA folder and pick it in Models > LoRA"
        # '*.qweight' = poids pre-quantifies (SVDQuant/Nunchaku, GPTQ-like). Signal net:
        # un checkpoint BF16/FP16 normal n'a jamais de 'qweight'.
        if has_qweight:
            return "SVDQuant/Nunchaku INT4"
    except Exception:
        pass
    return None


def _safetensors_dequant(path):
    """Renvoie le schema de quantification ComfyUI a dequantifier au chargement
    ('FP8', 'FP8 scaled' ou 'INT8 scaled'), sinon None (BF16/FP16 -> chemin normal).
    Format 'scaled' ComfyUI observe sur les checkpoints Civitai:
      X.weight (F8_E4M3 ou I8) + X.weight_scale (F32, scalaire ou par ligne [out,1])
      + X.comfy_quant (petit blob U8 descripteur, a jeter).
    NB: un bundle AIO dont SEUL l'encodeur texte est quantifie (transformer BF16)
    declenche aussi -> le loader dequant filtre le transformer et le laisse intact.
    U8 seul ne declenche pas: les blobs 'comfy_quant' sont U8 dans des fichiers sains."""
    try:
        hdr = _safetensors_header(path)
        has_fp8 = has_int = has_scale = False
        for k, v in hdr.items():
            if k == "__metadata__" or not isinstance(v, dict):
                continue
            dt = str(v.get("dtype", "")).upper()
            if dt.startswith("F8"):
                has_fp8 = True
            elif dt in ("I8", "I4", "U4", "INT8"):
                has_int = True
            if k.endswith(("weight_scale", "scale_weight")):
                has_scale = True
        if has_fp8:
            return "FP8 scaled" if has_scale else "FP8"
        if has_int and has_scale:
            return "INT8 scaled"
    except Exception:
        pass
    return None


# Architecture attendue dans les .gguf. Un GGUF de diffusion declare son archi dans
# 'general.architecture': les conversions ComfyUI-GGUF de Z-Image (unsloth, jayn7,
# QuantStack...) declarent 'lumina2' (S3-DiT, lignee Lumina). 'flux', 'qwen_image',
# 'llama'... = autres modeles qui exigent leur propre pipeline -> ecartes.
GGUF_ARCH = str(CONFIG.get("gguf_arch") or "lumina2").strip().lower()

_GGUF_FIXED = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
               6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def _gguf_skip(f, t):
    """Avance le flux au-dela d'une valeur GGUF sans la lire (strings et arrays inclus)."""
    import struct
    if t == 8:                                   # string
        f.seek(struct.unpack("<Q", f.read(8))[0], 1)
        return
    if t == 9:                                   # array
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        if et in _GGUF_FIXED:
            f.seek(struct.calcsize(_GGUF_FIXED[et]) * n, 1)
        else:
            for _ in range(n):
                _gguf_skip(f, et)
        return
    f.seek(struct.calcsize(_GGUF_FIXED[t]), 1)


def _gguf_arch(path, max_kv=64):
    """'general.architecture' d'un .gguf -- lit seulement l'en-tete (quelques Ko), jamais
    les poids. Renvoie 'lumina2' / 'flux' / 'qwen_image' / 'llama'... ou None si illisible
    (dans ce cas on ne filtre pas: mieux vaut tenter que d'ecarter un modele valide)."""
    import struct
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return None
            f.seek(4 + 8, 1)                     # version (u32) + tensor_count (u64)
            nkv = struct.unpack("<Q", f.read(8))[0]
            for _ in range(min(nkv, max_kv)):
                kl = struct.unpack("<Q", f.read(8))[0]
                if kl > 4096:                    # en-tete incoherent -> on abandonne
                    return None
                key = f.read(kl).decode("utf-8", "replace")
                t = struct.unpack("<I", f.read(4))[0]
                if key == "general.architecture" and t == 8:
                    n = struct.unpack("<Q", f.read(8))[0]
                    return f.read(n).decode("utf-8", "replace").strip().lower()
                _gguf_skip(f, t)
    except Exception as e:
        _dbg(f"gguf header read failed {path}: {e}")
    return None


# Prefixes de tenseurs du layout Z-Image ORIGINAL (celui que le loader GGUF de diffusers
# sait mapper -- conversions ComfyUI-GGUF: unsloth/jayn7/QuantStack, avec ou sans prefixe
# ComfyUI). Certains GGUF sont convertis par stable-diffusion.cpp avec un schema compact
# renomme: l'archi declaree est bonne mais AUCUNE cle ne matche -> tous les poids restent
# sur le device 'meta' et le .to(device) explose en "Cannot copy out of meta tensor".
# On detecte ce cas a l'en-tete pour refuser proprement.
_GGUF_OK_PREFIXES = ("layers.", "noise_refiner", "context_refiner", "final_layer",
                     "x_embedder", "cap_embedder", "t_embedder",
                     "model.diffusion_model.")


def _gguf_layout_unsupported(path):
    """Renvoie une raison (str) si le .gguf n'utilise PAS le layout de tenseurs Z-Image
    original attendu par diffusers, sinon None. Lecture d'en-tete seule (gguf mmap)."""
    try:
        from gguf import GGUFReader
        r = GGUFReader(path)
        names = [t.name for t in r.tensors]
        if not names:
            return None                      # illisible -> ne pas ecarter a tort
        if any(n.startswith(_GGUF_OK_PREFIXES) for n in names):
            return None
        return ("GGUF with a non-standard tensor layout (e.g. stable-diffusion.cpp "
                "conversion); diffusers cannot map it — use a ComfyUI-GGUF-style "
                "export (unsloth/jayn7) or the BF16/FP16 .safetensors build")
    except Exception as e:
        _dbg(f"gguf layout check failed {path}: {e}")
        return None


def _is_gguf_path(p):
    return bool(p) and str(p).lower().endswith(".gguf")


def _checkpoint_dirs():
    """Dossiers a scanner pour les checkpoints single-file: principal + extra (si defini),
    sans doublon de chemin."""
    dirs = [CHECKPOINTS_DIR]
    if CHECKPOINTS_EXTRA_DIR and CHECKPOINTS_EXTRA_DIR not in dirs:
        dirs.append(CHECKPOINTS_EXTRA_DIR)
    return dirs


def list_checkpoints():
    """Modeles Z-Image single-file (.safetensors, .gguf) des dossiers checkpoints
    (principal + extra, fusionnes dans une seule liste). Les FP8/INT8 'scaled' ComfyUI
    sont listes (dequantifies au chargement); restent exclus: LoRA egarees, SVDQuant/
    Nunchaku (runtime dedie requis), GGUF d'une autre architecture ou au layout
    stable-diffusion.cpp. En cas de meme nom de fichier, le dossier principal a la
    priorite."""
    out = []
    seen = set()
    for d in _checkpoint_dirs():
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f in seen:
                continue
            if not f.lower().endswith((".safetensors", ".ckpt", ".pt", ".sft", ".gguf")):
                continue
            if f.lower().endswith(".safetensors"):
                reason = _safetensors_unsupported(os.path.join(d, f))
                if reason:
                    _log(f"checkpoint skipped ({reason}): {f}")
                    continue
            if f.lower().endswith(".gguf"):
                a = _gguf_arch(os.path.join(d, f))
                # a=None -> en-tete illisible: on laisse passer (ne pas ecarter a tort).
                if a and a != GGUF_ARCH:
                    _log(f"checkpoint skipped (GGUF architecture '{a}', this build only "
                         f"loads '{GGUF_ARCH}' = Z-Image; that model needs its own "
                         f"pipeline and text encoder/VAE): {f}")
                    continue
                lay = _gguf_layout_unsupported(os.path.join(d, f))
                if lay:
                    _log(f"checkpoint skipped ({lay}): {f}")
                    continue
            seen.add(f)
            out.append(f)
    return sorted(out)


def resolve_checkpoint(name):
    """Chemin absolu d'un checkpoint single-file depuis son nom de fichier, cherche dans
    les dossiers checkpoints (principal puis extra). Renvoie name tel quel s'il est deja
    absolu; fallback sur le dossier principal si introuvable."""
    if not name or os.path.isabs(name):
        return name
    for d in _checkpoint_dirs():
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return os.path.join(CHECKPOINTS_DIR, name)


def list_loras():
    """LoRA (.safetensors / .ckpt / .pt) du dossier loras, RECURSIF (sous-dossiers inclus).
    Renvoie des chemins RELATIFS a LORAS_DIR avec des '/' (ex. 'sous-dossier/ma_lora.safetensors')
    -> set_loras / resolve les resolvent via os.path.join(LORAS_DIR, name)."""
    if not os.path.isdir(LORAS_DIR):
        return []
    exts = (".safetensors", ".ckpt", ".pt")
    out = []
    for root, _dirs, files in os.walk(LORAS_DIR):
        for f in files:
            if f.lower().endswith(exts):
                rel = os.path.relpath(os.path.join(root, f), LORAS_DIR).replace(os.sep, "/")
                out.append(rel)
    return sorted(out)


def set_checkpoints_dir(path):
    global CHECKPOINTS_DIR
    if path:
        CHECKPOINTS_DIR = path


def set_checkpoints_extra_dir(path):
    """Definit (ou efface avec '' / None) le dossier checkpoints supplementaire."""
    global CHECKPOINTS_EXTRA_DIR
    CHECKPOINTS_EXTRA_DIR = (path or "").strip()


def set_loras_dir(path):
    global LORAS_DIR
    if path:
        LORAS_DIR = path


def checkpoint_badge(name):
    """Etiquette courte de format pour un checkpoint (dropdown UI):
    'BF16 - 11.5 GB', 'GGUF Q6_K - 5.5 GB', 'FP8->bf16 - 5.7 GB (slow 1st load)'...
    Renvoie '' pour un repo HF (pas un fichier) ou si l'en-tete est illisible.
    Tout passe par l'en-tete memoise: aucun cout disque supplementaire au listing.
    ASCII only: ce libelle finit aussi dans les logs console (cp1252 sous Windows,
    ou une fleche unicode leve UnicodeEncodeError et tue le run)."""
    try:
        path = resolve_checkpoint(name)
        if not path or not os.path.isfile(path):
            return ""
        gb = os.path.getsize(path) / 1024**3
        if _is_gguf_path(path):
            import re
            m = re.search(r"(Q\d+[_A-Za-z0-9]*)", os.path.basename(path))
            return f"GGUF {m.group(1)} - {gb:.1f} GB" if m else f"GGUF - {gb:.1f} GB"
        dq = _safetensors_dequant(path)
        if dq:
            # 'FP8 scaled' / 'INT8 scaled' -> on garde le mot-cle court; le 1er
            # chargement paie le dequant, les suivants relisent le cache disque.
            short = dq.split()[0]
            cached = _dequant_cache_path(path)
            hint = "cached" if (cached and os.path.isfile(cached)) else "slow 1st load"
            return f"{short}->bf16 - {gb:.1f} GB ({hint})"
        return f"BF16 - {gb:.1f} GB"
    except Exception as e:
        _dbg(f"checkpoint_badge failed for {name}: {e}")
        return ""


def _read_safetensors_metadata(path):
    """Lit le header JSON (__metadata__) d'un .safetensors SANS charger les poids."""
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = f.read(n)
    return (json.loads(header.decode("utf-8")) or {}).get("__metadata__", {}) or {}


def lora_keywords(path):
    """Extrait les mots-cles / trigger words d'une LoRA depuis ses metadonnees:
    champs trigger explicites + top tags d'entrainement (ss_tag_frequency)."""
    if not path or not os.path.isfile(path):
        return ""
    try:
        meta = _read_safetensors_metadata(path)
    except Exception as e:
        _dbg(f"lora metadata read failed: {e}")
        return ""
    words = []
    for k in ("ss_trigger_words", "modelspec.trigger_phrase", "trigger_words",
              "activation text", "ss_activation_text"):
        v = meta.get(k)
        if v:
            words.append(v if isinstance(v, str) else ", ".join(map(str, v)))
    tf = meta.get("ss_tag_frequency")
    if tf:
        try:
            d = json.loads(tf) if isinstance(tf, str) else tf
            counts = {}
            for ds in d.values():
                for tag, c in ds.items():
                    counts[tag] = counts.get(tag, 0) + int(c)
            words.extend(sorted(counts, key=counts.get, reverse=True)[:15])
        except Exception:
            pass
    seen, out = set(), []
    for w in words:
        for part in str(w).split(","):
            part = part.strip()
            if part and part.lower() not in seen:
                seen.add(part.lower())
                out.append(part)
    return ", ".join(out)


def set_loras(slots):
    """Definit les LoRA actives. slots = liste de (nom_ou_None, poids). Resout les
    noms en chemins, ignore les None.

    NE recharge PAS le modele: les LoRA sont echangees A CHAUD sur le transformer deja
    en VRAM (_apply_loras, appele par _ensure_base au run suivant). Changer une LoRA
    coutait auparavant un rechargement complet (transformer + VAE + encodeur Qwen3)."""
    global LORAS
    new = []
    for name, weight in slots:
        if name and name not in ("None", "none", ""):
            p = name if os.path.isabs(name) else os.path.join(LORAS_DIR, name)
            new.append((p, float(weight)))
    if new != LORAS:
        LORAS = new
        _log("LoRAs -> " + (", ".join(f"{os.path.basename(p)}@{w}" for p, w in new) or "(none)")
             + " -> applied on next run (hot-swap, no model reload)")


def set_omni_model(repo):
    """Definit le modele Omni/Edit (repo HF ou dossier). Invalide le pipe omni."""
    global OMNI_MODEL
    repo = (repo or "").strip()
    if repo != OMNI_MODEL:
        OMNI_MODEL = repo
        _DERIVED.pop("omni", None)
        _log(f"Omni model -> {repo or '(none)'}")


def check_omni_available():
    """Teste l'existence des repos Omni/Edit sur Hugging Face (API publique)."""
    import urllib.request
    found = []
    for repo in ("Tongyi-MAI/Z-Image-Omni-Base", "Tongyi-MAI/Z-Image-Edit"):
        try:
            req = urllib.request.Request("https://huggingface.co/api/models/" + repo,
                                         headers={"User-Agent": "crispz-studio"})
            with urllib.request.urlopen(req, timeout=8) as r:
                if r.status == 200:
                    found.append(repo)
        except Exception:
            pass
    if found:
        return ("**Omni model available!** " + ", ".join(f"`{r}`" for r in found)
                + " - set it in config.txt `zimage_omni_model` (or Models tab).")
    return ("Not released yet. Z-Image-Omni-Base / Z-Image-Edit are still 'coming "
            "soon'. The Omni tab will work once they ship.")


def set_offload_mode(mode):
    """Change le mode d'offload CPU. Invalide le pipe (hooks poses au chargement)."""
    global OFFLOAD_MODE
    mode = mode if mode in OFFLOAD_CHOICES else "none"
    if mode != OFFLOAD_MODE:
        OFFLOAD_MODE = mode
        free_vram()
        _log(f"offload -> {OFFLOAD_MODE}: pipeline invalidated -> will reload")


# Seuil (Go) de VRAM occupee par d'AUTRES processus au-dela duquel on previent avant
# de charger un modele. Deux instances qui se partagent le GPU font deborder la VRAM en
# RAM partagee: les rendus passent de 2 s a 300+ s/step sans message d'erreur.
# 0 = garde desactivee.
try:
    GPU_BUSY_WARN_GB = float(CONFIG.get("gpu_busy_warn_gb", 2.0) or 0)
except Exception:
    GPU_BUSY_WARN_GB = 2.0


def gpu_foreign_vram_gb():
    """VRAM (Go) utilisee sur le GPU par des processus AUTRES que celui-ci.
    mem_get_info donne le libre/total reels du device; ce qu'on en occupe nous-memes
    est `memory_reserved` (l'allocateur torch). La difference vient d'ailleurs:
    autre instance de l'app, ComfyUI, un jeu, un navigateur en accel materielle."""
    if DEVICE != "cuda":
        return 0.0
    try:
        free, total = torch.cuda.mem_get_info()
        ours = torch.cuda.memory_reserved()
        return max(0.0, (total - free - ours) / 1024**3)
    except Exception as e:
        _dbg(f"mem_get_info unavailable: {e}")
        return 0.0


def gpu_busy_warning():
    """Message d'avertissement (str) si un autre processus occupe le GPU, sinon ''.
    Consomme par l'UI (banniere de statut) et la CLI (stderr) avant un chargement."""
    if GPU_BUSY_WARN_GB <= 0:
        return ""
    used = gpu_foreign_vram_gb()
    if used < GPU_BUSY_WARN_GB:
        return ""
    try:
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:
        total = 0.0
    return (f"another process is using {used:.1f} GB of VRAM"
            + (f" out of {total:.0f} GB" if total else "")
            + " - sharing the GPU makes renders spill to shared RAM "
              "(seconds -> minutes per step). Close the other app "
              "(ComfyUI, a second crispz instance, a game) for full speed.")


def free_vram():
    """Libere le pipeline de base + les pipelines derives et rend la VRAM
    (palier 3: unload sur inactivite ou endpoint /unload). Rechargement paresseux."""
    global _BASE_PIPE, _DERIVED, _LOADED_KEY, _APPLIED_LORAS, _CN_PIPE, _CN_PIPE_KEY, _CN_MODEL
    _BASE_PIPE = None
    _DERIVED = {}
    _LOADED_KEY = None
    _APPLIED_LORAS = []      # plus de pipe -> plus d'adaptateur pose
    # Le pipeline ControlNet partage les composants du base (donc devient invalide) et
    # le modele controlnet pese 6.7 Go a lui seul: on rend tout.
    _CN_PIPE = _CN_PIPE_KEY = _CN_MODEL = None
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# Au-dela de ce cote (px) on active l'attention slicing (whole-image 2K+ -> evite le
# spill VRAM 32 Go). En-dessous (tuiles 1024, txt2img 1024/1536) -> slicing OFF =
# attention SDPA native = RAPIDE (comme ComfyUI). Reglable via config attention_slice_above.
_SLICE_ABOVE = int(CONFIG.get("attention_slice_above", 1664))

# Garde-fou: au-dela de ce cote (px), un refine "whole image" (refine_tile=0) est auto-
# tuile (tuile 1024). Defaut = le seuil de slicing: au-dela, un whole-image serait slice
# (lent: ~120s en 2K) ET risque le spill VRAM (4K -> crash). Tuiler est plus rapide ET sur.
_AUTO_TILE_ABOVE = int(CONFIG.get("auto_refine_tile_above", _SLICE_ABOVE))

# Plafond de denoise pour le refine TUILE. En tuiles, chaque tuile est rediffusee avec le
# prompt global -> a fort denoise la diffusion reconstruit le sujet (ex: la tasse) DANS
# chaque tuile = duplications. On plafonne donc le denoise par tuile (le contenu existant
# guide alors la diffusion, facon Ultimate SD Upscale). Le refine "whole image" garde le
# denoise demande (pas de duplication possible: une seule passe sur toute la compo).
# Reglable via config refine_tile_denoise_cap (0 = pas de plafond).
_TILE_DENOISE_CAP = float(CONFIG.get("refine_tile_denoise_cap", 0.40))

# Prompt utilise pour le refine TUILE. Le prompt global decrit TOUTE la composition (pas
# la tuile) -> le passer a chaque tuile pousse la diffusion a recreer le sujet (la tasse)
# dans des tuiles qui ne sont que du fond. Par defaut on passe donc un prompt VIDE: chaque
# tuile se contente d'affiner le detail local. Valeurs config refine_tile_prompt:
#   "" (defaut) = prompt vide par tuile
#   "global"/"scene" = reutilise le prompt de la scene (ancien comportement)
#   tout autre texte = prompt generique applique a chaque tuile (ex: "high detail, sharp")
_TILE_PROMPT = str(CONFIG.get("refine_tile_prompt", ""))


def _tile_prompt(scene_prompt):
    """Prompt a utiliser par tuile selon la config (vide par defaut, anti-duplication)."""
    if _TILE_PROMPT.strip().lower() in ("global", "scene"):
        return scene_prompt or ""
    return _TILE_PROMPT


def _set_slicing(pipe, longest_side):
    """Regle le menagement VRAM selon le plus grand cote a traiter. Appele avant CHAQUE
    passe de diffusion (txt2img/refine/tuile/inpaint/outpaint/omni).

    ATTENTION, piege verifie: `pipe.enable_attention_slicing()` ne fait RIEN ici.
    DiffusionPipeline.set_attention_slice ne s'applique qu'aux modules exposant
    `set_attention_slice`, et NI ZImageTransformer2DModel NI AutoencoderKL ne le
    definissent (verifie sur diffusers 0.39.0.dev0) -- le pipeline les filtre en
    silence. Le vrai levier sur ce modele est le VAE: tiling/slicing plafonnent le pic
    de l'encode/decode, qui est la partie qui deborde en 2K+ (le transformer, lui,
    tient grace a SDPA). Le tiling VAE est deja pose au chargement; on le REAFFIRME ici
    en haute resolution (un from_pipe / un swap de transformer peut recreer le VAE)."""
    try:
        vae = getattr(pipe, "vae", None)
        if vae is None:
            return
        if int(longest_side) > _SLICE_ABOVE:
            vae.enable_slicing()
            vae.enable_tiling()
    except Exception as e:
        _dbg(f"vae slicing/tiling not applied: {e}")


def _vram_str():
    """Pic VRAM PyTorch reserve / total (pour reperer la saturation -> spill RAM partagee
    Windows = lenteur extreme, et TDR/'CUDA unknown error'). Ne voit PAS la VRAM des
    autres process (ComfyUI, etc.) -> utiliser nvidia-smi pour le total reel."""
    if DEVICE != "cuda":
        return ""
    try:
        resv = torch.cuda.memory_reserved() / 1024**3
        tot = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f" | VRAM {resv:.1f}/{tot:.0f} Go"
    except Exception:
        return ""


# ----------------------------------------------------------------------------
# Diagnostic: empreinte de l'etat PARTAGE entre les pipelines.
#
# Le pipeline ControlNet est monte sur les composants du base (VAE, encodeur texte,
# transformer) et le controlnet recoit en PLUS des references PARTAGEES vers les
# embedders du transformer (from_transformer). Tout ce qui reste modifie cote partage
# apres une passe ControlNet contamine donc les rendus SUIVANTS -- c'est exactement le
# bug ouvert (images sans rapport avec le prompt, sans erreur, jusqu'au redemarrage).
#
# Ces empreintes servent a le prendre sur le fait: on photographie l'etat partage
# avant/apres la passe et on ne journalise QUE ce qui a bouge. Niveau debug uniquement
# (--log-level debug): la sonde fonctionnelle fait une passe d'encodeur de texte.
# ----------------------------------------------------------------------------
_STATE_SNAP = {}
# Phrase temoin: le MEME texte doit toujours donner le MEME embedding. S'il change
# apres une passe ControlNet, l'encodeur de texte (partage) est abime en VRAM -> le
# modele genere alors une image coherente mais qui ignore le prompt (symptome observe).
_STATE_PROBE_PROMPT = "a red cube on a white table"
_STATE_PARAM_SAMPLE = 6


def _module_sig(mod, sample=_STATE_PARAM_SAMPLE):
    """Signature bon marche d'un nn.Module: device/dtype + somme des |poids| sur un
    echantillon FIXE de tenseurs + detection NaN/Inf. Suffisant pour reperer des poids
    abimes en VRAM (corruption silencieuse) sans checksummer 12 Go."""
    if mod is None:
        return "<none>"
    try:
        names = [n for n, _ in mod.named_parameters()]
        if not names:
            return f"{type(mod).__name__}<no params>"
        first = next(mod.parameters())
        step = max(1, len(names) // sample)
        picked = names[::step][:sample]
        params = dict(mod.named_parameters())
        parts = []
        for n in picked:
            p = params[n].detach()
            if not p.dtype.is_floating_point:
                parts.append(f"{n}=<{p.dtype}>")
                continue
            s = float(p.abs().sum(dtype=torch.float32))
            bad = "" if torch.isfinite(p).all() else " NaN/Inf!"
            parts.append(f"{n}={s:.6e}{bad}")
        return (f"{type(mod).__name__} dev={first.device} dtype={first.dtype} "
                f"n={len(names)} " + " ".join(parts))
    except Exception as e:
        return f"<sig failed: {e}>"


def _text_embed_sig(pipe):
    """Sonde FONCTIONNELLE: encode une phrase temoin et renvoie une empreinte de
    l'embedding. C'est le test qui tranche entre 'bug de logique' et 'encodeur de texte
    abime': meme phrase -> meme empreinte, toujours."""
    try:
        with torch.no_grad():
            emb, _ = pipe.encode_prompt(_STATE_PROBE_PROMPT,
                                        do_classifier_free_guidance=False)
        t = emb[0] if isinstance(emb, (list, tuple)) else emb
        t = t.detach().float()
        return (f"shape={tuple(t.shape)} sum={float(t.sum()):.6e} "
                f"absmean={float(t.abs().mean()):.6e} "
                f"finite={bool(torch.isfinite(t).all())}")
    except Exception as e:
        return f"<probe failed: {e}>"


def _shared_state_snapshot(probe=True):
    """Photographie l'etat partage: identite des objets, poids, scheduler, VAE, rope,
    hooks accelerate, et (si probe) l'embedding de la phrase temoin."""
    snap = {}
    base = _BASE_PIPE
    if base is None:
        snap["pipe"] = "<no base pipeline>"
        return snap
    try:
        snap["pipe:base"] = f"{type(base).__name__}#{id(base)}"
        snap["pipe:derived"] = ",".join(sorted(_DERIVED))
        snap["pipe:loaded_key"] = repr(_LOADED_KEY)
        snap["cn:pipe"] = f"{type(_CN_PIPE).__name__}#{id(_CN_PIPE)}" if _CN_PIPE else "<none>"
        snap["cn:model"] = f"#{id(_CN_MODEL)}" if _CN_MODEL is not None else "<none>"
        for name in ("transformer", "text_encoder", "vae"):
            comp = getattr(base, name, None)
            snap[f"id:{name}"] = f"#{id(comp)}"
            snap[f"sig:{name}"] = _module_sig(comp)
            snap[f"hook:{name}"] = type(getattr(comp, "_hf_hook", None)).__name__
            # Les pipes derives DOIVENT pointer sur les memes objets que le base.
            for kind, p in _DERIVED.items():
                if p is not base:
                    snap[f"same:{kind}.{name}"] = getattr(p, name, None) is comp
        tr = getattr(base, "transformer", None)
        # x_pad_token / cap_pad_token: minuscules et PARTAGES avec le controlnet
        # (from_transformer) -> canaris ideaux.
        for tok in ("x_pad_token", "cap_pad_token"):
            t = getattr(tr, tok, None)
            snap[f"tok:{tok}"] = ("<none>" if t is None else
                                  f"{tuple(t.shape)} {t.dtype} {t.device} "
                                  f"sum={float(t.detach().float().sum()):.6e}")
        rope = getattr(tr, "rope_embedder", None)
        f = getattr(rope, "freqs_cis", None)
        snap["rope:id"] = f"#{id(rope)}"
        snap["rope:freqs"] = ("<lazy/None>" if f is None else
                              " ".join(f"{tuple(x.shape)}@{x.device}" for x in f))
        sch = getattr(base, "scheduler", None)
        snap["sched:class"] = type(sch).__name__
        snap["sched:id"] = f"#{id(sch)}"
        snap["sched:config"] = repr(sorted(dict(sch.config).items())) if sch else "<none>"
        vae = getattr(base, "vae", None)
        for flag in ("use_tiling", "use_slicing"):
            snap[f"vae:{flag}"] = repr(getattr(vae, flag, "<missing>"))
        for key in ("force_upcast", "scaling_factor", "shift_factor"):
            snap[f"vae:cfg.{key}"] = repr(getattr(getattr(vae, "config", None), key, "<missing>"))
        snap["globals"] = (f"sampler={SAMPLER}/{SCHEDULE} guidance={GUIDANCE} "
                           f"offload={OFFLOAD_MODE}/{_effective_offload()} "
                           f"loras={_APPLIED_LORAS} cn_tile={CONTROLNET_TILE}")
        if probe:
            snap["probe:text_embed"] = _text_embed_sig(base)
    except Exception as e:
        snap["<snapshot error>"] = repr(e)
    return snap


def _shared_state_diff(label, probe=True):
    """Compare l'etat partage a la photo precedente et journalise CE QUI A BOUGE.
    No-op hors debug. A appeler autour de la passe ControlNet et a l'entree de
    generate() -- la premiere ligne 'CHANGED' designe le coupable."""
    global _STATE_SNAP
    if cz_core.LOG_LEVEL < 2:
        return
    try:
        snap = _shared_state_snapshot(probe=probe)
    except Exception as e:
        _dbg(f"state[{label}]: snapshot failed: {e}")
        return
    prev, _STATE_SNAP = _STATE_SNAP, snap
    if not prev:
        _dbg(f"state[{label}]: baseline ({len(snap)} keys)")
        return
    changed = [k for k in sorted(set(prev) | set(snap))
               if prev.get(k, "<absent>") != snap.get(k, "<absent>")]
    if not changed:
        _dbg(f"state[{label}]: unchanged ({len(snap)} keys)")
        return
    _dbg(f"state[{label}]: {len(changed)} CHANGED key(s) <-- shared state moved here")
    for k in changed:
        _dbg(f"  {k}\n      before: {prev.get(k, '<absent>')}\n      after : {snap.get(k, '<absent>')}")


# ----------------------------------------------------------------------------
# Z-Image (diffusers, BF16) : un pipeline "base" txt2img qui detient les composants,
# img2img / inpaint derives via from_pipe (poids partages, pas de VRAM en double).
# ----------------------------------------------------------------------------
def _lora_names(loras):
    return [f"cz_lora_{i}" for i in range(len(loras))]


def _clear_loras(pipe):
    """Retire TOUT adaptateur LoRA du pipe pour repartir d'un etat vierge.

    unload_lora_weights() seul laisse, selon les versions diffusers/peft, un peft_config
    residuel sur le transformer -> le load suivant avertit ('Already found a peft_config')
    et, comme on reutilise les memes noms d'adaptateurs (cz_lora_i), l'ancien adaptateur
    peut rester en place (mauvaise LoRA appliquee). On supprime donc explicitement les
    adaptateurs restants par nom apres l'unload."""
    try:
        pipe.unload_lora_weights()
    except Exception as e:
        _dbg(f"unload_lora_weights: {e}")
    try:
        listed = pipe.get_list_adapters() or {}
        names = sorted({n for lst in listed.values() for n in (lst or [])})
        if names:
            pipe.delete_adapters(names)
            _dbg(f"cleared leftover LoRA adapters: {names}")
    except Exception as e:
        _dbg(f"delete_adapters: {e}")


def _apply_loras(pipe, force=False):
    """Synchronise les adaptateurs LoRA du pipe avec LORAS, SANS recharger le modele.

    Le transformer reste en VRAM; seuls les adaptateurs PEFT bougent:
      - memes fichiers, poids differents -> set_adapters (immediat)
      - jeu de LoRA different            -> unload_lora_weights + reload des LoRA (~1s)
    Les pipes derives (from_pipe) partagent ce transformer -> ils suivent automatiquement.
    Renvoie True si applique, False si echec (le caller retombe sur un reload complet)."""
    global _APPLIED_LORAS
    if not force and _APPLIED_LORAS == LORAS:
        return True
    old_paths = [p for p, _ in _APPLIED_LORAS]
    new_paths = [p for p, _ in LORAS]
    try:
        if not force and old_paths and old_paths == new_paths:
            # Seuls les poids changent -> re-ponderation instantanee.
            pipe.set_adapters(_lora_names(LORAS), [float(w) for _, w in LORAS])
            _APPLIED_LORAS = list(LORAS)
            _log("LoRA weights updated in place (no reload): "
                 + ", ".join(f"{os.path.basename(p)}@{w}" for p, w in LORAS))
            return True
        if old_paths or force:
            _clear_loras(pipe)
        names, weights = [], []
        for i, (p, w) in enumerate(LORAS):
            if os.path.isfile(p):
                an = f"cz_lora_{i}"
                _log(f"applying LoRA: {os.path.basename(p)} (weight {w})")
                # Passer le dossier + weight_name (et non le chemin complet) : sinon
                # diffusers en mode offline (HF_HUB_OFFLINE) refuse "must specify a
                # weight_name". Marche aussi online et avec un fichier local direct.
                pipe.load_lora_weights(os.path.dirname(p) or ".",
                                       weight_name=os.path.basename(p), adapter_name=an)
                names.append(an)
                weights.append(float(w))
            else:
                _log(f"LoRA file not found, ignored: {p}")
        if names:
            pipe.set_adapters(names, weights)
        _APPLIED_LORAS = list(LORAS)
        if not force:
            _log("LoRAs hot-swapped (no model reload)")
        return True
    except Exception as e:
        _log(f"LoRA hot-swap failed ({e}); falling back to a full reload")
        _APPLIED_LORAS = []
        return False


# Cache disque des transformers dequantifies (FP8/INT8 ComfyUI -> bf16). Un dequant
# lit et convertit tout le fichier: ~5 min pour 5.7 Go sur un HDD. Le resultat bf16 est
# ecrit une fois ici, et les chargements suivants deviennent un simple single-file
# (~40 s). Vide/'auto' = <app>/cache/dequant, "off"/"none" = desactive.
_DQ_CACHE_CFG = str(CONFIG.get("dequant_cache", "auto") or "auto").strip()
try:
    DEQUANT_CACHE_MAX_GB = float(CONFIG.get("dequant_cache_max_gb", 60) or 0)
except Exception:
    DEQUANT_CACHE_MAX_GB = 60.0


def _dequant_cache_dir():
    """Dossier du cache de dequant, cree a la demande. None = cache desactive."""
    if _DQ_CACHE_CFG.lower() in ("off", "none", "0", "false"):
        return None
    d = (os.path.join(HERE, "cache", "dequant")
         if _DQ_CACHE_CFG.lower() in ("auto", "") else _DQ_CACHE_CFG)
    try:
        os.makedirs(d, exist_ok=True)
        return d
    except Exception as e:
        _dbg(f"dequant cache dir unavailable ({e})")
        return None


def _dequant_cache_path(src):
    """Chemin du bf16 cache pour un checkpoint source. La cle inclut taille+mtime:
    un fichier remplace (meme nom) ne reutilise jamais l'ancien cache."""
    d = _dequant_cache_dir()
    if not d:
        return None
    try:
        p, size, mtime = _file_key(src)
    except OSError:
        return None
    h = hashlib.sha1(f"{p.lower()}|{size}|{mtime}|bf16".encode("utf-8")).hexdigest()[:16]
    base = os.path.splitext(os.path.basename(src))[0][:48]
    return os.path.join(d, f"{base}.{h}.safetensors")


def _dequant_cache_prune(keep=None):
    """Plafonne le cache (dequant_cache_max_gb, 0 = illimite): supprime les fichiers
    les moins recemment UTILISES (atime, sinon mtime) jusqu'a repasser sous le seuil."""
    d = _dequant_cache_dir()
    if not d or DEQUANT_CACHE_MAX_GB <= 0:
        return
    try:
        files = []
        for f in os.listdir(d):
            fp = os.path.join(d, f)
            if not f.endswith(".safetensors") or not os.path.isfile(fp):
                continue
            st = os.stat(fp)
            files.append((max(st.st_atime, st.st_mtime), st.st_size, fp))
        total = sum(s for _t, s, _p in files)
        cap = DEQUANT_CACHE_MAX_GB * 1024**3
        for _t, size, fp in sorted(files):          # plus ancien acces d'abord
            if total <= cap:
                break
            if keep and os.path.abspath(fp) == os.path.abspath(keep):
                continue
            try:
                os.remove(fp)
                total -= size
                _log(f"dequant cache: evicted {os.path.basename(fp)} "
                     f"({size / 1024**3:.1f} GB, over the {DEQUANT_CACHE_MAX_GB:.0f} GB cap)")
            except OSError as e:
                _dbg(f"dequant cache evict failed {fp}: {e}")
    except Exception as e:
        _dbg(f"dequant cache prune failed: {e}")


def _dequant_cache_store(src, sd):
    """Ecrit le state dict dequantifie dans le cache (best effort: toute erreur est
    ignoree, le chargement courant a deja le dict en memoire). Ecriture atomique via
    un .tmp renomme -> une interruption ne laisse jamais un cache tronque."""
    dst = _dequant_cache_path(src)
    if not dst:
        return
    try:
        from safetensors.torch import save_file
        t0 = time.time()
        tmp = dst + ".tmp"
        # contiguous(): safetensors refuse les vues non contigues (issues des slices
        # de dequant); clone implicite, on est deja en RAM.
        save_file({k: v.contiguous() for k, v in sd.items()}, tmp)
        os.replace(tmp, dst)
        gb = os.path.getsize(dst) / 1024**3
        _log(f"dequant cache: saved {gb:.1f} GB in {time.time() - t0:.1f}s "
             f"-> next load of this checkpoint skips the dequant")
        _dequant_cache_prune(keep=dst)
    except Exception as e:
        _log(f"dequant cache: not saved ({e})")
        try:
            os.remove(dst + ".tmp")
        except OSError:
            pass


def _hadamard_ortho(n):
    """Matrice 'regular hadamard' du ConvRot comfy-quants -- ATTENTION, ce n'est PAS
    la construction de Sylvester: la base est ce H4 precis, etendu par produits de
    Kronecker jusqu'a n (puissance de 4), puis normalise 1/sqrt(n). Orthonormee ET
    symetrique -> la reconstruction re-multiplie simplement par la meme matrice.
    (Verifie contre src/comfy_quants/formats/convrot.py; avec un Sylvester la
    correlation aux poids de base tombe a ~0 -> bruit total.)"""
    h4 = torch.tensor([[1., 1., 1., -1.], [1., 1., -1., 1.],
                       [1., -1., 1., 1.], [-1., 1., 1., 1.]])
    H = h4
    while H.shape[0] < n:
        H = torch.kron(H, h4)
    if H.shape[0] != n:
        raise ValueError(f"convrot groupsize {n} is not a power of 4")
    return H / (float(n) ** 0.5)


def _load_dequant_state_dict(path):
    """Charge en RAM un checkpoint 'scaled' ComfyUI (FP8/INT8) et le dequantifie en
    DTYPE (bf16), tenseur par tenseur:
      - bundle AIO (transformer + encodeur texte + VAE): seules les cles
        'model.diffusion_model.*' sont gardees (VAE + encodeur Qwen3 = repo de base);
      - X.weight (F8/I8) * X.weight_scale (scalaire ou par ligne) -> bf16;
      - blob X.comfy_quant: si 'convrot' est declare (int8_tensorwise ComfyUI), la
        rotation de Hadamard par groupes (defaut 256) est DEFAITE apres le descale --
        sans ca les poids sont un bruit total (observe sur redzit222026HD);
      - les cles de quantification (weight_scale/scale_weight, comfy_quant, marqueur
        scaled_fp8) sont consommees/jetees.
    Le dict resultant part dans from_single_file (conversion de cles diffusers comprise).
    NB VRAM/RAM: dequantifie = empreinte d'un BF16 complet (~12 Go); le FP8 n'economise
    que le disque/telechargement, pas la memoire."""
    from safetensors import safe_open
    t0 = time.time()
    hdr = _safetensors_header(path)
    entries = [(k, v) for k, v in hdr.items()
               if k != "__metadata__" and isinstance(v, dict)]
    # Bundle AIO: ne garder que le transformer. (Sans prefixe ComfyUI = fichier
    # transformer-only au layout original -> pas de filtre.)
    if any(k.startswith("model.diffusion_model.") for k, _ in entries):
        entries = [(k, v) for k, v in entries
                   if k.startswith("model.diffusion_model.")]
    # Garde d'architecture: un checkpoint quantifie d'un AUTRE modele (cles sans
    # aucun marqueur Z-Image) chargerait des poids incoherents -> refus clair.
    if not any((".feed_forward." in k or "noise_refiner" in k or
                "context_refiner" in k or "cap_embedder" in k) for k, _ in entries):
        raise RuntimeError(
            f"{os.path.basename(path)}: quantized checkpoint does not look like a "
            "Z-Image transformer (different architecture); this build only loads "
            "Z-Image models.")
    # Lecture SEQUENTIELLE dans l'ordre PHYSIQUE du fichier (data_offsets): un HDD
    # s'effondre en acces aleatoire, et l'ordre des cles ne suit pas celui des donnees
    # (mesure sur un FP8 de 5.7 Go: 349s en ordre de cles -> lie au debit disque ainsi).
    entries.sort(key=lambda kv: kv[1].get("data_offsets", [0])[0])
    raw = {}
    qcfg = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k, _ in entries:
            if k.endswith(".comfy_quant"):   # blob JSON: format + convrot eventuels
                try:
                    qcfg[k[:-len(".comfy_quant")]] = json.loads(
                        bytes(f.get_tensor(k).tolist()).decode("utf-8"))
                except Exception as e:
                    _dbg(f"comfy_quant blob unreadable {k}: {e}")
                continue
            raw[k] = f.get_tensor(k)
    _had = {}                                # cache Hadamard par taille de groupe
    sd = {}
    n_dq = n_rot = 0
    for k in list(raw.keys()):
        if (k.endswith((".weight_scale", ".scale_weight", ".scale_input", ".input_scale"))
                or k.endswith("scaled_fp8")):
            continue                         # consommees via lookup / jetees (scale_input
                                             # = echelle d'ACTIVATION, pas de poids)
        t = raw.pop(k)
        if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2,
                       torch.int8, torch.uint8):
            s = None
            for cand in (k + "_scale",       # X.weight -> X.weight_scale (ComfyUI)
                         (k[:-len(".weight")] + ".scale_weight")
                         if k.endswith(".weight") else None):
                if cand and cand in raw:
                    s = raw[cand]
                    break
            t = t.to(torch.float32)
            if s is not None:                # scalaire ou [out,1] -> broadcast
                t = t * s.to(torch.float32)
            # ConvRot (int8_tensorwise comfy-quants): les poids stockes ont ete tournes
            # W_rot = (W.view(out, in/g, g) @ H.T).reshape(...) AVANT quantification ->
            # reconstruction = re-multiplier par H (orthonormee, symetrique) par groupe.
            cfg = qcfg.get(k[:-len(".weight")]) if k.endswith(".weight") else None
            if cfg and cfg.get("convrot"):
                g = int(cfg.get("convrot_groupsize", 256) or 256)
                if t.dim() == 2 and g > 1 and t.shape[1] % g == 0:
                    if g not in _had:
                        _had[g] = _hadamard_ortho(g)
                    t = (t.view(t.shape[0], -1, g) @ _had[g]).reshape(t.shape[0], -1)
                    n_rot += 1
            t = t.to(DTYPE)
            n_dq += 1
        elif t.is_floating_point() and t.dtype != DTYPE:
            t = t.to(DTYPE)
        sd[k] = t
    raw.clear()
    _log(f"dequantized {n_dq} tensors ({len(sd)} kept"
         + (f", {n_rot} un-rotated (ConvRot)" if n_rot else "")
         + f") to bf16 in {time.time() - t0:.1f}s")
    return sd


def _load_transformer():
    """Charge UNIQUEMENT le transformer courant (sans le reste du pipeline):
      - override GGUF quantifie (.gguf)   -> from_single_file + GGUFQuantizationConfig
      - override FP8/INT8 'scaled' ComfyUI -> dequant en RAM puis from_single_file(dict)
      - override single-file (.safetensors Civitai) -> from_single_file
      - override repo HF / dossier diffusers        -> sous-dossier 'transformer'
      - pas d'override                              -> transformer du repo de base
    Utilise au chargement complet ET pour l'echange a chaud (_swap_transformer)."""
    from diffusers import ZImageTransformer2DModel
    if ZIMAGE_TRANSFORMER:
        if _is_single_file(ZIMAGE_TRANSFORMER):
            # Garde: un fichier non chargeable (LoRA egaree, SVDQuant) selectionne via
            # config/CLI/prefs doit echouer avec un message actionnable, pas partir
            # chercher une config SD1.5 par defaut sur le Hub. (Sans effet sur les
            # .gguf: header safetensors illisible -> None.)
            bad = _safetensors_unsupported(ZIMAGE_TRANSFORMER)
            if bad:
                raise RuntimeError(f"{os.path.basename(ZIMAGE_TRANSFORMER)}: {bad}.")
            if _is_gguf_path(ZIMAGE_TRANSFORMER):
                # transformer Z-Image GGUF (quantifie) -> reste quantifie en memoire
                # (vraie economie de VRAM). VAE + encodeur texte = repo de base (cache).
                lay = _gguf_layout_unsupported(ZIMAGE_TRANSFORMER)
                if lay:
                    raise RuntimeError(
                        f"{os.path.basename(ZIMAGE_TRANSFORMER)}: {lay}.")
                from diffusers import GGUFQuantizationConfig
                _log(f"loading Z-Image transformer (GGUF, quantized): "
                     f"{ZIMAGE_TRANSFORMER} ...")
                # config/subfolder = structure du transformer depuis le repo de base
                # (cache), sinon from_single_file tente un repo par defaut.
                return _load_monitor(
                    f"transformer {os.path.basename(ZIMAGE_TRANSFORMER)} (GGUF)",
                    lambda: ZImageTransformer2DModel.from_single_file(
                        ZIMAGE_TRANSFORMER,
                        quantization_config=GGUFQuantizationConfig(compute_dtype=DTYPE),
                        config=BASE_REPO, subfolder="transformer",
                        torch_dtype=DTYPE))
            dq = _safetensors_dequant(ZIMAGE_TRANSFORMER)
            if dq:
                # Deja dequantifie une fois ? -> relire le bf16 du cache disque, c'est
                # un single-file normal (secondes) au lieu de re-convertir tout le
                # fichier (minutes sur HDD).
                cached = _dequant_cache_path(ZIMAGE_TRANSFORMER)
                if cached and os.path.isfile(cached):
                    _log(f"loading Z-Image transformer ({dq} -> bf16, from dequant "
                         f"cache): {os.path.basename(cached)}")
                    try:
                        os.utime(cached, None)       # marque l'usage pour le LRU
                    except OSError:
                        pass
                    return _load_monitor(
                        f"transformer {os.path.basename(ZIMAGE_TRANSFORMER)} (cached bf16)",
                        lambda: ZImageTransformer2DModel.from_single_file(
                            cached, config=BASE_REPO, subfolder="transformer",
                            torch_dtype=DTYPE))
                # FP8/INT8 'scaled' ComfyUI (majorite des builds Civitai legers) ->
                # dequant en RAM puis chargement du dict (conversion de cles diffusers
                # incluse: prefixe ComfyUI, split QKV fusionne...).
                _log(f"loading Z-Image transformer (single-file, {dq} ComfyUI -> "
                     f"dequantized to bf16): {ZIMAGE_TRANSFORMER} ...")
                sd = _load_dequant_state_dict(ZIMAGE_TRANSFORMER)
                _dequant_cache_store(ZIMAGE_TRANSFORMER, sd)
                return _load_monitor(
                    f"transformer {os.path.basename(ZIMAGE_TRANSFORMER)} ({dq})",
                    lambda: ZImageTransformer2DModel.from_single_file(
                        sd, config=BASE_REPO, subfolder="transformer",
                        torch_dtype=DTYPE))
            _log(f"loading Z-Image transformer (single-file): {ZIMAGE_TRANSFORMER} ...")
            # config/subfolder = structure du transformer depuis le repo de base (cache):
            # sans ca, un checkpoint non reconnu fait retomber from_single_file sur son
            # repo par defaut (SD1.5) -> 404, et le mode offline echoue.
            return _load_monitor(
                f"transformer {os.path.basename(ZIMAGE_TRANSFORMER)}",
                lambda: ZImageTransformer2DModel.from_single_file(
                    ZIMAGE_TRANSFORMER, config=BASE_REPO, subfolder="transformer",
                    torch_dtype=DTYPE))
        # repo HF / dossier diffusers -> charge le sous-dossier 'transformer'
        # (utile pour les modeles comme Juggernaut-Z dont le tokenizer est
        # incomplet: on garde VAE + encodeur + tokenizer du repo de base).
        _log(f"loading Z-Image transformer (repo subfolder): {ZIMAGE_TRANSFORMER} ...")
        return _load_monitor(
            f"transformer {ZIMAGE_TRANSFORMER}",
            lambda: ZImageTransformer2DModel.from_pretrained(
                ZIMAGE_TRANSFORMER, subfolder="transformer", torch_dtype=DTYPE))
    _log(f"loading Z-Image transformer (base repo): {BASE_REPO} ...")
    return _load_monitor(
        f"transformer {BASE_REPO}",
        lambda: ZImageTransformer2DModel.from_pretrained(
            BASE_REPO, subfolder="transformer", torch_dtype=DTYPE))


_CURRENT_TRANSFORMER = object()   # sentinelle: "prends le transformer courant"


def _effective_offload(tpath=_CURRENT_TRANSFORMER):
    """Offload REELLEMENT applique. Un transformer GGUF quantifie ne se deplace pas sur le
    GPU via .to(cuda) ni en sequential -> seul enable_model_cpu_offload le pose sur le GPU
    pendant le forward. On force donc 'model' pour un base GGUF, quel que soit le reglage.

    ATTENTION: la sentinelle n'est PAS None. None est une valeur LEGITIME de tpath (= pas
    d'override, on tourne sur le transformer du repo de base). Avec None comme sentinelle,
    _effective_offload(None) retombait sur ZIMAGE_TRANSFORMER, c'est-a-dire le NOUVEAU
    transformer: le garde-fou de _swap_transformer comparait alors le nouveau a lui-meme
    et laissait passer un echange a chaud repo de base -> GGUF, qui change pourtant
    l'offload effectif ('none' -> 'model')."""
    off = OFFLOAD_MODE
    t = ZIMAGE_TRANSFORMER if tpath is _CURRENT_TRANSFORMER else tpath
    if DEVICE == "cuda" and _is_gguf_path(t) and off != "model":
        off = "model"
    return off


def _swap_transformer(pipe):
    """Remplace SEULEMENT le transformer du pipeline deja en cache: le VAE, l'encodeur
    de texte Qwen3-4B, le tokenizer et le scheduler restent en VRAM (c'est eux le gros
    du temps de chargement). Valable uniquement a repo de base + offload identiques.

    Renvoie True si l'echange a reussi, False -> le caller fait un reload complet."""
    global _APPLIED_LORAS, _DERIVED
    t0 = time.time()
    old_path = _LOADED_KEY[1] if _LOADED_KEY else None
    # Passer de/vers un GGUF change l'offload EFFECTIF (un GGUF impose 'model') -> les
    # hooks accelerate et le placement different: on ne bricole pas, on recharge.
    if _effective_offload(old_path) != _effective_offload(ZIMAGE_TRANSFORMER):
        _log("transformer swap skipped (GGUF changes the effective offload) -> full reload")
        return False
    try:
        _log(f"switching Z-Image transformer -> {ZIMAGE_TRANSFORMER or BASE_REPO} "
             "(keeping VAE + text encoder in VRAM)")
        new_t = _load_transformer()
        old = getattr(pipe, "transformer", None)
        off = _effective_offload()
        # Offload: les hooks accelerate sont poses sur les composants. Il faut les retirer
        # avant l'echange, sinon le nouveau transformer n'en a pas et l'ancien garde les siens.
        if DEVICE == "cuda" and off in ("model", "sequential"):
            try:
                pipe.remove_all_hooks()
            except Exception as e:
                _dbg(f"remove_all_hooks: {e}")
        try:
            pipe.register_modules(transformer=new_t)   # API diffusers (met a jour le config)
        except Exception:
            pipe.transformer = new_t
        # Liberer l'ANCIEN transformer AVANT de poser le nouveau sur le GPU: sinon
        # ancien (12 Go) + nouveau (12 Go) + VAE/encodeur (~7 Go) depassent la VRAM
        # -> spill en RAM partagee qui ne se resorbe pas (mesure sur une grille XYZ
        # multi-checkpoints: 1.7 s/step -> 300-600 s/step, puis crash). Les pipes
        # derives (from_pipe) pointent aussi sur l'ancien -> a purger d'abord, sinon
        # `del old` ne libere rien (from_pipe est gratuit, il sera reconstruit).
        # Le pipeline ControlNet AUSSI: il tient le transformer, et from_transformer lui a
        # greffe des references vers ses embedders. L'oublier ici gardait l'ANCIEN
        # transformer (12 Go) vivant malgre `del old` -> exactement le debordement en RAM
        # partagee que ce bloc cherche a eviter, et un ControlNet lie a un transformer
        # qui n'est plus celui du pipeline.
        _free_controlnet()
        _DERIVED = {}
        del old
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        if DEVICE == "cuda":
            if off == "model":
                pipe.enable_model_cpu_offload()
            elif off == "sequential":
                pipe.enable_sequential_cpu_offload()
            else:
                new_t.to(DEVICE)
        # Les adaptateurs LoRA etaient poses sur l'ancien transformer -> a reposer.
        _APPLIED_LORAS = []
        if LORAS:
            _apply_loras(pipe, force=True)
        _log(f"transformer switched in {time.time() - t0:.1f}s "
             "(VAE + text encoder kept, no full reload)")
        return True
    except Exception as e:
        _log(f"transformer hot-swap failed ({e}); falling back to a full reload")
        _APPLIED_LORAS = []
        return False


def _ensure_base():
    """Charge (si besoin) le pipeline de base txt2img. Gere le transformer
    single-file (Civitai) et l'offload. Cache par (repo, transformer, offload).

    Deux echanges a chaud evitent un rechargement complet (transformer + VAE + encodeur
    Qwen3-4B, des dizaines de secondes):
      - LoRA differentes            -> _apply_loras (adaptateurs PEFT seuls)
      - transformer different, meme repo de base + offload -> _swap_transformer
        (on ne recharge QUE le transformer; VAE/encodeur/tokenizer restent en VRAM)."""
    global _BASE_PIPE, _DERIVED, _LOADED_KEY, _BASE_SCHED_CONFIG, _APPLIED_LORAS
    key = (BASE_REPO, ZIMAGE_TRANSFORMER, OFFLOAD_MODE)
    _dbg(f"_ensure_base key={key} cached={_LOADED_KEY}")
    if _BASE_PIPE is not None and _LOADED_KEY == key:
        if _apply_loras(_BASE_PIPE):
            _dbg("base pipeline: reusing cached (no reload)")
            return _BASE_PIPE
        _dbg("base pipeline: LoRA hot-swap failed -> free + reload")
        free_vram()
    elif _BASE_PIPE is not None:
        # Seul le transformer change (meme repo de base + meme offload) ? -> on ne recharge
        # QUE le transformer et on garde VAE + encodeur Qwen3 + tokenizer en VRAM.
        if (_LOADED_KEY and _LOADED_KEY[0] == BASE_REPO and _LOADED_KEY[2] == OFFLOAD_MODE
                and _swap_transformer(_BASE_PIPE)):
            _LOADED_KEY = key
            return _BASE_PIPE
        _dbg("base pipeline: key changed -> free + reload")
        free_vram()
    from diffusers import ZImagePipeline
    t0 = time.time()
    # Garde: un autre process qui squatte la VRAM fait deborder le chargement en RAM
    # partagee sans aucune erreur -> on previent AVANT de payer plusieurs minutes.
    _busy = gpu_busy_warning()
    if _busy:
        _log(f"WARNING: {_busy}")
    kwargs = {}
    if ZIMAGE_TRANSFORMER:
        kwargs["transformer"] = _load_transformer()
    _log(f"loading Z-Image base: {BASE_REPO} (offload={OFFLOAD_MODE}, dtype=bf16) ... "
         "first time downloads from HF, then cached")
    pipe = _load_monitor(f"Z-Image base {BASE_REPO}",
                         lambda: ZImagePipeline.from_pretrained(BASE_REPO, torch_dtype=DTYPE, **kwargs))
    # Capture le config natif (flow-matching) du scheduler -> base pour construire les
    # autres samplers (euler/dpm2a/dpmpp2m) sans perdre shift/flow params.
    try:
        _BASE_SCHED_CONFIG = dict(pipe.scheduler.config)
    except Exception:
        _BASE_SCHED_CONFIG = None
    # LoRA Z-Image (sur le transformer du base -> partage par les pipes derives).
    # force=True: pipe neuf, aucun adaptateur pose -> on (re)pose tout.
    _APPLIED_LORAS = []
    if LORAS:
        _apply_loras(pipe, force=True)
    # Attention slicing: POSE PAR APPEL via _set_slicing (selon la resolution traitee),
    # PAS au chargement. En tuile/1024 -> slicing OFF = attention SDPA native, rapide
    # (comme ComfyUI). Whole-image 2K+ -> slicing ON pour eviter le spill VRAM 32 Go.
    # enable_*_cpu_offload gere lui-meme le device -> ne PAS faire .to(cuda) alors.
    # IMPORTANT: un transformer GGUF quantifie ne se deplace PAS sur le GPU via .to(cuda)
    # (offload=none) ni en sequential -> il reste sur CPU = ULTRA lent (VRAM vide,
    # ~500s/step). Seul enable_model_cpu_offload (accelerate) le pose correctement sur le
    # GPU pendant le forward -> _effective_offload force 'model' pour un base GGUF.
    _off = _effective_offload()
    if _off != OFFLOAD_MODE:
        _log(f"GGUF base: offload '{OFFLOAD_MODE}' forced to '{_off}' (a GGUF does not "
             f"run on GPU with none/sequential -> would stay on CPU, ~500s/step)")
    if DEVICE == "cuda" and _off == "model":
        pipe.enable_model_cpu_offload()
    elif DEVICE == "cuda" and _off == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe = pipe.to(DEVICE)
    # VAE tiling/slicing: indispensable pour l'img2img/upscale. L'encode/decode VAE d'une
    # tuile 1024 + le modele complet en VRAM (transformer + encodeur Qwen3-4B ~8 Go) fait
    # deborder les 32 Go -> spill RAM partagee -> ~300s/step. Tuiler le VAE plafonne ce pic
    # (comme le "tiled decode" de ComfyUI). Le VAE est partage par les pipes derives.
    try:
        pipe.vae.config.force_upcast = False   # VAE en bf16 (fp32 lent sur Blackwell) -- TOUJOURS
    except Exception:
        pass
    try:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    except Exception as e:
        _dbg(f"VAE tiling not available: {e}")
    _apply_sampler(pipe)   # pose le sampler choisi (euler par defaut) sur le pipe de base
    _BASE_PIPE = pipe
    _DERIVED = {"txt2img": pipe}
    _LOADED_KEY = key
    _log(f"Z-Image base ready in {time.time() - t0:.1f}s (sampler={SAMPLER}/{SCHEDULE})")
    return pipe


def get_pipe(kind="img2img"):
    """Renvoie le pipeline demande. txt2img/img2img/inpaint derivent du base via
    from_pipe (poids partages). Omni a besoin de composants en plus (SigLIP) ->
    charge separement depuis un modele Omni dedie (CONFIG['zimage_omni_model'])."""
    base = _ensure_base()
    if kind in _DERIVED:
        _dbg(f"get_pipe('{kind}'): reuse derived")
        return _DERIVED[kind]
    if kind == "omni":
        return _load_omni()
    from diffusers import ZImageImg2ImgPipeline, ZImageInpaintPipeline
    cls = {"img2img": ZImageImg2ImgPipeline, "inpaint": ZImageInpaintPipeline}.get(kind)
    if cls is None:
        return base
    _log(f"deriving {kind} pipeline (shared weights, no extra VRAM)")
    # BUG diffusers: ZImage*Pipeline.from_pipe() UPCASTE tout le pipe (transformer + VAE)
    # en float32. Sur Blackwell (5090: pas de tensor cores fp32) l'img2img/inpaint devient
    # 100-300x plus lent que txt2img (transformer 0.5s -> 108s, mesure). On force bf16 a la
    # derivation, on recaste (composants partages avec le base), on coupe le re-upcast fp32
    # du VAE, et on vide le cache (les copies fp32 transitoires reservaient ~49 Go -> spill).
    # Un transformer GGUF est QUANTIFIE: pas de recast dtype (.to(DTYPE) leve "Casting a
    # quantized model is unsupported") -> torch_dtype=None explicite et pas de p.to(DTYPE)
    # (le compute_dtype est deja bf16).
    quantized = _is_gguf_path(ZIMAGE_TRANSFORMER)
    try:
        p = cls.from_pipe(base, torch_dtype=None) if quantized else cls.from_pipe(base, torch_dtype=DTYPE)
    except TypeError:
        p = cls.from_pipe(base)
    try:
        if not quantized:
            p = p.to(DTYPE)
        p.vae.config.force_upcast = False
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        _log(f"img2img bf16 recast failed ({e})")
    _apply_sampler(p)   # meme sampler que le base (au cas ou from_pipe recree le scheduler)
    # from_pipe recaste les composants PARTAGES (torch_dtype vaut float32 par defaut chez
    # diffusers) -> premier suspect du bug ouvert. On regarde ce que la derivation a
    # change sur le base, juste apres l'avoir faite.
    _shared_state_diff(f"after deriving '{kind}' from base")
    # Diagnostic vitesse: si le pipe derive n'est PAS sur cuda -> img2img/refine tourne
    # sur CPU = ultra lent. On le force sur DEVICE en mode plein VRAM (offload gere seul).
    # NB: offload EFFECTIF (un base GGUF force 'model' meme si l'UI dit 'none'): en
    # offload, un transformer "sur CPU" est normal -> un .to(cuda) casserait les hooks.
    try:
        tdev = next(p.transformer.parameters()).device
        if DEVICE == "cuda" and _effective_offload() == "none" and tdev.type != "cuda":
            _log(f"{kind} pipeline was on {tdev} -> moving to {DEVICE}")
            p = p.to(DEVICE)
            tdev = next(p.transformer.parameters()).device
        _log(f"{kind} pipeline ready: transformer={tdev}")
    except Exception as e:
        _dbg(f"device check failed: {e}")
    _DERIVED[kind] = p
    return p


def _load_omni():
    """Charge le pipeline Omni (multi-reference). Necessite un modele Z-Image
    Omni/Edit (avec encodeur SigLIP) -> CONFIG['zimage_omni_model'] ou env
    ZIMAGE_OMNI_MODEL. Pipeline separe (ne partage pas avec le base)."""
    global _DERIVED
    from diffusers import ZImageOmniPipeline
    repo = (OMNI_MODEL or os.environ.get("ZIMAGE_OMNI_MODEL")
            or CONFIG.get("zimage_omni_model") or "").strip()
    if not repo:
        raise RuntimeError(
            "Omni needs a dedicated Z-Image Omni/Edit model (with a SigLIP encoder that "
            "the Turbo/Base text-to-image models do not ship). As of now Tongyi has only "
            "released Z-Image-Turbo and Z-Image-Base; 'Z-Image-Omni-Base' and 'Z-Image-Edit' "
            "are still 'coming soon'. Once published, set 'zimage_omni_model' in config.txt "
            "to its HF repo id (likely 'Tongyi-MAI/Z-Image-Omni-Base' or 'Tongyi-MAI/"
            "Z-Image-Edit') or a local diffusers folder.")
    _log(f"loading Z-Image Omni: {repo} (offload={OFFLOAD_MODE}) ...")
    t0 = time.time()
    pipe = _load_monitor(f"Z-Image Omni {repo}",
                         lambda: ZImageOmniPipeline.from_pretrained(repo, torch_dtype=DTYPE))
    # Attention slicing pose par appel via _set_slicing (cf. _ensure_base).
    if DEVICE == "cuda" and OFFLOAD_MODE == "model":
        pipe.enable_model_cpu_offload()
    elif DEVICE == "cuda" and OFFLOAD_MODE == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe = pipe.to(DEVICE)
    _DERIVED["omni"] = pipe
    _log(f"Z-Image Omni ready in {time.time() - t0:.1f}s")
    return pipe


@_gpu_serial
def generate_omni(refs, prompt, negative, width, height, steps, seed):
    """Omni multi-reference: compose une image a partir de plusieurs images de
    reference + un prompt (ex. personne + vetement). ZImageOmniPipeline natif."""
    refs = [r for r in (refs or []) if r is not None]
    if not refs:
        raise ValueError("Omni needs at least one reference image.")
    pipe = get_pipe("omni")
    w = round_to_multiple(int(width))
    h = round_to_multiple(int(height))
    _log(f"omni: {len(refs)} ref(s) -> {w}x{h}, {int(steps)} steps, guidance {GUIDANCE:.1f} ...")
    _progress(0.1, f"Omni compose ({len(refs)} refs)...")
    _set_slicing(pipe, max(w, h))
    t0 = time.time()
    out = pipe(
        image=[r.convert("RGB") for r in refs],
        prompt=prompt or "",
        negative_prompt=(negative or None),
        width=w, height=h,
        num_inference_steps=int(steps),
        guidance_scale=GUIDANCE,
        generator=_make_generator(seed),
    ).images[0]
    _log(f"omni done in {time.time() - t0:.1f}s")
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


def load_pipe():
    """Compat: pipeline img2img (etage de raffinement)."""
    return get_pipe("img2img")


@_gpu_serial
def generate(prompt, width, height, steps, seed, negative_prompt=""):
    """txt2img Z-Image: genere une image depuis un prompt.
    Turbo -> GUIDANCE 0. Base -> GUIDANCE ~3.5-5 + plus de steps."""
    # Un ControlNet charge par un upscale precedent n'a rien a faire en VRAM pendant
    # une generation: il ne sert qu'au refine, et ses 6.7 Go font basculer un rendu
    # 1024+ dans la RAM partagee -> latents corrompus, sans erreur (cas reel observe:
    # 3 txt2img de suite ruines apres un upscale ControlNet). Il revient au prochain
    # refine (le modele reste en RAM, la remontee GPU est rapide).
    _free_controlnet()
    pipe = get_pipe("txt2img")
    # Rendre le cache d'allocations AVANT la passe, pas seulement apres. Un upscale qui
    # vient de finir laisse l'allocateur torch avec des blocs libres mais RESERVES: mesure
    # reelle juste apres un upscale ControlNet 2048x3072 -> alloc 19.3 Go mais reserved
    # 28.7/32 Go. Sous Windows (WDDM) la VRAM reservee est prise aux autres process et le
    # rendu suivant part avec ~3 Go de marge: c'est exactement le regime ou le pilote
    # deborde en RAM partagee et ou les latents sortent corrompus SANS erreur.
    # _refine_tiled fait deja ce menage avant sa boucle; generate ne le faisait qu'apres.
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        _dbg(f"txt2img start{_vram_str()}")
    # Bug ouvert (rendus corrompus a partir du 2e rendu): photographie l'etat partage a
    # CHAQUE txt2img. Si un rendu sort corrompu, le diff du rendu precedent dit ce qui a
    # bouge entre les deux. Niveau debug uniquement.
    _shared_state_diff("txt2img entry")
    w = round_to_multiple(int(width))
    h = round_to_multiple(int(height))
    _log(f"txt2img: {w}x{h}, {int(steps)} steps, guidance {GUIDANCE:.1f} ...")
    _dbg(f"txt2img seed={seed} dtype=bf16 device={DEVICE} offload={OFFLOAD_MODE} "
         f"transformer={'single-file' if ZIMAGE_TRANSFORMER else 'repo'}")
    if DEVICE == "cuda":
        _dbg(f"VRAM before: alloc={torch.cuda.memory_allocated()/1024**3:.2f} Go")
    _progress(0.1, f"Generating {w}x{h} ({int(steps)} steps)...")
    _set_slicing(pipe, max(w, h))
    t0 = time.time()
    img = pipe(
        prompt=prompt or "",
        negative_prompt=(negative_prompt or None),
        width=w, height=h,
        num_inference_steps=int(steps),
        guidance_scale=GUIDANCE,
        generator=_make_generator(seed),
    ).images[0]
    _log(f"txt2img done in {time.time() - t0:.1f}s")
    if DEVICE == "cuda":
        _dbg(f"VRAM peak: alloc={torch.cuda.max_memory_allocated()/1024**3:.2f} Go | "
             f"reserved={torch.cuda.max_memory_reserved()/1024**3:.2f} Go")
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return img


def round_to_multiple(x, m=32):
    """Alignement des dimensions. Defaut 32: le transformer Z-Image patchifie par 2 le
    latent VAE -> toute dimension pixel doit etre multiple de 32, sinon mismatch de
    tenseurs dans la diffusion (ex. 'size of tensor a (150) must match b (148)')."""
    return max(m, int(round(x / m) * m))


def set_force_ratio(spec):
    """Definit le ratio force pour upscale/img2img: 'W:H' / 'WxH' (ex '13:19', '832x1216')
    ou '' pour desactiver (ratio natif preserve). Pilote par le radio UI."""
    global FORCE_RATIO
    FORCE_RATIO = (spec or "").strip()
    _log(f"force ratio -> {FORCE_RATIO or '(off, ratio natif preserve)'}")


def set_force_ratio_mode(mode):
    """'crop' (recadrage centre) ou 'extend' (outpaint des bandes manquantes)."""
    global FORCE_RATIO_MODE
    FORCE_RATIO_MODE = "extend" if str(mode or "").strip().lower() == "extend" else "crop"
    _log(f"force ratio mode -> {FORCE_RATIO_MODE}")


def _parse_ratio(spec):
    """(w, h) depuis 'W:H', 'WxH', ou un label '832 x 1216 | 13:19'; sinon None."""
    import re
    if not spec:
        return None
    m = re.search(r"(\d+)\s*[:xX×]\s*(\d+)", str(spec))
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    return (a, b) if a > 0 and b > 0 else None


def _crop_to_ratio(image, ratio_w, ratio_h):
    """Recadre (centre) l'image au ratio ratio_w:ratio_h en gardant l'aire maximale."""
    image = image.convert("RGB")
    w, h = image.size
    target = float(ratio_w) / float(ratio_h)
    cur = w / h
    if abs(cur - target) < 1e-3:
        return image
    if cur > target:                       # trop large -> couper les cotes
        nw = max(1, int(round(h * target)))
        x0 = (w - nw) // 2
        return image.crop((x0, 0, x0 + nw, h))
    nh = max(1, int(round(w / target)))    # trop haut -> couper haut/bas
    y0 = (h - nh) // 2
    return image.crop((0, y0, w, y0 + nh))


def _extend_to_ratio(image, ratio_w, ratio_h, prompt, steps, seed):
    """Amene l'image au ratio cible en l'ETENDANT (outpaint) au lieu de recadrer:
    bandes symetriques ajoutees sur l'axe manquant et remplies par Z-Image via
    outpaint_directions -- le centre garde sa pleine resolution (seules les bandes
    sont generees, diffusion bornee a ~1 MP puis recomposition).

    Anti 'effet bande': une passe img2img legere (EXTEND_DENOISE) tourne sur l'image
    etendue, mais SEULES les bandes + une marge de transition feather sont recollees
    depuis cette passe -- le centre original reste PIXEL POUR PIXEL intact (la passe
    harmonise l'exposition/texture aux jointures sans jamais retoucher l'image)."""
    from PIL import ImageDraw, ImageFilter
    image = image.convert("RGB")
    w, h = image.size
    target = float(ratio_w) / float(ratio_h)
    cur = w / h
    if abs(cur - target) < 1e-3:
        return image
    if cur < target:                       # trop etroit -> elargir gauche + droite
        pad = target * h - w
        out = outpaint_directions(image, None, ["left", "right"], prompt, steps, seed,
                                  expand=pad / (2.0 * w))
    else:                                  # trop large -> etendre haut + bas
        pad = w / target - h
        out = outpaint_directions(image, None, ["top", "bottom"], prompt, steps, seed,
                                  expand=pad / (2.0 * h))
    if EXTEND_DENOISE > 0.001:
        _log(f"extend: seam-blend pass (img2img denoise {EXTEND_DENOISE:.2f}, "
             "original centre kept)")
        refined = _refine_whole(get_pipe("img2img"), out, EXTEND_DENOISE,
                                steps, prompt, seed)
        # Masque de recollage: blanc = prendre la passe harmonisee (bandes + marge de
        # transition A CHEVAL sur la jointure), noir = garder l'original. La marge
        # penetre dans l'image d'origine puis est feather -> raccord fondu, centre intact.
        ox, oy = (out.width - w) // 2, (out.height - h) // 2
        m = max(24, int(0.05 * min(out.size)))       # transition ~5% du petit cote
        mx, my = (m if ox > 0 else 0), (m if oy > 0 else 0)   # marge cote jointure SEULEMENT
        mask = Image.new("L", out.size, 255)
        ImageDraw.Draw(mask).rectangle(
            [ox + mx, oy + my, ox + w - mx, oy + h - my], fill=0)
        mask = mask.filter(ImageFilter.GaussianBlur(max(8, m // 3)))
        out = Image.composite(refined, out, mask)
    return out


def _reframe_canvas(image, ratio_w, ratio_h, overlap=8):
    """Place l'image dans un canevas plus grand au ratio cible (expansion sur 1 axe),
    + un masque (blanc = a remplir, noir = a garder, avec un petit overlap)."""
    from PIL import ImageDraw
    image = image.convert("RGB")
    w, h = image.size
    r = ratio_w / ratio_h
    # Alignement sur 32 (patch 2 x VAE 16): evite les erreurs de conv (no engine).
    if w / h < r:  # trop etroit -> elargir
        nw, nh = round_to_multiple(int(round(h * r)), 32), round_to_multiple(h, 32)
    else:          # trop large -> agrandir en hauteur
        nw, nh = round_to_multiple(w, 32), round_to_multiple(int(round(w / r)), 32)
    nw, nh = max(nw, round_to_multiple(w, 32)), max(nh, round_to_multiple(h, 32))
    ox, oy = (nw - w) // 2, (nh - h) // 2
    canvas = Image.new("RGB", (nw, nh), (127, 127, 127))
    canvas.paste(image, (ox, oy))
    mask = Image.new("L", (nw, nh), 255)
    ImageDraw.Draw(mask).rectangle(
        [ox + overlap, oy + overlap, ox + w - overlap, oy + h - overlap], fill=0)
    return canvas, mask, nw, nh


@_gpu_serial
def inpaint_run(background, mask, prompt, steps, denoise, seed):
    """Inpaint: regenere la zone blanche du masque selon le prompt
    (ZImageInpaintPipeline). background + mask = PIL (L: blanc = a changer)."""
    orig = background.convert("RGB")
    full_mask = mask
    # Diffusion bornee a ~1 MP (zone optimale du modele), puis recomposition pleine res.
    bg, work_mask, orig_size = _cap_work_res(orig, mask)
    w, h = bg.size
    pipe = get_pipe("inpaint")
    _log(f"inpaint: work {w}x{h} (orig {orig_size[0]}x{orig_size[1]}), {int(steps)} steps, "
         f"strength {float(denoise):.2f}, guidance {GUIDANCE:.1f} ...")
    _progress(0.1, "Inpainting...")
    _set_slicing(pipe, max(w, h))
    t0 = time.time()
    out = pipe(prompt=prompt or "", image=bg, mask_image=work_mask, strength=float(denoise),
               num_inference_steps=int(steps), guidance_scale=GUIDANCE,
               generator=_make_generator(seed)).images[0]
    # Recompose: hors-masque garde la pleine resolution; jointure fondue (feather).
    out = _composite_back(out, orig, full_mask, orig_size,
                          feather=max(2, int(min(orig_size) * 0.01)))
    _log(f"inpaint done in {time.time() - t0:.1f}s")
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


# Resolution cible "zone optimale" du modele Z-Image (~1 MP, comme les ratios txt2img).
# Le reframe vise ce budget pour ne PAS exploser le nombre de pixels (sortie 2-3 MP qui
# sort de la zone d'entrainement -> lent et qualite degradee).
MODEL_TARGET_PX = 1024 * 1024


def _ratio_canvas(ratio_w, ratio_h, target_px=MODEL_TARGET_PX):
    """Dimensions (multiples de 32) d'un canevas au ratio donne, a ~target_px pixels."""
    r = float(ratio_w) / float(ratio_h)
    nh = (target_px / r) ** 0.5
    nw = nh * r
    return round_to_multiple(int(round(nw)), 32), round_to_multiple(int(round(nh)), 32)


def _cap_work_res(image, mask, max_px=MODEL_TARGET_PX):
    """Borne la resolution de travail pour la diffusion: si image > max_px, renvoie une
    version reduite (multiples de 32) de (image, mask) + la taille d'origine pour
    recomposer ensuite. Evite de faire tourner le modele tres au-dessus de sa zone
    optimale (~1 MP) -> plus rapide et meilleure qualite."""
    w, h = image.size
    if w * h > max_px:
        s = (max_px / (w * h)) ** 0.5
        ww, wh = round_to_multiple(int(w * s), 32), round_to_multiple(int(h * s), 32)
    else:
        ww, wh = round_to_multiple(w, 32), round_to_multiple(h, 32)
    img_w = image.resize((ww, wh), Image.LANCZOS) if (ww, wh) != image.size else image
    msk_w = mask.resize((ww, wh), Image.NEAREST) if mask.size != (ww, wh) else mask
    return img_w, msk_w, (w, h)


def _composite_back(result, original, mask, orig_size, feather=0):
    """Recompose a la resolution d'origine: la zone masquee (blanc) vient de `result`
    (re-agrandi a orig_size), le reste vient de `original` -> le hors-masque garde la
    pleine resolution de l'image de depart. `feather` (px) floute le masque pour fondre
    la jointure (transition progressive original <-> genere, plus de ligne dure)."""
    if result.size != orig_size:
        result = result.resize(orig_size, Image.LANCZOS)
    if original.size != orig_size:
        original = original.resize(orig_size, Image.LANCZOS)
    m = (mask.resize(orig_size, Image.NEAREST) if mask.size != orig_size else mask).convert("L")
    if feather and feather > 0:
        from PIL import ImageFilter
        m = m.filter(ImageFilter.GaussianBlur(float(feather)))
    return Image.composite(result, original.convert("RGB"), m)


def reframe(image, ratio_w, ratio_h, fit, prompt, steps, seed, strength=1.0):
    """Recadre l'image au ratio cible en bornant la sortie a la resolution optimale du
    modele (~1 MP) -> plus d'explosion du nombre de pixels.
      fit='contain' : l'image entiere rentre dans le canevas (sans l'agrandir), les bords
                      ajoutes sont remplis par Z-Image (outpaint).
      fit='cover'   : l'image remplit le canevas au ratio puis est recadree au centre
                      (pas d'outpaint, simple reframe/crop)."""
    from PIL import ImageDraw
    img = image.convert("RGB")
    w, h = img.size
    nw, nh = _ratio_canvas(ratio_w, ratio_h)
    if str(fit).lower() == "cover":
        scale = max(nw / w, nh / h)
        rw2, rh2 = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = img.resize((rw2, rh2), Image.LANCZOS)
        left, top = (rw2 - nw) // 2, (rh2 - nh) // 2
        out = resized.crop((left, top, left + nw, top + nh))
        _log(f"reframe cover: {w}x{h} -> {nw}x{nh} (crop, no fill)")
        return out
    # contain -> on adapte l'original sans l'agrandir, puis on outpaint les bords.
    from PIL import ImageFilter
    scale = min(nw / w, nh / h, 1.0)
    rw2, rh2 = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = img.resize((rw2, rh2), Image.LANCZOS) if (rw2, rh2) != (w, h) else img
    ox, oy = (nw - rw2) // 2, (nh - rh2) // 2
    # Bords = extension floue des couleurs du bord (blurred edge fill, comme l'outpaint)
    # plutot qu'un gris -> continuite d'exposition; transparait si strength < 1.0.
    arr = np.pad(np.array(resized), [[oy, nh - rh2 - oy], [ox, nw - rw2 - ox], [0, 0]],
                 mode="edge")
    canvas = Image.fromarray(np.ascontiguousarray(arr))
    overlap = 8
    mask = Image.new("L", (nw, nh), 255)
    ImageDraw.Draw(mask).rectangle(
        [ox + overlap, oy + overlap, ox + rw2 - overlap, oy + rh2 - overlap], fill=0)
    blur_r = max(8, int(min(nw, nh) * 0.03))
    canvas = Image.composite(canvas.filter(ImageFilter.GaussianBlur(blur_r)), canvas, mask)
    pipe = get_pipe("inpaint")
    _log(f"reframe contain (outpaint): {w}x{h} -> {nw}x{nh}, {int(steps)} steps, "
         f"strength {float(strength):.2f}, guidance {GUIDANCE:.1f} ...")
    _progress(0.1, f"Reframe -> {nw}x{nh}...")
    _set_slicing(pipe, max(nw, nh))
    t0 = time.time()
    out = pipe(prompt=prompt or "", image=canvas, mask_image=mask, strength=float(strength),
               num_inference_steps=int(steps), guidance_scale=GUIDANCE,
               generator=_make_generator(seed)).images[0]
    if out.size != (nw, nh):
        out = out.resize((nw, nh), Image.LANCZOS)
    _log(f"reframe done in {time.time() - t0:.1f}s")
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


@_gpu_serial
def outpaint(image, ratio_w, ratio_h, prompt, steps, seed):
    """Compat (CLI --reframe et appels existants): reframe en mode 'contain' (outpaint),
    borne a la resolution optimale du modele."""
    return reframe(image, ratio_w, ratio_h, "contain", prompt, steps, seed)


def outpaint_directions(image, mask, directions, prompt, steps, seed, strength=1.0, expand=0.3):
    """Outpaint directionnel (facon Fooocus): agrandit l'image dans les directions
    choisies parmi left/right/top/bottom, chacune de `expand` (fraction de la dimension
    d'origine), en repliquant les pixels du bord (mode 'edge'), puis fait remplir les
    bandes ajoutees par Z-Image (ZImageInpaintPipeline). Un `mask` peint (L, blanc = a
    changer) est optionnel: il est conserve dans la zone d'origine et combine avec les
    bandes ajoutees (blanches)."""
    img = np.array(image.convert("RGB"))
    H, W = img.shape[:2]
    m = np.array(mask.convert("L")) if mask is not None else np.zeros((H, W), dtype=np.uint8)
    dirs = set(d.lower() for d in (directions or []))
    if "top" in dirs:
        p = int(H * expand)
        img = np.pad(img, [[p, 0], [0, 0], [0, 0]], mode="edge")
        m = np.pad(m, [[p, 0], [0, 0]], mode="constant", constant_values=255)
    if "bottom" in dirs:
        p = int(H * expand)
        img = np.pad(img, [[0, p], [0, 0], [0, 0]], mode="edge")
        m = np.pad(m, [[0, p], [0, 0]], mode="constant", constant_values=255)
    if "left" in dirs:
        p = int(W * expand)
        img = np.pad(img, [[0, 0], [p, 0], [0, 0]], mode="edge")
        m = np.pad(m, [[0, 0], [p, 0]], mode="constant", constant_values=255)
    if "right" in dirs:
        p = int(W * expand)
        img = np.pad(img, [[0, 0], [0, p], [0, 0]], mode="edge")
        m = np.pad(m, [[0, 0], [0, p]], mode="constant", constant_values=255)
    canvas = Image.fromarray(np.ascontiguousarray(img))
    mask_img = Image.fromarray(np.ascontiguousarray(m))
    full_size = canvas.size
    # Dilate un peu la zone a generer vers l'interieur -> le modele regenere une fine
    # bande de transition qui se raccorde a l'original (evite la jointure franche).
    from PIL import ImageFilter
    k = max(3, (int(min(full_size) * 0.02) // 2) * 2 + 1)
    mask_img = mask_img.filter(ImageFilter.MaxFilter(min(k, 15)))
    # "Blurred edge fill": on remplit la zone a generer avec une version FLOUE de
    # l'extension du bord (memes couleurs/tonalite que l'original) au lieu d'un bord
    # replique net. Avec strength < 1.0 ce flou transparait -> continuite d'exposition
    # (plus de bande plus claire) et le modele ajoute le detail par-dessus.
    blur_r = max(8, int(min(full_size) * 0.03))
    canvas = Image.composite(canvas.filter(ImageFilter.GaussianBlur(blur_r)), canvas, mask_img)
    # Diffusion bornee a ~1 MP (zone optimale), puis recomposition: le centre (image
    # d'origine) garde sa pleine resolution, seuls les bords ajoutes sont generes.
    work_img, work_mask, _ = _cap_work_res(canvas, mask_img)
    w2, h2 = work_img.size
    pipe = get_pipe("inpaint")
    _log(f"outpaint {sorted(dirs)}: {image.size[0]}x{image.size[1]} -> "
         f"{full_size[0]}x{full_size[1]} (work {w2}x{h2}), {int(steps)} steps, "
         f"guidance {GUIDANCE:.1f} ...")
    _progress(0.1, f"Outpaint -> {full_size[0]}x{full_size[1]}...")
    _set_slicing(pipe, max(w2, h2))
    t0 = time.time()
    out = pipe(prompt=prompt or "", image=work_img, mask_image=work_mask,
               strength=float(strength),
               num_inference_steps=int(steps), guidance_scale=GUIDANCE,
               generator=_make_generator(seed)).images[0]
    out = _composite_back(out, canvas, mask_img, full_size,
                          feather=max(4, int(min(full_size) * 0.015)))
    _log(f"outpaint done in {time.time() - t0:.1f}s")
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


def _make_generator(seed):
    return torch.Generator(DEVICE).manual_seed(int(seed)) if int(seed) >= 0 else None


@_gpu_serial
def _refine_whole(pipe, image, denoise, steps, prompt, seed):
    """Passe Z-Image img2img sur l'image entiere (ou une tuile). Le slicing est pose
    selon la taille reelle traitee: tuile 1024 -> OFF (rapide), whole 2K+ -> ON.
    L'entree est ALIGNEE /32 (resize) avant diffusion — le transformer patchifie le
    latent par 2, une dimension non /32 provoque un mismatch de tenseurs (150 vs 148) —
    puis le resultat est ramene a la taille d'origine (contrat des appelants preserve)."""
    _set_slicing(pipe, max(image.size))
    orig_size = image.size
    w = round_to_multiple(image.width, 32)
    h = round_to_multiple(image.height, 32)
    if (w, h) != image.size:
        _dbg(f"refine: input {image.size[0]}x{image.size[1]} not /32 -> resized {w}x{h}")
        image = image.resize((w, h), Image.LANCZOS)
    out = pipe(
        prompt=prompt or "",
        image=image,
        width=w, height=h,
        strength=float(denoise),
        num_inference_steps=int(steps),
        guidance_scale=GUIDANCE,
        generator=_make_generator(seed),
    ).images[0]
    if out.size != orig_size:
        out = out.resize(orig_size, Image.LANCZOS)
    return out


def _feather_mask_np(th, tw, overlap, left, right, top, bottom):
    """Masque (th, tw, 1) a rampe lineaire sur les bords qui jouxtent une autre tuile."""
    mask = np.ones((th, tw, 1), dtype=np.float32)
    f = int(overlap)
    if f > 0:
        ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
        if left:
            mask[:, :f, 0] *= ramp[np.newaxis, :]
        if right:
            mask[:, tw - f:, 0] *= ramp[::-1][np.newaxis, :]
        if top:
            mask[:f, :, 0] *= ramp[:, np.newaxis]
        if bottom:
            mask[th - f:, :, 0] *= ramp[::-1][:, np.newaxis]
    return mask


# ----------------------------------------------------------------------------
# ControlNet Tile: refine qui VERROUILLE la structure de l'image source.
# Un refine img2img classique reinvente librement le contenu a fort denoise (visages
# qui derivent, textes qui se deforment, duplication en tuiles). Le ControlNet Tile
# conditionne CHAQUE etape sur l'image de depart: la composition, les contours et le
# texte restent en place, seul le detail est regenere. C'est l'idiome "Ultimate SD
# Upscale + Tile" et le controlnet officiel Z-Image (alibaba-pai, distille 8 steps).
# ----------------------------------------------------------------------------
# DESACTIVE (2026-08-16) -- cause RACINE trouvee, et elle est structurelle.
#
# LE REFINE TUILE PAR CONTROLNET RECOPIE LA SCENE DANS CHAQUE TUILE. Constate a l'oeil
# sur un upscale 1024x1536 -> 2048x3072 d'un salon: parquet jonche de fauteuils et de
# cheminees miniatures, murs et fauteuils couverts de motifs graves, et en prompt vide
# des mains / des cheveux / du faux texte. Quatre configurations essayees, TOUTES
# mauvaises: prompt vide @0.75, prompt de scene @0.75, @1.0, @1.3.
#
# POURQUOI (verifie dans les sources diffusers 0.39): ZImageControlNetPipeline n'a PAS
# de parametre `strength` -- le mot n'apparait pas une seule fois dans le fichier, contre
# 13 fois dans pipeline_z_image_img2img.py -- et il n'existe pas de
# pipeline_z_image_controlnet_img2img (seulement controlnet et controlnet_inpaint).
# Le pipeline DEBRUITE DONC ENTIEREMENT depuis du bruit. Applique a une TUILE, cela veut
# dire generer une image complete dont le seul ancrage est l'image de controle: le
# modele compose une scene entiere dans chaque tuile. L'img2img classique, lui, part de
# la tuile reelle a denoise 0.35 -- il ne peut que la retoucher, jamais la recomposer.
# Ce n'est pas un defaut de plomberie, c'est l'approche qui ne tient pas en tuilage avec
# ce que diffusers expose aujourd'hui.
#
# CONTROLE QUI CONFIRME: la MEME passe sur l'image ENTIERE (1024x1536, refine_tile=0,
# scale 0.75) sort PROPRE -- aucune recopie, aucun meuble miniature, structure identique
# (cheminee, tableau, les deux fauteuils, parquet). Une seule "tuile" = l'image, donc
# rien a recopier. Le coupable est bien le TUILAGE, pas le ControlNet.
# MAIS elle a pris 559 s (9 min 20) contre 52 s pour les 15 tuiles: transformer (12 Go) +
# controlnet (6.7 Go) + activations pleine image = debordement en RAM partagee. Donc le
# seul mode qui donne un bon resultat est inutilisable en pratique sur 32 Go.
#
# CE QUI DEBLOQUERAIT: un ZImageControlNetImg2ImgPipeline (controlnet + debruitage
# PARTIEL) -- il rendrait le tuilage sain, puisque chaque tuile repartirait de son
# contenu reel au lieu d'etre generee de zero. A surveiller dans diffusers.
#
# LE GARDE-FOU NE VOIT RIEN: les tuiles fautives sortaient a gap 20-45 et correlation
# +0.21/+0.99, largement dans les seuils (75 / 0.20). Normal: _cn_structure_gap mesure
# l'accord BASSE FREQUENCE (64x64) par tuile, et une tuile qui repeuple un parquet de
# petits meubles garde la meme repartition de tons et la meme structure grossiere.
#
# NB: le bug initialement rapporte ("apres un upscale ControlNet, les rendus SUIVANTS
# n'ont plus de rapport avec le prompt") ne s'est PAS reproduit. Rejoue fidelement, le
# diff d'etat partage affiche 'unchanged' avant/apres chaque passe, et les txt2img qui
# suivent sont propres. L'image "facade de palais" qui avait motive le rapport etait
# tres probablement la SORTIE D'UPSCALE elle-meme, c'est-a-dire exactement la recopie
# decrite ci-dessus, et non un pipeline empoisonne.
#
# Ce qui est ECARTE, preuves a l'appui:
#  - le transformer n'est PAS altere par le ControlNet. from_transformer ne fait que lui
#    greffer des references PARTAGEES (t_embedder/all_x_embedder/cap_embedder/
#    rope_embedder/noise_refiner/context_refiner/x_pad_token/cap_pad_token). Verifie
#    empiriquement sur modeles miniatures CPU: apres greffe + forward controlnet +
#    forward transformer + liberation, les 185 cles d'etat du transformer sont
#    inchangees et une passe de reference redonne un resultat BIT A BIT identique.
#  - la liberation ne deplace plus rien sur CPU (bug distinct, corrige).
#  - la saturation VRAM (plafond de tuile: 31 Go/164 s par tuile -> 28.2 Go/4.5 s).
#  - le prompt arrive intact au pipeline: cz_ui passe le MEME `fp` a txt2img_run et a
#    _gen_meta, donc "metadonnee correcte + image sans rapport" = le modele a bien recu
#    le prompt et l'a ignore.
#
# RESTE OUVERT A COTE (sans rapport avec le ControlNet): sur les sorties du 2026-08-16,
# 7 txt2img d'affilee (17:32:58-17:35:52) puis 3 autres (17:58:14-17:59:34) sont sortis
# corrompus. Non reproduit. Piste: les pipes derives par from_pipe, qui fait
# `torch_dtype = kwargs.pop("torch_dtype", torch.float32)` puis `new_pipeline.to(dtype)`
# sur des composants PARTAGES avec le base; et en offload 'model' (force pour un GGUF) le
# pipe derive n'a pas les hooks accelerate du base (_all_hooks vide ->
# maybe_free_model_hooks ne fait rien), ce qui laisse la chaine d'offload du base dans un
# etat incoherent pour la generation SUIVANTE.
#
# METHODE: lancer avec --log-level debug. _shared_state_diff() photographie l'etat
# PARTAGE (poids, dtypes, devices, hooks, scheduler, VAE, rope + une sonde fonctionnelle
# sur l'encodeur de texte) et ne journalise QUE ce qui bouge: la premiere ligne
# 'CHANGED' designe le coupable.
CONTROLNET_TILE_AVAILABLE = False        # recopie la scene dans chaque tuile (cf. ci-dessus)
CONTROLNET_TILE = CONTROLNET_TILE_AVAILABLE and bool(CONFIG.get("controlnet_tile", False))
# repo HF + fichier, ou chemin local absolu vers un .safetensors controlnet.
CONTROLNET_MODEL = str(CONFIG.get(
    "controlnet_tile_model",
    "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1/"
    "Z-Image-Turbo-Fun-Controlnet-Tile-2.1-8steps.safetensors")).strip()
try:
    CONTROLNET_SCALE = float(CONFIG.get("controlnet_tile_scale", 0.75))
except Exception:
    CONTROLNET_SCALE = 0.75
# Seuil du garde-fou anti-bruit (ecart basse frequence sortie/entree). Mesure sur des
# cas reels: ~35 quand le controlnet fonctionne, ~107 avec un checkpoint incompatible.
try:
    CONTROLNET_SANITY_MAX = float(CONFIG.get("controlnet_tile_sanity_max", 75))
except Exception:
    CONTROLNET_SANITY_MAX = 75.0
try:
    CONTROLNET_SANITY_MIN_CORR = float(CONFIG.get("controlnet_tile_sanity_min_corr", 0.20))
except Exception:
    CONTROLNET_SANITY_MIN_CORR = 0.20
# Tuile de refine MAXIMALE quand le ControlNet est actif. Le controlnet (6.7 Go) et le
# transformer (12 Go) doivent cohabiter en VRAM pendant la passe: avec les activations
# d'une tuile 1024, un upscale 1728x3072 monte a 31/32 Go et bascule en RAM partagee
# (mesure reelle: 164 s par tuile au lieu de ~10 s). Une tuile 768 divise ces
# activations par ~2. 0 = ne pas plafonner (si tu as beaucoup de VRAM).
try:
    CONTROLNET_MAX_TILE = int(CONFIG.get("controlnet_tile_max_tile", 768))
except Exception:
    CONTROLNET_MAX_TILE = 768
_CN_MODEL = None          # ZImageControlNetModel charge (6.7 Go bf16)
_CN_PIPE = None           # pipeline ControlNet monte sur les composants du base
_CN_PIPE_KEY = None       # identite du transformer courant -> rebuild si swap


def _free_controlnet():
    """Rend la VRAM du ControlNet (6.7 Go) et jette le pipeline qui le porte.

    CRUCIAL: ce modele n'est utile QUE pendant un refine. Le laisser resident ampute
    la VRAM de tout le reste (txt2img compris) -- constate en vrai: apres un upscale
    ControlNet, trois txt2img 1024x1024 d'affilee sont sortis corrompus (latents
    massacres, pas d'erreur), et seul un redemarrage a retabli les rendus."""
    global _CN_PIPE, _CN_PIPE_KEY, _CN_MODEL
    if _CN_MODEL is None and _CN_PIPE is None:
        return
    # NE PAS faire _CN_MODEL.to("cpu"): from_transformer greffe dans le controlnet des
    # references PARTAGEES vers les embedders du transformer (t_embedder, x_embedder,
    # cap_embedder...). Les deplacer emmene ces morceaux du transformer sur CPU et la
    # generation suivante plante sur "mat1 is on cuda:0, others on cpu" (constate).
    # On LACHE simplement la reference: le GC libere les poids PROPRES du controlnet,
    # et les modules partages survivent puisque le transformer les detient toujours.
    _CN_PIPE = _CN_PIPE_KEY = _CN_MODEL = None
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    _log("ControlNet released (6.7 GB of VRAM back for generation; it reloads "
         "from cache on the next ControlNet refine)")
    _shared_state_diff("after ControlNet release")


def set_controlnet_tile(v):
    global CONTROLNET_TILE
    if v and not CONTROLNET_TILE_AVAILABLE:
        _log("ControlNet Tile is DISABLED in this build - ignoring. The tiled refine "
             "recopies the scene into every tile (miniature furniture over the floor, "
             "carved patterns on the walls): the ControlNet pipeline has no 'strength' "
             "and denoises fully, so each tile is generated from scratch instead of "
             "being retouched. Needs a ControlNet img2img pipeline in diffusers.")
        CONTROLNET_TILE = False
        return
    CONTROLNET_TILE = bool(v)
    _log(f"ControlNet Tile refine -> {'ON' if CONTROLNET_TILE else 'OFF'}")
    if not CONTROLNET_TILE:
        _free_controlnet()


def set_controlnet_scale(v):
    global CONTROLNET_SCALE
    try:
        CONTROLNET_SCALE = min(1.5, max(0.1, float(v)))
    except (TypeError, ValueError):
        pass
    return f"ControlNet strength: {CONTROLNET_SCALE}"


def _resolve_controlnet_path():
    """Chemin local du .safetensors controlnet. 'repo/sous/chemin.safetensors' ->
    telecharge une fois dans le cache HF; un chemin absolu est pris tel quel."""
    spec = CONTROLNET_MODEL
    if os.path.isabs(spec) or os.path.isfile(spec):
        return spec
    parts = spec.split("/")
    if len(parts) < 3:
        raise RuntimeError(
            f"controlnet_tile_model invalide: '{spec}' (attendu 'org/repo/fichier.safetensors' "
            "ou un chemin local absolu)")
    repo, fname = "/".join(parts[:2]), "/".join(parts[2:])
    from huggingface_hub import hf_hub_download
    _log(f"fetching ControlNet {fname} from {repo} (once, ~6.7 GB) ...")
    return hf_hub_download(repo, fname)


def _load_controlnet():
    """Charge (une fois) le ZImageControlNetModel."""
    global _CN_MODEL
    if _CN_MODEL is not None:
        return _CN_MODEL
    from diffusers import ZImageControlNetModel
    path = _resolve_controlnet_path()
    _log(f"loading ControlNet: {os.path.basename(path)} ...")
    _CN_MODEL = _load_monitor(
        f"controlnet {os.path.basename(path)}",
        lambda: ZImageControlNetModel.from_single_file(path, torch_dtype=DTYPE))
    return _CN_MODEL


def _ensure_controlnet_pipe():
    """Pipeline ControlNet monte sur les composants DEJA charges (VAE, encodeur texte,
    transformer): aucun poids en double a part le controlnet lui-meme.

    NB: ZImageControlNetPipeline.__init__ appelle from_transformer(controlnet,
    transformer), qui attache au controlnet des references PARTAGEES vers les embedders
    du transformer -> le pipeline est lie a CE transformer. Un swap de checkpoint doit
    donc le reconstruire, d'ou la cle d'identite."""
    global _CN_PIPE, _CN_PIPE_KEY
    from diffusers import ZImageControlNetPipeline
    base = _ensure_base()
    key = id(base.transformer)
    if _CN_PIPE is not None and _CN_PIPE_KEY == key:
        return _CN_PIPE
    cn = _load_controlnet()
    _log("building ControlNet pipeline (weights shared with the base pipeline)")
    pipe = ZImageControlNetPipeline(
        scheduler=base.scheduler, vae=base.vae, text_encoder=base.text_encoder,
        tokenizer=base.tokenizer, transformer=base.transformer, controlnet=cn)
    # Placement du controlnet. NE PAS appeler enable_*_cpu_offload sur CE pipeline:
    # ses composants (VAE, encodeur, transformer) sont ceux du base et portent DEJA les
    # hooks accelerate; re-hooker par-dessus laisse le controlnet sur CPU alors que les
    # activations arrivent en cuda -> "mat1 is on cuda:0, others on cpu" (constate).
    #
    # Resident en permanence, le controlnet coute 6.7 Go: avec le transformer (12 Go),
    # le VAE, l'encodeur et les activations d'une tuile 1024, un upscale 1728x3072
    # atteint 31/32 Go -> debordement en RAM partagee, 164 s PAR TUILE au lieu de ~10 s
    # (mesure). On l'offloade donc individuellement (accelerate): il monte sur le GPU
    # pour son forward et redescend juste apres, sans toucher aux composants partages.
    # PAS d'offload par forward du controlnet (accelerate cpu_offload_with_hook):
    # from_transformer lui greffe des references PARTAGEES vers les embedders du
    # transformer, donc le descendre sur CPU emmene des morceaux du transformer avec
    # lui -> "index is on cpu, different from other tensors on cuda:0" (essaye, constate).
    # La VRAM se gagne ailleurs: en reduisant la tuile de refine (cf. _refine_tiled).
    if DEVICE == "cuda":
        cn.to(DEVICE)
    try:
        pipe.vae.config.force_upcast = False
    except Exception:
        pass
    _apply_sampler(pipe)
    _CN_PIPE, _CN_PIPE_KEY = pipe, key
    return pipe


def _cn_structure_gap(src, out):
    """(ecart couleur, correlation de structure) entre l'image de controle et la sortie,
    en basse frequence (64x64).

    Les DEUX sont necessaires: l'ecart couleur seul laisse passer une sortie corrompue
    qui garde la repartition des tons (mesure sur des cas reels: tuile legitime 57,
    sortie corrompue 106), et la correlation seule a une marge etroite en tuilage
    (tuile legitime 0.32, corrompue 0.10). Une corruption echoue sur les deux."""
    a = np.asarray(src.convert("RGB").resize((64, 64), Image.LANCZOS), np.float32)
    b = np.asarray(out.convert("RGB").resize((64, 64), Image.LANCZOS), np.float32)
    ga = np.asarray(src.convert("L").resize((64, 64), Image.LANCZOS), np.float32)
    gb = np.asarray(out.convert("L").resize((64, 64), Image.LANCZOS), np.float32)
    ga, gb = ga - ga.mean(), gb - gb.mean()
    denom = float(np.sqrt((ga * ga).sum()) * np.sqrt((gb * gb).sum())) + 1e-9
    return float(np.abs(a - b).mean()), float((ga * gb).sum() / denom)


def _controlnet_refine(image, prompt, steps, seed, scale=None):
    """Une passe ControlNet Tile: l'image d'ENTREE sert de conditionnement, la sortie
    garde sa structure. Pas de `strength` ici (le pipeline denoise entierement); c'est
    `controlnet_conditioning_scale` qui dose fidelite (haut) vs invention (bas).

    Leve si la sortie ne ressemble plus du tout a l'entree: un controlnet Z-Image ne
    marche qu'avec un transformer de la lignee sur laquelle il a ete entraine, et un
    checkpoint incompatible ne donne PAS une erreur mais du BRUIT (mesure: ecart basse
    frequence ~107 contre ~35 quand tout va bien). Le caller retombe alors sur
    l'img2img classique plutot que de sortir une image ratee."""
    pipe = _ensure_controlnet_pipe()
    _shared_state_diff("before ControlNet pass")
    orig = image.size
    w = round_to_multiple(image.width, 32)
    h = round_to_multiple(image.height, 32)
    src = image.convert("RGB")
    if (w, h) != orig:
        src = src.resize((w, h), Image.LANCZOS)
    _set_slicing(pipe, max(w, h))
    out = pipe(prompt=prompt or "", control_image=src, height=h, width=w,
               controlnet_conditioning_scale=float(CONTROLNET_SCALE if scale is None else scale),
               num_inference_steps=int(steps), guidance_scale=GUIDANCE,
               generator=_make_generator(seed)).images[0]
    _shared_state_diff("after ControlNet pass")
    gap, corr = _cn_structure_gap(src, out)
    if gap > CONTROLNET_SANITY_MAX or corr < CONTROLNET_SANITY_MIN_CORR:
        raise RuntimeError(
            f"output does not match the control image (color gap {gap:.0f}/"
            f"{CONTROLNET_SANITY_MAX:.0f}, structure correlation {corr:+.2f}/"
            f"{CONTROLNET_SANITY_MIN_CORR:+.2f}): this checkpoint is most likely not "
            "compatible with the ControlNet. Use the official Z-Image-Turbo, or turn "
            "ControlNet Tile off")
    _dbg(f"controlnet tile: gap {gap:.1f}, correlation {corr:+.2f}")
    return out.resize(orig, Image.LANCZOS) if out.size != orig else out


def _refine_pass(image, denoise, steps, prompt, seed):
    """Une passe de refine, ControlNet Tile si active (structure verrouillee) sinon
    img2img classique. Repli automatique sur l'img2img si le ControlNet est
    indisponible (modele absent, VRAM...) -> un upscale ne casse jamais pour ca."""
    if CONTROLNET_TILE:
        try:
            return _controlnet_refine(image, prompt, steps, seed)
        except Exception as e:
            _log(f"ControlNet Tile unavailable ({e}); falling back to img2img refine")
    return _refine_whole(get_pipe("img2img"), image, denoise, steps, prompt, seed)


def _refine_tiled(pipe, image, denoise, steps, prompt, seed, tile, overlap):
    """Passe Z-Image en tuiles avec recomposition feather (facon Ultimate SD Upscale).
    Plafonne le pic VRAM (une tuile a la fois) et permet le 4K+ sans coutures.
    Memes rampe lineaire + overlap-add que esrgan_upscale, mais a scale 1 sur PIL."""
    w, h = image.size
    tile = round_to_multiple(tile)                       # multiple de 16 pour le VAE
    # ControlNet actif = 6.7 Go de plus a cote du transformer pendant toute la passe:
    # on plafonne la tuile, sinon la VRAM sature et l'upscale part en RAM partagee.
    if CONTROLNET_TILE and CONTROLNET_MAX_TILE > 0 and tile > CONTROLNET_MAX_TILE:
        _log(f"refine tiled: tile {tile} -> {CONTROLNET_MAX_TILE} (ControlNet needs "
             f"~6.7 GB alongside the model; regle: controlnet_tile_max_tile)")
        tile = round_to_multiple(CONTROLNET_MAX_TILE)
    overlap = max(0, min(int(overlap), tile - 16))
    if w <= tile and h <= tile:
        # Une seule tuile = image entiere -> pas de duplication possible: denoise demande.
        return _refine_pass(image, denoise, steps, prompt, seed)
    # Anti-duplication 1: prompt vide par tuile (le prompt global decrit toute la compo).
    #
    # SAUF avec le ControlNet Tile, ou vider le prompt CASSE la passe. Raison: l'img2img
    # classique part de la tuile bruitee a `denoise` (0.35) -- l'image source retient
    # deja le contenu, le prompt ne sert qu'a orienter le detail, et le vider evite que
    # chaque tuile redessine toute la scene. Le pipeline ControlNet, lui, DEBRUITE
    # ENTIEREMENT (pas de `strength`): le seul ancrage est l'image de controle a
    # conditioning_scale. Sur une zone plate (un parquet, un mur) ce signal ne suffit
    # pas, et sans prompt le modele invente -- constate: un upscale 2048x3072 de salon a
    # rendu un tiers bas peuple de mains, de cheveux et de faux texte, alors que le
    # garde-fou passait (gap 20-30, correlation +0.69/+0.72, tres au-dessus des seuils:
    # il mesure l'accord BASSE FREQUENCE par tuile, qui reste bon meme quand la tuile
    # invente un autre sujet).
    # L'anti-duplication n'a de toute facon pas lieu d'etre ici: c'est l'image de
    # controle qui verrouille la composition de chaque tuile, c'est tout son interet.
    if CONTROLNET_TILE:
        _log("refine tiled: ControlNet actif -> prompt de scene GARDE par tuile "
             "(le controlnet debruite entierement; sans prompt les zones plates inventent)")
    else:
        prompt = _tile_prompt(prompt)
        if not (prompt or "").strip():
            _log("refine tiled: prompt vide par tuile (anti-duplication; regle refine_tile_prompt).")
    # Anti-duplication 2 (filet): a fort denoise chaque tuile peut encore deriver.
    denoise = float(denoise)
    if _TILE_DENOISE_CAP > 0 and denoise > _TILE_DENOISE_CAP:
        _log(f"refine tiled: denoise {denoise:.2f} > plafond {_TILE_DENOISE_CAP:.2f} -> "
             f"reduit a {_TILE_DENOISE_CAP:.2f} (regle refine_tile_denoise_cap).")
        denoise = _TILE_DENOISE_CAP

    acc = np.zeros((h, w, 3), dtype=np.float32)
    weight = np.zeros((h, w, 1), dtype=np.float32)
    step = max(16, tile - overlap)
    ys = list(range(0, h, step))
    xs = list(range(0, w, step))
    total = len(ys) * len(xs)
    _log(f"refine: tiled {w}x{h}, tile {tile} overlap {overlap} -> {len(xs)}x{len(ys)} = {total} tiles")
    # L'etage ESRGAN vient de liberer de gros buffers: les rendre AVANT la boucle evite
    # de demarrer le refine avec un cache d'allocations qui pousse a la saturation.
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        _dbg(f"tiled refine start{_vram_str()}")
    i = 0
    for y in ys:
        for x in xs:
            if _STOP:
                _log("refine tiled: stop requested")
                break
            i += 1
            x2, y2 = min(x + tile, w), min(y + tile, h)
            x1, y1 = max(x2 - tile, 0), max(y2 - tile, 0)
            cw, ch = x2 - x1, y2 - y1
            _progress(0.45 + 0.5 * (i - 1) / max(1, total), f"Refine tile {i}/{total}")
            crop = image.crop((x1, y1, x2, y2))
            _t_tile = time.time()
            out = _refine_pass(crop, denoise, steps, prompt, seed)
            _log(f"  tile {i}/{total} ({cw}x{ch}) in {time.time() - _t_tile:.1f}s{_vram_str()}")
            if out.size != (cw, ch):
                out = out.resize((cw, ch), Image.LANCZOS)
            out_arr = np.asarray(out.convert("RGB"), dtype=np.float32) / 255.0
            mask = _feather_mask_np(ch, cw, overlap,
                                    left=x1 > 0, right=x2 < w, top=y1 > 0, bottom=y2 < h)
            acc[y1:y2, x1:x2, :] += out_arr * mask
            weight[y1:y2, x1:x2, :] += mask

    out = acc / np.clip(weight, 1e-6, None)
    return Image.fromarray((out * 255.0 + 0.5).astype(np.uint8))


# ----------------------------------------------------------------------------
# Orchestration : process_one, batch txt2img (run/_gen_meta restent dans app.py
# car run emet des gr.Error pour l'UI).
# ----------------------------------------------------------------------------
@_gpu_serial
def process_one(image, esrgan_model, factor, denoise, steps, prompt, seed, tile, overlap,
                refine_tile=DEFAULT_REFINE_TILE, refine_overlap=DEFAULT_REFINE_OVERLAP,
                do_esrgan=True, refine_first=False, apply_force_ratio=False):
    """Pipeline sur une PIL Image, renvoie (image, timings_dict).
    do_esrgan=False -> img2img pur (saute l'etage ESRGAN, refine sur l'image native).
    refine_first=True -> refine PUIS ESRGAN (la diffusion tourne a la resolution
    native = bien plus rapide), au lieu de ESRGAN PUIS refine (detail en haute-def).
    apply_force_ratio=True + FORCE_RATIO defini -> amene l'ENTREE au ratio choisi avant
    traitement: FORCE_RATIO_MODE 'crop' = recadrage centre (facon Fooocus), 'extend' =
    outpaint des bandes manquantes (rien n'est perdu). Sinon: ratio natif preserve."""
    timings = {"esrgan": 0.0, "refine": 0.0}
    image = image.convert("RGB")
    if apply_force_ratio and FORCE_RATIO:
        r = _parse_ratio(FORCE_RATIO)
        if r:
            _before = image.size
            if FORCE_RATIO_MODE == "extend":
                # max(6, steps): l'outpaint des bandes reste correct meme si l'upscale
                # tourne en pur ESRGAN (steps/denoise a ~0).
                image = _extend_to_ratio(image, r[0], r[1], prompt, max(6, int(steps)), seed)
                _verb = "extend (outpaint)"
            else:
                image = _crop_to_ratio(image, r[0], r[1])
                _verb = "crop"
            _log(f"force ratio {r[0]}:{r[1]} -> {_verb} {_before[0]}x{_before[1]} "
                 f"to {image.size[0]}x{image.size[1]}")
    w0, h0 = image.size
    use_esrgan = bool(do_esrgan and esrgan_model)
    do_refine = float(denoise) > 0.001
    _dbg(f"process_one in={w0}x{h0} factor={factor} denoise={denoise} steps={int(steps)} "
         f"do_esrgan={do_esrgan} refine_first={refine_first} esrgan={esrgan_model} "
         f"refine_tile={int(refine_tile)}")

    def _esrgan_stage(img):
        t0 = time.time()
        iw, ih = img.size
        _progress(0.15, f"ESRGAN upscale {iw}x{ih}...")
        model = load_esrgan(esrgan_model)
        _log(f"ESRGAN upscale: {iw}x{ih} (tile {int(tile)}) ...")
        up = esrgan_upscale(img, model, int(tile), int(overlap))
        # Cible = facteur applique a la taille d'origine (independant de l'ordre).
        target_w = round_to_multiple(w0 * factor)
        target_h = round_to_multiple(h0 * factor)
        up = up.resize((target_w, target_h), Image.LANCZOS)
        timings["esrgan"] += time.time() - t0
        _log(f"ESRGAN done in {timings['esrgan']:.1f}s -> {target_w}x{target_h}")
        return up

    def _refine_stage(img):
        t0 = time.time()
        pipe = load_pipe()
        rw, rh = img.size
        rt = int(refine_tile)
        # Garde-fou anti-crash: refine whole-image trop grand (4K+) -> auto-tuilage.
        if rt <= 0 and max(rw, rh) > _AUTO_TILE_ABOVE:
            rt = 1024
            _log(f"refine: image {rw}x{rh} > {_AUTO_TILE_ABOVE}px -> auto-tiling (tile 1024) "
                 "pour eviter le pic VRAM (regle: auto_refine_tile_above)")
        if rt > 0:
            out = _refine_tiled(pipe, img, denoise, steps, prompt, seed,
                                rt, int(refine_overlap) or 64)
        else:
            _log(f"Z-Image refine: whole image {rw}x{rh}, "
                 + (f"ControlNet Tile (scale {CONTROLNET_SCALE:.2f})" if CONTROLNET_TILE
                    else f"denoise {float(denoise):.2f}")
                 + f", {int(steps)} steps ...")
            _progress(0.5, f"Z-Image refine {rw}x{rh}...")
            out = _refine_pass(img, denoise, steps, prompt, seed)
        timings["refine"] += time.time() - t0
        return out

    result = image
    if refine_first:
        # refine sur l'image native (rapide) puis agrandissement ESRGAN.
        if do_refine:
            result = _refine_stage(result)
        if use_esrgan:
            result = _esrgan_stage(result)
    else:
        # ordre classique: ESRGAN (detailleur) puis refine a la resolution agrandie.
        if use_esrgan:
            result = _esrgan_stage(result)
        if do_refine:
            result = _refine_stage(result)

    if not use_esrgan and not do_refine:
        _log(f"process_one: nothing to do (no ESRGAN, denoise=0) on {w0}x{h0}")

    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    _progress(1.0, "Done")
    _log(f"process_one done | esrgan {timings['esrgan']:.1f}s + refine {timings['refine']:.1f}s "
         f"= {timings['esrgan'] + timings['refine']:.1f}s")
    return result, timings


@_gpu_serial
def txt2img_run(prompt, width, height, gen_steps, seed, negative_prompt="",
                upscale=False, esrgan_model=None, factor=2.0, denoise=0.30, steps=12,
                tile=DEFAULT_TILE, overlap=DEFAULT_OVERLAP,
                refine_tile=DEFAULT_REFINE_TILE, refine_overlap=DEFAULT_REFINE_OVERLAP,
                refine_first=False):
    """Genere une image (txt2img Z-Image) puis, si upscale=True, la passe dans le
    pipeline ESRGAN + refine. Renvoie (image, timings_dict)."""
    timings = {"txt2img": 0.0, "esrgan": 0.0, "refine": 0.0}
    t0 = time.time()
    base = generate(prompt, width, height, gen_steps, seed, negative_prompt)
    timings["txt2img"] = time.time() - t0
    if not upscale:
        return base, timings
    result, t = process_one(base, esrgan_model, factor, denoise, steps, prompt, seed,
                            tile, overlap, refine_tile=refine_tile, refine_overlap=refine_overlap,
                            refine_first=refine_first)
    timings["esrgan"] = t.get("esrgan", 0.0)
    timings["refine"] = t.get("refine", 0.0)
    return result, timings


def _gen_meta(mode, prompt, negative="", seed=None, steps=None, guidance=None,
              size=None, model=None, styles=None, extra=None):
    """Construit le dict de metadonnees de generation (pour sidecar/PNG)."""
    m = {"app": "crispz-studio", "mode": mode, "prompt": prompt or "",
         "negative": negative or "", "date": _now_stamp()}
    if seed is not None and int(seed) >= 0:
        m["seed"] = int(seed)
    if steps is not None:
        m["steps"] = int(steps)
    if guidance is not None:
        m["guidance"] = float(guidance)
    if size:
        m["size"] = f"{size[0]}x{size[1]}"
    # Noms de styles appliques (en plus des mots-cles deja injectes dans le prompt).
    _styles = [s for s in (styles or []) if s and s not in ("None", "none")]
    if _styles:
        m["styles"] = _styles
    m["sampler"] = f"{SAMPLER}/{SCHEDULE}"
    m["model"] = model or (ZIMAGE_TRANSFORMER or BASE_REPO)
    if LORAS:
        m["loras"] = [f"{os.path.basename(p)}@{w}" for p, w in LORAS]
    if extra:
        m.update(extra)
    return m
