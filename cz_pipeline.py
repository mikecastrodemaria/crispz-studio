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
import cz_hw

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

# Encodeur texte de remplacement (Models > Checkpoints > Text encoder). Vide = celui du
# repo de base, comme avant. Sinon un DOSSIER au format transformers (config.json +
# poids) ou un repo HF ('owner/repo', 'owner/repo/sous-dossier') -- ex. un Qwen3-4B
# "abliterated" de meme taille. Seul l'encodeur change: tokenizer, VAE et transformer
# restent ceux du repo de base. Le pipeline Omni (modele separe) garde le sien.
CFG_TEXT_ENCODER_KEY = "text_encoder"


def _resolve_text_encoder(env, prefs, config):
    """Encodeur au demarrage: env > preferences > config. Une cle PRESENTE dans les
    preferences gagne meme vide: c'est le choix "Default" fait dans l'UI, et une valeur
    de config.txt ne doit pas le defaire au redemarrage (un "" passait pour absent)."""
    v = str(env.get("ZIMAGE_TEXT_ENCODER") or "").strip()
    if v:
        return v
    if CFG_TEXT_ENCODER_KEY in prefs:
        return str(prefs.get(CFG_TEXT_ENCODER_KEY) or "").strip()
    return str(config.get(CFG_TEXT_ENCODER_KEY) or "").strip()


TEXT_ENCODER = _resolve_text_encoder(os.environ, _prefs, CONFIG)
# Celui qui est REELLEMENT charge ('' = celui du repo de base). Distinct de TEXT_ENCODER:
# un encodeur qui ne convient pas au repo courant est ecarte au chargement, et les
# metadonnees disent ce qui a tourne, pas ce qui etait demande.
_TEXT_ENCODER_ACTIVE = ""
TEXT_ENCODERS_DIR = str(os.environ.get("TEXT_ENCODERS_DIR") or _prefs.get("text_encoders_dir")
                        or CONFIG.get("text_encoders_dir") or "").strip()

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
# LoRA appelees DANS LE PROMPT via <lora:nom[:poids]> (syntaxe A1111), re-derivees a
# chaque run depuis le prompt par consume_prompt_loras. Separees des slots (LORAS) pour
# que retirer le tag du prompt suffise a les desactiver sans toucher aux slots.
PROMPT_LORAS = []
# Coupe-circuit config: prompt_lora_tags=false -> les tags sont juste retires du prompt
# (jamais envoyes a l'encodeur) mais plus resolus/actives.
PROMPT_LORA_TAGS = bool(CONFIG.get("prompt_lora_tags", True))
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
# (le plus rapide). model = decharge par sous-module (bon compromis). sequential = plus
# agressif, plus lent. N'est PAS de la quantif: les poids restent BF16, ils transitent
# RAM <-> GPU. 'auto' (defaut) = test de VRAM libre au chargement (cz_hw): un modele qui
# deborde la VRAM ne plante pas, il bascule en RAM partagee (Windows Sysmem Fallback) et
# rend 50-100x plus lentement SANS message d'erreur -> on ne promeut 'none' que si la
# carte a prouve qu'elle a la place. Ordre de resolution (le premier defini gagne):
# choix UI/CLI explicite > env CZ_OFFLOAD > config default_cpu_offload > auto.
OFFLOAD_CHOICES = ("auto", "none", "model", "sequential")
OFFLOAD_MODE = ((os.environ.get("CZ_OFFLOAD") or "").strip()
                or str(CONFIG.get("default_cpu_offload", "") or "").strip()).lower() or "auto"
if OFFLOAD_MODE not in OFFLOAD_CHOICES:
    _log(f"CZ_OFFLOAD/default_cpu_offload '{OFFLOAD_MODE}' unknown -> auto")
    OFFLOAD_MODE = "auto"
# Mode concret resolu pour 'auto' (pose par _resolve_auto au 1er chargement) et flag du
# filet de securite runtime (pose par le callback VRAM pendant le denoise).
_AUTO_OFFLOAD = ""
_VRAM_DOWNGRADE = False

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


# --- Encodeur texte de remplacement ---------------------------------------------------
# Z-Image lit l'AVANT-DERNIER etat cache de l'encodeur (hidden_states[-2]) et le
# transformer attend des embeddings larges de cap_feat_dim (2560, la largeur du Qwen3-4B):
# un encodeur ne convient que s'il a la meme famille, la meme largeur et le meme nombre de
# couches que celui du repo de base -- plus profond, l'avant-dernier etat serait une autre
# couche. Un Qwen3-4B "abliterated" ou fine-tune se branche tel quel. On le verifie a la
# config, AVANT de lire 8 Go.
_TE_FILE_EXTS = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".sft", ".gguf")


def _looks_single_file(p):
    """Vrai si le NOM est celui d'un fichier de poids, qu'il existe ou non (_is_single_file
    exige un fichier present: un chemin colle d'une autre machine lui echapperait)."""
    return bool(p) and str(p).lower().endswith(_TE_FILE_EXTS)


def _split_hf_src(src):
    """'owner/repo/sous/dossier' -> ('owner/repo', 'sous/dossier'). Les poids d'un
    encodeur publie sur HF sont souvent dans un sous-dossier du repo."""
    parts = [p for p in str(src).replace("\\", "/").split("/") if p]
    if len(parts) > 2:
        return "/".join(parts[:2]), "/".join(parts[2:])
    return str(src), None


def _enc_dims(cfg):
    """(largeur, couches, famille) d'une config transformers. Les VL rangent la partie
    texte sous 'text_config'; T5 dit d_model / num_layers."""
    c = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    h = c.get("hidden_size") or c.get("d_model")
    n = c.get("num_hidden_layers") or c.get("num_layers")
    return (int(h) if h else None, int(n) if n else None, cfg.get("model_type"))


def _base_text_encoder_config(base=None):
    """config.json de l'encodeur du repo de base, ou None si illisible."""
    base = (base or BASE_REPO or "").strip()
    try:
        cfg = os.path.join(base, "text_encoder", "config.json")
        if not os.path.isfile(cfg):
            from huggingface_hub import hf_hub_download
            try:
                cfg = hf_hub_download(base, "text_encoder/config.json", local_files_only=True)
            except Exception:
                cfg = hf_hub_download(base, "text_encoder/config.json")
        with open(cfg, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        _dbg(f"cannot read {base}'s text encoder config: {e}")
        return None


def _text_encoder_source(src):
    """Localise l'encodeur `src`: (config, dossier ou repo, sous-dossier) ou None.
    Dossier local: config.json a la racine ou dans text_encoder/. Repo HF: idem, ou le
    sous-dossier nomme dans l'id."""
    src = (src or "").strip()
    if not src:
        return None
    if os.path.isdir(src):
        for sub in (None, "text_encoder"):
            p = os.path.join(src, sub, "config.json") if sub else os.path.join(src, "config.json")
            if os.path.isfile(p):
                try:
                    with open(p, encoding="utf-8") as f:
                        return json.load(f), src, sub
                except Exception:
                    return None
        return None
    if os.path.exists(src) or _looks_single_file(src) or "\\" in src or os.path.isabs(src):
        return None
    repo, sub0 = _split_hf_src(src)
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        return None
    for sub in ([sub0] if sub0 else [None, "text_encoder"]):
        rel = f"{sub}/config.json" if sub else "config.json"
        for local in (True, False):          # le cache d'abord: marche hors ligne
            try:
                p = hf_hub_download(repo, rel, local_files_only=local)
                with open(p, encoding="utf-8") as f:
                    return json.load(f), repo, sub
            except Exception:
                continue
    return None


def _encoder_label(src):
    """Nom lisible d'un encodeur: le NOM du dossier -- jamais le chemin, qui finirait
    dans les PNG partages avec le nom de la session Windows -- ou l'id du repo HF."""
    src = (src or "").strip()
    if not src:
        return ""
    if os.path.isabs(src) or os.path.exists(src) or "\\" in src:
        parts = [p for p in src.replace("\\", "/").split("/") if p]
        if len(parts) >= 2 and parts[-1] == "text_encoder":
            return parts[-2]
        return parts[-1] if parts else src
    return src


def _text_encoder_problem(src, base=None):
    """Raison de refuser `src` comme encodeur du repo `base`, ou None s'il convient."""
    src = (src or "").strip()
    if not src:
        return None
    if src.lower().endswith(".gguf"):
        return ("a GGUF text encoder is a ComfyUI / llama.cpp file; this app loads the "
                "transformers folder (config.json + .safetensors)")
    if os.path.isfile(src) or _looks_single_file(src):
        return ("a single file carries no config.json; point to the FOLDER that holds "
                "config.json and the weights")
    found = _text_encoder_source(src)
    if found is None:
        return ("no config.json found, neither at its root nor in text_encoder/"
                if os.path.isdir(src) else
                "neither a folder on this machine nor a readable Hugging Face repo")
    ref_cfg = _base_text_encoder_config(base)
    if ref_cfg is None:
        return None                      # rien a comparer: le chargement tranchera
    (h, n, t), (rh, rn, rt) = _enc_dims(found[0]), _enc_dims(ref_cfg)
    b = (base or BASE_REPO)
    if t and rt and t != rt:
        return f"a '{t}' model, and {b} uses a '{rt}' text encoder"
    if h and rh and h != rh:
        return (f"hidden size {h}, and {b}'s encoder is {rh} wide: the transformer "
                f"cannot read its embeddings")
    if n and rn and n != rn:
        return (f"{n} layers, and {b}'s encoder has {rn}: Z-Image reads the "
                f"second-to-last layer, which would be another one")
    return None


def _encoder_class(base=None):
    """Classe transformers de l'encodeur, lue dans le model_index.json du repo de base:
    Qwen3Model pour Z-Image, alors que text_encoder/config.json dit Qwen3ForCausalLM.
    C'est la classe de model_index.json que diffusers charge et que le pipeline attend
    (il lit hidden_states, sans lm_head); un checkpoint Qwen3ForCausalLM s'y charge tel
    quel (transformers retire le prefixe 'model.' et laisse le lm_head de cote)."""
    base = (base or BASE_REPO or "").strip()
    try:
        p = os.path.join(base, "model_index.json")
        if not os.path.isfile(p):
            from huggingface_hub import hf_hub_download
            try:
                p = hf_hub_download(base, "model_index.json", local_files_only=True)
            except Exception:
                p = hf_hub_download(base, "model_index.json")
        with open(p, encoding="utf-8") as f:
            lib, cls = json.load(f)["text_encoder"]
        import importlib
        return getattr(importlib.import_module(lib), cls)
    except Exception as e:
        raise RuntimeError(f"cannot tell which class {base}'s text encoder uses "
                           f"({type(e).__name__}: {e})") from e


def _load_text_encoder(src, base=None):
    """Charge l'encodeur `src` en DTYPE, avec la classe du repo de base."""
    found = _text_encoder_source(src)
    if found is None:
        raise RuntimeError(f"{src}: no config.json")
    _cfg, where, sub = found
    kw = {"torch_dtype": DTYPE}
    if sub:
        kw["subfolder"] = sub
    return _encoder_class(base).from_pretrained(where, **kw)


def list_text_encoders():
    """Dossiers d'encodeur proposes dans l'onglet Models: les sous-dossiers a config.json
    de `text_encoders_dir`, ou de text_encoders / text_encoder / clip a cote des dossiers
    de checkpoints (principal ET extra: une bibliotheque partagee entre forks vit souvent
    dans l'extra) ou de leur parent (conventions ComfyUI et Forge)."""
    roots = [TEXT_ENCODERS_DIR] if TEXT_ENCODERS_DIR else []
    for cdir in _checkpoint_dirs():
        here = os.path.abspath(cdir or ".")
        for up in (os.path.dirname(here), os.path.dirname(os.path.dirname(here))):
            roots += [os.path.join(up, n) for n in ("text_encoders", "text_encoder", "clip")]
    out = []
    for r in dict.fromkeys(roots):       # sans doublon, ordre garde
        try:
            names = sorted(os.listdir(r))
        except OSError:
            continue
        for d in names:
            p = os.path.join(r, d)
            if p in out or not os.path.isdir(p):
                continue
            if (os.path.isfile(os.path.join(p, "config.json"))
                    or os.path.isfile(os.path.join(p, "text_encoder", "config.json"))):
                out.append(p)
    return out



def _hf_cache_dir():
    """Dossier du cache Hugging Face (suit HF_HUB_CACHE / HF_HOME), ou None."""
    try:
        from huggingface_hub import constants
        return constants.HF_HUB_CACHE
    except Exception:
        return None


def _scan_cached_encoders():
    """[(id HF, config)] du cache Hugging Face: depots qui ne sont PAS des pipelines
    diffusers (pas de model_index.json), dont une config -- a la racine ou dans un
    sous-dossier -- a ses poids a cote (revision la plus recente)."""
    root = _hf_cache_dir()
    if not root or not os.path.isdir(root):
        return []
    out = []
    for d in sorted(os.listdir(root)):
        if not d.startswith("models--"):
            continue
        repo = d[len("models--"):].replace("--", "/", 1)
        snaps = os.path.join(root, d, "snapshots")
        try:
            revs = sorted(os.listdir(snaps),
                          key=lambda r: os.path.getmtime(os.path.join(snaps, r)), reverse=True)
        except OSError:
            continue
        if not revs:
            continue
        snap = os.path.join(snaps, revs[0])
        if os.path.isfile(os.path.join(snap, "model_index.json")):
            continue                                 # un pipeline diffusers, pas un encodeur
        try:
            subs = [""] + sorted(s for s in os.listdir(snap) if os.path.isdir(os.path.join(snap, s)))
        except OSError:
            continue
        for s in subs:
            p = os.path.join(snap, s) if s else snap
            try:
                with open(os.path.join(p, "config.json"), encoding="utf-8") as f:
                    cfg = json.load(f)
                if not isinstance(cfg, dict):
                    continue
                if not any(fn.endswith(".safetensors") for fn in os.listdir(p)):
                    continue                         # config seule, poids pas telecharges
            except Exception:
                continue
            out.append((f"{repo}/{s}" if s else repo, cfg))
    return out


def list_cached_text_encoders(base=None):
    """Encodeurs COMPATIBLES deja telecharges dans le cache Hugging Face, en (nom, id HF).
    Un encodeur telecharge depuis HF vit dans ce cache, pas dans un dossier text_encoders:
    sans ce balayage, la liste de l'onglet Models ne le montrait pas (releve sur klein le
    2026-09-10). Compatible = meme famille, largeur et nombre de couches que l'encodeur du
    repo de base. La valeur est l'id HF: lisible dans les metadonnees."""
    ref_cfg = _base_text_encoder_config(base)
    if not ref_cfg:
        return []
    ref = _enc_dims(ref_cfg)
    return [(hid, hid) for hid, cfg in _scan_cached_encoders() if _enc_dims(cfg) == ref]


def cached_text_encoder_mismatches(base=None):
    """Encodeurs du cache HF de la MEME famille mais d'une autre taille que celui du repo
    de base: masques de la liste (ils seraient refuses), nommes a cote pour que l'on sache
    pourquoi. ([(id HF, largeur)], largeur attendue)."""
    ref_cfg = _base_text_encoder_config(base)
    if not ref_cfg:
        return [], None
    rh, rn, rt = _enc_dims(ref_cfg)
    out = []
    for hid, cfg in _scan_cached_encoders():
        h, n, t = _enc_dims(cfg)
        if t == rt and (h, n) != (rh, rn):
            out.append((hid, h))
    return out, rh


def set_text_encoder(src):
    """Choisit l'encodeur texte ('' = celui du repo de base). Un changement LIBERE le
    pipeline: l'encodeur se charge avec lui (from_pretrained(text_encoder=...)), sans
    echange a chaud sous les hooks d'offload, et les pipes derives (from_pipe), qui
    partagent l'encodeur du base, partent avec lui. Omni (modele separe) garde le sien."""
    global TEXT_ENCODER
    src = (src or "").strip()
    if src == TEXT_ENCODER:
        return
    TEXT_ENCODER = src
    free_vram()
    _log(f"text encoder -> {_encoder_label(src) or '(base repo)'} -> full reload on next run")


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


def set_prompt_loras(pairs):
    """Definit les LoRA appelees dans le prompt (liste de (chemin_abs, poids)). Appele a
    chaque run par consume_prompt_loras — y compris avec [] quand le prompt n'a plus de
    tag, pour que la LoRA se desactive au run suivant."""
    global PROMPT_LORAS
    new = [(p, float(w)) for p, w in (pairs or [])]
    if new != PROMPT_LORAS:
        PROMPT_LORAS = new
        _log("prompt LoRAs -> "
             + (", ".join(f"{os.path.basename(p)}@{w}" for p, w in new) or "(none)"))


def _effective_loras():
    """Slots (LORAS) + LoRA du prompt (PROMPT_LORAS), dedoublonnees par chemin: une LoRA
    presente des deux cotes garde le poids du PROMPT (le tag est le reglage le plus
    explicite). C'est CETTE liste que _apply_loras pose sur le transformer."""
    merged = {os.path.normcase(p): (p, float(w)) for p, w in LORAS}
    for p, w in PROMPT_LORAS:
        merged[os.path.normcase(p)] = (p, float(w))
    return list(merged.values())


def resolve_lora_name(name):
    """Resout un nom de tag <lora:...> vers un chemin RELATIF de list_loras(), ou None.
    Tolerant (insensible a la casse, '\\' acceptes): chemin relatif exact -> nom de
    fichier -> stem (sans extension) -> sous-chaine du stem si le match est UNIQUE
    (ambigu = non resolu: on ne devine pas entre deux fichiers)."""
    want = str(name or "").strip().replace("\\", "/").lower()
    if not want:
        return None
    files = list_loras()
    by_rel = {f.lower(): f for f in files}
    if want in by_rel:
        return by_rel[want]
    base = want.rsplit("/", 1)[-1]
    stem = base.rsplit(".", 1)[0] if base.lower().endswith((".safetensors", ".ckpt", ".pt")) \
        else base
    exact = [f for f in files if os.path.basename(f).lower() in (base, stem + ".safetensors",
                                                                 stem + ".ckpt", stem + ".pt")]
    if not exact:
        exact = [f for f in files
                 if os.path.splitext(os.path.basename(f))[0].lower() == stem]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        _log(f"lora tag '{name}': ambiguous ({len(exact)} files share this name), not resolved")
        return None
    partial = [f for f in files if stem in os.path.splitext(os.path.basename(f))[0].lower()]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        _log(f"lora tag '{name}': ambiguous ({len(partial)} partial matches), not resolved")
    return None


def consume_prompt_loras(prompt):
    """Point d'entree unique pour les tags <lora:nom[:poids]> d'un prompt utilisateur:
    les extrait, les resout dans LORAS_DIR et ACTIVE les LoRA trouvees pour ce run
    (PROMPT_LORAS, combinees aux slots par _apply_loras). Renvoie (prompt_nettoye,
    missing) — missing = noms introuvables localement; l'appelant decide de la suite
    (l'UI bloque avec un message + recherche CivitAI, la CLI sort en erreur). Les tags
    sont TOUJOURS retires du prompt: un fragment de syntaxe ne va jamais a l'encodeur.
    Poids absent -> LORA_WEIGHT; poids hors bornes -> ramene dans [min, max]."""
    from cz_prompt import extract_lora_tags
    clean, tags = extract_lora_tags(prompt)
    if not tags:
        set_prompt_loras([])
        return clean, []
    if not PROMPT_LORA_TAGS:
        _log(f"prompt lora tags disabled (config prompt_lora_tags=false): "
             f"{len(tags)} tag(s) removed from the prompt, not applied")
        set_prompt_loras([])
        return clean, []
    pairs, missing = [], []
    for name, w in tags:
        rel = resolve_lora_name(name)
        if not rel:
            missing.append(name)
            continue
        w = LORA_WEIGHT if w is None else float(w)
        cw = min(LORA_WEIGHT_MAX, max(LORA_WEIGHT_MIN, w))
        if cw != w:
            _log(f"lora tag '{name}': weight {w} clamped to {cw} "
                 f"(bounds {LORA_WEIGHT_MIN:g}..{LORA_WEIGHT_MAX:g})")
        pairs.append((os.path.join(LORAS_DIR, rel), cw))
    set_prompt_loras(pairs)
    return clean, missing


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
    """Change le mode d'offload CPU. Invalide le pipe (hooks poses au chargement).
    Valeur inconnue -> 'auto' (jamais 'none': le repli doit etre le mode SUR)."""
    global OFFLOAD_MODE, _AUTO_OFFLOAD
    mode = str(mode or "").strip().lower()
    mode = mode if mode in OFFLOAD_CHOICES else "auto"
    if mode != OFFLOAD_MODE:
        OFFLOAD_MODE = mode
        _AUTO_OFFLOAD = ""   # 'auto' refait le test VRAM au prochain chargement
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


# ---- Offload 'auto': test VRAM au chargement + filet de securite runtime (cz_hw) ----

def _hw_profile_path():
    """Profil des verdicts du test VRAM (JSON), a cote des autres caches."""
    return os.path.join(HERE, "cache", "hw_profile.json")


def _model_footprint_gb():
    """Empreinte VRAM (Go) du pipeline complet en offload 'none' (poids en VRAM,
    hors activations). Mesure Z-Image Turbo bf16 tout-en-VRAM: ~19 Go (transformer
    + encodeur Qwen3-4B + VAE). Surcharge possible via config 'model_footprint_gb'
    (ex. gros fine-tune, encodeur de remplacement plus lourd)."""
    try:
        v = float(CONFIG.get("model_footprint_gb", 0) or 0)
        if v > 0:
            return v
    except Exception:
        pass
    return 19.0


def _resolve_auto(retest=False):
    """Mode concret pour 'auto' (memoise pour le process). Le verdict est cache
    dans cache/hw_profile.json par (GPU, build torch/cuda, modele, dtype): le
    test ne coute qu'un mem_get_info par combinaison, puis une lecture JSON."""
    global _AUTO_OFFLOAD
    if _AUTO_OFFLOAD and not retest:
        return _AUTO_OFFLOAD
    mode, why = cz_hw.resolve(
        "auto", footprint_gb=_model_footprint_gb(),
        model_id=(ZIMAGE_TRANSFORMER or BASE_REPO), dtype="bf16",
        profile_path=_hw_profile_path(), retest=retest)
    _AUTO_OFFLOAD = mode
    _log(f"offload auto -> {mode} ({why})")
    return mode


def offload_status():
    """Ligne d'etat pour l'UI: mode demande + resolution 'auto' le cas echeant."""
    if OFFLOAD_MODE != "auto":
        return f"offload: {OFFLOAD_MODE} (explicit)"
    if not _AUTO_OFFLOAD:
        return "offload: auto (resolves at the next model load)"
    return f"offload: auto -> {_AUTO_OFFLOAD}"


def retest_offload():
    """Bouton 'Re-test VRAM' de l'UI: refait le test en ignorant le profil (autre
    app fermee/ouverte, driver change...). Invalide le pipe si le verdict change."""
    if OFFLOAD_MODE != "auto":
        return f"Offload is '{OFFLOAD_MODE}' (explicit) - select 'auto' to use the VRAM test."
    old = _AUTO_OFFLOAD
    mode = _resolve_auto(retest=True)
    if old and mode != old:
        free_vram()
        return f"auto -> {mode} (was {old}; the pipeline will reload)"
    return f"auto -> {mode}"


def _vram_guard_kwargs():
    """Filet de securite runtime: callback_on_step_end qui verifie APRES le 1er
    step de denoise en mode effectif 'none' que la VRAM n'est pas saturee (le
    test au chargement estime; un process tiers a pu arriver depuis, ou la
    resolution demandee depasse la marge). Sature -> flag + interruption du
    denoise; l'appelant bascule en 'model' et rejoue le job UNE fois.
    {} quand la garde est inutile (offload deja actif, pas de CUDA)."""
    if DEVICE != "cuda" or _effective_offload() != "none":
        return {}

    def _cb(pipe, i, t, cb_kwargs):
        global _VRAM_DOWNGRADE
        if i == 0 and cz_hw.vram_saturated():
            _VRAM_DOWNGRADE = True
            pipe._interrupt = True
        return cb_kwargs
    return {"callback_on_step_end": _cb}


def _pipe_guarded(pipe, **kwargs):
    """Appelle le pipe avec la garde VRAM quand elle est active. Un pipeline qui
    ne connait pas callback_on_step_end (TypeError) tourne sans garde: le filet
    est un bonus, jamais une cause d'echec. Renvoie la premiere image."""
    guard = _vram_guard_kwargs()
    if guard:
        try:
            return pipe(**kwargs, **guard).images[0]
        except TypeError:
            _dbg("callback_on_step_end unsupported -> VRAM guard disabled")
    return pipe(**kwargs).images[0]


def _consume_vram_downgrade():
    """Si la garde a declenche: applique la retrogradation vers 'model',
    l'enregistre dans le profil (le prochain boot demarre directement en 'model')
    et libere le pipe. True -> l'appelant rejoue le job une fois."""
    global _VRAM_DOWNGRADE, _AUTO_OFFLOAD
    if not _VRAM_DOWNGRADE:
        return False
    _VRAM_DOWNGRADE = False
    _log("WARNING: VRAM saturated after the first denoise step in offload 'none' "
         "-> the render would spill to shared RAM (50-100x slower, no error). "
         "Switching to 'model' and retrying the job once.")
    cz_hw.record_downgrade(_hw_profile_path(), ZIMAGE_TRANSFORMER or BASE_REPO,
                           "bf16", "model", "VRAM saturated after denoise step 1")
    if OFFLOAD_MODE == "auto":
        _AUTO_OFFLOAD = "model"
        free_vram()
    else:
        set_offload_mode("model")
    return True


def free_vram():
    """Libere le pipeline de base + les pipelines derives et rend la VRAM
    (palier 3: unload sur inactivite ou endpoint /unload). Rechargement paresseux."""
    global _BASE_PIPE, _DERIVED, _LOADED_KEY, _APPLIED_LORAS, _TEXT_ENCODER_ACTIVE
    _BASE_PIPE = None
    _DERIVED = {}
    _LOADED_KEY = None
    _APPLIED_LORAS = []      # plus de pipe -> plus d'adaptateur pose
    _TEXT_ENCODER_ACTIVE = ""  # ... ni d'encodeur de remplacement charge
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def is_oom(e):
    """Vrai si `e` est un manque de VRAM, sous ses deux formes: celle de l'allocateur de
    torch ("CUDA out of memory. Tried to allocate ...") et celle d'un appel CUDA direct
    ("CUDA error: out of memory"). La seconde arrive quand le cache de torch a tout
    reserve: un noyau charge a la demande ne trouve plus rien et ne peut rien reclamer a
    ce cache (porte de crispz-klein 1.36.4)."""
    s = str(e).lower()
    return "out of memory" in s or "alloc_failed" in s


def release_vram(offload=False, why=""):
    """Rend au pilote la VRAM que le cache de torch garde en reserve, sans rien decharger.

    torch ne vide son cache que quand SON allocateur echoue; les autres consommateurs
    echouent sans pouvoir le recuperer. offload=True remet aussi sur le CPU les modeles
    qu'un appel interrompu a laisses sur le GPU en offload 'model', pour CHAQUE pipeline
    charge avec ses hooks (le base, et le pipeline Omni quand il est charge a part). Sur
    crispz-klein, un transformer a moitie deplace par un OOM restait sur le GPU: 10,8 Go
    coinces, et chaque rendu suivant echouait jusqu'au redemarrage. `why` journalise
    l'etat de la VRAM apres coup."""
    if offload:
        seen = set()
        for p in [_BASE_PIPE, *_DERIVED.values()]:
            if p is None or id(p) in seen or not getattr(p, "_all_hooks", None):
                continue
            seen.add(id(p))
            try:
                p.maybe_free_model_hooks()   # diffusers: tout sur le CPU, hooks reposes
            except Exception as e:
                _dbg(f"release_vram: offload failed ({e})")
    gc.collect()
    if DEVICE != "cuda":
        return
    try:
        torch.cuda.empty_cache()
        if why:
            free, total = torch.cuda.mem_get_info()
            _log(f"VRAM released ({why}): {free / 1024 ** 3:.1f} GB free of "
                 f"{total / 1024 ** 3:.1f}, torch holds "
                 f"{torch.cuda.memory_allocated() / 1024 ** 3:.1f} GB "
                 f"(reserved {torch.cuda.memory_reserved() / 1024 ** 3:.1f})")
    except Exception as e:
        _dbg(f"release_vram: {e}")


def retry_on_oom(what, fn, *args, **kwargs):
    """Appelle fn(*args, **kwargs); sur un manque de VRAM, rend la VRAM (cache de torch,
    modeles restes sur le GPU) et retente UNE fois. Un second echec rend encore la VRAM
    avant de remonter l'erreur: le processus reste utilisable pour le rendu suivant."""
    err = None
    for attempt in (1, 2):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if not is_oom(e):
                raise
            # Le traceback retient les frames, donc leurs tenseurs sur le GPU: on le
            # lache AVANT de vider le cache, sinon empty_cache ne recupere rien.
            err = e.with_traceback(None)
            err.__context__ = err.__cause__ = None
        if attempt == 1:
            _log(f"{what}: out of VRAM ({str(err).strip().splitlines()[0]}), "
                 f"freeing it and retrying once")
        release_vram(offload=True, why=what)
    raise err


# Au-dela de ce cote (px) on active l'attention slicing (whole-image 2K+ -> evite le
# spill VRAM 32 Go). En-dessous (tuiles 1024, txt2img 1024/1536) -> slicing OFF =
# attention SDPA native = RAPIDE (comme ComfyUI). Reglable via config attention_slice_above.
_SLICE_ABOVE = int(CONFIG.get("attention_slice_above", 1664))

# Garde-fou: au-dela de ce cote (px), un refine "whole image" (refine_tile=0) est auto-
# tuile (tuile 1024). Defaut = le seuil de slicing: au-dela, un whole-image serait slice
# (lent: ~120s en 2K) ET risque le spill VRAM (4K -> crash). Tuiler est plus rapide ET sur.
_AUTO_TILE_ABOVE = int(CONFIG.get("auto_refine_tile_above", _SLICE_ABOVE))

# Taille de la tuile employee par cet auto-tuilage. "auto" (defaut) = calculee par
# _pick_refine_tile ; un entier fige la taille (ancien comportement : 1024).
# Mesure (RTX 5090, sortie 4096x4096, denoise 0.40, overlap 64) : le cout par pixel est
# PLAT de 768 a 1024 (1.78 / 1.83 / 1.79 us/px) et ne grimpe qu'au-dela (2.41 a 1536,
# 3.00 a 2048). Le temps suit donc la SURFACE TUILEE (n x tuile^2), pas la taille de la
# tuile. Or a 1024 la grille deborde : pas de 960 sur 4096 -> la derniere tuile est
# rabattue et recouvre la precedente sur 832px au lieu de 64, soit 1.56x la surface de
# l'image. A 896 le pas tombe juste (1.20x) -> 36.7s au lieu de 46.9s sur la meme image,
# a nombre de tuiles (25) et de coutures (8) IDENTIQUE.
# Bornes [768, 1024] : en dessous on multiplie tuiles et coutures et chaque tuile voit
# moins de contexte (le rendu derive - un arriere-plan flou se reconstruit differemment,
# verifie visuellement) ; au-dessus l'attention devient superlineaire.
_AUTO_TILE_MIN = int(CONFIG.get("auto_refine_tile_min", 768))
_AUTO_TILE_MAX = int(CONFIG.get("auto_refine_tile_max", 1024))
_AUTO_TILE_SIZE = str(CONFIG.get("auto_refine_tile", "auto")).strip().lower()


def _pick_refine_tile(w, h, overlap):
    """Tuile qui minimise la surface tuilee pour couvrir w x h (= le cout reel de la passe).

    A surface egale on garde la PLUS GRANDE tuile : moins de coutures et plus de contexte
    par tuile. Un entier dans auto_refine_tile court-circuite le calcul (taille figee)."""
    if _AUTO_TILE_SIZE not in ("auto", "", "0"):
        try:
            return round_to_multiple(int(_AUTO_TILE_SIZE))
        except ValueError:
            _log(f"config auto_refine_tile='{_AUTO_TILE_SIZE}' invalide (attendu 'auto' ou "
                 "un entier) -> calcul automatique")
    lo = max(256, _AUTO_TILE_MIN)
    hi = max(lo, _AUTO_TILE_MAX)
    ov = max(0, int(overlap))
    cands = []
    for t in range(lo, hi + 1, 32):
        step = max(16, t - ov)
        n = len(range(0, max(1, int(w)), step)) * len(range(0, max(1, int(h)), step))
        cands.append((n * t * t, -t, t))       # surface mini, puis plus grande tuile
    return min(cands)[2]

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
    """Synchronise les adaptateurs LoRA du pipe avec la liste EFFECTIVE (slots LORAS +
    LoRA du prompt PROMPT_LORAS), SANS recharger le modele.

    Le transformer reste en VRAM; seuls les adaptateurs PEFT bougent:
      - memes fichiers, poids differents -> set_adapters (immediat)
      - jeu de LoRA different            -> unload_lora_weights + reload des LoRA (~1s)
    Les pipes derives (from_pipe) partagent ce transformer -> ils suivent automatiquement.
    Renvoie True si applique, False si echec (le caller retombe sur un reload complet)."""
    global _APPLIED_LORAS
    eff = _effective_loras()
    if not force and _APPLIED_LORAS == eff:
        return True
    old_paths = [p for p, _ in _APPLIED_LORAS]
    new_paths = [p for p, _ in eff]
    try:
        if not force and old_paths and old_paths == new_paths:
            # Seuls les poids changent -> re-ponderation instantanee.
            pipe.set_adapters(_lora_names(eff), [float(w) for _, w in eff])
            _APPLIED_LORAS = list(eff)
            _log("LoRA weights updated in place (no reload): "
                 + ", ".join(f"{os.path.basename(p)}@{w}" for p, w in eff))
            return True
        if old_paths or force:
            _clear_loras(pipe)
        names, weights = [], []
        for i, (p, w) in enumerate(eff):
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
        _APPLIED_LORAS = list(eff)
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


# Plage de chaque format 8 bits: la valeur stockee la plus grande qu'un poids QUANTIFIE
# (poids / echelle) peut atteindre.
_QUANT_RANGE = {torch.float8_e4m3fn: 448.0, torch.float8_e5m2: 57344.0, torch.int8: 127.0}


def _stored_at_scale(t, s, qdtype, cfg=None):
    """Vrai si les poids stockes sont DEJA a leur echelle reelle, un weight_scale etant
    fourni en plus -- a ne pas appliquer. Porte de crispz-klein 1.34.1.

    Un FP8 'scaled' normal stocke poids / echelle: il REMPLIT la plage du format (448 en
    E4M3) et max|stocke| / (echelle x plage) vaut 1 / echelle (71 a 1 691 sur les 16
    fichiers FP8/INT8 de la bibliotheque). kleinFinalcutFP16FP8_comfyQuant stocke ses
    poids tels quels (0,375 sur 448) et fournit amax / 448 quand meme: rapport 1,03.
    Appliquer l'echelle rendait chaque poids 1 200 a 1 700 fois trop petit, et l'image
    sortait en bruit. Les echelles MX (uint8 = exposant E8M0) ne sont jamais concernees."""
    rng = _QUANT_RANGE.get(qdtype)
    fmt = str((cfg or {}).get("format", "")).lower()
    if rng is None or s.dtype == torch.uint8 or fmt.startswith("mx"):
        return False
    smax = float(s.detach().float().abs().max())
    if smax <= 0.0:
        return False
    amax = float(t.detach().float().abs().max())
    # 1. plage peu utilisee (un fichier normal la remplit)...
    if amax >= rng / 4:
        return False
    # 2. ... ET l'echelle decrit exactement les valeurs stockees: rapport ~1.
    ratio = amax / (smax * rng)
    return 0.5 <= ratio < 2.0


_PRESCALED = {}      # cle fichier -> bool (lu une fois par fichier et par session)


def _source_prescaled(src):
    """Le fichier source stocke-t-il ses poids deja a l'echelle ? Lu sur le PLUS PETIT
    tenseur quantifie qui porte une echelle: quelques Ko a lire. Toute erreur = False:
    la cle de cache ne change alors pas."""
    try:
        fk = _file_key(src)
    except OSError:
        return False
    if fk in _PRESCALED:
        return _PRESCALED[fk]
    res = False
    try:
        hdr = _safetensors_header(src)
        best = None
        for k, v in hdr.items():
            if not (isinstance(v, dict) and k.endswith(".weight")):
                continue
            if str(v.get("dtype", "")).upper() not in ("F8_E4M3", "F8_E5M2", "I8"):
                continue
            sk = next((c for c in (k + "_scale", k[:-len(".weight")] + ".scale_weight")
                       if c in hdr), None)
            if sk is None:
                continue
            n = 1
            for d in v.get("shape") or [1]:
                n *= int(d)
            if best is None or n < best[0]:
                best = (n, k, sk)
        if best:
            from safetensors import safe_open
            with safe_open(src, framework="pt", device="cpu") as f:
                t = f.get_tensor(best[1])
                s = f.get_tensor(best[2])
            res = _stored_at_scale(t, s, t.dtype)
    except Exception as e:
        _dbg(f"prescaled check failed on {os.path.basename(src)}: {e}")
        res = False
    _PRESCALED[fk] = res
    return res


def _dequant_cache_path(src, legacy=False):
    """Chemin du bf16 cache pour un checkpoint source. La cle inclut taille+mtime:
    un fichier remplace (meme nom) ne reutilise jamais l'ancien cache.

    Un fichier stocke deja a l'echelle (_source_prescaled) change de cle: son ancien
    cache a ete ecrit par le chargeur qui appliquait l'echelle a tort. Les autres
    fichiers gardent leur cle et leur cache. legacy=True rend l'ancienne cle."""
    d = _dequant_cache_dir()
    if not d:
        return None
    try:
        p, size, mtime = _file_key(src)
    except OSError:
        return None
    tag = "bf16"
    if not legacy and _source_prescaled(src):
        tag = "bf16-prescaled"
    h = hashlib.sha1(f"{p.lower()}|{size}|{mtime}|{tag}".encode("utf-8")).hexdigest()[:16]
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
        # L'ancien cache de CE fichier, ecrit sous l'ancienne cle (poids faux, cf.
        # _dequant_cache_path): remplace, donc supprime.
        old = _dequant_cache_path(src, legacy=True)
        if old and os.path.abspath(old) != os.path.abspath(dst) and os.path.isfile(old):
            try:
                ogb = os.path.getsize(old) / 1024**3
                os.remove(old)
                _log(f"dequant cache: removed the stale {os.path.basename(old)} "
                     f"({ogb:.1f} GB, written by the loader that applied the scale twice)")
            except OSError as e:
                _dbg(f"stale dequant cache not removed {old}: {e}")
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
    # comfy-quants declare le schema soit en blobs PAR TENSEUR (X.comfy_quant),
    # soit CENTRALEMENT dans __metadata__._quantization_metadata (variante
    # StableYogi: {"layers": {"blocks...": {"format": "int8_tensorwise",
    # "convrot": true, "convrot_groupsize": 256}}}). Ignorer cette variante
    # laisse la rotation en place -> poids en bruit total (observe sur les
    # INT8 Krea 2; meme format cote Z-Image). Les blobs par tenseur gagnent.
    try:
        qm = json.loads((hdr.get("__metadata__") or {}).get(
            "_quantization_metadata") or "{}")
        for lk, lv in (qm.get("layers") or {}).items():
            if isinstance(lv, dict):
                qcfg[lk] = lv
                qcfg["model.diffusion_model." + lk] = lv   # variante AIO prefixee
        if qcfg:
            _dbg(f"quantization metadata: {len(qm.get('layers') or {})} layer(s) "
                 "declared in header")
    except Exception as e:
        _dbg(f"_quantization_metadata unreadable: {e}")
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
    n_dq = n_rot = n_pre = 0
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
            qdt = t.dtype
            t = t.to(torch.float32)
            cfg0 = qcfg.get(k[:-len(".weight")]) if k.endswith(".weight") else None
            if s is not None and _stored_at_scale(t, s, qdt, cfg0):
                s = None                     # deja a l'echelle: cf. _stored_at_scale
                n_pre += 1
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
         + (f", {n_pre} already stored at scale: weight_scale NOT applied" if n_pre else "")
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
    l'offload effectif ('none' -> 'model') et exige un reload complet."""
    off = OFFLOAD_MODE
    if off == "auto":
        off = _resolve_auto()   # test VRAM (memoise + profil cache) -> mode concret
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
        if _effective_loras():
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
    global _TEXT_ENCODER_ACTIVE
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
    # Encodeur de remplacement: verifie a la config puis charge avec la classe que donne
    # le model_index.json du repo (Qwen3Model). Un encodeur qui ne convient pas (repo
    # change depuis le choix, dossier disparu, chargement en echec) est ecarte AVEC une
    # ligne de log et l'encodeur du repo tourne: une generation ne plante jamais pour ca,
    # et les metadonnees le disent (text_encoder_not_applied). img2img / inpaint
    # derivent du base via from_pipe et reprennent donc cet encodeur.
    _TEXT_ENCODER_ACTIVE = ""
    if TEXT_ENCODER:
        try:
            _why = _text_encoder_problem(TEXT_ENCODER)
        except Exception as e:        # la verification elle-meme echoue: on ecarte
            _why = f"it could not be checked ({type(e).__name__}: {e})"
        if not _why:
            try:
                kwargs["text_encoder"] = _load_monitor(
                    f"text encoder {_encoder_label(TEXT_ENCODER)}",
                    lambda: _load_text_encoder(TEXT_ENCODER))
                _TEXT_ENCODER_ACTIVE = TEXT_ENCODER
            except Exception as e:
                _why = f"it failed to load ({type(e).__name__}: {e})"
        if _why:
            _log(f"text encoder {_encoder_label(TEXT_ENCODER)} NOT used: {_why}. "
                 f"{BASE_REPO}'s own encoder runs instead; the image metadata says so "
                 f"(text_encoder_not_applied).")
        else:
            _log(f"text encoder: {_encoder_label(TEXT_ENCODER)} replaces {BASE_REPO}'s "
                 f"own (tokenizer, VAE and transformer unchanged)")
    _off_label = (f"auto->{_resolve_auto()}" if OFFLOAD_MODE == "auto" else OFFLOAD_MODE)
    _log(f"loading Z-Image base: {BASE_REPO} (offload={_off_label}, dtype=bf16) ... "
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
    if _effective_loras():
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
    _base_off = _resolve_auto() if OFFLOAD_MODE == "auto" else OFFLOAD_MODE
    if _off != _base_off:
        _log(f"GGUF base: offload '{_base_off}' forced to '{_off}' (a GGUF does not "
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
    _omni_off = _resolve_auto() if OFFLOAD_MODE == "auto" else OFFLOAD_MODE
    _log(f"loading Z-Image Omni: {repo} (offload={_omni_off}) ...")
    t0 = time.time()
    pipe = _load_monitor(f"Z-Image Omni {repo}",
                         lambda: ZImageOmniPipeline.from_pretrained(repo, torch_dtype=DTYPE))
    # Attention slicing pose par appel via _set_slicing (cf. _ensure_base).
    if DEVICE == "cuda" and _omni_off == "model":
        pipe.enable_model_cpu_offload()
    elif DEVICE == "cuda" and _omni_off == "sequential":
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
    pipe = get_pipe("txt2img")
    w = round_to_multiple(int(width))
    h = round_to_multiple(int(height))
    _log(f"txt2img: {w}x{h}, {int(steps)} steps, guidance {GUIDANCE:.1f} ...")
    _dbg(f"txt2img seed={seed} dtype=bf16 device={DEVICE} offload={OFFLOAD_MODE} "
         f"transformer={'single-file' if ZIMAGE_TRANSFORMER else 'repo'}")
    if DEVICE == "cuda":
        _dbg(f"VRAM before: alloc={torch.cuda.memory_allocated()/1024**3:.2f} Go")
    _progress(0.1, f"Generating {w}x{h} ({int(steps)} steps)...")
    t0 = time.time()
    # Deux tentatives maxi: si la garde VRAM declenche au 1er step (mode 'none'
    # trop optimiste), _consume_vram_downgrade bascule en 'model' et on rejoue.
    for _attempt in (0, 1):
        _set_slicing(pipe, max(w, h))
        img = _pipe_guarded(
            pipe,
            prompt=prompt or "",
            negative_prompt=(negative_prompt or None),
            width=w, height=h,
            num_inference_steps=int(steps),
            guidance_scale=GUIDANCE,
            generator=_make_generator(seed),
        )
        if not _consume_vram_downgrade():
            break
        pipe = get_pipe("txt2img")   # recharge avec l'offload retrograde
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
    # Deux tentatives maxi: garde VRAM au 1er step (cf. generate), puis retry en 'model'.
    for _attempt in (0, 1):
        _set_slicing(pipe, max(w, h))   # a reposer sur le pipe recharge du retry
        out = _pipe_guarded(
            pipe,
            prompt=prompt or "",
            image=image,
            width=w, height=h,
            strength=float(denoise),
            num_inference_steps=int(steps),
            guidance_scale=GUIDANCE,
            generator=_make_generator(seed),
        )
        if not _consume_vram_downgrade():
            break
        pipe = get_pipe("img2img")   # recharge avec l'offload retrograde
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


def _refine_tiled(pipe, image, denoise, steps, prompt, seed, tile, overlap):
    """Passe Z-Image en tuiles avec recomposition feather (facon Ultimate SD Upscale).
    Plafonne le pic VRAM (une tuile a la fois) et permet le 4K+ sans coutures.
    Memes rampe lineaire + overlap-add que esrgan_upscale, mais a scale 1 sur PIL."""
    w, h = image.size
    tile = round_to_multiple(tile)                       # multiple de 16 pour le VAE
    overlap = max(0, min(int(overlap), tile - 16))
    if w <= tile and h <= tile:
        # Une seule tuile = image entiere -> pas de duplication possible: denoise demande.
        return _refine_whole(pipe, image, denoise, steps, prompt, seed)
    # Anti-duplication 1: prompt vide par tuile (le prompt global decrit toute la compo).
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
            out = _refine_whole(pipe, crop, denoise, steps, prompt, seed)
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
            rt = _pick_refine_tile(rw, rh, int(refine_overlap) or 64)
            _log(f"refine: image {rw}x{rh} > {_AUTO_TILE_ABOVE}px -> auto-tiling (tile {rt}) "
                 "pour eviter le pic VRAM (regles: auto_refine_tile_above, auto_refine_tile)")
        if rt > 0:
            out = _refine_tiled(pipe, img, denoise, steps, prompt, seed,
                                rt, int(refine_overlap) or 64)
        else:
            _log(f"Z-Image refine: whole image {rw}x{rh}, denoise {float(denoise):.2f}, "
                 f"{int(steps)} steps ...")
            _progress(0.5, f"Z-Image refine {rw}x{rh}...")
            out = _refine_whole(pipe, img, denoise, steps, prompt, seed)
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


# Hash git court du build qui tourne, fige au demarrage. Ecrit dans chaque sidecar:
# pendant la chasse au bug mosaique, impossible de savoir si un rendu venait du code
# corrige ou d'un process pas encore redemarre -- cette cle tranche.
def _read_build():
    try:
        import subprocess
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True,
            text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


_BUILD = _read_build()


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
    # Encodeur de remplacement: celui qui a REELLEMENT tourne, par son NOM de dossier
    # (jamais le chemin, qui finirait dans les PNG partages). Demande mais ecarte au
    # chargement = l'image vient de l'encodeur du repo de base, et on nomme a part celui
    # qui n'a pas servi. Omni charge un modele separe avec son propre encodeur: rien a dire.
    if mode != "omni":
        if _TEXT_ENCODER_ACTIVE:
            m["text_encoder"] = _encoder_label(_TEXT_ENCODER_ACTIVE)
        elif TEXT_ENCODER:
            m["text_encoder_not_applied"] = _encoder_label(TEXT_ENCODER)
    # Liste EFFECTIVE (slots + tags <lora:...> du prompt): c'est ce qui a reellement
    # tourne -> indispensable pour reproduire le rendu depuis le sidecar.
    _eff_loras = _effective_loras()
    if _eff_loras:
        m["loras"] = [f"{os.path.basename(p)}@{w}" for p, w in _eff_loras]
    # Etat runtime qui change le CHEMIN d'execution: indispensable pour dater/attribuer
    # une corruption depuis les sidecars seuls (bug mosaique ouvert: sans ces cles il a
    # fallu reconstituer la config de chaque rendu de memoire). Import paresseux: le
    # detailer importe cz_pipeline dans ses fonctions, jamais l'inverse en tete de module.
    m["offload"] = f"{OFFLOAD_MODE}/{_effective_offload()}"
    m["build"] = _BUILD
    try:
        import cz_detailer
        m["detail_faces"] = bool(cz_detailer.DETAILER_ENABLED)
        m["detail_hands"] = bool(cz_detailer.HAND_ENABLED)
        m["hand_device"] = cz_detailer._HAND_DEVICE
    except Exception:
        pass
    if extra:
        m.update(extra)
    return m
