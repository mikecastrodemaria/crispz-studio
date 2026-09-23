"""Hardware-aware CPU-offload resolution for the crispz family.

'auto' (the default offload mode) resolves to a concrete mode -- none / model /
sequential -- at model-load time, from a real free-VRAM test:

    free VRAM (torch.cuda.mem_get_info, counts every process on the GPU)
        >= model footprint + activation margin  ->  'none' (fastest)
        otherwise                               ->  'model' / 'sequential'

Why a test and not a static table: on Windows (WDDM) the driver's *CUDA Sysmem
Fallback* silently spills an over-committed model to shared RAM over PCIe.
Nothing errors out -- renders just become 50-100x slower (measured: a 52 s
txt2img taking 5+ min with zero progress). Defaulting to the safe mode and
only promoting to 'none' when the card provably has room makes a fresh install
work on any GPU without manual settings.

The verdict is cached in a small JSON profile (cache/hw_profile.json) keyed by
GPU + torch/CUDA build + model + dtype, so the test runs once per combination.
A runtime downgrade (see the app's VRAM guard) is recorded in the same profile
so the next boot starts directly in the safe mode.

This module is STANDALONE (stdlib + torch only, torch imported lazily) and is
vendored as-is into every crispz-family app. App-specific knowledge -- the
model footprint, the profile path -- is passed in by the caller.
"""
import json
import os
import time

OFFLOAD_AUTO = "auto"
OFFLOAD_CONCRETE = ("none", "model", "sequential")
# Free VRAM (GB) under which a running generation counts as saturated: the next
# allocation spills to shared RAM. 0.5 GB of slack, same as the spec.
SATURATION_FREE_GB = 0.5


def _torch(torch_mod=None):
    """The torch module (lazy import). Tests inject a stub via torch_mod."""
    if torch_mod is not None:
        return torch_mod
    import torch
    return torch


def cuda_mem_gb(torch_mod=None):
    """(free_gb, total_gb) of CUDA device 0, or None (no CUDA / query failed).
    mem_get_info is the DRIVER's view: it counts every process on the GPU,
    which is exactly what matters on a shared card."""
    try:
        t = _torch(torch_mod)
        if not t.cuda.is_available():
            return None
        free, total = t.cuda.mem_get_info()
        return free / 1024 ** 3, total / 1024 ** 3
    except Exception:
        return None


def activation_margin_gb(width=1024, height=1024):
    """VRAM headroom (GB) the denoise pass needs on top of the weights.
    Measured >= 2.5 GB at 1024x1024; activations grow with the pixel count."""
    mpx = (int(width) * int(height)) / 1048576.0
    return max(2.5, 2.5 * mpx)


def fallback_mode(total_gb):
    """Concrete mode when the free-VRAM test says 'none' does not fit.
    NOT the same as a total-VRAM recommendation table: the test already ruled
    out 'none', so >= 11 GB gets 'model' (one submodule on GPU at a time) and
    smaller cards get 'sequential'. Threshold at 11 and not 12: a '12 GB' card
    exposes ~11.6-11.9 GB."""
    return "model" if float(total_gb) >= 11 else "sequential"


# ---------------------------------------------------------------------------
# Profile cache: one JSON file, {key: {mode, reason, free_at_test, ...}}.
# ---------------------------------------------------------------------------

def gpu_signature(torch_mod=None):
    """Stable id of (GPU, software stack): profile entries die with any change.
    torch.version.cuda stands in for the driver version (no portable query);
    a driver update usually comes with sysadmin churn that warrants a re-test
    anyway, and the UI has an explicit Re-test button."""
    try:
        t = _torch(torch_mod)
        if not t.cuda.is_available():
            return "no-cuda"
        p = t.cuda.get_device_properties(0)
        return (f"{p.name}|{p.total_memory // (1024 ** 2)}MB"
                f"|torch{t.__version__}|cuda{t.version.cuda}")
    except Exception:
        return "unknown-gpu"


def profile_key(model_id, dtype, torch_mod=None):
    return f"{gpu_signature(torch_mod)}|{model_id}|{dtype}"


def load_profile(path):
    """The whole profile dict; {} on any problem (missing, corrupt, unreadable)."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_profile(path, prof):
    """Atomic write (tmp + replace): an interruption never truncates the file."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(prof, f, indent=1)
        os.replace(tmp, path)
    except Exception:
        pass   # a lost profile only costs one re-test at the next boot


def record(profile_path, model_id, dtype, entry, torch_mod=None):
    """Store/overwrite this (gpu, model, dtype)'s entry ({mode, reason, ...})."""
    if not profile_path:
        return
    prof = load_profile(profile_path)
    entry = dict(entry)
    entry.setdefault("date", time.strftime("%Y-%m-%d %H:%M:%S"))
    prof[profile_key(model_id, dtype, torch_mod)] = entry
    _save_profile(profile_path, prof)


def record_downgrade(profile_path, model_id, dtype, mode, why, torch_mod=None):
    """Runtime safety net verdict -> profile, so the next boot starts there."""
    record(profile_path, model_id, dtype,
           {"mode": mode, "reason": f"runtime downgrade: {why}", "downgraded": True},
           torch_mod=torch_mod)


# ---------------------------------------------------------------------------
# The resolver.
# ---------------------------------------------------------------------------

def resolve(requested, *, footprint_gb, width=1024, height=1024, model_id="",
            dtype="bf16", profile_path="", torch_mod=None, retest=False):
    """Resolve an offload request to a concrete mode. -> (mode, reason).

    - a concrete request ('none'/'model'/'sequential') passes through untouched
      (explicit user choice wins, resolution order is the caller's job);
    - anything else is treated as 'auto':
        profile cache hit -> cached verdict (unless retest=True);
        no CUDA          -> 'none' (MPS/CPU/DirectML keep today's behaviour);
        free >= footprint + margin -> 'none', else fallback_mode(total).
      The fresh verdict is written to the profile.
    """
    req = str(requested or "").strip().lower()
    if req in OFFLOAD_CONCRETE:
        return req, "explicit setting"

    key = profile_key(model_id, dtype, torch_mod)
    if profile_path and not retest:
        hit = load_profile(profile_path).get(key)
        if isinstance(hit, dict) and hit.get("mode") in OFFLOAD_CONCRETE:
            return hit["mode"], f"cached verdict ({hit.get('reason', 'previous test')})"

    mem = cuda_mem_gb(torch_mod)
    if mem is None:
        return "none", "non-CUDA device: offload test not applicable"
    free, total = mem
    need = float(footprint_gb) + activation_margin_gb(width, height)
    if free >= need:
        mode = "none"
        reason = f"free {free:.1f} GB >= need {need:.1f} GB"
    else:
        mode = fallback_mode(total)
        reason = f"free {free:.1f} GB < need {need:.1f} GB (total {total:.0f} GB)"
    if profile_path:
        record(profile_path, model_id, dtype,
               {"mode": mode, "reason": reason,
                "free_at_test": round(free, 2), "need": round(need, 2)},
               torch_mod=torch_mod)
    return mode, reason


def vram_saturated(torch_mod=None, threshold_gb=SATURATION_FREE_GB):
    """True when the card is pinned: any further allocation spills to shared
    RAM (Windows Sysmem Fallback -> silent 50-100x slowdown). Checked by the
    runtime safety net after the first denoise step in 'none' mode."""
    mem = cuda_mem_gb(torch_mod)
    return mem is not None and mem[0] < float(threshold_gb)
