"""Unit tests for the dequant disk cache, the GPU-busy guard and the checkpoint
format badges. Synthetic safetensors only (a few KB), no GPU, no model download.

Run:  .venv/Scripts/python tests/test_dequant_cache.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import cz_pipeline as czp  # noqa: E402

TMP = tempfile.mkdtemp(prefix="cz_dqcache_")
_W = "model.diffusion_model.layers.0.feed_forward.w1.weight"


def _fp8_ckpt(name, scale=2.5):
    """Checkpoint FP8 'scaled' minimal (marqueur Z-Image inclus)."""
    w = torch.randn(4, 4).to(torch.float8_e4m3fn)
    p = os.path.join(TMP, name)
    save_file({_W: w, _W + "_scale": torch.tensor(scale)}, p)
    return p


def _with_cache(dirname):
    """Force le cache de dequant sur un dossier de test (et le rend actif)."""
    czp._DQ_CACHE_CFG = os.path.join(TMP, dirname)
    return czp._DQ_CACHE_CFG


def test_cache_path_is_stable_and_keyed_on_content():
    _with_cache("c1")
    p = _fp8_ckpt("a.safetensors")
    k1 = czp._dequant_cache_path(p)
    assert k1 and k1.endswith(".safetensors")
    assert czp._dequant_cache_path(p) == k1, "meme fichier -> meme cle"
    # meme NOM mais contenu different (taille/mtime) -> autre cle: pas de faux hit
    os.remove(p)
    p2 = _fp8_ckpt("a.safetensors", scale=3.0)
    save_file({_W: torch.randn(8, 8).to(torch.float8_e4m3fn)}, p2)
    assert czp._dequant_cache_path(p2) != k1


def test_store_then_reload_roundtrip():
    _with_cache("c2")
    p = _fp8_ckpt("b.safetensors")
    sd = czp._load_dequant_state_dict(p)
    czp._dequant_cache_store(p, sd)
    cached = czp._dequant_cache_path(p)
    assert os.path.isfile(cached), "le cache doit exister apres store"
    from safetensors.torch import load_file
    back = load_file(cached)
    assert set(back) == set(sd)
    assert torch.allclose(back[_W].float(), sd[_W].float())
    assert back[_W].dtype == czp.DTYPE, "le cache stocke du bf16, pas du FP8"
    # pas de .tmp laisse derriere (ecriture atomique)
    assert not os.path.exists(cached + ".tmp")


def test_cache_disabled_by_config():
    old = czp._DQ_CACHE_CFG
    try:
        czp._DQ_CACHE_CFG = "off"
        assert czp._dequant_cache_dir() is None
        assert czp._dequant_cache_path(_fp8_ckpt("c.safetensors")) is None
        czp._dequant_cache_store(_fp8_ckpt("d.safetensors"), {})   # no-op silencieux
    finally:
        czp._DQ_CACHE_CFG = old


def test_prune_evicts_least_recently_used():
    d = _with_cache("c3")
    os.makedirs(d, exist_ok=True)
    # 3 fichiers de ~1 Mo, atime croissant
    import time
    paths = []
    for i in range(3):
        fp = os.path.join(d, f"e{i}.safetensors")
        save_file({"w": torch.zeros(256, 512)}, fp)     # 512 Ko
        os.utime(fp, (1_000_000 + i * 1000, 1_000_000 + i * 1000))
        paths.append(fp)
    old_cap = czp.DEQUANT_CACHE_MAX_GB
    try:
        czp.DEQUANT_CACHE_MAX_GB = 1.0 / 1024        # 1 Mo -> il faut evincer
        czp._dequant_cache_prune()
        left = [p for p in paths if os.path.exists(p)]
        assert paths[0] not in left, "le moins recemment utilise part en premier"
        assert paths[-1] in left, "le plus recent survit"
    finally:
        czp.DEQUANT_CACHE_MAX_GB = old_cap


def test_prune_keeps_the_entry_just_written():
    d = _with_cache("c4")
    os.makedirs(d, exist_ok=True)
    fp = os.path.join(d, "fresh.safetensors")
    save_file({"w": torch.zeros(256, 512)}, fp)
    old_cap = czp.DEQUANT_CACHE_MAX_GB
    try:
        czp.DEQUANT_CACHE_MAX_GB = 1e-9              # tout devrait sauter
        czp._dequant_cache_prune(keep=fp)
        assert os.path.exists(fp), "l'entree qu'on vient d'ecrire n'est jamais evincee"
    finally:
        czp.DEQUANT_CACHE_MAX_GB = old_cap


def test_header_cache_avoids_rereads():
    p = _fp8_ckpt("h.safetensors")
    czp._HDR_CACHE.clear()
    h1 = czp._safetensors_header(p)
    assert czp._file_key(p) in czp._HDR_CACHE
    # 2e lecture: meme OBJET renvoye -> le disque n'a pas ete retape (le listing,
    # la detection de format et le badge lisent le meme en-tete).
    assert czp._safetensors_header(p) is h1
    # fichier remplace (mtime/taille) -> la cle change, l'en-tete est relu
    save_file({_W: torch.zeros(16, 16, dtype=torch.bfloat16)}, p)
    os.utime(p, (2_000_000, 2_000_000))
    h2 = czp._safetensors_header(p)
    assert h2 is not h1 and h2[_W]["dtype"] == "BF16"


def test_checkpoint_badge():
    _with_cache("c5")
    old_dir = czp.CHECKPOINTS_DIR
    try:
        czp.CHECKPOINTS_DIR = TMP
        p = _fp8_ckpt("badge_fp8.safetensors")
        b = czp.checkpoint_badge(os.path.basename(p))
        assert b.startswith("FP8->bf16") and "GB" in b and "slow 1st load" in b
        assert b.isascii(), "badge ASCII: il finit dans des logs console cp1252"
        # une fois le bf16 en cache, le badge le dit
        czp._dequant_cache_store(p, czp._load_dequant_state_dict(p))
        assert "cached" in czp.checkpoint_badge(os.path.basename(p))
        # BF16 simple
        bf = os.path.join(TMP, "badge_bf16.safetensors")
        save_file({_W: torch.zeros(4, 4, dtype=torch.bfloat16)}, bf)
        assert czp.checkpoint_badge("badge_bf16.safetensors").startswith("BF16")
        # GGUF -> quant lu dans le nom de fichier
        g = os.path.join(TMP, "z_image_turbo-Q6_K.gguf")
        with open(g, "wb") as f:
            f.write(b"GGUF" + b"\0" * 64)
        assert czp.checkpoint_badge("z_image_turbo-Q6_K.gguf").startswith("GGUF Q6_K")
        # repo HF (pas un fichier) -> pas de badge
        assert czp.checkpoint_badge("Tongyi-MAI/Z-Image-Turbo") == ""
    finally:
        czp.CHECKPOINTS_DIR = old_dir


def test_gpu_busy_warning_thresholds():
    old_warn, old_dev = czp.GPU_BUSY_WARN_GB, czp.DEVICE
    old_fn = czp.gpu_foreign_vram_gb
    try:
        czp.DEVICE = "cuda"
        czp.GPU_BUSY_WARN_GB = 2.0
        czp.gpu_foreign_vram_gb = lambda: 0.3
        assert czp.gpu_busy_warning() == "", "sous le seuil -> pas d'alerte"
        czp.gpu_foreign_vram_gb = lambda: 17.4
        msg = czp.gpu_busy_warning()
        assert "17.4 GB" in msg and "shared RAM" in msg
        czp.GPU_BUSY_WARN_GB = 0                      # garde desactivee
        assert czp.gpu_busy_warning() == ""
    finally:
        czp.GPU_BUSY_WARN_GB, czp.DEVICE = old_warn, old_dev
        czp.gpu_foreign_vram_gb = old_fn


def test_gpu_foreign_vram_is_zero_on_cpu():
    old = czp.DEVICE
    try:
        czp.DEVICE = "cpu"
        assert czp.gpu_foreign_vram_gb() == 0.0
    finally:
        czp.DEVICE = old


if __name__ == "__main__":
    for fn in (test_cache_path_is_stable_and_keyed_on_content,
               test_store_then_reload_roundtrip, test_cache_disabled_by_config,
               test_prune_evicts_least_recently_used,
               test_prune_keeps_the_entry_just_written,
               test_header_cache_avoids_rereads, test_checkpoint_badge,
               test_gpu_busy_warning_thresholds, test_gpu_foreign_vram_is_zero_on_cpu):
        fn()
        print(f"OK {fn.__name__}")
    print("All dequant-cache / GPU-guard / badge tests passed.")
