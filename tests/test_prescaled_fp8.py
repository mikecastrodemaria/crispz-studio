"""Poids FP8/INT8 stockes DEJA a l'echelle: le weight_scale fourni ne s'applique pas.
Porte de crispz-klein 1.34.1 (kleinFinalcutFP16FP8_comfyQuant rendait du bruit).

Run:  .venv/Scripts/python tests/test_prescaled_fp8.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors.torch import save_file

import cz_pipeline as P

torch.manual_seed(0)
E4 = torch.float8_e4m3fn
# Cles qui passent la garde d'architecture Z-Image du chargeur (cap_embedder, noise_refiner).
K1, K2 = "cap_embedder.probe.weight", "noise_refiner.probe.weight"


def _pair(n=64):
    w = torch.randn(n, n) * 0.02
    return w, (w.abs().max() / 448.0).reshape(())


def test_the_detector_separates_the_layouts():
    w, s = _pair()
    assert not P._stored_at_scale((w / s).to(E4).float(), s, E4)       # FP8 normal
    assert P._stored_at_scale(w.to(E4).float(), s, E4)                  # deja a l'echelle
    assert not P._stored_at_scale((w * 50).to(E4).float(), torch.tensor(0.5), E4)
    s8 = (w.abs().max() / 127.0).reshape(())
    q8 = torch.round(w / s8).clamp(-127, 127).to(torch.int8)
    assert not P._stored_at_scale(q8.float(), s8, torch.int8)          # INT8 normal
    full = torch.randint(-127, 128, (4, 3), dtype=torch.int8)
    full[0, 0] = 127
    assert not P._stored_at_scale(full.float(), torch.full((4, 1), 0.9), torch.int8)
    assert not P._stored_at_scale(w.to(E4).float(), torch.tensor([120], dtype=torch.uint8), E4)
    print("OK test_the_detector_separates_the_layouts")


def _tiny(path, prescaled, both=False):
    w1, s1 = _pair(32)
    w2, s2 = _pair(32)
    save_file({K1: (w1 if prescaled else w1 / s1).to(E4), K1 + "_scale": s1.float(),
               K2: (w2 if both else w2 / s2).to(E4), K2 + "_scale": s2.float()}, path)
    return w1, w2


def test_the_loader_reads_a_file_mixing_both_layouts():
    p = os.path.join(tempfile.mkdtemp(), "mixed.safetensors")
    w1, w2 = _tiny(p, prescaled=True)
    out = P._load_dequant_state_dict(p)
    for k, w in ((K1, w1), (K2, w2)):
        rel = ((out[k].float() - w).norm() / w.norm()).item()
        assert rel < 0.1, (k, rel)
    print("OK test_the_loader_reads_a_file_mixing_both_layouts")


def test_the_cache_key_changes_only_for_prescaled_files():
    d, cache = tempfile.mkdtemp(), tempfile.mkdtemp()
    pre, reg = os.path.join(d, "pre.safetensors"), os.path.join(d, "reg.safetensors")
    _tiny(pre, prescaled=True, both=True)
    _tiny(reg, prescaled=False)
    old = P._DQ_CACHE_CFG
    try:
        P._DQ_CACHE_CFG = cache
        assert P._dequant_cache_path(pre) != P._dequant_cache_path(pre, legacy=True)
        assert P._dequant_cache_path(reg) == P._dequant_cache_path(reg, legacy=True)
        stale = P._dequant_cache_path(pre, legacy=True)
        with open(stale, "wb") as f:
            f.write(b"x")
        P._dequant_cache_store(pre, {"x": torch.zeros(1)})
        assert os.path.isfile(P._dequant_cache_path(pre))
        assert not os.path.exists(stale), "cache faux laisse sur le disque"
    finally:
        P._DQ_CACHE_CFG = old
    print("OK test_the_cache_key_changes_only_for_prescaled_files")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("All prescaled-FP8 tests passed.")
