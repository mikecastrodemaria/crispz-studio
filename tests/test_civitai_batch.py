"""Unit tests for the batch CivitAI enrichment core (no network).

Run:  .venv/Scripts/python tests/test_civitai_batch.py
"""
import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_civitai            # noqa: E402
import cz_civitai_batch as B  # noqa: E402


def _mk(dirpath, name, sidecar=False, preview=False):
    p = os.path.join(dirpath, name)
    with open(p, "wb") as f:
        f.write(b"x")
    stem = os.path.splitext(p)[0]
    if sidecar:
        with open(stem + ".civitai.json", "w", encoding="utf-8") as f:
            json.dump({"modelId": 1, "versionId": 2}, f)
    if preview:
        with open(stem + ".preview.png", "wb") as f:
            f.write(b"\x89PNG")
    return p


def test_apply_shard_partition():
    files = [f"f{i}" for i in range(10)]
    s0 = B._apply_shard(files, "1/3")
    s1 = B._apply_shard(files, "2/3")
    s2 = B._apply_shard(files, "3/3")
    assert sorted(s0 + s1 + s2) == sorted(files)          # partition complete
    assert not (set(s0) & set(s1)) and not (set(s1) & set(s2))  # disjointe
    assert B._apply_shard(files, None) == files
    assert B._apply_shard(files, "9/3") == files          # invalide -> inchange


def test_needs_enrich():
    d = tempfile.mkdtemp()
    a = _mk(d, "a.safetensors")                            # rien -> a besoin
    b = _mk(d, "b.safetensors", sidecar=True, preview=True)  # complet -> pas besoin
    c = _mk(d, "c.safetensors", sidecar=True)              # sidecar sans preview -> besoin
    assert B._needs_enrich(a, overwrite=False) is True
    assert B._needs_enrich(b, overwrite=False) is False
    assert B._needs_enrich(c, overwrite=False) is True
    assert B._needs_enrich(b, overwrite=True) is True      # force -> toujours


def test_collect_files_and_shard():
    d = tempfile.mkdtemp()
    for n in ("m1.safetensors", "m2.safetensors", "note.txt"):
        _mk(d, n)
    os.makedirs(os.path.join(d, "sub"))
    _mk(os.path.join(d, "sub"), "m3.safetensors")
    files = B.collect_files("loras", loras_dir=d)
    names = sorted(os.path.basename(f) for f in files)
    assert names == ["m1.safetensors", "m2.safetensors", "m3.safetensors"]  # recursif, .txt exclu


def test_enrich_accounting(monkeypatch):
    d = tempfile.mkdtemp()
    a = _mk(d, "a.safetensors")                            # -> enrich (mock success)
    _mk(d, "b.safetensors", sidecar=True, preview=True)    # -> skip + refresh (updated)
    c = _mk(d, "c.safetensors")                            # -> enrich (mock failure)

    def fake_fetch(path, api_key=None, overwrite=False, progress=None, check_update=True):
        if os.path.basename(path) == "c.safetensors":
            return {"success": False, "message": "unknown hash"}
        return {"success": True, "message": "ok", "update_available": False}

    def fake_refresh(path, api_key=None):
        return {"success": True, "update_available": True}   # b a une nouvelle version

    monkeypatch.setattr(cz_civitai, "fetch_civitai_for_model", fake_fetch)
    monkeypatch.setattr(cz_civitai, "refresh_update_flag", fake_refresh)

    seen = []
    files = B.collect_files("loras", loras_dir=d)
    s = B.enrich(files, sleep=0, only_missing=True, check_updates=True,
                 progress=lambda i, n, name, ph, tx: seen.append((i, n)))
    assert s["total"] == 3
    assert s["enriched"] == 1 and s["failed"] == 1          # a ok, c failed
    assert s["skipped"] == 1 and s["updated"] == 1          # b skip + newer version
    assert any(w.startswith("c.safetensors") for w in s["warnings"])
    assert len(seen) >= 3                                    # progress appele par fichier
    # a et c sans reseau reel: le mock a bien ete utilise (pas d'appel HTTP)


def test_resolve_dirs_arg_priority():
    loras, cks = B.resolve_dirs(loras_dir="X:/L", checkpoints_dir="X:/C")
    assert loras.replace("\\", "/").endswith("X:/L".replace("\\", "/")) or loras.endswith("L")
    assert any(c.endswith("C") for c in cks)


if __name__ == "__main__":
    # mini-shim monkeypatch (sans pytest)
    class _MP:
        def __init__(self): self._undo = []
        def setattr(self, obj, name, val):
            self._undo.append((obj, name, getattr(obj, name)))
            setattr(obj, name, val)
        def undo(self):
            for obj, name, val in reversed(self._undo):
                setattr(obj, name, val)
    for fn in (test_apply_shard_partition, test_needs_enrich, test_collect_files_and_shard,
               test_resolve_dirs_arg_priority):
        fn(); print(f"OK {fn.__name__}")
    mp = _MP()
    try:
        test_enrich_accounting(mp); print("OK test_enrich_accounting")
    finally:
        mp.undo()
    print("All civitai_batch tests passed.")
