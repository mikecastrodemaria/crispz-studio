"""Unit tests for the parallel thumbnail rebuild (Asset Browser 'Rebuild ALL thumbnails').

Run:  .venv/Scripts/python tests/test_thumbs.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

import cz_assetbrowser as AB  # noqa: E402


def _outputs_dir(n=6):
    d = tempfile.mkdtemp()
    day = os.path.join(d, "2026-07-15")
    os.makedirs(day)
    for i in range(n):
        Image.new("RGB", (600, 400), (i * 30 % 255, 60, 120)).save(os.path.join(day, f"i{i}.png"))
    return d


def _models_dir(n=4, previews=True):
    md = tempfile.mkdtemp()
    for i in range(n):
        with open(os.path.join(md, f"m{i}.safetensors"), "wb") as f:
            f.write(b"x")
        if previews:
            Image.new("RGB", (300, 300), (10, i * 40 % 255, 90)).save(
                os.path.join(md, f"m{i}.preview.png"))
    return md


def test_force_rebuilds_everything_then_skips():
    d = _outputs_dir(6)
    jobs = AB._thumb_jobs_for("outputs", d)
    assert len(jobs) == 6
    r1 = AB._ab_gen_thumbs(jobs, 128, 85, force=True)
    assert r1 == {"total": 6, "made": 6, "skipped": 0, "failed": 0}
    r2 = AB._ab_gen_thumbs(jobs, 128, 85, force=False)      # a jour -> tout saute
    assert r2["skipped"] == 6 and r2["made"] == 0
    r3 = AB._ab_gen_thumbs(jobs, 128, 85, force=True)       # force -> tout refait
    assert r3["made"] == 6


def test_progress_is_called_once_per_file():
    d = _outputs_dir(5)
    jobs = AB._thumb_jobs_for("outputs", d)
    seen = []
    AB._ab_gen_thumbs(jobs, 128, 85, force=True, progress=lambda i, n, name: seen.append((i, n)))
    assert len(seen) == 5
    assert all(n == 5 for _i, n in seen)
    assert sorted(i for i, _n in seen) == [1, 2, 3, 4, 5]   # compteur strictement croissant


def test_parallel_and_serial_give_the_same_result():
    d = _outputs_dir(8)
    jobs = AB._thumb_jobs_for("outputs", d)
    r_par = AB._ab_gen_thumbs(jobs, 128, 85, force=True, workers=4)
    r_ser = AB._ab_gen_thumbs(jobs, 128, 85, force=True, workers=1)
    assert r_par["made"] == r_ser["made"] == 8 and r_par["failed"] == r_ser["failed"] == 0
    for _src, tp in jobs:
        assert os.path.isfile(tp)


def test_model_jobs_need_a_preview():
    # Hermetique: le layout des miniatures depend du cache configure (prefs machine).
    # On force les deux modes explicitement au lieu de subir preferences.json.
    out = tempfile.mkdtemp()
    md = _models_dir(4, previews=True)
    old = AB._prefs.get("ab_cache_dir")
    try:
        AB._prefs["ab_cache_dir"] = "output"          # ancien layout a cote des images
        jobs = AB._thumb_jobs_for("loras", out, loras_dir=md)
        assert len(jobs) == 4
        assert all(s.endswith(".preview.png") for s, _t in jobs)
        assert all("/thumbs/loras/" in t.replace("\\", "/") for _s, t in jobs)
        AB._prefs["ab_cache_dir"] = tempfile.mkdtemp()  # cache dedie (defaut app-folder)
        jobs = AB._thumb_jobs_for("loras", out, loras_dir=md)
        assert all("/crispz-thumbs/" in t.replace("\\", "/") and "/loras/" in t.replace("\\", "/")
                   for _s, t in jobs)
        # sans preview -> rien a miniaturiser (pas d'erreur)
        md2 = _models_dir(3, previews=False)
        assert AB._thumb_jobs_for("models", out, checkpoints_dir=md2) == []
    finally:
        if old is None:
            AB._prefs.pop("ab_cache_dir", None)
        else:
            AB._prefs["ab_cache_dir"] = old


def test_missing_dir_is_not_an_error():
    out = tempfile.mkdtemp()
    assert AB._thumb_jobs_for("loras", out, loras_dir="D:/nope_not_here") == []
    assert AB._ab_gen_thumbs([], 128, 85, force=True)["total"] == 0


def test_broken_source_counts_as_failed_not_crash():
    d = _outputs_dir(2)
    day = os.path.join(d, "2026-07-15")
    with open(os.path.join(day, "broken.png"), "wb") as f:
        f.write(b"not an image")          # PIL va echouer dessus
    jobs = AB._thumb_jobs_for("outputs", d)
    r = AB._ab_gen_thumbs(jobs, 128, 85, force=True)
    assert r["failed"] == 1 and r["made"] == 2, r      # le lot continue malgre l'echec


def test_rebuild_thumbs_returns_summary_with_kind():
    d = _outputs_dir(3)
    r = AB.rebuild_thumbs("outputs", d, force=True)
    assert r["kind"] == "outputs" and r["made"] == 3


if __name__ == "__main__":
    for fn in (test_force_rebuilds_everything_then_skips, test_progress_is_called_once_per_file,
               test_parallel_and_serial_give_the_same_result, test_model_jobs_need_a_preview,
               test_missing_dir_is_not_an_error, test_broken_source_counts_as_failed_not_crash,
               test_rebuild_thumbs_returns_summary_with_kind):
        fn()
        print(f"OK {fn.__name__}")
    print("All thumbnail tests passed.")
