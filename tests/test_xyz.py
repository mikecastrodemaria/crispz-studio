"""Unit tests for the X/Y/Z grid pure helpers (parse, validate, build, assemble).

Run:  .venv/Scripts/python tests/test_xyz.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

import cz_ui  # noqa: E402


def _base_vals(prompt="a red cat on a sofa"):
    vals = [None] * 36
    vals[0] = prompt
    vals[13], vals[14] = 1024, 1024
    vals[15], vals[16], vals[17], vals[18] = 8, 1, 42, 0.0
    vals[32] = "out"
    return vals


def test_parse_values():
    assert cz_ui._xyz_parse_values("4, 8, 12") == ["4", "8", "12"]
    assert cz_ui._xyz_parse_values(' "red, bright", blue ') == ["red, bright", "blue"]
    assert cz_ui._xyz_parse_values("") == []
    assert cz_ui._xyz_parse_values("  ,  ,") == []


def test_match():
    ch = ["euler", "unipc"]
    assert cz_ui._xyz_match("EULER", ch) == ("euler", None)
    assert cz_ui._xyz_match("uni", ch) == ("unipc", None)
    assert cz_ui._xyz_match("xxx", ch)[0] is None                 # introuvable
    assert cz_ui._xyz_match("e", ["euler", "exponential"])[0] is None  # ambigu


def test_validate():
    vals, ms = _base_vals(), {"loras": []}
    ok, err = cz_ui._xyz_validate_axis("Steps", ["4", "8"], vals, ms)
    assert ok == [4, 8] and err is None
    ok, err = cz_ui._xyz_validate_axis("Steps", ["4", "abc"], vals, ms)
    assert ok is None and "Steps" in err
    ok, err = cz_ui._xyz_validate_axis("Prompt S/R", ["dog", "wolf"], vals, ms)
    assert ok is None and "not found" in err                       # 'dog' absent du prompt
    ok, err = cz_ui._xyz_validate_axis("Prompt S/R", ["cat", "dog", "fox"], vals, ms)
    assert ok == ["cat", "dog", "fox"] and err is None
    ok, err = cz_ui._xyz_validate_axis("LoRA weight", ["0.5"], vals, ms)
    assert ok is None and "no active LoRA" in err
    ok, err = cz_ui._xyz_validate_axis("LoRA weight", ["0.5", "1.0"], vals,
                                       {"loras": [("D:/l.safetensors", 1.0)]})
    assert ok == [0.5, 1.0] and err is None
    ok, err = cz_ui._xyz_validate_axis("Sampler", ["uni"], vals, ms)
    assert ok == ["unipc"] and err is None


def test_build_jobs():
    base_vals = _base_vals()
    base_ms = {"base_repo": "Tongyi-MAI/Z-Image-Turbo", "transformer": None,
               "loras": [], "sampler": "euler", "schedule": "sgm_uniform"}
    axes = [("Steps", [4, 8, 12]), ("Guidance", [0.0, 3.5])]
    jobs, meta = cz_ui._xyz_build_jobs(axes, base_vals, base_ms)
    assert len(jobs) == 6
    assert jobs[0]["vals"][15] == 4 and jobs[0]["vals"][18] == 0.0
    assert jobs[5]["vals"][15] == 12 and jobs[5]["vals"][18] == 3.5
    assert jobs[0]["xyz"] == {"gid": meta["gid"], "ix": 0, "iy": 0, "iz": 0}
    assert jobs[5]["xyz"]["ix"] == 2 and jobs[5]["xyz"]["iy"] == 1
    assert "Steps=4" in jobs[0]["label"] and "Guidance=0.0" in jobs[0]["label"]
    assert base_vals[15] == 8, "base snapshot must stay untouched (pure)"
    assert meta["x"] == ("Steps", ["4", "8", "12"]) and meta["y"][0] == "Guidance"


def test_build_jobs_sr():
    base_vals = _base_vals("a red cat on a sofa")
    base_ms = {"loras": []}
    axes = [("Prompt S/R", ["cat", "dog", "fox"])]
    jobs, _meta = cz_ui._xyz_build_jobs(axes, base_vals, base_ms)
    assert len(jobs) == 3
    assert jobs[0]["vals"][0] == "a red cat on a sofa"     # 1re valeur = terme -> inchange
    assert jobs[1]["vals"][0] == "a red dog on a sofa"
    assert jobs[2]["vals"][0] == "a red fox on a sofa"


def test_assemble():
    tmp = tempfile.mkdtemp()
    meta = {"gid": "test", "x": ("Steps", ["4", "8"]), "y": ("Denoise", ["0.2", "0.4"]),
            "z": None, "out_dir": tmp}
    cells = {(0, 0, 0): Image.new("RGB", (640, 480), (200, 40, 40)),
             (1, 0, 0): Image.new("RGB", (640, 480), (40, 200, 40)),
             (0, 1, 0): Image.new("RGB", (640, 480), (40, 40, 200))}
    # (1,1,0) manquante -> placeholder attendu, pas d'exception
    paths = cz_ui._xyz_assemble(meta, cells, thumb=128)
    assert len(paths) == 1 and os.path.isfile(paths[0])
    sheet = Image.open(paths[0])
    assert sheet.width > 2 * 128 and sheet.height > 2 * 128
    assert "xyz_test" in paths[0]


def test_csv_join_roundtrip():
    vals = ["plain", "with, comma", 'with "quote"', "4x-Clear.safetensors"]
    joined = cz_ui._xyz_csv_join(vals)
    assert cz_ui._xyz_parse_values(joined) == vals


def test_suggestions():
    fill, ph = cz_ui._xyz_suggestions("Steps")            # calibrage numerique
    assert fill == "4, 8, 12, 20, 28" and "4, 8" in ph
    fill, ph = cz_ui._xyz_suggestions("Sampler")          # liste fermee
    assert cz_ui._xyz_parse_values(fill) == ["euler", "unipc"]
    fill, ph = cz_ui._xyz_suggestions("Performance")
    assert "Turbo (8 steps)" in cz_ui._xyz_parse_values(fill)
    fill, ph = cz_ui._xyz_suggestions("Checkpoint")       # repos officiels presents
    assert "Tongyi-MAI/Z-Image-Turbo" in cz_ui._xyz_parse_values(fill)
    fill, ph = cz_ui._xyz_suggestions("Prompt S/R")       # pas d'insertion, aide seule
    assert fill == "" and "search term" in ph
    fill, ph = cz_ui._xyz_suggestions("(none)")
    assert fill == "" and "pick an axis" in ph


def test_fill_preserves_user_input():
    upd = cz_ui._ui_xyz_fill("Steps", "5, 9")             # champ non vide -> intouche
    assert "value" not in upd
    upd = cz_ui._ui_xyz_fill("Steps", "  ")               # vide -> rempli
    assert upd["value"] == "4, 8, 12, 20, 28"
    upd = cz_ui._ui_xyz_fill("Prompt S/R", "")            # rien a inserer
    assert "value" not in upd


if __name__ == "__main__":
    for fn in (test_parse_values, test_match, test_validate, test_build_jobs,
               test_build_jobs_sr, test_assemble, test_csv_join_roundtrip,
               test_suggestions, test_fill_preserves_user_input):
        fn()
        print(f"OK {fn.__name__}")
    print("All xyz tests passed.")
