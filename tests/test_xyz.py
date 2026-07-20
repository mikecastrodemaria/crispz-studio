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


def test_cli_apply():
    import cz_cli
    p = {"prompt": "a red cat", "gen_steps": 8, "guidance": 0.0, "seed": 42,
         "esrgan": None, "factor": 2.0, "denoise": 0.3, "tile": 512, "refine_tile": 0}
    cz_cli._xyz_cli_apply("Steps", 12, p, {})              # kind=val -> param abstrait
    cz_cli._xyz_cli_apply("Guidance", 3.5, p, {})
    cz_cli._xyz_cli_apply("Denoise", 0.4, p, {})
    assert p["gen_steps"] == 12 and p["guidance"] == 3.5 and p["denoise"] == 0.4
    cz_cli._xyz_cli_apply("Performance", "Base CFG (28 steps)", p, {})
    assert p["gen_steps"] == 28 and p["guidance"] == 4.0
    cz_ui._XYZ_AXES["Prompt S/R"]["_term"] = "cat"
    cz_cli._xyz_cli_apply("Prompt S/R", "cat", p, {})      # 1re valeur = inchange
    assert p["prompt"] == "a red cat"
    cz_cli._xyz_cli_apply("Prompt S/R", "dog", p, {})
    assert p["prompt"] == "a red dog"


def test_cli_axis_name_resolution():
    ax = [k for k in cz_ui._XYZ_AXES if k != "(none)"]
    assert cz_ui._xyz_match("step", ax) == ("Steps", None)          # partiel unique
    assert cz_ui._xyz_match("GUIDANCE", ax) == ("Guidance", None)   # casse ignoree
    assert cz_ui._xyz_match("prompt", ax) == ("Prompt S/R", None)
    assert cz_ui._xyz_match("ile", ax)[0] is None                   # ambigu (Tile/Refine tile)
    assert cz_ui._xyz_match("tile", ax) == ("Tile", None)           # exact (casse ignoree) gagne


def _with_fake_loras(names):
    """Remplace temporairement la liste des LoRA disponibles (les tests ne doivent
    pas dependre du contenu du dossier loras de la machine)."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        orig_list, orig_dir = cz_ui.list_loras, cz_ui.cz_pipeline.LORAS_DIR
        cz_ui.list_loras = lambda: list(names)
        cz_ui.cz_pipeline.LORAS_DIR = "D:/loras"
        try:
            yield
        finally:
            cz_ui.list_loras = orig_list
            cz_ui.cz_pipeline.LORAS_DIR = orig_dir
    return _ctx()


_FAKE_LORAS = ["epochs/ollie_e000010.safetensors",
               "epochs/ollie_e000020.safetensors",
               "epochs/ollie_e000030.safetensors",
               "style/other, comma.safetensors"]


def test_lora_name_validate():
    vals, ms = _base_vals(), {"loras": []}
    with _with_fake_loras(_FAKE_LORAS):
        # resolution par fragment unique + case temoin None
        ok, err = cz_ui._xyz_validate_axis("LoRA", ["e000010", "e000030", "None"], vals, ms)
        assert err is None, err
        assert ok == [("epochs/ollie_e000010.safetensors", None),
                      ("epochs/ollie_e000030.safetensors", None), ("None", None)], ok
        # fragment ambigu -> refus, pas de choix au hasard
        ok, err = cz_ui._xyz_validate_axis("LoRA", ["ollie"], vals, ms)
        assert ok is None and "ambiguous" in err, (ok, err)
        # inconnu -> refus
        ok, err = cz_ui._xyz_validate_axis("LoRA", ["nope"], vals, ms)
        assert ok is None and "not found" in err, (ok, err)
        # nom contenant une virgule (protege par des guillemets en amont)
        ok, err = cz_ui._xyz_validate_axis("LoRA", ["other, comma"], vals, ms)
        assert err is None and ok[0][0] == "style/other, comma.safetensors", (ok, err)


def test_lora_name_weight_validate():
    vals, ms = _base_vals(), {"loras": []}
    with _with_fake_loras(_FAKE_LORAS):
        ok, err = cz_ui._xyz_validate_axis(
            "LoRA + weight", ["e000010:0.5", "e000020:0.9"], vals, ms)
        assert err is None, err
        assert ok == [("epochs/ollie_e000010.safetensors", 0.5),
                      ("epochs/ollie_e000020.safetensors", 0.9)], ok
        # poids manquant / illisible
        ok, err = cz_ui._xyz_validate_axis("LoRA + weight", ["e000010"], vals, ms)
        assert ok is None and "name:weight" in err, (ok, err)
        ok, err = cz_ui._xyz_validate_axis("LoRA + weight", ["e000010:abc"], vals, ms)
        assert ok is None and "invalid weight" in err, (ok, err)


def test_lora_name_apply_and_labels():
    base_vals = _base_vals()
    base_ms = {"loras": [("D:/loras/old.safetensors", 0.65)],
               "sampler": "euler", "schedule": "sgm_uniform"}
    with _with_fake_loras(_FAKE_LORAS):
        axes = [("LoRA", [("epochs/ollie_e000010.safetensors", None),
                          ("epochs/ollie_e000020.safetensors", None),
                          ("None", None)])]
        jobs, meta = cz_ui._xyz_build_jobs(axes, base_vals, base_ms)
    assert len(jobs) == 3
    # l'axe "LoRA" conserve le poids courant (0.65)
    # _path_for_lora = os.path.join -> separateurs mixtes sous Windows, comme pour
    # les slots LoRA normaux (set_loras s'en accommode).
    assert jobs[0]["ms"]["loras"] == [
        (os.path.join("D:/loras", "epochs/ollie_e000010.safetensors"), 0.65)], \
        jobs[0]["ms"]["loras"]
    assert jobs[2]["ms"]["loras"] == []                     # None -> aucun LoRA
    assert base_ms["loras"] == [("D:/loras/old.safetensors", 0.65)], "snapshot de base intact"
    # etiquettes : nom de base, sans extension, colonnes distinctes
    assert meta["x"][1] == ["ollie_e000010", "ollie_e000020", "None"], meta["x"]
    assert len(set(meta["x"][1])) == 3
    assert "LoRA=ollie_e000010" in jobs[0]["label"], jobs[0]["label"]


def test_lora_name_weight_apply():
    base_vals = _base_vals()
    base_ms = {"loras": [], "sampler": "euler", "schedule": "sgm_uniform"}
    with _with_fake_loras(_FAKE_LORAS):
        axes = [("LoRA + weight", [("epochs/ollie_e000010.safetensors", 0.5),
                                   ("epochs/ollie_e000020.safetensors", 0.9)])]
        jobs, meta = cz_ui._xyz_build_jobs(axes, base_vals, base_ms)
    assert jobs[0]["ms"]["loras"] == [
        (os.path.join("D:/loras", "epochs/ollie_e000010.safetensors"), 0.5)]
    assert jobs[1]["ms"]["loras"] == [
        (os.path.join("D:/loras", "epochs/ollie_e000020.safetensors"), 0.9)]
    assert meta["x"][1] == ["ollie_e000010@0.5", "ollie_e000020@0.9"], meta["x"]


def test_lora_label_truncates_left():
    """Les LoRA compares ne different que par leur suffixe : une troncature par la
    droite rendrait toutes les colonnes identiques."""
    long_a = "x" * 40 + "_e000010.safetensors"
    long_b = "x" * 40 + "_e000020.safetensors"
    a = cz_ui._xyz_fmt_value("LoRA", (long_a, None))
    b = cz_ui._xyz_fmt_value("LoRA", (long_b, None))
    assert a != b, (a, b)
    assert a.endswith("_e000010") and b.endswith("_e000020"), (a, b)
    assert len(a) <= 28, a


def test_lora_suggestions():
    with _with_fake_loras(_FAKE_LORAS):
        fill, ph = cz_ui._xyz_suggestions("LoRA")
        # la liste inseree doit se re-parser (le nom a virgule est guillemete)
        assert cz_ui._xyz_parse_values(fill) == ["None"] + _FAKE_LORAS, fill
        assert "e.g." in ph
        fill_w, _ = cz_ui._xyz_suggestions("LoRA + weight")
        vals = cz_ui._xyz_parse_values(fill_w)
        assert vals[0] == "None" and all(":" in v for v in vals[1:]), vals
        # chaque suggestion pondere doit repasser la validation
        ok, err = cz_ui._xyz_validate_axis("LoRA + weight", vals[1:], _base_vals(), {"loras": []})
        assert err is None, err
    # aucun LoRA disponible -> placeholder explicite, pas de plantage
    with _with_fake_loras([]):
        fill, ph = cz_ui._xyz_suggestions("LoRA")
        assert "no LoRA found" in ph, ph


if __name__ == "__main__":
    for fn in (test_parse_values, test_match, test_validate, test_build_jobs,
               test_build_jobs_sr, test_assemble, test_csv_join_roundtrip,
               test_suggestions, test_fill_preserves_user_input,
               test_cli_apply, test_cli_axis_name_resolution,
               test_lora_name_validate, test_lora_name_weight_validate,
               test_lora_name_apply_and_labels, test_lora_name_weight_apply,
               test_lora_label_truncates_left, test_lora_suggestions):
        fn()
        print(f"OK {fn.__name__}")
    print("All xyz tests passed.")
