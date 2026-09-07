"""L'onglet Models de l'Asset Browser sortait VIDE sur une install normale.

Il ne scannait que le dossier de checkpoints PRINCIPAL et ne reconnaissait que
.safetensors. Sur une machine qui range ses modeles ailleurs (dossier "extra",
autre disque) le principal est vide: l'onglet affichait 0 modele alors que la
bibliotheque en comptait 22, GGUF jamais listes au passage.

Run:  .venv/Scripts/python tests/test_asset_browser_dirs.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_assetbrowser as AB

TMP = tempfile.mkdtemp(prefix="cz_ab_dirs_")


def _mk(d, *names):
    os.makedirs(d, exist_ok=True)
    for n in names:
        open(os.path.join(d, n), "wb").write(b"\0" * 16)
    return d


def test_extra_dir_is_scanned():
    main = _mk(os.path.join(TMP, "main"), "alpha.safetensors")
    extra = _mk(os.path.join(TMP, "extra"), "beta.safetensors", "gamma.gguf")
    out = os.path.join(TMP, "out1")
    names = {e["name"] for e in AB._scan_catalog([main, extra], out, "models")}
    assert names == {"alpha", "beta", "gamma"}, names
    # un seul dossier reste accepte (l'ancienne signature)
    assert {e["name"] for e in AB._scan_catalog(main, out, "models")} == {"alpha"}
    print("OK test_extra_dir_is_scanned")


def test_gguf_counts_as_a_model_but_not_as_a_lora():
    d = _mk(os.path.join(TMP, "mixed"), "m.safetensors", "q.gguf", "l.pt")
    out = os.path.join(TMP, "out2")
    assert {e["name"] for e in AB._scan_catalog(d, out, "models")} == {"m", "q", "l"}
    # cote LoRA, un .gguf n'a rien a faire
    assert {e["name"] for e in AB._scan_catalog(d, out, "loras")} == {"m", "l"}
    print("OK test_gguf_counts_as_a_model_but_not_as_a_lora")


def test_same_name_the_main_folder_wins():
    """Meme regle que list_checkpoints: pas deux entrees pour un meme nom."""
    main = _mk(os.path.join(TMP, "m2"), "dup.safetensors")
    extra = _mk(os.path.join(TMP, "e2"), "dup.safetensors")
    out = os.path.join(TMP, "out3")
    items = AB._scan_catalog([main, extra], out, "models")
    assert len(items) == 1, items
    print("OK test_same_name_the_main_folder_wins")


def test_missing_dirs_are_not_an_error():
    out = os.path.join(TMP, "out4")
    assert AB._scan_catalog([], out, "models") == []
    assert AB._scan_catalog(None, out, "models") == []
    assert AB._scan_catalog([os.path.join(TMP, "nope")], out, "models") == []
    print("OK test_missing_dirs_are_not_an_error")


if __name__ == "__main__":
    try:
        test_extra_dir_is_scanned()
        test_gguf_counts_as_a_model_but_not_as_a_lora()
        test_same_name_the_main_folder_wins()
        test_missing_dirs_are_not_an_error()
        print("ALL OK")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
