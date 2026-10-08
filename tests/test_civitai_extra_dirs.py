"""Fiche CivitAI d'un modele range dans un dossier SUPPLEMENTAIRE.

Le catalogue de l'explorateur liste le dossier principal ET le(s) dossier(s)
supplementaire(s), mais le bouton "Fetch from CivitAI" joignait le chemin relatif au seul
dossier principal : tout modele d'une bibliotheque rangee hors du dossier de l'app -- le
cas normal -- repondait "model file not found". Trouve sur crispz-klein.

Ni reseau ni modele : on verifie seulement la resolution du chemin.

Run:  .venv/Scripts/python tests/test_civitai_extra_dirs.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_pipeline as P  # noqa: E402
import cz_ui  # noqa: E402


class _Dirs:
    """Un dossier principal vide + un dossier supplementaire qui contient tout."""

    def __init__(self):
        self.main = tempfile.mkdtemp()
        self.extra = tempfile.mkdtemp()

    def __enter__(self):
        self.old = (P.LORAS_DIR, P.CHECKPOINTS_DIR, P.CHECKPOINTS_EXTRA_DIR)
        P.LORAS_DIR = P.CHECKPOINTS_DIR = self.main
        P.CHECKPOINTS_EXTRA_DIR = self.extra
        self._enter_loras()
        return self

    def __exit__(self, *exc):
        (P.LORAS_DIR, P.CHECKPOINTS_DIR, P.CHECKPOINTS_EXTRA_DIR) = self.old
        self._exit_loras()
        for d in (self.main, self.extra):
            shutil.rmtree(d, ignore_errors=True)
        return False

    def put(self, rel, where=None):
        p = os.path.join(where or self.extra, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(b"x")
        return p

    def _enter_loras(self):
        pass        # ce fork n'a qu'un seul dossier de LoRA

    def _exit_loras(self):
        pass


def test_a_lora_resolves_in_the_single_lora_folder():
    with _Dirs() as d:
        real = d.put("Style/noir.safetensors", where=d.main)
        got = cz_ui._civitai_model_path("Style/noir.safetensors", "loras")
        assert os.path.isfile(got), got
        assert os.path.normcase(got) == os.path.normcase(real), (got, real)


def test_a_checkpoint_of_the_extra_folder_is_found():
    with _Dirs() as d:
        real = d.put("base-model.safetensors")
        got = cz_ui._civitai_model_path("base-model.safetensors", "models")
        assert os.path.normcase(got) == os.path.normcase(real), (got, real)


def test_a_model_missing_everywhere_falls_back_to_the_main_folder():
    """Pas de fichier -> le chemin du dossier principal, et le message reste clair
    ("model file not found") au lieu d'une exception."""
    with _Dirs() as d:
        got = cz_ui._civitai_model_path("ghost.safetensors", "loras")
        assert os.path.normcase(got) == os.path.normcase(
            os.path.join(d.main, "ghost.safetensors")), got
        assert not os.path.isfile(got)


def test_an_empty_name_resolves_to_nothing():
    with _Dirs():
        assert cz_ui._civitai_model_path("", "loras") == ""
        assert cz_ui._civitai_model_path(None, "models") == ""


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"All {len(tests)} CivitAI extra-folder tests passed.")
