"""Extra LoRA folders: the library lives outside the app folder.

loras_dir + loras_extra_dirs are merged into ONE list. Everything that lists or resolves
a LoRA has to read that list -- the slots, the Asset Browser, the CivitAI enrichment --
or a library shared with ComfyUI or Forge is simply invisible.

Neither GPU nor model: the LoRAs are empty files.

Run:  .venv/Scripts/python tests/test_lora_extra_dirs.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_pipeline as P  # noqa: E402


class _Lib:
    """A nearly empty main folder and an extra one that holds the library."""

    def __init__(self, extra=1):
        self.main = tempfile.mkdtemp()
        self.extras = [tempfile.mkdtemp() for _ in range(extra)]

    def __enter__(self):
        self.old = (P.LORAS_DIR, list(P.LORAS_EXTRA_DIRS))
        P.LORAS_DIR, P.LORAS_EXTRA_DIRS = self.main, list(self.extras)
        return self

    def __exit__(self, *exc):
        P.LORAS_DIR, P.LORAS_EXTRA_DIRS = self.old
        for d in [self.main] + self.extras:
            shutil.rmtree(d, ignore_errors=True)
        return False

    def put(self, where, rel):
        p = os.path.join(where, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(b"x")
        return p


def test_the_listing_merges_every_folder():
    with _Lib(extra=2) as lib:
        lib.put(lib.main, "here.safetensors")
        lib.put(lib.extras[0], "Style/noir.safetensors")
        lib.put(lib.extras[1], "deep/er/one.ckpt")
        lib.put(lib.extras[0], "notes.txt")                 # not a LoRA
        assert P.list_loras() == ["Style/noir.safetensors", "deep/er/one.ckpt",
                                  "here.safetensors"], P.list_loras()


def test_a_name_resolves_to_the_folder_that_has_it():
    with _Lib() as lib:
        real = lib.put(lib.extras[0], "Style/noir.safetensors")
        got = P.resolve_lora_path("Style/noir.safetensors")
        assert os.path.normcase(got) == os.path.normcase(real), (got, real)
        assert os.path.isfile(got)


def test_the_main_folder_wins_on_a_duplicate_name():
    """Two folders, one file name: the one the user owns comes first."""
    with _Lib() as lib:
        mine = lib.put(lib.main, "same.safetensors")
        lib.put(lib.extras[0], "same.safetensors")
        assert P.list_loras() == ["same.safetensors"], P.list_loras()
        assert os.path.normcase(P.resolve_lora_path("same.safetensors")) \
            == os.path.normcase(mine)


def test_a_name_found_nowhere_falls_back_to_the_main_folder():
    """So the caller reports 'not found' against a path that means something."""
    with _Lib() as lib:
        got = P.resolve_lora_path("ghost.safetensors")
        assert os.path.normcase(got) == os.path.normcase(
            os.path.join(lib.main, "ghost.safetensors")), got


def test_an_absolute_path_is_left_alone():
    with _Lib() as lib:
        p = lib.put(lib.extras[0], "abs.safetensors")
        assert P.resolve_lora_path(p) == p


def test_the_setter_takes_a_list_or_a_semicolon_string():
    old = list(P.LORAS_EXTRA_DIRS)
    try:
        P.set_loras_extra_dirs("X:/a;Y:/b")
        assert P.LORAS_EXTRA_DIRS == ["X:/a", "Y:/b"], P.LORAS_EXTRA_DIRS
        P.set_loras_extra_dirs(["X:/a", "X:/a", "Z:/c"])     # deduplicated
        assert P.LORAS_EXTRA_DIRS == ["X:/a", "Z:/c"], P.LORAS_EXTRA_DIRS
        P.set_loras_extra_dirs("")
        assert P.LORAS_EXTRA_DIRS == [], P.LORAS_EXTRA_DIRS
    finally:
        P.LORAS_EXTRA_DIRS = old


def test_the_slots_resolve_through_the_extra_folders():
    """set_loras stores absolute paths: a slot picked from the merged list must point at
    the real file, not at a name under the main folder that does not exist."""
    with _Lib() as lib:
        real = lib.put(lib.extras[0], "Style/noir.safetensors")
        old = list(P.LORAS)
        try:
            P.set_loras([("Style/noir.safetensors", 0.8)])
            assert len(P.LORAS) == 1, P.LORAS
            assert os.path.normcase(P.LORAS[0][0]) == os.path.normcase(real), P.LORAS
            assert os.path.isfile(P.LORAS[0][0])
        finally:
            P.LORAS = old


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"All {len(tests)} extra LoRA folder tests passed.")
