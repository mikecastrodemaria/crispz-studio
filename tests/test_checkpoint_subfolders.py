"""Checkpoints filed in SUBFOLDERS of the checkpoints folder(s).

The user sorts the Z-Image models into ZIB/ and ZIT/ under the checkpoints folder; the
dropdown only listed the root. The listing now walks the main AND the extra folder
recursively, shows the path relative to its folder ('ZIB/model.safetensors'), and
resolve_checkpoint turns that entry -- or a bare file name saved by an older preset --
back into the absolute path. Same walk for the LoRAs, which the <lora:name> tag
resolves by file name too.

Neither GPU nor model: the files are empty.

Run:  .venv/Scripts/python tests/test_checkpoint_subfolders.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_pipeline as P  # noqa: E402


class _Dirs:
    """A main checkpoints folder + an extra one, with the LoRAs in a third."""

    def __enter__(self):
        self.main, self.extra, self.loras = (tempfile.mkdtemp() for _ in range(3))
        self.old = (P.CHECKPOINTS_DIR, P.CHECKPOINTS_EXTRA_DIR,
                    P.LORAS_DIR, list(P.LORAS_EXTRA_DIRS))
        P.CHECKPOINTS_DIR, P.CHECKPOINTS_EXTRA_DIR = self.main, self.extra
        P.LORAS_DIR, P.LORAS_EXTRA_DIRS = self.loras, []
        return self

    def __exit__(self, *exc):
        (P.CHECKPOINTS_DIR, P.CHECKPOINTS_EXTRA_DIR,
         P.LORAS_DIR, P.LORAS_EXTRA_DIRS) = self.old
        for d in (self.main, self.extra, self.loras):
            shutil.rmtree(d, ignore_errors=True)
        return False

    def put(self, where, rel):
        p = os.path.join(where, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(b"x")
        return p


def _same(a, b):
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def test_the_listing_walks_the_subfolders_of_both_folders():
    with _Dirs() as d:
        d.put(d.main, "root.safetensors")
        d.put(d.main, "ZIB/base.safetensors")
        d.put(d.main, "ZIT/turbo.gguf")
        d.put(d.extra, "deep/er/shared.safetensors")
        d.put(d.main, ".cache/hf.safetensors")             # a cache: never a model
        d.put(d.main, "ZIB/notes.txt")                      # not a model
        d.put(d.main, "ZIB/partial.crdownload")             # a download in progress
        got = P.list_checkpoints()
        assert got == ["ZIB/base.safetensors", "ZIT/turbo.gguf",
                       "deep/er/shared.safetensors", "root.safetensors"], got


def test_an_entry_resolves_to_its_own_folder():
    with _Dirs() as d:
        real = d.put(d.extra, "ZIB/base.safetensors")
        assert _same(P.resolve_checkpoint("ZIB/base.safetensors"), real)
        assert _same(P.resolve_checkpoint("ZIB" + chr(92) + "base.safetensors"), real)   # Windows form
        assert P.checkpoint_badge("ZIB/base.safetensors").startswith("BF16")


def test_the_main_folder_wins_on_a_duplicate_relative_path():
    with _Dirs() as d:
        mine = d.put(d.main, "ZIB/same.safetensors")
        d.put(d.extra, "ZIB/same.safetensors")
        assert P.list_checkpoints() == ["ZIB/same.safetensors"], P.list_checkpoints()
        assert _same(P.resolve_checkpoint("ZIB/same.safetensors"), mine)


def test_a_bare_name_saved_before_the_model_was_filed_away_still_loads():
    """A preset or a pref holds 'base.safetensors'; the file has since moved to ZIB/."""
    with _Dirs() as d:
        real = d.put(d.main, "ZIB/base.safetensors")
        assert _same(P.resolve_checkpoint("base.safetensors"), real)
        assert _same(P.resolve_checkpoint("BASE.safetensors"), real)       # case-insensitive
        # Unknown everywhere: the main folder, so the caller reports 'not found'.
        assert _same(P.resolve_checkpoint("ghost.safetensors"),
                     os.path.join(d.main, "ghost.safetensors"))


def test_the_catalogue_name_of_an_absolute_path():
    """The Asset Browser focus: the models catalogue is indexed by the relative path."""
    with _Dirs() as d:
        real = d.put(d.extra, "ZIT/turbo.safetensors")
        assert P.checkpoint_relname(real) == "ZIT/turbo.safetensors"
        assert P.checkpoint_relname("ZIT/turbo.safetensors") == "ZIT/turbo.safetensors"
        assert P.checkpoint_relname(os.path.join(tempfile.gettempdir(), "x", "out.safetensors")) \
            == "out.safetensors"


def test_a_lora_in_a_subfolder_is_listed_resolved_and_found_by_its_tag():
    with _Dirs() as d:
        real = d.put(d.loras, "Style/noir.safetensors")
        d.put(d.loras, "root.safetensors")
        d.put(d.loras, ".cache/junk.safetensors")
        assert P.list_loras() == ["Style/noir.safetensors", "root.safetensors"], P.list_loras()
        assert _same(P.resolve_lora_path("Style/noir.safetensors"), real)
        assert _same(P.resolve_lora_path("Style" + chr(92) + "noir.safetensors"), real)
        # <lora:noir>, <lora:noir.safetensors>, <lora:Style/noir> all land on the file
        for tag in ("noir", "noir.safetensors", "Style/noir", "style/NOIR.safetensors"):
            assert P.resolve_lora_name(tag) == "Style/noir.safetensors", (tag, P.resolve_lora_name(tag))


def test_the_ui_focus_and_presets_accept_subfolder_entries():
    import cz_ui
    with _Dirs() as d:
        d.put(d.main, "ZIB/base.safetensors")
        # No catalogue rebuild against these temp folders (it writes under the output dir).
        keep = cz_ui.ab_build_catalog
        cz_ui.ab_build_catalog = lambda *a, **k: None
        try:
            _msg, url = cz_ui._asset_focus_url("models", "ZIB/base.safetensors")
        finally:
            cz_ui.ab_build_catalog = keep
        assert url.endswith("&focus=ZIB/base.safetensors"), url
        # The auto-preset is named after the file alone, not the subfolder.
        assert cz_ui._preset_sanitize(os.path.splitext(os.path.basename("ZIB/base.safetensors"))[0]) == "base"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} tests passed")
