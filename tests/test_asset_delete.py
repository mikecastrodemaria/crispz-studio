"""Unit tests for the Asset Browser's delete (no server, no UI).

Regression guard: `delete_asset(rel)` resolved `rel` against DEFAULT_OUTPUT_DIR, the folder
from config.txt — while the Asset Browser opens on the folder the UI currently points at.
As soon as those differed, every delete answered "not found" and the SPA removed the card
anyway, so the image looked deleted and was back on the next refresh.

The SPA now sends the folder it is showing (it is served from inside it, so it reads it off
its own URL) and the answer is checked. Because that value comes from the page, only a
folder the app itself opened the browser for is accepted.

Run:  .venv/Scripts/python tests/test_asset_delete.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_assetbrowser as AB  # noqa: E402


def _out_dir(with_file="2026-10-04/img.png"):
    d = tempfile.mkdtemp()
    if with_file:
        p = os.path.join(d, *with_file.split("/"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(b"\x89PNG not really")
        return d, p
    return d, None


def test_an_unregistered_folder_is_refused():
    """The folder comes from the page: an unchecked value would let the public endpoint
    delete anything on disk."""
    d, p = _out_dir()
    assert AB.delete_asset("2026-10-04/img.png", d) == "folder not allowed"
    assert os.path.isfile(p), "the file must not have been touched"


def test_a_registered_folder_deletes():
    d, p = _out_dir()
    AB.register_output_dir(d)                     # what opening the browser does
    assert AB.delete_asset("2026-10-04/img.png", d) == "deleted"
    assert not os.path.exists(p)


def test_the_sidecar_and_the_thumbnail_go_too():
    d, p = _out_dir()
    with open(p + ".json", "w", encoding="utf-8") as f:
        f.write("{}")
    AB.register_output_dir(d)
    assert AB.delete_asset("2026-10-04/img.png", d) == "deleted"
    assert not os.path.exists(p) and not os.path.exists(p + ".json")


def test_the_configured_folder_is_legal_from_the_start():
    """The SPA now ALWAYS sends a folder, so the default one must be accepted without
    anything having been opened first -- otherwise the fix breaks the common case."""
    assert os.path.abspath(AB._ab_resolve_dir(AB.DEFAULT_OUTPUT_DIR)) in AB.ALLOWED_OUTPUT_DIRS


def test_a_missing_file_says_so():
    d, _ = _out_dir(with_file=None)
    AB.register_output_dir(d)
    assert AB.delete_asset("2026-10-04/nope.png", d) == "not found"


def test_path_traversal_is_refused():
    d, _ = _out_dir()
    AB.register_output_dir(d)
    outside = tempfile.mkdtemp()
    victim = os.path.join(outside, "keepme.txt")
    with open(victim, "w", encoding="utf-8") as f:
        f.write("x")
    rel = os.path.relpath(victim, d).replace("\\", "/")
    assert AB.delete_asset(rel, d) == "not found"
    assert os.path.isfile(victim), "a path outside the folder must survive"


def test_the_spa_reads_its_folder_from_its_own_url_and_checks_the_answer():
    """The two halves of the fix on the page side: delAsset must send abRoot() and must
    stop removing the card when the app did not actually delete."""
    from cz_assets import ASSET_BROWSER_HTML as H
    assert "function abRoot()" in H
    assert "gcall('delete_asset',[e.file,abRoot()])" in H.replace(" ", "")
    assert "if(res!=='deleted')" in H.replace(" ", "")


if __name__ == "__main__":
    for fn in (test_an_unregistered_folder_is_refused,
               test_a_registered_folder_deletes,
               test_the_sidecar_and_the_thumbnail_go_too,
               test_the_configured_folder_is_legal_from_the_start,
               test_a_missing_file_says_so,
               test_path_traversal_is_refused,
               test_the_spa_reads_its_folder_from_its_own_url_and_checks_the_answer):
        fn()
        print(f"OK {fn.__name__}")
    print("All asset-delete tests passed.")
