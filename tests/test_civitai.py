"""Unit tests for the local/pure CivitAI helpers (no network hit).

Run:  .venv/Scripts/python tests/test_civitai.py
"""
import os
import sys
import json
import hashlib
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_civitai  # noqa: E402


def _tmpfile(data=b"hello crispz"):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "model.safetensors")
    with open(p, "wb") as f:
        f.write(data)
    return p


def test_compute_sha256_progress():
    p = _tmpfile(b"x" * (3 << 20))          # 3 MB -> plusieurs chunks de 1 MB
    events = []
    h = cz_civitai._compute_sha256(p, progress=lambda ph, fr, tx: events.append((ph, fr)))
    assert h == hashlib.sha256(b"x" * (3 << 20)).hexdigest()
    assert events and all(ph == "hash" for ph, _ in events)
    fracs = [fr for _, fr in events]
    assert fracs == sorted(fracs) and abs(fracs[-1] - 1.0) < 1e-9     # croissant -> 100%


def test_model_sha256_reads_sidecar_no_hash():
    p = _tmpfile(b"anything")
    sha = "a" * 64
    with open(os.path.splitext(p)[0] + ".metadata.json", "w", encoding="utf-8") as f:
        json.dump({"sha256": sha.upper()}, f)
    # doit lire le sidecar (minuscule) sans hasher le fichier -> progress jamais appele
    called = []
    got = cz_civitai.model_sha256(p, progress=lambda *a: called.append(a))
    assert got == sha and not called


def test_fetch_missing_file_no_network():
    res = cz_civitai.fetch_civitai_for_model("D:/nope/does-not-exist.safetensors")
    assert res["success"] is False and "not found" in res["message"]


def test_has_preview_and_sidecar_load():
    p = _tmpfile()
    assert cz_civitai.has_preview(p) is False
    assert cz_civitai.load_civitai_sidecar(p) == {}
    stem = os.path.splitext(p)[0]
    with open(stem + ".preview.png", "wb") as f:
        f.write(b"\x89PNG")
    with open(stem + ".civitai.json", "w", encoding="utf-8") as f:
        json.dump({"trainedWords": ["foo"], "examples": [{"url": "u", "prompt": "p"}]}, f)
    assert cz_civitai.has_preview(p) is True
    sc = cz_civitai.load_civitai_sidecar(p)
    assert sc["trainedWords"] == ["foo"] and sc["examples"][0]["prompt"] == "p"


def test_examples_from_reads_meta_prompt():
    """Les images by-hash portent un meta REMPLI -> le prompt doit etre extrait.
    Regression: on lisait l'endpoint /images dont 'meta' est toujours null -> 0 prompt."""
    imgs = [
        {"url": "u1", "width": 8, "height": 9, "meta": {"prompt": "  a nordic woman  "}},
        {"url": "u2", "meta": None},                 # parametres non publies
        {"url": "u3", "meta": {"prompt": ""}},       # meta sans prompt
        {"no_url": 1, "meta": {"prompt": "x"}},      # sans url -> ignoree
    ]
    ex = cz_civitai._examples_from(imgs)
    assert len(ex) == 3                                    # la 4e est ignoree
    assert ex[0]["prompt"] == "a nordic woman" and ex[0]["has_prompt"] is True
    assert ex[0]["width"] == 8 and ex[0]["height"] == 9
    assert ex[1]["prompt"] == "" and ex[1]["has_prompt"] is False   # meta None -> honnete
    assert ex[2]["has_prompt"] is False


def test_examples_from_respects_limit():
    imgs = [{"url": f"u{i}", "meta": {"prompt": "p"}} for i in range(20)]
    assert len(cz_civitai._examples_from(imgs, limit=8)) == 8


def test_get_version_by_hash_carries_images(monkeypatch=None):
    """by-hash doit remonter ses images (elles contiennent les prompts) -> 0 requete de plus."""
    payload = {"id": 42, "modelId": 7, "name": "v1", "baseModel": "Z-Image",
               "trainedWords": ["trg"], "model": {"name": "M"},
               "images": [{"url": "u", "meta": {"prompt": "hello"}}]}
    old = cz_civitai._api_get
    cz_civitai._api_get = lambda ep, params=None, api_key=None, timeout=20: payload
    try:
        ver = cz_civitai.get_version_by_hash("a" * 64)
    finally:
        cz_civitai._api_get = old
    assert ver["images"] and ver["images"][0]["meta"]["prompt"] == "hello"
    assert ver["trainedWords"] == ["trg"] and ver["versionId"] == 42


def test_api_get_falls_back_to_global_key():
    """api_key=None doit utiliser la cle globale (sinon les appels internes partent
    anonymes et ratent les contenus gates/NSFW)."""
    seen = {}

    class _R:
        def read(self): return b"{}"
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        return _R()

    old_open, old_key = cz_civitai.urllib.request.urlopen, cz_civitai.API_KEY
    cz_civitai.urllib.request.urlopen = fake_urlopen
    cz_civitai.API_KEY = "SECRET123"
    try:
        cz_civitai._api_get("/models/1")                    # sans api_key explicite
        assert "token=SECRET123" in seen["url"], seen["url"]
        cz_civitai._api_get("/models/1", api_key="OTHER")   # explicite -> prioritaire
        assert "token=OTHER" in seen["url"]
        cz_civitai.API_KEY = None                           # pas de cle -> pas de token
        cz_civitai._api_get("/models/1")
        assert "token=" not in seen["url"]
    finally:
        cz_civitai.urllib.request.urlopen = old_open
        cz_civitai.API_KEY = old_key


if __name__ == "__main__":
    for fn in (test_compute_sha256_progress, test_model_sha256_reads_sidecar_no_hash,
               test_fetch_missing_file_no_network, test_has_preview_and_sidecar_load,
               test_examples_from_reads_meta_prompt, test_examples_from_respects_limit,
               test_get_version_by_hash_carries_images, test_api_get_falls_back_to_global_key):
        fn()
        print(f"OK {fn.__name__}")
    print("All civitai tests passed.")
