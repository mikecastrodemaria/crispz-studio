"""CivitAI LoRA search + download (Models > LoRA > Search CivitAI).

search_loras / download_model_file / _try_remove are copied from crispz-studio unchanged.
The RANKING is not: studio compares the CivitAI base label for EQUALITY with the name the
app gives its own model, and CivitAI does not spell them the same way -- its 'Z-Image only'
box, ticked by default, empties every search even when real Z-Image LoRAs are in the
results. So the comparison here is a normalised PREFIX, and what CivitAI really answers is
pinned below as DATA -- 'ZImageTurbo', 'ZImageBase', 'Z-Image' -- read from the API and from the sidecars of a real LoRA
folder on this machine, never guessed.

No network: _api_get and the download stream are stubbed. What is checked:
  - the configured family really matches the labels CivitAI returns (the studio regression);
  - the family filter keeps the family and drops the foreign bases;
  - the ranking puts the family first, and the preferred base ahead of it when there is one;
  - search_loras flattens one entry per model VERSION and survives a network failure;
  - a download whose SHA256 does not match is refused AND the file is removed;
  - an existing file is never overwritten.

Run:  .venv/Scripts/python tests/test_civitai_search.py
"""
import hashlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_civitai  # noqa: E402
import cz_ui  # noqa: E402

# The base labels CivitAI really returns for this app's LoRAs. NOT guessed: read from the
# API and from the '*.civitai.json' sidecars of a real LoRA folder.
_REAL_LABELS = ("ZImageTurbo", "ZImageBase", "Z-Image")
_FOREIGN = "SDXL 1.0"


def _model(mid, name, versions):
    return {"id": mid, "name": name, "creator": {"username": "someone"},
            "nsfw": False, "modelVersions": versions}


def _ver(vid, base, fname="a.safetensors", sha="", size=2048):
    return {"id": vid, "name": f"v{vid}", "baseModel": base,
            "files": [{"name": fname, "primary": True, "sizeKB": size,
                       "downloadUrl": f"https://civitai.com/api/download/models/{vid}",
                       "hashes": {"SHA256": sha}}],
            "images": [{"url": "https://img/x.jpg"}]}


class _Api:
    """Stubs cz_civitai._api_get with a fixed payload (or None = network failure)."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def __enter__(self):
        self.real = cz_civitai._api_get

        def fake(endpoint, params=None, api_key=None):
            self.calls.append((endpoint, params))
            return self.payload

        cz_civitai._api_get = fake
        return self

    def __exit__(self, *exc):
        cz_civitai._api_get = self.real
        return False


# The foreign base FIRST, so a passing ranking test cannot be CivitAI's order by accident.
_PAYLOAD = {"items": [_model(1, "Foreign", [_ver(11, _FOREIGN)])]
            + [_model(10 + i, f"Fam{i}", [_ver(100 + i, b)])
               for i, b in enumerate(_REAL_LABELS)]}


def _bases(dd_update):
    """The base model of each candidate, read back from the dropdown labels."""
    return [lbl.split("[", 1)[1].split("]", 1)[0] for lbl in dd_update["choices"]]


def test_the_configured_family_matches_the_labels_civitai_really_returns():
    """The regression this guards is crispz-studio's: it compared for EQUALITY against its
    own name for the base ('Z-Image') while CivitAI answers 'ZImageTurbo' / 'ZImageBase',
    so the filter silently emptied every search. Each real label must be matched by the
    configured family -- otherwise the whole panel looks broken and says 'no result'."""
    fam, pref = cz_ui._civitai_base_prefixes()
    assert fam, "the family prefix must not be empty (it would match every base)"
    for label in _REAL_LABELS:
        assert cz_civitai._norm_base(label).startswith(fam), \
            f"{label!r} is not matched by the family {fam!r}"
    assert not cz_civitai._norm_base(_FOREIGN).startswith(fam), \
        "the family prefix is so short that it matches a foreign base too"
    if pref:
        assert pref.startswith(fam), "the preferred base must be inside the family"
        assert any(cz_civitai._norm_base(b).startswith(pref) for b in _REAL_LABELS), \
            "the preferred base matches none of the real labels"


def test_search_flattens_one_entry_per_version():
    with _Api({"items": [_model(1, "Two", [_ver(11, _FOREIGN), _ver(12, _FOREIGN)])]}):
        out = cz_civitai.search_loras("x")
    assert len(out) == 2, out
    assert [c["versionId"] for c in out] == [11, 12]
    assert out[0]["modelName"] == "Two" and out[0]["url"].endswith("/models/1")


def test_a_network_failure_is_an_empty_list_not_an_exception():
    with _Api(None):
        assert cz_civitai.search_loras("x") == []
    assert cz_civitai.search_loras("   ") == []          # empty query: no call at all


def test_the_family_comes_first_and_the_preferred_base_ahead_of_it():
    """The foreign base is first in the payload and must end up last. When a preferred base
    is configured, it comes ahead of the rest of the family."""
    with _Api(_PAYLOAD):
        _q, dd, state, status = cz_ui._ui_civitai_lora_search("x", False)
    got = _bases(dd)
    assert len(state) == len(_REAL_LABELS) + 1, state
    assert got[-1] == _FOREIGN, got
    _fam, pref = cz_ui._civitai_base_prefixes()
    if pref:
        assert cz_civitai._norm_base(got[0]).startswith(pref), got
        assert f"with a {cz_ui._CIV_PREFERRED} base" in status, status
    else:
        assert "with a " not in status, status    # no sub-tier -> no note about one


def test_the_family_filter_drops_the_foreign_bases():
    with _Api(_PAYLOAD):
        _q, dd, _s, _st = cz_ui._ui_civitai_lora_search("x", True)
    got = _bases(dd)
    assert _FOREIGN not in got, got
    assert len(got) == len(_REAL_LABELS), got


def test_no_result_names_the_filter_as_the_likely_cause():
    with _Api({"items": [_model(1, "Foreign", [_ver(11, _FOREIGN)])]}):
        _q, dd, state, status = cz_ui._ui_civitai_lora_search("x", True)
    assert dd["choices"] == [] and state == {}
    assert cz_ui._CIV_FILTER_LABEL in status, status


class _Stream:
    """Stubs urlopen with a fixed body, so the download never touches the network."""

    def __init__(self, body):
        self.body = body

    def __enter__(self):
        import urllib.request
        self.real = urllib.request.urlopen
        body = self.body

        class _R:
            headers = {"Content-Length": str(len(body))}

            def __init__(self):
                self._b = io.BytesIO(body)

            def read(self, n):
                return self._b.read(n)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        urllib.request.urlopen = lambda *a, **k: _R()
        return self

    def __exit__(self, *exc):
        import urllib.request
        urllib.request.urlopen = self.real
        return False


def _no_enrich():
    """fetch_civitai_for_model would hit the network after a download."""
    real = cz_civitai.fetch_civitai_for_model
    cz_civitai.fetch_civitai_for_model = lambda *a, **k: None
    return real


def test_a_sha256_mismatch_refuses_and_removes_the_file():
    """The whole point of checking during the stream: a corrupted LoRA must not be left on
    disk looking valid."""
    body = b"not-the-announced-bytes"
    real = _no_enrich()
    try:
        with _Stream(body), tempfile.TemporaryDirectory() as d:
            cand = {"downloadUrl": "https://x/y", "fileName": "bad.safetensors",
                    "sha256": "0" * 64, "sizeKB": 1}
            res = cz_civitai.download_model_file(cand, d)
            assert res["success"] is False, res
            assert "SHA256 mismatch" in res["message"], res
            assert os.listdir(d) == [], os.listdir(d)      # no .part left either
    finally:
        cz_civitai.fetch_civitai_for_model = real


def test_a_matching_sha256_lands_the_file():
    body = b"the-real-bytes"
    real = _no_enrich()
    try:
        with _Stream(body), tempfile.TemporaryDirectory() as d:
            cand = {"downloadUrl": "https://x/y", "fileName": "ok.safetensors",
                    "sha256": hashlib.sha256(body).hexdigest(), "sizeKB": 1}
            res = cz_civitai.download_model_file(cand, d)
            assert res["success"] is True, res
            assert "verified" in res["message"], res
            # The file AND its hash sidecar: _cache_sha256 writes '<stem>.civitai.json' so
            # the next scan does not re-read the whole file to recompute the hash.
            assert sorted(os.listdir(d)) == ["ok.civitai.json", "ok.safetensors"], \
                os.listdir(d)
            assert res["path"] == os.path.join(d, "ok.safetensors"), res
    finally:
        cz_civitai.fetch_civitai_for_model = real


def test_an_existing_file_is_never_overwritten():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "there.safetensors")
        with open(p, "wb") as f:
            f.write(b"mine")
        res = cz_civitai.download_model_file(
            {"downloadUrl": "https://x/y", "fileName": "there.safetensors"}, d)
        assert res["success"] is True and "already exists" in res["message"], res
        with open(p, "rb") as f:
            assert f.read() == b"mine", "the existing file was overwritten"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"All {len(tests)} CivitAI search/download tests passed.")
