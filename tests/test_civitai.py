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


def test_sha256_is_cached_and_reused():
    """Regression: le hash etait recalcule a CHAQUE passe (des centaines de Go relus).
    Il doit etre persiste dans le sidecar et reutilise."""
    p = _tmpfile(b"y" * 4096)
    calls = []
    real = cz_civitai._compute_sha256

    def counting(path, progress=None):
        calls.append(path)
        return real(path, progress=progress)

    cz_civitai._compute_sha256 = counting
    try:
        h1 = cz_civitai.model_sha256(p)          # calcule + met en cache
        h2 = cz_civitai.model_sha256(p)          # doit lire le cache
    finally:
        cz_civitai._compute_sha256 = real
    assert h1 == h2 == hashlib.sha256(b"y" * 4096).hexdigest()
    assert len(calls) == 1, "le 2e appel doit venir du cache, pas d'un re-hash"
    sc = cz_civitai.load_civitai_sidecar(p)
    assert sc["sha256"] == h1 and sc["sha256_size"] == 4096


def test_sha256_cache_invalidated_when_size_changes():
    p = _tmpfile(b"z" * 100)
    cz_civitai.model_sha256(p)
    with open(p, "wb") as f:                     # modele remplace -> taille differente
        f.write(b"z" * 200)
    assert cz_civitai._cached_sha256(p) is None, "cache perime doit etre rejete"
    assert cz_civitai.model_sha256(p) == hashlib.sha256(b"z" * 200).hexdigest()


def test_metadata_sidecar_wins_over_our_cache():
    p = _tmpfile(b"w" * 64)
    cz_civitai.model_sha256(p)                   # remplit notre cache
    ext = "b" * 64
    with open(os.path.splitext(p)[0] + ".metadata.json", "w", encoding="utf-8") as f:
        json.dump({"sha256": ext}, f)
    assert cz_civitai.model_sha256(p) == ext     # convention externe prioritaire


def test_fetch_sidecar_merge_keeps_hash_cache(monkeypatch=None):
    """Le fetch reecrit le sidecar: il ne doit PAS effacer le cache de hash."""
    p = _tmpfile(b"m" * 32)
    sha = cz_civitai.model_sha256(p)
    payload = {"id": 5, "modelId": 9, "name": "v", "baseModel": "Z", "trainedWords": [],
               "model": {"name": "M"}, "images": [{"url": "u", "meta": {"prompt": "hi"}}]}
    old_get, old_upd = cz_civitai._api_get, cz_civitai._update_fields
    cz_civitai._api_get = lambda ep, params=None, api_key=None, timeout=20: payload
    cz_civitai._update_fields = lambda *a, **k: {"update_available": False,
                                                 "latest_versionId": None,
                                                 "latest_versionName": ""}
    try:
        res = cz_civitai.fetch_civitai_for_model(p, check_update=False)
    finally:
        cz_civitai._api_get, cz_civitai._update_fields = old_get, old_upd
    assert res["success"] is True
    sc = cz_civitai.load_civitai_sidecar(p)
    assert sc["sha256"] == sha, "le fetch a ecrase le cache de hash"
    assert sc["modelId"] == 9 and sc["examples"][0]["prompt"] == "hi"


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


def _with_model_payload(payload, fn):
    old = cz_civitai._api_get
    cz_civitai._api_get = lambda ep, params=None, api_key=None, timeout=20: payload
    try:
        return fn()
    finally:
        cz_civitai._api_get = old


# Page CivitAI typique: la derniere version publiee l'est pour une AUTRE base.
_MIXED_BASES = {"modelVersions": [
    {"id": 300, "name": "3.0 (Krea2)", "baseModel": "Krea 2"},
    {"id": 200, "name": "2.0", "baseModel": "Z-Image"},
    {"id": 100, "name": "1.0", "baseModel": "Z Image"},     # libelle variant -> meme base
]}


def test_latest_version_ignores_other_base_models():
    """Regression: '3.0 (Krea2)' etait signale comme update d'un LoRA Z-Image alors qu'il
    ne tourne pas dessus. La derniere version de la MEME base doit gagner."""
    got = _with_model_payload(_MIXED_BASES,
                              lambda: cz_civitai.get_latest_version(9, base_model="Z-Image"))
    assert got["id"] == 200 and got["name"] == "2.0"
    # Sans filtre (base locale inconnue) -> comportement historique: la plus recente.
    raw = _with_model_payload(_MIXED_BASES, lambda: cz_civitai.get_latest_version(9))
    assert raw["id"] == 300


def test_update_flag_not_raised_by_a_new_base_model():
    upd = _with_model_payload(
        _MIXED_BASES, lambda: cz_civitai._update_fields(9, 200, base_model="Z-Image"))
    assert upd["update_available"] is False, "Krea2 n'est pas un update pour du Z-Image"
    # Vraie mise a jour: on est sur la 1.0 Z-Image -> la 2.0 Z-Image (pas la 3.0 Krea2).
    upd = _with_model_payload(
        _MIXED_BASES, lambda: cz_civitai._update_fields(9, 100, base_model="z image"))
    assert upd["update_available"] is True and upd["latest_versionId"] == 200
    assert upd["latest_versionName"] == "2.0"


def test_update_flag_infers_base_from_our_own_version():
    """Vieux sidecar sans 'baseModel': la base est deduite de NOTRE versionId dans la
    reponse (deja telechargee) -> le filtre marche sans requete supplementaire."""
    upd = _with_model_payload(_MIXED_BASES, lambda: cz_civitai._update_fields(9, 200))
    assert upd["update_available"] is False
    upd = _with_model_payload(_MIXED_BASES, lambda: cz_civitai._update_fields(9, 100))
    assert upd["update_available"] is True and upd["latest_versionId"] == 200


def test_update_flag_when_no_version_shares_our_base():
    """Le fichier local est d'une base absente de la page (ou renommee cote API) ->
    pas d'update plutot qu'un faux positif."""
    upd = _with_model_payload(
        _MIXED_BASES, lambda: cz_civitai._update_fields(9, 200, base_model="Flux.1 D"))
    assert upd["update_available"] is False and upd["latest_versionId"] is None


def test_update_flag_when_api_omits_base_models():
    """Si l'API ne renseigne aucun baseModel, l'info est indisponible (pas
    contradictoire) -> on ne filtre pas et on garde le comportement historique."""
    payload = {"modelVersions": [{"id": 300, "name": "3.0"}, {"id": 200, "name": "2.0"}]}
    upd = _with_model_payload(
        payload, lambda: cz_civitai._update_fields(9, 200, base_model="Z-Image"))
    assert upd["update_available"] is True and upd["latest_versionId"] == 300


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
               test_get_version_by_hash_carries_images, test_api_get_falls_back_to_global_key,
               test_sha256_is_cached_and_reused, test_sha256_cache_invalidated_when_size_changes,
               test_metadata_sidecar_wins_over_our_cache,
               test_fetch_sidecar_merge_keeps_hash_cache,
               test_latest_version_ignores_other_base_models,
               test_update_flag_not_raised_by_a_new_base_model,
               test_update_flag_infers_base_from_our_own_version,
               test_update_flag_when_no_version_shares_our_base,
               test_update_flag_when_api_omits_base_models):
        fn()
        print(f"OK {fn.__name__}")
    print("All civitai tests passed.")
