"""Unit tests for the persistent job queue (save on mutation, restore at startup).

Run:  .venv/Scripts/python tests/test_queue_persist.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

import cz_ui  # noqa: E402

TMP = tempfile.mkdtemp(prefix="cz_queue_")


def _isolate():
    """Redirige le store vers un dossier de test et repart d'une file vide."""
    cz_ui._Q_STORE = os.path.join(TMP, "queue.json")
    cz_ui._Q_ASSETS = os.path.join(TMP, "assets")
    cz_ui.Q_PERSIST = True
    for p in (cz_ui._Q_STORE, cz_ui._Q_STORE + ".tmp"):
        if os.path.exists(p):
            os.remove(p)


def _job(prompt="a red cat", n=40):
    vals = [None] * n
    vals[cz_ui._Q_IDX["prompt"]] = prompt
    vals[cz_ui._Q_IDX["width"]] = 1024
    vals[cz_ui._Q_IDX["height"]] = 1024
    vals[cz_ui._Q_IDX["gen_steps"]] = 8
    vals[cz_ui._Q_IDX["image_number"]] = 1
    vals[cz_ui._Q_IDX["seed"]] = 42
    ms = {"base_repo": "Tongyi-MAI/Z-Image-Turbo", "transformer": "ck.safetensors",
          "loras": [("D:/loras/a.safetensors", 0.8)], "sampler": "euler",
          "schedule": "sgm_uniform"}
    return {"vals": vals, "ms": ms, "label": f"job {prompt}"}


def test_roundtrip_keeps_settings_and_lora_tuples():
    _isolate()
    cz_ui._q_persist([_job("a"), _job("b")])
    back = cz_ui._q_load()
    assert len(back) == 2
    assert back[0]["vals"][cz_ui._Q_IDX["prompt"]] == "a"
    assert back[1]["label"] == "job b"
    assert back[0]["ms"]["sampler"] == "euler"
    # les LoRA doivent revenir en TUPLES (le JSON les rendrait en listes) sinon
    # set_loras / la comparaison d'etat modele cassent au premier job restaure
    lo = back[0]["ms"]["loras"]
    assert lo and isinstance(lo[0], tuple) and lo[0][1] == 0.8


def test_history_is_not_persisted():
    _isolate()
    j = _job()
    j["vals"][cz_ui._Q_HISTORY_IDX] = [Image.new("RGB", (8, 8))] * 3   # galerie de session
    cz_ui._q_persist([j])
    assert cz_ui._q_load()[0]["vals"][cz_ui._Q_HISTORY_IDX] is None


def test_input_image_survives_restart():
    _isolate()
    j = _job()
    img = Image.new("RGB", (24, 16), (10, 200, 30))
    j["vals"][cz_ui._Q_IDX["use_input"]] = True
    j["vals"][5] = img                       # un composant image quelconque
    cz_ui._q_persist([j])
    back = cz_ui._q_load()[0]["vals"][5]
    assert isinstance(back, Image.Image) and back.size == (24, 16)
    assert back.getpixel((0, 0)) == (10, 200, 30)


def test_missing_asset_degrades_to_none():
    _isolate()
    j = _job()
    j["vals"][5] = Image.new("RGB", (8, 8))
    cz_ui._q_persist([j])
    for f in os.listdir(cz_ui._Q_ASSETS):    # l'utilisateur a vide cache/
        os.remove(os.path.join(cz_ui._Q_ASSETS, f))
    back = cz_ui._q_load()
    assert len(back) == 1 and back[0]["vals"][5] is None, "job garde, image perdue"


def test_unserializable_value_does_not_break_the_save():
    _isolate()
    j = _job()
    j["vals"][6] = object()                  # ni JSON ni image
    cz_ui._q_persist([j])
    back = cz_ui._q_load()
    assert len(back) == 1 and back[0]["vals"][6] is None


def test_corrupt_store_returns_empty_queue():
    _isolate()
    with open(cz_ui._Q_STORE, "w", encoding="utf-8") as f:
        f.write("{not json at all")
    assert cz_ui._q_load() == []


def test_persist_disabled():
    _isolate()
    cz_ui.Q_PERSIST = False
    try:
        cz_ui._q_persist([_job()])
        assert not os.path.exists(cz_ui._Q_STORE)
        assert cz_ui._q_load() == []
    finally:
        cz_ui.Q_PERSIST = True


def test_mutations_write_through():
    _isolate()
    items = []
    # add: la signature est (*vals, items)
    j = _job("through")
    cz_ui._ui_queue_add(*j["vals"], items)
    assert len(cz_ui._q_load()) == 1
    cz_ui._ui_queue_remove(items, 0)
    assert cz_ui._q_load() == []


if __name__ == "__main__":
    for fn in (test_roundtrip_keeps_settings_and_lora_tuples,
               test_history_is_not_persisted, test_input_image_survives_restart,
               test_missing_asset_degrades_to_none,
               test_unserializable_value_does_not_break_the_save,
               test_corrupt_store_returns_empty_queue, test_persist_disabled,
               test_mutations_write_through):
        fn()
        print(f"OK {fn.__name__}")
    print("All queue-persistence tests passed.")
