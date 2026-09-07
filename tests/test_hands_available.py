"""Le detailer de mains etait declare absent alors qu'il etait pret a tourner.

A l'execution il ne demande qu'onnxruntime + un .onnx exporte une fois dans cache/.
'ultralytics' ne sert qu'a cet export, et la docstring de cz_detailer._ensure_hand_onnx
interdit formellement de le laisser vivre dans le process de diffusion (il corrompt
les poids partages pendant les transferts d'offload). Gater la feature sur l'import
d'ultralytics punissait donc exactement ceux qui avaient suivi ce conseil.

Run:  .venv/Scripts/python tests/test_hands_available.py
"""
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_core
import cz_detailer
import cz_protocol as C


def _onnx_path():
    stem = os.path.splitext(os.path.basename(cz_detailer._HAND_MODEL))[0]
    return os.path.join(cz_core.HERE, "cache", stem + ".onnx")


def _with_find_spec(missing, fn):
    """Execute fn en faisant disparaitre les modules nommes dans `missing`."""
    real = importlib.util.find_spec

    def fake(name, *a, **k):
        return None if name in missing else real(name, *a, **k)
    importlib.util.find_spec = fake
    try:
        return fn()
    finally:
        importlib.util.find_spec = real


def test_exported_onnx_is_enough():
    """Sans ultralytics mais avec le .onnx: la feature EST disponible."""
    if not os.path.isfile(_onnx_path()):
        print("SKIP test_exported_onnx_is_enough (no exported detector in cache/)")
        return
    assert _with_find_spec({"ultralytics"}, C._hands_available) is True
    print("OK test_exported_onnx_is_enough")


def test_ultralytics_alone_is_enough():
    """Avec ultralytics, l'export peut se faire a la demande -> disponible."""
    real = importlib.util.find_spec

    def fake(name, *a, **k):
        return object() if name == "ultralytics" else real(name, *a, **k)
    importlib.util.find_spec = fake
    try:
        assert C._hands_available() is True
    finally:
        importlib.util.find_spec = real
    print("OK test_ultralytics_alone_is_enough")


def test_no_onnxruntime_means_no():
    """onnxruntime est la seule dependance vraiment indispensable a l'execution."""
    assert _with_find_spec({"onnxruntime"}, C._hands_available) is False
    print("OK test_no_onnxruntime_means_no")


if __name__ == "__main__":
    test_exported_onnx_is_enough()
    test_ultralytics_alone_is_enough()
    test_no_onnxruntime_means_no()
    print("ALL OK")
