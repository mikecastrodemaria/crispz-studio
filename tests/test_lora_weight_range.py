"""Unit tests for the configurable LoRA weight range (lora_weight_min / lora_weight_max).

Run:  .venv/Scripts/python tests/test_lora_weight_range.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_pipeline as P  # noqa: E402


def _range_with(cfg):
    """Recalcule les bornes avec un CONFIG temporaire."""
    old = P.CONFIG
    P.CONFIG = cfg
    try:
        return P._lora_weight_range()
    finally:
        P.CONFIG = old


def test_default_is_symmetric_and_allows_negatives():
    assert _range_with({}) == (-2.0, 2.0)


def test_custom_range_is_honoured():
    assert _range_with({"lora_weight_min": -1.0, "lora_weight_max": 1.5}) == (-1.0, 1.5)
    # interdire les negatifs = min a 0
    assert _range_with({"lora_weight_min": 0, "lora_weight_max": 2}) == (0.0, 2.0)


def test_ints_and_strings_are_accepted():
    assert _range_with({"lora_weight_min": -3, "lora_weight_max": 3}) == (-3.0, 3.0)
    assert _range_with({"lora_weight_min": "-1.5", "lora_weight_max": "1.5"}) == (-1.5, 1.5)


def test_bad_values_fall_back_to_default():
    assert _range_with({"lora_weight_min": "abc", "lora_weight_max": 2}) == (-2.0, 2.0)
    assert _range_with({"lora_weight_min": None, "lora_weight_max": None}) == (-2.0, 2.0)


def test_inverted_or_empty_range_falls_back():
    assert _range_with({"lora_weight_min": 2, "lora_weight_max": -2}) == (-2.0, 2.0)   # inverse
    assert _range_with({"lora_weight_min": 1, "lora_weight_max": 1}) == (-2.0, 2.0)    # vide


def test_module_exposes_range_and_clamps_default_weight():
    assert hasattr(P, "LORA_WEIGHT_MIN") and hasattr(P, "LORA_WEIGHT_MAX")
    assert P.LORA_WEIGHT_MIN < P.LORA_WEIGHT_MAX
    # le poids par defaut doit etre dans les bornes (sinon curseur hors plage)
    assert P.LORA_WEIGHT_MIN <= P.LORA_WEIGHT <= P.LORA_WEIGHT_MAX


def test_negative_weight_survives_set_loras():
    """Un poids negatif ne doit etre ni rejete ni clampe par la couche modele."""
    import tempfile
    d = tempfile.mkdtemp()
    p = os.path.join(d, "slider.safetensors")
    with open(p, "wb") as f:
        f.write(b"x")
    P.LORAS = []
    P.set_loras([(p, -1.0)])
    assert P.LORAS == [(p, -1.0)]
    P.LORAS = []


if __name__ == "__main__":
    for fn in (test_default_is_symmetric_and_allows_negatives, test_custom_range_is_honoured,
               test_ints_and_strings_are_accepted, test_bad_values_fall_back_to_default,
               test_inverted_or_empty_range_falls_back,
               test_module_exposes_range_and_clamps_default_weight,
               test_negative_weight_survives_set_loras):
        fn()
        print(f"OK {fn.__name__}")
    print("All lora weight range tests passed.")
