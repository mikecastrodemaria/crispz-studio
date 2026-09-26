"""Tests <lora:...> in-prompt tags (family CLI protocol). No torch, no GPU.
Run:  .venv/Scripts/python tests/test_protocol_loras.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_protocol as cp  # noqa: E402


def test_lora_tag_with_weight_moves_to_loras_and_cleans_prompt():
    spec, _w = cp.validate_spec(
        {"protocol": 1,
         "prompt": "a knight in the rain <lora:ink-style.safetensors:0.8>, "
                   "dramatic light"})
    assert spec["loras"] == ["ink-style.safetensors:0.8"]
    assert "<lora" not in spec["prompt"]
    assert spec["prompt"] == "a knight in the rain, dramatic light"


def test_lora_tag_without_weight_and_case_insensitive():
    spec, _w = cp.validate_spec(
        {"protocol": 1, "prompt": "<LORA:flat>: a cat"})
    assert spec["loras"] == ["flat"]
    assert spec["prompt"] == ": a cat".strip(" ,") or spec["prompt"]
    assert "<" not in spec["prompt"]


def test_explicit_spec_loras_win_over_prompt_tags():
    spec, _w = cp.validate_spec(
        {"protocol": 1, "loras": ["ink-style.safetensors:1.0"],
         "prompt": "a cat <lora:ink-style.safetensors:0.3> "
                   "<lora:extra:0.5>"})
    # same file already explicit -> the explicit weight wins; the other is added
    assert spec["loras"] == ["ink-style.safetensors:1.0", "extra:0.5"]


def test_prompt_with_only_lora_tags_is_a_clean_error():
    try:
        cp.validate_spec({"protocol": 1, "prompt": "<lora:a:0.5> <lora:b>"})
    except cp.SpecError as e:
        assert e.code == 2 and "only" in str(e)
    else:
        raise AssertionError("tags-only prompt should raise")


def test_prompt_without_tags_untouched():
    spec, _w = cp.validate_spec(
        {"protocol": 1, "prompt": "a cat < not a tag > 1:2"})
    assert spec["prompt"] == "a cat < not a tag > 1:2"
    assert spec["loras"] == []


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"All {len(tests)} lora-tag tests passed.")
