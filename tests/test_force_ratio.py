"""Unit tests for the forced-aspect-ratio modes (crop / extend) on Upscale/img2img.
The extend path stubs outpaint_directions (no GPU): only the geometry is checked.

Run:  .venv/Scripts/python tests/test_force_ratio.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

import cz_pipeline as czp  # noqa: E402


def _stub_outpaint(image, mask, directions, prompt, steps, seed, strength=1.0, expand=0.3):
    """Reproduit UNIQUEMENT le padding geometrique d'outpaint_directions (pas de GPU)."""
    import numpy as np
    img = np.array(image.convert("RGB"))
    H, W = img.shape[:2]
    d = set(x.lower() for x in directions)
    if "top" in d:
        img = np.pad(img, [[int(H * expand), 0], [0, 0], [0, 0]], mode="edge")
    if "bottom" in d:
        img = np.pad(img, [[0, int(H * expand)], [0, 0], [0, 0]], mode="edge")
    if "left" in d:
        img = np.pad(img, [[0, 0], [int(W * expand), 0], [0, 0]], mode="edge")
    if "right" in d:
        img = np.pad(img, [[0, 0], [0, int(W * expand)], [0, 0]], mode="edge")
    return Image.fromarray(np.ascontiguousarray(img))


def test_extend_reaches_target_ratio():
    old = czp.outpaint_directions
    czp.outpaint_directions = _stub_outpaint
    old_dn = czp.EXTEND_DENOISE
    czp.EXTEND_DENOISE = 0.0        # pas de passe d'harmonisation GPU dans le test
    try:
        # portrait 512x768 -> 16:9 : elargit, ne coupe rien
        out = czp._extend_to_ratio(Image.new("RGB", (512, 768)), 16, 9, "", 6, 1)
        assert out.size[1] == 768                      # hauteur intacte
        assert abs(out.size[0] / out.size[1] - 16 / 9) < 0.02
        # paysage 1024x512 -> 1:1 : etend en hauteur
        out = czp._extend_to_ratio(Image.new("RGB", (1024, 512)), 1, 1, "", 6, 1)
        assert out.size[0] == 1024
        assert abs(out.size[0] / out.size[1] - 1.0) < 0.02
        # deja au ratio -> intact, zero outpaint
        out = czp._extend_to_ratio(Image.new("RGB", (1024, 1024)), 1, 1, "", 6, 1)
        assert out.size == (1024, 1024)
    finally:
        czp.outpaint_directions = old
        czp.EXTEND_DENOISE = old_dn


def test_crop_still_crops():
    out = czp._crop_to_ratio(Image.new("RGB", (512, 768)), 16, 9)
    assert out.size[0] == 512 and out.size[1] < 768
    assert abs(out.size[0] / out.size[1] - 16 / 9) < 0.02


def test_mode_setter_normalises():
    old = czp.FORCE_RATIO_MODE
    try:
        czp.set_force_ratio_mode("EXTEND")
        assert czp.FORCE_RATIO_MODE == "extend"
        czp.set_force_ratio_mode("nimporte quoi")
        assert czp.FORCE_RATIO_MODE == "crop"
        czp.set_force_ratio_mode(None)
        assert czp.FORCE_RATIO_MODE == "crop"
    finally:
        czp.FORCE_RATIO_MODE = old


def test_ui_radio_mapping():
    import cz_ui
    old_r, old_m = czp.FORCE_RATIO, czp.FORCE_RATIO_MODE
    try:
        cz_ui._ui_set_force_ratio("Extend (outpaint)", "1152 x 896  (9:7)")
        assert czp.FORCE_RATIO and czp.FORCE_RATIO_MODE == "extend"
        cz_ui._ui_set_force_ratio("Crop to fit", "1152 x 896  (9:7)")
        assert czp.FORCE_RATIO and czp.FORCE_RATIO_MODE == "crop"
        cz_ui._ui_set_force_ratio("Off", "1152 x 896  (9:7)")
        assert czp.FORCE_RATIO == ""
    finally:
        czp.FORCE_RATIO, czp.FORCE_RATIO_MODE = old_r, old_m


if __name__ == "__main__":
    for fn in (test_extend_reaches_target_ratio, test_crop_still_crops,
               test_mode_setter_normalises, test_ui_radio_mapping):
        fn()
        print(f"OK {fn.__name__}")
    print("All force-ratio tests passed.")
