#!/usr/bin/env python3
"""Smoke / non-regression test for crispz-studio (no GPU, no model load).

Exercises the pure helpers + UI build. Run with the project venv:
  .venv\\Scripts\\python tools\\smoke_test.py
Exit 0 if all checks pass, 1 otherwise.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app  # noqa: E402
from PIL import Image  # noqa: E402

ok = 0
fail = 0


def check(name, cond):
    global ok, fail
    if cond:
        ok += 1
        print(f"  [PASS] {name}")
    else:
        fail += 1
        print(f"  [FAIL] {name}")


print("== crispz-studio smoke test ==")

# Styles: prompt template + merged negative
p, n = app._apply_styles("a cat", "blurry", ["Fooocus Cinematic"]) if "Fooocus Cinematic" in app.STYLES \
    else app._apply_styles("a cat", "blurry", list(app.STYLES)[:1])
check("apply_styles returns (prompt, negative)", isinstance(p, str) and isinstance(n, str) and "blurry" in n)

# Filenames
fn = app._format_filename("txt2img", 42, 1024, 1024, 0)
check("format_filename dated+seed+size", "txt2img" in fn and "seed42" in fn and "1024x1024" in fn)

# Editor image extraction
im = Image.new("RGB", (8, 8))
check("editor_img dict", app._editor_img({"composite": im, "background": None}) is im)
check("editor_img pil", app._editor_img(im) is im)
check("editor_img none", app._editor_img(None) is None)

# Mask extraction
from PIL import ImageDraw  # noqa: E402
bg = Image.new("RGB", (64, 64)); comp = bg.copy()
ImageDraw.Draw(comp).rectangle([10, 10, 40, 40], fill=(255, 255, 255))
_, mask = app._editor_to_image_mask({"background": bg, "composite": comp, "layers": []})
check("inpaint mask non-empty", mask is not None and mask.getbbox() is not None)

# Reframe canvas alignment (32)
c, m, nw, nh = app._reframe_canvas(Image.new("RGB", (512, 512)), 16, 9)
check("reframe canvas 32-aligned", nw % 32 == 0 and nh % 32 == 0 and nw >= 512)

# Metadata round-trip (png/jpg/webp) without sidecar
d = tempfile.mkdtemp()
meta = app._gen_meta("txt2img", "a red car", "blurry", 7, 8, 0.0, (64, 64))
for ext in ("png", "jpg", "webp"):
    fp = os.path.join(d, "t." + ext)
    app.save_image(Image.new("RGB", (64, 64), (200, 30, 30)), fp, ext, meta=meta)
    if os.path.isfile(fp + ".json"):
        os.remove(fp + ".json")  # force read from embedded
    got = app._read_image_meta(fp).get("prompt")
    check(f"metadata embedded+read ({ext})", got == "a red car")

# LoRA spec / multi-lora
app.set_loras([("a.safetensors", 0.8), ("None", 1.0), ("b.safetensors", 0.5)])
check("multi-lora set (skip None)", len(app.LORAS) == 2)
app.set_loras([("None", 1.0)])
check("lora clear", app.LORAS == [])

# Setters
check("set_log_level", "debug" in app.set_log_level("debug").lower())
app.set_log_level("info")
check("set_faceswap_restore", app.set_faceswap_restore(True, 0.8) and app.FACESWAP_RESTORE)
app.set_faceswap_restore(False, 0.8)

# Compose instruction has placeholder
check("compose instruction has {descriptions}", "{descriptions}" in app.COMPOSE_INSTRUCTION)
check("improve instruction has {prompt}", "{prompt}" in app.IMPROVE_INSTRUCTION)

# Gallery load (out folder may be empty -> still returns a tuple)
g, st, msg = app._gallery_load("out", "Newest", "")
check("gallery_load returns list+status", isinstance(g, list) and "image" in msg)

# Config
check("CONFIG loaded", isinstance(app.CONFIG, dict) and len(app.CONFIG) > 5)

# UI builds
try:
    app.build_ui()
    check("build_ui()", True)
except Exception as e:
    check(f"build_ui() ({e})", False)

print(f"\n== {ok} passed, {fail} failed ==")
sys.exit(0 if fail == 0 else 1)
