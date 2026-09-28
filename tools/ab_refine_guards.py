# -*- coding: utf-8 -*-
"""An A/B of the TILE refine: whole-image vs tile, with and without the
anti-duplication guard rails.

It answers three questions the timings alone do not settle:

    A  whole-image       : a single pass (no duplication possible) = the reference
    B  tile + guard rails: an empty prompt per tile + a capped denoise = the shipped behaviour
    C  tile WITHOUT the guard rails: the global prompt on every tile, the raw denoise =
                           what they avoid

C must show the subject DUPLICATED in the tiles that are nothing but background. Should B
and C look alike, it means the prompt describes the subject too little on that image --
not that the guard rails are broken.

A TRAP when comparing A and B: the cap makes B run at the CAPPED denoise, not at the
denoise asked for. So comparing A at 0.60 with B asked at 0.60 mixes two effects. To
isolate the tiling, re-run A at B's EFFECTIVE denoise (= the cap) -- hence --only, and
--recrop, which re-crops without re-diffusing anything.

Measured on a 4096x4096 output (RTX 5090, Z-Image, 8 steps): whole-image 581s against
~37s in tile 896 at an equal effective denoise, and as a bonus less mottled skin and the
subject's geometry preserved (the 4K whole-image moves and shrinks the subject).

Run:  .venv/Scripts/python tools/ab_refine_guards.py --src <image_4k.png>
      .venv/Scripts/python tools/ab_refine_guards.py --src <img> --only A --denoise 0.40
      .venv/Scripts/python tools/ab_refine_guards.py --recrop

"""
import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

import cz_pipeline as P  # noqa: E402

# Under out/ (gitignored): the trial renders do not end up in the repo.
DEFAULT_OUT = os.path.join("out", "ab_refine_guards")
# A prompt that DESCRIBES THE SUBJECT: it is the one the diffusion will copy into
# every tile when it is passed as it is (case C). A landscape prompt would trigger nothing.
PROMPT = ("a young woman holding a straw broom, standing in a misty forest, "
          "shallow depth of field, natural light")
SEED, STEPS, OVERLAP = 1234, 8, 64

# 100% crops at the same framing for every variant, as a fraction of the image (so as
# to follow any source size). Each one answers a precise question:
#   subject = the subject: judges the skin, the hair, the fine detail, the geometry
#   seam    = falls on a tile crossing for a 4096 output in tile 896
#             (approximate elsewhere): that is where a seam would show
#   bg      = pure background, far from the subject: that is where the duplication
#             appears (case C)
CROPS_REL = {"subject": (0.354, 0.049, 0.604, 0.299),
             "seam": (0.281, 0.281, 0.531, 0.531),
             "bg": (0.062, 0.586, 0.312, 0.836)}


def _boxes(size):
    w, h = size
    return {k: (int(a * w), int(b * h), int(c * w), int(d * h))
            for k, (a, b, c, d) in CROPS_REL.items()}


def _crops_for(path, out_dir):
    im = Image.open(path).convert("RGB")
    stem = os.path.splitext(os.path.basename(path))[0]
    for name, box in _boxes(im.size).items():
        im.crop(box).save(os.path.join(out_dir, "crops", f"{stem}__{name}.png"))
    return len(CROPS_REL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="the ALREADY upscaled image (4K+) to refine; required unless --recrop")
    ap.add_argument("--ckpt", help="the checkpoint to use (default: the one already configured)")
    ap.add_argument("--denoise", type=float, default=0.60,
                    help="for B/C: above the cap, otherwise it never engages")
    ap.add_argument("--only", action="append", choices=["A", "B", "C"],
                    help="render only these variants (repeatable)")
    ap.add_argument("--recrop", action="store_true",
                    help="diffuse nothing: re-crop the PNGs already rendered")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--overlap", type=int, default=OVERLAP)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()

    os.makedirs(os.path.join(a.out, "crops"), exist_ok=True)

    if a.recrop:
        n = 0
        for p in sorted(glob.glob(os.path.join(a.out, "[ABC]_*.png"))):
            n += _crops_for(p, a.out)
            print(f"re-cropped {os.path.basename(p)}")
        print(f"{n} crop(s) regenerated in {os.path.join(a.out, 'crops')}")
        return

    assert a.src, "--src is required (the 4K image to refine)"
    want = set(a.only or ["A", "B", "C"])
    if want & {"B", "C"}:
        assert a.denoise > P._TILE_DENOISE_CAP, (
            f"--denoise {a.denoise} <= the cap {P._TILE_DENOISE_CAP}: the guard rail "
            "would not engage and the A/B would show nothing")

    img = Image.open(a.src).convert("RGB")
    print(f"source {img.size} | denoise {a.denoise} | cap {P._TILE_DENOISE_CAP} "
          f"| per-tile prompt = {P._TILE_PROMPT!r} | variants {sorted(want)}")

    if a.ckpt:
        P.set_zimage_transformer(a.ckpt)
    pipe = P.load_pipe()
    tile = P._pick_refine_tile(img.width, img.height, a.overlap)
    print(f"tile picked by the auto-tiling: {tile}")

    runs = []
    if "A" in want:
        runs.append(("A_whole_image", lambda: P._refine_whole(
            pipe, img, a.denoise, a.steps, a.prompt, a.seed)))
    if "B" in want:
        runs.append((f"B_tiled{tile}_guards", lambda: P._refine_tiled(
            pipe, img, a.denoise, a.steps, a.prompt, a.seed, tile, a.overlap)))
    if "C" in want:
        runs.append((f"C_tiled{tile}_no_guards", lambda: _tiled_unguarded(
            pipe, img, a.denoise, a.steps, a.prompt, a.seed, tile, a.overlap)))

    for name, fn in runs:
        t0 = time.time()
        try:
            out = fn()
        except torch.cuda.OutOfMemoryError:
            print(f"{name}: OOM -- which is exactly why the auto-tiling exists")
            torch.cuda.empty_cache()
            continue
        dt = time.time() - t0
        path = os.path.join(a.out, f"{name}_den{a.denoise}.png")
        out.save(path)
        _crops_for(path, a.out)
        print(f"{name}: {dt:.1f}s -> {os.path.basename(path)}")

    print(f"\nImages in {a.out} -- compare the 'bg' crop: if the subject reappears "
          "there, the guard rails are doing their job.")


def _tiled_unguarded(pipe, image, denoise, steps, prompt, seed, tile, overlap):
    """_refine_tiled with both guard rails NEUTRALISED (the global prompt, the raw denoise)."""
    saved_prompt, saved_cap = P._TILE_PROMPT, P._TILE_DENOISE_CAP
    P._TILE_PROMPT, P._TILE_DENOISE_CAP = "global", 0.0
    try:
        return P._refine_tiled(pipe, image, denoise, steps, prompt, seed, tile, overlap)
    finally:
        P._TILE_PROMPT, P._TILE_DENOISE_CAP = saved_prompt, saved_cap


if __name__ == "__main__":
    main()
