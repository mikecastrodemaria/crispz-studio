# -*- coding: utf-8 -*-
"""A/B du refine TUILE: whole-image vs tuile, avec et sans les garde-fous anti-duplication.

Repond a trois questions que les chronos seuls ne tranchent pas:

    A  whole-image        : une seule passe (aucune duplication possible) = la reference
    B  tuile + garde-fous : prompt vide par tuile + denoise plafonne = le comportement livre
    C  tuile SANS garde-fous : prompt global a chaque tuile, denoise brut = ce qu'ils evitent

C doit montrer le sujet DUPLIQUE dans les tuiles qui ne sont que du fond. Si B et C se
ressemblent, c'est que le prompt decrit trop peu le sujet sur cette image -- pas que les
garde-fous sont casses.

PIEGE en comparant A et B: le plafond fait tourner B au denoise PLAFONNE, pas au denoise
demande. Comparer A a 0.60 et B demande a 0.60 melange donc deux effets. Pour isoler le
tuilage, relancer A au denoise EFFECTIF de B (= le plafond) -- d'ou --only, et --recrop
qui recoupe sans rien rediffuser.

Mesure sur une sortie 4096x4096 (RTX 5090, Z-Image, 8 steps): whole-image 581s contre
~37s en tuile 896 a denoise effectif egal, avec en prime une peau moins marbree et la
geometrie du sujet preservee (le whole-image 4K deplace et retrecit le sujet).

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

# Sous out/ (gitignore): les rendus d'essai ne partent pas dans le depot.
DEFAULT_OUT = os.path.join("out", "ab_refine_guards")
# Un prompt qui DECRIT LE SUJET: c'est lui que la diffusion recopiera dans chaque tuile
# si on le passe tel quel (cas C). Un prompt de paysage ne declencherait rien.
PROMPT = ("a young woman holding a straw broom, standing in a misty forest, "
          "shallow depth of field, natural light")
SEED, STEPS, OVERLAP = 1234, 8, 64

# Crops 100% au meme cadrage pour toutes les variantes, en fraction de l'image (pour
# suivre n'importe quelle taille de source). Chacun repond a une question precise:
#   subject = le sujet: juge la peau, les cheveux, le detail fin, la geometrie
#   seam    = tombe sur un croisement de tuiles pour une sortie 4096 en tuile 896
#             (approximatif ailleurs): c'est la que se verrait une couture
#   bg      = fond pur, loin du sujet: c'est la qu'apparait la duplication (cas C)
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
    ap.add_argument("--src", help="image DEJA agrandie (4K+) a raffiner; requis sauf --recrop")
    ap.add_argument("--ckpt", help="checkpoint a utiliser (defaut: celui deja configure)")
    ap.add_argument("--denoise", type=float, default=0.60,
                    help="pour B/C: au-dessus du plafond, sinon il ne s'engage pas")
    ap.add_argument("--only", action="append", choices=["A", "B", "C"],
                    help="ne rendre que ces variantes (repetable)")
    ap.add_argument("--recrop", action="store_true",
                    help="ne rien diffuser: recouper les PNG deja rendus")
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
            print(f"recoupe {os.path.basename(p)}")
        print(f"{n} crop(s) regeneres dans {os.path.join(a.out, 'crops')}")
        return

    assert a.src, "--src est requis (l'image 4K a raffiner)"
    want = set(a.only or ["A", "B", "C"])
    if want & {"B", "C"}:
        assert a.denoise > P._TILE_DENOISE_CAP, (
            f"--denoise {a.denoise} <= plafond {P._TILE_DENOISE_CAP}: le garde-fou ne "
            "s'engagerait pas et l'A/B ne montrerait rien")

    img = Image.open(a.src).convert("RGB")
    print(f"source {img.size} | denoise {a.denoise} | plafond {P._TILE_DENOISE_CAP} "
          f"| prompt par tuile = {P._TILE_PROMPT!r} | variantes {sorted(want)}")

    if a.ckpt:
        P.set_zimage_transformer(a.ckpt)
    pipe = P.load_pipe()
    tile = P._pick_refine_tile(img.width, img.height, a.overlap)
    print(f"tuile choisie par l'auto: {tile}")

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
            print(f"{name}: OOM -- c'est exactement pourquoi l'auto-tuilage existe")
            torch.cuda.empty_cache()
            continue
        dt = time.time() - t0
        path = os.path.join(a.out, f"{name}_den{a.denoise}.png")
        out.save(path)
        _crops_for(path, a.out)
        print(f"{name}: {dt:.1f}s -> {os.path.basename(path)}")

    print(f"\nImages dans {a.out} -- comparer le crop 'bg': si le sujet y reapparait, "
          "les garde-fous font bien leur travail.")


def _tiled_unguarded(pipe, image, denoise, steps, prompt, seed, tile, overlap):
    """_refine_tiled avec les deux garde-fous NEUTRALISES (prompt global, denoise brut)."""
    saved_prompt, saved_cap = P._TILE_PROMPT, P._TILE_DENOISE_CAP
    P._TILE_PROMPT, P._TILE_DENOISE_CAP = "global", 0.0
    try:
        return P._refine_tiled(pipe, image, denoise, steps, prompt, seed, tile, overlap)
    finally:
        P._TILE_PROMPT, P._TILE_DENOISE_CAP = saved_prompt, saved_cap


if __name__ == "__main__":
    main()
