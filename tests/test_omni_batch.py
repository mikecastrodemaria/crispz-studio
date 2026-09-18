"""Reference (Omni) : lot « Image number », detaileur et « Upscale after generate », comme en
txt2img. Omni ne faisait qu'une image et ignorait ces trois reglages sans un mot (porte de
crispz-klein 1.36.3). Generation Omni, upscale et detaileur sont remplaces : ni GPU ni modele.

Run:  .venv/Scripts/python tests/test_omni_batch.py
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

import cz_detailer
import cz_pipeline
import cz_ui


def _call(**over):
    """_ui_generate en mode Reference (Omni), passe par NOMS de parametres."""
    base = dict(prompt="a car", negative="", styles=[], style_random=False, use_input=True,
                input_image=None, input_mode="Reference (Omni)",
                ref1=Image.new("RGB", (32, 32)), ref2=None, ref3=None, ref4=None,
                faceswap_enable=False, faceswap_src=None, width=64, height=64, gen_steps=4,
                image_number=1, seed=10, guidance=cz_pipeline.GUIDANCE,
                offload_mode=cz_pipeline.OFFLOAD_MODE, esrgan_model="esrgan.pth",
                do_esrgan=False, do_refine=False, refine_first=False, factor=2.0, denoise=0.3,
                refine_steps=8, tile=512, overlap=64, refine_tile=0, refine_overlap=64,
                save_mode="display", output_dir="out", output_format="png", history=[],
                auto_upscale=False, progress=lambda f, desc=None: None)
    base.update(over)
    params = inspect.signature(cz_ui._ui_generate).parameters
    return cz_ui._ui_generate(**{k: v for k, v in base.items() if k in params})


class _Stubs:
    """generate_omni, process_one et le detaileur remplaces ; un modele Omni suppose configure."""

    def __init__(self, detailer=False):
        self.omni, self.up, self.faces, self.hands = [], [], [], []
        self.detailer = detailer

    def __enter__(self):
        self.saved = (cz_ui.generate_omni, cz_ui.process_one, cz_pipeline.OMNI_MODEL,
                      cz_detailer.DETAILER_ENABLED, cz_detailer.HAND_ENABLED,
                      cz_detailer.detail_faces, cz_detailer.detail_hands)

        def omni(refs, p, n, w, h, st, seed, **kw):
            self.omni.append((p, seed))
            return Image.new("RGB", (32, 32))

        def up(img, *a, **kw):
            self.up.append(img.size)
            return Image.new("RGB", (64, 64)), {"esrgan": 0.0, "refine": 0.0}

        def faces(img, p, seed, **kw):
            self.faces.append(seed)
            return img, 1

        def hands(img, p, seed, **kw):
            self.hands.append(seed)
            return img, 1

        cz_ui.generate_omni, cz_ui.process_one = omni, up
        cz_pipeline.OMNI_MODEL = cz_pipeline.OMNI_MODEL or "test/omni"
        cz_detailer.DETAILER_ENABLED = cz_detailer.HAND_ENABLED = self.detailer
        cz_detailer.detail_faces, cz_detailer.detail_hands = faces, hands
        return self

    def __exit__(self, *exc):
        (cz_ui.generate_omni, cz_ui.process_one, cz_pipeline.OMNI_MODEL,
         cz_detailer.DETAILER_ENABLED, cz_detailer.HAND_ENABLED,
         cz_detailer.detail_faces, cz_detailer.detail_hands) = self.saved
        return False


def _seeds(n):
    return [10 + i for i in range(n)] if not cz_pipeline._NO_SEED_INCREMENT else [10] * n


def test_omni_batch_makes_n_images_with_seed_plus_i():
    with _Stubs() as st:
        gal, rep, _h, _h2 = _call(image_number=3)
    assert [s for _p, s in st.omni] == _seeds(3), st.omni
    assert len(gal) == 3 and "omni x3" in rep, rep
    assert st.up == [], st.up                          # case decochee : pas d'upscale
    print("OK test_omni_batch_makes_n_images_with_seed_plus_i")


def test_omni_upscale_only_when_asked():
    with _Stubs() as st:
        _g, rep, _h, _h2 = _call(auto_upscale=True)
    assert st.up == [(32, 32)], st.up
    assert "omni+upscale" in rep and "64x64" in rep, rep
    print("OK test_omni_upscale_only_when_asked")


def test_omni_detailer_runs_on_each_final_image():
    with _Stubs(detailer=True) as st:
        _call(image_number=2, auto_upscale=True)
    assert st.faces == _seeds(2) and st.hands == _seeds(2), (st.faces, st.hands)
    print("OK test_omni_detailer_runs_on_each_final_image")


def test_an_upscale_failure_keeps_the_omni_image():
    with _Stubs() as st:
        def boom(*a, **k):
            raise RuntimeError("esrgan missing")

        cz_ui.process_one = boom
        gal, rep, _h, _h2 = _call(auto_upscale=True)
    assert len(gal) == 1 and "upscale skipped" in rep and "32x32" in rep, rep
    assert "omni+upscale" not in rep, rep
    print("OK test_an_upscale_failure_keeps_the_omni_image")


if __name__ == "__main__":
    test_omni_batch_makes_n_images_with_seed_plus_i()
    test_omni_upscale_only_when_asked()
    test_omni_detailer_runs_on_each_final_image()
    test_an_upscale_failure_keeps_the_omni_image()
    print("All Omni batch tests passed.")
