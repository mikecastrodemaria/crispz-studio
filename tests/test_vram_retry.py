"""Manque de VRAM : vidage + nouvel essai (porte de crispz-klein 1.36.4).

Sur crispz-klein, un lot Reference (Omni) avec Upscale after generate et le detaileur
manquait de VRAM a la 4e image, puis chaque rendu echouait jusqu'au redemarrage.
Couvre :
  - is_oom reconnait les deux formes (allocateur de torch, appel CUDA direct) ;
  - retry_on_oom vide la VRAM et retente UNE fois, la revide si le nouvel essai echoue,
    et laisse passer les autres erreurs ;
  - release_vram remet sur le CPU chaque pipeline charge avec ses hooks (offload model),
    une seule fois par objet ;
  - le detaileur retente une passe, puis saute les zones restantes s'il manque encore ;
  - la boucle Omni de l'UI retente, et dit quoi baisser si le nouvel essai echoue.

Ni GPU ni modele : les appels au pipeline sont remplaces.

Run:  .venv/Scripts/python tests/test_vram_retry.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image  # noqa: E402

import cz_pipeline  # noqa: E402
import cz_detailer  # noqa: E402
import cz_ui  # noqa: E402
from test_omni_batch import _call, _Stubs  # noqa: E402

OOM = "CUDA error: out of memory\nCUDA kernel errors might be asynchronously reported"


class _Releases:
    """Remplace cz_pipeline.release_vram le temps d'un test et compte les appels."""

    def __init__(self):
        self.calls = []

    def __enter__(self):
        self.real = cz_pipeline.release_vram
        cz_pipeline.release_vram = lambda offload=False, why="": self.calls.append(offload)
        return self

    def __exit__(self, *exc):
        cz_pipeline.release_vram = self.real
        return False


def test_is_oom_matches_both_forms():
    assert cz_pipeline.is_oom(RuntimeError(OOM))
    assert cz_pipeline.is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert cz_pipeline.is_oom(RuntimeError("CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate"))
    assert not cz_pipeline.is_oom(RuntimeError("CUDA error: an illegal memory access"))
    assert not cz_pipeline.is_oom(ValueError("Edit needs at least one input image."))


def test_retry_on_oom_frees_the_vram_and_retries_once():
    tries = []

    def flaky(a, b=0):
        tries.append((a, b))
        if len(tries) == 1:
            raise RuntimeError(OOM)
        return a + b

    with _Releases() as rel:
        assert cz_pipeline.retry_on_oom("test", flaky, 2, b=3) == 5
    assert tries == [(2, 3), (2, 3)], tries
    assert rel.calls == [True], rel.calls          # les poids reviennent aussi sur le CPU


def test_retry_on_oom_frees_again_when_the_retry_fails():
    tries = []

    def always(*a):
        tries.append(a)
        raise RuntimeError(OOM)

    with _Releases() as rel:
        try:
            cz_pipeline.retry_on_oom("test", always, 1)
        except RuntimeError as e:
            assert cz_pipeline.is_oom(e), e
        else:
            raise AssertionError("le second echec doit remonter")
    assert len(tries) == 2, tries                  # un seul nouvel essai, pas une boucle
    assert rel.calls == [True, True], rel.calls    # revidee avant de remonter l'erreur


def test_retry_on_oom_leaves_other_errors_alone():
    tries = []

    def broken():
        tries.append(1)
        raise ValueError("bad input")

    with _Releases() as rel:
        try:
            cz_pipeline.retry_on_oom("test", broken)
        except ValueError:
            pass
        else:
            raise AssertionError("la ValueError doit passer")
    assert tries == [1] and rel.calls == [], (tries, rel.calls)


def test_release_vram_offloads_every_hooked_pipe_once():
    class FakePipe:
        def __init__(self, hooks):
            self._all_hooks, self.freed = hooks, 0

        def maybe_free_model_hooks(self):
            self.freed += 1

    saved = (cz_pipeline._BASE_PIPE, cz_pipeline._DERIVED)
    try:
        base, derived, omni = FakePipe(["hook"]), FakePipe([]), FakePipe(["hook"])
        cz_pipeline._BASE_PIPE = base
        cz_pipeline._DERIVED = {"txt2img": base, "img2img": derived, "omni": omni}
        cz_pipeline.release_vram()
        assert (base.freed, omni.freed) == (0, 0)  # un simple vidage garde les poids
        cz_pipeline.release_vram(offload=True)
        assert (base.freed, derived.freed, omni.freed) == (1, 0, 1), \
            (base.freed, derived.freed, omni.freed)
        cz_pipeline._BASE_PIPE, cz_pipeline._DERIVED = None, {}
        cz_pipeline.release_vram(offload=True)     # rien de charge : pas d'erreur
    finally:
        cz_pipeline._BASE_PIPE, cz_pipeline._DERIVED = saved


def _detail(refine):
    """cz_detailer._detail_regions sur deux zones, avec une passe de refine simulee."""
    real = (cz_pipeline.get_pipe, cz_pipeline._refine_whole)
    cz_pipeline.get_pipe = lambda kind="img2img": object()
    cz_pipeline._refine_whole = refine
    try:
        with _Releases() as rel:
            img, done = cz_detailer._detail_regions(
                Image.new("RGB", (512, 512)), [(40, 40, 140, 140), (300, 300, 400, 400)],
                "a face", 7, 4, 0.3, "face", min_size=10)
    finally:
        cz_pipeline.get_pipe, cz_pipeline._refine_whole = real
    return img, done, rel.calls


def test_detailer_retries_a_pass_that_ran_out_of_vram():
    tries = []

    def refine(pipe, work, denoise, steps, prompt, seed):
        tries.append(seed)
        if len(tries) == 1:
            raise RuntimeError(OOM)
        return work

    img, done, releases = _detail(refine)
    assert done == 2 and len(tries) == 3, (done, tries)   # zone 1 deux fois, zone 2 une
    assert releases == [True], releases
    assert img.size == (512, 512)


def test_detailer_skips_the_other_regions_when_still_out_of_vram():
    tries = []

    def refine(pipe, work, denoise, steps, prompt, seed):
        tries.append(seed)
        raise RuntimeError(OOM)

    img, done, releases = _detail(refine)
    # Zone 1 : essai + nouvel essai. La zone 2 n'est pas tentee : elle echouerait pareil.
    assert done == 0 and len(tries) == 2, (done, tries)
    assert releases == [True, True], releases
    assert img.size == (512, 512)


def test_ui_omni_retries_after_running_out_of_vram():
    tries = []

    def flaky(refs, p, n, w, h, st, seed, **kw):
        tries.append(seed)
        if len(tries) == 1:
            raise RuntimeError(OOM)
        return Image.new("RGB", (32, 32))

    with _Stubs(), _Releases() as rel:
        cz_ui.generate_omni = flaky                # _Stubs remet l'original en sortie
        gal, rep = _call(image_number=1)[:2]
    assert tries == [10, 10] and len(gal) == 1, (tries, gal)
    assert "VRAM" not in rep, rep
    # Le vidage du nouvel essai (poids sur le CPU), puis celui entre deux images.
    assert rel.calls == [True, False], rel.calls


def test_ui_omni_reports_what_to_lower_when_the_retry_fails():
    def always(refs, p, n, w, h, st, seed, **kw):
        raise RuntimeError(OOM)

    with _Stubs(), _Releases() as rel:
        cz_ui.generate_omni = always
        gal, rep = _call(image_number=1)[:2]
    assert not gal, gal
    assert "Omni error" in rep and "VRAM full" in rep and "restart" in rep, rep
    assert rel.calls == [True, True], rel.calls


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"All {len(tests)} VRAM retry tests passed.")
