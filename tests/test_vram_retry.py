"""Out of VRAM: an emptying + a retry (ported from crispz-klein 1.36.4).

On crispz-klein, a Reference (Omni) batch with Upscale after generate and the detailer
ran out of VRAM at the 4th image, then every render failed until the restart.
It covers:
  - is_oom recognises both forms (torch's allocator, a direct CUDA call);
  - retry_on_oom frees the VRAM and retries ONCE, frees it again when the retry fails,
    and lets the other errors through;
  - release_vram puts back on the CPU every pipeline loaded with its hooks (offload model),
    once per object only;
  - the detailer retries a pass, then skips the remaining areas when it is still short;
  - the UI's Omni loop retries, and says what to lower when the retry fails.

Neither a GPU nor a model: the pipeline calls are stubbed.

Run:  .venv/Scripts/python tests/test_vram_retry.py

"""
import importlib.util
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

# cz_detailer._feather_mask needs cv2, which arrives with the FaceSwap deps
# (insightface) and is absent from an install without them, and from the CI runner.
_HAS_CV2 = importlib.util.find_spec("cv2") is not None


class _Releases:
    """Replaces cz_pipeline.release_vram for the length of a test and counts the calls."""

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
    assert rel.calls == [True], rel.calls          # the weights come back onto the CPU too


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
            raise AssertionError("the second failure must propagate")
    assert len(tries) == 2, tries                  # a single retry, not a loop
    assert rel.calls == [True, True], rel.calls    # emptied again before the error is raised


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
            raise AssertionError("the ValueError must pass through")
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
        assert (base.freed, omni.freed) == (0, 0)  # a plain emptying keeps the weights
        cz_pipeline.release_vram(offload=True)
        assert (base.freed, derived.freed, omni.freed) == (1, 0, 1), \
            (base.freed, derived.freed, omni.freed)
        cz_pipeline._BASE_PIPE, cz_pipeline._DERIVED = None, {}
        cz_pipeline.release_vram(offload=True)     # nothing loaded: no error
    finally:
        cz_pipeline._BASE_PIPE, cz_pipeline._DERIVED = saved


def _detail(refine):
    """cz_detailer._detail_regions on two areas, with a simulated refine pass."""
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
    if not _HAS_CV2:
        print("SKIP test_detailer_retries_a_pass_that_ran_out_of_vram (no cv2)")
        return
    tries = []

    def refine(pipe, work, denoise, steps, prompt, seed):
        tries.append(seed)
        if len(tries) == 1:
            raise RuntimeError(OOM)
        return work

    img, done, releases = _detail(refine)
    assert done == 2 and len(tries) == 3, (done, tries)   # area 1 twice, area 2 once
    assert releases == [True], releases
    assert img.size == (512, 512)


def test_detailer_skips_the_other_regions_when_still_out_of_vram():
    if not _HAS_CV2:
        print("SKIP test_detailer_skips_the_other_regions_when_still_out_of_vram (no cv2)")
        return
    tries = []

    def refine(pipe, work, denoise, steps, prompt, seed):
        tries.append(seed)
        raise RuntimeError(OOM)

    img, done, releases = _detail(refine)
    # Area 1: a try + a retry. Area 2 is not attempted: it would fail the same way.
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
        cz_ui.generate_omni = flaky                # _Stubs puts the original back on the way out
        gal, rep = _call(image_number=1)[:2]
    assert tries == [10, 10] and len(gal) == 1, (tries, gal)
    assert "VRAM" not in rep, rep
    # The retry's emptying (the weights on the CPU), then the one between two images.
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
