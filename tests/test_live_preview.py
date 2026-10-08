"""The live preview, formed in the result gallery (_with_live_preview).

The render runs in a thread and the wrapper hands out, while it works, the frames the
denoise drops into cz_pipeline -- then the real result. This file covers what breaks when
that is got wrong: the ORDER (frames before the result, never after), errors (they must
reach the interface unchanged), the flag (always lowered) and the gradio CONTEXT, without
which the progress bar would go silent inside the thread.

Neither GPU nor model: the render is simulated and the images are 8x8.

Run:  .venv/Scripts/python tests/test_live_preview.py
"""
import contextvars
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

import cz_pipeline  # noqa: E402
import cz_ui  # noqa: E402


def _push(step):
    """What the denoise does at every step: drop an image into the slot."""
    with cz_pipeline._PREVIEW_LOCK:
        if not cz_pipeline._PREVIEW["busy"]:
            return
        cz_pipeline._PREVIEW.update(img=Image.new("RGB", (8, 8), (step * 40, 0, 0)),
                                    step=step, total=3)
        cz_pipeline._PREVIEW["seq"] += 1


class _Fast:
    """Quick polling and the preview on, whatever this machine's config.txt says."""

    def __enter__(self):
        self.poll, self.on = cz_ui._LIVE_PREVIEW_POLL, cz_pipeline.LIVE_PREVIEW_ENABLED
        cz_ui._LIVE_PREVIEW_POLL = 0.01
        cz_pipeline.LIVE_PREVIEW_ENABLED = True
        return self

    def __exit__(self, *exc):
        cz_ui._LIVE_PREVIEW_POLL = self.poll
        cz_pipeline.LIVE_PREVIEW_ENABLED = self.on
        return False


RESULT = (["final.png"], "report", [], [])


def _render(steps=3, pause=0.06):
    def _fn(*a, **kw):
        for i in range(1, steps + 1):
            _push(i)
            time.sleep(pause)
        return RESULT
    return _fn


def test_the_frames_come_out_before_the_result():
    with _Fast():
        out = list(cz_ui._with_live_preview(_render())())
    assert len(out) >= 2, out                       # at least one frame + the result
    assert out[-1] == RESULT, out[-1]               # the result comes LAST
    for frame in out[:-1]:
        assert isinstance(frame, tuple) and len(frame) == 4, frame
        assert isinstance(frame[0], list) and frame[0], frame   # the gallery, and only it
    assert cz_pipeline._PREVIEW["busy"] is False


def test_the_result_is_never_overwritten_by_a_late_frame():
    """The very defect the separate component was there to avoid: a preview frame landing
    AFTER the result erased it. Impossible here -- it all comes out of one generator."""
    with _Fast():
        out = list(cz_ui._with_live_preview(_render(steps=5, pause=0.03))())
    assert out[-1] == RESULT
    assert RESULT not in out[:-1]


def test_an_exception_reaches_the_interface_unchanged():
    def boom(*a, **kw):
        raise ValueError("render broke")
    with _Fast():
        try:
            list(cz_ui._with_live_preview(boom)())
        except ValueError as e:
            assert str(e) == "render broke", e
        else:
            raise AssertionError("the error must surface, not be swallowed")
    assert cz_pipeline._PREVIEW["busy"] is False    # flag lowered despite the failure


def test_the_gradio_context_is_carried_into_the_render_thread():
    """Without the copied context, progress() inside the thread would do NOTHING, in
    silence: the progress bar would stay empty for the whole render."""
    probe = contextvars.ContextVar("cz_probe")
    probe.set("set by the calling thread")
    seen = {}

    def _fn(*a, **kw):
        seen["v"] = probe.get("lost")
        return RESULT

    with _Fast():
        list(cz_ui._with_live_preview(_fn)())
    assert seen["v"] == "set by the calling thread", seen


def test_with_the_preview_off_the_handler_is_called_directly():
    on = cz_pipeline.LIVE_PREVIEW_ENABLED
    cz_pipeline.LIVE_PREVIEW_ENABLED = False
    try:
        out = list(cz_ui._with_live_preview(lambda *a, **k: RESULT)())
    finally:
        cz_pipeline.LIVE_PREVIEW_ENABLED = on
    assert out == [RESULT], out                     # a single output, no frame at all
    assert cz_pipeline._PREVIEW["busy"] is False


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"All {len(tests)} live preview tests passed.")
