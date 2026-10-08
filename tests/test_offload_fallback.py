"""A card that refuses an offload mode: the ladder down (_place_pipe).

Putting a big model on the card in 'none' copies the whole thing at once, and that can
fail in the DRIVER rather than in torch's allocator -- "CUDA error: out of memory", or
the opaque "CUDA error: unknown error". It used to end in a raw traceback. Each mode
further down needs less VRAM, so the placement walks down the ladder and says so.

Neither GPU nor model: the pipe is simulated.

Run:  .venv/Scripts/python tests/test_offload_fallback.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_pipeline as P  # noqa: E402

DRIVER = ("CUDA error: unknown error\n"
          "CUDA kernel errors might be asynchronously reported at some other API call")


class FakePipe:
    """Refuses every mode listed in `refuse`, records what was tried."""

    def __init__(self, refuse=()):
        self.refuse, self.tried, self.device = set(refuse), [], "cpu"

    def _maybe(self, mode):
        self.tried.append(mode)
        if mode in self.refuse:
            raise RuntimeError(DRIVER)

    def to(self, device, *a, **kw):
        if str(device) == "cpu":          # the undo between two attempts
            self.device = "cpu"
            return self
        self._maybe("none")
        self.device = str(device)
        return self

    def enable_model_cpu_offload(self, *a, **kw):
        self._maybe("model")
        self.device = "model-offload"

    def enable_sequential_cpu_offload(self, *a, **kw):
        self._maybe("sequential")
        self.device = "sequential-offload"


class _OnCuda:
    def __enter__(self):
        self.dev = P.DEVICE
        P.DEVICE = "cuda"
        return self

    def __exit__(self, *exc):
        P.DEVICE = self.dev
        return False


def test_the_asked_mode_is_used_when_the_card_accepts_it():
    with _OnCuda():
        pipe = FakePipe()
        out = P._place_pipe(pipe, "none")
        assert out is pipe and pipe.device == "cuda", pipe.device
        assert pipe.tried == ["none"], pipe.tried       # no ladder walked for nothing


def test_a_refused_none_falls_back_to_model():
    """The real case: a 32 GB card that will not take the whole model at once."""
    with _OnCuda():
        pipe = FakePipe(refuse=["none"])
        out = P._place_pipe(pipe, "none")
        assert out is pipe and pipe.device == "model-offload", pipe.device
        assert pipe.tried == ["none", "model"], pipe.tried


def test_it_keeps_going_down_to_sequential():
    with _OnCuda():
        pipe = FakePipe(refuse=["none", "model"])
        P._place_pipe(pipe, "none")
        assert pipe.tried == ["none", "model", "sequential"], pipe.tried
        assert pipe.device == "sequential-offload"


def test_everything_refused_raises_the_FIRST_error():
    """The first one describes the mode the user actually asked for -- the later ones
    are the fallbacks and would send them chasing the wrong thing."""
    with _OnCuda():
        pipe = FakePipe(refuse=["none", "model", "sequential"])
        try:
            P._place_pipe(pipe, "none")
        except RuntimeError as e:
            assert "unknown error" in str(e), e
        else:
            raise AssertionError("it must re-raise once no mode is left")
        assert pipe.tried == ["none", "model", "sequential"], pipe.tried


def test_asking_for_model_never_walks_back_up_to_none():
    """'none' needs MORE VRAM: climbing back up would be the wrong way round."""
    with _OnCuda():
        pipe = FakePipe(refuse=["model"])
        P._place_pipe(pipe, "model")
        assert pipe.tried == ["model", "sequential"], pipe.tried


def test_without_a_card_the_pipe_is_simply_moved():
    dev = P.DEVICE
    P.DEVICE = "cpu"
    try:
        pipe = FakePipe(refuse=["none", "model", "sequential"])
        out = P._place_pipe(pipe, "none")       # no ladder, no refusal: plain .to("cpu")
        assert out is pipe and pipe.tried == [], pipe.tried
    finally:
        P.DEVICE = dev


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"All {len(tests)} offload fallback tests passed.")
