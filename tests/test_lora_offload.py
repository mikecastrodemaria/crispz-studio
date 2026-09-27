"""A LoRA applied under offload (ported from crispz-klein 1.36.6).

Two DoRA LoRAs chosen as edit LoRAs put an end to the session on klein: the
loading failed on "Cannot copy out of meta tensor", then ALL the renders that followed
on "Cannot generate a cpu tensor from a generator of type cuda". The same applying code here.
It covers:
  - the LoRAs are loaded with low_cpu_mem_usage=False (no layer on 'meta',
    a filtered key keeps its init value);
  - a load that fails after diffusers has removed the offload hooks gets them
    put back, instead of leaving the pipe on the CPU;
  - restore_offload repairs a pipe left on the CPU and touches nothing otherwise.

Neither a GPU nor a model: the pipe is simulated, the LoRAs are tiny files.

Run:  .venv/Scripts/python tests/test_lora_offload.py

"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import cz_pipeline as P  # noqa: E402

META = ("Cannot copy out of meta tensor; no data! Please use "
        "torch.nn.Module.to_empty() instead of torch.nn.Module.to()")


class FakePipe:
    """A minimal pipe with diffusers' behaviour: loading a LoRA first removes
    the offload hooks, and only puts them back on a success."""

    def __init__(self, hooks=True, fail=False):
        self._all_hooks = ["hook"] if hooks else []
        self._execution_device = torch.device("cuda" if hooks else "cpu")
        self.fail = fail
        self.loaded, self.enabled, self.adapters, self.cleared = [], 0, None, 0

    def load_lora_weights(self, *a, **kw):
        self._all_hooks = []                       # diffusers removes the offload
        self._execution_device = torch.device("cpu")
        self.loaded.append((a, kw))
        if self.fail:
            raise RuntimeError(META)
        self.enable_model_cpu_offload()             # ... and puts it back when all goes well

    def enable_model_cpu_offload(self, *a, **kw):
        self._all_hooks = ["hook"]
        self._execution_device = torch.device("cuda")
        self.enabled += 1

    def enable_sequential_cpu_offload(self, *a, **kw):
        self.enabled += 1

    def to(self, *a, **kw):
        self._execution_device = torch.device("cuda")
        return self

    def set_adapters(self, names, weights):
        self.adapters = (list(names), list(weights))

    def get_list_adapters(self):
        return {}

    def unload_lora_weights(self):
        self.cleared += 1

    def delete_adapters(self, *a, **kw):
        pass


def _lora(path, hidden=1024, rank=4):
    """A tiny LoRA in the PEFT dialect."""
    save_file({
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(rank, hidden),
        "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": torch.zeros(hidden, rank),
    }, path)
    return path


class _OnCuda:
    """A machine with a card, offload 'model': the state in which the bug happens."""

    def __enter__(self):
        self.dev, self.off = P.DEVICE, P._effective_offload
        P.DEVICE = "cuda"
        P._effective_offload = lambda *a, **k: "model"
        return self

    def __exit__(self, *exc):
        P.DEVICE, P._effective_offload = self.dev, self.off
        return False


def _want(paths):
    """A LoRA set asked for, none applied yet (the globals vary from fork to fork)."""
    P._APPLIED_LORAS = []
    P.LORAS = list(paths)
    if hasattr(P, "PROMPT_LORAS"):
        P.PROMPT_LORAS = []
    if hasattr(P, "_APPLIED_LOKRS"):
        P._APPLIED_LOKRS = []


def test_loras_are_loaded_with_real_tensors():
    tmp = tempfile.mkdtemp()
    try:
        p = _lora(os.path.join(tmp, "dora.safetensors"))
        pipe = FakePipe()
        _want([(p, 0.8)])
        assert P._apply_loras(pipe) is True
        assert len(pipe.loaded) == 1, pipe.loaded
        _a, kw = pipe.loaded[0]
        assert kw.get("low_cpu_mem_usage") is False, kw
        assert pipe.adapters == (["cz_lora_0"], [0.8]), pipe.adapters
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_failed_load_puts_the_offload_back():
    tmp = tempfile.mkdtemp()
    try:
        p = _lora(os.path.join(tmp, "dora.safetensors"))
        pipe = FakePipe(fail=True)
        _want([(p, 0.8)])
        with _OnCuda():
            assert P._apply_loras(pipe) is False     # -> the caller reloads everything
        # Without the restoration, the pipe would stay on the CPU and everything after would fail.
        assert pipe._all_hooks and str(pipe._execution_device) == "cuda", pipe._all_hooks
        assert pipe.enabled == 1, pipe.enabled
        assert P._APPLIED_LORAS == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_restore_offload_only_acts_on_a_pipe_left_on_the_cpu():
    with _OnCuda():
        broken = FakePipe(hooks=False)               # laisse sur le CPU
        assert P.restore_offload(broken, "a test") is True
        assert broken.enabled == 1 and str(broken._execution_device) == "cuda"
        healthy = FakePipe()                          # already on the card
        assert P.restore_offload(healthy) is False
        assert healthy.enabled == 0
        P._effective_offload = lambda *a, **k: "none"
        plain = FakePipe(hooks=False)
        assert P.restore_offload(plain) is True       # offload 'none' -> simple .to(cuda)
        assert plain.enabled == 0 and str(plain._execution_device) == "cuda"
        P.DEVICE = "cpu"                              # a machine with no card: we touch nothing
        assert P.restore_offload(FakePipe(hooks=False)) is False


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"All {len(tests)} LoRA offload tests passed.")
