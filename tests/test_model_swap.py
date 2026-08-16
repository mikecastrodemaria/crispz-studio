"""Unit tests for the transformer hot-swap (_swap_transformer) — no model is loaded.

Regression guard for: switching Z-Image checkpoint A -> B (same base repo) used to
free_vram() and reload the WHOLE pipeline (transformer + VAE + Qwen3-4B text encoder).
Only the transformer must be reloaded; the base components stay in VRAM.

Run:  .venv/Scripts/python tests/test_model_swap.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_pipeline as P  # noqa: E402


class FakeT:
    def __init__(self, tag):
        self.tag = tag
        self.moved = None

    def to(self, dev):
        self.moved = dev
        return self


class FakePipe:
    def __init__(self, transformer=None):
        self.transformer = transformer
        self.calls = []

    def register_modules(self, **kw):
        self.calls.append(("register", tuple(kw)))
        for k, v in kw.items():
            setattr(self, k, v)

    def remove_all_hooks(self):
        self.calls.append(("remove_hooks",))

    def enable_model_cpu_offload(self):
        self.calls.append(("offload_model",))

    def unload_lora_weights(self):
        self.calls.append(("unload_lora",))

    def load_lora_weights(self, folder, weight_name=None, adapter_name=None):
        self.calls.append(("load_lora", weight_name))

    def set_adapters(self, names, weights):
        self.calls.append(("set_adapters", tuple(names)))


def _swap_env(monkey_new, offload="none", loras=None):
    P.OFFLOAD_MODE = offload
    P.LORAS = list(loras or [])
    P._APPLIED_LORAS = []
    P._DERIVED = {"img2img": object()}          # doit etre vide apres le swap
    P._load_transformer = lambda: monkey_new


def test_swap_keeps_pipe_and_replaces_only_transformer():
    old_t, new_t = FakeT("old"), FakeT("new")
    pipe = FakePipe(old_t)
    _swap_env(new_t)
    assert P._swap_transformer(pipe) is True
    assert pipe.transformer is new_t, "le transformer doit etre remplace"
    assert ("register", ("transformer",)) in pipe.calls
    assert P._DERIVED == {}, "les pipes derives pointaient sur l'ancien transformer"


def test_swap_reapplies_loras_on_the_new_transformer():
    d = tempfile.mkdtemp()
    lp = os.path.join(d, "a.safetensors")
    with open(lp, "wb") as f:
        f.write(b"x")
    new_t = FakeT("new")
    pipe = FakePipe(FakeT("old"))
    _swap_env(new_t, loras=[(lp, 0.6)])
    assert P._swap_transformer(pipe) is True
    # les adaptateurs etaient sur l'ancien transformer -> reposes sur le nouveau
    assert ("load_lora", "a.safetensors") in pipe.calls
    assert P._APPLIED_LORAS == [(lp, 0.6)]


def test_swap_offload_removes_and_reapplies_hooks():
    if P.DEVICE != "cuda":
        print("   (skip: offload path is CUDA-only on this machine)")
        return
    new_t = FakeT("new")
    pipe = FakePipe(FakeT("old"))
    _swap_env(new_t, offload="model")
    assert P._swap_transformer(pipe) is True
    assert ("remove_hooks",) in pipe.calls and ("offload_model",) in pipe.calls


def test_swap_failure_falls_back():
    pipe = FakePipe(FakeT("old"))
    P.OFFLOAD_MODE = "none"
    P.LORAS = []
    P._load_transformer = lambda: (_ for _ in ()).throw(RuntimeError("corrupt file"))
    assert P._swap_transformer(pipe) is False    # -> le caller fera free_vram + reload


def test_set_zimage_transformer_does_not_free_the_pipe():
    """Coeur du fix: changer de checkpoint single-file ne doit plus jeter le pipeline."""
    sentinel = object()
    P._BASE_PIPE = sentinel
    P.ZIMAGE_TRANSFORMER = "D:/models/A.safetensors"
    P.set_zimage_transformer("D:/models/B.safetensors")
    assert P.ZIMAGE_TRANSFORMER == "D:/models/B.safetensors"
    assert P._BASE_PIPE is sentinel, "le pipe doit rester charge (swap du transformer seul)"
    P._BASE_PIPE = None


def test_set_zimage_model_single_file_does_not_free():
    # _is_single_file exige un fichier REEL (comme resolve_checkpoint en fournit)
    d = tempfile.mkdtemp()
    ck = os.path.join(d, "Juggernaut.safetensors")
    with open(ck, "wb") as f:
        f.write(b"x")
    sentinel = object()
    P._BASE_PIPE = sentinel
    P.ZIMAGE_TRANSFORMER = None
    P.set_zimage_model(ck)
    assert P.ZIMAGE_TRANSFORMER == ck
    assert P._BASE_PIPE is sentinel, "un checkpoint single-file ne doit pas jeter le pipe"
    P._BASE_PIPE = None


def test_set_zimage_model_new_base_repo_still_reloads():
    """Le repo de base change -> VAE/encodeur changent aussi -> reload complet obligatoire."""
    P._BASE_PIPE = object()
    P._LOADED_KEY = ("old/repo", None, "none")
    P.BASE_REPO = "old/repo"
    P.set_zimage_model("Tongyi-MAI/Z-Image-Turbo")
    assert P.BASE_REPO == "Tongyi-MAI/Z-Image-Turbo"
    assert P._BASE_PIPE is None, "changer de repo de base doit liberer le pipe"
    assert P._LOADED_KEY is None


if __name__ == "__main__":
    _real = P._load_transformer
    try:
        for fn in (test_swap_keeps_pipe_and_replaces_only_transformer,
                   test_swap_reapplies_loras_on_the_new_transformer,
                   test_swap_offload_removes_and_reapplies_hooks,
                   test_swap_failure_falls_back,
                   test_set_zimage_transformer_does_not_free_the_pipe,
                   test_set_zimage_model_single_file_does_not_free,
                   test_set_zimage_model_new_base_repo_still_reloads):
            fn()
            print(f"OK {fn.__name__}")
    finally:
        P._load_transformer = _real
    print("All model swap tests passed.")
