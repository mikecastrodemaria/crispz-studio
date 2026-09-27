"""Hardware detection + settings recommendations for crispz.

Prints a readable summary. Used by run.bat / run.sh / boot_check.bat.

Exit codes (so that the .bat scripts can react):
    0 = all is well
    1 = PyTorch absent
    2 = CUDA unavailable (CPU only)
    3 = INCOMPATIBLE: this torch build does not support this GPU's architecture
        (that is the RTX 50xx + non-cu128 torch case -> "WinError 127 torch_cuda.dll")

"""
import sys

# The NVIDIA architectures by compute capability. Serves to name the GPU and to know
# which CUDA minimum it requires (Blackwell = 12.8, otherwise the default build is enough).
ARCHS = [
    (12, 0, "Blackwell (RTX 50xx)", "12.8"),
    (9, 0, "Hopper (H100)", "12.0"),
    (8, 9, "Ada Lovelace (RTX 40xx)", "11.8"),
    (8, 6, "Ampere (RTX 30xx)", "11.1"),
    (8, 0, "Ampere (A100)", "11.0"),
    (7, 5, "Turing (RTX 20xx / GTX 16xx)", "10.0"),
    (7, 0, "Volta", "9.0"),
    (6, 1, "Pascal (GTX 10xx)", "8.0"),
]


def arch_name(major, minor):
    for ma, mi, name, cuda in ARCHS:
        if (major, minor) >= (ma, mi):
            return name, cuda
    return f"older (sm_{major}{minor})", "?"


def offload_reco(vram_gb):
    """The offload mode advised according to the VRAM.

    A landmark measured on this project: a bf16 FLUX transformer weighs ~23.8 GB and its
    T5 encoder ~9.5 GB (~33 GB in total) -> it does not fit in 32 GB, hence the
    offload. A Q8 GGUF of the same model drops to ~12.7 GB and fits with room to spare.
    'sequential' moves every submodule on every forward: very slow
    (measured ~3 s/step against ~1.1 s/step in 'model'), to be kept for the small cards.
    """
    if vram_gb >= 30:
        return ("none", "compact models (GGUF Q8, Z-Image) fit whole. "
                        "For a big bf16 model (~33 GB), switch to 'model'.")
    if vram_gb >= 20:
        return ("model", "a whole transformer fits on the GPU; the text encoder is "
                         "evicted once the prompt is encoded.")
    # The threshold at 11 and not 12: a card sold as "12 GB" exposes ~11.6-11.9 GB.
    # Putting those in 'sequential' would cost ~3x the time per step (measured) for nothing.
    if vram_gb >= 11:
        return ("model", "prefer the GGUF quantizations (Q8 ~12.7 GB, Q4 ~7 GB) "
                         "to keep some headroom.")
    if vram_gb >= 7:
        return ("sequential", "tight card: GGUF Q4 advised, 1024px max, "
                              "and expect slow steps.")
    return ("sequential", "very limited VRAM: GGUF Q4, 768-1024px, ESRGAN alone if needed.")


def main():
    try:
        import torch
    except ImportError:
        print("[ERROR] PyTorch missing.")
        return 1

    print(f"torch {torch.__version__} | cuda {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("CUDA unavailable: generation will run on the CPU (very slow, not advised).")
        print("Advice: no NVIDIA GPU here, prefer the ESRGAN pass alone (denoise = 0).")
        return 2

    i = 0
    props = torch.cuda.get_device_properties(i)
    name = torch.cuda.get_device_name(i)
    cap = torch.cuda.get_device_capability(i)
    vram_gb = props.total_memory / (1024 ** 3)
    bf16 = cap[0] >= 8                     # Ampere et plus
    gen, cuda_min = arch_name(*cap)
    sm = f"sm_{cap[0]}{cap[1]}"

    print(f"GPU             : {name}")
    print(f"Architecture    : {gen}  [{sm}]")
    print(f"VRAM            : {vram_gb:.1f} GB")
    print(f"BF16 native     : {'yes' if bf16 else 'no (Turing/Pascal, FP16 advised)'}")

    # --- THE check that counts: can this torch build compile for this GPU ? ---
    # A torch without the card's sm_ loads but breaks on the 1st CUDA allocation
    # ("WinError 127 ... torch_cuda.dll", or "no kernel image is available").
    try:
        arch_list = torch.cuda.get_arch_list()
    except Exception:
        arch_list = []
    supported = (not arch_list) or (sm in arch_list)
    print(f"Support {sm:<7}: {'yes' if supported else 'NO'}"
          f"  (torch build: {', '.join(arch_list[-4:]) if arch_list else 'unknown'})")
    if not supported:
        print()
        print("=" * 62)
        print(f"[INCOMPATIBLE] This PyTorch build has no {sm} kernels.")
        print(f"   {gen} needs CUDA {cuda_min}+; this torch is built for CUDA "
              f"{torch.version.cuda}.")
        print("   Typical symptom: 'WinError 127 ... torch_cuda.dll' or")
        print("   'no kernel image is available for execution on the device'.")
        print("   Fix:")
        print("     pip uninstall -y torch torchvision torchaudio")
        print(f"     pip install torch torchvision torchaudio "
              f"--index-url https://download.pytorch.org/whl/cu{cuda_min.replace('.', '')}")
        print("=" * 62)
        return 3

    # --- Recommendations (tiered according to the real VRAM) ---
    off, why = offload_reco(vram_gb)
    if vram_gb >= 20:
        tile, note = 0, "whole image (tile=0)"
    elif vram_gb >= 12:
        tile, note = 768, "tile 768, overlap 32"
    elif vram_gb >= 8:
        tile, note = 512, "tile 512, overlap 32"
    else:
        tile, note = 384, "tile 384, overlap 32, lower it on OOM"
    if vram_gb >= 24:
        zsize = "up to 2048px a side, whole image"
    elif vram_gb >= 12:
        zsize = "up to ~1536px, tile the diffusion pass beyond that (refine_tile)"
    else:
        zsize = "stay <= 1024px on the diffusion pass"

    print()
    print("--- Suggested settings (config.txt / Advanced tab) ---")
    print(f"CPU offload     : {off}  <- {why}")
    print(f"Tiling ESRGAN   : {note}   (default_tile={tile})")
    print(f"Diffusion pass  : {zsize}")
    print(f"Dtype           : {'BF16 (default)' if bf16 else 'FP16 (set DTYPE=torch.float16)'}")
    print(f"Attention slice : {'not needed' if vram_gb >= 16 else 'useful (already automatic in the code)'}")
    print(f"Denoise         : 0.20-0.30 conservative, 0.30-0.40 with a detailed prompt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
