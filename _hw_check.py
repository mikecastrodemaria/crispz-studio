"""Detection hardware + reco reglages pour crispz.

Imprime un resume lisible. Utilise par run.bat / run.sh / boot_check.bat.

Codes de sortie (pour que les scripts .bat puissent reagir):
    0 = tout va bien
    1 = PyTorch absent
    2 = CUDA indisponible (CPU seulement)
    3 = INCOMPATIBLE: ce build torch ne supporte pas l'architecture de ce GPU
        (c'est le cas RTX 50xx + torch non-cu128 -> "WinError 127 torch_cuda.dll")
"""
import sys

# Architectures NVIDIA par compute capability. Sert a nommer le GPU et a savoir
# quel CUDA minimum il exige (Blackwell = 12.8, sinon le build par defaut suffit).
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
    """Mode d'offload conseille selon la VRAM.

    Repere mesure sur ce projet: un transformer FLUX bf16 pese ~23,8 Go et son
    encodeur T5 ~9,5 Go (~33 Go au total) -> il ne tient pas dans 32 Go, d'ou
    l'offload. Un GGUF Q8 du meme modele tombe a ~12,7 Go et tient largement.
    'sequential' deplace chaque sous-module a chaque forward: tres lent
    (mesure ~3 s/step contre ~1,1 s/step en 'model'), a reserver aux petites cartes.
    """
    if vram_gb >= 30:
        return ("none", "compact models (GGUF Q8, Z-Image) fit whole. "
                        "For a big bf16 model (~33 GB), switch to 'model'.")
    if vram_gb >= 20:
        return ("model", "a whole transformer fits on the GPU; the text encoder is "
                         "evicted once the prompt is encoded.")
    # Seuil a 11 et non 12: une carte vendue "12 Go" expose ~11,6-11,9 Go. Les mettre
    # en 'sequential' couterait ~3x le temps par step (mesure) sans necessite.
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

    # --- LE check qui compte: ce build torch sait-il compiler pour ce GPU ? ---
    # Un torch sans le sm_ de la carte se charge mais casse a la 1re allocation
    # CUDA ("WinError 127 ... torch_cuda.dll", ou "no kernel image is available").
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

    # --- Recommandations (echelonnees selon la VRAM reelle) ---
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
