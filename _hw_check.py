"""Detection hardware + reco reglages pour crispz.
Imprime un resume lisible. Utilise par run.bat / run.sh.
"""
import sys


def main():
    try:
        import torch
    except ImportError:
        print("[ERREUR] PyTorch absent.")
        sys.exit(1)

    print(f"torch {torch.__version__} | cuda {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("CUDA non disponible: Z-Image Turbo tournera en CPU (tres lent, deconseille).")
        print("Reco: machine sans GPU NVIDIA, prefere la passe ESRGAN seule (denoise = 0).")
        return

    i = 0
    name = torch.cuda.get_device_name(i)
    cap = torch.cuda.get_device_capability(i)
    vram_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
    bf16 = cap[0] >= 8  # Ampere et plus

    print(f"GPU             : {name}")
    print(f"Compute capabil.: {cap[0]}.{cap[1]}")
    print(f"VRAM            : {vram_gb:.1f} Go")
    print(f"BF16 natif      : {'oui' if bf16 else 'non (Turing/Pascal, FP16 conseille)'}")

    # Reco tile ESRGAN
    if vram_gb >= 20:
        tile, note = 0, "image entiere (tile=0)"
    elif vram_gb >= 12:
        tile, note = 768, "tile 768, overlap 32"
    elif vram_gb >= 8:
        tile, note = 512, "tile 512, overlap 32"
    else:
        tile, note = 384, "tile 384, overlap 32, baisser si OOM"

    # Reco passe Z-Image
    if vram_gb >= 24:
        zsize = "jusqu'a 2048px de cote en image entiere"
    elif vram_gb >= 12:
        zsize = "jusqu'a ~1536px, au-dela: faire l'upscale en 2 passes ou attendre le tiling diffusion"
    else:
        zsize = "rester <= 1024px sur la passe diffusion"

    print()
    print("--- Reco reglages ---")
    print(f"Tiling ESRGAN   : {note}")
    print(f"Passe Z-Image   : {zsize}")
    print(f"Dtype           : {'BF16 (defaut app.py)' if bf16 else 'modifier DTYPE = torch.float16 dans app.py'}")
    print(f"Denoise         : 0.20-0.30 conservateur, 0.30-0.40 avec prompt detaille")
    print(f"Steps           : 12-16 a strength 0.30")
    print(f"Attention slicing: {'inutile' if vram_gb >= 16 else 'utile (deja active dans app.py)'}")


if __name__ == "__main__":
    main()
