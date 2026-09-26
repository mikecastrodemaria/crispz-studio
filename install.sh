#!/usr/bin/env bash
# Install pour crispz (Linux / macOS / WSL).
#
# Defaut: venv .venv ISOLE (n'herite PAS du site-packages global) installe
# depuis requirements-lock.txt -> environnement reproductible, versions
# maitrisees, aucun risque de casser un autre projet. Telecharge son propre
# torch (~3,5 Go).
#
#   --shared    ancien comportement: venv --system-site-packages qui HERITE du
#               torch global. Plus leger sur le disque, mais fait aussi heriter
#               diffusers/accelerate/numpy/pillow -> versions non choisies et
#               partagees avec les autres forks crispz.
#   --no-venv   installe directement sur le Python courant.

set -e
cd "$(dirname "$0")"

# Pipeline attendu pour cette famille de modele. SEULE ligne qui differe entre
# crispz-studio (ZImage), crispz-krea (Flux) et crispz-qwen-edit (Qwen).
CHECK_PIPE=ZImageImg2ImgPipeline

USE_VENV=1
ISOLATED=1
FACESWAP=1
FACESWAP_MODEL=0
for a in "$@"; do
    case "$a" in
        --no-venv|--system) USE_VENV=0 ;;
        --shared) ISOLATED=0 ;;
        --no-faceswap) FACESWAP=0 ;;
        --faceswap-model) FACESWAP_MODEL=1 ;;
    esac
done
[ "$USE_VENV" -eq 0 ] && ISOLATED=0

echo "=== crispz - install ==="
if [ "$ISOLATED" -eq 1 ]; then
    echo "Mode: ISOLATED venv (reproducible, dedicated torch)"
else
    echo "Mode: shared / system venv (inherits the global torch)"
fi
echo

# 1) Python de base
if command -v python3.10 >/dev/null 2>&1; then
    PYCMD="python3.10"
elif command -v python3 >/dev/null 2>&1; then
    PYCMD="python3"
else
    echo "[ERROR] Python not found. Install Python 3.10+."
    exit 1
fi
echo "Base Python: $PYCMD"
$PYCMD --version
echo

# 2) torch + CUDA. En mode ISOLE, torch vient de requirements-lock.txt: rien a
#    verifier ici. En mode partage/systeme, il doit deja etre present.
if [ "$ISOLATED" -eq 1 ]; then
    echo "Isolated mode: torch is installed in the venv from the lock. Nothing to check."
else
    set +e
    $PYCMD -c "import torch,sys; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda); sys.exit(0 if torch.cuda.is_available() else 2)"
    rc=$?
    set -e
    if [ $rc -eq 1 ]; then
        echo
        echo "[ERROR] PyTorch not found. Install your PyTorch + CUDA build first."
        echo "Example (CUDA 12.8):"
        echo "  $PYCMD -m pip install torch --index-url https://download.pytorch.org/whl/cu128"
        echo "Or run again without --shared for an isolated venv that installs its own torch."
        exit 1
    elif [ $rc -eq 2 ]; then
        echo "[WARN] CUDA unavailable. Generating on the CPU will be very slow."
    fi
fi
echo

# 3) xformers casse ? le neutraliser cote SYSTEME. Utile seulement si le venv
#    herite du global (mode --shared) ou en --no-venv.
if [ "$ISOLATED" -ne 1 ]; then
    if $PYCMD -c "import xformers" >/dev/null 2>&1; then
        if ! $PYCMD -c "import xformers.ops" >/dev/null 2>&1; then
            echo "[WARN] xformers installed but does not load (torch ABI mismatch). Uninstalling."
            $PYCMD -m pip uninstall -y xformers
        else
            echo "xformers OK."
        fi
    fi
    echo
fi

# 4) venv
RUNPY="$PYCMD"
if [ "$USE_VENV" -eq 1 ]; then
    if [ ! -d ".venv" ]; then
        if [ "$ISOLATED" -eq 1 ]; then
            echo "Creating the .venv (ISOLATED)..."
            "$PYCMD" -m venv ".venv" || \
                echo "[WARN] could not create the venv -> installing on the current Python."
        else
            echo "Creating the .venv (--system-site-packages: inherits your torch)..."
            "$PYCMD" -m venv --system-site-packages ".venv" || \
                echo "[WARN] could not create the venv -> installing on the current Python."
        fi
    else
        echo ".venv already there, reused as is."
        echo "  For a clean start: rm -rf .venv then run this again."
    fi
    if [ -x ".venv/bin/python" ]; then
        RUNPY=".venv/bin/python"
        "$RUNPY" -m pip install --quiet --upgrade pip setuptools wheel
    fi
else
    echo "Mode --no-venv: installing on the current Python."
fi
echo "Install interpreter: $RUNPY"
echo

# 5) Installer les deps. En mode isole on prefere le lock (versions exactes
#    validees, torch cu128 inclus). Sinon requirements.txt (bornes larges).
REQFILE=requirements.txt
if [ "$ISOLATED" -eq 1 ] && [ -f requirements-lock.txt ]; then
    REQFILE=requirements-lock.txt
fi
echo "Installing the dependencies from $REQFILE ..."
if [ "$REQFILE" = "requirements-lock.txt" ]; then
    echo "  (includes torch cu128, ~3.5 GB to download the first time)"
fi
# Pillow est filtre du fichier: gradio 5.50 declare pillow<12 et refuserait de
# resoudre avec le pin 12.x. Il est pose juste apres, en --no-deps. Le pin reste
# dans le lock pour que Dependabot voie la version corrigee.
# onnxruntime (build CPU) est filtre lui aussi: rembg le tire en dependance, et une
# fois installe il MASQUE onnxruntime-gpu a l'import (meme nom de module, le CPU
# gagne) -> faceswap, GFPGAN/CodeFormer et rembg tombent silencieusement sur le CPU
# malgre onnxruntime-gpu present. On ne garde que le build GPU.
REQTMP="${TMPDIR:-/tmp}/cz_req_nopillow.txt"
grep -vE '^(pillow|onnxruntime)==' "$REQFILE" > "$REQTMP"
$RUNPY -m pip install -r "$REQTMP"
echo

# 5bis) Pillow, installe A PART et en --no-deps.
#   gradio 5.50 declare "pillow<12.0", mais les CVE Pillow (dont celles
#   atteignables via les images que l'utilisateur ouvre) ne sont corrigees qu'en
#   12.x. Cette borne de gradio est conservatrice: verifie sur cette base de
#   code, Pillow 12 fonctionne. On installe donc apres coup, sans redeclencher
#   la resolution qui refuserait la combinaison.
PILLOW_PIN="pillow==12.3.0"
echo "Installing $PILLOW_PIN (apart: works around gradio's pillow<12 bound)..."
$RUNPY -m pip install --no-deps --upgrade "$PILLOW_PIN" \
    || echo "[WARN] Pillow install failed -> inherited version kept."
echo

# 6) Verifier que diffusers expose le pipeline de cette famille de modele
$RUNPY -c "from diffusers import $CHECK_PIPE; print('$CHECK_PIPE OK')"
echo

# 7) Deps optionnelles. Le lock les contient deja; en mode non-isole il faut
#    encore passer par les fichiers dedies.
if [ "$FACESWAP" -eq 1 ] && [ "$REQFILE" != "requirements-lock.txt" ]; then
    echo "Installing the FaceSwap deps (insightface + onnxruntime-gpu)..."
    $RUNPY -m pip install -r requirements-faceswap.txt || \
        echo "[WARN] FaceSwap install failed (not blocking). The feature stays off."
    echo "Installing the extras (rembg for Remove BG)..."
    $RUNPY -m pip install -r requirements-extra.txt || \
        echo "[WARN] extras install failed (not blocking)."
    echo
fi

# 8) Dossiers de modeles
mkdir -p upscale_models checkpoints loras faceswap
echo "Folders ready: upscale_models (ESRGAN), checkpoints, loras, faceswap."
echo

# 9) Config locale: copie config-sample.txt -> config.txt si absent
if [ ! -f config.txt ] && [ -f config-sample.txt ]; then
    cp config-sample.txt config.txt
    echo "config.txt created from config-sample.txt (edit it for your settings)."
fi
echo

# 10) Modele inswapper (FaceSwap) - opt-in (~528 Mo, licence): --faceswap-model
if [ "$FACESWAP_MODEL" -eq 1 ]; then
    if [ ! -f faceswap/inswapper_128.onnx ]; then
        echo "Downloading the inswapper_128.onnx model (~528 MB)..."
        $RUNPY -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx', 'faceswap/inswapper_128.onnx'); print('inswapper OK')"
    else
        echo "inswapper model already there."
    fi
    echo
fi

echo "=== Install OK. Run: ./run.sh ==="
echo "    Options: --shared (venv inheriting the global torch)  --no-venv (current Python)"
echo "             --no-faceswap (skip insightface)  --faceswap-model (download inswapper)"
