#!/usr/bin/env bash
# Install pour crispz - Z-Image upscaler + detailer (Linux / macOS / WSL)
# Par defaut: cree un venv .venv (--system-site-packages) qui HERITE de ton torch
# et isole les deps de crispz. Option --no-venv (ou --system) pour installer
# directement sur le Python courant. Ne reinstalle JAMAIS torch.

set -e
cd "$(dirname "$0")"

USE_VENV=1
for a in "$@"; do
    case "$a" in
        --no-venv|--system) USE_VENV=0 ;;
    esac
done

echo "=== crispz - install ==="
echo

# 1) Python de base (doit deja avoir torch)
if command -v python3.10 >/dev/null 2>&1; then
    PYCMD="python3.10"
elif command -v python3 >/dev/null 2>&1; then
    PYCMD="python3"
else
    echo "[ERREUR] Python introuvable. Installe Python 3.10+."
    exit 1
fi
echo "Python de base: $PYCMD"
$PYCMD --version
echo

# 2) Verifier torch + CUDA sur le Python de base (NE PAS reinstaller)
set +e
$PYCMD -c "import torch,sys; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda); sys.exit(0 if torch.cuda.is_available() else 2)"
rc=$?
set -e
if [ $rc -eq 1 ]; then
    echo
    echo "[ERREUR] PyTorch introuvable. Installe ton build PyTorch + CUDA d'abord."
    echo "Exemple (CUDA 12.8):"
    echo "  $PYCMD -m pip install torch --index-url https://download.pytorch.org/whl/cu128"
    exit 1
elif [ $rc -eq 2 ]; then
    echo "[AVERT] CUDA non disponible. Z-Image en CPU sera tres lent."
fi
echo

# 3) xformers casse ? le neutraliser cote SYSTEME (un venv --system-site-packages
#    en herite, donc un xformers casse casserait aussi le venv).
if $PYCMD -c "import xformers" >/dev/null 2>&1; then
    if ! $PYCMD -c "import xformers.ops" >/dev/null 2>&1; then
        echo "[AVERT] xformers installe mais ne charge pas (ABI torch incompatible). Desinstallation."
        $PYCMD -m pip uninstall -y xformers
    else
        echo "xformers OK."
    fi
fi
echo

# 4) venv optionnel (defaut) avec --system-site-packages (herite de torch)
RUNPY="$PYCMD"
if [ "$USE_VENV" -eq 1 ]; then
    if [ ! -d ".venv" ]; then
        echo "Creation du venv .venv (--system-site-packages: herite de ton torch)..."
        "$PYCMD" -m venv --system-site-packages ".venv" || \
            echo "[AVERT] creation du venv impossible -> install sur le Python courant."
    fi
    if [ -x ".venv/bin/python" ]; then
        RUNPY=".venv/bin/python"
        if ! "$RUNPY" -c "import torch" >/dev/null 2>&1; then
            echo "[AVERT] torch non visible dans le venv -> repli sur le Python courant."
            RUNPY="$PYCMD"
        fi
    fi
else
    echo "Mode --no-venv: install sur le Python courant."
fi
echo "Interpreteur d'install: $RUNPY"
echo

# 5) Installer les deps (hors torch)
echo "Installation des dependances..."
$RUNPY -m pip install -r requirements.txt
echo

# 6) Verifier ZImageImg2ImgPipeline
$RUNPY -c "from diffusers import ZImageImg2ImgPipeline; print('ZImageImg2ImgPipeline OK')"
echo

# 7) Dossier upscale_models
mkdir -p upscale_models
echo "Dossier upscale_models pret. Depose tes .pth dedans, ou pointe ESRGAN_DIR."
echo
echo "=== Install OK. Lance: ./run.sh  (ou ./run.sh --no-venv) ==="
