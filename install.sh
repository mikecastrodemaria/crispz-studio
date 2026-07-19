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
    echo "Mode: venv ISOLE (reproductible, torch dedie)"
else
    echo "Mode: venv partage / systeme (herite du torch global)"
fi
echo

# 1) Python de base
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

# 2) torch + CUDA. En mode ISOLE, torch vient de requirements-lock.txt: rien a
#    verifier ici. En mode partage/systeme, il doit deja etre present.
if [ "$ISOLATED" -eq 1 ]; then
    echo "Mode isole: torch sera installe dans le venv depuis le lock. Rien a verifier."
else
    set +e
    $PYCMD -c "import torch,sys; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda); sys.exit(0 if torch.cuda.is_available() else 2)"
    rc=$?
    set -e
    if [ $rc -eq 1 ]; then
        echo
        echo "[ERREUR] PyTorch introuvable. Installe ton build PyTorch + CUDA d'abord."
        echo "Exemple (CUDA 12.8):"
        echo "  $PYCMD -m pip install torch --index-url https://download.pytorch.org/whl/cu128"
        echo "Ou relance sans --shared pour un venv isole qui installe son propre torch."
        exit 1
    elif [ $rc -eq 2 ]; then
        echo "[AVERT] CUDA non disponible. La generation en CPU sera tres lente."
    fi
fi
echo

# 3) xformers casse ? le neutraliser cote SYSTEME. Utile seulement si le venv
#    herite du global (mode --shared) ou en --no-venv.
if [ "$ISOLATED" -ne 1 ]; then
    if $PYCMD -c "import xformers" >/dev/null 2>&1; then
        if ! $PYCMD -c "import xformers.ops" >/dev/null 2>&1; then
            echo "[AVERT] xformers installe mais ne charge pas (ABI torch incompatible). Desinstallation."
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
            echo "Creation du venv .venv (ISOLE)..."
            "$PYCMD" -m venv ".venv" || \
                echo "[AVERT] creation du venv impossible -> install sur le Python courant."
        else
            echo "Creation du venv .venv (--system-site-packages: herite de ton torch)..."
            "$PYCMD" -m venv --system-site-packages ".venv" || \
                echo "[AVERT] creation du venv impossible -> install sur le Python courant."
        fi
    else
        echo "Venv .venv deja present, reutilise en l'etat."
        echo "  Pour repartir propre: rm -rf .venv puis relance."
    fi
    if [ -x ".venv/bin/python" ]; then
        RUNPY=".venv/bin/python"
        "$RUNPY" -m pip install --quiet --upgrade pip setuptools wheel
    fi
else
    echo "Mode --no-venv: install sur le Python courant."
fi
echo "Interpreteur d'install: $RUNPY"
echo

# 5) Installer les deps. En mode isole on prefere le lock (versions exactes
#    validees, torch cu128 inclus). Sinon requirements.txt (bornes larges).
REQFILE=requirements.txt
if [ "$ISOLATED" -eq 1 ] && [ -f requirements-lock.txt ]; then
    REQFILE=requirements-lock.txt
fi
echo "Installation des dependances depuis $REQFILE ..."
if [ "$REQFILE" = "requirements-lock.txt" ]; then
    echo "  (inclut torch cu128, ~3,5 Go de telechargement la premiere fois)"
fi
$RUNPY -m pip install -r "$REQFILE"
echo

# 6) Verifier que diffusers expose le pipeline de cette famille de modele
$RUNPY -c "from diffusers import $CHECK_PIPE; print('$CHECK_PIPE OK')"
echo

# 7) Deps optionnelles. Le lock les contient deja; en mode non-isole il faut
#    encore passer par les fichiers dedies.
if [ "$FACESWAP" -eq 1 ] && [ "$REQFILE" != "requirements-lock.txt" ]; then
    echo "Installation des deps FaceSwap (insightface + onnxruntime-gpu)..."
    $RUNPY -m pip install -r requirements-faceswap.txt || \
        echo "[AVERT] echec install FaceSwap (non bloquant). La feature restera desactivee."
    echo "Installation des extras (rembg pour Remove BG)..."
    $RUNPY -m pip install -r requirements-extra.txt || \
        echo "[AVERT] echec install extras (non bloquant)."
    echo
fi

# 8) Dossiers de modeles
mkdir -p upscale_models checkpoints loras faceswap
echo "Dossiers prets: upscale_models (ESRGAN), checkpoints, loras, faceswap."
echo

# 9) Config locale: copie config-sample.txt -> config.txt si absent
if [ ! -f config.txt ] && [ -f config-sample.txt ]; then
    cp config-sample.txt config.txt
    echo "config.txt cree depuis config-sample.txt (edite-le pour tes reglages)."
fi
echo

# 10) Modele inswapper (FaceSwap) - opt-in (~528 Mo, licence): --faceswap-model
if [ "$FACESWAP_MODEL" -eq 1 ]; then
    if [ ! -f faceswap/inswapper_128.onnx ]; then
        echo "Telechargement du modele inswapper_128.onnx (~528 Mo)..."
        $RUNPY -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx', 'faceswap/inswapper_128.onnx'); print('inswapper OK')"
    else
        echo "Modele inswapper deja present."
    fi
    echo
fi

echo "=== Install OK. Lance: ./run.sh ==="
echo "    Options: --shared (venv qui herite du torch global)  --no-venv (Python courant)"
echo "             --no-faceswap (sauter insightface)  --faceswap-model (telecharger inswapper)"
