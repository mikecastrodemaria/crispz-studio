#!/usr/bin/env bash
# Install for crispz (Linux / macOS / WSL).
#
# Default: an ISOLATED .venv (it does NOT inherit the global site-packages)
# installed from requirements-lock.txt -> a reproducible environment, versions
# under control, no risk of breaking another project. It downloads its own
# torch (~3.5 GB).
#
#   --shared    the old behaviour: a --system-site-packages venv that INHERITS
#               the global torch. Lighter on the disk, but it also inherits
#               diffusers/accelerate/numpy/pillow -> versions nobody chose, and
#               shared with the other crispz forks.
#   --no-venv   installs straight onto the current Python.

set -e
cd "$(dirname "$0")"

# The pipeline expected for this model family. The ONLY line that differs between
# crispz-studio (ZImage), crispz-krea (Flux) and crispz-qwen-edit (Qwen).
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

# 1) The base Python
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

# 2) torch + CUDA. In ISOLATED mode torch comes from requirements-lock.txt:
#    nothing to check here. In shared/system mode, it must already be present.
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

# 3) A broken xformers ? neutralise it SYSTEM-side. Only useful when the venv
#    inherits the global one (--shared mode) or under --no-venv.
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

# 5) Install the deps. In isolated mode we prefer the lock (exact, validated
#    versions, torch cu128 included). Otherwise requirements.txt (wide bounds).
REQFILE=requirements.txt
if [ "$ISOLATED" -eq 1 ] && [ -f requirements-lock.txt ]; then
    REQFILE=requirements-lock.txt
fi
echo "Installing the dependencies from $REQFILE ..."
if [ "$REQFILE" = "requirements-lock.txt" ]; then
    echo "  (includes torch cu128, ~3.5 GB to download the first time)"
fi
# Pillow is filtered out of the file: gradio 5.50 declares pillow<12 and would refuse
# to resolve with the 12.x pin. It is installed just after, with --no-deps. The pin
# stays in the lock so that Dependabot sees the fixed version.
# onnxruntime (the CPU build) is filtered out too: rembg pulls it as a dependency, and
# once installed it HIDES onnxruntime-gpu at import time (the same module name, the CPU
# one wins) -> faceswap, GFPGAN/CodeFormer and rembg fall back silently to the CPU
# although onnxruntime-gpu is there. We keep the GPU build only.
REQTMP="${TMPDIR:-/tmp}/cz_req_nopillow.txt"
grep -vE '^(pillow|onnxruntime)==' "$REQFILE" > "$REQTMP"
$RUNPY -m pip install -r "$REQTMP"
echo

# 5bis) Pillow, installed APART and with --no-deps.
#   gradio 5.50 declares "pillow<12.0", but the Pillow CVEs (including the ones
#   reachable through the images the user opens) are only fixed in 12.x. That bound
#   of gradio's is conservative: checked against this code base, Pillow 12 works.
#   So we install it afterwards, without triggering again the resolution that would
#   refuse the combination.
PILLOW_PIN="pillow==12.3.0"
echo "Installing $PILLOW_PIN (apart: works around gradio's pillow<12 bound)..."
$RUNPY -m pip install --no-deps --upgrade "$PILLOW_PIN" \
    || echo "[WARN] Pillow install failed -> inherited version kept."
echo

# 6) Check that diffusers exposes the pipeline of this model family
$RUNPY -c "from diffusers import $CHECK_PIPE; print('$CHECK_PIPE OK')"
echo

# 7) Optional deps. The lock already contains them; in non-isolated mode the
#    dedicated files are still the way in.
if [ "$FACESWAP" -eq 1 ] && [ "$REQFILE" != "requirements-lock.txt" ]; then
    echo "Installing the FaceSwap deps (insightface + onnxruntime-gpu)..."
    $RUNPY -m pip install -r requirements-faceswap.txt || \
        echo "[WARN] FaceSwap install failed (not blocking). The feature stays off."
    echo "Installing the extras (rembg for Remove BG)..."
    $RUNPY -m pip install -r requirements-extra.txt || \
        echo "[WARN] extras install failed (not blocking)."
    echo
fi

# 8) Model folders
mkdir -p upscale_models checkpoints loras faceswap
echo "Folders ready: upscale_models (ESRGAN), checkpoints, loras, faceswap."
echo

# 9) Local config: copies config-sample.txt -> config.txt when absent
if [ ! -f config.txt ] && [ -f config-sample.txt ]; then
    cp config-sample.txt config.txt
    echo "config.txt created from config-sample.txt (edit it for your settings)."
fi
echo

# 10) The inswapper model (FaceSwap) - opt-in (~528 MB, licence): --faceswap-model
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
