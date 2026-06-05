#!/usr/bin/env bash
# Lance crispz (UI Gradio) avec detection hardware.
# Utilise .venv s'il existe; --no-venv (ou --system) force le Python courant.

set -e
cd "$(dirname "$0")"

USE_VENV=1
for a in "$@"; do
    case "$a" in
        --no-venv|--system) USE_VENV=0 ;;
    esac
done

# Python de base
if command -v python3.10 >/dev/null 2>&1; then
    PYCMD="python3.10"
elif command -v python3 >/dev/null 2>&1; then
    PYCMD="python3"
else
    echo "[ERREUR] Python introuvable."
    exit 1
fi

RUNPY="$PYCMD"
if [ "$USE_VENV" -eq 1 ] && [ -x ".venv/bin/python" ]; then
    RUNPY=".venv/bin/python"
fi

# ESRGAN_DIR par defaut si non defini
if [ -z "$ESRGAN_DIR" ]; then
    export ESRGAN_DIR="$(pwd)/upscale_models"
fi

echo "=== crispz - run ==="
echo "Python     = $RUNPY"
echo "ESRGAN_DIR = $ESRGAN_DIR"
echo
echo "--- Detection hardware ---"
$RUNPY _hw_check.py
echo

echo "--- Lancement de l'UI Gradio ---"
echo "Ouvre http://127.0.0.1:7860 dans ton navigateur"
echo
$RUNPY app.py
