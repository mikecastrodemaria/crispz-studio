#!/usr/bin/env bash
# Launches crispz (the Gradio UI) with hardware detection.
# Uses .venv when it exists; --no-venv (or --system) forces the current Python.

set -e
cd "$(dirname "$0")"

USE_VENV=1
for a in "$@"; do
    case "$a" in
        --no-venv|--system) USE_VENV=0 ;;
        --debug) export CRISPZ_LOG_LEVEL=debug ;;
    esac
done

# The base Python
if command -v python3.10 >/dev/null 2>&1; then
    PYCMD="python3.10"
elif command -v python3 >/dev/null 2>&1; then
    PYCMD="python3"
else
    echo "[ERROR] Python not found."
    exit 1
fi

RUNPY="$PYCMD"
if [ "$USE_VENV" -eq 1 ] && [ -x ".venv/bin/python" ]; then
    RUNPY=".venv/bin/python"
fi

# The default ESRGAN_DIR when it is not set
if [ -z "$ESRGAN_DIR" ]; then
    export ESRGAN_DIR="$(pwd)/upscale_models"
fi

echo "=== crispz-studio - run ==="
echo "Python     = $RUNPY"
echo "ESRGAN_DIR = $ESRGAN_DIR"
[ -n "$CRISPZ_LOG_LEVEL" ] && echo "Log level  = $CRISPZ_LOG_LEVEL  (run.sh --debug)"
echo
echo "--- Hardware detection ---"
$RUNPY _hw_check.py
echo

echo "--- Starting the Gradio UI ---"
echo "Open http://127.0.0.1:7860 in your browser"
echo
$RUNPY app.py
