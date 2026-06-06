@echo off
title crispz-studio - RTX 5090 (local)
cd /d "%~dp0"
echo ============================================
echo  crispz-studio - RTX 5090 (local 127.0.0.1)
echo ============================================
echo.
REM Optimisations CUDA (sans danger, BF16 deja utilise par Z-Image)
set NVIDIA_TF32_OVERRIDE=1
set CUDA_CACHE_MAXSIZE=4294967296
set CUDA_AUTO_BOOST=1
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set GRADIO_SERVER_PORT=7860
REM Delegue au run.bat (detection venv + ESRGAN_DIR + lancement)
call "%~dp0run.bat" %*
