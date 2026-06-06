@echo off
title crispz-studio - RTX 5090 (LAN)
cd /d "%~dp0"
echo ============================================
echo  crispz-studio - RTX 5090 - Accessible sur le LAN
echo ============================================
echo.
set NVIDIA_TF32_OVERRIDE=1
set CUDA_CACHE_MAXSIZE=4294967296
set CUDA_AUTO_BOOST=1
set CUDA_DEVICE_ORDER=PCI_BUS_ID
REM Gradio lit ces variables nativement: 0.0.0.0 = ecoute toutes les interfaces.
set GRADIO_SERVER_NAME=0.0.0.0
set GRADIO_SERVER_PORT=7860
echo Acces LAN : depuis une autre machine du reseau, ouvre :
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do echo    http://%%a:7860
echo.
call "%~dp0run.bat" %*
