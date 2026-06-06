@echo off
setlocal enabledelayedexpansion
title crispz-studio - Boot Check RTX 5090
color 0A
cd /d "%~dp0"

echo ====================================================
echo    crispz-studio - RTX 5090 Boot Diagnostic
echo ====================================================
echo.

REM --- Interpreteur Python (venv prioritaire) ---
set "RUNPY="
if exist ".venv\Scripts\python.exe" set "RUNPY=.venv\Scripts\python.exe"
if not defined RUNPY (
    where py >nul 2>&1 && ( set "RUNPY=py -3.10" ) || ( set "RUNPY=python" )
)

REM --- 1. GPU ---
echo [1/5] Detection GPU...
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,temperature.gpu --format=csv,noheader,nounits > "%TEMP%\cz_gpu.txt" 2>nul
if errorlevel 1 (
    echo    [ERREUR] nvidia-smi introuvable. Verifie les drivers NVIDIA.
) else (
    for /f "tokens=1,2,3,4,5 delims=," %%a in (%TEMP%\cz_gpu.txt) do (
        echo    GPU        : %%a
        echo    Driver     : %%b
        echo    VRAM Total : %%c MB
        echo    VRAM Libre : %%d MB
        echo    Temp       : %%e C
    )
)
echo.

REM --- 2. Python ---
echo [2/5] Interpreteur Python : !RUNPY!
!RUNPY! --version 2>nul
if errorlevel 1 ( echo    [ERREUR] Python introuvable. & pause & exit /b 1 )
echo.

REM --- 3. PyTorch / CUDA ---
echo [3/5] PyTorch / CUDA...
!RUNPY! -c "import torch; print('    torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| bf16', torch.cuda.is_bf16_supported())" 2>nul
if errorlevel 1 ( echo    [ATTENTION] torch non detecte -> lance install.bat. )
echo.

REM --- 4. Pipelines Z-Image ---
echo [4/5] diffusers Z-Image...
!RUNPY! -c "from diffusers import ZImagePipeline, ZImageImg2ImgPipeline; print('    ZImage pipelines OK')" 2>nul
if errorlevel 1 ( echo    [ATTENTION] ZImage pipelines indisponibles -> install.bat. )
echo.

REM --- 5. Modeles ---
echo [5/5] Modeles...
set "ZDIR=D:\Github\sdlibs\models\Stable-diffusion\Z-Image"
if exist "%ZDIR%" (
    set "N=0"
    for %%f in ("%ZDIR%\*.safetensors") do set /a N+=1
    echo    Checkpoints Z-Image ^(%ZDIR%^) : !N!
) else (
    echo    [INFO] %ZDIR% introuvable (regle checkpoints_dir dans config.txt).
)
echo.

REM --- Optimisations CUDA ---
set NVIDIA_TF32_OVERRIDE=1
set CUDA_CACHE_MAXSIZE=4294967296
set CUDA_AUTO_BOOST=1
set CUDA_DEVICE_ORDER=PCI_BUS_ID

echo ====================================================
echo    Checks termines. Lancement de crispz-studio...
echo ====================================================
timeout /t 2 /nobreak >nul
endlocal
call "%~dp0run.bat" %*
