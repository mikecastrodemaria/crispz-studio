@echo off
REM Generic boot check for crispz-studio (replaces the old rtx5090 scripts).
REM
REM Diagnoses the machine BEFORE launching the app, whatever the card
REM (RTX 50xx / 40xx / 30xx / 20xx...), and stops dead when the configuration
REM cannot work -- rather than letting the app crash halfway through.
REM
REM The decisive check is done by _hw_check.py: it compares the GPU's sm_XX with
REM the list of architectures compiled into the installed torch build. That is
REM what catches the "RTX 50xx + non-cu128 torch" case -- the WinError 127
REM on torch_cuda.dll.
REM
REM   --no-run   diagnose only, do not launch the app
REM   --no-update  do not look for a GitHub update (or CRISPZ_NO_UPDATE_CHECK=1)
REM   --lan      listen on the LAN (0.0.0.0) instead of 127.0.0.1
REM   --web      LAN + a Cloudflare tunnel (a public URL)
REM   any other argument is passed on to run.bat
REM
REM WARNING about --lan / --web: the app has NO authentication at all and serves
REM the output folder + the model folders. See "Scope" in SECURITY.md.

setlocal enabledelayedexpansion
title crispz-studio - Boot Check
cd /d "%~dp0"

set "NORUN=0"
set "NOUPDATE=0"
set "EXPOSE="
set "PASSTHRU="
:argloop
if "%~1"=="" goto argdone
if /I "%~1"=="--no-run" (
    set "NORUN=1"
) else if /I "%~1"=="--no-update" (
    set "NOUPDATE=1"
) else if /I "%~1"=="--lan" (
    set "EXPOSE=lan"
) else if /I "%~1"=="--web" (
    set "EXPOSE=web"
) else (
    set "PASSTHRU=!PASSTHRU! %~1"
)
shift
goto argloop
:argdone

echo ====================================================
echo    crispz-studio - Boot Check
echo ====================================================
echo.

REM --- The interpreter (the venv wins) ---
set "RUNPY="
if exist ".venv\Scripts\python.exe" set "RUNPY=.venv\Scripts\python.exe"
if not defined RUNPY (
    where py >nul 2>&1 && ( set "RUNPY=py -3.10" ) || ( set "RUNPY=python" )
)

echo [1/5] Python : !RUNPY!
!RUNPY! --version 2>nul
if errorlevel 1 (
    echo    [ERROR] Python not found. Install Python 3.10+ then run install.bat.
    pause & exit /b 1
)
echo.

REM --- GitHub update: OFFERED, never imposed (see _update_check.py) ---
REM Offered only when it is safe: none of the commits to fetch touches a file modified
REM here, nor adds a file already present outside git. With no answer within 20 s: N,
REM and the app starts as it is. --no-update or CRISPZ_NO_UPDATE_CHECK=1: the step is
REM skipped. Offline, without git or without a tracked branch: it says so and moves on.
if "!NOUPDATE!"=="0" (
    echo [UPDATE] GitHub updates...
    !RUNPY! _update_check.py
    set "UPD=!errorlevel!"
    if "!UPD!"=="10" (
        choice /C YN /T 20 /D N /M "    Update now? N by default in 20 s"
        if errorlevel 2 (
            echo    Starting without updating. Later: update.bat
        ) else (
            call "%~dp0update.bat"
            if errorlevel 1 (
                echo    [ERROR] Update interrupted, see above. The app was not started.
                pause & exit /b 1
            )
            echo    Update done. Back to the checks...
        )
    )
    if "!UPD!"=="11" echo    Starting without updating.
    echo.
)

REM --- 2. Driver / card state (raw information) ---
echo [2/5] Driver NVIDIA...
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu --format=csv,noheader,nounits > "%TEMP%\cz_gpu.txt" 2>nul
if errorlevel 1 (
    echo    [INFO] nvidia-smi not found ^(no NVIDIA GPU, or drivers missing^).
) else (
    for /f "tokens=1,2,3,4,5 delims=," %%a in (%TEMP%\cz_gpu.txt) do (
        echo    Card    : %%a
        echo    Driver  : %%b
        echo    VRAM    : %%d / %%c MB used   ^| Temp: %%e C
    )
    del "%TEMP%\cz_gpu.txt" >nul 2>&1
)
echo.

REM --- 3. THE check: does torch support THIS card ? + recommendations ---
echo [3/5] PyTorch / GPU / suggested settings...
echo.
!RUNPY! _hw_check.py
set "HW=!errorlevel!"
echo.
if "!HW!"=="1" (
    echo    [ERROR] PyTorch missing -^> run install.bat.
    pause & exit /b 1
)
if "!HW!"=="3" (
    echo    [BLOCKING] torch does not support this card ^(see the fix above^).
    echo    The app would crash on the first CUDA allocation. Stopping.
    pause & exit /b 3
)
if "!HW!"=="2" echo    [WARN] CPU mode: generating will be very slow.

REM --- 4. The diffusers pipeline of this model family ---
echo [4/5] diffusers...
!RUNPY! -c "from diffusers import ZImagePipeline, ZImageImg2ImgPipeline; print('    ZImage pipelines OK')" 2>nul
if errorlevel 1 echo    [WARNING] ZImage pipelines unavailable -^> run install.bat / update.bat.
echo.

REM --- 5. Models: we read the REAL folders of the config, not a hardcoded path ---
echo [5/5] Models...
REM %% : in batch a literal %% is written double, otherwise cmd eats the Python format.
!RUNPY! -c "import os,cz_pipeline as p;[print('    %%-11s %%3d file(s)  %%s' %% (n, (len([f for f in os.listdir(d) if f.lower().endswith(('.safetensors','.gguf','.ckpt','.pt','.sft'))]) if os.path.isdir(d) else 0), d if os.path.isdir(d) else '(folder missing)')) for n,d in (('checkpoints',p.CHECKPOINTS_DIR),('extra',p.CHECKPOINTS_EXTRA_DIR),('loras',p.LORAS_DIR)) if d]" 2>nul
if errorlevel 1 echo    [INFO] could not read the config ^(config.txt missing? run install.bat^).
echo.

REM --- CUDA optimisations (no effect without an NVIDIA GPU) ---
set NVIDIA_TF32_OVERRIDE=1
set CUDA_CACHE_MAXSIZE=4294967296
set CUDA_AUTO_BOOST=1
set CUDA_DEVICE_ORDER=PCI_BUS_ID
REM A fixed port (inherited from the old run_quality_*.bat): keeps Gradio from
REM going to 7861+ when a previous instance has not released the port yet.
if not defined GRADIO_SERVER_PORT set GRADIO_SERVER_PORT=7860

REM --- Network exposure (--lan / --web): Gradio reads these variables natively ---
set "CF_PORT=7860"
if defined EXPOSE (
    echo ----------------------------------------------------
    echo  [SECURITY] Network exposure requested ^(--!EXPOSE!^).
    echo  crispz-studio has NO authentication and serves your output folder
    echo  as well as your model folders. Only expose it on a network you
    echo  trust. See the "Scope" section of SECURITY.md.
    echo ----------------------------------------------------
    set GRADIO_SERVER_NAME=0.0.0.0
    set GRADIO_SERVER_PORT=!CF_PORT!
    echo LAN access:
    for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do echo    http://%%a:!CF_PORT!
    echo.
)
if /I "!EXPOSE!"=="web" (
    REM Personal config, NOT versioned (see cloudflare.local.bat.example):
    REM   CF_TUNNEL = a named cloudflared tunnel, otherwise an ephemeral quick tunnel.
    set "CF_TUNNEL="
    if exist "%~dp0cloudflare.local.bat" call "%~dp0cloudflare.local.bat"
    if defined CF_PORT set GRADIO_SERVER_PORT=!CF_PORT!
    where cloudflared >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] cloudflared not found in PATH.
        echo    Install it: winget install --id Cloudflare.cloudflared
        pause & exit /b 1
    )
    if defined CF_TUNNEL (
        echo [Cloudflare] Named tunnel: !CF_TUNNEL!
        start "Cloudflare Tunnel" cloudflared tunnel run !CF_TUNNEL!
    ) else (
        echo [Cloudflare] Ephemeral quick tunnel: the https://xxxx.trycloudflare.com URL
        echo              shows up in the "Cloudflare Tunnel" window.
        start "Cloudflare Tunnel" cloudflared tunnel --url http://localhost:!CF_PORT!
    )
    echo.
)

if "!NORUN!"=="1" (
    echo ====================================================
    echo    Diagnostic done ^(--no-run: app not started^).
    echo ====================================================
    endlocal & exit /b 0
)
echo ====================================================
echo    Checks OK. Starting crispz-studio...
echo ====================================================
timeout /t 2 /nobreak >nul
call "%~dp0run.bat" %PASSTHRU%
if /I "!EXPOSE!"=="web" (
    echo.
    echo ----------------------------------------------------
    echo  Stopped. Remember to close the Cloudflare tunnel window.
    echo ----------------------------------------------------
)
endlocal
