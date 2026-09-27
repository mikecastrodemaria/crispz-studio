@echo off
REM Install for crispz (Windows).
REM
REM Default: an ISOLATED .venv (it does NOT inherit the global site-packages)
REM installed from requirements-lock.txt -> a reproducible environment, versions
REM under control, no risk of breaking another project. It downloads its own
REM torch (~3.5 GB).
REM
REM   --shared    the old behaviour: a --system-site-packages venv that INHERITS
REM               the global torch. Lighter on the disk, but it also inherits
REM               diffusers/accelerate/numpy/pillow -> versions nobody chose, and
REM               shared with the other crispz forks.
REM   --no-venv   installs straight onto the current Python.

setlocal enabledelayedexpansion
cd /d "%~dp0"

REM The pipeline expected for this model family. The ONLY line that differs
REM between crispz-studio (ZImage), crispz-krea (Flux) and crispz-qwen-edit (Qwen).
set CHECK_PIPE=ZImageImg2ImgPipeline

REM --- flags ---
set USE_VENV=1
set ISOLATED=1
set FACESWAP=1
set FACESWAP_MODEL=0
:argloop
if "%~1"=="" goto argdone
if /I "%~1"=="--no-venv" set USE_VENV=0
if /I "%~1"=="--system" set USE_VENV=0
if /I "%~1"=="--shared" set ISOLATED=0
if /I "%~1"=="--no-faceswap" set FACESWAP=0
if /I "%~1"=="--faceswap-model" set FACESWAP_MODEL=1
shift
goto argloop
:argdone
if "!USE_VENV!"=="0" set ISOLATED=0

echo === crispz - install Windows ===
if "!ISOLATED!"=="1" (
    echo Mode: ISOLATED venv ^(reproducible, dedicated torch^)
) else (
    echo Mode: shared / system venv ^(inherits the global torch^)
)
echo.

REM 1) The base Python
where py >nul 2>&1
if errorlevel 1 (
    where python >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python not found. Install Python 3.10+ from python.org.
        exit /b 1
    )
    set PYCMD=python
) else (
    py -3.10 -c "import sys" >nul 2>&1
    if errorlevel 1 ( set PYCMD=py ) else ( set PYCMD=py -3.10 )
)
echo Base Python: !PYCMD!
!PYCMD! --version
echo.

REM 2) torch + CUDA. In ISOLATED mode torch comes from requirements-lock.txt: we
REM    check nothing here. In shared/system mode, it must already be present.
if "!ISOLATED!"=="1" (
    echo Isolated mode: torch is installed in the venv from the lock. Nothing to check.
) else (
    !PYCMD! -c "import torch,sys; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda); sys.exit(0 if torch.cuda.is_available() else 2)"
    if errorlevel 2 (
        echo.
        echo [WARN] PyTorch present but CUDA unavailable. Generating on the CPU will be very slow.
        goto torch_ok
    )
    if errorlevel 1 (
        echo.
        echo [ERROR] PyTorch not found. Install your PyTorch + CUDA build first, then run this again.
        echo Example ^(CUDA 12.8^): !PYCMD! -m pip install torch --index-url https://download.pytorch.org/whl/cu128
        echo Or run again without --shared for an isolated venv that installs its own torch.
        exit /b 1
    )
)
:torch_ok
echo.

REM 3) A broken xformers ? neutralise it SYSTEM-side. Only useful when the venv
REM    inherits the global one (--shared mode) or under --no-venv.
if not "!ISOLATED!"=="1" (
    !PYCMD! -c "import xformers.ops" >nul 2>&1
    if not errorlevel 1 (
        echo xformers OK.
    ) else (
        !PYCMD! -c "import xformers" >nul 2>&1
        if not errorlevel 1 (
            echo [WARN] xformers installed but does not load ^(torch DLL/ABI mismatch^). Uninstalling.
            !PYCMD! -m pip uninstall -y xformers
        )
    )
    echo.
)

REM 4) venv
set RUNPY=!PYCMD!
if "!USE_VENV!"=="1" (
    if not exist ".venv\Scripts\python.exe" (
        if "!ISOLATED!"=="1" (
            echo Creating the .venv ^(ISOLATED^)...
            !PYCMD! -m venv .venv
        ) else (
            echo Creating the .venv ^(--system-site-packages: inherits torch^)...
            !PYCMD! -m venv --system-site-packages .venv
        )
    ) else (
        echo .venv already there, reused as is.
        echo   For a clean start: delete .venv then run this again.
    )
    if exist ".venv\Scripts\python.exe" (
        set RUNPY=.venv\Scripts\python.exe
        .venv\Scripts\python.exe -m pip install --quiet --upgrade pip setuptools wheel
    ) else (
        echo [WARN] could not create the venv -^> falling back to the current Python.
    )
) else (
    echo Mode --no-venv: installing on the current Python.
)
echo Install interpreter: !RUNPY!
echo.

REM 5) Install the deps. In isolated mode we prefer the lock (exact, validated
REM    versions, torch cu128 included). Otherwise requirements.txt (wide bounds).
set REQFILE=requirements.txt
if "!ISOLATED!"=="1" if exist "requirements-lock.txt" set REQFILE=requirements-lock.txt
echo Installing the dependencies from !REQFILE! ...
if "!REQFILE!"=="requirements-lock.txt" echo   ^(includes torch cu128, ~3.5 GB to download the first time^)
REM Pillow is filtered out of the file: gradio 5.50 declares pillow^<12 and would
REM refuse to resolve with the 12.x pin. It is installed just after, with --no-deps.
REM The pin stays in the lock so that Dependabot sees the fixed version.
REM onnxruntime (the CPU build) is filtered out too: rembg pulls it as a dependency,
REM and once installed it HIDES onnxruntime-gpu at import time (the same module name,
REM the CPU one wins) -^> faceswap, GFPGAN/CodeFormer and rembg fall back silently
REM to the CPU although onnxruntime-gpu is there. We keep the GPU build only.
set "REQTMP=%TEMP%\cz_req_nopillow.txt"
findstr /V /B /C:"pillow==" /C:"onnxruntime==" "!REQFILE!" > "!REQTMP!"
!RUNPY! -m pip install -r "!REQTMP!"
if errorlevel 1 (
    echo [ERROR] pip install failed. Check the log above.
    exit /b 1
)
echo.

REM 5bis) Pillow, installed APART and with --no-deps.
REM   gradio 5.50 declares "pillow<12.0", but the Pillow CVEs (including the ones
REM   reachable through the images the user opens) are only fixed in 12.x. That
REM   bound of gradio's is conservative: checked against this code base, Pillow 12
REM   works. So we install it afterwards, without triggering again the resolution
REM   that would refuse the combination.
set "PILLOW_PIN=pillow==12.3.0"
echo Installing !PILLOW_PIN! ^(apart: works around gradio's pillow^<12 bound^)...
!RUNPY! -m pip install --no-deps --upgrade "!PILLOW_PIN!"
if errorlevel 1 echo [WARN] Pillow install failed -^> inherited version kept.
echo.

REM 6) Check that diffusers exposes the pipeline of this model family
!RUNPY! -c "from diffusers import !CHECK_PIPE!; print('!CHECK_PIPE! OK')"
if errorlevel 1 (
    echo [ERROR] diffusers does not contain !CHECK_PIPE!.
    exit /b 1
)
echo.

REM 7) Optional deps. The lock already contains them; in non-isolated mode the
REM    dedicated files are still the way in.
if "!FACESWAP!"=="1" if not "!REQFILE!"=="requirements-lock.txt" (
    echo Installing the FaceSwap deps ^(insightface + onnxruntime-gpu^)...
    !RUNPY! -m pip install -r requirements-faceswap.txt
    if errorlevel 1 echo [WARN] FaceSwap install failed ^(not blocking^). The feature stays off.
    echo Installing the extras ^(rembg for Remove BG^)...
    !RUNPY! -m pip install -r requirements-extra.txt
    if errorlevel 1 echo [WARN] extras install failed ^(not blocking^).
    echo.
)

REM 8) Model folders
for %%D in (upscale_models checkpoints loras faceswap) do if not exist "%%D" mkdir "%%D"
echo Folders ready: upscale_models (ESRGAN), checkpoints, loras, faceswap.
echo.

REM 9) Local config: copies config-sample.txt -> config.txt when absent
if not exist "config.txt" (
    if exist "config-sample.txt" (
        copy /Y "config-sample.txt" "config.txt" >nul
        echo config.txt created from config-sample.txt ^(edit it for your settings^).
    )
)
echo.

REM 10) The inswapper model (FaceSwap) - opt-in ^(528 MB, licence^): --faceswap-model
if "!FACESWAP_MODEL!"=="1" (
    if not exist "faceswap\inswapper_128.onnx" (
        echo Downloading the inswapper_128.onnx model ^(~528 MB^)...
        !RUNPY! -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx', 'faceswap/inswapper_128.onnx'); print('inswapper OK')"
    ) else (
        echo inswapper model already there.
    )
    echo.
)

echo === Install OK. Run run.bat ===
echo     Options: --shared ^(venv inheriting the global torch^)  --no-venv ^(current Python^)
echo              --no-faceswap ^(skip insightface^)  --faceswap-model ^(download inswapper^)
endlocal
