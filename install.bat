@echo off
REM Install pour crispz (Windows).
REM
REM Defaut: venv .venv ISOLE (n'herite PAS du site-packages global) installe
REM depuis requirements-lock.txt -> environnement reproductible, versions
REM maitrisees, aucun risque de casser un autre projet. Telecharge son propre
REM torch (~3,5 Go).
REM
REM   --shared    ancien comportement: venv --system-site-packages qui HERITE du
REM               torch global. Plus leger sur le disque, mais fait aussi heriter
REM               diffusers/accelerate/numpy/pillow -> versions non choisies et
REM               partagees avec les autres forks crispz.
REM   --no-venv   installe directement sur le Python courant.

setlocal enabledelayedexpansion
cd /d "%~dp0"

REM Pipeline attendu pour cette famille de modele. SEULE ligne qui differe
REM entre crispz-studio (ZImage), crispz-krea (Flux) et crispz-qwen-edit (Qwen).
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

REM 1) Python de base
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

REM 2) torch + CUDA. En mode ISOLE, torch vient de requirements-lock.txt: on ne
REM    verifie rien ici. En mode partage/systeme, il doit deja etre present.
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

REM 3) xformers casse ? le neutraliser cote SYSTEME. Utile seulement si le venv
REM    herite du global (mode --shared) ou en --no-venv.
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

REM 5) Installer les deps. En mode isole on prefere le lock (versions exactes
REM    validees, torch cu128 inclus). Sinon requirements.txt (bornes larges).
set REQFILE=requirements.txt
if "!ISOLATED!"=="1" if exist "requirements-lock.txt" set REQFILE=requirements-lock.txt
echo Installing the dependencies from !REQFILE! ...
if "!REQFILE!"=="requirements-lock.txt" echo   ^(includes torch cu128, ~3.5 GB to download the first time^)
REM Pillow est filtre du fichier: gradio 5.50 declare pillow^<12 et refuserait
REM de resoudre avec le pin 12.x. Il est pose juste apres, en --no-deps.
REM Le pin reste dans le lock pour que Dependabot voie la version corrigee.
REM onnxruntime (build CPU) est filtre lui aussi: rembg le tire en dependance, et
REM une fois installe il MASQUE onnxruntime-gpu a l'import (meme nom de module,
REM le CPU gagne) -^> faceswap, GFPGAN/CodeFormer et rembg tombent silencieusement
REM sur le CPU malgre onnxruntime-gpu present. On ne garde que le build GPU.
set "REQTMP=%TEMP%\cz_req_nopillow.txt"
findstr /V /B /C:"pillow==" /C:"onnxruntime==" "!REQFILE!" > "!REQTMP!"
!RUNPY! -m pip install -r "!REQTMP!"
if errorlevel 1 (
    echo [ERROR] pip install failed. Check the log above.
    exit /b 1
)
echo.

REM 5bis) Pillow, installe A PART et en --no-deps.
REM   gradio 5.50 declare "pillow<12.0", mais les CVE Pillow (dont celles
REM   atteignables via les images que l'utilisateur ouvre) ne sont corrigees
REM   qu'en 12.x. Cette borne de gradio est conservatrice: verifie sur cette
REM   base de code, Pillow 12 fonctionne. On installe donc apres coup, sans
REM   redeclencher la resolution qui refuserait la combinaison.
set "PILLOW_PIN=pillow==12.3.0"
echo Installing !PILLOW_PIN! ^(apart: works around gradio's pillow^<12 bound^)...
!RUNPY! -m pip install --no-deps --upgrade "!PILLOW_PIN!"
if errorlevel 1 echo [WARN] Pillow install failed -^> inherited version kept.
echo.

REM 6) Verifier que diffusers expose le pipeline de cette famille de modele
!RUNPY! -c "from diffusers import !CHECK_PIPE!; print('!CHECK_PIPE! OK')"
if errorlevel 1 (
    echo [ERROR] diffusers does not contain !CHECK_PIPE!.
    exit /b 1
)
echo.

REM 7) Deps optionnelles. Le lock les contient deja; en mode non-isole il faut
REM    encore passer par les fichiers dedies.
if "!FACESWAP!"=="1" if not "!REQFILE!"=="requirements-lock.txt" (
    echo Installing the FaceSwap deps ^(insightface + onnxruntime-gpu^)...
    !RUNPY! -m pip install -r requirements-faceswap.txt
    if errorlevel 1 echo [WARN] FaceSwap install failed ^(not blocking^). The feature stays off.
    echo Installing the extras ^(rembg for Remove BG^)...
    !RUNPY! -m pip install -r requirements-extra.txt
    if errorlevel 1 echo [WARN] extras install failed ^(not blocking^).
    echo.
)

REM 8) Dossiers de modeles
for %%D in (upscale_models checkpoints loras faceswap) do if not exist "%%D" mkdir "%%D"
echo Folders ready: upscale_models (ESRGAN), checkpoints, loras, faceswap.
echo.

REM 9) Config locale: copie config-sample.txt -> config.txt si absent
if not exist "config.txt" (
    if exist "config-sample.txt" (
        copy /Y "config-sample.txt" "config.txt" >nul
        echo config.txt created from config-sample.txt ^(edit it for your settings^).
    )
)
echo.

REM 10) Modele inswapper (FaceSwap) - opt-in ^(528 Mo, licence^): --faceswap-model
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
