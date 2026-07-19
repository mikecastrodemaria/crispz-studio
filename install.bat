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
    echo Mode: venv ISOLE ^(reproductible, torch dedie^)
) else (
    echo Mode: venv partage / systeme ^(herite du torch global^)
)
echo.

REM 1) Python de base
where py >nul 2>&1
if errorlevel 1 (
    where python >nul 2>&1
    if errorlevel 1 (
        echo [ERREUR] Python introuvable. Installe Python 3.10+ depuis python.org.
        exit /b 1
    )
    set PYCMD=python
) else (
    py -3.10 -c "import sys" >nul 2>&1
    if errorlevel 1 ( set PYCMD=py ) else ( set PYCMD=py -3.10 )
)
echo Python de base: !PYCMD!
!PYCMD! --version
echo.

REM 2) torch + CUDA. En mode ISOLE, torch vient de requirements-lock.txt: on ne
REM    verifie rien ici. En mode partage/systeme, il doit deja etre present.
if "!ISOLATED!"=="1" (
    echo Mode isole: torch sera installe dans le venv depuis le lock. Rien a verifier.
) else (
    !PYCMD! -c "import torch,sys; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda); sys.exit(0 if torch.cuda.is_available() else 2)"
    if errorlevel 2 (
        echo.
        echo [AVERT] PyTorch present mais CUDA non disponible. La generation en CPU sera tres lente.
        goto torch_ok
    )
    if errorlevel 1 (
        echo.
        echo [ERREUR] PyTorch introuvable. Installe d'abord ton build PyTorch + CUDA, puis relance.
        echo Exemple ^(CUDA 12.8^): !PYCMD! -m pip install torch --index-url https://download.pytorch.org/whl/cu128
        echo Ou relance sans --shared pour un venv isole qui installe son propre torch.
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
            echo [AVERT] xformers installe mais ne charge pas ^(DLL/ABI torch incompatible^). Desinstallation.
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
            echo Creation du venv .venv ^(ISOLE^)...
            !PYCMD! -m venv .venv
        ) else (
            echo Creation du venv .venv ^(--system-site-packages: herite de torch^)...
            !PYCMD! -m venv --system-site-packages .venv
        )
    ) else (
        echo Venv .venv deja present, reutilise en l'etat.
        echo   Pour repartir propre: supprime .venv puis relance.
    )
    if exist ".venv\Scripts\python.exe" (
        set RUNPY=.venv\Scripts\python.exe
        .venv\Scripts\python.exe -m pip install --quiet --upgrade pip setuptools wheel
    ) else (
        echo [AVERT] creation du venv impossible -^> repli sur le Python courant.
    )
) else (
    echo Mode --no-venv: install sur le Python courant.
)
echo Interpreteur d'install: !RUNPY!
echo.

REM 5) Installer les deps. En mode isole on prefere le lock (versions exactes
REM    validees, torch cu128 inclus). Sinon requirements.txt (bornes larges).
set REQFILE=requirements.txt
if "!ISOLATED!"=="1" if exist "requirements-lock.txt" set REQFILE=requirements-lock.txt
echo Installation des dependances depuis !REQFILE! ...
if "!REQFILE!"=="requirements-lock.txt" echo   ^(inclut torch cu128, ~3,5 Go de telechargement la premiere fois^)
!RUNPY! -m pip install -r !REQFILE!
if errorlevel 1 (
    echo [ERREUR] echec pip install. Verifie le log ci-dessus.
    exit /b 1
)
echo.

REM 6) Verifier que diffusers expose le pipeline de cette famille de modele
!RUNPY! -c "from diffusers import !CHECK_PIPE!; print('!CHECK_PIPE! OK')"
if errorlevel 1 (
    echo [ERREUR] diffusers ne contient pas !CHECK_PIPE!.
    exit /b 1
)
echo.

REM 7) Deps optionnelles. Le lock les contient deja; en mode non-isole il faut
REM    encore passer par les fichiers dedies.
if "!FACESWAP!"=="1" if not "!REQFILE!"=="requirements-lock.txt" (
    echo Installation des deps FaceSwap ^(insightface + onnxruntime-gpu^)...
    !RUNPY! -m pip install -r requirements-faceswap.txt
    if errorlevel 1 echo [AVERT] echec install FaceSwap ^(non bloquant^). La feature restera desactivee.
    echo Installation des extras ^(rembg pour Remove BG^)...
    !RUNPY! -m pip install -r requirements-extra.txt
    if errorlevel 1 echo [AVERT] echec install extras ^(non bloquant^).
    echo.
)

REM 8) Dossiers de modeles
for %%D in (upscale_models checkpoints loras faceswap) do if not exist "%%D" mkdir "%%D"
echo Dossiers prets: upscale_models (ESRGAN), checkpoints, loras, faceswap.
echo.

REM 9) Config locale: copie config-sample.txt -> config.txt si absent
if not exist "config.txt" (
    if exist "config-sample.txt" (
        copy /Y "config-sample.txt" "config.txt" >nul
        echo config.txt cree depuis config-sample.txt ^(edite-le pour tes reglages^).
    )
)
echo.

REM 10) Modele inswapper (FaceSwap) - opt-in ^(528 Mo, licence^): --faceswap-model
if "!FACESWAP_MODEL!"=="1" (
    if not exist "faceswap\inswapper_128.onnx" (
        echo Telechargement du modele inswapper_128.onnx ^(~528 Mo^)...
        !RUNPY! -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx', 'faceswap/inswapper_128.onnx'); print('inswapper OK')"
    ) else (
        echo Modele inswapper deja present.
    )
    echo.
)

echo === Install OK. Lance run.bat ===
echo     Options: --shared ^(venv qui herite du torch global^)  --no-venv ^(Python courant^)
echo              --no-faceswap ^(sauter insightface^)  --faceswap-model ^(telecharger inswapper^)
endlocal
