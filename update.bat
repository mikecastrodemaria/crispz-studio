@echo off
REM Update crispz-studio (Windows): recupere les commits GitHub puis remet les
REM dependances en phase avec le lock, SANS casser l'installation existante.
REM
REM Fait, dans l'ordre:
REM   1. sauvegarde des versions installees (rollback possible)
REM   2. git pull (en refusant d'ecraser des modifications locales non commitees)
REM   3. reinstall des deps UNIQUEMENT si le fichier de deps a change
REM   4. verification que torch/CUDA et le pipeline fonctionnent encore
REM
REM Protection torch: une resolution transitive peut remplacer un build +cuXXX
REM par une roue CPU et casser le GPU. On releve la version avant/apres et on
REM alerte si elle a change.
REM
REM   --force-deps   reinstaller les deps meme si rien n'a change
REM   --no-pull      sauter le git pull (resynchroniser les deps seulement)
REM   --shared       utiliser requirements.txt au lieu du lock (venv partage)

setlocal enabledelayedexpansion
title crispz-studio - Update
cd /d "%~dp0"

set "FORCEDEPS=0"
set "DOPULL=1"
set "ISOLATED=1"
:argloop
if "%~1"=="" goto argdone
if /I "%~1"=="--force-deps" set "FORCEDEPS=1"
if /I "%~1"=="--no-pull" set "DOPULL=0"
if /I "%~1"=="--shared" set "ISOLATED=0"
shift
goto argloop
:argdone

echo === crispz-studio - update ===
echo.

REM --- Interpreteur ---
set "RUNPY="
if exist ".venv\Scripts\python.exe" set "RUNPY=.venv\Scripts\python.exe"
if not defined RUNPY (
    where py >nul 2>&1 && ( set "RUNPY=py -3.10" ) || ( set "RUNPY=python" )
)
echo Interpreter: !RUNPY!

REM --- 0. Etat avant: version torch + empreinte du fichier de deps ---
set REQFILE=requirements.txt
if "!ISOLATED!"=="1" if exist "requirements-lock.txt" set REQFILE=requirements-lock.txt
set "TORCH_BEFORE="
for /f "delims=" %%v in ('!RUNPY! -c "import torch;print(torch.__version__)" 2^>nul') do set "TORCH_BEFORE=%%v"
if defined TORCH_BEFORE (
    echo torch installed: !TORCH_BEFORE!
    !RUNPY! -m pip freeze > "%TEMP%\cz_pip_before.txt" 2>nul
    echo   ^(version snapshot: %TEMP%\cz_pip_before.txt^)
) else (
    echo torch not installed ^(first install? run install.bat^).
)
set "HASH_BEFORE="
if exist "!REQFILE!" for /f "delims=" %%h in ('certutil -hashfile "!REQFILE!" MD5 ^| findstr /R "^[0-9a-f][0-9a-f]*$"') do set "HASH_BEFORE=%%h"
echo.

REM --- 1. git pull ---
if "!DOPULL!"=="1" (
    where git >nul 2>&1
    if errorlevel 1 (
        echo [WARN] git not found -^> pull skipped. Update the files by hand.
    ) else (
        REM Ne jamais ecraser du travail local: _update_check.py --guard bloque si les
        REM commits a recuperer touchent un fichier modifie ici, ou ajoutent un fichier
        REM deja present ici hors de git. Ailleurs, git pull --ff-only CONSERVE les
        REM modifications locales: config, tests, wildcards non suivis ne bloquent plus.
        !RUNPY! _update_check.py --guard
        if errorlevel 1 (
            echo.
            echo   Commit / stash those files first, or run again with --no-pull to
            echo   resync the dependencies only.
            pause & exit /b 1
        )
        echo Fetching the commits ^(git pull^)...
        git pull --ff-only
        if errorlevel 1 (
            echo [ERROR] git pull failed ^(diverged branch?^). Sort it out by hand.
            pause & exit /b 1
        )
    )
    echo.
)

REM --- 2. Le fichier de deps a-t-il change ? ---
set "HASH_AFTER="
if exist "!REQFILE!" for /f "delims=" %%h in ('certutil -hashfile "!REQFILE!" MD5 ^| findstr /R "^[0-9a-f][0-9a-f]*$"') do set "HASH_AFTER=%%h"
set "NEEDDEPS=0"
if not "!HASH_BEFORE!"=="!HASH_AFTER!" set "NEEDDEPS=1"
if "!FORCEDEPS!"=="1" set "NEEDDEPS=1"
if not defined TORCH_BEFORE set "NEEDDEPS=1"

if "!NEEDDEPS!"=="0" (
    echo Dependencies: !REQFILE! unchanged -^> nothing to reinstall.
    echo   ^(--force-deps to force it^)
) else (
    echo Dependencies: updating from !REQFILE! ...
    set "REQTMP=%TEMP%\cz_req_nopillow.txt"
    findstr /V /B /C:"pillow==" "!REQFILE!" > "!REQTMP!"
    !RUNPY! -m pip install -r "!REQTMP!"
    if not errorlevel 1 (
        REM Pillow est hors du lock (borne pillow^<12 de gradio) -> pose a part,
        REM sinon un update ferait REGRESSER la version corrigee. Cf. install.bat.
        !RUNPY! -m pip install --no-deps --upgrade "pillow==12.3.0" >nul 2>&1
    )
    if errorlevel 1 (
        echo [ERROR] pip install failed. The environment may be inconsistent.
        echo   Possible restore: !RUNPY! -m pip install -r "%TEMP%\cz_pip_before.txt"
        pause & exit /b 1
    )
)
echo.

REM --- 3. torch a-t-il ete remplace ? (piege classique: build +cuXXX -^> CPU) ---
set "TORCH_AFTER="
for /f "delims=" %%v in ('!RUNPY! -c "import torch;print(torch.__version__)" 2^>nul') do set "TORCH_AFTER=%%v"
if defined TORCH_BEFORE if not "!TORCH_BEFORE!"=="!TORCH_AFTER!" (
    echo [WARNING] torch changed: !TORCH_BEFORE!  -^>  !TORCH_AFTER!
    echo    If the +cuXXX suffix is gone, the GPU will not be used any more.
    echo    Restore: !RUNPY! -m pip install torch==!TORCH_BEFORE! --index-url https://download.pytorch.org/whl/cu128
    echo.
)

REM --- 4. Verifications finales ---
echo Checking the install...
!RUNPY! _hw_check.py
set "HW=!errorlevel!"
echo.
if "!HW!"=="3" (
    echo [BLOCKING] torch no longer supports this card ^(see the fix above^).
    pause & exit /b 3
)
!RUNPY! -c "from diffusers import ZImagePipeline, ZImageImg2ImgPipeline; print('diffusers: ZImage pipelines OK')"
if errorlevel 1 (
    echo [ERROR] diffusers no longer ships the ZImage pipelines.
    echo   Run install.bat again, or restore: !RUNPY! -m pip install -r "%TEMP%\cz_pip_before.txt"
    pause & exit /b 1
)
!RUNPY! -c "import cz_ui; print('app: imports OK')"
if errorlevel 1 (
    echo [ERROR] the application no longer imports. See the traceback above.
    pause & exit /b 1
)
echo.

REM --- 5. Nouveautes de config: signaler les cles ajoutees dans le sample ---
if exist "config.txt" if exist "config-sample.txt" (
    !RUNPY! -c "import json;a=json.load(open('config.txt',encoding='utf-8'));b=json.load(open('config-sample.txt',encoding='utf-8'));n=[k for k in b if k not in a and not k.startswith('_')];print('New config keys available: '+', '.join(n) if n else 'config.txt is up to date.')" 2>nul
    echo   ^(config.txt is never overwritten: add the keys you want by hand^)
)
echo.

echo === Update OK ===
if exist "CHANGELOG.md" echo What's new: see CHANGELOG.md
echo Run: run.bat  ^(or boot_check.bat for a full diagnostic^)
REM Code de sortie explicite: boot_check.bat distingue ainsi une mise a jour terminee
REM d'une mise a jour en echec (un avertissement de l'etape 5 laissait errorlevel a 1).
endlocal & exit /b 0
