@echo off
REM Update crispz-studio (Windows): fetches the GitHub commits then brings the
REM dependencies back in line with the lock, WITHOUT breaking the existing install.
REM
REM What it does, in order:
REM   1. saves the installed versions (a rollback stays possible)
REM   2. git pull (refusing to overwrite uncommitted local changes)
REM   3. reinstalls the deps ONLY when the deps file has changed
REM   4. checks that torch/CUDA and the pipeline still work
REM
REM torch protection: a transitive resolution can replace a +cuXXX build with a
REM CPU wheel and break the GPU. We note the version before/after and warn when
REM it has changed.
REM
REM   --force-deps   reinstall the deps even when nothing has changed
REM   --no-pull      skip the git pull (resynchronise the deps only)
REM   --shared       use requirements.txt instead of the lock (a shared venv)

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

REM --- The interpreter ---
set "RUNPY="
if exist ".venv\Scripts\python.exe" set "RUNPY=.venv\Scripts\python.exe"
if not defined RUNPY (
    where py >nul 2>&1 && ( set "RUNPY=py -3.10" ) || ( set "RUNPY=python" )
)
echo Interpreter: !RUNPY!

REM --- 0. The state before: the torch version + the fingerprint of the deps file ---
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
        REM Never overwrite local work: _update_check.py --guard blocks when the commits
        REM to fetch touch a file modified here, or add a file already present here outside
        REM git. Otherwise, git pull --ff-only KEEPS the local changes: an untracked config,
        REM tests or wildcards no longer block anything.
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

REM --- 2. Has the deps file changed ? ---
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
        REM Pillow is outside the lock (gradio's pillow^<12 bound) -> installed apart,
        REM otherwise an update would REGRESS the fixed version. See install.bat.
        !RUNPY! -m pip install --no-deps --upgrade "pillow==12.3.0" >nul 2>&1
    )
    if errorlevel 1 (
        echo [ERROR] pip install failed. The environment may be inconsistent.
        echo   Possible restore: !RUNPY! -m pip install -r "%TEMP%\cz_pip_before.txt"
        pause & exit /b 1
    )
)
echo.

REM --- 3. Has torch been replaced ? (the classic trap: a +cuXXX build -^> CPU) ---
set "TORCH_AFTER="
for /f "delims=" %%v in ('!RUNPY! -c "import torch;print(torch.__version__)" 2^>nul') do set "TORCH_AFTER=%%v"
if defined TORCH_BEFORE if not "!TORCH_BEFORE!"=="!TORCH_AFTER!" (
    echo [WARNING] torch changed: !TORCH_BEFORE!  -^>  !TORCH_AFTER!
    echo    If the +cuXXX suffix is gone, the GPU will not be used any more.
    echo    Restore: !RUNPY! -m pip install torch==!TORCH_BEFORE! --index-url https://download.pytorch.org/whl/cu128
    echo.
)

REM --- 4. Final checks ---
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

REM --- 5. Config news: report the keys added in the sample ---
if exist "config.txt" if exist "config-sample.txt" (
    !RUNPY! -c "import json;a=json.load(open('config.txt',encoding='utf-8'));b=json.load(open('config-sample.txt',encoding='utf-8'));n=[k for k in b if k not in a and not k.startswith('_')];print('New config keys available: '+', '.join(n) if n else 'config.txt is up to date.')" 2>nul
    echo   ^(config.txt is never overwritten: add the keys you want by hand^)
)
echo.

echo === Update OK ===
if exist "CHANGELOG.md" echo What's new: see CHANGELOG.md
echo Run: run.bat  ^(or boot_check.bat for a full diagnostic^)
REM An explicit exit code: that is how boot_check.bat tells a finished update from a
REM failed one (a warning from step 5 used to leave errorlevel at 1).
endlocal & exit /b 0
