@echo off
REM Lance crispz (UI Gradio) avec detection hardware.
REM Utilise .venv s'il existe; --no-venv (ou --system) force le Python courant.

setlocal enabledelayedexpansion
cd /d "%~dp0"

set USE_VENV=1
:argloop
if "%~1"=="" goto argdone
if /I "%~1"=="--no-venv" set USE_VENV=0
if /I "%~1"=="--system" set USE_VENV=0
shift
goto argloop
:argdone

REM Python de base
where py >nul 2>&1
if errorlevel 1 (
    set PYCMD=python
) else (
    py -3.10 -c "import sys" >nul 2>&1
    if errorlevel 1 ( set PYCMD=py ) else ( set PYCMD=py -3.10 )
)

set RUNPY=!PYCMD!
if "!USE_VENV!"=="1" if exist ".venv\Scripts\python.exe" set RUNPY=.venv\Scripts\python.exe

REM ESRGAN_DIR: priorite a la variable existante, sinon dossier sdlibs s'il existe, sinon local
if "%ESRGAN_DIR%"=="" (
    if exist "D:\Github\sdlibs\models\ESRGAN" (
        set ESRGAN_DIR=D:\Github\sdlibs\models\ESRGAN
    ) else (
        set ESRGAN_DIR=%~dp0upscale_models
    )
)

echo === crispz - run ===
echo Python     = !RUNPY!
echo ESRGAN_DIR = !ESRGAN_DIR!
echo.
echo --- Detection hardware ---
!RUNPY! _hw_check.py
echo.

echo --- Lancement de l'UI Gradio ---
echo Ouvre http://127.0.0.1:7860 dans ton navigateur
echo.
!RUNPY! app.py
endlocal
