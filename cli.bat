@echo off
REM CLI interactive pour crispz.
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

where py >nul 2>&1
if errorlevel 1 ( set PYCMD=python ) else (
    py -3.10 -c "import sys" >nul 2>&1
    if errorlevel 1 ( set PYCMD=py ) else ( set PYCMD=py -3.10 )
)

set RUNPY=!PYCMD!
if "!USE_VENV!"=="1" if exist ".venv\Scripts\python.exe" set RUNPY=.venv\Scripts\python.exe

if "%ESRGAN_DIR%"=="" (
    if exist "D:\Github\sdlibs\models\ESRGAN" (
        set ESRGAN_DIR=D:\Github\sdlibs\models\ESRGAN
    ) else (
        set ESRGAN_DIR=%~dp0upscale_models
    )
)

!RUNPY! cli_interactive.py
endlocal
