@echo off
REM crispz-studio - batch CivitAI enrichment (previews / trigger words / example prompts /
REM new-version warnings). Pass-through args, e.g.:  civitai_index.bat --kind loras --force
REM Run several of these at once (or use civitai_index_parallel.bat) to fetch in parallel.
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if exist "env\Scripts\python.exe" set "PY=env\Scripts\python.exe"
if exist "venv\Scripts\python.exe" set "PY=venv\Scripts\python.exe"
"%PY%" cz_civitai_batch.py %*
endlocal
