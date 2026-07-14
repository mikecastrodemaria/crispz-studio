@echo off
REM crispz-studio - launch N parallel CivitAI enrichment shards (disjoint file lists).
REM Usage:  civitai_index_parallel.bat [N]     (default N=4; kind=all)
REM Each shard is an independent process (its own window). CivitAI rate-limits: keep N
REM modest and set a CivitAI API key in the app (Advanced) or pass --api-key per shard.
setlocal enabledelayedexpansion
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "N=%~1"
if "%N%"=="" set "N=4"
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if exist "env\Scripts\python.exe" set "PY=env\Scripts\python.exe"
if exist "venv\Scripts\python.exe" set "PY=venv\Scripts\python.exe"
echo Launching %N% parallel shard(s), kind=all ...
for /L %%i in (1,1,%N%) do (
  start "civitai shard %%i/%N%" "%PY%" cz_civitai_batch.py --kind all --shard %%i/%N%
)
echo Done launching. Each shard runs in its own window.
endlocal
