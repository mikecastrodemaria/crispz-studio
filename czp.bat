@echo off
REM crispz family CLI protocol v1 (JSON in/out). Examples:
REM   czp caps
REM   czp gen --spec spec.json
REM Routes to the running app when there is one; loads the pipeline otherwise.
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" cz_protocol.py %*
) else (
    python cz_protocol.py %*
)
