@echo off
REM Boot check + LAN access (0.0.0.0:7860). See boot_check.bat for the detail.
REM NO authentication at all: a trusted network only (see SECURITY.md).
call "%~dp0boot_check.bat" --lan %*
