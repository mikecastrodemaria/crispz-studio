@echo off
REM Boot check + LAN + a Cloudflare tunnel (a public URL).
REM NO authentication at all: the app becomes reachable from the Internet (see SECURITY.md).
REM Personal config: cloudflare.local.bat (CF_TUNNEL / CF_PORT), not versioned.
call "%~dp0boot_check.bat" --web %*
