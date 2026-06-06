@echo off
title crispz-studio - RTX 5090 (WEB via Cloudflare)
cd /d "%~dp0"
echo ============================================
echo  crispz-studio - RTX 5090 - LAN + WEB (Cloudflare Tunnel)
echo ============================================
echo.
set NVIDIA_TF32_OVERRIDE=1
set CUDA_CACHE_MAXSIZE=4294967296
set CUDA_AUTO_BOOST=1
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set GRADIO_SERVER_NAME=0.0.0.0

REM --- Config Cloudflare PERSO en local (NON versionnee, voir cloudflare.local.bat.example) ---
REM   CF_TUNNEL = nom de ton tunnel cloudflared nomme (sinon: quick tunnel ephemere)
REM   CF_PORT   = port local expose
set "CF_TUNNEL="
set "CF_PORT=7860"
if exist "%~dp0cloudflare.local.bat" call "%~dp0cloudflare.local.bat"
set GRADIO_SERVER_PORT=%CF_PORT%

where cloudflared >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] cloudflared introuvable dans le PATH.
    echo Installe-le : winget install --id Cloudflare.cloudflared
    pause
    exit /b 1
)

if defined CF_TUNNEL (
    echo [Cloudflare] Tunnel nomme : %CF_TUNNEL%  ^(route geree par ta config Cloudflare^)
    start "Cloudflare Tunnel" cloudflared tunnel run %CF_TUNNEL%
) else (
    echo [Cloudflare] Quick tunnel ephemere : l'URL https://xxxx.trycloudflare.com
    echo            s'affiche dans la fenetre "Cloudflare Tunnel".
    start "Cloudflare Tunnel" cloudflared tunnel --url http://localhost:%CF_PORT%
)
echo.
echo Acces LAN local :
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do echo    http://%%a:%CF_PORT%
echo.

call "%~dp0run.bat" %*
echo.
echo ----------------------------------------------------
echo  Arrete. Pense a fermer la fenetre du tunnel Cloudflare.
echo ----------------------------------------------------
