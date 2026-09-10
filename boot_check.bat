@echo off
REM Boot check generique crispz-studio (remplace les anciens scripts rtx5090).
REM
REM Diagnostique la machine AVANT de lancer l'app, quelle que soit la carte
REM (RTX 50xx / 40xx / 30xx / 20xx...), et s'arrete net si la configuration ne
REM peut pas fonctionner -- plutot que de laisser l'app planter en cours de route.
REM
REM Le check decisif est fait par _hw_check.py: il compare le sm_XX du GPU a la
REM liste d'architectures compilees dans le build torch installe. C'est ce qui
REM detecte le cas "RTX 50xx + torch non-cu128" (WinError 127 torch_cuda.dll).
REM
REM   --no-run   diagnostiquer seulement, ne pas lancer l'app
REM   --no-update  ne pas chercher de mise a jour GitHub (ou CRISPZ_NO_UPDATE_CHECK=1)
REM   --lan      ecouter sur le LAN (0.0.0.0) au lieu de 127.0.0.1
REM   --web      LAN + tunnel Cloudflare (URL publique)
REM   tout autre argument est transmis a run.bat
REM
REM ATTENTION --lan / --web: l'app n'a AUCUNE authentification et sert le dossier
REM de sortie + les dossiers de modeles. Voir "Scope" dans SECURITY.md.

setlocal enabledelayedexpansion
title crispz-studio - Boot Check
cd /d "%~dp0"

set "NORUN=0"
set "NOUPDATE=0"
set "EXPOSE="
set "PASSTHRU="
:argloop
if "%~1"=="" goto argdone
if /I "%~1"=="--no-run" (
    set "NORUN=1"
) else if /I "%~1"=="--no-update" (
    set "NOUPDATE=1"
) else if /I "%~1"=="--lan" (
    set "EXPOSE=lan"
) else if /I "%~1"=="--web" (
    set "EXPOSE=web"
) else (
    set "PASSTHRU=!PASSTHRU! %~1"
)
shift
goto argloop
:argdone

echo ====================================================
echo    crispz-studio - Boot Check
echo ====================================================
echo.

REM --- Interpreteur (venv prioritaire) ---
set "RUNPY="
if exist ".venv\Scripts\python.exe" set "RUNPY=.venv\Scripts\python.exe"
if not defined RUNPY (
    where py >nul 2>&1 && ( set "RUNPY=py -3.10" ) || ( set "RUNPY=python" )
)

echo [1/5] Python : !RUNPY!
!RUNPY! --version 2>nul
if errorlevel 1 (
    echo    [ERREUR] Python introuvable. Installe Python 3.10+ puis lance install.bat.
    pause & exit /b 1
)
echo.

REM --- Mise a jour GitHub: PROPOSEE, jamais imposee (voir _update_check.py) ---
REM Proposee seulement si elle est sure: aucun des commits a recuperer ne touche un
REM fichier modifie ici ni n'ajoute un fichier deja present hors de git. Sans reponse
REM en 20 s: N, l'app demarre telle quelle. --no-update ou CRISPZ_NO_UPDATE_CHECK=1:
REM etape sautee. Hors ligne, sans git ou sans branche suivie: elle le dit et passe.
if "!NOUPDATE!"=="0" (
    echo [MAJ] Mises a jour GitHub...
    !RUNPY! _update_check.py
    set "UPD=!errorlevel!"
    if "!UPD!"=="10" (
        choice /C ON /T 20 /D N /M "    Mettre a jour maintenant ? N par defaut dans 20 s"
        if errorlevel 2 (
            echo    Demarrage sans mise a jour. Plus tard : update.bat
        ) else (
            call "%~dp0update.bat"
            if errorlevel 1 (
                echo    [ERREUR] Mise a jour interrompue, voir ci-dessus. L'app n'est pas lancee.
                pause & exit /b 1
            )
            echo    Mise a jour faite. Suite des verifications...
        )
    )
    if "!UPD!"=="11" echo    Demarrage sans mise a jour.
    echo.
)

REM --- 2. Etat du driver / de la carte (informations brutes) ---
echo [2/5] Driver NVIDIA...
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu --format=csv,noheader,nounits > "%TEMP%\cz_gpu.txt" 2>nul
if errorlevel 1 (
    echo    [INFO] nvidia-smi introuvable ^(pas de GPU NVIDIA, ou drivers absents^).
) else (
    for /f "tokens=1,2,3,4,5 delims=," %%a in (%TEMP%\cz_gpu.txt) do (
        echo    Carte   : %%a
        echo    Driver  : %%b
        echo    VRAM    : %%d / %%c MB utilises   ^| Temp: %%e C
    )
    del "%TEMP%\cz_gpu.txt" >nul 2>&1
)
echo.

REM --- 3. LE check: torch supporte-t-il CETTE carte ? + recommandations ---
echo [3/5] PyTorch / GPU / reglages conseilles...
echo.
!RUNPY! _hw_check.py
set "HW=!errorlevel!"
echo.
if "!HW!"=="1" (
    echo    [ERREUR] PyTorch absent -^> lance install.bat.
    pause & exit /b 1
)
if "!HW!"=="3" (
    echo    [BLOQUANT] torch ne supporte pas cette carte ^(voir le correctif ci-dessus^).
    echo    L'app planterait a la premiere allocation CUDA. Arret.
    pause & exit /b 3
)
if "!HW!"=="2" echo    [AVERT] Mode CPU: la generation sera tres lente.

REM --- 4. Pipeline diffusers de cette famille de modele ---
echo [4/5] diffusers...
!RUNPY! -c "from diffusers import ZImagePipeline, ZImageImg2ImgPipeline; print('    ZImage pipelines OK')" 2>nul
if errorlevel 1 echo    [ATTENTION] ZImage pipelines indisponibles -^> lance install.bat / update.bat.
echo.

REM --- 5. Modeles: on lit les VRAIS dossiers de la config, pas un chemin en dur ---
echo [5/5] Modeles...
REM %% : en batch un %% litteral s'ecrit double, sinon cmd mange le format Python.
!RUNPY! -c "import os,cz_pipeline as p;[print('    %%-11s %%3d fichier(s)  %%s' %% (n, (len([f for f in os.listdir(d) if f.lower().endswith(('.safetensors','.gguf','.ckpt','.pt','.sft'))]) if os.path.isdir(d) else 0), d if os.path.isdir(d) else '(dossier absent)')) for n,d in (('checkpoints',p.CHECKPOINTS_DIR),('extra',p.CHECKPOINTS_EXTRA_DIR),('loras',p.LORAS_DIR)) if d]" 2>nul
if errorlevel 1 echo    [INFO] impossible de lire la config ^(config.txt absent ? lance install.bat^).
echo.

REM --- Optimisations CUDA (sans effet si pas de GPU NVIDIA) ---
set NVIDIA_TF32_OVERRIDE=1
set CUDA_CACHE_MAXSIZE=4294967296
set CUDA_AUTO_BOOST=1
set CUDA_DEVICE_ORDER=PCI_BUS_ID
REM Port fixe (heritage des anciens run_quality_*.bat): evite que Gradio parte
REM sur 7861+ quand une instance precedente n'a pas encore libere le port.
if not defined GRADIO_SERVER_PORT set GRADIO_SERVER_PORT=7860

REM --- Exposition reseau (--lan / --web): Gradio lit ces variables nativement ---
set "CF_PORT=7860"
if defined EXPOSE (
    echo ----------------------------------------------------
    echo  [SECURITE] Exposition reseau demandee ^(--!EXPOSE!^).
    echo  crispz-studio n'a AUCUNE authentification et sert ton dossier de
    echo  sortie ainsi que tes dossiers de modeles. N'expose que sur un reseau
    echo  de confiance. Voir la section "Scope" de SECURITY.md.
    echo ----------------------------------------------------
    set GRADIO_SERVER_NAME=0.0.0.0
    set GRADIO_SERVER_PORT=!CF_PORT!
    echo Acces LAN :
    for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do echo    http://%%a:!CF_PORT!
    echo.
)
if /I "!EXPOSE!"=="web" (
    REM Config perso NON versionnee (cf. cloudflare.local.bat.example):
    REM   CF_TUNNEL = tunnel cloudflared nomme, sinon quick tunnel ephemere.
    set "CF_TUNNEL="
    if exist "%~dp0cloudflare.local.bat" call "%~dp0cloudflare.local.bat"
    if defined CF_PORT set GRADIO_SERVER_PORT=!CF_PORT!
    where cloudflared >nul 2>&1
    if errorlevel 1 (
        echo [ERREUR] cloudflared introuvable dans le PATH.
        echo    Installe-le : winget install --id Cloudflare.cloudflared
        pause & exit /b 1
    )
    if defined CF_TUNNEL (
        echo [Cloudflare] Tunnel nomme : !CF_TUNNEL!
        start "Cloudflare Tunnel" cloudflared tunnel run !CF_TUNNEL!
    ) else (
        echo [Cloudflare] Quick tunnel ephemere: l'URL https://xxxx.trycloudflare.com
        echo              s'affiche dans la fenetre "Cloudflare Tunnel".
        start "Cloudflare Tunnel" cloudflared tunnel --url http://localhost:!CF_PORT!
    )
    echo.
)

if "!NORUN!"=="1" (
    echo ====================================================
    echo    Diagnostic termine ^(--no-run: app non lancee^).
    echo ====================================================
    endlocal & exit /b 0
)
echo ====================================================
echo    Checks OK. Lancement de crispz-studio...
echo ====================================================
timeout /t 2 /nobreak >nul
call "%~dp0run.bat" %PASSTHRU%
if /I "!EXPOSE!"=="web" (
    echo.
    echo ----------------------------------------------------
    echo  Arrete. Pense a fermer la fenetre du tunnel Cloudflare.
    echo ----------------------------------------------------
)
endlocal
