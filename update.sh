#!/usr/bin/env bash
# Update crispz-studio (Unix): recupere les commits GitHub puis remet les
# dependances en phase avec le lock, SANS casser l'installation existante.
#
#   --force-deps   reinstaller les deps meme si rien n'a change
#   --no-pull      sauter le git pull (resynchroniser les deps seulement)
#   --shared       utiliser requirements.txt au lieu du lock
set -uo pipefail
cd "$(dirname "$0")"

FORCEDEPS=0; DOPULL=1; ISOLATED=1
for a in "$@"; do
  case "$a" in
    --force-deps) FORCEDEPS=1 ;;
    --no-pull)    DOPULL=0 ;;
    --shared)     ISOLATED=0 ;;
  esac
done

echo "=== crispz-studio - update ==="
RUNPY=python3
[ -x ".venv/bin/python" ] && RUNPY=".venv/bin/python"
[ -x "env/bin/python" ] && RUNPY="env/bin/python"
# git-bash / msys2 sous Windows: le venv est en Scripts/, pas bin/. Sans ca on
# tomberait sur le python du shell (souvent sans pip) au lieu de celui du projet.
[ -x ".venv/Scripts/python.exe" ] && RUNPY=".venv/Scripts/python.exe"
echo "Interpreter: $RUNPY"

REQFILE=requirements.txt
[ "$ISOLATED" = "1" ] && [ -f requirements-lock.txt ] && REQFILE=requirements-lock.txt

# 0) etat avant (rollback + detection du remplacement de torch)
TORCH_BEFORE="$("$RUNPY" -c 'import torch;print(torch.__version__)' 2>/dev/null || true)"
SNAP="${TMPDIR:-/tmp}/cz_pip_before.txt"
if [ -n "$TORCH_BEFORE" ]; then
  echo "torch installed: $TORCH_BEFORE"
  "$RUNPY" -m pip freeze > "$SNAP" 2>/dev/null || true
  echo "  (version snapshot: $SNAP)"
else
  echo "torch not installed (first install? run install.sh)."
fi
hash_of() { [ -f "$1" ] && (md5sum "$1" 2>/dev/null || md5 -q "$1" 2>/dev/null) | awk '{print $1}'; }
HASH_BEFORE="$(hash_of "$REQFILE")"
echo

# 1) git pull -- refuse d'ecraser des modifications locales non commitees
if [ "$DOPULL" = "1" ]; then
  if ! command -v git >/dev/null 2>&1; then
    echo "[WARN] git not found -> pull skipped."
  elif ! "$RUNPY" _update_check.py --guard; then
    # Bloque seulement si les commits a recuperer touchent un fichier modifie ici ou
    # ajoutent un fichier deja present hors de git (cf. _update_check.py).
    echo
    echo "  Commit / stash those files first, or run again with --no-pull to"
    echo "  resync the dependencies only."
    exit 1
  else
    echo "Fetching the commits (git pull)..."
    git pull --ff-only || { echo "[ERROR] git pull failed (diverged branch?)."; exit 1; }
  fi
  echo
fi

# 2) deps: seulement si le fichier a change (ou --force-deps)
HASH_AFTER="$(hash_of "$REQFILE")"
NEEDDEPS=0
[ "$HASH_BEFORE" != "$HASH_AFTER" ] && NEEDDEPS=1
[ "$FORCEDEPS" = "1" ] && NEEDDEPS=1
[ -z "$TORCH_BEFORE" ] && NEEDDEPS=1
if [ "$NEEDDEPS" = "0" ]; then
  echo "Dependencies: $REQFILE unchanged -> nothing to reinstall. (--force-deps to force it)"
else
  echo "Dependencies: updating from $REQFILE ..."
  # Pillow est hors du lock (borne pillow<12 de gradio) -> pose a part, sinon un
  # update ferait REGRESSER la version corrigee. Cf. install.sh.
  REQTMP="${TMPDIR:-/tmp}/cz_req_nopillow.txt"
  grep -v '^pillow==' "$REQFILE" > "$REQTMP"
  if "$RUNPY" -m pip install -r "$REQTMP"; then
    "$RUNPY" -m pip install --no-deps --upgrade "pillow==12.3.0" >/dev/null 2>&1 || true
  else
    echo "[ERROR] pip install failed. Possible restore:"
    echo "  $RUNPY -m pip install -r $SNAP"
    exit 1
  fi
fi
echo

# 3) torch a-t-il ete remplace ? (piege: build +cuXXX -> roue CPU)
TORCH_AFTER="$("$RUNPY" -c 'import torch;print(torch.__version__)' 2>/dev/null || true)"
if [ -n "$TORCH_BEFORE" ] && [ "$TORCH_BEFORE" != "$TORCH_AFTER" ]; then
  echo "[WARNING] torch changed: $TORCH_BEFORE -> $TORCH_AFTER"
  echo "   If the +cuXXX suffix is gone, the GPU will not be used any more."
  echo "   Restore: $RUNPY -m pip install torch==$TORCH_BEFORE --index-url https://download.pytorch.org/whl/cu128"
  echo
fi

# 4) verifications
echo "Checking the install..."
"$RUNPY" _hw_check.py; HW=$?
echo
if [ "$HW" = "3" ]; then
  echo "[BLOCKING] torch does not support this card (see the fix above)."
  exit 3
fi
"$RUNPY" -c "from diffusers import ZImagePipeline, ZImageImg2ImgPipeline; print('diffusers: ZImage pipelines OK')" || {
  echo "[ERROR] diffusers no longer ships the ZImage pipelines."
  echo "  Run install.sh again, or restore: $RUNPY -m pip install -r $SNAP"; exit 1; }
"$RUNPY" -c "import cz_ui; print('app: imports OK')" || {
  echo "[ERROR] the application no longer imports."; exit 1; }
echo

# 5) nouvelles cles de config apparues dans le sample
if [ -f config.txt ] && [ -f config-sample.txt ]; then
  "$RUNPY" -c "import json;a=json.load(open('config.txt',encoding='utf-8'));b=json.load(open('config-sample.txt',encoding='utf-8'));n=[k for k in b if k not in a and not k.startswith('_')];print('New config keys available: '+', '.join(n) if n else 'config.txt is up to date.')" 2>/dev/null
  echo "  (config.txt is never overwritten: add the keys you want by hand)"
fi
echo
echo "=== Update OK ==="
[ -f CHANGELOG.md ] && echo "What's new: see CHANGELOG.md"
echo "Run: ./run.sh"
