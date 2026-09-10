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
echo "Interpreteur: $RUNPY"

REQFILE=requirements.txt
[ "$ISOLATED" = "1" ] && [ -f requirements-lock.txt ] && REQFILE=requirements-lock.txt

# 0) etat avant (rollback + detection du remplacement de torch)
TORCH_BEFORE="$("$RUNPY" -c 'import torch;print(torch.__version__)' 2>/dev/null || true)"
SNAP="${TMPDIR:-/tmp}/cz_pip_before.txt"
if [ -n "$TORCH_BEFORE" ]; then
  echo "torch installe: $TORCH_BEFORE"
  "$RUNPY" -m pip freeze > "$SNAP" 2>/dev/null || true
  echo "  (snapshot des versions: $SNAP)"
else
  echo "torch non installe (premiere install ? lance install.sh)."
fi
hash_of() { [ -f "$1" ] && (md5sum "$1" 2>/dev/null || md5 -q "$1" 2>/dev/null) | awk '{print $1}'; }
HASH_BEFORE="$(hash_of "$REQFILE")"
echo

# 1) git pull -- refuse d'ecraser des modifications locales non commitees
if [ "$DOPULL" = "1" ]; then
  if ! command -v git >/dev/null 2>&1; then
    echo "[AVERT] git introuvable -> pull saute."
  elif ! "$RUNPY" _update_check.py --guard; then
    # Bloque seulement si les commits a recuperer touchent un fichier modifie ici ou
    # ajoutent un fichier deja present hors de git (cf. _update_check.py).
    echo
    echo "  Commit / stash ces fichiers d'abord, ou relance avec --no-pull pour ne"
    echo "  resynchroniser que les dependances."
    exit 1
  else
    echo "Recuperation des commits (git pull)..."
    git pull --ff-only || { echo "[ERREUR] git pull a echoue (branche divergente ?)."; exit 1; }
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
  echo "Dependances: $REQFILE inchange -> rien a reinstaller. (--force-deps pour forcer)"
else
  echo "Dependances: mise a jour depuis $REQFILE ..."
  # Pillow est hors du lock (borne pillow<12 de gradio) -> pose a part, sinon un
  # update ferait REGRESSER la version corrigee. Cf. install.sh.
  REQTMP="${TMPDIR:-/tmp}/cz_req_nopillow.txt"
  grep -v '^pillow==' "$REQFILE" > "$REQTMP"
  if "$RUNPY" -m pip install -r "$REQTMP"; then
    "$RUNPY" -m pip install --no-deps --upgrade "pillow==12.3.0" >/dev/null 2>&1 || true
  else
    echo "[ERREUR] pip install a echoue. Restauration possible:"
    echo "  $RUNPY -m pip install -r $SNAP"
    exit 1
  fi
fi
echo

# 3) torch a-t-il ete remplace ? (piege: build +cuXXX -> roue CPU)
TORCH_AFTER="$("$RUNPY" -c 'import torch;print(torch.__version__)' 2>/dev/null || true)"
if [ -n "$TORCH_BEFORE" ] && [ "$TORCH_BEFORE" != "$TORCH_AFTER" ]; then
  echo "[ATTENTION] torch a change: $TORCH_BEFORE -> $TORCH_AFTER"
  echo "   Si le suffixe +cuXXX a disparu, le GPU ne sera plus utilise."
  echo "   Restauration: $RUNPY -m pip install torch==$TORCH_BEFORE --index-url https://download.pytorch.org/whl/cu128"
  echo
fi

# 4) verifications
echo "Verification de l'installation..."
"$RUNPY" _hw_check.py; HW=$?
echo
if [ "$HW" = "3" ]; then
  echo "[BLOQUANT] torch ne supporte pas cette carte (voir le correctif ci-dessus)."
  exit 3
fi
"$RUNPY" -c "from diffusers import ZImagePipeline, ZImageImg2ImgPipeline; print('diffusers: ZImage pipelines OK')" || {
  echo "[ERREUR] diffusers ne fournit plus les pipelines ZImage."
  echo "  Relance install.sh, ou restaure: $RUNPY -m pip install -r $SNAP"; exit 1; }
"$RUNPY" -c "import cz_ui; print('app: imports OK')" || {
  echo "[ERREUR] l'application ne s'importe plus."; exit 1; }
echo

# 5) nouvelles cles de config apparues dans le sample
if [ -f config.txt ] && [ -f config-sample.txt ]; then
  "$RUNPY" -c "import json;a=json.load(open('config.txt',encoding='utf-8'));b=json.load(open('config-sample.txt',encoding='utf-8'));n=[k for k in b if k not in a and not k.startswith('_')];print('Nouvelles cles de config disponibles: '+', '.join(n) if n else 'config.txt a jour.')" 2>/dev/null
  echo "  (config.txt n'est jamais ecrase: ajoute les cles voulues a la main)"
fi
echo
echo "=== Update OK ==="
[ -f CHANGELOG.md ] && echo "Nouveautes: voir CHANGELOG.md"
echo "Lance: ./run.sh"
