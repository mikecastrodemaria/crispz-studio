"""Mise a jour GitHub proposee au demarrage (boot_check.bat), et garde de securite
d'update.bat / update.sh. Bibliotheque standard seulement: ce script tourne AVANT que
l'app ne s'importe, et doit marcher meme quand ses dependances sont cassees.

  python _update_check.py           boot: cherche, affiche, dit si c'est sur
  python _update_check.py --guard   update: bloque si le pull toucherait du travail local

Codes de sortie:
  0   rien a proposer: a jour, hors ligne, pas de git, pas de branche suivie, desactive
      (CRISPZ_NO_UPDATE_CHECK=1)
  10  mise a jour disponible ET sure: boot_check.bat propose alors O/N
  11  mise a jour disponible mais bloquee: branche divergente, ou elle toucherait un
      fichier modifie ici, ou ecraserait un fichier present ici hors de git
En mode --guard: 0 = `git pull --ff-only` peut tourner, 11 = bloque.

La regle de securite est celle de git, en plus strict: un pull en avance rapide conserve
les modifications locales des fichiers qu'il ne touche pas. Il refuse de toucher un fichier
modifie ici -- et il ECRASE sans rien dire un fichier IGNORE que le depot ajoute (tests/
est ignore dans ces depots, et les tests y sont ajoutes de force). On bloque les deux.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
try:
    FETCH_TIMEOUT = int(os.environ.get("CRISPZ_UPDATE_TIMEOUT", "20") or 20)
except ValueError:
    FETCH_TIMEOUT = 20
SHOW = 8                                  # commits listes au plus


def _git(*args, timeout=15):
    """(code, sortie) d'une commande git dans le depot; (None, raison) si git manque ou
    si la commande depasse son delai (un reseau qui pend ne doit pas bloquer le boot)."""
    try:
        p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        return p.returncode, ((p.stdout or "") + (p.stderr or "" if p.returncode else "")).strip()
    except FileNotFoundError:
        return None, "git introuvable"
    except subprocess.TimeoutExpired:
        return None, f"pas de reponse en {timeout} s"


def _lines(out):
    return [ln.strip() for ln in (out or "").splitlines() if ln.strip()]


def assess(fetch=True):
    """Etat de la mise a jour, sous forme de dict. status: 'skip', 'uptodate', 'safe' ou
    'blocked'. Separe de main() pour les tests."""
    code, out = _git("rev-parse", "--is-inside-work-tree")
    if code is None:
        return {"status": "skip", "why": out}
    if code != 0 or out != "true":
        return {"status": "skip", "why": "ce dossier n'est pas un depot git"}
    code, up = _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    if code != 0 or not up:
        return {"status": "skip", "why": "la branche courante ne suit aucune branche distante"}
    if fetch:
        code, out = _git("fetch", "--quiet", up.split("/", 1)[0], timeout=FETCH_TIMEOUT)
        if code != 0:
            last = _lines(out)[-1][:90] if _lines(out) else "fetch en echec"
            return {"status": "skip", "why": f"GitHub injoignable ({last})"}
    _, behind = _git("rev-list", "--count", "HEAD..@{u}")
    _, ahead = _git("rev-list", "--count", "@{u}..HEAD")
    behind = int(behind) if str(behind).isdigit() else 0
    ahead = int(ahead) if str(ahead).isdigit() else 0
    if behind == 0:
        return {"status": "uptodate", "upstream": up, "ahead": ahead}
    _, log = _git("log", "--oneline", "--no-decorate", f"-{SHOW}", "HEAD..@{u}")
    info = {"upstream": up, "behind": behind, "ahead": ahead, "log": _lines(log)}
    if ahead:
        return {**info, "status": "blocked",
                "why": f"branche divergente: {ahead} commit(s) local(aux) absent(s) de GitHub"}
    # Ce que les commits a recuperer touchent (un renommage compte comme suppression + ajout).
    _, changed = _git("diff", "--name-only", "--no-renames", "HEAD", "@{u}")
    _, added = _git("diff", "--name-only", "--no-renames", "--diff-filter=A", "HEAD", "@{u}")
    changed, added = set(_lines(changed)), set(_lines(added))
    # Travail local: fichiers suivis modifies, indexes ou non.
    _, local = _git("diff", "--name-only", "HEAD")
    local = set(_lines(local))
    overlap = sorted(changed & local)
    clobber = sorted(p for p in added if os.path.lexists(os.path.join(ROOT, p)))
    info["local"] = sorted(local)
    if overlap or clobber:
        return {**info, "status": "blocked", "overlap": overlap, "clobber": clobber,
                "why": "la mise a jour toucherait du travail local"}
    return {**info, "status": "safe"}


def _plural(n, word):
    return f"{n} {word}{'s' if n > 1 else ''}"


def main(argv):
    try:
        sys.stdout.reconfigure(errors="replace")      # console cmd: pas d'UTF-8 garanti
    except Exception:
        pass
    guard = "--guard" in argv
    if not guard and os.environ.get("CRISPZ_NO_UPDATE_CHECK", "") == "1":
        print("    Verification desactivee (CRISPZ_NO_UPDATE_CHECK=1).")
        return 0
    st = assess(fetch=True)
    s = st["status"]
    if s == "skip":
        print(f"    Pas de verification: {st['why']}.")
        return 0
    if s == "uptodate":
        extra = f" ({_plural(st['ahead'], 'commit')} local non pousse)" if st.get("ahead") else ""
        print(f"    A jour{extra}.")
        return 0
    n = st["behind"]
    print(f"    {_plural(n, 'nouveau commit').replace('nouveau commits', 'nouveaux commits')} "
          f"sur GitHub ({st['upstream']}):")
    for ln in st["log"]:
        print(f"      {ln[:100]}")
    if n > len(st["log"]):
        print(f"      ... et {n - len(st['log'])} autre(s)")
    if s == "blocked":
        print(f"    [BLOQUE] {st['why']}.")
        for p in st.get("overlap", []):
            print(f"      modifie ici ET par la mise a jour : {p}")
        for p in st.get("clobber", []):
            print(f"      present ici hors de git, ajoute par la mise a jour : {p}")
        print("    Rien n'a ete touche.")
        return 11
    kept = st.get("local") or []
    if kept:
        print(f"    {_plural(len(kept), 'fichier')} modifie(s) ici, que la mise a jour ne "
              f"touche pas: conserve(s) tel(s) quel(s).")
    return 0 if guard else 10


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
