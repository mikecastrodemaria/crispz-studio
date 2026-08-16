"""Verifie que toute cle lue via CONFIG.get("...") dans le code existe dans
config-sample.txt (et inversement, signale les cles documentees mais jamais lues).

Sans ca, une option ajoutee au code reste invisible pour l'utilisateur: elle n'est
ni dans le fichier d'exemple ni dans le tutoriel. Aucune dependance (regex + json).

Usage:  python tools/check_config_keys.py        (code de sortie 1 si manquantes)
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "config-sample.txt")

# CONFIG.get("cle"...) et CONFIG.get('cle'...) au premier niveau seulement: les
# sous-blocs (asset_browser.*, xyz_grid.*) sont lus via un dict intermediaire.
_RE = re.compile(r"""CONFIG\.get\(\s*["']([A-Za-z0-9_]+)["']""")


def main():
    with open(SAMPLE, encoding="utf-8") as f:
        sample = json.load(f)
    documented = {k for k in sample if not k.startswith("_")}
    # les cles d'aide '_x_help' documentent la cle 'x'
    helped = {k[1:-5] for k in sample if k.startswith("_") and k.endswith("_help")}

    used = {}
    for f in sorted(os.listdir(ROOT)):
        if not (f.startswith("cz_") or f == "app.py") or not f.endswith(".py"):
            continue
        with open(os.path.join(ROOT, f), encoding="utf-8") as fh:
            for key in _RE.findall(fh.read()):
                used.setdefault(key, set()).add(f)

    missing = sorted(k for k in used if k not in documented)
    unused = sorted(k for k in documented if k not in used)
    no_help = sorted(k for k in used if k in documented and k not in helped)

    for k in missing:
        print(f"MISSING from config-sample.txt: {k}  (read in {', '.join(sorted(used[k]))})")
    if unused:
        print(f"note: documented but never read: {', '.join(unused)}")
    if no_help:
        print(f"note: no '_<key>_help' entry for: {', '.join(no_help)}")
    if missing:
        print(f"\n{len(missing)} config key(s) missing from config-sample.txt")
        return 1
    print(f"config keys OK ({len(used)} read, all documented)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
