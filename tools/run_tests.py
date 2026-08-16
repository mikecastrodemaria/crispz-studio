"""Lance toute la suite tests/test_*.py dans des processus separes et resume.

Processus separes volontairement: les tests manipulent l'etat GLOBAL des modules
(CHECKPOINTS_DIR, FORCE_RATIO, caches...) et se pollueraient l'un l'autre dans un
meme interpreteur. Aucune dependance externe (pas de pytest).

Usage:
    .venv/Scripts/python tools/run_tests.py            # tout
    .venv/Scripts/python tools/run_tests.py xyz quant  # ceux dont le nom matche
    .venv/Scripts/python tools/run_tests.py -v         # sortie complete des echecs
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TESTS = os.path.join(ROOT, "tests")


def main(argv):
    verbose = "-v" in argv or "--verbose" in argv
    filters = [a for a in argv if not a.startswith("-")]
    if not os.path.isdir(TESTS):
        print(f"no tests directory at {TESTS}")
        return 0
    files = sorted(f for f in os.listdir(TESTS)
                   if f.startswith("test_") and f.endswith(".py")
                   and (not filters or any(k.lower() in f.lower() for k in filters)))
    if not files:
        print(f"no test file matches {filters}")
        return 1
    failed, t_all = [], time.time()
    for f in files:
        t0 = time.time()
        # UTF-8 force: les tests impriment des libelles non-ASCII, et une console
        # Windows en cp1252 ferait echouer le print et non le test lui-meme.
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        p = subprocess.run([sys.executable, os.path.join(TESTS, f)],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", cwd=ROOT, env=env)
        ok = p.returncode == 0
        print(f"{'PASS' if ok else 'FAIL'}  {f:<34} {time.time() - t0:5.1f}s")
        if not ok:
            failed.append(f)
            tail = (p.stdout or "") + (p.stderr or "")
            print("\n".join(tail.strip().splitlines()[-(200 if verbose else 12):]))
            print()
    print(f"\n{len(files) - len(failed)}/{len(files)} passed "
          f"in {time.time() - t_all:.1f}s"
          + (f" | FAILED: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
