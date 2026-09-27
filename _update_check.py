"""A GitHub update offered at startup (boot_check.bat), and the safety guard of
update.bat / update.sh. The standard library only: this script runs BEFORE the app
imports, and must work even when its dependencies are broken.

  python _update_check.py           boot: looks, displays, says whether it is safe
  python _update_check.py --guard   update: blocks when the pull would touch local work

Exit codes:
  0   nothing to offer: up to date, offline, no git, no tracked branch, disabled
      (CRISPZ_NO_UPDATE_CHECK=1)
  10  an update is available AND safe: boot_check.bat then offers Y/N
  11  an update is available but blocked: a diverged branch, or it would touch a file
      modified here, or overwrite a file present here outside git
In --guard mode: 0 = `git pull --ff-only` may run, 11 = blocked.

The safety rule is git's own, only stricter: a fast-forward pull keeps the local changes
of the files it does not touch. It refuses to touch a file modified here -- and it
OVERWRITES without a word an IGNORED file that the repo adds (tests/ is ignored in these
repos, and the tests are force-added there). We block both.

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
    """(code, output) of a git command in the repo; (None, a reason) when git is missing or
    when the command runs past its timeout (a hanging network must not block the boot)."""
    try:
        p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        return p.returncode, ((p.stdout or "") + (p.stderr or "" if p.returncode else "")).strip()
    except FileNotFoundError:
        return None, "git not found"
    except subprocess.TimeoutExpired:
        return None, f"no answer in {timeout} s"


def _lines(out):
    return [ln.strip() for ln in (out or "").splitlines() if ln.strip()]


def assess(fetch=True):
    """The state of the update, as a dict. status: 'skip', 'uptodate', 'safe' or
    'blocked'. Kept apart from main() for the tests."""
    code, out = _git("rev-parse", "--is-inside-work-tree")
    if code is None:
        return {"status": "skip", "why": out}
    if code != 0 or out != "true":
        return {"status": "skip", "why": "this folder is not a git repository"}
    code, up = _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    if code != 0 or not up:
        return {"status": "skip", "why": "the current branch tracks no remote branch"}
    if fetch:
        code, out = _git("fetch", "--quiet", up.split("/", 1)[0], timeout=FETCH_TIMEOUT)
        if code != 0:
            last = _lines(out)[-1][:90] if _lines(out) else "fetch failed"
            return {"status": "skip", "why": f"GitHub unreachable ({last})"}
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
                "why": f"diverged branch: {ahead} local commit(s) missing from GitHub"}
    # What the commits to fetch touch (a rename counts as a deletion + an addition).
    _, changed = _git("diff", "--name-only", "--no-renames", "HEAD", "@{u}")
    _, added = _git("diff", "--name-only", "--no-renames", "--diff-filter=A", "HEAD", "@{u}")
    changed, added = set(_lines(changed)), set(_lines(added))
    # Local work: tracked files modified, staged or not.
    _, local = _git("diff", "--name-only", "HEAD")
    local = set(_lines(local))
    overlap = sorted(changed & local)
    clobber = sorted(p for p in added if os.path.lexists(os.path.join(ROOT, p)))
    info["local"] = sorted(local)
    if overlap or clobber:
        return {**info, "status": "blocked", "overlap": overlap, "clobber": clobber,
                "why": "the update would touch local work"}
    return {**info, "status": "safe"}


def _plural(n, word):
    return f"{n} {word}{'s' if n > 1 else ''}"


def main(argv):
    try:
        sys.stdout.reconfigure(errors="replace")      # the cmd console: no guaranteed UTF-8
    except Exception:
        pass
    guard = "--guard" in argv
    if not guard and os.environ.get("CRISPZ_NO_UPDATE_CHECK", "") == "1":
        print("    Check disabled (CRISPZ_NO_UPDATE_CHECK=1).")
        return 0
    st = assess(fetch=True)
    s = st["status"]
    if s == "skip":
        print(f"    No check: {st['why']}.")
        return 0
    if s == "uptodate":
        extra = f" ({_plural(st['ahead'], 'commit')} local, not pushed)" if st.get("ahead") else ""
        print(f"    Up to date{extra}.")
        return 0
    n = st["behind"]
    print(f"    {_plural(n, 'new commit')} on GitHub ({st['upstream']}):")
    for ln in st["log"]:
        print(f"      {ln[:100]}")
    if n > len(st["log"]):
        print(f"      ... and {n - len(st['log'])} more")
    if s == "blocked":
        print(f"    [BLOCKED] {st['why']}.")
        for p in st.get("overlap", []):
            print(f"      changed here AND by the update: {p}")
        for p in st.get("clobber", []):
            print(f"      here outside git, added by the update: {p}")
        print("    Nothing was touched.")
        return 11
    kept = st.get("local") or []
    if kept:
        print(f"    {_plural(len(kept), 'file')} changed here that the update does not "
              f"touch: kept as they are.")
    return 0 if guard else 10


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
