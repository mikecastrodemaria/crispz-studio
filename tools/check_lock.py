"""Checks that every package declared in requirements.txt, requirements-extra.txt and
requirements-faceswap.txt is also pinned in requirements-lock.txt.

WHY. requirements.txt gives the bounds, requirements-lock.txt gives the exact versions of
a validated environment -- but the lock is written by hand, so it drifts. A package added
on one side and forgotten on the other installs fine through install.bat (which reads
requirements.txt) and is simply ABSENT from an isolated venv built from the lock. The
failure is silent and late: on 2026-09-28 the five forks were each missing some of
trustmark, c2pa-python, ultralytics, onnx, gguf and hf_xet that way, several of them under
a comment inherited from another fork claiming this one had no use for the package.
Symptoms: "check unavailable" in the Provenance section, the hands pass permanently
unavailable, a .gguf refused at load time, a 17 GB download falling back to plain HTTP.

The check is deliberately ONE-WAY. The lock legitimately holds packages no requirements
file names (torch, safetensors, scipy, opencv-python...): they are transitive deps pinned
on purpose. Listing those would be noise.

It also checks that every lock entry really is pinned: a bare name in a lock file is a
version nobody chose.

No dependency (a regex + the standard library).

Usage:  python tools/check_lock.py        (exit code 1 on a missing or unpinned package)

"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK = "requirements-lock.txt"
SOURCES = ("requirements.txt", "requirements-extra.txt", "requirements-faceswap.txt")

# pip's own rule: a '#' only opens a comment at the start of a line or after a blank.
# That leaves the '#egg=' / '#subdirectory=' fragments of a URL alone.
_COMMENT = re.compile(r"(^|\s)#.*$")
# a name at the start of a line: pip stops it at the first character that is neither a
# letter, a digit, '.', '-' nor '_' (so at '[', '=', '<', '>', ';', ' ').
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")
# 'name @ git+https://...' (the PEP 508 direct reference the lock uses for diffusers).
_DIRECT = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*@\s*(\S+)")
# the bare 'git+https://host/org/repo@commit' form requirements.txt uses: no name is
# given, the repo IS the package.
_GIT_URL = re.compile(r"/([A-Za-z0-9._-]+?)(?:\.git)?(?:@[^/]*)?$")


def _norm(name):
    """PEP 503: hf_xet, hf-xet and HF.Xet all name the same package."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def _parse(path):
    """{normalised name: (name as written, is it pinned?)} for one requirements file."""
    found = {}
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = _COMMENT.sub("", raw).strip()
            if not line or line.startswith("-"):
                continue  # blank, or an option (--extra-index-url, -r, -e...)
            line = line.split(";", 1)[0].strip()  # drop the environment marker
            m = _DIRECT.match(line)
            if m:  # the name is authoritative, the URL carries the pin
                found[_norm(m.group(1))] = (m.group(1), "@" in m.group(2))
            elif line.startswith("git+"):
                m = _GIT_URL.search(line)
                if m:
                    found[_norm(m.group(1))] = (m.group(1), "@" in line.rsplit("/", 1)[-1])
            else:
                m = _NAME.match(line)
                if m:
                    found[_norm(m.group(0))] = (m.group(0), "==" in line)
    return found


def main():
    lock = _parse(os.path.join(ROOT, LOCK))

    missing = []
    for src in SOURCES:
        path = os.path.join(ROOT, src)
        if not os.path.exists(path):
            continue
        for key, (name, _) in sorted(_parse(path).items()):
            if key not in lock:
                missing.append((name, src))

    unpinned = sorted(name for name, pinned in lock.values() if not pinned)

    for name, src in missing:
        print("MISSING from %s: %s  (declared in %s)" % (LOCK, name, src))
    for name in unpinned:
        print("NOT PINNED in %s: %s  (a lock entry needs == or a commit)" % (LOCK, name))

    if missing or unpinned:
        print("\n%d package(s) missing from the lock, %d not pinned"
              % (len(missing), len(unpinned)))
        return 1
    print("lock OK (%d package(s) pinned, every declared dependency covered)" % len(lock))
    return 0


if __name__ == "__main__":
    sys.exit(main())
