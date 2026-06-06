#!/usr/bin/env python3
"""Check whether the Z-Image Omni/Edit models are published on Hugging Face yet.

Exit code 0 if at least one is available, 1 otherwise. Prints a one-line status.
Used by the daily watcher (and runnable by hand). No deps beyond stdlib.
"""
import json
import sys
import urllib.request

REPOS = ["Tongyi-MAI/Z-Image-Omni-Base", "Tongyi-MAI/Z-Image-Edit"]


def repo_exists(repo, timeout=10):
    url = "https://huggingface.co/api/models/" + repo
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "crispz-watcher"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False  # 401/404/timeout -> treat as not available


def main():
    found = [r for r in REPOS if repo_exists(r)]
    if found:
        print("AVAILABLE: " + ", ".join(found)
              + " -> set 'zimage_omni_model' in crispz-studio/config.txt")
        return 0
    print("not yet: " + ", ".join(REPOS) + " (still 'coming soon')")
    return 1


if __name__ == "__main__":
    sys.exit(main())
