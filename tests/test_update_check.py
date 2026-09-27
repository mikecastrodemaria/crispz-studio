"""The GitHub update offered at startup: when it is safe, when it blocks.

It works on REAL temporary git repos (a bare repo plays GitHub), with no network:
  - up to date -> nothing to offer;
  - behind, a clean tree -> safe (boot_check.bat offers Y/N);
  - a local change on a file the update touches -> blocked;
  - a local change ELSEWHERE -> safe, and the file is announced as kept;
  - a file present here outside git, which the update ADDS -> blocked (git overwrites
    an ignored file without a word; tests/ is ignored in these repos);
  - a diverged branch -> blocked; no tracked branch / not a repo -> nothing.

Run:  .venv/Scripts/python tests/test_update_check.py

"""
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import _update_check as U

ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
       "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}


def git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, env=ENV, check=True, capture_output=True)


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


class World:
    """origin (a bare repo) + work (the user's clone) + dev (the one that pushes)."""

    def __init__(self):
        self.root = tempfile.mkdtemp(prefix="upd_")
        self.origin = os.path.join(self.root, "origin.git")
        git(self.root, "init", "--bare", "-b", "main", self.origin)
        seed = os.path.join(self.root, "seed")
        git(self.root, "clone", self.origin, seed)
        git(seed, "checkout", "-b", "main")
        write(os.path.join(seed, "a.txt"), "a\n")
        write(os.path.join(seed, "b.txt"), "b\n")
        write(os.path.join(seed, ".gitignore"), "tests/\n")
        git(seed, "add", ".")
        git(seed, "commit", "-m", "init")
        git(seed, "push", "-u", "origin", "main")
        self.work = os.path.join(self.root, "work")
        self.dev = os.path.join(self.root, "dev")
        git(self.root, "clone", self.origin, self.work)
        git(self.root, "clone", self.origin, self.dev)

    def push(self, rel, text, msg, force=False):
        write(os.path.join(self.dev, rel), text)
        git(self.dev, "add", *(["-f"] if force else []), rel)
        git(self.dev, "commit", "-m", msg)
        git(self.dev, "push")

    def assess(self):
        U.ROOT = self.work
        return U.assess(fetch=True)

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)


def test_up_to_date_offers_nothing():
    w = World()
    try:
        assert w.assess()["status"] == "uptodate"
    finally:
        w.close()
    print("OK test_up_to_date_offers_nothing")


def test_behind_and_clean_is_safe():
    w = World()
    try:
        w.push("a.txt", "a2\n", "change a")
        st = w.assess()
        assert st["status"] == "safe" and st["behind"] == 1, st
        assert any("change a" in ln for ln in st["log"]), st
        assert U.main([]) == 10 and U.main(["--guard"]) == 0
    finally:
        w.close()
    print("OK test_behind_and_clean_is_safe")


def test_a_local_change_on_a_touched_file_blocks():
    w = World()
    try:
        w.push("a.txt", "a2\n", "change a")
        write(os.path.join(w.work, "a.txt"), "mon travail\n")
        st = w.assess()
        assert st["status"] == "blocked" and st["overlap"] == ["a.txt"], st
        assert U.main([]) == 11 and U.main(["--guard"]) == 11
    finally:
        w.close()
    print("OK test_a_local_change_on_a_touched_file_blocks")


def test_a_local_change_elsewhere_is_kept():
    """The forks' case: test_queue.py modified here, the update touches something else."""
    w = World()
    try:
        w.push("a.txt", "a2\n", "change a")
        write(os.path.join(w.work, "b.txt"), "mon travail\n")
        st = w.assess()
        assert st["status"] == "safe" and st["local"] == ["b.txt"], st
    finally:
        w.close()
    print("OK test_a_local_change_elsewhere_is_kept")


def test_an_ignored_file_the_update_adds_blocks():
    """tests/ is ignored and the tests are force-added there: git WOULD OVERWRITE the
    local file of the same name without a word."""
    w = World()
    try:
        w.push("tests/test_new.py", "amont\n", "add test", force=True)
        write(os.path.join(w.work, "tests", "test_new.py"), "le mien\n")
        st = w.assess()
        assert st["status"] == "blocked" and st["clobber"] == ["tests/test_new.py"], st
    finally:
        w.close()
    print("OK test_an_ignored_file_the_update_adds_blocks")


def test_an_untracked_folder_elsewhere_does_not_block():
    """wildcards/_backup-*/ exists in the forks: it must not block the update."""
    w = World()
    try:
        w.push("a.txt", "a2\n", "change a")
        write(os.path.join(w.work, "wildcards", "_backup", "x.txt"), "x\n")
        assert w.assess()["status"] == "safe"
    finally:
        w.close()
    print("OK test_an_untracked_folder_elsewhere_does_not_block")


def test_a_diverged_branch_blocks():
    w = World()
    try:
        w.push("a.txt", "a2\n", "change a")
        write(os.path.join(w.work, "b.txt"), "local\n")
        git(w.work, "commit", "-am", "local commit")
        st = w.assess()
        assert st["status"] == "blocked" and "diverged" in st["why"], st
    finally:
        w.close()
    print("OK test_a_diverged_branch_blocks")


def test_no_upstream_or_no_repo_offers_nothing():
    d = tempfile.mkdtemp(prefix="upd_norepo_")
    try:
        U.ROOT = d
        assert U.assess()["status"] == "skip"
        git(d, "init", "-b", "main")
        write(os.path.join(d, "a.txt"), "a\n")
        git(d, "add", ".")
        git(d, "commit", "-m", "x")
        st = U.assess()
        assert st["status"] == "skip" and "no remote branch" in st["why"], st
        assert U.main([]) == 0
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("OK test_no_upstream_or_no_repo_offers_nothing")


def test_the_switch_turns_the_boot_check_off_but_not_the_guard():
    w = World()
    try:
        w.push("a.txt", "a2\n", "change a")
        write(os.path.join(w.work, "a.txt"), "mon travail\n")
        U.ROOT = w.work
        os.environ["CRISPZ_NO_UPDATE_CHECK"] = "1"
        try:
            assert U.main([]) == 0
            assert U.main(["--guard"]) == 11, "la garde d'update.bat doit rester active"
        finally:
            del os.environ["CRISPZ_NO_UPDATE_CHECK"]
    finally:
        w.close()
    print("OK test_the_switch_turns_the_boot_check_off_but_not_the_guard")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("All update-check tests passed.")
