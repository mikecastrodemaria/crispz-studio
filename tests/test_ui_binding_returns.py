"""A handler wired on outputs=None must return nothing.

Gradio 5 warns on EVERY firing when a returned value has no
component to receive it ("A function returned too many output values"). On a
slider, that means a pair of warnings per movement -- noise that
ends up hiding a real warning.

The test re-reads cz_ui's wirings and checks that no handler wired on
outputs=None holds a `return <value>`.

It lives in the five forks: it was klein-only, which is precisely how two
.then(set_sampler, [sampler_dd], None) chains survived in studio, krea and krea2
until 2026-09-28.

Run:  .venv/Scripts/python tests/test_ui_binding_returns.py

"""
import inspect
import io
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_ui

# .then and .submit count too: two .then(set_sampler, [sampler_dd], None) chains
# warned on every "Apply CivitAI recommended settings" click in studio, krea and
# krea2, and this pattern did not look at .then, so it saw nothing.
BIND = re.compile(r"\.(?:change|click|input|release|select|then|submit)\("
                  r"\s*([A-Za-z_][\w.]*)\s*,"
                  r"\s*\[[^\]]*\]\s*,\s*None\s*\)")


def test_no_handler_returns_into_the_void():
    src = io.open(os.path.join(os.path.dirname(cz_ui.__file__), "cz_ui.py"),
                  encoding="utf-8").read()
    names = sorted(set(BIND.findall(src)))
    assert names, "no outputs=None wiring found: has the pattern changed?"
    offenders = []
    for n in names:
        obj = cz_ui
        try:
            for part in n.split("."):
                obj = getattr(obj, part)
            body = inspect.getsource(obj).split(":", 1)[1]
        except Exception:
            continue
        if re.search(r"^\s+return\s+\S", body, re.M):
            offenders.append(n)
    assert not offenders, (
        f"wired on outputs=None but return a value -> Gradio warns on every "
        f"firing: {offenders}")
    print(f"OK test_no_handler_returns_into_the_void ({len(names)} wirings checked)")


if __name__ == "__main__":
    test_no_handler_returns_into_the_void()
    print("ALL OK")
