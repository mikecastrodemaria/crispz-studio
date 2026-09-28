"""A handler wired on no outputs must return nothing.

Gradio 5 warns on EVERY firing when a returned value has no component to receive it
("A function returned too many output values"). On a slider that is a pair of warnings per
movement -- noise that ends up hiding a real warning.

This reads cz_ui.py's wirings from the AST, not with a regex. The regex version missed four
forms Gradio treats exactly like `outputs=None`:
  - `outputs=None` passed as a KEYWORD;
  - `outputs=[]`, an empty list;
  - the outputs argument simply OMITTED (it defaults to None);
  - anything that is not `.change/.click/.input/.release/.select` -- two
    `.then(set_sampler, [sampler_dd], None)` chains warned on every "Apply CivitAI
    recommended settings" click in three forks and the pattern did not look at .then.
And it could not follow a LAMBDA nor a handler living in another module
(cz_detailer.set_enabled), which together are most of the wirings here.

A detector nobody checks is worse than none, so this file TESTS ITS OWN DETECTOR first,
against a sample holding known-bad and known-good wirings, and only then applies it to
cz_ui.py.

Run:  .venv/Scripts/python tests/test_ui_binding_returns.py

"""
import ast
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_ui  # noqa: E402

# Every Gradio event that takes (fn, inputs, outputs).
EVENTS = {"change", "click", "input", "release", "select", "then", "submit",
          "upload", "clear", "blur", "focus", "stop", "edit", "like"}


def _is_none(node):
    return isinstance(node, ast.Constant) and node.value is None


def _defs_of(src):
    """{name: node} for every function defined in `src`, nested ones included."""
    out = {}
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.setdefault(n.name, n)
    return out


def _dotted(node):
    """`f` -> "f", `mod.f` -> "mod.f", anything else -> None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _returns_value(node, resolve):
    """Does calling `node` hand a value back to Gradio?

    A def: it holds a `return <something not None>`.
    A lambda: its body is the value -- but `lambda m: setter(m)` returns whatever `setter`
    returns, so the call is followed. Without that, every such wrapper looks guilty and the
    detector cries wolf (it did, on krea2's quant_dd).
    """
    if isinstance(node, ast.Lambda):
        if _is_none(node.body):
            return False
        if isinstance(node.body, ast.Call):
            inner = resolve(_dotted(node.body.func))
            return True if inner is None else _returns_value(inner, resolve)
        return True
    for n in ast.walk(node):
        if isinstance(n, ast.Return) and n.value is not None and not _is_none(n.value):
            return True
    return False


def find_offenders(src, resolve):
    """[(line, handler, how)] for the wirings that return into the void, + how many wirings
    with no outputs were examined."""
    offenders, examined = [], 0
    for n in ast.walk(ast.parse(src)):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr in EVENTS and n.args):
            continue
        kw = {k.arg: k.value for k in n.keywords}
        if "outputs" in kw:
            outs, how = kw["outputs"], "outputs=None"
        elif len(n.args) >= 3:
            outs, how = n.args[2], "outputs=None"
        else:
            outs, how = None, "outputs omitted"
        if isinstance(outs, (ast.List, ast.Tuple)) and not outs.elts:
            how = "outputs empty"
        elif not (outs is None or _is_none(outs)):
            continue
        examined += 1
        handler = n.args[0]
        if _is_none(handler):
            continue                      # .click(None, ...) attaches JS only
        target = handler if isinstance(handler, ast.Lambda) else resolve(_dotted(handler))
        if target is None:
            continue                      # not resolvable: not accused
        if _returns_value(target, resolve):
            offenders.append((n.lineno, _dotted(handler) or "<lambda>", how))
    return offenders, examined


# --- the detector, tested on a sample before it is trusted on the real file -------------
SAMPLE = '''
def gives_back(v):
    return f"status: {v}"

def gives_nothing(v):
    _ = v

def bare_return(v):
    if v:
        return
    return None

sl.change(gives_back, [sl], None)                 # 1 bad: outputs=None
sl.change(gives_back, inputs=[sl], outputs=None)  # 2 bad: keyword
sl.change(gives_back, [sl], [])                   # 3 bad: empty list
sl.change(gives_back, [sl])                       # 4 bad: outputs omitted
btn.click(fn, [sl], None).then(gives_back, [sl], None)   # 5 bad: .then
sl.change(lambda v: gives_back(v), [sl], None)    # 6 bad: lambda over a returning call
sl.change(gives_nothing, [sl], None)              # good
sl.change(bare_return, [sl], None)                # good: every path returns None
sl.change(lambda v: gives_nothing(v), [sl], None) # good: lambda over a silent call
sl.change(lambda v: None, [sl], None)             # good
sl.change(gives_back, [sl], [status])             # good: it has somewhere to go
sl.change(None, [sl], None)                       # good: JS-only
'''


def test_the_detector_catches_every_form_and_only_those():
    defs = _defs_of(SAMPLE)
    offenders, examined = find_offenders(SAMPLE, defs.get)
    lines = sorted(l for l, _n, _h in offenders)
    assert len(offenders) == 6, f"{len(offenders)} caught instead of 6: {offenders}"
    hows = {h for _l, _n, h in offenders}
    assert hows == {"outputs=None", "outputs empty", "outputs omitted"}, hows
    # 12, not 11: the `.then` line carries TWO wirings with no outputs -- the .click that
    # opens the chain and the .then that follows it. Only the second one is guilty.
    assert examined == 12, f"{examined} wirings with no outputs instead of 12"
    assert len(lines) == len(set(lines)), "a wiring reported twice"
    print(f"OK test_the_detector_catches_every_form_and_only_those ({len(offenders)} known"
          f" offenders found, {examined - len(offenders)} innocents left alone)")


def test_no_handler_returns_into_the_void():
    root = os.path.dirname(cz_ui.__file__)
    src = io.open(os.path.join(root, "cz_ui.py"), encoding="utf-8").read()
    local = _defs_of(src)
    cache = {}

    def resolve(name):
        """A local function, or one from a sibling cz_* module."""
        if not name:
            return None
        if "." not in name:
            return local.get(name)
        mod, fn = name.rsplit(".", 1)
        if mod not in cache:
            path = os.path.join(root, mod + ".py")
            try:
                cache[mod] = _defs_of(io.open(path, encoding="utf-8").read())
            except (OSError, SyntaxError):
                cache[mod] = {}
        return cache[mod].get(fn)

    offenders, examined = find_offenders(src, resolve)
    assert examined >= 8, (f"only {examined} wirings with no outputs found: has the Gradio "
                           f"API or the wiring style changed?")
    assert not offenders, ("wired on no outputs but return a value -> Gradio warns on every "
                           "firing: " + "; ".join(f"cz_ui.py:{l} {n} ({h})"
                                                  for l, n, h in offenders))
    print(f"OK test_no_handler_returns_into_the_void ({examined} wirings with no outputs "
          f"checked)")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
    print("ALL OK")
