"""Le monologue interne d'un modele de raisonnement ne doit JAMAIS finir dans le prompt.

Regression: avec un modele thinking (Qwen3, DeepSeek-R1, Kimi...), Describe /
Improve / Vision Mix renvoyaient "Okay, the user wants a prompt for..." colle
devant le vrai prompt, qui partait tel quel dans le text encoder.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_ollama as O

CASES = [
    # (entree, sortie attendue)
    ("<think>Okay, the user wants a cat.</think>a fluffy cat, studio light",
     "a fluffy cat, studio light"),
    ("<thinking>\nlong\nmulti-line\n</thinking>\n\na fluffy cat",
     "a fluffy cat"),
    ("<Think>CASE INSENSITIVE</Think>a cat", "a cat"),
    ("<reasoning>x</reasoning> a cat", "a cat"),
    # fermeture orpheline: le modele pensait avant le 1er token capture
    ("Okay, let me think about this.</think>a fluffy cat", "a fluffy cat"),
    # ouverture jamais fermee: il n'y a QUE du raisonnement -> rien a garder
    ("<think>truncated reasoning that never closes", ""),
    # deux blocs
    ("<think>a</think>one <think>b</think>two", "one two"),
    # texte normal: inchange
    ("a fluffy cat, studio light", "a fluffy cat, studio light"),
    ("", ""),
    (None, ""),
]


def test_strip_thinking():
    for raw, want in CASES:
        got = O._strip_thinking(raw)
        assert got == want, f"{raw!r} -> {got!r} (attendu {want!r})"
    print("OK test_strip_thinking")


def test_gen_opts_disables_thinking():
    opts = O._ollama_gen_opts()
    assert opts.get("think") is False, opts
    assert opts.get("stream") is False, opts
    print("OK test_gen_opts_disables_thinking")


def test_http_retries_without_think():
    """Un modele qui ne connait pas 'think' repond 400 -> on rejoue sans le champ."""
    import urllib.error
    seen = []

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"response": "ok"}'

    def fake_urlopen(req, timeout=None):
        import json as _j
        payload = _j.loads(req.data.decode())
        seen.append(payload)
        if "think" in payload:
            raise urllib.error.HTTPError(req.full_url, 400, "does not support thinking",
                                         None, None)
        return _Resp()

    # Le transport est prompt_improve.http (ouvreur sans proxy): on remplace son open().
    import prompt_improve
    old = prompt_improve._OPENER.open
    prompt_improve._OPENER.open = fake_urlopen
    try:
        out = O._ollama_http("/api/generate", {"model": "m", "prompt": "p", "think": False})
    finally:
        prompt_improve._OPENER.open = old
    assert out == {"response": "ok"}, out
    assert len(seen) == 2, seen
    assert "think" in seen[0] and "think" not in seen[1], seen
    print("OK test_http_retries_without_think")


if __name__ == "__main__":
    test_strip_thinking()
    test_gen_opts_disables_thinking()
    test_http_retries_without_think()
    print("All Ollama thinking tests passed.")
