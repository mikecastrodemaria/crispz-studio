"""Improve prompt wired into the tool (cz_ollama + UI handlers + CLI), against a fake
local Ollama server. No GPU, no real Ollama.

- positive: the model picked for Improve, else the first installed; directives travel;
  a failure keeps the text untouched (no silent local keyword fallback any more);
- negative: an empty box starts from the standard negative; Ollama down -> that
  negative is inserted as is, with a warning;
- config compatibility: a CUSTOM ollama_improve_prompt becomes the positive instruction,
  the shipped one does not; keep_alive falls back on ollama_keep_alive;
- transport: a 'localhost' URL works (rewritten to 127.0.0.1);
- CLI: --improve / --improve-negative / --directives rewrite before generating.

Run:  .venv/Scripts/python tests/test_improve_wiring.py
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

import prompt_improve  # noqa: E402
import cz_ollama  # noqa: E402

DOWN = "http://127.0.0.1:9"      # nothing listens there


class FakeOllama(BaseHTTPRequestHandler):
    bodies = []
    reply = "IMPROVED"

    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send({"models": [{"name": "qwen3:8b"}, {"name": "llama3.1:8b"}]})
        else:
            self._send({"error": "nope"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if self.path == "/api/generate":
            FakeOllama.bodies.append(body)
            return self._send({"response": FakeOllama.reply})
        self._send({"error": "nope"}, 404)


_SRV = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
threading.Thread(target=_SRV.serve_forever, daemon=True).start()
URL = f"http://127.0.0.1:{_SRV.server_address[1]}"


def _last():
    return FakeOllama.bodies[-1]


def _value(update):
    """Value carried by a gr.update() dict, or None when the box is left untouched."""
    return update.get("value") if isinstance(update, dict) else update


# ----------------------------------------------------------- cz_ollama ---
def test_shipped_improve_instruction_is_not_a_customization():
    s = cz_ollama._improve_settings({"ollama_improve_prompt":
                                     cz_ollama._SHIPPED_IMPROVE_INSTRUCTIONS[0]})
    assert "positive_instruction" not in s


def test_custom_legacy_instruction_becomes_the_positive_instruction():
    s = cz_ollama._improve_settings({"ollama_improve_prompt": "Be epic.\n\nPROMPT: {prompt}"})
    assert s["positive_instruction"] == "Be epic.\n\nPROMPT: {prompt}"
    s = cz_ollama._improve_settings({"ollama_improve_prompt": "Be epic. {prompt}",
                                     "ollama_improve": {"positive_instruction": "Mine {prompt}"}})
    assert s["positive_instruction"] == "Mine {prompt}"


def test_keep_alive_falls_back_on_ollama_keep_alive():
    assert cz_ollama._improve_settings({})["keep_alive"] == 0
    assert cz_ollama._improve_settings({"ollama_keep_alive": "2m"})["keep_alive"] == "2m"
    assert cz_ollama._improve_settings({"ollama_keep_alive": "2m",
                                        "ollama_improve": {"keep_alive": 0}})["keep_alive"] == 0


def test_transport_accepts_a_localhost_url():
    port = _SRV.server_address[1]
    out = cz_ollama._ollama_http("/api/tags", base=f"http://localhost:{port}", timeout=5)
    assert [m["name"] for m in out["models"]] == ["qwen3:8b", "llama3.1:8b"]


def test_list_text_models_lists_every_model():
    assert cz_ollama.list_text_models(base=URL) == ["qwen3:8b", "llama3.1:8b"]


# ---------------------------------------------------------- UI handlers ---
def test_ui_improve_uses_the_picked_model_and_directives():
    import cz_ui
    upd, status = cz_ui._ui_improve("a fox", "llama3.1:8b", URL, "in French")
    assert _value(upd) == "IMPROVED" and "llama3.1:8b" in status and "directives" in status
    assert _last()["model"] == "llama3.1:8b"
    assert "in French" in _last()["prompt"] and _last()["prompt"].endswith("PROMPT: a fox")


def test_ui_improve_without_model_takes_the_first_installed():
    import cz_ui
    upd, status = cz_ui._ui_improve("a fox", None, URL)
    assert _value(upd) == "IMPROVED" and _last()["model"] == "qwen3:8b"


def test_ui_improve_empty_prompt_makes_no_call():
    import cz_ui
    n = len(FakeOllama.bodies)
    upd, status = cz_ui._ui_improve("   ", None, URL)
    assert _value(upd) is None and len(FakeOllama.bodies) == n


def test_ui_improve_failure_keeps_the_text():
    import cz_ui
    upd, status = cz_ui._ui_improve("a fox", "llama3.1:8b", DOWN)
    assert _value(upd) is None, "the prompt box must stay untouched"
    assert status.startswith("⚠") and "unreachable" in status


def test_ui_improve_negative_empty_box_starts_from_the_standard_negative():
    import cz_ui
    upd, status = cz_ui._ui_improve_negative("", "llama3.1:8b", URL)
    assert _value(upd) == "IMPROVED"
    assert "NEGATIVE PROMPT: " + prompt_improve.default_negative() in _last()["prompt"]


def test_ui_improve_negative_empty_box_and_ollama_down_inserts_the_standard_negative():
    import cz_ui
    upd, status = cz_ui._ui_improve_negative("", "llama3.1:8b", DOWN)
    assert _value(upd) == prompt_improve.default_negative()
    assert status.startswith("⚠") and "standard negative" in status


def test_ui_improve_negative_filled_box_and_ollama_down_keeps_the_text():
    import cz_ui
    upd, status = cz_ui._ui_improve_negative("blurry", "llama3.1:8b", DOWN)
    assert _value(upd) is None and status.startswith("⚠")


def test_ui_detect_lists_text_models_for_improve():
    import cz_ui
    vision, improve, status = cz_ui._ui_detect_ollama(URL)
    assert improve["choices"] == ["qwen3:8b", "llama3.1:8b"]


def test_ui_toggle_panel():
    import cz_ui
    opened, upd = cz_ui._ui_toggle_panel(False)
    assert opened is True and upd["visible"] is True
    opened, upd = cz_ui._ui_toggle_panel(True)
    assert opened is False and upd["visible"] is False


# ------------------------------------------------------------------- CLI ---
def _cli(argv):
    import cz_cli
    got = {}

    def fake_txt2img_run(prompt, w, h, gen_steps, seed, negative, **kw):
        got.update(prompt=prompt, negative=negative)
        return Image.new("RGB", (32, 32)), {"txt2img": 0.0}

    real, real_url = cz_cli.txt2img_run, cz_ollama.OLLAMA_URL
    cz_cli.txt2img_run = fake_txt2img_run
    try:
        rc = cz_cli.cli_main(["--txt2img", "--save-mode", "display", "--quiet"] + argv)
    finally:
        cz_cli.txt2img_run = real
        cz_ollama.OLLAMA_URL = real_url
    return rc, got


def test_cli_improve_rewrites_the_prompt_before_generating():
    cz_ollama.OLLAMA_URL = URL
    rc, got = _cli(["--prompt", "a fox", "--improve", "--directives", "winter night",
                    "--improve-model", "llama3.1:8b"])
    assert rc in (0, None) and got["prompt"] == "IMPROVED"
    assert "winter night" in _last()["prompt"] and _last()["model"] == "llama3.1:8b"


def test_cli_improve_negative_with_ollama_down_uses_the_standard_negative():
    cz_ollama.OLLAMA_URL = DOWN
    rc, got = _cli(["--prompt", "a fox", "--improve-negative"])
    assert rc in (0, None)
    assert got["prompt"] == "a fox" and got["negative"] == prompt_improve.default_negative()


def test_cli_improve_with_ollama_down_generates_nothing():
    cz_ollama.OLLAMA_URL = DOWN
    rc, got = _cli(["--prompt", "a fox", "--improve"])
    assert rc == 2 and not got


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"All {len(tests)} improve wiring tests passed.")
    _SRV.shutdown()
