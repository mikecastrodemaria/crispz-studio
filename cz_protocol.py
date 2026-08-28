"""crispz family - protocole CLI v1: spec JSON en entree, JSON en sortie.

Contrat partage par la famille (crispz-studio, crispz-qwen-edit, crispz-krea*,
et le futur comics2crispz qui les pilote) :

    czp caps                       -> capacites de l'outil + instance qui tourne
    czp gen --spec spec.json       -> genere UNE image, imprime le resultat JSON

Principe : le CLI est le CONTRAT, pas le processus. `gen` route vers l'instance
Gradio qui tourne (endpoints api_name 'cli_caps' / 'cli_gen', meme mecanique
HTTP que les SPA Asset Browser / Comic Studio) : la queue de l'app serialise le
GPU et le modele reste chaud. Sans instance, execution locale (chemin froid,
batch de nuit). Les imports lourds (torch, pipeline) ne se font QUE dans ce
chemin local : `caps` et le routage ne chargent jamais rien.

Sortie : toujours UNE ligne JSON sur stdout. Codes retour :
    0 ok · 1 erreur d'execution · 2 spec invalide · 3 op/protocole non
    supporte · 4 aucune route (ni instance, ni --local possible).

v1 : une image par appel (l'appelant boucle) ; refs/guidance acceptes mais
non appliques (warning, jamais silencieux) ; sur la route remote, spec.model
est ignore avec warning (on ne change pas le modele de l'instance de
l'utilisateur sous ses pieds).
"""

import os
import sys
import json
import time
import argparse
import re
import urllib.request

from cz_core import CONFIG, APP_VERSION

PROTOCOL = 1
TOOL = "crispz-studio"
OPS = ("caps", "gen")

# Champs de spec connus du protocole (v1). Un champ inconnu = warning, jamais
# une erreur : un spec ecrit pour un autre outil de la famille doit passer.
SPEC_FIELDS = ("protocol", "op", "prompt", "negative", "width", "height",
               "seed", "steps", "guidance", "refs", "loras", "model", "input",
               "out_dir", "count")

_DEF_URL = "http://127.0.0.1:7860"


def instance_url():
    cfg = CONFIG.get("cli_protocol")
    return (cfg or {}).get("instance_url", _DEF_URL) if isinstance(cfg, dict) \
        else _DEF_URL


def caps_dict():
    """Capacites de CET outil. Leger: config seulement, aucun modele charge.
    supports.refs reste False tant que la v1 n'applique pas les refs Omni."""
    return {"ok": True, "protocol": PROTOCOL, "tool": TOOL,
            "version": APP_VERSION, "ops": list(OPS),
            "supports": {"loras": True, "refs": False, "seed": True,
                         "negative": True, "arbitrary_size": True}}


# ----------------------------------------------------------------------------
# Spec
# ----------------------------------------------------------------------------
class SpecError(Exception):
    def __init__(self, msg, code=2):
        super().__init__(msg)
        self.code = code


def validate_spec(spec, op="gen"):
    """Normalise un spec 'gen'. Renvoie (spec_normalise, warnings).
    Leve SpecError(code=2) si invalide, SpecError(code=3) si protocole/op
    hors contrat. Ne tronque jamais en silence."""
    if not isinstance(spec, dict):
        raise SpecError("spec must be a JSON object")
    try:
        proto = int(spec.get("protocol"))
    except (TypeError, ValueError):
        raise SpecError("missing/invalid 'protocol' (expected 1)")
    if proto != PROTOCOL:
        raise SpecError(f"protocol {proto} not supported (this tool speaks "
                        f"{PROTOCOL})", code=3)
    sop = spec.get("op") or op
    if sop != op:
        raise SpecError(f"spec op '{sop}' does not match command '{op}'")
    if op not in OPS:
        raise SpecError(f"op '{op}' not supported by {TOOL} (v1: {OPS})",
                        code=3)
    warnings = [f"unknown field '{k}' ignored" for k in spec
                if k not in SPEC_FIELDS]
    out = {"protocol": PROTOCOL, "op": op}
    out["prompt"] = str(spec.get("prompt") or "").strip()
    if op == "gen" and not out["prompt"]:
        raise SpecError("empty 'prompt'")
    out["negative"] = str(spec.get("negative") or "").strip()
    for key, default in (("width", 1024), ("height", 1024)):
        try:
            v = int(spec.get(key) or default)
        except (TypeError, ValueError):
            raise SpecError(f"invalid '{key}'")
        if not 64 <= v <= 4096:
            raise SpecError(f"'{key}' out of range (64-4096): {v}")
        out[key] = v
    try:
        out["seed"] = int(spec.get("seed", -1))
    except (TypeError, ValueError):
        raise SpecError("invalid 'seed'")
    steps = spec.get("steps")
    if steps is not None:
        try:
            steps = int(steps)
        except (TypeError, ValueError):
            raise SpecError("invalid 'steps'")
    out["steps"] = steps
    if spec.get("guidance") is not None:
        warnings.append("'guidance' not applied (v1: instance settings win)")
    refs = spec.get("refs") or []
    if refs:
        warnings.append(f"{len(refs)} ref(s) not applied (v1: txt2img only)")
    out["loras"] = [str(x) for x in (spec.get("loras") or [])]
    out["model"] = (str(spec["model"]).strip()
                    if spec.get("model") else None)
    out["out_dir"] = (str(spec["out_dir"]).strip()
                      if spec.get("out_dir") else None)
    count = spec.get("count")
    if count is not None and int(count) != 1:
        warnings.append("count forced to 1 (v1: one image per call, loop on "
                        "the caller side)")
    return out, warnings


# ----------------------------------------------------------------------------
# Execution locale (imports lourds ICI seulement)
# ----------------------------------------------------------------------------
def run_gen(spec, warnings=None, route="local"):
    """Genere UNE image depuis un spec VALIDE. Charge le pipeline si besoin
    (chemin froid) - ne jamais appeler pendant qu'une instance rend sur le
    meme GPU: passer par la route remote, c'est son role."""
    import cz_pipeline
    from cz_imageio import build_output_path, save_image

    warnings = list(warnings or [])
    t0 = time.time()
    if spec.get("model"):
        cz_pipeline.set_zimage_model(spec["model"])
    slots = []
    for s in spec.get("loras") or []:
        head, _, tail = str(s).rpartition(":")
        try:
            slots.append((head, float(tail)))
        except ValueError:
            slots.append((str(s), cz_pipeline.LORA_WEIGHT))
    if slots:
        cz_pipeline.set_loras(slots)
    steps = spec.get("steps") or int(CONFIG.get("default_gen_steps", 8))
    img, timings = cz_pipeline.txt2img_run(
        spec["prompt"], spec["width"], spec["height"], steps,
        spec.get("seed", -1), spec.get("negative", ""))
    seed_used = cz_pipeline._LAST_SEED
    path = build_output_path(None, "local", spec.get("out_dir"), "png",
                             tag="czp_txt2img", seed=seed_used, size=img.size)
    save_image(img, path, "png",
               meta={"mode": "txt2img", "prompt": spec["prompt"],
                     "negative": spec.get("negative", ""), "seed": seed_used,
                     "steps": steps, "size": list(img.size),
                     "loras": spec.get("loras") or []})
    return {"ok": True, "protocol": PROTOCOL, "tool": TOOL,
            "version": APP_VERSION, "route": route,
            "images": [os.path.abspath(path)], "seed_used": seed_used,
            "timings": {"total_s": round(time.time() - t0, 2),
                        **{k: round(v, 2) for k, v in (timings or {}).items()}},
            "warnings": warnings}


def handle_gen_json(spec_json):
    """Cote INSTANCE (endpoint api_name='cli_gen'): valide + genere. Renvoie
    toujours un dict {ok: ...} - l'appelant czp n'a qu'un chemin d'erreur."""
    try:
        spec, warnings = validate_spec(json.loads(spec_json or "{}"))
        if spec.get("model"):
            warnings.append("model override ignored on the remote route (the "
                            "running instance keeps its model)")
            spec["model"] = None
        return run_gen(spec, warnings, route="remote")
    except SpecError as e:
        return {"ok": False, "protocol": PROTOCOL, "tool": TOOL,
                "error": str(e), "exit_code": e.code}
    except Exception as e:
        return {"ok": False, "protocol": PROTOCOL, "tool": TOOL,
                "error": f"{type(e).__name__}: {e}", "exit_code": 1}


# ----------------------------------------------------------------------------
# Route remote (HTTP direct sur les endpoints Gradio, comme les SPA)
# ----------------------------------------------------------------------------
def _gradio_call(url, name, data, post_timeout=10, get_timeout=3600):
    """POST /gradio_api/call/<name> -> event_id, puis GET du flux et extraction
    de la premiere sortie. Renvoie la string, ou leve urllib.error/ValueError."""
    req = urllib.request.Request(
        f"{url.rstrip('/')}/gradio_api/call/{name}",
        data=json.dumps({"data": data}).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=post_timeout) as r:
        j = json.loads(r.read().decode("utf-8"))
    eid = j.get("event_id") or j.get("hash")
    if not eid:
        raise ValueError("no event id (endpoint missing? app too old?)")
    with urllib.request.urlopen(
            f"{url.rstrip('/')}/gradio_api/call/{name}/{eid}",
            timeout=get_timeout) as r:
        text = r.read().decode("utf-8")
    m = re.search(r"data:\s*(\[[\s\S]*?\])\s*(?:\n|$)", text)
    if not m:
        raise ValueError("empty API response")
    return json.loads(m.group(1))[0]


def probe_instance(url, timeout=4):
    """caps de l'instance qui tourne a `url`, ou None si rien ne repond.
    Court: ne bloque jamais un appel local plus de quelques secondes."""
    try:
        raw = _gradio_call(url, "cli_caps", [], post_timeout=timeout,
                           get_timeout=max(timeout, 8))
        caps = json.loads(raw) if isinstance(raw, str) else raw
        return caps if isinstance(caps, dict) and caps.get("ok") else None
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Entree CLI (czp.bat / python cz_protocol.py)
# ----------------------------------------------------------------------------
def _emit(payload, code):
    print(json.dumps(payload, ensure_ascii=False))
    return code


def _read_spec(path):
    # utf-8-sig: PowerShell 5.1 (Set-Content/Out-File -Encoding utf8) ecrit un
    # BOM - le refuser casserait le premier test venu sous Windows. Le strip
    # supplementaire couvre le BOM arrive par stdin.
    raw = sys.stdin.read() if path == "-" \
        else open(path, encoding="utf-8-sig").read()
    return json.loads(raw.lstrip("\ufeff"))


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="czp", description="crispz family CLI protocol v1 (JSON in/out)")
    parser.add_argument("op", choices=["caps", "gen", "edit", "upscale"])
    parser.add_argument("--spec", metavar="FILE",
                        help="spec JSON ('-' = stdin); required for gen")
    parser.add_argument("--local", action="store_true",
                        help="force local execution (night batch), never route")
    parser.add_argument("--remote", metavar="URL",
                        help="force this instance (error 4 if unreachable)")
    args = parser.parse_args(argv)

    if args.op == "caps":
        caps = caps_dict()
        url = args.remote or instance_url()
        inst = probe_instance(url)
        caps["instance"] = ({"running": True, "url": url,
                             "tool": inst.get("tool"),
                             "version": inst.get("version")} if inst
                            else {"running": False, "url": url})
        return _emit(caps, 0)

    if args.op not in OPS:
        return _emit({"ok": False, "tool": TOOL,
                      "error": f"op '{args.op}' not supported by {TOOL} "
                               f"(v1: {', '.join(OPS)})"}, 3)

    if not args.spec:
        return _emit({"ok": False, "tool": TOOL,
                      "error": "gen requires --spec <file|->"}, 2)
    try:
        spec, warnings = validate_spec(_read_spec(args.spec), op=args.op)
    except SpecError as e:
        return _emit({"ok": False, "tool": TOOL, "error": str(e)}, e.code)
    except Exception as e:
        return _emit({"ok": False, "tool": TOOL,
                      "error": f"unreadable spec: {e}"}, 2)

    # Routage: remote si une instance repond (sa queue serialise le GPU),
    # local sinon. --local / --remote court-circuitent la detection.
    if not args.local:
        url = args.remote or instance_url()
        inst = probe_instance(url)
        if inst:
            if spec.get("model"):
                warnings.append("model override ignored on the remote route "
                                "(the running instance keeps its model)")
                spec["model"] = None
            try:
                raw = _gradio_call(url, "cli_gen", [json.dumps(spec)])
                res = json.loads(raw) if isinstance(raw, str) else raw
            except Exception as e:
                return _emit({"ok": False, "tool": TOOL,
                              "error": f"remote call failed: {e}"}, 1)
            res.setdefault("warnings", [])
            res["warnings"] = warnings + [w for w in res["warnings"]
                                          if w not in warnings]
            return _emit(res, 0 if res.get("ok") else
                         int(res.get("exit_code", 1)))
        if args.remote:
            return _emit({"ok": False, "tool": TOOL,
                          "error": f"no instance at {args.remote}"}, 4)

    try:
        return _emit(run_gen(spec, warnings), 0)
    except Exception as e:
        return _emit({"ok": False, "tool": TOOL,
                      "error": f"{type(e).__name__}: {e}"}, 1)


if __name__ == "__main__":
    sys.exit(main())
