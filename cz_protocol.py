"""crispz family - CLI protocol v1: JSON spec in, JSON out.

Contract shared by the whole family (crispz-studio, crispz-qwen-edit,
crispz-krea*, and comics2crispz which drives them):

    czp caps                       -> this tool's capabilities + running instance
    czp gen --spec spec.json       -> generate ONE image, print the JSON result

Principle: the CLI is the CONTRACT, not the process. `gen` routes to the
running Gradio instance (hidden api_name endpoints 'cli_caps' / 'cli_gen',
same HTTP mechanics as the Asset Browser / Comic Studio SPAs): the app's queue
serializes the GPU and the model stays warm. Without an instance, local
execution (cold path, night batch). Heavy imports (torch, pipeline) happen
ONLY on that local path: `caps` and the routing never load anything.

Output: always ONE JSON line on stdout. Exit codes:
    0 ok - 1 run error - 2 invalid spec - 3 unsupported op/protocol -
    4 no route (no instance and no local execution possible).

v1: one image per call (the caller loops); guidance is accepted but not
applied (warning, never silent); on the remote route spec.model is ignored
with a warning (we never swap the model of the user's running instance under
their feet).

loras: spec.loras = ["file.safetensors:0.8", ...], hot-swapped per call. The
prompt may also carry <lora:file[:weight]> tags (A1111/Civitai habit): they
are extracted at validation time and merged into spec.loras (explicit entries
win on duplicates), so style/character LoRAs travel INSIDE panel texts too.

refs (v2): spec.refs = LOCAL image paths (the protocol is machine-local).
When an Omni model is configured (zimage_omni_model), generation goes through
generate_omni (multi-reference, character consistency); otherwise refs are
dropped WITH a warning and supports.refs announces it upfront. A ref missing
on disk is an error (code 2), never a silently degraded generation. At most
4 refs (Omni UI limit), extras are cut with a warning.
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
OPS = ("caps", "gen", "upscale")

# Spec fields known to the protocol (v1). An unknown field is a warning,
# never an error: a spec written for another family tool must go through.
SPEC_FIELDS = ("protocol", "op", "prompt", "negative", "width", "height",
               "seed", "steps", "guidance", "refs", "loras", "model", "input",
               "out_dir", "count", "detail_faces", "detail_hands",
               "factor", "denoise")

_DEF_URL = "http://127.0.0.1:7860"

MAX_REFS = 4                      # Omni pipeline limit (same as the UI)

# Whether this model family has a multi-reference/omni pipeline AT ALL.
# crispz-krea / crispz-krea2 hard-set this to False (their _load_omni raises:
# no instruction-edit model exists for Krea) - config can never turn refs on
# there, and caps must say so instead of promising a capability that raises.
OMNI_SUPPORTED = True
# Default omni repo when the family ships one out of the box
# (crispz-qwen-edit: Qwen-Image-Edit). Empty = config/env must set it.
OMNI_DEFAULT = ""


def _omni_configured():
    """Is an Omni model (multi-reference) available? CONFIG/env only - same
    rule as cz_pipeline.OMNI_MODEL, without importing the pipeline (caps must
    stay light)."""
    if not OMNI_SUPPORTED:
        return False
    return bool((os.environ.get("ZIMAGE_OMNI_MODEL")
                 or CONFIG.get("zimage_omni_model") or OMNI_DEFAULT).strip())


def instance_url():
    cfg = CONFIG.get("cli_protocol")
    return (cfg or {}).get("instance_url", _DEF_URL) if isinstance(cfg, dict) \
        else _DEF_URL


def _faces_available():
    """Is the face-detection stack (insightface) installed? Light check:
    module spec only, nothing imported - caps must stay fast."""
    import importlib.util
    return importlib.util.find_spec("insightface") is not None


def _models_dir(kind):
    """Dossier des modeles ('checkpoints'|'loras'), memes priorites que
    cz_pipeline (env > preferences > config > defaut) mais sans l'importer:
    caps doit rester leger."""
    from cz_core import HERE, _prefs
    return (os.environ.get(kind.upper() + "_DIR")
            or _prefs.get(f"{kind}_dir") or CONFIG.get(f"{kind}_dir")
            or os.path.join(HERE, kind))


def _list_model_files(kind, exts=(".safetensors", ".gguf")):
    """Fichiers modeles disponibles (chemins relatifs POSIX, tries). Simple
    listage disque - la SOURCE DE VERITE des modeles reste chaque outil, les
    appelants (wizard comics2crispz) ne configurent aucun chemin."""
    d = _models_dir(kind)
    out = []
    if os.path.isdir(d):
        for root, dirs, files in os.walk(d):
            dirs[:] = [x for x in dirs
                       if x not in ("_index", ".cache", "recipes")]
            for f in files:
                if f.lower().endswith(exts):
                    out.append(os.path.relpath(os.path.join(root, f), d)
                               .replace("\\", "/"))
    return sorted(out, key=str.lower)


def _model_loaded():
    """Modele actif de CE process: renseigne cote instance (endpoint
    cli_caps), '' cote czp froid (cz_pipeline pas importe - et on ne
    l'importe PAS pour ca)."""
    mod = sys.modules.get("cz_pipeline")
    if mod is None:
        return ""
    t = getattr(mod, "ZIMAGE_TRANSFORMER", None)
    if t:
        return os.path.basename(str(t))
    return str(getattr(mod, "BASE_REPO", "") or "")


def _hands_available():
    """Le detailer de mains (YOLOv8) demande le paquet optionnel
    'ultralytics'. Check leger, rien d'importe."""
    import importlib.util
    return importlib.util.find_spec("ultralytics") is not None


def caps_dict():
    """This tool's capabilities. Light: config + disk listing, no model
    loaded. supports.refs = an Omni model is configured (multi-reference
    generation will actually work); supports.faces = the instance can serve
    face detection (cli_faces endpoint). models/loras = the files this tool
    can use (its own dirs - callers never configure model paths);
    model_loaded = what THIS process has active (instance side)."""
    return {"ok": True, "protocol": PROTOCOL, "tool": TOOL,
            "version": APP_VERSION, "ops": list(OPS),
            "supports": {"loras": True, "refs": _omni_configured(),
                         "max_refs": MAX_REFS, "seed": True,
                         "negative": True, "arbitrary_size": True,
                         "faces": _faces_available(),
                         "detail_faces": _faces_available(),
                         "detail_hands": _hands_available()},
            "model_loaded": _model_loaded(),
            "models": _list_model_files("checkpoints"),
            "loras": _list_model_files("loras")}


# ----------------------------------------------------------------------------
# Spec
# ----------------------------------------------------------------------------
# <lora:file[:weight]> tags INSIDE the prompt (the A1111/Civitai habit).
# Extracted at validation time so every route (remote endpoint, local run)
# and every caller (czp, comics2crispz panel texts) gets them, and the text
# encoder never sees the tag itself. Explicit spec.loras win on duplicates
# (first weight wins, the family rule).
_LORA_TAG = re.compile(r"<lora:([^:>]+?)(?::([0-9.]+))?>", re.IGNORECASE)


def _extract_prompt_loras(prompt, loras):
    """Moves <lora:...> tags from the prompt into `loras` (in place).
    Returns the cleaned prompt."""
    def _sub(m):
        name = m.group(1).strip()
        if name:
            seen = {str(x).split(":", 1)[0].strip().lower() for x in loras}
            if name.lower() not in seen:
                loras.append(f"{name}:{m.group(2)}" if m.group(2) else name)
        return " "
    out = _LORA_TAG.sub(_sub, prompt)
    if out != prompt:
        out = re.sub(r"\s{2,}", " ", out)
        out = re.sub(r"\s+,", ",", out).strip(" ,")
    return out


class SpecError(Exception):
    def __init__(self, msg, code=2):
        super().__init__(msg)
        self.code = code


def validate_spec(spec, op="gen"):
    """Normalize a 'gen' spec. Returns (normalized_spec, warnings).
    Raises SpecError(code=2) when invalid, SpecError(code=3) when the
    protocol/op is out of contract. Never truncates silently."""
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
    if op == "upscale":
        inp = str(spec.get("input") or "").strip()
        if not inp:
            raise SpecError("upscale requires 'input' (image path)")
        if not os.path.isfile(inp):
            raise SpecError(f"input not found on disk: {inp} (paths are "
                            f"LOCAL - the caller resolves them)")
        out["input"] = inp
        try:
            out["factor"] = float(spec.get("factor") or 2.0)
        except (TypeError, ValueError):
            raise SpecError("invalid 'factor'")
        if not 1.0 <= out["factor"] <= 8.0:
            raise SpecError(f"'factor' out of range (1-8): {out['factor']}")
        den = spec.get("denoise")
        if den is not None:
            try:
                den = float(den)
            except (TypeError, ValueError):
                raise SpecError("invalid 'denoise'")
            if not 0.0 <= den <= 1.0:
                raise SpecError("'denoise' out of range (0-1)")
        out["denoise"] = den
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
    refs = [str(r) for r in (spec.get("refs") or []) if str(r).strip()]
    if refs and not _omni_configured():
        warnings.append(f"{len(refs)} ref(s) ignored: no omni model "
                        f"configured (set zimage_omni_model) - character "
                        f"consistency by refs is OFF")
        refs = []
    if len(refs) > MAX_REFS:
        warnings.append(f"{len(refs)} refs, keeping the first {MAX_REFS} "
                        f"(omni pipeline limit)")
        refs = refs[:MAX_REFS]
    for r in refs:
        if not os.path.isfile(r):
            raise SpecError(f"ref not found on disk: {r} (refs are LOCAL "
                            f"file paths - the caller resolves them)")
    out["refs"] = refs
    out["loras"] = [str(x) for x in (spec.get("loras") or [])]
    out["prompt"] = _extract_prompt_loras(out["prompt"], out["loras"])
    if op == "gen" and not out["prompt"]:
        raise SpecError("prompt contained only <lora:...> tags - describe "
                        "the image too")
    out["model"] = (str(spec["model"]).strip()
                    if spec.get("model") else None)
    out["out_dir"] = (str(spec["out_dir"]).strip()
                      if spec.get("out_dir") else None)
    count = spec.get("count")
    if count is not None and int(count) != 1:
        warnings.append("count forced to 1 (v1: one image per call, loop on "
                        "the caller side)")
    # Detailer ADetailer-style: true/false force, absent (None) = le reglage
    # courant de l'outil (config face_detailer) pour les visages, OFF pour
    # les mains (ultralytics optionnel - jamais implicite).
    for key in ("detail_faces", "detail_hands"):
        v = spec.get(key)
        out[key] = None if v is None else bool(v)
    return out, warnings


# ----------------------------------------------------------------------------
# Local execution (heavy imports HERE only)
# ----------------------------------------------------------------------------
def run_gen(spec, warnings=None, route="local"):
    """Generate ONE image from a VALIDATED spec. Loads the pipeline when
    needed (cold path) - never call this while an instance is rendering on
    the same GPU: that is what the remote route is for."""
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
    # The seed is resolved HERE, the same way the UI does (cz_ui.run): a -1
    # becomes a concrete value BEFORE generating -> seed_used is always exact
    # and replayable. (Do NOT read cz_pipeline._LAST_SEED afterwards: only the
    # UI path sets it, it would report the instance's last UI render.)
    import random
    seed_used = int(spec.get("seed", -1))
    if seed_used < 0:
        seed_used = random.randint(0, 2**31 - 1)
    cz_pipeline._LAST_SEED = seed_used          # the UI's 'Reuse last seed' sees it
    refs = spec.get("refs") or []
    if refs:
        # Omni route (multi-reference): refs were already validated by
        # validate_spec (files exist, omni configured).
        from PIL import Image
        t_o = time.time()
        imgs = [Image.open(r) for r in refs]
        try:
            img = cz_pipeline.generate_omni(
                imgs, spec["prompt"], spec.get("negative", ""),
                spec["width"], spec["height"], steps, seed_used)
        finally:
            for im in imgs:
                try:
                    im.close()
                except Exception:
                    pass
        timings = {"omni": time.time() - t_o}
    else:
        img, timings = cz_pipeline.txt2img_run(
            spec["prompt"], spec["width"], spec["height"], steps,
            seed_used, spec.get("negative", ""))
    # Passe detailer (visages ADetailer-style, mains YOLO) - la meme que le
    # CLI --detail-faces/--detail-hands. Un echec (paquet optionnel absent,
    # detecteur KO) DEGRADE en warning, jamais en panneau perdu.
    timings = dict(timings or {})
    nf = nh = 0
    df, dh = spec.get("detail_faces"), spec.get("detail_hands")
    if df or dh or df is None:
        try:
            import cz_detailer
            if df is None:
                df = bool(getattr(cz_detailer, "DETAILER_ENABLED", False))
            if df:
                t_d = time.time()
                img, nf = cz_detailer.detail_faces(img, spec["prompt"],
                                                   seed_used, steps)
                timings["detail_faces"] = time.time() - t_d
            if dh:
                t_d = time.time()
                img, nh = cz_detailer.detail_hands(img, spec["prompt"],
                                                   seed_used, steps)
                timings["detail_hands"] = time.time() - t_d
        except Exception as e:
            warnings.append(f"detailer skipped: {e}")
    path = build_output_path(None, "local", spec.get("out_dir"), "png",
                             tag=("czp_omni" if refs else "czp_txt2img"),
                             seed=seed_used, size=img.size)
    save_image(img, path, "png",
               meta={"mode": "omni" if refs else "txt2img",
                     "prompt": spec["prompt"],
                     "negative": spec.get("negative", ""), "seed": seed_used,
                     "steps": steps, "size": list(img.size),
                     "loras": spec.get("loras") or [],
                     "refs": [os.path.basename(r) for r in refs]})
    return {"ok": True, "protocol": PROTOCOL, "tool": TOOL,
            "version": APP_VERSION, "route": route,
            "images": [os.path.abspath(path)], "seed_used": seed_used,
            "refs_used": len(refs), "loras": list(spec.get("loras") or []),
            "faces_refined": nf, "hands_refined": nh,
            "timings": {"total_s": round(time.time() - t0, 2),
                        **{k: round(v, 2) for k, v in (timings or {}).items()}},
            "warnings": warnings}


def run_upscale(spec, warnings=None, route="local"):
    """Upscale UNE image (ESRGAN + refine, le pipeline de l'outil) depuis un
    spec VALIDE. Sortie print de la famille: les cases BD generees a ~1 MP
    remontent a la resolution d'impression par ici. Le prompt (optionnel)
    guide la passe de refine - JAMAIS le prompt de scene sur un crop, regle
    maison: l'appelant envoie une description LOCALE ou rien."""
    import cz_pipeline
    import cz_esrgan
    from PIL import Image
    from cz_imageio import build_output_path, save_image

    warnings = list(warnings or [])
    t0 = time.time()
    models = cz_esrgan.list_esrgan_models()
    model = None
    if spec.get("model"):
        model = spec["model"] if spec["model"] in models else None
        if model is None:
            warnings.append(f"ESRGAN model '{spec['model']}' not found - "
                            f"using {models[0] if models else 'none'}")
    if model is None:
        model = models[0] if models else None
    if model is None:
        raise RuntimeError(f"no ESRGAN model in {cz_esrgan.ESRGAN_DIR}")
    denoise = spec.get("denoise")
    if denoise is None:
        denoise = float(CONFIG.get("default_denoise", 0.30))
    steps = spec.get("steps") or int(CONFIG.get("default_refine_steps",
                                                CONFIG.get("default_steps",
                                                           12)))
    with Image.open(spec["input"]) as im:
        img, timings = cz_pipeline.process_one(
            im.convert("RGB"), model, spec.get("factor", 2.0), denoise,
            steps, spec.get("prompt", ""), spec.get("seed", -1),
            int(CONFIG.get("default_tile", 0)),
            int(CONFIG.get("default_overlap", 16)))
    path = build_output_path(None, "local", spec.get("out_dir"), "png",
                             tag="czp_upscale", seed=spec.get("seed", -1),
                             size=img.size)
    save_image(img, path, "png",
               meta={"mode": "upscale", "source": spec["input"],
                     "factor": spec.get("factor"), "denoise": denoise,
                     "model": model, "size": list(img.size)})
    return {"ok": True, "protocol": PROTOCOL, "tool": TOOL,
            "version": APP_VERSION, "route": route,
            "images": [os.path.abspath(path)],
            "size": list(img.size), "esrgan_model": model,
            "timings": {"total_s": round(time.time() - t0, 2),
                        **{k: round(v, 2) for k, v in (timings or {}).items()
                           if isinstance(v, (int, float))}},
            "warnings": warnings}


def handle_upscale_json(spec_json):
    """Cote INSTANCE (endpoint api_name='cli_upscale')."""
    try:
        spec, warnings = validate_spec(json.loads(spec_json or "{}"),
                                       op="upscale")
        return run_upscale(spec, warnings, route="remote")
    except SpecError as e:
        return {"ok": False, "protocol": PROTOCOL, "tool": TOOL,
                "error": str(e), "exit_code": e.code}
    except Exception as e:
        return {"ok": False, "protocol": PROTOCOL, "tool": TOOL,
                "error": f"{type(e).__name__}: {e}", "exit_code": 1}


def handle_gen_json(spec_json):
    """INSTANCE side (api_name='cli_gen' endpoint): validate + generate.
    Always returns a dict {ok: ...} - the czp caller has a single error
    path."""
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
# Remote route (direct HTTP on the Gradio endpoints, like the SPAs)
# ----------------------------------------------------------------------------
def _gradio_call(url, name, data, post_timeout=10, get_timeout=3600):
    """POST /gradio_api/call/<name> -> event_id, then GET the stream and
    extract the first output. Returns the string, or raises
    urllib.error/ValueError."""
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
    """caps of the instance running at `url`, or None if nothing answers.
    Short: never delays a local call by more than a few seconds."""
    try:
        raw = _gradio_call(url, "cli_caps", [], post_timeout=timeout,
                           get_timeout=max(timeout, 8))
        caps = json.loads(raw) if isinstance(raw, str) else raw
        return caps if isinstance(caps, dict) and caps.get("ok") else None
    except Exception:
        return None


# ----------------------------------------------------------------------------
# CLI entry (czp.bat / python cz_protocol.py)
# ----------------------------------------------------------------------------
def _emit(payload, code):
    print(json.dumps(payload, ensure_ascii=False))
    return code


def _read_spec(path):
    # utf-8-sig: PowerShell 5.1 (Set-Content/Out-File -Encoding utf8) writes
    # a BOM - refusing it would break the first test anyone runs on Windows.
    # The extra strip covers a BOM arriving through stdin.
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

    # Routing: remote when an instance answers (its queue serializes the
    # GPU), local otherwise. --local / --remote bypass the detection.
    endpoint = "cli_upscale" if args.op == "upscale" else "cli_gen"
    if not args.local:
        url = args.remote or instance_url()
        inst = probe_instance(url)
        if inst:
            if args.op == "gen" and spec.get("model"):
                # (upscale: spec.model = modele ESRGAN, sans danger remote)
                warnings.append("model override ignored on the remote route "
                                "(the running instance keeps its model)")
                spec["model"] = None
            try:
                raw = _gradio_call(url, endpoint, [json.dumps(spec)])
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
        runner = run_upscale if args.op == "upscale" else run_gen
        return _emit(runner(spec, warnings), 0)
    except Exception as e:
        return _emit({"ok": False, "tool": TOOL,
                      "error": f"{type(e).__name__}: {e}"}, 1)


if __name__ == "__main__":
    sys.exit(main())
