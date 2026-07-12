"""crispz-studio - CLI (mode batch / scripting) + serveur HTTP persistant (FastAPI).

Extrait de app.py (step 8). Importe l'UI et l'orchestration depuis cz_ui (qui branche
tous les cz_*); ne redefinit rien du pipeline. app.py se contente d'appeler cli_main.
"""

import os
import sys
import time
import glob

from PIL import Image

import cz_core
import cz_pipeline
import cz_esrgan
from cz_ui import (  # noqa: F401
    # constantes / chemins / defauts
    DEVICE, HERE, PREFS_PATH, PRESETS, SUPPORTED_FORMATS,
    DEFAULT_MODEL, DEFAULT_FACTOR, DEFAULT_DENOISE, DEFAULT_STEPS, DEFAULT_TILE,
    DEFAULT_OVERLAP, DEFAULT_REFINE_TILE, DEFAULT_REFINE_OVERLAP, DEFAULT_SAVE_MODE,
    DEFAULT_OUTPUT_DIR, DEFAULT_OUTPUT_FORMAT,
    # orchestration / pipeline (re-exportes par cz_ui)
    run, process_one, txt2img_run, outpaint, build_ui,
    free_vram, set_offload_mode, set_guidance, set_sampler, SAMPLER_CHOICES,
    set_schedule, SCHEDULE_CHOICES, set_esrgan_dir, set_zimage_model,
    set_zimage_transformer, set_loras_dir, set_loras, list_esrgan_models,
    build_output_path, save_image, _faceswap, _remove_bg,
    _ollama_vision_models, _ollama_describe, _ollama_compose,
    apply_preset_to_args, _report_vram, _reset_vram_peak, _format_timings,
    _append_time_log, _save_prefs_keys, _parse_log_level, _log, _ab_resolve_dir,
)


def _disable_brotli():
    """Neutralise le brotli_middleware de Gradio (bug h11 'Content-Length' a l'envoi
    de gros resultats). Patch sur le symbole importe par gradio.routes."""
    class _Passthrough:
        def __init__(self, app, *a, **k):
            self.app = app

        async def __call__(self, scope, receive, send):
            await self.app(scope, receive, send)
    import importlib
    for modname in ("gradio.routes", "gradio.brotli_middleware"):
        try:
            m = importlib.import_module(modname)
            if hasattr(m, "BrotliMiddleware"):
                m.BrotliMiddleware = _Passthrough
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Palier 3 : serveur HTTP persistant (FastAPI), load paresseux + unload sur idle
# ----------------------------------------------------------------------------
def serve_main(host="127.0.0.1", port=7861, idle_timeout=300):
    """Petit serveur HTTP. Le modele Z-Image se charge au premier /upscale et reste
    chaud (plus de rechargement entre appels -> temps stables). Apres idle_timeout
    secondes sans requete, la VRAM est rendue (utile pour cohabiter avec Fooocus).
    Endpoints: GET /health, GET /models, POST /upscale, POST /unload."""
    try:
        import threading
        import uvicorn
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except Exception as e:
        print("[serve] FastAPI/uvicorn required: pip install fastapi uvicorn", file=sys.stderr)
        print(f"[serve] detail: {e}", file=sys.stderr)
        return 1

    os.makedirs(cz_esrgan.ESRGAN_DIR, exist_ok=True)
    app = FastAPI(title="crispz")
    lock = threading.Lock()
    state = {"last": time.time()}

    class UpscaleReq(BaseModel):
        input: str
        model: str = DEFAULT_MODEL
        factor: float = DEFAULT_FACTOR
        denoise: float = DEFAULT_DENOISE
        steps: int = DEFAULT_STEPS
        prompt: str = ""
        seed: int = -1
        tile: int = DEFAULT_TILE
        overlap: int = DEFAULT_OVERLAP
        refine_tile: int = DEFAULT_REFINE_TILE
        refine_overlap: int = DEFAULT_REFINE_OVERLAP
        cpu_offload: str = "none"
        preset: str = "Custom"
        save_mode: str = "local"
        output_dir: str = DEFAULT_OUTPUT_DIR
        output_format: str = DEFAULT_OUTPUT_FORMAT

    @app.get("/health")
    def health():
        return {"status": "ok", "device": DEVICE, "pipe_loaded": cz_pipeline._BASE_PIPE is not None,
                "offload": cz_pipeline.OFFLOAD_MODE, "idle_timeout": idle_timeout}

    @app.get("/models")
    def models():
        return {"esrgan_dir": cz_esrgan.ESRGAN_DIR, "models": list_esrgan_models()}

    @app.post("/unload")
    def unload():
        with lock:
            free_vram()
        return {"status": "unloaded"}

    @app.post("/upscale")
    def upscale(req: UpscaleReq):
        if not os.path.isfile(req.input):
            raise HTTPException(status_code=400, detail=f"input not found: {req.input}")
        avail = list_esrgan_models()
        if not avail:
            raise HTTPException(status_code=400, detail=f"no ESRGAN model in {cz_esrgan.ESRGAN_DIR}")
        # preset (s'il est fourni) sert de base; sinon les champs de la requete.
        p = PRESETS.get(req.preset or "Custom") or {}
        def pick(name, val):
            return p.get(name, val)
        model = req.model if req.model in avail else avail[0]
        with lock:
            state["last"] = time.time()
            set_offload_mode(pick("cpu_offload", req.cpu_offload))
            img = Image.open(req.input)
            result, t = process_one(
                img, model, pick("factor", req.factor), pick("denoise", req.denoise),
                pick("steps", req.steps), req.prompt, req.seed,
                pick("tile", req.tile), pick("overlap", req.overlap),
                refine_tile=pick("refine_tile", req.refine_tile),
                refine_overlap=pick("refine_overlap", req.refine_overlap),
            )
            _srcbase = os.path.splitext(os.path.basename(req.input))[0]
            dst = build_output_path(req.input, req.save_mode, req.output_dir, req.output_format,
                                    tag=f"{_srcbase}_upscaled", seed=req.seed, size=result.size)
            if dst:
                save_image(result, dst, req.output_format)
            state["last"] = time.time()
        return {"output": os.path.abspath(dst) if dst else None,
                "size": list(result.size),
                "esrgan_s": round(t.get("esrgan", 0.0), 2),
                "refine_s": round(t.get("refine", 0.0), 2),
                "total_s": round(t.get("esrgan", 0.0) + t.get("refine", 0.0), 2)}

    def _idle_watch():
        period = min(30, max(5, idle_timeout // 4)) if idle_timeout > 0 else 30
        while True:
            time.sleep(period)
            if idle_timeout > 0 and cz_pipeline._BASE_PIPE is not None and (time.time() - state["last"]) > idle_timeout:
                with lock:
                    if cz_pipeline._BASE_PIPE is not None and (time.time() - state["last"]) > idle_timeout:
                        free_vram()
                        print(f"[serve] model unloaded after {idle_timeout}s idle", file=sys.stderr)

    if idle_timeout and idle_timeout > 0:
        threading.Thread(target=_idle_watch, daemon=True).start()
    print(f"[serve] crispz on http://{host}:{port}  (idle unload: {idle_timeout}s)", file=sys.stderr)
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


# ----------------------------------------------------------------------------
# CLI (mode batch / scripting)
# ----------------------------------------------------------------------------
def _xyz_cli_apply(name, value, p, base_ms):
    """Applique la valeur d'un axe cote CLI: kind=val -> cle abstraite du dict p
    (spec['param']); modeles/sampler via les setters; Performance/S/R comme dans l'UI."""
    from cz_ui import _XYZ_AXES, PERFORMANCE, resolve_checkpoint, ZIMAGE_BASE_REPOS
    spec = _XYZ_AXES[name]
    kind = spec.get("kind")
    if kind == "val":
        p[spec["param"]] = value
    elif kind == "ms" and spec.get("key") == "sampler":
        set_sampler(value)
    elif kind == "ms":
        set_schedule(value)
    elif kind == "checkpoint":
        if value in ZIMAGE_BASE_REPOS:
            set_zimage_transformer("")
            set_zimage_model(value)
        else:
            set_zimage_transformer(resolve_checkpoint(value))
    elif kind == "lora_weight":
        set_loras([(path, float(value)) for path, _w in (base_ms.get("loras") or [])])
    elif kind == "performance":
        st, g = PERFORMANCE[value]
        p["gen_steps"], p["guidance"] = int(st), float(g)
    elif kind == "sr":
        term = spec["_term"]
        if str(value) != term:
            p["prompt"] = str(p["prompt"]).replace(term, str(value))


def _xyz_cli_run(args, parser, model_name):
    """Grille X/Y/Z en CLI (--txt2img --xyz 'Axe=v1,v2' x1-3): un rendu par combo,
    sauvegarde habituelle, puis planche(s) annotee(s) via cz_ui._xyz_assemble.
    Ctrl+C = planche partielle avec les cellules deja rendues."""
    from cz_ui import (_XYZ_AXES, _xyz_parse_values, _xyz_validate_axis, _xyz_match,
                       _xyz_assemble, XYZ_FEATURE_ENABLED, XYZ_MAX_JOBS, XYZ_THUMB,
                       _gen_meta)
    if not XYZ_FEATURE_ENABLED:
        parser.error("--xyz: xyz_grid is disabled in config.txt")
    if len(args.xyz) > 3:
        parser.error("--xyz can be used at most 3 times (X, Y, Z)")
    ax_names = [k for k in _XYZ_AXES if k != "(none)"]
    upscale_only = {"ESRGAN model", "Factor", "Denoise", "Tile", "Refine tile"}
    fake_vals = [None] * 36
    fake_vals[0] = args.prompt
    base_ms = {"loras": list(cz_pipeline.LORAS)}
    axes = []
    for spec in args.xyz:
        if "=" not in spec:
            parser.error(f"--xyz expects AXIS=v1,v2,... (got: {spec})")
        name_raw, _, vals_raw = spec.partition("=")
        name, err = _xyz_match(name_raw.strip(), ax_names)
        if err:
            parser.error(f"--xyz axis: {err}")
        if name in upscale_only and not args.upscale:
            parser.error(f"--xyz {name} only affects the upscale pass -> add --upscale")
        values, err = _xyz_validate_axis(name, _xyz_parse_values(vals_raw), fake_vals, base_ms)
        if err:
            parser.error(f"--xyz: {err}")
        if _XYZ_AXES[name].get("kind") == "sr":
            _XYZ_AXES[name]["_term"] = str(values[0])
        axes.append((name, values))
    names = [n for n, _v in axes]
    if len(set(names)) != len(names):
        parser.error("--xyz: each axis must vary a different parameter")
    total = 1
    for _n, v in axes:
        total *= len(v)
    if total > XYZ_MAX_JOBS:
        parser.error(f"--xyz: {total} combos > max_jobs ({XYZ_MAX_JOBS}); reduce the lists "
                     f"(or raise xyz_grid.max_jobs in config.txt)")
    (xn, xv) = axes[0]
    (yn, yv) = axes[1] if len(axes) > 1 else (None, [None])
    (zn, zv) = axes[2] if len(axes) > 2 else (None, [None])
    gid = time.strftime("%Y%m%d_%H%M%S")
    cells, done, quiet = {}, 0, (args.quiet or args.print_output)
    try:
        for iz, z in enumerate(zv):
            for iy, y in enumerate(yv):
                for ix, x in enumerate(xv):
                    p = {"prompt": args.prompt, "gen_steps": args.gen_steps,
                         "guidance": args.guidance, "seed": args.seed,
                         "esrgan": model_name, "factor": args.factor,
                         "denoise": args.denoise, "tile": args.tile,
                         "refine_tile": args.refine_tile}
                    combo = [(xn, x)] + ([(yn, y)] if yn else []) + ([(zn, z)] if zn else [])
                    for aname, aval in combo:
                        _xyz_cli_apply(aname, aval, p, base_ms)
                    set_guidance(p["guidance"])
                    label = " ".join(f"{n}={v}" for n, v in combo)
                    _log(f"combo {done + 1}/{total}: {label}", mod="xyz")
                    img, t = txt2img_run(
                        p["prompt"], args.gen_width, args.gen_height, p["gen_steps"],
                        p["seed"], args.negative, upscale=args.upscale,
                        esrgan_model=p["esrgan"], factor=p["factor"], denoise=p["denoise"],
                        steps=args.steps, tile=p["tile"], overlap=args.overlap,
                        refine_tile=p["refine_tile"], refine_overlap=args.refine_overlap,
                        refine_first=args.refine_first)
                    if args.save_mode != "display":
                        try:
                            dst = build_output_path(None, args.save_mode, args.output_dir,
                                                    args.output_format, tag="xyz",
                                                    seed=p["seed"], size=img.size,
                                                    index=done + 1)
                            if dst:
                                save_image(img, dst, args.output_format, meta=_gen_meta(
                                    "xyz", p["prompt"], args.negative, p["seed"],
                                    p["gen_steps"], p["guidance"], img.size,
                                    extra={"xyz": label}))
                        except Exception as e:
                            _log(f"save failed ({e}); cell kept for the sheet", mod="xyz")
                    thumb = img.copy()
                    thumb.thumbnail((XYZ_THUMB, XYZ_THUMB), Image.LANCZOS)
                    cells[(ix, iy, iz)] = thumb
                    done += 1
                    if not quiet:
                        print(f"[{done}/{total}] {label}  ({t['txt2img']:.1f}s)")
    except KeyboardInterrupt:
        _log(f"interrupted after {done}/{total}; assembling partial sheet(s)", mod="xyz")
    meta = {"gid": gid, "x": (xn, [str(v) for v in xv]),
            "y": (yn, [str(v) for v in yv]) if yn else None,
            "z": (zn, [str(v) for v in zv]) if zn else None,
            "out_dir": args.output_dir}
    for s in _xyz_assemble(meta, cells, thumb=XYZ_THUMB):
        print(os.path.abspath(s))
    return 0


def cli_main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="crispz-studio CLI (txt2img + upscale). No args: launches the Gradio UI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cli", action="store_true", help="Force CLI mode (otherwise: launches the UI)")
    # Sources : fichier, glob, dossier
    parser.add_argument("-i", "--input", help="Image, glob (in/*.png) or source FOLDER for batch")
    parser.add_argument("--input-folder", help="Explicit alias for the batch folder (otherwise -i works too)")
    # Sortie
    parser.add_argument("-o", "--output",
                        help="Output file (single mode, overrides auto naming). "
                             "If a folder: equivalent to --save-mode local --output-dir <that folder>.")
    parser.add_argument("--save-mode", choices=["display", "local", "alongside", "custom"],
                        default=DEFAULT_SAVE_MODE,
                        help="display=no save | local=output_dir relative to project | "
                             "alongside=same folder as the source | custom=output_dir as-is")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Output folder for --save-mode local/custom")
    parser.add_argument("--output-format", choices=list(SUPPORTED_FORMATS),
                        default=DEFAULT_OUTPUT_FORMAT, help="Output format (png/webp/jpg)")
    # Pipeline
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL,
                        help="ESRGAN model (file in ESRGAN_DIR). Fallback: first found.")
    parser.add_argument("--factor", type=float, default=DEFAULT_FACTOR, help="Net upscale factor")
    parser.add_argument("--denoise", type=float, default=DEFAULT_DENOISE, help="Z-Image strength (0 = ESRGAN only)")
    parser.add_argument("--no-refine", action="store_true",
                        help="Skip the (slow) Z-Image img2img refine pass -> ESRGAN upscale only "
                             "(shortcut for --denoise 0).")
    parser.add_argument("--refine-first", action="store_true",
                        help="Refine at native resolution THEN ESRGAN upscale (~4-16x faster "
                             "refine), instead of ESRGAN-then-refine.")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Diffusion steps")
    parser.add_argument("--prompt", default="", help="Optional prompt")
    parser.add_argument("--seed", type=int, default=-1, help="Seed (-1 = random)")
    parser.add_argument("--tile", type=int, default=DEFAULT_TILE, help="ESRGAN tile size (0 = disabled)")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP, help="ESRGAN tiling overlap")
    parser.add_argument("--refine-tile", type=int, default=DEFAULT_REFINE_TILE,
                        help="Z-Image diffusion tile size (0 = whole image). >0 tiles the "
                             "refine pass: caps VRAM and enables 4K+ without seams. Try 1024-1280.")
    parser.add_argument("--refine-overlap", type=int, default=DEFAULT_REFINE_OVERLAP,
                        help="Overlap (feather) of the Z-Image diffusion tiles")
    parser.add_argument("--cpu-offload", choices=list(cz_pipeline.OFFLOAD_CHOICES), default="none",
                        help="CPU offload of the diffusion pass (VRAM). none=all in VRAM | "
                             "model=offload per submodule (good tradeoff) | "
                             "sequential=more aggressive, slower. Requires accelerate.")
    parser.add_argument("--guidance", type=float, default=0.0,
                        help="CFG guidance scale. 0 for Z-Image Turbo (default). "
                             "Z-Image Base needs ~3.5-5 (and ~20+ steps).")
    parser.add_argument("--sampler", choices=list(SAMPLER_CHOICES), default=None,
                        help="Sampler: euler (native flow, default) or unipc (UniPC multistep). "
                             "Default from config default_sampler. (DPM++/DPM2a unavailable: "
                             "Z-Image forces custom sigmas.)")
    parser.add_argument("--schedule", choices=list(SCHEDULE_CHOICES), default=None,
                        help="Sigma schedule (ComfyUI-style): sgm_uniform (native Z-Image, "
                             "default), beta, karras, exponential. Default from default_schedule.")
    parser.add_argument("--no-esrgan", action="store_true",
                        help="img2img only: skip the ESRGAN upscale, just run the Z-Image refine "
                             "on the input at native size (no enlargement).")
    parser.add_argument("--preset", choices=list(PRESETS), default="Custom",
                        help="Use-case preset (auto settings). Explicit flags override it.")
    # Text -> Image (txt2img)
    parser.add_argument("--txt2img", action="store_true",
                        help="Generate an image from --prompt (Z-Image txt2img) instead of "
                             "reading -i. Add --upscale to also run ESRGAN + refine.")
    parser.add_argument("--gen-width", type=int, default=1024, help="txt2img width (mult. of 16)")
    parser.add_argument("--gen-height", type=int, default=1024, help="txt2img height (mult. of 16)")
    parser.add_argument("--gen-steps", type=int, default=8, help="txt2img steps (Z-Image Turbo)")
    parser.add_argument("--negative", default="", help="Negative prompt (txt2img)")
    parser.add_argument("--upscale", action="store_true",
                        help="In --txt2img: run the ESRGAN + refine upscale on the generated image")
    parser.add_argument("--xyz", action="append", default=[], metavar="AXIS=V1,V2,...",
                        help="X/Y/Z grid (with --txt2img): vary a parameter across values; repeat "
                             "up to 3 times for X, Y, Z axes. Same axes/rules as the UI grid "
                             "(quotes protect commas; Prompt S/R: first value = search term). Ends "
                             "with annotated contact sheet(s) in <output>/xyz_<timestamp>/.")
    # Nouvelles features en CLI (mêmes que l'UI)
    parser.add_argument("--lora", action="append", default=[], metavar="NAME[:WEIGHT]",
                        help="Apply a LoRA (file in the loras dir, or a path), optional :weight "
                             "(default 1.0). Repeatable, e.g. --lora a.safetensors:0.8 --lora b")
    parser.add_argument("--loras-dir", help="Override the LoRA folder (for --lora names)")
    parser.add_argument("--remove-bg", action="store_true",
                        help="Remove the background of -i (rembg) -> transparent PNG, then exit.")
    parser.add_argument("--reframe", metavar="W:H",
                        help="Outpaint -i to a target aspect ratio (e.g. 16:9 / 9:16), then exit. "
                             "--prompt guides the fill; --gen-steps sets the steps.")
    parser.add_argument("--faceswap-src", metavar="PATH",
                        help="Post-process: swap the face in the (txt2img/reframe) result with this "
                             "source face. Needs insightface + an inswapper model.")
    parser.add_argument("--vision-mix", nargs="+", metavar="IMG",
                        help="Describe these reference images with an Ollama vision model and merge "
                             "them into ONE prompt, then run txt2img from it.")
    parser.add_argument("--ollama-model", help="Ollama vision model for --vision-mix "
                                               "(default: first detected vision model)")
    # Server (stage 3)
    parser.add_argument("--serve", action="store_true",
                        help="Run a persistent HTTP server (lazy model load + idle unload) "
                             "instead of the UI/one-shot. Requires fastapi + uvicorn.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (--serve)")
    parser.add_argument("--port", type=int, default=7861, help="Server port (--serve)")
    parser.add_argument("--idle-timeout", type=int, default=300,
                        help="Seconds of inactivity before the server frees VRAM (0 = never)")
    # Chemins config / Z-Image
    parser.add_argument("--esrgan-dir", help="Override ESRGAN_DIR for this run")
    parser.add_argument("--zimage-model",
                        help="Override Z-Image: HF repo, diffusers folder, OR a single-file "
                             ".safetensors (Civitai) used as the transformer (VAE+encoder from base).")
    parser.add_argument("--zimage-transformer",
                        help="Single-file .safetensors transformer override (Civitai), keeping "
                             "the VAE + Qwen3 encoder from --zimage-model / the base repo.")
    parser.add_argument("--save-paths", action="store_true",
                        help="Save --esrgan-dir and --zimage-model to preferences.json")
    # Reports
    parser.add_argument("--list-models", action="store_true", help="List ESRGAN models then exit")
    parser.add_argument("--time-log", default=None,
                        help="If set, append the time of each run to this file (TSV)")
    parser.add_argument("--quiet", action="store_true", help="Reduce stdout verbosity")
    parser.add_argument("--log-level", choices=["quiet", "info", "debug"], default=None,
                        help="Console log level on stderr. debug = full params/state (dev). "
                             "Default from env CRISPZ_LOG_LEVEL or 'info'.")
    parser.add_argument("--report-vram", action="store_true",
                        help="Print the run VRAM peak on stderr (line '[VRAM] ...'). "
                             "Used to size coexistence with Fooocus.")
    parser.add_argument("--print-output", action="store_true",
                        help="Print ONLY the absolute output path on stdout (one per saved "
                             "image), nothing else. For external integration (Fooocus). "
                             "Implies a silent stdout; the VRAM peak stays on stderr.")
    args = parser.parse_args(argv)
    apply_preset_to_args(args, argv if argv is not None else sys.argv[1:])
    # Raccourci: --no-refine == --denoise 0 (saute la passe img2img lente).
    if args.no_refine:
        args.denoise = 0.0

    if args.log_level:
        cz_core.LOG_LEVEL = _parse_log_level(args.log_level)
    elif args.quiet:
        cz_core.LOG_LEVEL = 0
    _log(f"log level = {cz_core.LOG_LEVEL} (0=quiet 1=info 2=debug)")

    if args.esrgan_dir:
        set_esrgan_dir(args.esrgan_dir)
    if args.zimage_model:
        set_zimage_model(args.zimage_model)
    if args.zimage_transformer:
        set_zimage_transformer(args.zimage_transformer)
    set_offload_mode(args.cpu_offload)
    set_guidance(args.guidance)
    if args.sampler:
        set_sampler(args.sampler)
    if args.schedule:
        set_schedule(args.schedule)

    # LoRA(s) en CLI: --lora NAME[:WEIGHT] (repetable)
    if args.loras_dir:
        set_loras_dir(args.loras_dir)
    if args.lora:
        slots = []
        for spec in args.lora:
            head, _, tail = spec.rpartition(":")
            try:
                slots.append((head, float(tail)))   # NAME:WEIGHT
            except ValueError:
                slots.append((spec, cz_pipeline.LORA_WEIGHT))    # NAME (poids par defaut)
        set_loras(slots)

    def _maybe_faceswap(img):
        if args.faceswap_src and os.path.isfile(args.faceswap_src):
            try:
                return _faceswap(img, Image.open(args.faceswap_src))
            except Exception as e:
                _log(f"faceswap skipped: {e}")
        return img

    # --remove-bg : detoure -i puis termine
    if args.remove_bg:
        if not args.input or not os.path.isfile(args.input):
            parser.error("--remove-bg requires -i <image>")
        res = _remove_bg(Image.open(args.input))
        sm = args.save_mode if args.save_mode != "display" else "local"
        base = os.path.splitext(os.path.basename(args.input))[0]
        dst = args.output if (args.output and not os.path.isdir(args.output)) else \
            build_output_path(args.input, sm, args.output_dir, "png", tag=f"{base}_nobg")
        save_image(res, dst, "png")
        print(os.path.abspath(dst))
        return 0

    # --reframe W:H : outpaint -i puis termine
    if args.reframe:
        if not args.input or not os.path.isfile(args.input):
            parser.error("--reframe requires -i <image>")
        try:
            rw, rh = [int(x) for x in str(args.reframe).split(":")]
        except Exception:
            parser.error("--reframe expects W:H, e.g. 16:9")
        res = _maybe_faceswap(outpaint(Image.open(args.input), rw, rh, args.prompt,
                                       args.gen_steps, args.seed))
        sm = args.save_mode if args.save_mode != "display" else "local"
        base = os.path.splitext(os.path.basename(args.input))[0]
        dst = args.output if (args.output and not os.path.isdir(args.output)) else \
            build_output_path(args.input, sm, args.output_dir, args.output_format,
                              tag=f"{base}_reframe", seed=args.seed, size=res.size)
        save_image(res, dst, args.output_format)
        print(os.path.abspath(dst))
        return 0

    # --vision-mix IMG... : decrit + fusionne en un prompt, puis txt2img
    if args.vision_mix:
        imgs = [Image.open(p) for p in args.vision_mix if os.path.isfile(p)]
        if not imgs:
            parser.error("--vision-mix: no valid image path")
        vmodel = args.ollama_model or (_ollama_vision_models() or [None])[0]
        if not vmodel:
            parser.error("--vision-mix needs an Ollama vision model (none detected)")
        caps = [_ollama_describe(im, vmodel) for im in imgs]
        args.prompt = _ollama_compose(caps, vmodel)
        args.txt2img = True
        if not (args.quiet or args.print_output):
            print(f"[vision-mix] {vmodel} -> {args.prompt}", file=sys.stderr)

    if args.serve:
        return serve_main(args.host, args.port, args.idle_timeout)

    # Mode txt2img (Text -> Image, + upscale optionnel)
    if args.txt2img:
        if not args.prompt:
            parser.error("--txt2img requires --prompt")
        os.makedirs(cz_esrgan.ESRGAN_DIR, exist_ok=True)
        model_name = None
        if args.upscale:
            avail = list_esrgan_models()
            if not avail:
                parser.error(f"--upscale needs an ESRGAN model in {cz_esrgan.ESRGAN_DIR}")
            model_name = args.model if args.model in avail else avail[0]
        if args.xyz:
            return _xyz_cli_run(args, parser, model_name)
        if args.report_vram:
            _reset_vram_peak()
        result, t = txt2img_run(
            args.prompt, args.gen_width, args.gen_height, args.gen_steps, args.seed,
            args.negative, upscale=args.upscale, esrgan_model=model_name,
            factor=args.factor, denoise=args.denoise, steps=args.steps,
            tile=args.tile, overlap=args.overlap,
            refine_tile=args.refine_tile, refine_overlap=args.refine_overlap,
            refine_first=args.refine_first)
        result = _maybe_faceswap(result)
        # Sortie : -o fichier, sinon output_dir (sauf save-mode display)
        dst = None
        if args.output and not (os.path.isdir(args.output) or args.output.endswith(("/", "\\"))):
            dst = args.output
            os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
        elif args.save_mode != "display":
            out_dir = args.output or args.output_dir
            if not os.path.isabs(out_dir):
                out_dir = os.path.join(HERE, out_dir)
            os.makedirs(out_dir, exist_ok=True)
            ext = args.output_format if args.output_format in SUPPORTED_FORMATS else "png"
            dst = os.path.join(out_dir, f"txt2img.{ext}")
        if dst:
            save_image(result, dst, args.output_format)
            if args.print_output:
                print(os.path.abspath(dst))
        quiet = args.quiet or args.print_output
        if not quiet:
            parts = [f"txt2img {result.size[0]}x{result.size[1]} in {t['txt2img']:.1f}s"]
            if args.upscale:
                parts.append(f"esrgan {t['esrgan']:.1f}s + refine {t['refine']:.1f}s")
            if dst:
                parts.append(f"-> {dst}")
            print("  |  ".join(parts))
        if args.report_vram:
            _report_vram()
        return 0

    if args.save_paths:
        _save_prefs_keys({"esrgan_dir": cz_esrgan.ESRGAN_DIR, "zimage_model": cz_pipeline.BASE_REPO})
        print(f"Saved to {PREFS_PATH}: esrgan_dir={cz_esrgan.ESRGAN_DIR}, zimage_model={cz_pipeline.BASE_REPO}")
        if not args.input and not args.input_folder:
            return 0

    os.makedirs(cz_esrgan.ESRGAN_DIR, exist_ok=True)
    models = list_esrgan_models()

    if args.list_models:
        if not models:
            print(f"No model in {cz_esrgan.ESRGAN_DIR}")
        else:
            for m in models:
                print(m)
        return 0

    # Pas de --cli et pas d'entree -> UI
    if not args.cli and not args.input and not args.input_folder:
        _disable_brotli()  # evite le bug h11 'Content-Length' a l'envoi des resultats
        # + dossiers modeles (LoRA/checkpoints) pour servir leurs previews dans l'Asset Browser
        _model_dirs = [p for p in (getattr(cz_pipeline, "LORAS_DIR", ""),
                                   getattr(cz_pipeline, "CHECKPOINTS_DIR", ""),
                                   getattr(cz_pipeline, "CHECKPOINTS_EXTRA_DIR", ""))
                       if p and os.path.isdir(p)]
        build_ui().launch(allowed_paths=[os.path.join(HERE, "styles", "samples"),
                                         os.path.join(HERE, "tags"),
                                         _ab_resolve_dir(DEFAULT_OUTPUT_DIR)] + _model_dirs)
        return 0

    if not models and not args.no_esrgan:
        parser.error(f"No ESRGAN model in {cz_esrgan.ESRGAN_DIR} (or use --no-esrgan for img2img only)")

    model_name = (args.model if args.model in models else (models[0] if models else None))

    if args.report_vram:
        _reset_vram_peak()

    # Resoudre les entrees : dossier > glob > fichier unique
    source_folder = args.input_folder
    if not source_folder and args.input and os.path.isdir(args.input):
        source_folder = args.input
        args.input = None

    # --output (compat) : si c'est un dossier, equivalent a --save-mode local --output-dir <dossier>
    save_mode = args.save_mode
    output_dir = args.output_dir
    explicit_output_file = None
    if args.output:
        if os.path.isdir(args.output) or args.output.endswith(("/", "\\")):
            save_mode = "custom" if os.path.isabs(args.output) else "local"
            output_dir = args.output
        else:
            explicit_output_file = args.output
            save_mode = "custom"

    # --print-output: stdout reserve aux chemins de sortie (contrat machine).
    # Le pic VRAM, lui, reste sur stderr et n'est donc pas pollue.
    quiet = args.quiet or args.print_output

    # Mode batch dossier
    if source_folder:
        last_result, last_source, report = run(
            None, source_folder, model_name, args.factor, args.denoise, args.steps,
            args.prompt, args.seed, args.tile, args.overlap,
            save_mode=save_mode, output_dir=output_dir,
            output_format=args.output_format, time_log_path=args.time_log,
            print_output=args.print_output,
            refine_tile=args.refine_tile, refine_overlap=args.refine_overlap,
            do_esrgan=not args.no_esrgan, refine_first=args.refine_first,
        )
        if not quiet:
            print(report)
        if args.report_vram:
            _report_vram()
        return 0

    # Mode unique : glob possible
    paths = sorted(glob.glob(args.input)) if any(c in args.input for c in "*?[") else [args.input]
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        parser.error(f"No file matches {args.input}")

    # Si plusieurs fichiers via glob, on les passe un par un
    for p in paths:
        if not quiet:
            print(f"-> {p}")
        img = Image.open(p)
        # explicit_output_file ne s'applique qu'au premier fichier
        if explicit_output_file and len(paths) == 1:
            result, t = process_one(img, model_name, args.factor, args.denoise, args.steps,
                                    args.prompt, args.seed, args.tile, args.overlap,
                                    refine_tile=args.refine_tile, refine_overlap=args.refine_overlap,
                                    do_esrgan=not args.no_esrgan, refine_first=args.refine_first)
            os.makedirs(os.path.dirname(os.path.abspath(explicit_output_file)) or ".", exist_ok=True)
            save_image(result, explicit_output_file, args.output_format)
            if args.print_output:
                print(os.path.abspath(explicit_output_file))
            _append_time_log(args.time_log, p, explicit_output_file, t, "custom", args.output_format)
            if not quiet:
                print(_format_timings(t, src_path=p, dst_path=explicit_output_file))
        else:
            # mode standard: build_output_path applique le save_mode
            last_result, last_source, report = run(
                p, None, model_name, args.factor, args.denoise, args.steps,
                args.prompt, args.seed, args.tile, args.overlap,
                save_mode=save_mode, output_dir=output_dir,
                output_format=args.output_format, time_log_path=args.time_log,
                print_output=args.print_output,
                refine_tile=args.refine_tile, refine_overlap=args.refine_overlap,
                do_esrgan=not args.no_esrgan, refine_first=args.refine_first,
            )
            if not quiet:
                print(report)
    if args.report_vram:
        _report_vram()
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
