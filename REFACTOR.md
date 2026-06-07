# crispz-studio — Refactor plan (app.py modularization)

`app.py` is ~3.5k lines. It works and is covered by `smoke_test.py`, but should be
split for maintainability. This is a **staged** plan: each step keeps `app.py`
importing the moved names so nothing else changes, and each step is validated with
`tools/check.bat` (py_compile + smoke) **and** an in-browser pass (see VALIDATION.md).

Do NOT do it all at once. One module per PR, smoke-green between each.

## Progress (app.py: 3601 -> 3115 lines so far)
- [x] **Step 0 — `cz_assets.py`**: static strings (ASSET_BROWSER_HTML/CZ_JS/FOOOCUS_CSS).
- [x] **Step 1 — `cz_core.py`**: CONFIG/paths/DEFAULT_*/profiles/instructions/prefs/
  DEVICE/DTYPE/logging + `_pil_to_b64_jpeg`. LOG_LEVEL read via `cz_core.LOG_LEVEL`.
- [x] **Step 2 — `cz_ollama.py`**: _ollama_* + OLLAMA_URL/KEEP_ALIVE/CPU + _ollama_gen_opts.
- [x] **Step 3 — `cz_imageio.py`**: filenames/paths/save_image/_exif/_read_image_meta/
  _list_output_files. (`_gen_meta` stays in app — reads model state.)
- [x] **Step 4 — `cz_assetbrowser.py`**: ab_reindex + thumbs + delete_asset.
  Each step validated: `is`-identity of moved names + build_ui + smoke 19/19.
- [x] **Step 5 — `cz_prompt.py`**: styles + wildcards (STYLES/_apply_styles/_pick_styles
  + WILDCARDS_DIR/set_wildcards_dir/_apply_wildcards/list_wildcards). App reads the live
  `cz_prompt.WILDCARDS_DIR`. Validated: style applied, setter propagation, smoke 19/19.
- [ ] **Step 6 — `cz_esrgan.py` / `cz_face.py`**: leaf compute. Mutable caches
  (_ESRGAN_CACHE / _FACE_APP/_FACE_SWAPPER/_FACE_RESTORE_SESSION/_BLIP) + setters must
  move WITH the functions so bare refs stay valid.
- [ ] **Step 7 — `cz_models.py` + `cz_generate.py`** (HIGH risk): BASE_REPO/
  ZIMAGE_TRANSFORMER/LORAS/OFFLOAD_MODE/GUIDANCE + pipe caches + setters + generate.
  Keep them in ONE module so the many bare refs stay intra-module.
- [ ] **Step 8 — `cz_ui.py` + `cz_cli.py`** (HIGH risk): build_ui wiring + argparse/serve.

Recommend: do steps 5-8 with an in-browser pass each (generate, switch model, LoRA,
faceswap, wildcards) since smoke does not exercise the stateful generation/UI paths.

## Target layout
```
crispz_studio/
  config.py     # CONFIG load, DEFAULT_*, paths, profiles, _log/_dbg, LOG_LEVEL
  models.py     # _ensure_base/get_pipe, checkpoints/LoRA/transformer, list_*, FP8 filter
  generate.py   # generate / txt2img_run / img2img run / outpaint / inpaint / omni
  esrgan.py     # spandrel load + tiled upscale
  face.py       # faceswap + GFPGAN restore
  ollama.py     # describe / improve / compose / vision-model detect
  prompt.py     # styles, wildcards, _pick_styles, _apply_*
  imageio.py    # save_image, metadata (PNG/EXIF/sidecar), filenames, gallery list
  assetbrowser.py # SPA + ab_reindex
  ui.py         # build_ui (Gradio) + handlers
  cli.py        # argparse + cli_main + serve
app.py          # thin entrypoint -> from crispz_studio.cli import main
```

## Ordering (lowest risk first)
1. **config.py** — pure constants/loaders, no deps on the rest. Everything imports it.
2. **imageio.py** — save/metadata/filenames (depends on config only).
3. **prompt.py** — styles/wildcards (config + STYLES).
4. **ollama.py** — HTTP + describe/improve/compose (config + imageio b64).
5. **esrgan.py**, **face.py** — leaf compute (config + numpy/torch).
6. **models.py**, **generate.py** — pipelines (config + esrgan).
7. **assetbrowser.py** — SPA + reindex (config + imageio).
8. **ui.py** + **cli.py** last (wire everything).

## Gotchas
- Many helpers read module globals (CONFIG, HERE, DEVICE, BASE_REPO, LORAS, OFFLOAD_MODE…).
  Put the **mutable runtime state** (BASE_REPO, ZIMAGE_TRANSFORMER, LORAS, OFFLOAD_MODE,
  GUIDANCE…) in a small `state.py` (or config.py) and have setters there, so modules
  share one source of truth instead of duplicating globals.
- `_log`/`_dbg` must live in config.py (everyone uses them).
- Avoid circular imports: ui.py imports the rest; the rest must NOT import ui.py.
- Gradio component wiring stays entirely in ui.py.

## Safety net
- `tools/check.bat` / CI on every step.
- `smoke_test.py` exercises the moved functions via `import app` (keep the re-exports
  in app.py until the move is complete).
- A full in-browser pass (VALIDATION.md) after the UI move.
