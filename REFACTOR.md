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
- [x] **Step 6 — `cz_esrgan.py` / `cz_face.py`**: leaf compute. Mutable caches
  (_ESRGAN_CACHE / _FACE_APP/_FACE_SWAPPER/_FACE_RESTORE_SESSION/_BLIP) + setters must
  move WITH the functions so bare refs stay valid.
  - [x] **6a — `cz_esrgan.py`**: ESRGAN_DIR + _ESRGAN_CACHE + set_esrgan_dir +
    list_esrgan_models + load_esrgan + _pil_to_tensor/_tensor_to_pil + esrgan_upscale.
    App reads the live `cz_esrgan.ESRGAN_DIR` (~13 sites). Validated: `is`-identity +
    setter clears cache + build_ui + smoke 19/19. (in-browser pass pending)
  - [x] **6b — `cz_face.py`**: _BLIP/_local_caption + _FACE_APP/_FACE_SWAPPER/
    _FACE_RESTORE_SESSION + faceswap/restore + set_faceswap_restore + _remove_bg.
    app keeps thin wrappers: `set_faceswap_restore` delegates + mirrors
    cz_face.FACESWAP_RESTORE back into app (smoke/UI read app.FACESWAP_RESTORE live);
    `_faceswap` passes app's CHECKPOINTS_DIR (still here until step 7) to
    cz_face._faceswap(checkpoints_dir=...). _local_caption/_remove_bg re-exported.
    Validated: is-identity + live app<->cz_face sync + build_ui + smoke 19/19.
    (in-browser pass pending)
- [ ] **Step 7 — `cz_models.py` + `cz_generate.py`** (HIGH risk): BASE_REPO/
  ZIMAGE_TRANSFORMER/LORAS/OFFLOAD_MODE/GUIDANCE + pipe caches + setters + generate.
  Keep them in ONE module so the many bare refs stay intra-module.
- [ ] **Step 8 — `cz_ui.py` + `cz_cli.py`** (HIGH risk): build_ui wiring + argparse/serve.

Recommend: do steps 6-8 with an in-browser pass each (generate, switch model, LoRA,
faceswap, wildcards) since smoke does not exercise the stateful generation/UI paths.

## HOW TO RESUME (fresh session) — read this first
Modules already extracted (DONE, green): cz_assets, cz_core, cz_ollama, cz_imageio,
cz_assetbrowser, cz_prompt. app.py is 3017 lines and imports them. Remaining = the
mutable-state core: ESRGAN, FaceSwap, models/generate, ui/cli.

Pattern that worked (repeat it):
1. Create `cz_<name>.py`. It imports from cz_core (and other DONE modules), NEVER app.
2. Move the functions AND their mutable globals + setters TOGETHER so intra-module refs
   stay bare. Only cross-module mutable state is read as `cz_<name>.NAME` from app.
3. In app.py: delete the moved block, add `from cz_<name> import (...)` (+ `import
   cz_<name>` if you must read a mutable global like `cz_esrgan.ESRGAN_DIR`).
4. DO NOT use Edit replace_all on `ESRGAN_DIR` (it is a substring of DEFAULT_ESRGAN_DIR)
   or on `BASE_REPO`/`GUIDANCE` (substrings of DEFAULT_*). Edit each call site by hand
   with enough surrounding context.
5. Validate every step BEFORE commit:
   - `python -m py_compile app.py cz_<name>.py`
   - import + `is`-identity asserts for moved names + `app.build_ui()`
   - `.venv\Scripts\python tools\smoke_test.py`  (must stay 19/19)
   - IN-BROWSER (this is the new safety net smoke can't give): run.bat, then
     generate 1 image, switch checkpoint, apply a LoRA, run FaceSwap, use a wildcard.
   - one commit per module (no "Co-Authored-By"; see CLAUDE.local.md).

Step 6 = cz_esrgan.py (ESRGAN_DIR + _ESRGAN_CACHE + set_esrgan_dir + list_esrgan_models
+ _load/_esrgan_upscale) AND cz_face.py (_FACE_APP/_FACE_SWAPPER/_FACE_RESTORE_SESSION/
_BLIP + setters + detect/swap/restore). ESRGAN_DIR is read in ~13 places (generate,
upscale, API, CLI, build_ui) -> read as cz_esrgan.ESRGAN_DIR there.
Step 7 = cz_models.py + cz_generate.py: BASE_REPO/ZIMAGE_TRANSFORMER/LORAS/OFFLOAD_MODE/
GUIDANCE + _BASE_PIPE/_DERIVED/_LOADED_KEY caches + _ensure_base/get_pipe + checkpoint/
LoRA/transformer setters + generate/txt2img/img2img/outpaint/inpaint/omni + _gen_meta.
Keep ALL of this in ONE module (cz_pipeline.py is fine) so the many bare refs stay valid.
Step 8 = move build_ui (+ handlers) and cli_main/serve out last; app.py becomes a thin
entrypoint `from cz_cli import main`.

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
