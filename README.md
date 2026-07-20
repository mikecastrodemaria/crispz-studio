# crispz-studio

> Z-Image txt2img + upscaler/detailer studio (a Fooocus-style fork of
> [crispz](https://github.com/mikecastrodemaria/crispz)).
> Current version: **1.7.1** — see [CHANGELOG.md](CHANGELOG.md).

![crispz-studio — Z-Image creation + enhancement studio](assets/screenshot.png)

A standalone Z-Image **creation + enhancement** tool, **100% local**, no ComfyUI /
SwarmUI. On top of crispz's upscaler it adds:

- **Text → Image** (`ZImagePipeline`): generate from a prompt, with an optional
  **Upscale after generate** toggle (under the Generate button) that auto-chains each
  image through the ESRGAN + refine pipeline — no manual step. CLI equivalent:
  `--txt2img --upscale` (see README_CLI.md).
- **Image → Upscale** (the crispz pipeline): Real-ESRGAN + Z-Image refine, 4K tiling.
- **Job queue**: `+ Queue` snapshots ALL current settings (incl. model, LoRAs, sampler)
  into a labeled job list; `Run queue` chains them unattended (overnight batches with
  different models/settings). **Stop pauses the queue** — remaining jobs are kept. VRAM
  is purged automatically only when the model changes between jobs.
- **X/Y/Z grid**: vary 1–3 parameters (checkpoint, sampler, steps, guidance, denoise,
  ESRGAN model, LoRA file, LoRA weight, Performance preset, Prompt S/R…) — every combo becomes a
  queued job and the run ends with an **annotated contact sheet** per Z value (X columns ×
  Y rows) saved in the output folder and shown in the gallery.
- **Tag autocomplete** in the prompt/negative fields: suggestions as you type from tag
  CSVs (downloaded once into `tags/`; drop any `.csv` there to add a source) merged with
  your local `__wildcards__`. Dropdown under the caret, ↑/↓ + Tab/Enter, Escape;
  popularity-ranked, indexed (~sub-ms per keystroke).
- **Inpaint / Outpaint** (one tab, 3 modes): **Brush** repaints a painted mask ·
  **Expand sides** outpaints Left/Right/Top/Bottom (+ **Center**) ~30% per side ·
  **Reframe** to a new aspect ratio (**Contain** = keep the whole image and fill the
  borders, **Cover** = crop to fill). All bounded to the model's ~1 MP sweet spot (no
  pixel blow-up), with blurred-edge fill + feathered seams for clean blends, an optional
  **Auto-describe** (local captioner, no Ollama, automatic when the prompt is empty) to
  guide coherent fills, and an optional **Harmonize** pass (light img2img refine over the
  whole result) to unify grain/light and remove the "added zone" look. Steps follow the
  model.
- **Remove Background** (rembg) and **Face Swap** with optional **GFPGAN restore**.
- **Models**: one **Z-Image checkpoint** dropdown merging the official base repos
  (Turbo / Z-Image) with single-file `.safetensors` from a main **and** an optional
  extra folder, a **Transformer override** (diffusers repo/folder, e.g. Juggernaut-Z),
  and **multi-LoRA** (configurable **1–10 slots** + trigger words). Picking a model also
  auto-syncs the Performance preset. FP8 and INT8/INT4-quantized checkpoints are skipped
  (diffusers can't load them) - use the BF16/FP16 build.
- **Presets (Fooocus-style)** (Settings > ⭐ Presets): **save / load / update / delete**
  presets — a preset bundles prompt, styles, size, steps/CFG, sampler, checkpoint,
  transformer + LoRAs, and **Load** switches the model/LoRAs too. Stored in `presets/*.json`.
  A **basic preset is auto-created for every loadable model** (on startup and when you
  Refresh the checkpoint list) if it doesn't already exist yet — named after the model,
  with steps/CFG from its profile. Existing presets are never overwritten; skipped
  FP8/INT8-INT4 models get none.
- **Seed**: **♻️ Reuse last seed** (refills the real seed of the previous render) + **Fix
  seed** (no +1 per image). A random `-1` seed is resolved to a concrete value so it is
  actually saved in the metadata.
- **Advanced tab** — **metadata scheme** (`crispz` / **a1111** for **Civitai** upload
  compatibility), **read wildcards in order**, **also save pre-upscale image**, live
  **LoRA-slot count**, and the **Hugging Face token** (gated models).
- **PNG Info**: drop an image into *Input Image* to read its embedded **prompt + params**
  (crispz, **A1111/Civitai**, or ComfyUI) and send them to the fields.
- **Ollama (optional)**: **Describe** (image→prompt), **Improve prompt**, and **Vision
  Mix** (blend several reference images into one prompt). Models unload from VRAM after
  use. Without Ollama, **Describe** uses a local BLIP captioner and **Improve prompt**
  falls back to a local rule-based pass — both work fully offline.
- **Fooocus-style UI**: big contained preview + batch gallery (arrows + fullscreen),
  prompt + Generate + **Stop**, dark theme, Settings (aspect/performance/batch **1–30**),
  **277 styles** (search + hover previews), and a **crop editor** on every image input.
- **Asset Browser** (standalone gallery, new tab): opens **instantly** (indexing +
  thumbnails in the background, shimmer placeholder → real thumbnail); images save into
  **`out/YYYY-MM-DD/`** date subfolders; a **subfolder sidebar** with counts + per-folder
  **hide** + a **Hidden** toggle (persisted), **defaults to today**; **metadata keyword
  search**, per-image copy/delete, NSFW blur; and **Outputs / LoRAs / Models** source tabs
  (models show a Civitai preview if one sits next to the `.safetensors`, else a placeholder
  + trigger words). A **🔎 Fetch from CivitAI** button (per model, in its lightbox) looks the
  model up by **SHA256** and pulls its **preview + trigger words + example images** (saved as
  `<name>.preview.png` + `<name>.civitai.json`). The fetch shows **live progress** (spinner +
  bar: real `Hashing… %` when the file must be hashed, then Querying / Downloading) with an
  inline ✅/⚠️ result. **Example images are clickable** → a full-screen viewer shows each
  example **large with its generation prompt** (Copy prompt) and **← / →** to browse. A small
  **🖼️ icon** next to each **LoRA** dropdown and the **Z-Image checkpoint** dropdown
  (Advanced) opens the Asset Browser **straight to that model's card** (its preview /
  trigger words / examples). A **🔄 Fetch all missing** button (LoRAs / Models tabs)
  enriches the whole folder in one go (same as the standalone `civitai_index.bat` /
  `.sh` — see below); models with a **newer version on CivitAI** get a **⚠ update** badge.
  A **🖼 Rebuild ALL thumbnails (force)** button re-generates every thumbnail of the
  current tab from scratch (parallel, live progress) — for when a thumbnail is corrupt or
  you changed `thumbnail_size`.
  Plus a per-session history in the app. The **Output folder** can point
  anywhere (even another drive); a folder typed into the UI at runtime is auto-authorised,
  so the browser opens without a Gradio *"File not allowed"* error. (In `config.txt`, write
  Windows paths with `/` or `\\` — a single `\` is an illegal JSON escape.)
- **Metadata saved** with every image: PNG text chunk + EXIF (jpg/webp) + `.json`
  sidecar — prompt, negative, seed, steps, guidance, size, model, LoRAs, **applied
  style names**, and the **sampler/schedule**. **Dated, unique filenames** (date +
  tag + seed + size).
- **`config.txt`** for all defaults + the Ollama instruction strings.
- **Reference (Omni)** native multi-image compose: code ready, UI hidden until the
  Z-Image Omni/Edit model ships.

Tabbed Gradio UI + scriptable CLI + persistent server (`--serve`).

> **CLI cheat sheet:** see **[README_CLI.md](README_CLI.md)** for one-block examples of
> every mode (txt2img, upscale, LoRA, Vision Mix, Remove BG, Reframe, Face Swap, server).

### Launchers (Windows)

| Script | What it does |
|---|---|
| `run.bat` | Standard local launch (127.0.0.1:7860). |
| `xyz_example.bat` | Ready-to-run **X/Y/Z grid** CLI example (`xyz_example.bat "your prompt"`) — 2×2 Steps × Guidance, prints the sheet path. Unix: `xyz_example.sh`. |
| `boot_check_rtx5090.bat` | GPU / venv / torch / models diagnostic, then launch. |
| `run_quality_rtx5090.bat` | Local launch + RTX-5090 CUDA env. |
| `run_quality_rtx5090_lan.bat` | **LAN**: listens on `0.0.0.0`, prints your LAN URL. |
| `run_quality_rtx5090_web.bat` | **Web via Cloudflare tunnel** (named tunnel or ephemeral quick tunnel). |

They set `GRADIO_SERVER_NAME` / `GRADIO_SERVER_PORT` (Gradio reads them) and call `run.bat`.

**Cloudflare (private):** the web launcher reads `cloudflare.local.bat` (your tunnel
name/port) — this file is **gitignored**, never committed. Copy
`cloudflare.local.bat.example` to `cloudflare.local.bat` and fill it in. Leave
`CF_TUNNEL` empty for an ephemeral `*.trycloudflare.com` quick tunnel (no personal
config). Needs `cloudflared` (`winget install --id Cloudflare.cloudflared`).

> Roadmap status: **Phase 3** (FaceSwap) is implemented and works. **Phase 2**
> (Omni multi-reference) is coded but **its UI stays hidden** until a Z-Image
> Omni/Edit model is released (none yet) — use **img2img** for a reference image
> in the meantime. ControlNet / Inpaint are next. See the parent crispz repo for
> the upscale internals.

## Configuration (`config.txt`)

All defaults and the Ollama prompt strings live in a JSON config. The repo ships a
generic **`config-sample.txt`**; copy it to **`config.txt`** (your local copy,
gitignored) and edit that:

```bash
cp config-sample.txt config.txt    # Windows: copy config-sample.txt config.txt
```

Load order: `config.txt` → `config-sample.txt` → built-in defaults. See
**`config_modification_tutorial.txt`** for every key (filename pattern, Ollama
prompts, the Omni / FaceSwap model paths, etc.).

## Styles, Describe & Improve (Ollama)

- **Styles** tab (Advanced): 277 Fooocus/SDXL styles, a **search** box and a
  **thumbnail preview** gallery. Selected styles wrap your prompt and merge their
  negatives. (Sample thumbnails live in `styles/samples/`, local only.)
- **Describe** (Input Image → Describe): caption an image into a prompt using an
  Ollama **vision** model (auto-detected, vision-only list), or a **local captioner**
  (no Ollama needed).
- **Improve prompt**: rewrites the current prompt via the same Ollama model. URL +
  model in Advanced → **Prompt AI**. Tune the instructions in `config.txt`. **Without
  Ollama** it falls back to a local rule-based pass that appends quality tags
  (`improve_local_keywords` in `config.txt`) — instant, no model.
- **Local captioner** (no Ollama): the Describe fallback and the **Auto-describe**
  toggle in Inpaint / Outpaint use a local BLIP model, set by `caption_model` in
  `config.txt` (or **Prompt AI → Caption model**): `blip-large` (default, richer) ·
  `blip-base` (lighter). The model downloads on first use.

## Using a reference image (multi-reference status)

**Available now — img2img.** To guide generation with a reference image, use
**Input Image → Upscale or img2img** (uncheck "ESRGAN upscale" for a pure
img2img refine). One reference image + your prompt.

**Available now — Vision Mix.** Input Image → **Vision Mix** tab: drop up to 4
references, then either *"Vision Mix → prompt"* (fills the prompt) or
*"Vision Mix & Generate"* (one click: blend + generate). A vision model captions
each image and the LLM merges them into ONE prompt (e.g. person + outfit +
setting). Needs Ollama with a real vision model (llava, qwen-vl, …) set in
Advanced → Prompt AI. The merge instruction is `ollama_compose_prompt`, and
`ollama_vision_blocklist` hides models that wrongly claim vision — both in
`config.txt`. Vision Mix blends ideas/style, not exact pixels (that's what the
true Omni model, kept for later, will do).

**Multi-reference compose (person + outfit, etc.)** needs a model that can read
several reference images. The options and their status for Z-Image:

| Approach | Status for Z-Image |
|---|---|
| **Omni** (`ZImageOmniPipeline`, native multi-ref) | code ready; **model not released** (Z-Image-Omni-Base / Z-Image-Edit "coming soon") |
| **ControlNet** (`ZImageControlNetPipeline`) | pipeline exists; **no Z-Image ControlNet model yet** |
| **IP-Adapter** (what Fooocus uses for image prompts on SDXL) | **none for Z-Image yet** |

Fooocus's "Image Prompt / multi-reference" relies on **IP-Adapter + ControlNet**,
which are SDXL-family components — they don't apply to Z-Image. So for Z-Image the
only reference path today is **img2img**; true multi-reference waits on the models
above.

### Reference (Omni) — hidden until a model exists

The **Reference (Omni)** tab is **hidden by default** (no usable model). The code
is in place: once a Z-Image Omni/Edit model ships, set it in `config.txt`
(`"zimage_omni_model": "<HF repo or local diffusers folder>"`) **and restart** —
the tab appears and multi-reference works. Use **Models → Check Omni availability**
to see if it has been released.

## Job queue

Queue several generations with different settings and run them unattended.

- **`+ Queue`** (under the prompt area) freezes a complete snapshot: every Generate
  setting **plus the current model state** (checkpoint/transformer, LoRAs + weights,
  sampler/schedule). Jobs are self-contained — you can change the model afterwards, each
  job restores its own. The button shows the pending count.
- **Job queue panel** (accordion): labeled list, select a job, **Up / Down / Remove /
  Clear**, then **`Run queue`** to execute in order (normal progress bar, history and
  file saving as usual).
- **Stop = pause**: the current job is interrupted, remaining jobs stay queued; press
  `Run queue` to resume. A failed job logs `[crispz][queue] …` and the queue continues.
- VRAM purge between jobs happens **only** when the model actually changes (the existing
  model-cache invalidation does the work — zero cost for same-model series).
- Config (`config.txt`): `"job_queue": {"enabled": true}`. Set `false` to remove the
  panel entirely (no components created, zero cost).
- v1 limits: in-memory queue (cleared on page reload), sequential execution.

## X/Y/Z grid

Compare parameter variations side by side on an annotated contact sheet.

1. Open **X/Y/Z grid** (accordion under the Job queue), pick the **X axis** (and
   optionally Y and Z) and type the values, comma-separated — quotes protect commas
   (`"red, bright", blue`). The field's placeholder adapts to the chosen axis, and the
   **`⤵ suggest`** button pre-fills it (app lists for closed choices, calibration values
   for numeric axes) — it never overwrites what you already typed.
2. **Build grid → queue**: every combo becomes a job in the Job queue (validated first:
   numbers cast, closed lists matched case-insensitively — `uni` resolves to `unipc` —
   combo count capped by `max_jobs`).
3. **Run queue**. When the grid has run, one **annotated sheet per Z value** (X in
   columns, Y in rows, 512 px cells, missing cells drawn as placeholders) is saved under
   `<output>/xyz_<timestamp>/` and appended to the result gallery. Pause/resume keeps the
   collected cells, so the final sheet is complete.

Axes: `Checkpoint`, `Sampler`, `Schedule`, `Steps`, `Guidance`, `Seed`, `ESRGAN model`,
`Factor`, `Denoise`, `Tile`, `Refine tile`, `LoRA` (swap the file in LoRA slot 1),
`LoRA + weight` (swap file *and* weight), `LoRA weight` (all active LoRAs),
`Performance` (applies the preset), `Prompt S/R` (first value = search term, then its
replacements; the term must exist in the prompt).

**Comparing LoRA epochs.** The `LoRA` axis swaps the *file* in slot 1 while keeping the
weight you set in the Models panel — what you want when the same LoRA was trained over
several epochs, or re-uploaded as several CivitAI versions. Hit **`⤵ suggest`** to drop
the full list of available LoRAs into the field, then delete the ones you don't want to
compare. Names match like every other closed list: any unambiguous fragment works, so
`e000020` is enough — an ambiguous or unknown one is rejected at **Build** time, not
mid-series. `None` is a valid value for a LoRA-free control cell, and other active slots
are left untouched. `LoRA + weight` varies both at once, written `name:weight`
(`ollie_e10:0.6, ollie_e20:0.9`) — the suggest button pre-fills each entry with the
current weight, so you only edit the numbers. Cell labels show the base name without
extension, trimmed from the *left* so the `_e000020` suffix that tells your epochs apart
stays visible.

Config (`config.txt`): `"xyz_grid": {"enabled": true, "max_jobs": 100, "thumb": 512}` —
requires `job_queue`; `enabled=false` removes the panel entirely.

Also available from the CLI: `--txt2img --xyz "Steps=4,8,12" --xyz "Guidance=0, 3.5"`
(see README_CLI.md) — same axes and validation, Ctrl+C assembles a partial sheet.

## Tag autocomplete (prompt fields)

Suggestions appear under the caret while typing in the **prompt** and **negative**
fields (from 2 typed characters in the current comma-delimited token).

- **Keys**: ↑/↓ navigate · **Tab / Enter** insert · **Escape** close · click works too.
  Inserted tags get underscores replaced by spaces; `__wildcard__` entries are kept
  verbatim.
- **Sources**: the CSVs in `tag_autocomplete.sources` are downloaded **once** into
  `tags/` (atomic, with console progress). Drop any extra `.csv` in `tags/` to add a
  source — rich format `name,category,count,"alias1,alias2"` or one word per line.
  Your **wildcards** are merged in as `__name__` entries with top priority. Aliases
  match too (shown alongside the tag).
- **Performance**: the index is built once in the browser (popularity sort, dedup,
  2-char prefix buckets, early exit at `max_results`). Timings are logged in the
  browser console: `[tagac] ready in N ms` and a rolling per-keystroke average.
- Config (`config.txt`):
  `"tag_autocomplete": {"enabled": true, "max_results": 8, "sources": [<urls>]}` —
  `enabled=false` downloads nothing and injects no script (zero cost).

## Inpaint / Outpaint (Advanced tab)

One tab with three modes. The image editor, prompt, **Steps** (from the model
Performance) and **Strength** are shared across modes:

- **Brush (inpaint)** — paint a mask over the area to change, describe the result in the
  prompt, run. Brush size is set from the editor toolbar (click the brush icon).
- **Expand sides (outpaint)** — check **Left / Right / Top / Bottom** (or **Center** for
  all four) to grow the canvas ~30% per side; Z-Image fills the new borders.
- **Reframe (ratio)** — pick a target aspect ratio + **Contain** (keep the whole image
  and fill the borders) or **Cover** (crop to fill).

All modes are bounded to the model's **~1 MP sweet spot** (no pixel blow-up). Border
fills use a **blurred-edge init + feathered seams** so new content matches the original's
colors, and the unmasked area keeps its full resolution.

Tips for clean outpaint/reframe:

- **Describe the result** in the prompt (the full outfit/scene). **Auto-describe** runs
  automatically when the prompt is empty (local BLIP captioner, no Ollama) to keep the
  fill coherent with the center — or check it to also prepend a description to your prompt.
- **Strength** ~0.65–0.8 blends best (keeps the edge colors); ~1.0 adds more new detail
  but a more visible transition.
- **Harmonize** (checkbox) runs a light final img2img refine over the whole result to
  unify grain/light and remove any remaining "added zone" look.
- The local caption model is set in **Prompt AI → Caption model** (`blip-large` /
  `blip-base`).

## Face Swap — post-process  *(Phase 3, optional)*

Input Image → **Face Swap** tab: a source face + “Apply face swap to result”.
Works on any mode (txt2img / img2img / omni). The installer sets up the deps by
default (`insightface` + `onnxruntime-gpu`); skip with `install.bat --no-faceswap`,
or install manually:

```bash
.venv/Scripts/python -m pip install -r requirements-faceswap.txt
```

and an **inswapper model**: drop `inswapper_128.onnx` in the `faceswap/` folder
(auto-detected), or set `faceswap_model_path` / `faceswap_model_url` in `config.txt`.
The face-detection model (buffalo_l) downloads automatically on first use. If the
dep/model is missing, the run still succeeds and the report says `faceswap skipped`.

> The inswapper weights are not redistributed here (license). Get them from a
> Hugging Face mirror, e.g. `ezioruan/inswapper_128.onnx`. Local model files
> (`faceswap/`, `*.onnx`) are gitignored.

## Text -> Image

```bash
# Generate only
python app.py --txt2img --prompt "a serene mountain lake, cinematic" \
    --gen-width 1024 --gen-height 1024 --gen-steps 8 --seed 42 \
    --save-mode local --output-dir out

# Generate then upscale (ESRGAN + Z-Image refine)
python app.py --txt2img --prompt "portrait of an old fisherman" --upscale \
    --factor 2 --denoise 0.30 -m 4x-ClearRealityV1_Soft.safetensors \
    --save-mode local --output-dir out
```

In the UI, use the **Text -> Image** tab. Z-Image Turbo runs at `guidance 0` with few
steps (8 is a good default).

## Civitai / single-file Z-Image model

```bash
# Pass a .safetensors directly as the model (treated as the transformer)
python app.py --txt2img --prompt "..." \
    --zimage-model "D:/models/zimage_civitai.safetensors"

# Or keep an HF/diffusers base and override only the transformer
python app.py --zimage-transformer "D:/models/zimage_civitai.safetensors" ...
```

The single-file is loaded as the **transformer**; the **VAE + Qwen3 text encoder**
still come from the base repo (`Tongyi-MAI/Z-Image-Turbo` by default). Use a
**BF16/FP16** checkpoint - FP8/GGUF ComfyUI variants do not load cleanly in diffusers.

### Turbo vs Base (`--guidance`)

Z-Image comes in two flavors that need different inference settings:

| Model | Guidance (CFG) | Steps |
|---|---|---|
| **Z-Image Turbo** (distilled) | `--guidance 0` (default) | ~8 |
| **Z-Image Base** (full) | `--guidance 3.5-5` | ~20-28 |

```bash
# Turbo checkpoint
python app.py --txt2img --prompt "..." --zimage-model "D:/models/z-image-turbo.safetensors" \
    --gen-steps 8 --guidance 0 --save-mode local --output-dir out

# Base checkpoint (needs real CFG + more steps)
python app.py --txt2img --prompt "..." --zimage-model "D:/models/z-image-base.safetensors" \
    --gen-steps 24 --guidance 4 --save-mode local --output-dir out
```

Next to the **CFG guidance** slider there are two dropdowns, ComfyUI-style:

- **Sampler**: `euler` (native flow-matching, default) or `unipc` (UniPC multistep).
  Both accept the pipeline's custom sigmas + dynamic shift. The diffusers DPM++ 2M /
  DPM2a schedulers reject custom sigmas, so they are **not available** for Z-Image
  (this is a diffusers limitation, unlike ComfyUI). An incompatible choice falls
  back to Euler.
- **Schedule** (the sigma schedule, = ComfyUI's "scheduler"): `sgm_uniform` (the
  native Z-Image linear schedule, default), `beta`, `karras`, `exponential`. These
  remap the sigmas on top of the model's dynamic shift.

Both apply to txt2img / img2img / inpaint (not Omni). CLI: `--sampler`, `--schedule`.
For a **Z-Image Base** checkpoint (e.g. Civitai), the typical recipe is CFG ~4-5,
~30 steps (Performance "Base CFG"), sampler `euler`, schedule `sgm_uniform` or `beta`.

## Switching models in the UI (Advanced → Models)

The **Z-Image checkpoint** dropdown is the single place to switch model. It merges,
in one list:

- the official base repos **`Tongyi-MAI/Z-Image-Turbo`** and **`Tongyi-MAI/Z-Image`**
  (pulled from Hugging Face on first use), then
- every single-file `.safetensors` found in your **Checkpoints folder** **and** the
  optional **Extra checkpoints folder** (both merged into the same list).

What each choice does:

| You pick… | Effect | Performance preset (auto) |
|---|---|---|
| **Tongyi-MAI/Z-Image-Turbo** | full base repo (distilled) | **Turbo (8 steps)** |
| **Tongyi-MAI/Z-Image** | full base repo (needs real CFG) | **Base CFG (28 steps)** |
| a local `.safetensors` | used as the **transformer** (VAE + Qwen3 encoder kept from the current base repo) | from the model profile |

Switching the dropdown automatically syncs **steps, guidance and the Performance
radio**. The change is applied on the next **Generate**.

**Switching between two `.safetensors` (or clearing an override) reloads only the
transformer** — the VAE, the Qwen3-4B text encoder and the tokenizer stay in VRAM, so it
takes seconds instead of a full reload. Only picking a **different base repo** reloads
everything (its VAE/encoder genuinely differ). Same for LoRAs: they are hot-swapped, and
changing just a weight is instant.

Steps:

1. **Models → Checkpoints folder** (and, if you keep models elsewhere, **Extra
   checkpoints folder**) → **Refresh**. Both folders feed the single dropdown.
2. Pick an entry in **Z-Image checkpoint** — a base repo or a local file.
3. **Generate** → it loads with your selection.

### Community full-repo models (e.g. Juggernaut-Z-Image)

Some community models ship as a **full diffusers repo** but with an **incomplete
tokenizer** (only `tokenizer.json`), so loading them as the **base** fails. Load
just their **transformer** instead and keep the base components from Turbo:

1. **Models → "Transformer override (HF repo / diffusers folder)"** = e.g.
   `RunDiffusion/Juggernaut-Z-Image` → **Apply**.
2. Keep the **Z-Image checkpoint** dropdown on `Tongyi-MAI/Z-Image-Turbo` (provides
   VAE + Qwen3 encoder + tokenizer).
3. **Generate** (downloads the transformer once, ~12 GB).

CLI equivalent: `--zimage-transformer RunDiffusion/Juggernaut-Z-Image`.

Juggernaut-Z is a **Z-Image Base** fine-tune → set **Performance = "Base CFG"**
(guidance ~6, 25-45 steps). Tested working.

**Gotchas**

- Checkpoints must be **BF16/FP16**. FP8 / INT8-INT4 / GGUF / SVDQ (ComfyUI) variants
  do **not** load in diffusers and are **auto-hidden** from the checkpoint list (a line
  is logged: `checkpoint skipped (FP8 | INT8/INT4 quantized, ...)`). Pick the BF16 build.
- If the checkpoint is a **Z-Image Base** model (not Turbo), set **Performance →
  "Base CFG (28 steps)"** (guidance ~4, more steps), otherwise the result is flat.
- Verify what loaded with `run.bat --debug`:
  `[crispz] loading Z-Image transformer (single-file): …`.

**Persist the folders** (so you don't re-type them) in `config.txt`:

```json
"checkpoints_dir": "C:\\path\\to\\models\\Stable-diffusion\\Z-Image",
"checkpoints_extra_dir": "",
"loras_dir": "C:\\path\\to\\models\\Lora"
```

### LoRA (up to 3, combinable)

**Models → LoRA**: set the folder → **Refresh** → pick **up to 3 LoRAs**, each with
its own **weight** (range **`-2..2`**, configurable via `lora_weight_min` /
`lora_weight_max`). A **negative weight inverts the LoRA's effect** — a "skinny slider"
LoRA at `-1` pushes the other way; `0` disables it. They are combined (`set_adapters`) on
the transformer (shared
by txt2img/img2img) and applied on the next run **without reloading the model** —
changing a weight is instant, swapping LoRA files takes ~1 s. Selecting LoRAs auto-fills their
merged **keywords / trigger words** (read from the file metadata); **Add to prompt**
appends them.

## Disabling the upscale (pure txt2img / pure img2img)

- **txt2img only** (no upscale): the default. Don't pass `--upscale` (CLI), or leave
  the "Upscale after generation" checkbox off (UI).
- **img2img only** (refine without ESRGAN enlargement): `--no-esrgan` (CLI), or
  uncheck **"ESRGAN upscale"** in the Image -> Upscale tab. The Z-Image refine runs
  on the input at its native size.
- **ESRGAN only** (fast upscale, skip the slow refine): `--no-refine` (CLI, shortcut
  for `--denoise 0`), or uncheck **"Refine (img2img)"** in the Image -> Upscale tab.
  The two stages are independent toggles: the diffusion refine runs at the *upscaled*
  resolution, so it is the slow part — turn it off when you just want a clean enlarge.

```bash
# img2img only: Z-Image refine on the input, no enlargement
python app.py --cli -i in.png --no-esrgan --denoise 0.30 --save-mode local --output-dir out

# ESRGAN only: fast upscale, no diffusion refine
python app.py --cli -i in.png -m 4x-ClearRealityV1_Soft.safetensors --no-refine \
    --factor 2 --save-mode local --output-dir out
```

---

## Installation

### Requirements

- Python 3.10+
- PyTorch **already installed** with your CUDA build (the project targets
  PyTorch 2.7+ / CUDA 12.8). **NEVER reinstall torch** from this project; it
  aligns with your existing environment.
- An NVIDIA GPU with >= 8 GB VRAM is recommended. RTX 5090 tested in native BF16,
  whole-image up to 2048px without trouble.

### Provided install scripts

```bash
# Linux / macOS / WSL
./install.sh
./run.sh           # Gradio UI + hardware detection
./cli.sh           # interactive CLI with preferences

# Windows
install.bat
run.bat
cli.bat
```

The install scripts:
- find a base Python that already has PyTorch (never touch torch),
- by default create a **`.venv` virtual environment with
  `--system-site-packages`**: it **inherits your system torch/CUDA** (no torch
  reinstall) while **isolating crispz's own deps** (diffusers, gradio, spandrel)
  from your global Python,
- automatically uninstall a broken `xformers` (built for the wrong torch
  version -> DLL load error when diffusers loads),
- install the other deps from `requirements.txt`,
- verify that `ZImageImg2ImgPipeline` loads,
- create the `upscale_models/` folder.

`run.sh` / `cli.sh` (and the `.bat`) automatically use `.venv` if it exists.

### venv or not: the `--no-venv` flag

The venv is the default (keeps your global Python clean). To install/run directly
on the current interpreter instead (the old behavior), pass `--no-venv` (or
`--system`) to any script:

```bash
./install.sh --no-venv      # install on the current Python
./run.sh --no-venv          # run on the current Python
```

```bat
install.bat --no-venv
run.bat --no-venv
```

If venv creation fails, the scripts fall back to the current Python automatically.

Equivalent manual install (no venv):

```bash
pip install -r requirements.txt
```

### Known environment pitfalls

- **Incompatible `xformers`.** If an `xformers` version is installed but built
  for a different torch (e.g. `xformers` for torch 2.9 while you have torch 2.8),
  diffusers crashes with `DLL load failed while importing _C` when loading the
  VAE. Fix: `pip uninstall xformers`. The native SDPA in torch 2.7+ is enough.
- **`transformers` too old.** `ZImageImg2ImgPipeline` loads an encoder that
  imports `Dinov2WithRegistersConfig`, available since transformers >= 4.49.
  The requirements pin this lower bound.
- **diffusers from git.** Z-Image is only in diffusers from source (not in the
  releases at the time of writing), hence the `git+...` in requirements.
- **Gradio pinned `<6`.** Gradio 6's Brotli middleware has an h11 bug that spams
  `Too little data for declared Content-Length` in the console when a response is
  interrupted (non-fatal, but noisy). Requirements pin `gradio<6` to avoid it.

---

## Configurable paths (ESRGAN_DIR + Z-Image)

Two paths are configurable, persisted in `preferences.json`. Resolution order on
each launch:

1. Environment variable (`ESRGAN_DIR`, `ZIMAGE_MODEL`)
2. `preferences.json` at the project root
3. Default: `./upscale_models` for ESRGAN, `Tongyi-MAI/Z-Image-Turbo` for Z-Image

Three ways to change them:

- **Gradio UI**: **Advanced → Models** tab. Pick the model in the **Z-Image
  checkpoint** dropdown (it reloads on next Generate), set the **Checkpoints /
  Extra checkpoints / ESRGAN** folders, then **Refresh ESRGAN** or **Save paths**
  (writes `preferences.json`).
- **CLI**: `--esrgan-dir <path>`, `--zimage-model <repo_or_path>`, `--save-paths`
  to persist (with or without `-i`).
- **Interactive CLI** (`cli.sh` / `cli.bat`): first prompt = ESRGAN folder +
  Z-Image model. Saved to `preferences.json` if you choose to keep them.

`zimage_model` accepts either an HF repo (e.g. `Tongyi-MAI/Z-Image-Turbo`) or a
local path to an already-downloaded `diffusers` folder.

## ESRGAN models

Drop at least one `.pth` or `.safetensors` into `./upscale_models`, or point
`ESRGAN_DIR` (env or prefs) at an existing folder.

A few useful picks:
- `RealESRGAN_x4plus.pth` (general)
- `4x-UltraSharp.pth` (sharp, versatile)
- `4x-ClearRealityV1_Soft.safetensors` (soft, good on portraits/scenes)
- `4xFaceUpDAT.pth` (portraits/faces)

`spandrel` detects the architecture and the scale (x2 / x4) automatically.

---

## Z-Image (first run)

No file to provide: on first launch, `diffusers` fetches the Z-Image transformer,
the VAE and the Qwen3-4B text encoder from Hugging Face, then everything is cached
locally. Subsequent runs are offline.

**Loading progress** — because the first load downloads several GB and then reads them
into VRAM, it can take minutes. The terminal shows a live one-line status
(`[crispz][load] Z-Image base... 45s | 3.2 GB in VRAM`, or `... (downloading / reading,
first run only)` before allocation starts) and the UI progress bar advances with it.
Turn it off or tune it in `config.txt`:
`"load_progress": {"enabled": true, "target_vram_gb": 14.0, "heartbeat_s": 2.0}`
(`enabled: false` loads directly with no monitor thread).

---

## Running

### 1) Gradio UI (default)

```bash
python app.py
```

UI at http://127.0.0.1:7860. It includes:

- **Before/after slider** (`gradio_imageslider`) that overlays source and result
  with a mouse cursor. Falls back to two side-by-side images if the component is
  not installed.
- **Timing report** under the image: ESRGAN, Z-Image refine, total, source path,
  save path.
- **"Save" section** with the same modes as the CLI.
- **Batch mode**: if you fill in "OR source folder", the uploaded image is ignored
  and the app processes the whole folder.

### 2) Scriptable CLI

```bash
# Single image, explicit settings
python app.py --cli -i my_image.jpg \
    --save-mode local --output-dir out --output-format png \
    -m 4x-ClearRealityV1_Soft.safetensors \
    --factor 2 --denoise 0.30 --steps 12 --tile 760 --overlap 32

# Batch over a whole folder
python app.py --cli -i ./my_images --save-mode local --output-dir out --output-format webp

# Save next to each source ("alongside" mode)
python app.py --cli -i ./my_images --save-mode alongside --output-format jpg

# Display only (no file written), just the timing on stdout
python app.py --cli -i my_image.jpg --save-mode display --denoise 0

# With a TSV log to track timings
python app.py --cli -i ./my_images --save-mode local --time-log runs.tsv
```

### 3) Interactive CLI with preferences

```bash
./cli.sh   # or cli.bat on Windows
```

Prompts for each setting (paths, models, source, pipeline, save, time-log) with a
default value from `preferences.json`. Offers to save the choices at the end of the
session.

## Presets (use cases)

A "Use case" dropdown in the UI (and `--preset` on the CLI) fills the settings for a
given scenario. On the CLI, any explicit flag overrides the preset.

| Preset | What it sets |
|---|---|
| `Custom` | Nothing (default). |
| `Photo (balanced)` | factor 2, denoise 0.30, 12 steps. |
| `Subtle (clean-up)` | factor 2, denoise 0.12, 16 steps. Stays very close to the input. |
| `Detailed (creative)` | factor 2, denoise 0.40, 16 steps. More invented detail. |
| `Portrait (faces)` | factor 2, denoise 0.22, 14 steps. |
| `4K (tiled)` | factor 4, tile 1024, overlap 64, `--cpu-offload model`. |
| `Low VRAM (8-12GB)` | ESRGAN tile 512, diffusion tile 1024, `--cpu-offload sequential`. |

```bash
python app.py --cli -i in.png --preset "4K (tiled)" --save-mode local --output-dir out
python app.py --cli -i in.png --preset "Detailed (creative)" --denoise 0.32   # flag wins
```

## Mapping UI <-> CLI <-> preferences.json

Every UI setting has a CLI flag and a prefs key:

| UI / interactive CLI | CLI flag | preferences.json | Default |
|---|---|---|---|
| ESRGAN_DIR | `--esrgan-dir` | `esrgan_dir` | `./upscale_models` |
| Z-Image model | `--zimage-model` | `zimage_model` | `Tongyi-MAI/Z-Image-Turbo` |
| Source image | `-i` (file or glob) | - | - |
| Batch source folder | `-i` (folder) or `--input-folder` | - | - |
| ESRGAN model | `-m` / `--model` | `model` | `4x-ClearRealityV1_Soft.safetensors` |
| Use-case preset | `--preset` | - | `Custom` |
| Upscale factor | `--factor` | `factor` | `2.0` |
| Denoise (strength) | `--denoise` | `denoise` | `0.30` |
| Skip refine (ESRGAN only) | `--no-refine` | "Refine (img2img)" checkbox (off) | refine on |
| Diffusion steps | `--steps` | `steps` | `12` |
| Prompt | `--prompt` | `prompt` | `""` |
| Seed | `--seed` | `seed` | `-1` |
| ESRGAN tile | `--tile` | `tile` | `760` |
| Overlap | `--overlap` | `overlap` | `32` |
| Sampler | `--sampler {euler,unipc}` | "Sampler" dropdown (next to CFG) | `default_sampler` (`euler`) |
| Sigma schedule | `--schedule {sgm_uniform,beta,karras,exponential}` | "Schedule" dropdown | `default_schedule` (`sgm_uniform`) |
| CPU offload (diffusion) | `--cpu-offload` | - | `none` |
| Diffusion tile (4K+) | `--refine-tile` | - | `0` (whole image) |
| Diffusion tile overlap | `--refine-overlap` | - | `64` |
| Save mode | `--save-mode` | `save_mode` | `display` |
| Output folder | `--output-dir` | `output_dir` | `out` |
| Output format | `--output-format` | `output_format` | `png` |
| Time log (CLI) | `--time-log <file.tsv>` | `time_log` | (empty) |
| Save paths (CLI) | `--save-paths` | - | - |
| List models (CLI) | `--list-models` | - | - |
| VRAM peak on stderr (CLI) | `--report-vram` | - | - |
| Output path only (CLI) | `--print-output` | - | - |

Save modes:

| save_mode | Behavior |
|---|---|
| `display` | Writes nothing. UI renders the image + timing. CLI prints the report. |
| `local` | Writes to `output_dir`, resolved **relative to the project** if not absolute. |
| `alongside` | Writes to the **same folder as the source**. Requires a source path (CLI or batch folder). |
| `custom` | Writes to `output_dir` as-is (typically an absolute path). |

Default naming: `{source_name}_upscaled.{png|webp|jpg}`. On the CLI, `-o` accepts
a file (overrides auto naming), a folder (equivalent to
`--save-mode local --output-dir <folder>`), or is omitted (uses
`--save-mode` / `--output-dir`).

Full `preferences.json` example:

```json
{
  "esrgan_dir": "C:/path/to/models/ESRGAN",
  "zimage_model": "Tongyi-MAI/Z-Image-Turbo",
  "model": "4x-ClearRealityV1_Soft.safetensors",
  "factor": 2.0,
  "denoise": 0.30,
  "steps": 12,
  "prompt": "",
  "seed": -1,
  "tile": 760,
  "overlap": 32,
  "save_mode": "local",
  "output_dir": "out",
  "output_format": "png",
  "time_log": ""
}
```

## Timing report

`run()` returns (and prints / logs) the time of each stage:

- `esrgan` : stage 1 (Real-ESRGAN + Lanczos resize)
- `refine` : stage 2 (Z-Image img2img). 0s if `denoise <= 0`.
- `total`  : sum

crispz also prints `[crispz] ...` stage logs to **stderr** (loading ESRGAN, loading
or reusing the Z-Image pipeline, stage timings, per-tile progress). This fills the
otherwise-silent model-load gaps and tells you whether a run reloaded the pipeline or
reused the cached one. Silenced with `--quiet`; on stderr, so it never pollutes
`--print-output`.

The UI shows a Markdown block under the image. The CLI prints the report on
stdout (unless `--quiet`). With `--time-log <file>`, each run appends a TSV line:

```
<iso-timestamp>\t<src>\t<dst>\tesrgan=2.24s\trefine=1.87s\tmode=local\tfmt=png
```

---

## External integration (Fooocus, scripts)

Two flags make it easy to call crispz from another tool (separate process):

- `--print-output` : stdout contains ONLY the absolute path of each saved image
  (one per line), nothing else. The human-readable report is suppressed. This is
  the machine-parsable contract for retrieving the result.
- `--report-vram` : run VRAM peak on **stderr** (line `[VRAM] pic alloue:
  X.XX Go | pic reserve: Y.YY Go`). On stderr, so it does not pollute the stdout
  of `--print-output`. Used to size VRAM coexistence (e.g. with Fooocus).

```bash
# The caller reads the output path on stdout, VRAM on stderr
dst=$(python app.py --cli -i in.png --save-mode local --output-dir out \
    --print-output --report-vram 2>vram.log)
echo "upscaled image: $dst"
```

`--print-output` requires a save mode that writes a file
(`local` / `alongside` / `custom`). In `display` nothing is written, so nothing
is printed.

---

## VRAM offload (`--cpu-offload`)

The Z-Image refinement pass is the heavy VRAM consumer. By default it runs fully
on the GPU. To shrink the peak (so crispz can coexist with another GPU app, e.g. a
loaded Fooocus), `--cpu-offload` streams the diffusion weights between RAM and GPU.
This is NOT quantization: weights stay BF16, they just move RAM <-> GPU. Requires
`accelerate` (already in `requirements.txt`). Available in the UI too (Tiling/VRAM
accordion) and on the CLI.

| Mode | What it does |
|---|---|
| `none` (default) | Everything in VRAM. Fastest, highest peak. |
| `model` | Offload per submodule. Good tradeoff: ~half the peak, similar speed. |
| `sequential` | Most aggressive, lowest peak, a bit slower. |

Measured (RTX 5090, source 832x1216 -> x2 = 1664x2432, denoise 0.30, 12 steps):

| Mode | Peak allocated | Peak reserved | Time |
|---|---|---|---|
| `none` | 28.48 GB | 32.35 GB (spills to shared RAM) | ~59s |
| `model` | 13.54 GB | 24.02 GB | ~52s |
| `sequential` | 9.20 GB | 9.22 GB | ~61s |

```bash
python app.py --cli -i in.png --save-mode local --output-dir out \
    --cpu-offload sequential --report-vram
```

**Recommended (32 GB card):**

- **2K output (<= ~2048 px):** `--cpu-offload model`. Fits in ~24 GB and is the
  fastest. `none` needs 32.35 GB and spills into Windows shared memory on a 32 GB
  card, which is slower.
- **4K+:** add `--refine-tile 1024` (diffusion tiling) on top of `--cpu-offload
  model` (or `sequential`). Whole-image at 4K OOMs.
- **Sharing the GPU with another app:** `--cpu-offload sequential` (~9 GB).

In the Fooocus Extra plugin the host SDXL model is unloaded before each call, so
crispz gets almost the whole card. `model` is still the best 2K pick because `none`
sits right at the 32 GB limit and spills.

---

## Speed (making img2img / upscale faster)

The img2img **refine** is the slow part of the upscale path because the diffusion
runs at the **post-ESRGAN** resolution (x2 = 4x the pixels, x4 = 16x). Levers,
fastest first:

- **Skip the refine** when you only need a clean enlarge: uncheck **"Refine
  (img2img)"** (UI) or `--no-refine` (CLI). ESRGAN alone is near-instant.
- **Refine before upscale**: check **"Refine before upscale (faster)"** (UI) or
  `--refine-first` (CLI). The diffusion runs at the *native* resolution, then ESRGAN
  enlarges -> the refine is ~4-16x faster (a touch less high-res detail). Default
  via `default_refine_first` in `config.txt`.
- **`--cpu-offload model`**: on a 32 GB card the x2 refine at ~2K *spills* into
  Windows shared memory in `none` mode (slow); `model` fits in ~24 GB with no spill
  and is actually **faster**. This is the single biggest fix if the refine crawls.
- **Fewer refine steps / lower denoise** (effective steps = `steps x denoise`).
- **Attention slicing is now per-pass, by resolution** (`attention_slice_above`, default
  1664 px): OFF for tiles / 1024-1536 (native SDPA attention = fast, like ComfyUI), ON
  only for whole-image 2K+ (caps the VRAM peak, avoids the shared-RAM spill). This is the
  big one for **tiled upscale** (`--refine-tile 1024`): slicing no longer slows the tiles.
- **`--cpu-offload none`** on a big card (5090): `model`/`sequential` stream weights
  RAM<->GPU every step (much slower); only use them when you actually lack VRAM. With
  `--refine-tile`, VRAM is already capped, so keep offload `none`.
- TF32 matmul is enabled on CUDA.

```bash
# Fast img2img + upscale: refine small, then ESRGAN to x2
python app.py --cli -i in.png -m 4x-ClearRealityV1_Soft.safetensors --refine-first \
    --factor 2 --denoise 0.30 --save-mode local --output-dir out
```

---

## Server mode (`--serve`)

For repeated upscales, the per-call model load (Z-Image + Qwen3-4B encoder) dominates
and makes timings very uneven. `--serve` runs a small HTTP server that loads the model
**lazily on the first request** and keeps it **warm**, then **frees the VRAM after
`--idle-timeout` seconds** of inactivity (so it can coexist with another GPU app).
Requires `fastapi` + `uvicorn`.

```bash
python app.py --serve --host 127.0.0.1 --port 7861 --idle-timeout 300
```

Endpoints:

| Method | Path | Body / result |
|---|---|---|
| GET | `/health` | `{status, device, pipe_loaded, offload, idle_timeout}` |
| GET | `/models` | `{esrgan_dir, models:[...]}` |
| POST | `/upscale` | JSON (`input` path + any setting, incl. `preset`) -> `{output, size, esrgan_s, refine_s, total_s}` |
| POST | `/unload` | Frees the VRAM now -> `{status:"unloaded"}` |

```bash
curl -s http://127.0.0.1:7861/upscale -H "Content-Type: application/json" -d '{
  "input": "in.png", "preset": "4K (tiled)",
  "save_mode": "local", "output_dir": "out"
}'
```

Measured benefit (RTX 5090, 2K): first call ~66s (cold, model load), next call ~46s
(warm). The model stays resident between calls until the idle timeout fires.

---

## Useful settings

| Setting | Advice |
|---|---|
| **Denoise (strength)** | 0.05-0.25 = subtle, stays very close to the input. 0.25-0.40 = creative, more detail injected. Beyond ~0.40, Z-Image starts to reinvent. At high denoise, a **detailed caption prompt** greatly improves coherence. |
| **Denoise + tiled refine (4K+)** | When the refine is **tiled** (4K, or auto-tiled above `auto_refine_tile_above`), each tile is re-diffused independently. The **global prompt describes the whole scene, not the tile** -> passing it to every tile makes the model redraw the subject (you get the teacup / butterfly repeated in several tiles). Two guards: (1) `refine_tile_prompt` -> per-tile prompt, **empty by default** so each tile only refines local detail (set to `"global"` for the old behavior, or a generic string like `"high detail, sharp focus"`); (2) `refine_tile_denoise_cap` (default **0.40**) caps the per-tile denoise as a safety net. Whole-image refine keeps your prompt and denoise (no duplication possible). |
| **Steps** | Effective steps ~= `steps * strength`. At strength 0.30, 12-16 steps give enough denoising steps. |
| **Guidance** | Fixed at 0.0 (Z-Image Turbo). |
| **Prompt** | Optional. Empty works very well if denoise <= 0.30. |
| **Factor** | ESRGAN runs at native x4, then Lanczos resizes to the requested factor. For a clean x2, the image goes through a raw x4. |
| **ESRGAN tiling** | 0 (whole image) on 24+ GB VRAM. 512-768 otherwise. Overlap 32 by default, increase if you see seams. |

The `_hw_check.py` script (called by `run.sh` / `run.bat`) detects your GPU and
gives recommendations based on VRAM, compute capability (BF16 available from
Ampere = CC 8.0), and the max image size for the diffusion pass.

---

## Reference settings

A reliable starting point (source ~832x1216 -> x2), model
`4x-ClearRealityV1_Soft.safetensors`, factor 2, denoise 0.30, steps 12,
tile 760, overlap 32:

```bash
python app.py --cli -i my_image.jpg -o out/my_image_upscaled.png \
    -m 4x-ClearRealityV1_Soft.safetensors \
    --factor 2 --denoise 0.30 --steps 12 --tile 760 --overlap 32
```

---

## High resolution (4K+): diffusion tiling

By default the Z-Image pass runs on the whole image. That is ideal up to ~2048px on
the long side; beyond that you exceed the training resolution (artifacts) and the
VRAM peak explodes.

`--refine-tile <px>` (0 = off) tiles the diffusion pass, Ultimate SD Upscale style:
each tile is refined separately and recomposed with linear feathering over
`--refine-overlap` (so seams are invisible). This both **caps the VRAM peak** (one
tile at a time, independent of the final size) and **enables 4K+**. Try a tile of
1024-1280 (rounded to a multiple of 16) with overlap 64.

```bash
# 4K refine, tiled, seam-free
python app.py --cli -i in.png --factor 4 --denoise 0.30 --steps 12 \
    --refine-tile 1024 --refine-overlap 64 \
    --save-mode local --output-dir out
```

Measured (RTX 5090, base 832x1216 -> x4 = 3328x4864, tile 1024, denoise 0.30):
EXIT in ~86s, VRAM peak ~21.7 / 23.0 GB (vs OOM for whole-image 4K). Combine with
`--cpu-offload` for an even lower peak. Whole-image mode (`--refine-tile 0`) stays
the default and the best choice under ~2048px (no regression).

---

## License

CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial). See `LICENSE.txt`.
