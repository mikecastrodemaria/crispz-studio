# Changelog

All notable changes to crispz-studio. One versioned entry per feature.
The app version lives in `cz_core.py` (`APP_VERSION`) and is shown in the browser tab title.

## 1.5.2 — 2026-07-05 — Fix: empty "Apply override" no longer clears the checkpoint

- Selecting a checkpoint in **Z-Image checkpoint** applies it automatically. Clicking the
  transformer-override **Apply** button with an **empty** field used to call
  `set_zimage_transformer("")`, silently wiping that selection (the terminal then showed
  `transformer -> (repo de base)` and Generate loaded the plain base repo).
- The button is now a no-op on an empty field (returns a clear hint instead of clearing),
  and was relabeled **"Apply override"** (secondary) to distinguish it from the main
  checkpoint dropdown. To go back to the plain base repo, pick an official repo in the
  dropdown.

## 1.5.1 — 2026-07-05 — Fix: tensor-size mismatch on non-/32 image dimensions

- Fixes `Upscale/img2img failed: The size of tensor a (150) must match the size of
  tensor b (148)` — hit e.g. with **Force aspect ratio** crops whose height/width is a
  multiple of 16 but not 32 (1200, 848…). The Z-Image transformer patchifies the VAE
  latent by 2, so **every pixel dimension must be a multiple of 32**.
- `round_to_multiple` default is now **32** (txt2img sizes, refine tiles, ESRGAN targets
  all align), and `_refine_whole` snaps its input to /32 (resize) before diffusion then
  restores the original size — callers and tiled overlap-add contracts unchanged.

## 1.5.0 — 2026-07-05 — X/Y/Z grid in the CLI

The comparison grid is no longer UI-only: `--xyz "AXIS=v1,v2,…"` (repeat up to 3 times
for X, Y, Z) with `--txt2img` runs every combo and ends with the same annotated contact
sheet(s) in `<output>/xyz_<timestamp>/` (paths printed on stdout).

```bash
python app.py --cli --txt2img --prompt "a red cat" \
    --xyz "Steps=4,8,12" --xyz "Guidance=0, 3.5" --save-mode local
```

- Same axes and validation as the UI grid (shared helpers): case-insensitive axis and
  closed-list resolution (`step` → `Steps`, `uni` → `unipc`), quotes protect commas,
  Prompt S/R checked against `--prompt`, duplicate axes rejected, `max_jobs` cap.
  Upscale-only axes (ESRGAN model, Factor, Denoise, Tile, Refine tile) require
  `--upscale` (clear error otherwise).
- Each combo is saved as a normal output (tag `xyz`, metadata includes the combo);
  **Ctrl+C assembles a partial sheet** with the cells rendered so far.
- Respects `xyz_grid.enabled` (config) — disabled = clear error, nothing runs.
- Ready-to-run example scripts: `xyz_example.bat` / `xyz_example.sh`
  (`xyz_example.bat "your prompt"` → 2×2 Steps × Guidance grid; edit the `--xyz` lines
  to change the axes). Fails loudly with a non-zero exit code on error.
- Files: `cz_cli.py` (`--xyz`, runner), `cz_ui.py` (axes table gains abstract `param`
  names shared with the CLI), `xyz_example.bat`/`.sh`, `tests/test_xyz.py`
  (CLI apply + axis resolution).

## 1.4.0 — 2026-07-05 — Tag autocomplete in prompt fields

Type-ahead suggestions in the **prompt** and **negative prompt** fields.

- **Sources**: CSVs listed in `tag_autocomplete.sources` are downloaded **once at first
  launch** into `tags/` (atomic tmp+rename, one-line console progress); any `.csv` you
  drop into `tags/` becomes a source too (rich `name,category,count,"aliases"` format or
  one word per line). Local assets are merged in: your **wildcards** appear as
  `__name__` entries at top priority.
- **Client**: vanilla JS injected only when enabled (`gr.Blocks(head=…)`). Index built
  once — global popularity sort, cross-source dedup, **2-char prefix buckets, early
  exit** — then a dropdown under the caret: ↑/↓ navigate, **Tab/Enter** insert (current
  comma-delimited token replaced, underscores → spaces, `__wildcards__` kept verbatim),
  **Escape** closes. Aliases match too (shown in gray with the matched alias). Startup
  and per-keystroke timings logged in the browser console (`[tagac] ready in N ms`,
  rolling average per 50 keystrokes).
- **Zero-cost off**: `"tag_autocomplete": {"enabled": false}` → `cz_tags` never imported,
  nothing downloaded, no script injected.
- New generic helper `cz_core.download_with_progress` (atomic, 64 KB blocks, one-line
  progress) — also used by the inswapper/GFPGAN downloads from this version on.
- Files: `cz_tags.py` (new), `cz_assets.py` (`TAG_AC_JS`), `cz_ui.py`, `cz_core.py`,
  `cz_cli.py` (`tags/` served), `config-sample.txt`, `.gitignore` (`tags/`),
  `tests/test_tagac.py`.

## 1.3.0 — 2026-07-05 — Contextual suggestions for X/Y/Z value fields

- Each value field adapts to the axis picked in the neighboring dropdown: the
  **placeholder** shows contextual examples, and a **`⤵ suggest`** button inserts a
  ready-to-prune list — app lists for closed choices (Sampler, Schedule, Performance,
  Checkpoint incl. both folders, ESRGAN models), classic calibration values for numeric
  axes (Steps `4, 8, 12, 20, 28`, Guidance `0, 2, 3.5, 5`, Denoise `0.2, 0.3, 0.4`…),
  syntax hint for Prompt S/R.
- The fill button never overwrites a non-empty field; values containing commas/quotes are
  CSV-quoted so the inserted text re-parses exactly (round-trip tested).
- Case-insensitive partial matching at build time (from 1.2.0) completes the loop:
  suggestions can be shortened by hand (`uni` → `unipc`).
- Config: sub-key `"suggest": true` of the `xyz_grid` block; `false` = no buttons, no
  handlers, static placeholders.
- Files: `cz_ui.py`, `config-sample.txt`, `tests/test_xyz.py`.

## 1.2.0 — 2026-07-05 — X/Y/Z comparison grid

Compare parameter variations on an annotated contact sheet, powered by the job queue.

- **X/Y/Z grid panel** (accordion under the Job queue): pick 1–3 axes and their values
  (comma-separated; quotes protect commas). **Build grid → queue** turns every combo
  into a queued job; run/pause/reorder like any other jobs.
- **Axes**: Checkpoint, Sampler, Schedule, Steps, Guidance, Seed, ESRGAN model, Factor,
  Denoise, Tile, Refine tile, LoRA weight (applies to all active LoRAs), **Performance**
  (applies the whole preset), **Prompt S/R** (a1111-style search & replace: first value =
  search term, next values = replacements; validated against the prompt at build time).
- **Validation at build**: numeric casts, closed lists resolved case-insensitively (unique
  substring accepted, e.g. `uni` → `unipc`), duplicate axes rejected, combo count capped
  (`max_jobs`, default 100).
- **Contact sheets** (Pillow, no new dependency): one annotated sheet per Z value — X in
  columns, Y in rows, letterboxed cells (`thumb`, default 512 px), missing cells drawn as
  placeholders — saved under `<output>/xyz_<timestamp>/` and appended to the result
  gallery. Cells are accumulated across pause/resume, so a paused grid still ends with a
  complete sheet.
- Config block: `"xyz_grid": {"enabled": true, "max_jobs": 100, "thumb": 512}` (requires
  `job_queue`); `enabled=false` creates nothing (zero cost).
- Files: `cz_ui.py` (axes table, validation, plan builder, assembler, panel),
  `config-sample.txt`, `tests/test_xyz.py`.

## 1.1.0 — 2026-07-04 — Job queue

Queue up generations with different settings and run them unattended (e.g. overnight).

- **`+ Queue`** snapshots ALL current settings: the full Generate parameter set **plus the
  global model state** (checkpoint/transformer, active LoRAs + weights, sampler, schedule),
  so each job is self-contained and reproducible regardless of what is loaded later.
  The button label shows the pending count (`+ Queue (3)`).
- **Job queue panel** (accordion under the prompt area): readable labels
  (`txt2img · model · 1024x768 · 8 steps · seed 42 · x2 · "prompt…"`), select a job and
  **Up / Down / Remove / Clear**.
- **`Run queue`** executes jobs in order in the normal progress window; the session
  history and saved outputs accumulate as usual. Before each job the model state is
  restored through the existing setters, so **VRAM is purged automatically only when the
  model actually changes** between jobs (zero cost otherwise).
- **Stop pauses the queue**: the current job is interrupted (existing Stop behavior) and
  the remaining jobs stay queued — press `Run queue` again to resume. A failing job is
  logged (`[crispz][queue] …`) and the queue continues with the next one.
- Config block (`config.txt`): `"job_queue": {"enabled": true}` — set `false` to remove
  the panel entirely (no components, no handlers, zero cost).
- Files: `cz_ui.py` (panel + handlers + pure helpers), `cz_core.py` (`APP_VERSION`,
  module-prefixed logs), `config-sample.txt`, `tests/test_queue.py`.
- Limits (v1): the queue lives in memory (cleared on page reload); jobs are not editable
  in place (remove + re-queue); execution is sequential.

## 1.0.0 — 2026-07-04 — Baseline

Everything up to and including: unified Inpaint/Outpaint editor (brush / expand sides /
reframe, ~1 MP bound, harmonize), auto-upscale after generate, local BLIP captioner +
auto-describe, unified Z-Image checkpoint dropdown (+ extra folder, Performance
auto-sync), multi-LoRA, face swap + GFPGAN, remove background, Asset Browser (instant
open, day filter, placeholders), Ollama integration with offline fallbacks, CLI and
server mode.
