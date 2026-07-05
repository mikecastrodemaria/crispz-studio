# Changelog

All notable changes to crispz-studio. One versioned entry per feature.
The app version lives in `cz_core.py` (`APP_VERSION`) and is shown in the browser tab title.

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
