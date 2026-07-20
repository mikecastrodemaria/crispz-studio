# Changelog

All notable changes to crispz-studio. One versioned entry per feature.
The app version lives in `cz_core.py` (`APP_VERSION`) and is shown in the browser tab title.

## 1.12.0 — 2026-07-20 — X/Y/Z grid: compare LoRA files (epochs, versions)

Comparing several trainings of the same LoRA — epochs of one run, or successive CivitAI
versions — meant editing the Models panel and rebuilding a grid by hand. Two axes now do
it in one build.

- **`LoRA` axis**: swaps the *file* in LoRA slot 1 and keeps the weight set in the Models
  panel. Other active slots are left untouched. `None` is a valid value → control cell
  with no LoRA.
- **`LoRA + weight` axis**: varies both at once, written `name:weight`
  (`ollie_e10:0.6, ollie_e20:0.9`), for when the best weight differs per epoch. Split on
  the *last* `:` so Windows paths survive.
- **`⤵ suggest` lists the available LoRAs** (same mechanism as Checkpoint): the button
  drops the full list into the field, ready to prune. For `LoRA + weight` each entry is
  pre-filled with the current weight, so only the numbers need editing. The inserted list
  is CSV-quoted, so a filename containing a comma round-trips.
- Names resolve like every other closed list (`_xyz_match`): any unambiguous fragment
  works (`e000020`), ambiguous or unknown ones are rejected at **Build** time rather than
  mid-series.
- Cell labels show the base name without extension, truncated **from the left** — LoRA
  being compared usually differ only by their `_e000020` suffix, so trimming the end
  would have made every column read the same.
- Available from the CLI too: `--xyz "LoRA=ollie_e10, ollie_e20, None"`.
- Files: `cz_ui.py` (`_XYZ_AXES` + `lora_name` / `lora_name_weight` in `_xyz_suggestions`
  / `_xyz_validate_axis` / `_xyz_apply`, new `_xyz_fmt_value` + `_xyz_current_lora_weight`),
  `cz_cli.py` (`_xyz_cli_apply`, labels), `tests/test_xyz.py` (+6 tests: resolution,
  ambiguity, weights, apply, left-truncation, suggest round-trip).

## 1.11.4 — 2026-07-20 — Fix: asset-browser thumbnails corrupted files being served

Generating thumbnails while the Asset Browser SPA was displaying them produced
`h11 LocalProtocolError: Too much data for declared Content-Length` bursts in the console,
and broken images in the page. `FileResponse` takes `Content-Length` from an `os.stat`,
then re-reads the file to send it; `im.save(dst)` truncates `dst` to 0 and grows it, so a
request landing in that window declared one size and sent another. With 8 worker threads
over hundreds of files, the window was wide open.

- Thumbnails are now written to a temp file then `os.replace()`d (atomic): a reader sees
  either the previous complete file or the new one, never one mid-write. Same treatment
  for the other served files rewritten in place — `index.html`, `manifest.json` (both the
  reindex and the stub) and `<kind>.json`; `ab_open_fast` had the same race by design,
  since it spawns a background reindex that rewrites the manifest the SPA is polling.
- Side effect fixed: a truncated thumbnail kept a fresh mtime, so the
  `getmtime(thumb) >= getmtime(src)` check considered it up to date and it stayed corrupt.
- `os.replace` retries on Windows `PermissionError` (a destination held open by the
  serving thread), ~1 s with capped backoff; on definitive failure the thumbnail is
  counted as failed and regenerated next pass rather than written unsafely.
- Measured on a 1 writer / 4 reader race: 1725 Content-Length mismatches before, 0 after.
- Files: `cz_assetbrowser.py` (`_write_atomic_text`, `_replace_retry`, `_ab_make_thumb`).

## 1.11.3 — 2026-07-16 — Fix: LoRA hot-swap left stale adapters ("Already found a peft_config")

Switching LoRA in the UI logged a PEFT warning — *"Already found a `peft_config` attribute
in the model. This will lead to having multiple adapters."* — because
`unload_lora_weights()` does not reliably clear the transformer's `peft_config` in this
diffusers version. Since the hot-swap reuses the same adapter names (`cz_lora_i`), a stale
adapter could remain and the wrong LoRA be applied.

- `_apply_loras` now clears via a new `_clear_loras(pipe)`: `unload_lora_weights()` **then**
  an explicit `delete_adapters(get_list_adapters())` to remove any leftover adapter by name
  — so a swap A→B leaves only B registered, no accumulation.
- Safe by construction: the extra calls are wrapped in try/except and fall back to the
  previous behavior on any error.
- Files: `cz_pipeline.py` (`_clear_loras`), `tests/test_lora_hotswap.py` (+2 tests
  modelling the adapter lifecycle: a swap leaves only the new adapter; removing all clears
  the registry).

## 1.11.2 — 2026-07-16 — Fix: "Image number (batch)" was ignored in img2img / Input image

With **Input image** checked, `_ui_generate` called `run()` exactly once and returned a
single image — the **Image number (batch)** slider was silently dropped, so it only ever
worked in txt2img.

- The img2img/upscale branch now loops like txt2img: **n images**, seed **+1 per image**
  (or fixed if *Fix seed (no +1 per image)* is checked), **wildcards and random style
  re-rolled per image**, **Stop** honoured between images. The report lists each image
  (`1/4 (seed 1234)`); every image keeps its real saved filename for download.
- A **seed `-1`** is now resolved to a concrete value up front (as in txt2img), so
  **♻️ Reuse last seed** and the image metadata finally work in img2img too.
- **Refine (img2img) unchecked** = denoise 0 = no diffusion pass, so the output is
  deterministic and a batch would just write n identical files: the batch is clamped
  to 1 in that case (logged).
- Files: `cz_ui.py` (`_ui_generate`), `tools/smoke_test.py` (3 checks), `VALIDATION.md`.

## 1.11.1 — 2026-07-15 — Fix: SVDQuant/Nunchaku checkpoints were not filtered out

The README says FP8 / SVDQ (ComfyUI) checkpoints do not load in diffusers, and
`_safetensors_unsupported` filtered FP8 and `weight_scale`-style INT8/INT4 — but it
missed **SVDQuant / Nunchaku**, which uses a different convention: no `weight_scale`,
weights named `*.qweight`. Such a file stayed in the checkpoint dropdown and only failed
at load time.

- Detection added: any `*.qweight` tensor -> `"SVDQuant/Nunchaku INT4"`, skipped at
  startup with the reason like FP8. A normal BF16/FP16 checkpoint never has `qweight`,
  so there is no false positive.
- Verified on a real file (`…_svdqInt4R32Flux1Dev.safetensors`: 380 `qweight` keys,
  dtypes I32/BF16/I8, zero `weight_scale` — which is exactly why the old rule missed it)
  and against 9 other real checkpoints (BF16 -> kept, FP8 -> still caught).
- Files: `cz_pipeline.py` (`_safetensors_unsupported`).

## 1.11.0 — 2026-07-15 — "Rebuild ALL thumbnails (force)" button + parallel thumbnail generation

- New **🖼 Rebuild ALL thumbnails (force)** button in the Asset Browser header. It applies
  to the **tab you are on** — **Models**, **LoRAs** or **Outputs** — and force-regenerates
  every thumbnail from scratch (useful after a corrupt/partial thumbnail or a
  `thumbnail_size` change, which the normal "skip if up to date" rule would never redo).
- Runs **in the background with live progress**, reusing the same job + polling
  infrastructure as the CivitAI batch: a toast shows **`Thumbnails 42/177 — name`**, then
  a summary (`X rebuilt · Y failed · Z total`) and the tab reloads.
- **Thumbnail generation is now parallel** (`ThreadPoolExecutor`, `min(8, cpu)` by
  default, tunable via `asset_browser.thumb_workers`). PIL releases the GIL while
  decoding/resizing, so this speeds up the normal background indexing too, not just the
  new button.
- **Cache-busting**: rebuilt thumbnails keep the same URL, so the browser would have kept
  showing the old images — the SPA now appends a token after a rebuild.
- Defensive: a corrupt source counts as `failed` and the batch continues; a missing
  folder or a model with no preview yields no job instead of an error.
- The pre-existing Advanced ▸ Asset Browser "reindex" button (outputs only, synchronous)
  is unchanged.
- Files: `cz_assetbrowser.py` (`_ab_gen_thumbs` gains `force`/`progress`/`workers`,
  new `_thumb_jobs_for` + `rebuild_thumbs`, `_thumb_workers`), `cz_ui.py`
  (`_api_thumbs_rebuild` + `thumbs_rebuild` endpoint; the job registry and its endpoint
  are renamed `_BG_JOBS` / `job_progress` since they now serve three job types),
  `cz_assets.py` (button, handler, cache-buster), `tests/test_thumbs.py`.

## 1.10.2 — 2026-07-15 — Fix: the model SHA256 is cached (re-runs no longer re-hash the library)

`_compute_sha256` computed the hash but **never stored it**, so every batch pass re-read
every model in full just to obtain the same hash. Measured on a real library: **310 of
324 models have no `.metadata.json` sidecar → 416 GB re-read on each run.**

- The hash is now **persisted in `<name>.civitai.json`** (`sha256` + `sha256_size`) and
  reused. Lookup order: external `<name>.metadata.json` (Civitai-Helper convention) →
  our cache → compute (then cache).
- Cached **even when the model is unknown to CivitAI**, so those files stop being
  re-hashed on every pass too.
- **Invalidation**: the cache is rejected if the file size changed (model replaced /
  different version) → recompute.
- The fetch now **merges** the sidecar instead of overwriting it, so writing the CivitAI
  data no longer wipes the hash cache it had just saved. Sidecar writes (fetch +
  update-flag refresh) are now **atomic** (tmp + `os.replace`).
- `_needs_enrich` now tests `modelId` rather than "sidecar exists", so a sidecar holding
  only the cached hash is not mistaken for an enriched model.
- Net effect: the first pass still hashes what it must; **subsequent passes read no model
  bytes at all**.
- Files: `cz_civitai.py` (`_cached_sha256`, `_cache_sha256`, `model_sha256`, merged +
  atomic sidecar writes), `cz_civitai_batch.py` (`_needs_enrich`), `tests/test_civitai.py`
  (+4 tests: cache reused, stale-on-size-change, external sidecar wins, fetch keeps the
  cache).

## 1.10.1 — 2026-07-15 — Fix: example prompts were never fetched + API key ignored on some calls

Every CivitAI example was stored with an empty prompt (measured: **1130 / 1130**), so the
viewer showed "no prompt" for all of them.

- **Root cause**: examples came from the `/images` endpoint, which now returns
  **`"meta": null`** — CivitAI no longer publishes generation parameters there. The
  prompt was never in the response we were reading.
- **Fix**: the **`/model-versions/by-hash` response — which we already request — carries
  an `images` array with a *populated* `meta`** (prompt, steps, cfg, sampler…).
  `get_version_by_hash` now returns it and the fetch uses it, so prompts arrive with
  **zero extra requests** (`/images` is kept only as a fallback when a version has no
  showcase image). Verified end-to-end on a real model: **2/2 examples with prompt**.
- **API key was ignored on some calls**: `_api_get` only used a key when one was passed
  explicitly, so `get_latest_version` / `refresh_update_flag` (called by the batch with
  `api_key=None`) went out **anonymous** and missed gated/NSFW content. `_api_get` now
  falls back to the global key (UI → `preferences.json` → config).
- **HTTP errors are visible**: 401/403 (missing/invalid key) and 429 (rate limit) are now
  logged instead of being buried in debug — with a hint when no key is set.
- Missing prompts are now **honest**: examples carry a `has_prompt` flag and the viewer
  says *"the uploader did not publish the generation parameters for this image"* instead
  of implying a bug. The fetch message reports coverage (`3 example(s) (2 with prompt)`).
- **Backfilling existing sidecars**: previously fetched models have empty prompts. Re-run
  with `--all` to re-query metadata **without** re-downloading previews:
  `civitai_index.bat --kind all --all` (or `./civitai_index.sh --kind all --all`).
- Files: `cz_civitai.py` (`_api_get` key fallback + HTTP logging, `get_version_by_hash`
  images, new `_examples_from`), `cz_assetbrowser.py` / `cz_assets.py` (`has_prompt`),
  `tests/test_civitai.py` (+4 tests).

## 1.10.0 — 2026-07-15 — Negative LoRA weights + configurable weight range

The LoRA **Weight** sliders were hard-capped at `0..2`, so **negative weights were
impossible** — even though they are meaningful: a LoRA at a negative weight pushes *away*
from what it was trained on (a "skinny slider" at `-1` gives the opposite effect, an "age
slider" at `-0.5` swings the other way).

- Slider range is now **`-2..2` by default** and **configurable**:
  `"lora_weight_min": -2.0` / `"lora_weight_max": 2.0` in `config.txt`.
  Set `lora_weight_min` to `0` to forbid negatives.
- `default_lora_weight` is **clamped into the range**, so the slider can never start
  outside its own bounds.
- Defensive: non-numeric values, or `min >= max`, fall back to `-2..2` **and log why**
  (no silent surprise).
- The model layer never clamped weights (`set_loras`, X/Y/Z `LoRA weight` axis and the
  CLI `--lora NAME:WEIGHT` all pass floats straight through), so negatives work
  end-to-end — the UI slider was the only thing in the way.
- The LoRA panel now states the active range and that negatives invert the effect; both
  keys are documented in `config-sample.txt` and `config_modification_tutorial.txt`.
- Files: `cz_pipeline.py` (`_lora_weight_range`, `LORA_WEIGHT_MIN/MAX`, clamped
  `LORA_WEIGHT`), `cz_ui.py` (slider bounds + hint), `config-sample.txt`,
  `config_modification_tutorial.txt`, `tests/test_lora_weight_range.py`.

## 1.9.0 — 2026-07-15 — Switching Z-Image checkpoint reloads only the transformer

Same idea as the LoRA hot-swap (1.8.1), applied to the model itself. Switching from one
Z-Image checkpoint to another (**Z-Image checkpoint** dropdown, or the transformer
override) used to `free_vram()` and reload the **whole** pipeline — including the
**Qwen3-4B text encoder** and the VAE, which had not changed.

- When the **base repo and offload mode are unchanged** and only the **transformer**
  differs, `_ensure_base` now calls the new **`_swap_transformer`**: it loads *only* the
  new transformer and swaps it into the cached pipeline (`register_modules`), keeping the
  **VAE + Qwen3 text encoder + tokenizer + scheduler in VRAM**. The old transformer is
  freed (`empty_cache`).
- Covers all the "Z-Image → Z-Image" moves: single-file ↔ single-file, single-file →
  base repo's own transformer (clearing the override), and repo-subfolder overrides.
- Consistency taken care of: derived img2img/inpaint pipes (`from_pipe`) pointed at the
  **old** transformer → `_DERIVED` is cleared (rebuilding is free, weights are shared);
  LoRA adapters lived on the old transformer → they are **re-applied** to the new one.
  Under CPU offload the accelerate hooks are removed and re-attached around the swap.
- Safe fallback: any failure logs and falls back to the previous full reload.
  **Changing the base repo still reloads everything** (VAE/encoder genuinely change).
- New shared `_load_transformer()` used by both the full load and the swap.
- Files: `cz_pipeline.py` (`_load_transformer`, `_swap_transformer`, `_ensure_base`,
  `set_zimage_transformer`, `set_zimage_model`), `cz_ui.py` (status wording),
  `tests/test_model_swap.py` (7 tests incl. regression guards: a single-file switch must
  not free the pipe; a base-repo change must still free it).

## 1.8.1 — 2026-07-15 — Fix: switching a LoRA no longer reloads the whole model

Enabling / changing / removing a LoRA used to **reload the entire Z-Image pipeline**
(transformer + VAE + **Qwen3-4B text encoder**) — tens of seconds for what should be
instant, even though the model was already in VRAM.

- Cause: `set_loras()` called `free_vram()` (wiping `_BASE_PIPE`), and the base cache key
  included `tuple(LORAS)`, so any LoRA change invalidated the loaded pipeline.
- Fix: LoRAs are now **hot-swapped on the cached pipe** via the PEFT backend
  (new `_apply_loras`), and the cache key is back to `(repo, transformer, offload)`:
  - **weight-only change → `set_adapters`**, instant, nothing re-read from disk;
  - **different LoRA set → `unload_lora_weights` + reload of the LoRA files only** (~1 s);
  - derived pipes (img2img / inpaint, built with `from_pipe`) share the transformer, so
    they follow automatically.
- Safe fallback: if the hot-swap raises (e.g. missing PEFT backend), `_ensure_base` falls
  back to the previous full-reload path, so behaviour is never worse than before.
- Model/transformer/offload changes still reload, as they must.
- Files: `cz_pipeline.py` (`_apply_loras`, `_APPLIED_LORAS`, `set_loras`, `_ensure_base`,
  `free_vram`), `tests/test_lora_hotswap.py` (8 tests incl. regression guards: `set_loras`
  must not free the pipe, the cache key must not contain the LoRAs).

## 1.8.0 — 2026-07-14 — Batch CivitAI enrichment (.bat/.sh script + "Fetch all" button + new-version warnings)

Enrich a whole folder at once instead of one model at a time, from the UI **or** from a
standalone script you can run in parallel.

- **Standalone `cz_civitai_batch.py`** (imports no torch → starts instantly). Scans the
  LoRA / checkpoint folders and fetches the **missing** CivitAI info for each model
  (preview + trigger words + **example prompts**), skipping ones already done but still
  **refreshing the "newer version" flag**.
  ```
  python cz_civitai_batch.py --kind {loras,models,all} [--force] [--all]
         [--shard i/m] [--sleep 0.5] [--api-key KEY]
  ```
  `--shard i/m` splits the file list into disjoint subsets so **several processes can run
  in parallel**. Prints a per-model progress line + a final `enriched/skipped/updated/
  failed` summary; non-zero exit only if everything failed.
- **Wrappers**: `civitai_index.bat` / `.sh` (pass-through args, finds the venv Python,
  forces UTF-8) and `civitai_index_parallel.bat` / `.sh` (`[N]`, default 4) that launch
  **N parallel shards** — this is the intended "batch in parallel" workflow.
- **"🔄 Fetch all missing" button** in the Asset Browser (LoRAs / Models tabs): runs the
  same core in a background thread with a live toast (`Batch 12/48 — name…`), then a
  summary and catalog reload. New `civitai_fetch_all` Gradio endpoint (polled via the
  existing `civitai_progress`).
- **New-version warnings**: `fetch` and the batch now compare the local version to the
  latest on CivitAI (`get_latest_version`) and store `update_available` +
  `latest_versionName` in `<name>.civitai.json`. The Asset Browser shows a **⚠ update**
  badge on the card and a "Newer version on CivitAI: …" line in the lightbox.
- **Example prompts**: already captured since 1.7.2; the batch path reuses the same fetch,
  so they are filled in bulk too.
- **Config** `"civitai_batch": {"enabled": true, "sleep": 0.5, "check_updates": true}` —
  `enabled:false` hides the "Fetch all" button (the per-model 🔎 still works); `sleep` is
  rate-limit friendly; `check_updates:false` skips the extra version request.
- Files: `cz_civitai_batch.py` (new), `cz_civitai.py` (`get_latest_version`,
  `refresh_update_flag`, `update_available` in the sidecar), `cz_ui.py`
  (`civitai_fetch_all` endpoint), `cz_assetbrowser.py` (catalog `update`/`latest`, SPA
  render flag), `cz_assets.py` (button + toast + badge + version line),
  `civitai_index.bat/.sh`, `civitai_index_parallel.bat/.sh`, `config-sample.txt`,
  `tests/test_civitai_batch.py`.
- **Rate limits**: CivitAI throttles; keep parallel shards modest and set a CivitAI API key
  (Advanced) for heavy runs.

## 1.7.3 — 2026-07-14 — Jump to a LoRA / checkpoint in the Asset Browser (🖼️ icon)

Fooocus2026-style shortcut: a small **🖼️ icon** sits next to each **LoRA** dropdown
(**Advanced ▸ LoRA**) and next to the **Z-Image checkpoint** dropdown (**Advanced ▸
Models**). Clicking it opens the **Asset Browser in a new tab, already on the right source
tab and focused on that item** — its lightbox (preview + trigger words + example images
from 1.7.1/1.7.2) opens immediately.

- The browser is opened at `index.html?src=loras|models&focus=<file>`; the SPA reads the
  query on load, switches source, clears the folder filter and opens the matching card.
- The catalog is (re)built synchronously before the tab opens so the target is present.
- Base HF repos (Turbo/Base — no local file) just open the **Models** tab (nothing to
  focus). `None` LoRA slots open the **LoRAs** tab.
- Files: `cz_ui.py` (`_asset_focus_url` + 🖼️ buttons wired to each LoRA / the checkpoint),
  `cz_assets.py` (query parsing + `_tryFocus`).

## 1.7.2 — 2026-07-14 — CivitAI fetch: live progress + example viewer with prompts

Two UX fixes on the Asset Browser's **🔎 Fetch from CivitAI** button (1.7.1).

- **Live progress instead of a silent freeze.** The fetch now runs in a background
  thread and the model lightbox shows a status line with a spinner + progress bar that
  advances through the real phases: **`Hashing model file… 42%`** (a *real* byte-percentage
  — the only slow step, and only when there is no `<name>.metadata.json` sidecar) →
  `Querying CivitAI…` → `Fetching example images…` → `Downloading preview…`, then an inline
  ✅/⚠️ result (no more blocking `alert()`). The button is disabled while it runs.
  New Gradio endpoint `civitai_progress`; the client polls it every ~400 ms.
- **Example images are now clickable.** Each CivitAI example opens a full-screen viewer
  showing the image **large** with its **generation prompt** underneath (+ **Copy prompt**
  and *Open image*), and **← / →** (mouse or keyboard) to browse between examples. The
  example prompts were already downloaded into `<name>.civitai.json` — the catalog now
  carries them through (`{url, prompt, width, height}`) instead of the URL alone.
- No new dependency, no new config; purely additive to the existing button. Robust:
  any error is shown inline and never blocks the browser.
- Files: `cz_civitai.py` (`fetch_civitai_for_model(progress=…)` + real hash %),
  `cz_ui.py` (threaded job registry + `civitai_progress` endpoint),
  `cz_assetbrowser.py` (keep example prompts in the catalog),
  `cz_assets.py` (status bar, polling, example viewer + CSS).

## 1.7.1 — 2026-07-12 — Asset Browser: CivitAI enrichment (previews / trigger words / examples)

- New **`cz_civitai.py`** (technique from Fooocus2026): looks a model/LoRA up on **CivitAI
  by its SHA256** — read from the sibling `<name>.metadata.json` when present, so multi-GB
  checkpoints are **not** re-hashed — then fetches **trigger words** + top **example images**
  and saves `<name>.preview.png` (the sidecar convention the Asset Browser already scans) +
  `<name>.civitai.json`.
- **Asset Browser** (LoRAs / Models tabs): a **🔎 Fetch from CivitAI** button in the model
  lightbox (a `civitai_fetch` Gradio API endpoint) downloads the preview + trigger words,
  rebuilds the catalog and reloads — the placeholder becomes a real preview. The lightbox
  now shows **example images** + a **CivitAI page** link; the catalog reads trigger words
  from `<name>.civitai.json` (falling back to the safetensors header).
- Optional **CivitAI API key** — paste it in **Advanced > CivitAI access** (saved to
  `preferences.json`) or set `civitai_api_key` in config; for gated/NSFW previews and to
  avoid rate limits. Most public models work without one.
- Files: `cz_civitai.py`, `cz_assetbrowser.py`, `cz_ui.py`, `cz_assets.py`, `cz_core.py`
  (`APP_VERSION` 1.7.1), `config-sample.txt`, `config_modification_tutorial.txt`.

## 1.7.0 — 2026-07-12 — Presets, seed reuse, Advanced tab, PNG Info, a1111 metadata, Asset Browser overhaul

A large UI/UX pass (all in the `cz_*` modules).

- **Presets (Fooocus-style)** — new *⭐ Presets* accordion (Settings). A preset bundles
  prompt/negative, styles, size, steps/CFG, sampler/schedule, image number, checkpoint,
  transformer override and LoRAs into `presets/<name>.json`. **Load** applies the widgets
  AND the model/LoRAs (a chained silent checkpoint apply keeps the preset's steps/CFG);
  **Save as new / Update selected / Delete / refresh**. `presets/` gitignored except
  `example.json`.
- **Seed management** — *♻️ Reuse last seed* button (refills the field with the previous
  render's real seed) + *Fix seed (no +1 per image)* toggle. A `-1` random seed is now
  resolved to a concrete value before generation, so the metadata stores the real seed
  (previously it saved `-1`).
- **Advanced tab** — new *Advanced* tab (after Save) for advanced settings; the *Hugging
  Face access (gated models)* block moved here from Models.
- **Input Image → PNG Info** — a "Read prompt / metadata from an image" reader (a filepath
  uploader that preserves PNG chunks) parses crispz, **A1111/Civitai** (`parameters`) and
  ComfyUI metadata, with *Send prompt* / *Send seed* to the fields.
- **Metadata scheme** (`metadata_scheme`, Advanced > Metadata) — `crispz` (default) or
  `a1111`, which also writes an A1111/Civitai `parameters` PNG chunk so **Civitai reads the
  prompt/seed/params** on upload (crispz chunk + sidecar kept in both).
- **Read wildcards in order** (`wildcards_in_order`, Advanced > Generation) — a batch sweeps
  each wildcard file line by line (deterministic) instead of picking random lines.
- **Also save pre-upscale image** (`save_pre_upscale`) — in txt2img + auto-upscale, also
  save the base txt2img image (before ESRGAN/refine), tagged `txt2img`.
- **Configurable LoRA slots** (`lora_slots`, default 3) — 1–10 slots; a live slider in
  Advanced > Generation shows/hides them (persisted in `preferences.json`).
- **Asset Browser overhaul** — the output gallery now opens as a **standalone page in a new
  tab** via a button; **instant open** (manifest written immediately, thumbnails generated
  in the background behind a shimmer placeholder that swaps to the real thumbnail); images
  save into **`out/YYYY-MM-DD/`** date subfolders (`date_subfolders`, recursive scan);
  **per-image delete**; a **subfolder sidebar** with counts, per-folder **hide** and a
  **Hidden** toggle (persisted in localStorage), defaulting to the current day; **keyword
  search** over the embedded metadata; and **Outputs / LoRAs / Models** source tabs
  (LoRAs/Models show a Civitai preview if one sits next to the `.safetensors`, else a
  placeholder + trigger words).
- Files: `cz_ui.py`, `cz_assets.py`, `cz_assetbrowser.py`, `cz_imageio.py`, `cz_prompt.py`,
  `cz_pipeline.py`, `cz_cli.py`, `cz_core.py` (`APP_VERSION` 1.7.0), `config-sample.txt`,
  `config_modification_tutorial.txt`, `presets/example.json`.

## 1.6.0 — 2026-07-07 — Model-loading progress in the terminal and UI

The first model load downloads from Hugging Face and then reads several GB into VRAM —
previously a long silent gap (the report was `317.3s` with no sign of progress). The
blocking `from_pretrained` now runs in a daemon thread while a heartbeat (every ~2 s)
reports where the load is:

- **Terminal**: a single rewritten line `[crispz][load] Z-Image base... 45s | 3.2 GB in
  VRAM` (during the first-run download, before anything is allocated, it reads
  `... 12s (downloading / reading, first run only)`).
- **UI**: the Gradio progress bar advances — honest %, based on **VRAM allocated /
  `target_vram_gb`** once loading into memory starts (capped 95 %), a small time-based
  bar during the download phase.
- Applied to the three heavy loads: **Z-Image base**, the **single-file transformer**
  (Civitai checkpoint), and **Z-Image Omni**.
- **Zero-cost off**: `"load_progress": {"enabled": false}` loads directly with no monitor
  thread. `target_vram_gb` (default 14) and `heartbeat_s` (default 2) are tunable.
- The monitor never swallows errors — a failed load re-raises exactly as before.
- Files: `cz_pipeline.py` (`_load_monitor` + pure `_fmt_load`/`_load_pct`), `cz_core.py`
  (`APP_VERSION` 1.6.0), `config-sample.txt`, `tests/test_load.py`.

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
