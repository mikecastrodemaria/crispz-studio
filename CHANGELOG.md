# Changelog

All notable changes to crispz-studio. One versioned entry per feature.
The app version lives in `cz_core.py` (`APP_VERSION`) and is shown in the browser tab title.

## Unreleased — AI provenance: C2PA reading + TrustMark invisible watermark (EU AI Act art. 50)

New optional brick `cz_provenance.py` (CPU only, the GPU is never touched), for
machine-readable AI disclosure as required by EU AI Act Article 50 (applicable
Aug 2 2026; systems already on the market have until Dec 2 2026):

- **Read** (PNG Info + CLI `--provenance -i img.png`): a **Provenance** section shows
  any embedded **C2PA / Content Credentials** manifest (issuer, claim generator,
  signature state — Firefly/ChatGPT/Gemini outputs carry these), automatically and for
  free; the **🔍 Check invisible watermark** button decodes a **TrustMark** watermark
  on demand (lazy model init ~4 s once, then ~0.1 s/image). The UI wording is
  deliberate: *absence of marks proves nothing* — it never claims "not AI"/"authentic".
- **Write** (`save_image()`, the single choke point all saves go through — txt2img,
  upscale, inpaint, reframe, queue, CLI, HTTP endpoints): with
  `provenance_watermark: "on"` every saved image gets an invisible TrustMark watermark
  carrying `provenance_wm_id` (**max 9 ASCII chars** — the error-corrected payload is
  ~68 bits; longer ids are truncated). Verified by test to survive PNG **and JPEG q95**
  re-encoding. Default **off**.
- Deps: `trustmark` + `c2pa-python` in `requirements-extra.txt` (graceful degradation
  when absent, same pattern as rembg). Windows note: install trustmark with
  `PYTHONUTF8=1` (its sdist crashes on cp1252). Tests: `tests/test_provenance.py`
  (7 cases, skip-if-missing so CI stays light). C2PA *writing* (signed manifests) is
  deliberately not included yet — it needs a signing-certificate decision first.
- Caution: TrustMark loads a small torch model **in-process** (CPU). Given the
  resident-YOLO offload-corruption precedent below, watch the first renders with
  `provenance_watermark: on` + GGUF/offload `model`; the watermark hook runs at save
  time only, and stays off by default.
## Unreleased — `<lora:name:weight>` in the prompt + CivitAI search for missing LoRAs

A LoRA can now be called **directly from the prompt** (A1111 syntax): `<lora:my_lora>`
or `<lora:my_lora:0.8>`, in the UI, the CLI (`--prompt`) and the HTTP server. The tag
is stripped **before text encoding** (a syntax fragment must never reach the encoder),
the name is resolved in the LoRA folder (case-insensitive; file name, stem, or unique
partial match — ambiguous names are refused rather than guessed) and the file is applied
**on top of the LoRA slots** via the existing hot-swap (no model reload). Same file in a
slot and in the prompt → the prompt weight wins; no weight → `default_lora_weight`;
out-of-range → clamped to `lora_weight_min..max`. The effective list (slots + prompt) is
what lands in the image metadata, so renders stay reproducible from their sidecar.
A tag injected by a **wildcard** is stripped too but not applied (the run's LoRAs are
fixed before wildcard expansion). Config: `prompt_lora_tags` (default true).

**Missing LoRA = no render, never a silent ignore**: the run stops before any model
load, names the missing file, and pre-fills the new **Models > 🧩 LoRA > 🔎 Search
CivitAI** panel — search by name (`/models?query=…&types=LORA`), one candidate per
version with its **base model** (Z-Image first, strict filter by default), **⬇
Download** into the LoRA folder with streamed **SHA256 verification** (mismatch →
file deleted + clear message), hash cached in the `.civitai.json` sidecar, preview +
trigger words fetched, slot dropdowns refreshed. CLI exits with code 2 and the same
guidance; the server answers HTTP 400.

## Unreleased — the REAL mechanism: a resident torch YOLO model corrupts offload transfers

The CPU-detection fix was not the end of it. On the GGUF path (offload `model`, forced
for every GGUF) the corruption came back — hands pass fine, next render mosaic, then
NaN (`RuntimeWarning: invalid value encountered in cast` in diffusers'
`image_processor`). A systematic bisection in isolated processes (one scenario per
process, since the poison survives within a process) established, deterministically:

| scenario (Q8_0 GGUF, offload model) | renders after |
|---|---|
| control / detect-only / refine-only / deep imports / RNG advance / raw `torch.load` of the .pt | clean, bit-stable |
| **`YOLO(path)` loaded (no predict, CPU) + 2 crop refines** | **mosaic, degrading** |

**Checksum proof**: in a clean process the text encoder's total weight sum is stable
across CPU↔GPU offload trips (7.223877e7, drift ≤ 1.5e-8) and the same sentence always
encodes to the same embedding. With the torch YOLO model merely **resident in
memory** — never used — the weights read 7.341337e7 right after the refines and keep
drifting on every subsequent call (7.3456e7, 7.3686e7): **the offload transfers
corrupt the shared model weights in memory**, progressively — which is exactly why
renders degrade from subtle drift to mosaic to NaN/black, why the damage survives
checkbox toggles, and why only a restart cures it. Reproduced identically on Q6_K,
**and on a safetensors checkpoint (sickOllie) with offload `model` forced** — so this
is not a GGUF bug: the trigger is the **offload path** (GGUF merely forces it), and a
GGUF was simply the only configuration running offload `model` in daily use.
No torch global flag, no thread, no env var changes: the corruption vector is the mere
presence of the ultralytics-built module during transfers (allocator-layout /
transfer-race sensitivity; exact torch/WDDM internals not pinned down).

**Fix — total isolation**: `ultralytics` is never imported in the app process anymore.
The hand detector is exported **once to ONNX** (`cache/<model>.onnx`) in a
**subprocess** (where ultralytics lives and dies), and runtime detection is a pure
`onnxruntime` session — the same stack insightface has used all along without a single
incident — with letterbox preprocessing and numpy NMS. `hand_detailer_device` now
selects the onnxruntime provider (`cpu` default). One-time export needs `ultralytics`
+ `onnx` (requirements-extra.txt); absent → clear log, no crash.

**Validated**: full hands pass and face+hands pass on Q8_0/offload model → following
renders bit-stable (0.64/0.64/0.64); positive control with the old in-process torch
load on the same build still poisons (0.64→0.36→0.30), proving the harness catches it.
ONNX detection finds the same hands as the torch predictor on the reference image.

## Unreleased — `_effective_offload(None)` ambiguity re-fixed after the revert

Re-lands a real fix that was swept away by the ControlNet revert (dc66910). `None` is a
*legitimate* value for `tpath` (no override → the base repo's transformer), but it also
served as the "use the current transformer" sentinel — so `_effective_offload(None)`
silently evaluated `ZIMAGE_TRANSFORMER`. `_swap_transformer`'s guard therefore compared
the **new** transformer with itself and let a base-repo → GGUF hot-swap through, even
though the effective offload changes (`none` → `model`, forced for every GGUF) and a
full reload is required. Now uses a dedicated sentinel object; regression test in
`tests/test_model_swap.py`.

## Unreleased — mosaic corruption SOLVED: the hand detector poisoned the GPU

The mystery corruption that plagued 2026-08-16/17 — renders coming out as mosaic
garbage, unrelated to the prompt, no error, until restart — is root-caused and fixed.

**The culprit: the hand detailer's YOLO detection running on the GPU.** The ultralytics
`predict()` call poisons the process CUDA/torch state: the render *during* which the
hand pass runs comes out fine, and **every diffusion render after it is destroyed**.
Nailed by a controlled matrix on 2026-08-17 with per-render execution metadata
(`detail_*_run` sidecar keys): clean base render → hands pass `H:1` (one hand refined,
image clean) → next render total garbage, reproducible, user-confirmed
(base OK, base+face OK, base+hands KO). The refine code shared with the face detailer
is innocent (identical code path, weeks of clean service with insightface detection;
refine-on-crop repro without YOLO also clean).

This retroactively explains the "first render clean, everything after corrupted until
restart" signature seen since the hand detailer shipped (d6143c0, 2026-08-16 13:15 —
the corruption's first-ever appearance was the same day at 17:32), across all models
(GGUF, sickOllie, base repo) and through every unrelated revert.

**Fix**: hand detection now runs on **CPU** by default (`hand_detailer_device: "cpu"`).
The YOLOv8n model is 6 MB; CPU detection costs ~0.1 s per image. `"cuda"` remains
accepted to re-test the conflict after an ultralytics/torch upgrade. The exact
mechanism inside ultralytics/torch is still unidentified — the isolation is the fix.

**New forensics** (kept): every sidecar now records `detail_faces` / `detail_hands`
(checkbox state), `detail_faces_run` / `detail_hands_run` (-1 not attempted, -2 pass
crashed, ≥0 regions actually refined) and `offload` (configured/effective). Note for
test matrices: the job queue does NOT snapshot the detailer checkboxes (module flags
read at run time) — use direct Generate clicks when varying them per step.

## Unreleased — ControlNet Tile: root cause found, feature stays OFF

`CONTROLNET_TILE_AVAILABLE` remains **False**. It was flipped on to resume the
investigation, and the investigation ended it: **the tiled ControlNet refine recopies
the scene into every tile.**

Seen plainly on a 1024×1536 → 2048×3072 upscale of a living room: the parquet ends up
littered with miniature armchairs and fireplaces, walls and armchairs get carved
patterns, and with an empty per-tile prompt whole hands, hair and fake signage appear.
Four configurations tried, **all bad** — empty prompt @0.75, scene prompt @0.75, @1.0
and @1.3.

**Why, from the diffusers 0.39 source.** `ZImageControlNetPipeline` has no `strength`
parameter — the word does not appear once in the file, against 13 times in
`pipeline_z_image_img2img.py` — and there is no `pipeline_z_image_controlnet_img2img`
(only `controlnet` and `controlnet_inpaint`). So the pipeline **denoises fully from
noise**. Applied to a *tile*, that means generating a complete image whose only anchor is
the control image, and the model duly composes an entire scene inside each tile. Plain
img2img cannot do this: it starts from the real tile at denoise 0.35, so it can only
retouch. This is the approach failing under tiling, not a plumbing defect.

**The control experiment confirms it.** The same pass run on the *whole* image
(1024×1536, `refine_tile=0`, scale 0.75) comes out **clean** — no recopying, no miniature
furniture, structure identical (fireplace, painting, both armchairs, herringbone). One
"tile" that *is* the image, so there is nothing to recopy: the culprit is the tiling, not
the ControlNet. But it took **559 s** (9 min 20) against 52 s for the 15 tiles —
transformer (12 GB) + ControlNet (6.7 GB) + whole-image activations spill into shared
RAM. So the only mode that produces a good result is unusable in practice on 32 GB.

**What would unblock it**: a `ZImageControlNetImg2ImgPipeline` (ControlNet + *partial*
denoise) — it would make tiling sound again, since each tile would restart from its real
content instead of being generated from scratch. Worth watching in diffusers.

**The guard cannot see it.** The offending tiles scored gap 20–45 and correlation
+0.21/+0.99, comfortably inside the thresholds (75 / 0.20). `_cn_structure_gap` measures
*low-frequency* (64×64) agreement per tile, and a tile that repopulates a parquet with
small furniture keeps the same tonal distribution and the same coarse structure.

**The originally reported bug did not reproduce.** Replayed faithfully, the shared-state
diff prints `unchanged` before and after every ControlNet pass and the following txt2img
renders are clean. The "palace facade" image that motivated the report was most likely
**the upscale output itself** — that is, exactly the recopying described above — rather
than a poisoned pipeline.

**Still open, unrelated to the ControlNet**: on the 2026-08-16 outputs, 7 consecutive
txt2img (17:32:58–17:35:52) and 3 more (17:58:14–17:59:34) came out corrupted. Not
reproduced. Lead and instrumentation below.

**What the output metadata actually shows** (renders of 2026-08-16, measured with a
high-frequency-energy ratio over every PNG of the day — normal renders sit at
0.05–0.28, corrupted ones at 0.38–0.56):

| window | checkpoint | verdict |
|---|---|---|
| 13:00–14:30 | `z_image_turbo-Q6_K.gguf` | clean (≈45 renders) |
| 15:26–16:01 | `Tongyi-MAI/Z-Image-Turbo` | clean |
| 17:19–17:32:49 | `sickOllie_v1.safetensors` | clean |
| **17:32:58–17:35:52** | `sickOllie_v1.safetensors` | **corrupted, 7 renders in a row** |
| 17:57:17 | `z_image_turbo-Q6_K.gguf` | clean (1st render after load) |
| **17:58:14–17:59:34** | `z_image_turbo-Q6_K.gguf` | **corrupted, 3 renders in a row** |

The 17:58–17:59 series came out *after* the commit that disables the ControlNet
(17:42:56) — only conclusive if the app was restarted in between, which is not recorded.
What is certain is that this second corruption is **not** the ControlNet forward pass
mutating shared state (proof below); it is a distinct problem that happens to have
surfaced in the same working window.

**The signature**: structure stays globally plausible while local texture is destroyed
into a mosaic, and the content drifts away from the prompt — with no error, until a full
reload (restart, or a checkpoint switch that forces one) clears it. On 2026-08-16 it also
*ramped in* rather than flipping on (17:32:03 → 0.17, 17:32:33 → 0.28, 17:32:49 → 0.23,
17:32:58 → 0.52), which points at a resource margin being eaten, not a state flip.

**The reported repro no longer reproduces.** Replayed faithfully on the current code
(sickOllie, offload `none`, 19.3 GB resident → ControlNet upscale to 2048×3072, 15 tiles
at 3.1–3.3 s → three plain txt2img): the shared-state diff prints `unchanged (31 keys)`
before and after **every** ControlNet pass and at every txt2img entry, and the three
following renders score 0.135 / 0.132 / 0.114 — the same range as the two before it
(0.122 / 0.130). The fixes already committed plus the ones below appear to have closed
the path; the instrumentation stays in so the next occurrence names its own cause.

**What that run did expose — the VRAM margin.** Right after the upscale, a plain
1024×1536 txt2img ran with `alloc 19.3 GB` but **`reserved 28.7 / 32 GB`**, and peaked at
27.4 allocated / 28.7 reserved versus 21.7 / 23.9 for the identical render *before* the
upscale. Under Windows/WDDM reserved VRAM is taken from the other processes on the card,
so every render after an upscale was starting with ~3 GB of headroom on a **GPU shared
with the user** — precisely the regime where the driver spills into shared RAM and
latents come out corrupted with no error.

**Ruled out, with proof:**
- the ControlNet does **not** alter the transformer. `from_transformer` only grafts
  *shared references* into the ControlNet. Verified empirically on miniature CPU models:
  after grafting → ControlNet forward → transformer forward → release, all 185 state
  keys of the transformer are unchanged and a reference pass is **bit-for-bit
  identical**;
- the prompt reaches the pipeline intact — `cz_ui` passes the same `fp` to both
  `txt2img_run` and `_gen_meta`, so "correct metadata + unrelated image" means the model
  received the prompt and ignored it;
- the release path no longer moves anything to CPU (separate bug, fixed);
- VRAM saturation (fixed by the tile cap: 31 GB / 164 s per tile → 28.2 GB / 4.5 s).

**Main lead — the pipes derived by `from_pipe`.** `DiffusionPipeline.from_pipe` does
`torch_dtype = kwargs.pop("torch_dtype", torch.float32)` then `new_pipeline.to(dtype)`,
and the components are **shared with the base** — so deriving a pipe recasts the base's
transformer. In offload `model` (forced for a GGUF) the derived pipe also lacks the
base's accelerate hooks (`_all_hooks` empty → `maybe_free_model_hooks` does nothing),
which leaves the base's offload chain inconsistent for the *next* generation. The face
and hand detailers derive `img2img` **after** the first render — which matches
"1st render clean, 2nd onwards corrupted" exactly.

### Fixed along the way

- **`generate()` started a render on a full allocator.** It emptied the CUDA cache
  *after* the pass but not before, so a txt2img that followed an upscale began with
  ~9 GB of freed-but-reserved blocks still held (28.7 / 32 GB measured). It now does the
  same `gc.collect()` + `empty_cache()` before the pass that `_refine_tiled` has done
  since d1761c2 — measured: the render now starts at 19.6 GB instead of 28.7 GB.
  **Only when offload is off.** A first version of this ran unconditionally and produced
  **entirely black images from the second render onwards** on `z_image_turbo-Q6_K.gguf`
  at 832×1216 (render #1 correct, #2 and #3 `max=0`), while the same sequence on the base
  repo without offload stayed fine. In offload `model` — forced for every GGUF — the
  weights round-trip CPU↔GPU on each forward, and this cleanup between two renders breaks
  them. There is nothing to gain there anyway: under offload the pipeline tops out around
  8.5 GB, and the VRAM margin this cleanup targets only exists when the whole model stays
  resident.
- **`_effective_offload(None)` was ambiguous.** `None` is a *legitimate* value for
  `tpath` (no override → the base repo's transformer), but it was also the "use the
  current transformer" sentinel — so `_effective_offload(None)` silently evaluated
  `ZIMAGE_TRANSFORMER`. `_swap_transformer`'s guard therefore compared the **new**
  transformer with itself and let a base-repo → GGUF hot-swap through, even though the
  effective offload changes (`none` → `model`) and a full reload is required. Now uses a
  dedicated sentinel object. Regression test in `tests/test_model_swap.py`.
- **`_swap_transformer` did not release the ControlNet.** It cleared `_DERIVED` before
  `del old` precisely so nothing would keep the old transformer alive — but `_CN_PIPE`
  holds it too (and `from_transformer` grafted its embedders). The old transformer
  (12 GB) survived the swap, which is the very shared-RAM spill that block exists to
  prevent.

### New: shared-state diff (`--log-level debug`)

`_shared_state_diff()` photographs everything shared between the pipelines — weights
(sampled per module, with NaN/Inf detection), dtypes, devices, accelerate hooks,
scheduler, VAE flags, rope state, object identities — plus a **functional probe**: a
fixed sentinel sentence is pushed through the text encoder and its embedding
fingerprinted. Same sentence must always give the same fingerprint; if it moves, the
shared text encoder is damaged, which is what produces a coherent image that ignores the
prompt. Only what changed is logged, so the first `CHANGED` line names the culprit.
Wired into `generate()`, `get_pipe()` (right after a `from_pipe` derivation), and around
the ControlNet pass and release.

## Unreleased — 🔒 ControlNet Tile refine (structure-locked upscale)

The refine pass can now run through the official **Z-Image Tile ControlNet**
(`alibaba-pai`, distilled for 8 steps) instead of plain img2img: every diffusion step is
conditioned on the source image, so composition, faces and text stay in place and only
detail is regenerated. This is the fix for "my upscale changed the picture" and for tile
duplication at high denoise — `controlnet_tile_scale` (0.75) replaces `denoise` as the
fidelity knob.

The ControlNet pipeline is built **on the components already in VRAM** (VAE, text
encoder, transformer): nothing is loaded twice apart from the ControlNet itself, and it
is rebuilt automatically when you swap checkpoints (it holds shared references to the
transformer's embedders). UI: *ControlNet Tile refine* + strength slider in the Upscale
tab. CLI: `--controlnet-tile`, `--controlnet-scale`. Config: `controlnet_tile`,
`controlnet_tile_model` (repo/file or a local path), `controlnet_tile_scale`.

**Two traps found while testing, both handled:**
- Calling `enable_model_cpu_offload()` on this pipeline leaves the ControlNet on CPU
  (its components already carry the base pipeline's accelerate hooks) →
  `mat1 is on cuda:0, other tensors on cpu`. The ControlNet is now simply placed on the
  GPU once and the shared components keep the base's hooks.
- **A ControlNet only works with a transformer of the lineage it was trained on, and an
  incompatible checkpoint returns pure noise instead of an error.** The pass now
  measures the low-frequency gap between its output and the control image (~35 when it
  works, ~107 when it doesn't, measured) and, past `controlnet_tile_sanity_max` (60),
  discards the result, explains why, and falls back to the normal img2img refine — an
  upscale never returns a broken image because of this.

Validated: identical results between the shared-component pipeline and a stock diffusers
one built from scratch (sharpness 985 vs 1002, structure gap 44.4 vs 45.4).

## Unreleased — 🖐 Hand detailer

Hands are the weak point of every diffusion model, so the ADetailer-style pass now
works on them too: **🖐 Detail hands** next to *Detail faces* runs the same circuit
(detect → expanded crop → upscale to the model's sweet spot → img2img refine → feathered
elliptical paste) with a **tighter margin** (0.35 vs 0.6 — widening it would regenerate
the forearm and the background) and a **higher default denoise** (0.4 — fingers need
more than a face). Faces are processed first, then hands.

Detection uses a **YOLOv8 hand model** through `ultralytics`, an **optional** dependency
(`requirements-extra.txt`): without it the feature logs an actionable message and does
nothing — the face detailer is untouched (it uses insightface). The ~6 MB model is
pulled once from `Bingsu/adetailer`; `hand_detailer_model` also accepts an absolute path
to your own `.pt`. Config: `hand_detailer`, `hand_detailer_denoise`,
`hand_detailer_max_hands`, `hand_detailer_margin`, `hand_detailer_conf`. CLI:
`--detail-hands`, `--hand-denoise`. Also exposed on the server's `/txt2img`.

Validated on a real render (768×1024, both palms visible): 2 hands detected in 1.1 s,
both refined in 9.6 s, mean pixel difference **6.5 inside the hand boxes and 0.0000
everywhere else** — palm creases and skin texture appear, the face and clothes are
byte-identical.

## Unreleased — Persistent queue, HTTP txt2img/edit endpoints, real CI, slicing fix

- **The job queue survives a restart.** It is written to `cache/queue.json` on every
  mutation *and* after each finished job, then restored at startup (the accordion opens
  itself and says how many jobs came back). Input images are stored beside it and
  reloaded; the session gallery is not persisted; anything unserialisable degrades to
  `None` instead of losing the whole file. Disable with `job_queue.persist: false`.
- **`--serve` grew past `/upscale`**: **`POST /txt2img`** (prompt, size, steps,
  guidance, sampler/schedule, optional chained upscale and face detailer) and
  **`POST /edit`** (`mode: inpaint | expand | reframe` — mask file, directional sides,
  or target ratio with contain/cover). Same engine as the UI, model stays warm between
  calls. Verified live against a running server: txt2img 512², expand left+right
  512→818 px, reframe cover 1376×768, and a bad mode returns a clean 400.
- **CI actually tests now.** The workflow byte-compiles every module (was: `app.py`
  only), validates the JSON files, checks that **every `CONFIG.get()` key is documented
  in `config-sample.txt`** (`tools/check_config_keys.py` — it immediately found two
  undocumented keys), and runs the whole suite on CPU torch. New `tools/run_tests.py`
  runs the suite locally in isolated processes (15/15 in ~67 s).
- **Fix: attention slicing was a no-op.** `_set_slicing` called
  `pipe.enable_attention_slicing()` above 1664 px, but `DiffusionPipeline` only forwards
  that to modules exposing `set_attention_slice` — and neither `ZImageTransformer2DModel`
  nor `AutoencoderKL` define it (verified on diffusers 0.39.0.dev0), so the call was
  silently dropped. It now (re)asserts **VAE tiling/slicing**, which is the mechanism
  that actually caps the 2K+ memory peak.

## Unreleased — Dequant disk cache, GPU-busy guard, format badges

Three quality-of-life fixes born from a 7-checkpoint benchmark session.

- **Dequant cache** — a FP8/INT8 checkpoint had to be converted to bf16 on *every*
  load (~5-6 min for 5.7 GB off an HDD). The bf16 result is now written once to
  `<app>/cache/dequant/` and later loads read it back as a plain single-file. Keyed on
  path + size + mtime (a replaced file never reuses a stale entry), atomic write
  (`.tmp` + rename), and an LRU cap (`dequant_cache_max_gb`, default 60 GB, 0 =
  unlimited). Config `dequant_cache`: `auto` (default) / a custom path / `off`.
- **GPU-busy guard** — two processes sharing the GPU silently push renders into shared
  RAM (measured: 1.7 s/step → 300-600 s/step, then a crash). Before loading, the app
  now compares the device's real free VRAM against what this process reserved and warns
  when someone else holds more than `gpu_busy_warn_gb` (default 2 GB): console, CLI
  stderr, and the Models status line. 0 disables it.
- **Format badges** — the checkpoint dropdown now shows `[BF16 · 11.5 GB]`,
  `[GGUF Q6_K · 5.5 GB]`, `[FP8→bf16 · 5.7 GB (slow 1st load)]` or `(cached)` next to
  each model, so the cost of a switch is visible before clicking. The dropdown *value*
  stays the raw file name — presets, XYZ, CLI and prefs are untouched. Safetensors
  headers are memoised per (path, size, mtime), so listing + badges + loading share a
  single read.

Tests: `tests/test_dequant_cache.py` (9 cases — key stability, roundtrip, disabled
mode, LRU eviction, keep-fresh-entry, header memoisation, badges, guard thresholds).
Measured on a real 5.7 GB FP8 off the HDD: **first load 249.9 s** (238 s of dequant +
8 s writing the 11.5 GB cache), **second load 0.2 s** to build the transformer — the
dequant pass is gone and the checkpoint behaves like any BF16 single-file. Weights
byte-identical between the two passes.

## Unreleased — Fix: hot-swap freed the old transformer too late (VRAM spill)

**Why.** On a multi-checkpoint XYZ grid, the second swap put the machine in the mud:
the NEW transformer (12 GB) was moved to the GPU **before** the old one (12 GB) was
freed — with the VAE + Qwen3 encoder (~7 GB) that overflows 32 GB of VRAM into shared
system RAM, and PyTorch never recovers: renders went from 1.7 s/step to **300-600
s/step** (68 min for 8 steps) until the process died mid-grid.

**What.** `_swap_transformer` now purges the derived pipes and deletes the old
transformer (+ `empty_cache`) **before** placing the new one on the GPU — peak VRAM
during a swap is one transformer, not two. Ported to the whole family. Model-swap
tests green, and the same 7-checkpoint grid (3× FP8, ConvRot INT8, AIO bundle, GGUF
Q4, bf16 reference) then completed end-to-end with every render back to **3-14 s**
and all seven portraits clean on the contact sheet.

## Unreleased — XYZ grid: full-Prompt A/B axis + type-ahead suggestions

**Why.** Comparing whole prompts needed Prompt S/R gymnastics, and filling the
Checkpoint/LoRA value fields meant copy-pasting long file names by hand.

**What.**
- **New `Prompt` axis**: each value is a **complete prompt** (quotes protect embedded
  commas) — true A/B/C testing, combinable with any other axis (e.g. Prompt ×
  Checkpoint). CLI too: `--xyz "Prompt=a, \"b, with comma\", c"`, and `--prompt`
  becomes optional when a Prompt axis is given. Sheet/job labels truncate long prompts.
  NB: the case-insensitive shorthand `prompt` now resolves to this axis (exact match
  wins); `prompt s` still reaches `Prompt S/R`.
- **Type-ahead in the X/Y/Z value fields** (after 3 typed characters, ↑/↓ + Tab/Enter,
  Escape, click): suggests from the **checkpoint/LoRA lists validated at startup** on
  the `Checkpoint` / `LoRA` / `LoRA + weight` axes (`:1` auto-appended for the latter),
  and from the local **`__wildcards__`** on the `Prompt` / `Prompt S/R` axes when the
  current token starts with `__`. CSV segments are respected (completion only touches
  the segment being typed). Disable with `xyz_grid.suggest: false` (same key as the
  ⤵ suggest button). Validated live in the browser: checkpoint filter + insert, second
  segment after a comma, wildcard expansion inside a prompt, `:1` suffix.

## Unreleased — Fix: ConvRot INT8 checkpoints rendered pure noise

**Why.** `redcraft22INT8INT4_redzit222026HD` loaded structurally but rendered a pixel
mosaic: its `comfy_quant` blobs declare
`{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}` — the ComfyUI
**ConvRot** scheme stores weights ROTATED (grouped Hadamard on the input dim,
anti-outlier) and un-rotates activations at runtime. Plain `int8 × scale` dequant
yields rotated weights = total noise.

**What.** The dequant loader now parses the `comfy_quant` JSON blobs and, when
`convrot` is declared, **applies the inverse rotation** after descaling
(`(w.view(out, in/g, g) @ H).reshape(...)`). Crucial detail: comfy-quants'
`regular_hadamard` is **not** the Sylvester construction — it is a specific **H4 base
extended by Kronecker products** (hence "groupsize is a power of 4"), orthonormal and
symmetric; with a Sylvester matrix the correlation to the base weights stays ~0.
Validated three ways: synthetic roundtrip (export recipe reproduced, max error < 2 %),
**correlation 1.00** between the dequantized redzit layer and the base Z-Image-Turbo
weights (0.004 before the fix), and a clean GPU render from the previously-broken
checkpoint. Ported to the qwen-edit and krea loaders.

## Unreleased — CLI parity: expand / inpaint-mask / reframe-fit / force-ratio flags

**Why.** The Inpaint/Outpaint tab features and the forced-ratio radio had no CLI
equivalents — batch/headless users couldn't reach them.

**What.** New flags, same behaviour as the UI:
- **`--expand left,right,top,bottom`** (or `all`) + `--expand-ratio` (default 0.3): the
  "Expand sides" directional outpaint on `-i`, then exit.
- **`--inpaint-mask mask.png`** + `--inpaint-denoise` (default 1.0): repaint the WHITE
  area of the mask on `-i`, guided by `--prompt`.
- **`--reframe-fit contain|cover`**: `--reframe` now exposes both modes (contain =
  outpaint to the ratio, default; cover = fill + centre-crop, no generation).
- **`--force-ratio W:H`** + **`--force-ratio-mode crop|extend`**: the forced-ratio
  radio for Upscale/img2img runs; the `-o` single-file path now honours it too (it
  ignored `FORCE_RATIO` while the standard path applied it).
Validated end-to-end: crop 1024² → 1024×576, cover 1376×768 (no fill), expand
right+bottom 1024² → 1331² (GPU), inpaint-mask renders the prompt exactly in the
masked circle (GPU). Documented in README_CLI.md.

## Unreleased — Force aspect ratio: new "Extend (outpaint)" mode next to crop

**Why.** "Force aspect ratio on Upscale/img2img" could only centre-crop the input to the
target ratio (Fooocus-style) — the edges were lost. Requested: reach the ratio by
**extending** the image instead.

**What.** The checkbox becomes a radio — **Off / Crop to fit / Extend (outpaint)**. In
Extend mode the missing bands (symmetric, on the short axis) are generated by Z-Image via
the existing `outpaint_directions` machinery: the centre keeps its full resolution
(diffusion capped at ~1 MP then recomposed), blurred-edge fill for exposure continuity,
nothing is cropped. Steps are clamped to ≥6 so a pure-ESRGAN upscale (denoise 0) still
gets clean bands. A **seam-blend pass** then removes the exposure/texture junction lines
the raw outpaint leaves: one light img2img pass runs over the extended image (denoise
`force_ratio_extend_denoise`, default **0.22**, 0 = off) but only the **bands + a
feathered transition margin** (~5%, straddling the seam) are composited from it — the
original centre stays pixel-for-pixel untouched. Config: `force_ratio_mode` (`crop` default /
`extend`), env `CZ_FORCE_RATIO_MODE`. Validated on GPU: 512×512 → 910×512 (16:9, ratio
1.777), scene continued plausibly and seam-free with the harmonize pass; unit tests
cover the geometry (both axes + no-op), the setter and the UI radio mapping
(`tests/test_force_ratio.py`).

## Unreleased — Load (almost) every CivitAI Z-Image build: GGUF + FP8/INT8 "scaled" checkpoints

**Why.** Most Z-Image fine-tunes on CivitAI ship as ComfyUI **FP8/INT8 "scaled"**
safetensors (half the size of BF16) or as **GGUF** quants — both were skipped from the
model list with "not loadable by diffusers", leaving only the ~12 GB BF16 builds usable.

**What.**
- **GGUF checkpoints** (`.gguf`, Q3..Q8): listed from the checkpoints folders and loaded
  via `from_single_file` + `GGUFQuantizationConfig` — the transformer **stays quantized
  in VRAM** (real memory saving). Guards ported from the Qwen fork: architecture check
  (`general.architecture` must be **`lumina2`**, what ComfyUI-GGUF conversions of Z-Image
  declare — configurable via `gguf_arch`), tensor-layout check (stable-diffusion.cpp
  compact renames are refused with a clear message), and `_effective_offload` forces
  offload `model` for a GGUF base (a quantized transformer doesn't reach the GPU via
  `.to(cuda)`/sequential — it would silently run on CPU). Derived img2img/inpaint pipes
  skip the bf16 recast ("Casting a quantized model is unsupported"), and hot-swapping
  from/to a GGUF falls back to a full reload (the effective offload changes).
- **FP8 / FP8-scaled / INT8-scaled safetensors** (ComfyUI format: `X.weight` F8/I8 +
  `X.weight_scale` scalar-or-per-row + `X.comfy_quant` blob): dequantized **in RAM to
  BF16** at load (`w × scale`), then fed to `from_single_file` as a state dict — the
  diffusers key conversion (ComfyUI prefix, fused-QKV split) still applies. Tensors are
  read in **file-offset order** (an HDD collapses on random access). Note: dequantized
  FP8 has the memory footprint of a full BF16 — the saving is on disk/download only.
- **AIO bundles** (transformer + text encoder + VAE in one file): only the
  `model.diffusion_model.*` keys are kept — a 16.9 GB bundle whose FP8 lives only in the
  bundled text encoder loads as a plain BF16 transformer (base VAE/encoder are reused).
- **Still refused, clearly**: misfiled LoRAs, SVDQuant/Nunchaku INT4 (needs the nunchaku
  runtime), quantized checkpoints of a different architecture ("does not look like a
  Z-Image transformer"), foreign-architecture or sd.cpp-layout GGUFs.

Validated on real CivitAI files: pure-FP8 5.7 GB (intorealism V80) and INT8-per-row
6.7 GB (redzit) both load to 6.15 B bf16 params with zero meta tensors; the 16.9 GB AIO
loads in ~5 s (no dequant needed); a quantized non-Z-Image checkpoint (ernieRedmix) is
rejected with the clear architecture message; Z-Image Turbo Q4_K_M GGUF loads and lists.

## Unreleased — Fix: a LoRA picked as checkpoint no longer hunts for an SD1.5 config

**Why.** A LoRA file misfiled in a checkpoints folder (e.g. `ZITnsfwLoRAv3.safetensors`)
could be selected as the transformer: diffusers cannot recognise the state dict, falls
back to its default single-file repo and dies with
`OSError: stable-diffusion-v1-5/stable-diffusion-v1-5 does not appear to have a file
named config.json` — observed on a Pinokio install.

**What.** `_safetensors_unsupported` now detects LoRA state dicts from the header
(kohya `lora_down/up` + `lora_unet_/lora_te` prefixes, peft `lora_A/B`): such files are
skipped from the checkpoint list and force-loading one raises "LoRA file, not a
checkpoint - move it to the LoRA folder and pick it in Models > LoRA". The Z-Image
`from_single_file` also passes `config=BASE_REPO, subfolder="transformer"` (as the Qwen
fork already did), so a valid-but-unrecognised transformer never falls back to SD1.5 and
offline mode keeps working. Validated: 3 real Z-Image LoRAs detected, a real checkpoint
accepted and actually loaded through the new path (25 s, valid model), the forced-LoRA
guard raises the clear error, smoke 22/22.

## Unreleased — Fix: concurrent generations no longer corrupt the shared scheduler

**Why.** Gradio does not serialise events from DIFFERENT listeners: a manual **Generate**
still running while **Run queue** starts its first job (or the face detailer refining)
meant two threads calling the same shared pipeline and stepping the SAME scheduler — its
internal index ran past the end and the job died with
`IndexError: index 31 is out of bounds for dimension 0 with size 31`
(`scheduling_flow_match_euler_discrete.step`), observed live on crispz-qwen-edit.

**What.** A process-wide **GPU lock** (`_GPU_LOCK`, re-entrant) now serialises every
generation entry point (`generate`, `txt2img_run`, `process_one`, `_refine_whole`,
`outpaint`, `inpaint_run`, `generate_omni` — decorator `@_gpu_serial`): a second request
simply waits for the GPU instead of racing it. Same-thread nesting (txt2img→generate,
process_one→refine, detailer) stays free thanks to the RLock. Validated: 4 threads on a
locked function show zero overlap, nesting does not deadlock, all entry points wrapped,
smoke 22/22.

## Unreleased — Thumbnail cache: app-folder default, UI field, and CLI flags for the new features

- **New default location**: the Asset Browser thumbnail cache now lives in **`<app>/cache/`**
  (gitignored) instead of inside the output folder — the app folder is usually on a fast
  disk, so the grid stays fast even when outputs are on an HDD/NAS, with zero configuration.
  The special value **`output`** restores the old next-to-the-images layout
  (`<out>/_index/thumbs`, relative URLs); any path still works as a custom cache.
- **UI field**: *Save > Asset Browser > Thumbnail cache folder* (+ Save) — persisted to
  `preferences.json` (which now overrides `asset_browser.cache_dir` from config), applied
  immediately for writing; a restart is needed to *serve* from a brand-new path
  (`allowed_paths` is fixed at launch).
- **CLI**: the recent features are usable headless too — **`--detail-faces`** (+
  `--detailer-denoise`) runs the auto face detailer after a `--txt2img` render, and
  **`--metadata-scheme a1111`** writes the Civitai-readable `parameters` chunk. `--auth`
  was already in. Validated: default resolves to the app cache (the 10 951 migrated
  thumbnails are picked up unchanged — same output-dir slug), `output`/custom overrides,
  the UI handler, `--help`, build_ui, smoke 22/22.

## 1.16.0 — 2026-08-04 — Release: Fooocus-parity pass (auth, CivitAI consensus, face detailer) + SSD thumb cache

Consolidates everything since **1.15.0** — the gap-analysis pass against Fooocus2026:
optional **login page** (auth) for LAN/tunnel exposure, **CivitAI community recommended
settings** with one-click apply, **PNG Info ✨ Apply all**, one-click **🎲 Vary
(subtle/strong)**, the **🔧 auto face detailer** (ADetailer-style), the Asset Browser
**thumbnail cache on a fast disk** (`asset_browser.cache_dir`), **17 exact aspect
ratios**, the `simple` schedule alias, the same-base-model **⚠ update badge** fix, and
**occlusion-aware Face Swap** blending. Details in the sections below.

### 🔧 Auto face detailer (ADetailer-style)

**Why (last real Fooocus-parity gap).** Small faces in a wide shot come out soft: the
model spends ~1 Mpix on the whole scene, so a 150 px face gets almost none of it.
Fooocus's *Enhance* / A1111's *ADetailer* fix this by re-rendering each face at high
resolution; crispz had nothing equivalent.

**What.** New `cz_detailer.py` + a **🔧 Detail faces** checkbox under the Generate button
(module flag — the queue and X/Y/Z grid snapshots are untouched). After each render (and
after the optional auto-upscale), faces are detected with insightface buffalo_l (already
loaded for Face Swap; detection no longer requires the inswapper model —
`cz_face.detect_faces` / `_ensure_face_detector` factored out) and each face, biggest
first (up to `face_detailer_max_faces`, default 4), is:
enlarged crop (+60 % context) → scaled to the model's ~832 px sweet spot → **Z-Image
img2img** (same prompt/seed, `face_detailer_denoise` 0.35, live slider in Advanced >
Generation) → scaled back → pasted through a **feathered elliptical mask** (no square
edges, same technique as the Face Swap GFPGAN paste). Full-frame portraits are skipped
(nothing to gain), failures degrade to the untouched image.
Validated: geometry/mask units, build_ui, smoke 22/22 — then **end-to-end on a real GPU
render** (832×1216 classical-portrait scene, seed 20260804): txt2img 4.5 s, one face
detected, refined at denoise 0.35 / 12 steps in ~30 s (one-time insightface load +
img2img-pipe derivation included; subsequent images are much faster). Pixel-diff check:
**4.62 mean inside the face crop, 0.0000 outside** — the feathered paste is surgical,
the rest of the image is untouched to the pixel. Visually: a ring-shaped artifact on the
forehead ornament became a clean dot, skin gradients and lashes tightened, identity fully
preserved, no visible seam.

### PNG Info "✨ Apply all" + one-click "🎲 Vary" (Fooocus parity)

- **PNG Info** could only send the prompt and the seed. The new **✨ Apply all** button
  loads *everything* the image carries — prompt, negative, seed, steps, CFG, size
  (width/height), sampler/schedule — like Fooocus's full parameter load. crispz's own
  `sampler/schedule` notation is applied as-is (aliases like `simple` normalised); A1111/
  CivitAI sampler names go through the same conservative mapping as the CivitAI
  recommended settings (`Euler a` → euler, `DPM++ 2M Karras` → keeps the sampler, applies
  the karras schedule) and the status line spells out what was applied vs kept.
- **Vary (subtle / strong)** — two buttons in the Upscale/img2img tab arm a pure img2img
  pass in one click (ESRGAN off, refine on, denoise **0.25** / **0.6**), tick *Input
  Image* and open its panel; drop an image and press Generate. The report line explains
  what was armed.

### Security: optional login page (`auth` / `--auth` / `CRISPZ_AUTH`)

**Why.** The UI can be exposed on a LAN or through a tunnel (cloudflared) — until now with
no protection at all: anyone with the URL could generate images and browse/delete outputs.

**What.** Optional auth, off by default (localhost unchanged). Set config `auth` to
`"user:password"` (several accounts via commas), or pass `--auth user:pw`, or the
`CRISPZ_AUTH` env var: Gradio then shows a login page and gates every route — verified
live: without login, `file=` serving and API endpoints return 401; a wrong password is
rejected; a no-auth launch behaves exactly as before.

### CivitAI: community "recommended settings" (consensus) + one-click apply

**Why (Fooocus2026 parity).** CivitAI's example images publish their generation `meta`
(sampler, cfgScale, steps, size). Fooocus2026 analyses them into consensus settings;
crispz fetched previews/triggers/examples but ignored the settings.

**What.** `cz_civitai.analyze_settings` computes the consensus (median steps/CFG, majority
sampler/size, with the number of images used) and stores it as `"recommended"` in
`<name>.civitai.json` on every fetch. The Asset Browser model card shows a **Community
settings** block. In **Models > Checkpoints**, a new **📊 Apply CivitAI recommended
settings** button fetches (with progress — hashing a 12 GB checkpoint without a cached
hash takes minutes on an HDD) and applies steps/CFG plus sampler/schedule when a Z-Image
equivalent exists (`Euler*` → euler, `UniPC` → unipc, `LCM` → lcm, `*Karras*` → karras
schedule…); DPM++-style samplers with no equivalent are reported and left unchanged.
Validated against live CivitAI: consensus `{steps 9, CFG 1.0, sampler Euler}` from 5
community images, applied as steps=9 / CFG=1.0 / euler.

### Asset Browser: thumbnail cache on a fast disk (`asset_browser.cache_dir`)

**Why.** Thumbnails are the Asset Browser's hot path — one file per image, re-read on
every grid paint — and they were always written next to the images. With the output
folder on a slow HDD (plus antivirus write-scanning), serving a single 84 KB thumbnail
cold took **~1.1 s**; a grid of thousands crawled.

**What.** New `asset_browser.cache_dir` (config, empty by default). When set to a fast
disk (e.g. `"D:/crispz-cache"`), thumbnails go to `<cache_dir>/crispz-thumbs/<slug>/`
(one slug per output folder — several output folders never collide) and the manifests
reference them by absolute `/gradio_api/file=` URL; the launcher serves that folder
automatically. Empty keeps the previous `<out>/_index/thumbs` layout, so nothing changes
for existing setups. The path logic is centralised (`_thumbs_root` / `_thumb_paths`) and
used everywhere thumbs are built, checked, or deleted — `delete_asset` and freshness
checks follow the cache. The cache is disposable: delete it any time, it rebuilds on
demand. Measured on a 10 576-image folder (HDD → SSD): cold thumbnail ~1.1 s → **~3 ms**,
30 grid thumbnails in **88 ms** total.

### Aspect ratio: 8 sizes to 17, sorted, with the exact CivitAI ratios

The dropdown only carried the Fooocus list, whose portrait/landscape entries
(`832 x 1216`, `768 x 1344`, `1536 x 640`) are *approximations* of 2:3, 9:16 and 21:9,
and which had no 5:4 or 4:3 at all. Following a CivitAI or ComfyUI recipe meant accepting
a different framing or dropping to the CLI (`--gen-width` / `--gen-height`).

Nine sizes added, all multiples of 16, all **exact** ratios except where they mirror an
existing Fooocus label: `1280 x 1024` / `1024 x 1280` (5:4, 4:5), `1280 x 960` /
`960 x 1280` (4:3, 3:4), `1536 x 1024` / `1024 x 1536` (3:2, 2:3), `1536 x 864` /
`864 x 1536` (16:9, 9:16), `640 x 1536` (the missing portrait ultra-wide). The Fooocus
entries stay as they are — existing presets and seeds depend on them.

The list is now sorted from squarest to widest, each landscape size followed by its
portrait counterpart, instead of the historical order. Cost runs 1,0 to 1,6 Mpix: past
~1,3 Mpix generation is slower and a model trained around a megapixel can drift in
composition (duplicated subject), so the big ones are for when a recipe calls for them.


### Schedule: `simple` accepted as an alias of `sgm_uniform`

**Why.** ComfyUI and CivitAI recipes name the native flow schedule `simple`. Ours was
only reachable as `sgm_uniform`, so every copied recipe needed a mental translation and
`--schedule simple` was rejected outright.

**What.** `simple` is now accepted wherever a schedule is written (`default_schedule`,
`ZIMAGE_SCHEDULE`, `--schedule`, the XYZ `Schedule` axis) and normalised back to
`sgm_uniform`, so metadata and presets keep one name for one curve. It is **not** a
second entry in the UI dropdown: same schedule, not a new option. The sigmas the pipeline
hands to the scheduler are `linspace(1, 1/steps, steps)` — exactly what ComfyUI's
`simple` produces on a flow model, so the alias is an equality, not an approximation.

### CivitAI: the ⚠ update badge now compares within the same base model

**The bug.** The check took `modelVersions[0]` from `GET /models/<id>` — the most recent
version of the *page*, whatever it was trained on. A LoRA page that moves on to another
base (a *3.0 (Krea2)* release over a Z-Image *2.0*) flagged every local copy as outdated,
pointing at a file that would not even load in the app.

**The fix.** `get_latest_version` takes the local `baseModel` (from the sidecar, or
deduced from our own version inside the same response for pre-existing sidecars) and only
considers versions published for that base — normalised comparison, so *Z-Image* /
*Z Image* / *zimage* match. No version shares our base → no update, rather than a false
positive. If the API returns no `baseModel` at all, nothing is filtered: the information
is missing, not contradictory. Stale flags clear on the next `civitai_index` pass (or
**🔄 Fetch all missing**), which re-checks already-enriched models without re-downloading.

### Face Swap: occlusion-aware blending, and ONNX actually on the GPU

**The bug.** inswapper renders a 128 px face and insightface pastes it back through a
plain rectangle (`img_white` in `model_zoo/inswapper.py`), which is blind to depth: on a
shot of someone eating, the ice cream and the hand holding it sat inside that rectangle
and were repainted by the generated face. The mouth area came out broken on every image
where an object touches the face.

**The fix.** `_faceswap` now uses `paste_back=False` and composites itself, through a
mask built from an **occlusion pass** (XSeg, `dfl_xseg.onnx`) intersected with a
**face-region pass** (BiSeNet, `bisenet_resnet_34.onnx`), both computed on the *original*
frame — the only place the occluding object is still visible. The restore pass is masked
the same way, so it cannot repaint over an occlusion either. Added **LAB colour matching**
between the swapped and original face, and **CodeFormer** as the default enhancer
(`faceswap_restore_model`, with a `faceswap_restore_fidelity` weight) over GFPGAN.
`faceswap_restore` now defaults to **on**: without it a 128 px swap is visibly soft.
Each model is fetched once and every pass degrades cleanly to a log line if absent.
Cost: ~90 ms per face on GPU.

**GPU.** `requirements-lock.txt` pinned both `onnxruntime` and `onnxruntime-gpu` (rembg
pulls the CPU build). Same module name, so the CPU build **shadowed** the GPU one and
every ONNX pass — swap, restore, masks, rembg — had been running on CPU. The installers
now filter the CPU build out of the lock, like the existing Pillow filter.

**Not** a replacement for Fooocus-style FaceSwap, which conditions the diffusion itself
(IP-Adapter face) rather than pasting afterwards. No IP-Adapter or ControlNet exists for
Z-Image-Turbo today — only LoRAs — so the post-process remains the only route here.

## 1.15.0 — 2026-07-28 — Release: Asset Browser rearchitecture, security, GPU-agnostic tooling

Consolidates everything since **v1.11.2**. Nothing new here beyond the last build change
below — this entry marks the release boundary.

**Asset Browser — the big one.** Opening it re-read the PNG metadata of every image on
every open (**295 s** for 9 278 images) and shipped a **9,42 MB** manifest to the browser,
while the SPA stopped polling after 180 s — so it never finished filling in. Rebuilt on the
Fooocus design: a metadata cache (295 s → **3,7 s**), a tiny `days.json` plus one manifest
per day (**5 400× less data** on open), and incremental indexing at save time (~15 ms/image,
no rescan). Global search was kept, which Fooocus does not have on its Outputs tab.

**Security.** 37 Dependabot alerts triaged against what the code actually calls: Pillow,
protobuf and sentencepiece upgraded (**21 closed, 0 advisories left** on those pins), the
other 16 assessed unreachable and dismissed with per-package reasons — documented in
`SECURITY.md`, and the four Dependabot PRs closed with the reasoning.

**Tooling.** `boot_check.bat` replaces the RTX-5090-only script and works on any card; its
decisive check compares the GPU's `sm_XX` against the installed torch build, catching the
`WinError 127 torch_cuda.dll` class of failure *before* launch. `update.bat`/`.sh` add the
missing post-`git pull` step. New `lcm` sampler; LoRA weights can go negative (`-2..2`).

- Last change in this release: the `pillow==12.3.0` pin stays in `requirements-lock.txt`
  (so Dependabot sees the fixed version) and `install.*` / `update.*` filter that one line
  out before `pip` runs, installing Pillow separately with `--no-deps`. Commenting it out
  had fixed the install but left Dependabot matching the whole advisory range.

## 1.14.1 — 2026-07-27 — Fix install (Pillow/gradio), sampler status, drop the RTX-5090 scripts

Fallout from testing `install.bat` / `update.bat` / `run.bat` end to end.

- **`install.bat` was broken** by the 1.13.1 security bump: `gradio 5.50` declares
  `pillow<12.0`, so pinning `pillow==12.3.0` made a clean install fail with
  `ResolutionImpossible`. The running venv had not noticed because the upgrade used
  `--no-deps`. There is **no Pillow below 12 that fixes those CVEs** (11.3.0 is the last
  11.x and leaves 19 advisories open), so Pillow is now installed **separately, after the
  lock, with `--no-deps`** by `install.bat`/`.sh` — and re-applied by `update.bat`/`.sh` so
  an update cannot silently regress it. gradio's bound is conservative; Pillow 12 is
  verified working here (build_ui, save + metadata round-trip, thumbnails, `RankFilter`,
  `crop`, WebP). The `<6` gradio pin stays deliberate.
- **Sampler dropdown warning fixed**: `set_sampler` / `set_schedule` return a status
  string but were wired to `None` outputs, so Gradio logged *"returned too many output
  values"* on every change (any sampler, not just the new `lcm`). The status is now
  **displayed** next to the dropdowns instead of being discarded.
- **RTX-5090 scripts removed**: `boot_check_rtx5090.bat`, `run_quality_rtx5090.bat`,
  `_lan`, `_web`. They are superseded by `boot_check.bat` / `_lan` / `_web`, which are
  card-agnostic. Their one behaviour not yet covered — a fixed `GRADIO_SERVER_PORT=7860` —
  was carried over. No launcher (including the Pinokio one) referenced them.
- `config.txt` regains the inline `_help` documentation it had lost historically, plus the
  24 keys added since it was created — user values preserved.
- Files: `install.bat`, `install.sh`, `update.bat`, `update.sh`, `requirements-lock.txt`,
  `cz_ui.py`, `boot_check.bat`, `README.md`, `VALIDATION.md`.

## 1.14.0 — 2026-07-27 — Smart boot check (any GPU) + update scripts

`boot_check_rtx5090.bat` only knew one card and hardcoded a model path. Replaced by a
generic diagnostic, and the missing post-`git pull` step now exists.

- **`boot_check.bat`** — works on any NVIDIA card. Beyond driver/VRAM/temperature, it runs
  the check that actually matters: **is the GPU's `sm_XX` in the installed torch build's
  arch list?** That is precisely the *"RTX 50xx + non-cu128 torch"* failure
  (`WinError 127 … torch_cuda.dll`) hit earlier in this project — it is now caught **before
  launch**, with the exact `pip install --index-url …` line to fix it, and the script stops
  instead of letting the app die at the first CUDA allocation.
- **Recommendations scale with the card**: `_hw_check.py` now maps compute capability to a
  generation name (Blackwell / Ada / Ampere / Turing / Pascal) and derives CPU offload,
  ESRGAN tiling, max resolution and dtype from the real VRAM. Thresholds come from figures
  measured in this project (FLUX bf16 ≈ 33 GB with its encoder, GGUF Q8 ≈ 12,7 GB,
  `sequential` ≈ 3 s/step vs `model` ≈ 1,1 s/step). The 12 GB tier is keyed at 11 GB, since
  a "12 GB" card reports ~11,9 — putting it on `sequential` would have cost 3x for nothing.
- **Model folders are read from `config.txt`** instead of a hardcoded `D:\…\Z-Image`.
- **`boot_check_lan.bat` / `boot_check_web.bat`** — same diagnostic then LAN (`0.0.0.0`) or
  Cloudflare tunnel; both print a **no-authentication warning** first, consistent with
  `SECURITY.md`. `boot_check_rtx5090.bat` is kept as an alias.
- **`update.bat` / `update.sh`** — the post-GitHub-update step: refuses to `git pull` over
  uncommitted work (shows what is dirty), reinstalls dependencies **only if the requirements
  file changed** (md5), **warns if `torch` was replaced** (a transitive resolve can swap a
  `+cu128` build for a CPU wheel — that happened here once), re-runs the hardware check,
  verifies diffusers and `cz_ui` still import, and lists **new config keys** from
  `config-sample.txt` (`config.txt` is never overwritten). Flags: `--no-pull`,
  `--force-deps`, `--shared`.
- All `.bat` written with **CRLF**: `goto` silently fails on LF-only batch files.
- Files: `boot_check.bat`, `boot_check_lan.bat`, `boot_check_web.bat`,
  `boot_check_rtx5090.bat` (alias), `update.bat`, `update.sh`, `_hw_check.py`, `README.md`.

## 1.13.1 — 2026-07-27 — Security: upgrade Pillow / protobuf / sentencepiece (21 Dependabot alerts)

Adding `requirements-lock.txt` made Dependabot match **pinned versions** instead of ranges,
raising 37 alerts. Each was triaged against what the code actually calls.

- **Upgraded**: `pillow 11.3.0 -> 12.3.0`, `protobuf 6.31.0 -> 7.35.1`,
  `sentencepiece 0.1.96 -> 0.2.2` — **21 alerts closed**, and the advisory database reports
  **0 remaining** for those three pins.
- Pillow was the priority: it is the only flagged package that parses **files the user
  supplies** (Input image, PNG Info), so its 18 advisories (PSD/FITS/JPEG2000 OOB, font and
  PDF decompression bombs, `RankFilter`, `ImageCmsTransform`, `crop`/`paste` overflow…)
  were genuinely reachable.
- **Deliberately not upgraded** (documented per-package in `SECURITY.md`): rembg (server-only
  flaws, server never started, patched line needs Python 3.11 while this runs 3.10), gradio
  (`gr.load()`/OAuth/audio unused, Windows traversal needs Python 3.13+; fix needs a 6.x
  major bump the `<6` pin deliberately excludes), transformers (`Trainer`/LightGlue unused,
  `trust_remote_code` never set; fix needs 5.x), torch (`jit.script`/`lstm_cell`/
  `unpack_sequence` never called; fixes have no `+cu128` build, so upgrading would break
  RTX 5090 support to close unreachable flaws).
- `torch` was protected during the upgrade (`pip install --no-deps`) — the earlier incident
  where a transitive resolve replaced `2.8.0+cu128` with a CPU build must not repeat.
- Verified on the real environment: `torch 2.8.0+cu128` + CUDA still load, `build_ui()`
  builds, the image chain (save + metadata round-trip + thumbnail + `RankFilter` + `crop` +
  WebP) works under Pillow 12, and the test suites pass.
- `SECURITY.md`'s alert section was rewritten: it still claimed the repo had *no lockfile*,
  which is what changed.
- Files: `requirements-lock.txt`, `SECURITY.md`.

## 1.13.0 — 2026-07-27 — Asset Browser: per-day index + incremental indexing (Fooocus architecture)

Follow-up to 1.12.2. The metadata cache fixed the *server* side (295 s -> 3,7 s), but the
browser still downloaded and rendered a **9,42 MB manifest with all 9 278 images** on every
open. Aligned on the Fooocus design, which was studied for this.

- **`_index/days.json`** — a tiny index (`{date, count}` per day, ~200 bytes for 42 days).
  The page reads *that* on open, so the sidebar and the current day appear immediately
  instead of waiting for a multi-megabyte manifest.
- **One `manifest.json` per day**, written *inside* the day folder (Fooocus convention).
  The SPA loads only the day being displayed: **9,42 MB -> 1,48 MB** for the largest day
  (938 images), and typically far less. **5400x less data** for the initial load.
- **Incremental indexing** — new `on_image_saved()` hook (crispz's `on_image_logged`),
  called from `save_image()`: thumbnail + day manifest + `days.json` are updated **as the
  image is written** (~15 ms), so the browser no longer needs a folder rescan to be current.
  Idempotent (re-saving the same file does not duplicate it) and silent by contract — any
  failure is logged and *never* breaks a generation.
- **Global search preserved.** Fooocus only searches within the LoRA/Models tabs; crispz
  searches all output metadata, so that was kept: typing a query loads the remaining days
  in the background (cached in memory) and searches across everything.
- **Backwards compatible**: the global `_index/manifest.json` is still written, and the SPA
  falls back to it when `days.json` is absent (index not migrated yet).
- `_entry_for()` is now the single definition of a manifest entry, shared by the full
  reindex and the incremental hook, so the two paths cannot drift apart.
- Files: `cz_assetbrowser.py` (`_write_day_manifests`, `on_image_saved`, `_bump_days_index`,
  `_entry_for`, `_INCR_LOCK`), `cz_imageio.py` (hook in `save_image`), `cz_assets.py`
  (SPA: `days.json` -> per-day load, search loads all days), `tests/test_ab_index.py`
  (+5 tests: per-day manifests, incremental add, idempotence, never raises, identical
  entry shape between both paths).

## 1.12.2 — 2026-07-27 — Asset Browser: metadata cache (reindex 80x faster)

Opening the Asset Browser re-read the PNG metadata of **every** image on every open —
~25 ms each. Measured on a real 9 278-image library: **295 s per open**, while the SPA
gives up polling after 180 s (90 x 2 s). The manifest was therefore written *after* the
page stopped listening: "it doesn't refresh" and "images are missing".

- `ab_reindex` now keeps a **metadata cache** (`_index/meta_cache.json`, rel -> mtime+size
  signature + parsed meta). Unchanged files are served from it; only new or modified
  images are re-read.
- Measured on the same 9 278 images: **295 s -> 3,7 s (80x)**. Well under the polling
  window, so the gallery fills in as intended.
- Cache follows deletions (entries for vanished files are dropped, no unbounded growth)
  and is **defensive**: a corrupt/unreadable cache is ignored and rebuilt, never fatal.
- Each pass logs what it did: `indexed N image(s) in X.Xs (H from meta cache, R read)`.
- Files: `cz_assetbrowser.py` (`_load_meta_cache`, `_save_meta_cache`, `_meta_cached`),
  `tests/test_ab_index.py` (5 tests: cache hit, modified file re-read, fresh metadata
  reaches the manifest, deletions pruned, corrupt cache non-fatal).

## 1.12.1 — 2026-07-27 — New `lcm` sampler (LCM flow-matching)

- **Sampler** gains **`lcm`** (`FlowMatchLCMScheduler`) next to `euler` and `unipc`:
  designed for **few steps with guidance ~0-1**, so it suits distilled / Turbo checkpoints.
  Works with all four schedules (`sgm_uniform` / `beta` / `karras` / `exponential`) — the
  12 sampler×schedule combinations were verified to build.
- Falls back to `euler` (with a log line) if the installed diffusers has no
  `FlowMatchLCMScheduler`.
- **Why not `dpmpp_sde`** (recommended by some Civitai model cards): the Z-Image pipeline
  imposes custom `sigmas`, and `DPMSolverSDEScheduler.set_timesteps` does not accept them
  (it also needs the `torchsde` package). Same reason DPM++ 2M / DPM2a are not exposed.
  Note that ComfyUI's **`simple` scheduler == our default `sgm_uniform`**, and ComfyUI's
  **CFG 1.0 == our guidance 0** — so those model cards are already satisfied by the
  defaults.
- Files: `cz_pipeline.py` (`SAMPLER_CHOICES`, `_build_scheduler`), `README.md`,
  `config_modification_tutorial.txt`, `tests/test_xyz.py` (sampler suggestions now derive
  from `SAMPLER_CHOICES` instead of a hardcoded list).

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
