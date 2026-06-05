# crispz-studio

> Z-Image txt2img + upscaler/detailer studio (a Fooocus-style fork of
> [crispz](https://github.com/mikecastrodemaria/crispz)).

A standalone Z-Image **creation + enhancement** tool, **100% local**, no ComfyUI /
SwarmUI. On top of crispz's upscaler it adds:

- **Text → Image** (`ZImagePipeline`): generate from a prompt, optionally upscale.
- **Image → Upscale** (the crispz pipeline): Real-ESRGAN + Z-Image refine.
- **Single-file `.safetensors` models** (Civitai): use a Z-Image transformer
  checkpoint via `from_single_file` (VAE + Qwen3 encoder kept from the base repo).
- **Shared VRAM**: txt2img and img2img reuse the same loaded weights (`from_pipe`).

Tabbed Gradio UI + scriptable CLI + persistent server (`--serve`).

> Roadmap: Phase 2 = ControlNet + Omni-edit (image mixing). Phase 3 = faceswap
> (post-process). See the parent crispz repo for the upscale internals.

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

- **Gradio UI**: "Paths / models" accordion at the top. Buttons "Refresh ESRGAN
  list", "Apply Z-Image" (invalidates the pipe so it reloads), and "Save to
  preferences.json".
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
| Diffusion steps | `--steps` | `steps` | `12` |
| Prompt | `--prompt` | `prompt` | `""` |
| Seed | `--seed` | `seed` | `-1` |
| ESRGAN tile | `--tile` | `tile` | `760` |
| Overlap | `--overlap` | `overlap` | `32` |
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
  "esrgan_dir": "D:/Github/sdlibs/models/ESRGAN",
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
