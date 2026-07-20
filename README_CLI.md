# crispz-studio — CLI cheat sheet

All commands are `python app.py ...` (use your venv: `.venv\Scripts\python app.py ...`).
Run `python app.py --help` for the full flag list. No args → launches the UI.

Common output flags (work with most modes):
`--save-mode {display,local,alongside,custom}` · `--output-dir <dir>` · `-o <file|dir>`
· `--output-format {png,jpg,webp}` · `--print-output` (print only the saved path)
· `--report-vram` · `--log-level {quiet,info,debug}`

---

## Text → Image (txt2img)

```bash
# Basic (Z-Image Turbo: 8 steps, guidance 0)
python app.py --txt2img --prompt "a serene mountain lake, cinematic" \
    --gen-width 1024 --gen-height 1024 --gen-steps 8 --seed 42 \
    --save-mode local --output-dir out

# Z-Image Base / Juggernaut-Z (needs CFG)
python app.py --txt2img --prompt "cinematic portrait of a sea captain" \
    --gen-width 960 --gen-height 1440 --gen-steps 24 --guidance 6 \
    --save-mode local --output-dir out

# Generate then upscale in ONE command (txt2img chained into ESRGAN + Z-Image refine).
# This is the CLI equivalent of the UI's "Upscale after generate" checkbox.
python app.py --txt2img --prompt "portrait of an old fisherman" --upscale \
    --factor 2 --denoise 0.30 -m 4x-ClearRealityV1_Soft.safetensors \
    --save-mode local --output-dir out

# Faster chained variant: refine at native resolution THEN ESRGAN (--refine-first).
# Skip the refine pass entirely with --denoise 0 (ESRGAN-only upscale).
python app.py --txt2img --prompt "..." --upscale --refine-first \
    --factor 2 --denoise 0.30 -m 4x-ClearRealityV1_Soft.safetensors
```

## X/Y/Z comparison grid  — `--xyz "AXIS=v1,v2,…"` (repeat ×1–3)

Every combo is generated, saved as a normal output, and the run ends with annotated
contact sheet(s) in `<output>/xyz_<timestamp>/` (one per Z value; paths printed).

> Ready-to-run example: **`xyz_example.bat "your prompt"`** (Windows) or
> **`./xyz_example.sh "your prompt"`** (Unix) — a 2×2 Steps × Guidance grid; edit the
> `--xyz` lines inside to change the axes.

```bash
# 3×2 grid: Steps × Guidance
python app.py --txt2img --prompt "a red cat" \
    --xyz "Steps=4,8,12" --xyz "Guidance=0, 3.5" --save-mode local

# Compare checkpoints (partial names resolve case-insensitively), one sheet per seed
python app.py --txt2img --prompt "..." \
    --xyz "Checkpoint=Z-Image-Turbo, intoreal" --xyz "Seed=42, 1234"

# Prompt S/R (first value = search term) + an upscale axis (needs --upscale)
python app.py --txt2img --prompt "a red cat" --upscale \
    --xyz "prompt=cat, dog, fox" --xyz "Denoise=0.2, 0.4"

# Compare LoRA epochs (partial names resolve; None = control cell without LoRA)
python app.py --txt2img --prompt "..." \
    --xyz "LoRA=ollie_e000010, ollie_e000020, ollie_e000030, None"

# Vary LoRA file AND weight together
python app.py --txt2img --prompt "..." \
    --xyz "LoRA + weight=ollie_e10:0.6, ollie_e20:0.9"
```

Axes: `Checkpoint`, `Sampler`, `Schedule`, `Steps`, `Guidance`, `Seed`, `ESRGAN model`,
`Factor`, `Denoise`, `Tile`, `Refine tile`, `LoRA` (file in slot 1), `LoRA + weight`
(`name:weight`), `LoRA weight`, `Performance`, `Prompt S/R`.
Axis names and closed-list values resolve case-insensitively (`step` → `Steps`,
`uni` → `unipc`); quotes protect commas; upscale-only axes require `--upscale`.
**Ctrl+C assembles a partial sheet** with the cells rendered so far.

## Choose / switch the Z-Image model

```bash
# Single-file checkpoint (Civitai BF16/FP16) as the transformer
python app.py --txt2img --prompt "..." \
    --zimage-transformer "D:/.../Z-Image/Juggernaut_Z_V1_bf16.safetensors" --guidance 6 --gen-steps 24

# Transformer from a diffusers repo/folder (keeps base VAE/encoder)
python app.py --txt2img --prompt "..." \
    --zimage-transformer "RunDiffusion/Juggernaut-Z-Image" --guidance 6

# Full diffusers base
python app.py --txt2img --prompt "..." --zimage-model "Tongyi-MAI/Z-Image-Turbo"
```

## LoRA (up to 3, combinable)  — `--lora NAME[:WEIGHT]`

```bash
# One LoRA at weight 0.8 (file in the loras dir)
python app.py --txt2img --prompt "..." --lora mystyle.safetensors:0.8

# Several LoRAs + custom folder
python app.py --txt2img --prompt "..." --loras-dir "D:/.../loras" \
    --lora char.safetensors:0.9 --lora style.safetensors:0.5
```

## Vision Mix (reference images → one prompt → generate)  — needs Ollama vision model

```bash
python app.py --vision-mix person.jpg outfit.jpg --save-mode local --output-dir out
# pick the model explicitly
python app.py --vision-mix a.png b.png --ollama-model llava:13b-v1.6 --gen-steps 8
```

## Upscale / detail an existing image (img2img)

```bash
# Upscale x2 (ESRGAN + refine)
python app.py --cli -i my_image.jpg -m 4x-ClearRealityV1_Soft.safetensors \
    --factor 2 --denoise 0.30 --save-mode local --output-dir out

# img2img only (refine, NO ESRGAN enlargement)
python app.py --cli -i in.png --no-esrgan --denoise 0.30 --save-mode local --output-dir out

# Whole folder (batch)
python app.py --cli -i ./my_images --save-mode local --output-dir out --output-format webp

# 4K+ with diffusion tiling (caps VRAM)
python app.py --cli -i in.png --factor 4 --denoise 0.30 --steps 12 \
    --refine-tile 1024 --refine-overlap 128 --save-mode local --output-dir out
```

## Batch CivitAI enrichment  — `cz_civitai_batch.py` (standalone, runs in parallel)

Fetch the **missing** CivitAI info (preview + trigger words + example prompts) for every
model in your LoRA / checkpoint folders, and flag models that have a **newer version** on
CivitAI. This is a separate script (no torch import → starts instantly); the Asset
Browser's **🔄 Fetch all missing** button runs the exact same core.

```bash
# All LoRAs + checkpoints, only the ones missing info
python cz_civitai_batch.py --kind all

# Backfill: re-query metadata for models already fetched (fills in example prompts,
# refreshes trigger words / version flags) WITHOUT re-downloading previews
python cz_civitai_batch.py --kind all --all

# Only LoRAs, re-download everything (overwrite existing previews/sidecars)
python cz_civitai_batch.py --kind loras --force

# Wrappers (find the venv Python, force UTF-8): pass-through args
civitai_index.bat --kind models          # Windows
./civitai_index.sh --kind models         # Unix

# Run in PARALLEL: split into N disjoint shards (one process each)
python cz_civitai_batch.py --kind all --shard 1/4   # + 2/4, 3/4, 4/4 in other terminals
civitai_index_parallel.bat 4             # Windows: launches 4 shards at once
./civitai_index_parallel.sh 4            # Unix
```

Flags: `--kind {loras,models,all}` · `--force` (overwrite) · `--all` (re-query every file,
don't overwrite previews) · `--shard i/m` · `--loras-dir` / `--checkpoints-dir` ·
`--api-key` · `--sleep 0.5` (seconds between requests) · `--no-check-updates`.

**What a re-run actually costs** (it does *not* redo everything):

| | Re-done on a re-run? |
|---|---|
| **SHA256 of the models** | **No** — cached in `<name>.civitai.json` after the first pass (and in `<name>.metadata.json` if some other tool wrote one). Only the *first* run reads the files. |
| **Previews** (`<name>.preview.png`) | **No** — kept unless `--force`. |
| **Metadata** (prompts, triggers, version flag) | Only with `--all` or `--force`, or for models still missing info: 2 API calls each. |
| **Asset Browser thumbnails** | Regenerated only if missing or older than the source; reopening the browser rebuilds the catalog in the background. |

So the expensive pass is the **first** one (it hashes the library). After that, a full
`--all` re-run is essentially just the API calls + `--sleep` per model.
**CivitAI rate-limits** — keep the shard count modest and set a CivitAI API key (in the
app's Advanced tab, or `--api-key`) for large runs.

## Remove background  — `--remove-bg` (local, rembg)

```bash
python app.py --remove-bg -i photo.jpg --save-mode local --output-dir out
# -> out/<name>_nobg_*.png (transparent)
```

## Reframe / Outpaint  — `--reframe W:H`

```bash
python app.py --reframe 16:9 -i square.png --prompt "extend the scenery" \
    --gen-steps 12 --guidance 6 --save-mode local --output-dir out
```

## Face swap (post-process)  — `--faceswap-src` (needs insightface + inswapper)

```bash
# generate then swap the face from a source photo
python app.py --txt2img --prompt "studio portrait, soft light" \
    --faceswap-src myface.jpg --save-mode local --output-dir out

# reframe + face swap
python app.py --reframe 16:9 -i in.png --faceswap-src myface.jpg --save-mode local --output-dir out
```

## Presets (use cases)

```bash
python app.py --cli -i in.png --preset "4K (tiled)" --save-mode local --output-dir out
python app.py --cli -i in.png --preset "Detailed (creative)" --denoise 0.32   # explicit flag wins
```

## VRAM offload (coexist with other apps)

```bash
python app.py --cli -i in.png --cpu-offload sequential --report-vram \
    --save-mode local --output-dir out
# --cpu-offload {none|model|sequential}
```

## Persistent server (load once, many requests)

```bash
python app.py --serve --host 127.0.0.1 --port 7861 --idle-timeout 300
# POST /upscale (JSON), GET /health, POST /unload
```

## External integration (Fooocus / scripts)

```bash
# stdout = ONLY the saved path; VRAM peak stays on stderr
dst=$(python app.py --cli -i in.png --save-mode local --output-dir out \
    --print-output --report-vram 2>vram.log)
echo "saved: $dst"
```

---

### Notes
- BF16/FP16 checkpoints only — FP8 / GGUF / SVDQ (ComfyUI) do **not** load in diffusers.
- Turbo → `--guidance 0`, ~8 steps. Base / Juggernaut-Z → `--guidance ~6`, ~22-35 steps.
- Output filenames are dated + unique (`{date}_{tag}_seed{seed}_{w}x{h}`), set by
  `filename_pattern` in `config.txt`.
- Defaults (size, steps, save mode, Ollama prompts, etc.) live in `config.txt`
  (see `config_modification_tutorial.txt`).
