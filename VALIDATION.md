# crispz-studio — Validation checklist (in-browser)

`smoke_test.py` covers the pure logic (no GPU/browser). The items below need a
quick **manual** check in the running UI. Run `run.bat`, hard-refresh
(Ctrl+Shift+R), then tick each.

## Generation
- [ ] **txt2img**: prompt + Generate → image shown, saved in `out/`, sidecar `.json` next to it.
- [ ] **Batch**: Image number = 4 → 4 images, browse with the gallery arrows ◀▶ + fullscreen button.
- [ ] **img2img / upscale**: Input Image → upload → Generate → upscaled result.
- [ ] **Batch img2img**: Input Image + Refine checked, Image number = 3 → 3 variations
      (seeds n, n+1, n+2, listed in the report). Refine **unchecked** → 1 image only
      (no diffusion = identical copies).

## Crop editor (inputs)
- [ ] Any image input (Describe / Source face / Vision Mix / Upscale / Reframe):
      upload → use the **crop** tool → the cropped area is what gets used.

## Inpaint
- [ ] Input Image → **Inpaint** → upload, paint white over an area, write a prompt → **Inpaint**.
- [ ] Result shows only the painted area changed. (Empty mask → clear message.)

## Reframe / Remove BG
- [ ] **Reframe (outpaint)** 16:9 → image expanded, borders filled.
- [ ] **Remove BG** → transparent PNG result.

## Face Swap
- [ ] **Face Swap**: source face + "Apply face swap" → face replaced.
- [ ] **Restore face (GFPGAN)** on → face sharper, **no square box** around it.

## Vision Mix (Ollama)
- [ ] Advanced > Prompt AI > **Detect** → vision models listed (no qwen3.6).
- [ ] **Vision Mix**: 2 refs → "Vision Mix & Generate" → composed image.

## Styles / Wildcards
- [ ] **Styles**: hover preview shows a thumbnail; search filters; "Random style each image" works.
- [ ] **Wildcards**: pick a file → contents load → **Insert __name__** → in prompt → Generate expands it.

## Galleries
- [ ] **History (this session)** accordion: renders accumulate.
- [ ] **Gallery (output folder)**: Refresh / sort / filter / select (metadata) / **Copy image** / Delete / Blur.
- [ ] **Asset Browser**: Models > Reindex + open link → new tab: grid, search, lightbox (◀▶/Esc), copy.

## Models
- [ ] **Checkpoints** dropdown switches model; picking one auto-tunes steps/CFG (profile).
- [ ] **LoRA**: folder + Refresh **persists** (survives restart); keywords auto-fill.
- [ ] **Transformer override** (e.g. Juggernaut repo/folder) loads + auto Base-CFG.
- [ ] FP8 checkpoints are **absent** from the dropdown.

## Launchers / sharing (optional)
- [ ] `run_quality_rtx5090_lan.bat` → open `http://<LAN-IP>:7860` from another device.
- [ ] `run_quality_rtx5090_web.bat` (+ `cloudflare.local.bat`) → tunnel URL works.
- [ ] Pinokio: install `mikecastrodemaria/crispz-studio.pinokio` → Install → Start → Open Web UI.

## Notes
- GPU tight with Ollama? set `ollama_cpu: true` (CPU captions) or `ollama_keep_alive: 0` (already default).
- Before committing: run `tools\check.bat` (py_compile + smoke test).
