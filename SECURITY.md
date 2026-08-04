# Security Policy

## Reporting a vulnerability

Please use **private vulnerability reporting**, which is enabled on this repository:
*Security* tab → *Report a vulnerability*. That keeps the report confidential until a
fix is available.

Do **not** open a public issue for a security problem.

## Supported versions

crispz-studio is developed on `main` with no maintenance branches: only the **latest
commit on `main`** receives fixes. The current version is in `cz_core.py`
(`APP_VERSION`) and shown in the browser tab title.

## Scope

crispz-studio is a **local desktop application**. `build_ui().launch()`
(`cz_cli.py`) runs with Gradio's defaults: it binds **127.0.0.1** and creates **no
public share link**.

By default the app has **no authentication**. An **optional login page** exists: set
`auth` in `config.txt` to `"user:password"` (several accounts via commas), or pass
`--auth user:pw`, or set the `CRISPZ_AUTH` environment variable — Gradio then gates
every route (UI, API endpoints, and `file=` serving all return 401 until login).
**Enable it before** exposing the app beyond localhost — Gradio's `share=True`,
`server_name="0.0.0.0"`, a tunnel, or a reverse proxy. Exposure without auth is outside
the supported configuration and is done at your own risk. In particular,
`launch(allowed_paths=[...])` deliberately grants the web UI read access to your output
folder and your LoRA / checkpoint directories so it can serve previews; on an exposed
unauthenticated instance that becomes file disclosure.

Reports that depend on the app being deliberately exposed to a network, or on the
operator loading model files they do not trust, are considered configuration choices
rather than vulnerabilities in crispz-studio.

## Known Dependabot alerts

Since `requirements-lock.txt` was added, Dependabot matches **pinned versions** instead of
ranges, and reports every advisory that touches them. Each one is either **fixed by an
upgrade** or **assessed as unreachable** in this application. The scope above is what makes
that distinction meaningful: crispz-studio is a local app bound to 127.0.0.1, and the
alerts that need a network-exposed service are out of the supported configuration.

### Fixed by upgrading

| Package | Was | Now | Alerts closed |
|---|---|---|---|
| pillow | 11.3.0 | **12.3.0** | 18 (PSD/FITS/JPEG2000/McIdas OOB, font + PDF decompression bombs, `RankFilter`, `ImageCmsTransform`, `paste`/`crop` overflow, TGA RLE, `WindowsViewer`) |
| protobuf | 6.31.0 | **7.35.1** | 2 (JSON recursion bypass, DoS) |
| sentencepiece | 0.1.96 | **0.2.2** | 1 (heap overflow) |

Pillow was the priority: it is the one flagged package that parses **files the user
supplies** (Input image, PNG Info drop), so those advisories were genuinely reachable.
Verified against the GitHub advisory database: **0 advisories remain** for the three
pinned versions.

### Assessed — not reachable in this application

| Package | Alerts | Why it does not apply |
|---|---|---|
| rembg | 4 | All four target the **rembg HTTP server** (`/api/remove`, CORS, custom-model path traversal). crispz imports `remove()` as a **library** and never starts that server. Two of the four have **no patched release at all**, and the patched line (2.0.75+) requires **Python ≥ 3.11** while this app runs on 3.10. |
| gradio | 6 | `gr.load()` SSRF and both OAuth flaws — **neither API is used** (`git grep 'gr\.load(\|oauth'` is empty). Windows absolute-path traversal needs **Python 3.13+** (this app runs 3.10). Audio cache key needs `gr.Audio`, which does not exist here. Cookie injection needs a network-exposed instance. Patching would require **gradio 6.x**, a major bump; the `<6` pin is deliberate (Brotli/h11 middleware bug, documented in `requirements.txt`). |
| transformers | 3 | The `Trainer` RCE needs `Trainer` (no training here), the LightGlue RCE needs that model (never invoked), and the general RCE path needs `trust_remote_code=True` — **never set** anywhere in the codebase. Patching needs **transformers 5.x**, which diffusers' current pin does not support. |
| torch | 3 | `torch.jit.script`, `lstm_cell` and `unpack_sequence` are **not called** anywhere in the codebase. All three are Moderate/Low, and their fixes land in torch 2.10+/2.13+ — there is no `+cu128` build of those for the RTX 5090 setup this project targets, so upgrading would break GPU support to close unreachable flaws. |

The upgrades were verified on the real environment: `torch 2.8.0+cu128` and CUDA still
load, `build_ui()` still builds, the full image chain (save + metadata round-trip +
thumbnail + `RankFilter` + `crop` + WebP) still works, and the test suites pass.

### transformers — CVE-2026-5241 (LightGlue model loading)

The vulnerable path requires `AutoModel.from_pretrained()` on a LightGlue repository,
where `LightGlueConfig` reads `trust_remote_code` from an untrusted `config.json` and
propagates it into nested `AutoConfig.from_pretrained()` calls. None of those
preconditions exist here:

```
$ git grep -nEi 'lightglue|superglue|superpoint' -- '*.py'   # no match
$ git grep -nEi 'trust_remote_code' -- '*.py'                # no match
$ git grep -nEi 'Auto[A-Za-z]*\.from_pretrained' -- '*.py'   # no match

$ git grep -n 'from transformers' -- '*.py'
cz_face.py:71:    from transformers import BlipProcessor, BlipForConditionalGeneration
```

transformers is used **only** through explicitly named classes (`BlipProcessor` /
`BlipForConditionalGeneration`, for the captioner). The flaw lives in the `Auto*`
resolution step, which reads an untrusted config to pick a class — naming the class
directly means that step never runs. Everything else goes through diffusers.

The LightGlue code does ship inside the installed transformers package; it is simply
never invoked by this application.

### gradio — CVE-2026-10783 (audio cache key)

The flaw is in `save_audio_to_cache`, reachable only through the Audio component. The
app has no audio surface at all:

```
$ git grep -nEi 'gr\.Audio|save_audio_to_cache' -- '*.py'    # no match
```

These assessments are re-checked when the pins change. If you believe one is wrong,
please report it through the private reporting channel above.
