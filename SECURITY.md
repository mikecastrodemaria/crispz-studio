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

The app has **no authentication of any kind**. Exposing it beyond localhost — Gradio's
`share=True`, `server_name="0.0.0.0"`, or a reverse proxy — is outside the supported
configuration and is done at your own risk. In particular, `launch(allowed_paths=[...])`
deliberately grants the web UI read access to your output folder and your LoRA /
checkpoint directories so it can serve previews; on an exposed instance that becomes
file disclosure.

Reports that depend on the app being deliberately exposed to a network, or on the
operator loading model files they do not trust, are considered configuration choices
rather than vulnerabilities in crispz-studio.

## Known Dependabot alerts (assessed, not applicable)

This repository has **no lockfile** — dependencies are declared as ranges in
`requirements.txt`. Dependabot therefore reports *"the currently installed version
can't be determined"* and matches the **declared range** against the advisory range,
not the version actually installed. Both open alerts are artifacts of that matching:

| Alert | Package | Severity | Advisory range | Patched | Declared here |
|---|---|---|---|---|---|
| #3 | transformers | high | `< 5.5.0` | 5.5.0 | `>=4.51,<5` |
| #4 | gradio | low | `< 6.15.1` | 6.15.1 | `>=4.44,<6` |

Both advisory ranges sweep in an entire earlier release line: the transformers flaw was
found in **5.2.0** but the range covers all of 4.x, and the gradio flaw was found in
**6.14.0** but the range covers all of 5.x. Neither upper bound here can reach the
patched version, and the `<6` bound on gradio is deliberate (Brotli middleware bug with
h11, documented in `requirements.txt`).

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
