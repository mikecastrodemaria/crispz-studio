"""CLI interactive pour crispz.

Demande chaque reglage avec un defaut (chargé depuis preferences.json si present),
puis propose de sauver les choix. Couvre 1:1 ce que l'UI Gradio expose.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PREFS_PATH = os.path.join(HERE, "preferences.json")

DEFAULTS = {
    "esrgan_dir": "",
    "zimage_model": "",
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
    "time_log": "",
}


def load_prefs():
    if not os.path.isfile(PREFS_PATH):
        return dict(DEFAULTS)
    try:
        with open(PREFS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        merged = dict(DEFAULTS)
        merged.update({k: v for k, v in data.items() if k in DEFAULTS})
        return merged
    except Exception as e:
        print(f"[WARN] preferences.json unreadable ({e}), falling back to defaults.")
        return dict(DEFAULTS)


def save_prefs(prefs):
    with open(PREFS_PATH, "w", encoding="utf-8") as f:
        json.dump(prefs, f, indent=2, ensure_ascii=False)
    print(f"Saved: {PREFS_PATH}")


def ask(label, default, cast=str):
    shown = f"{default}" if default != "" else "(empty)"
    raw = input(f"{label} [{shown}]: ").strip()
    if raw == "":
        return default
    try:
        return cast(raw)
    except ValueError:
        print(f"  Invalid value, keeping {shown}.")
        return default


def ask_choice(label, choices, default):
    txt = "/".join(c.upper() if c == default else c for c in choices)
    raw = input(f"{label} [{txt}]: ").strip().lower()
    if raw == "":
        return default
    if raw in choices:
        return raw
    print(f"  Invalid choice, keeping {default}.")
    return default


def ask_yes_no(label, default=True):
    suffix = "[Y/n]" if default else "[y/N]"
    raw = input(f"{label} {suffix}: ").strip().lower()
    if raw == "":
        return default
    return raw in ("o", "oui", "y", "yes")


def main():
    sys.path.insert(0, HERE)
    import app

    prefs = load_prefs()

    print("=== crispz - interactive CLI ===")

    # 1) Chemins / modeles
    esrgan_dir = ask("ESRGAN folder", prefs.get("esrgan_dir") or app.ESRGAN_DIR, str)
    app.set_esrgan_dir(esrgan_dir)
    zimage_model = ask("Z-Image model (HF repo or local path)",
                       prefs.get("zimage_model") or app.BASE_REPO, str)
    app.set_zimage_model(zimage_model)

    models = app.list_esrgan_models()
    if not models:
        print(f"[ERROR] No ESRGAN model in {app.ESRGAN_DIR}.")
        return 1

    print(f"\nESRGAN_DIR: {app.ESRGAN_DIR}")
    print(f"Z-Image   : {app.BASE_REPO}")
    print(f"ESRGAN models available: {len(models)}\n")

    # 2) Source : fichier ou dossier
    src = input("Source image OR folder (batch): ").strip().strip('"')
    while not src:
        src = input("  Required: ").strip().strip('"')
    is_batch = os.path.isdir(src)

    # 3) Modele ESRGAN
    print("\nESRGAN models:")
    for i, m in enumerate(models):
        marker = " *" if m == prefs.get("model") else ""
        print(f"  [{i}] {m}{marker}")
    default_idx = models.index(prefs["model"]) if prefs.get("model") in models else 0
    raw = input(f"Choice [{default_idx}]: ").strip()
    try:
        model_name = models[int(raw)] if raw else models[default_idx]
    except (ValueError, IndexError):
        model_name = models[default_idx]
        print(f"  Invalid choice, keeping {model_name}")

    # 4) Pipeline
    factor = ask("Factor (1.0-4.0)", prefs["factor"], float)
    denoise = ask("Denoise (0.0-0.8, ~0.30 advised)", prefs["denoise"], float)
    steps = ask("Diffusion steps (4-30)", prefs["steps"], int)
    prompt = ask("Optional prompt", prefs["prompt"], str)
    seed = ask("Seed (-1 = random)", prefs["seed"], int)
    tile = ask("ESRGAN tile (0 = off)", prefs["tile"], int)
    overlap = ask("Tiling overlap", prefs["overlap"], int)

    # 5) Sauvegarde
    print("\nSave mode:")
    print("  display    = no save, prints the timing only")
    print("  local      = saves in <output_dir>, relative to the project")
    print("  alongside  = saves next to the source file")
    print("  custom     = saves in <output_dir> as given (absolute)")
    save_mode = ask_choice("save_mode", ["display", "local", "alongside", "custom"], prefs["save_mode"])
    if save_mode in ("local", "custom"):
        output_dir = ask("output_dir", prefs["output_dir"], str)
    else:
        output_dir = prefs["output_dir"]
    output_format = ask_choice("Output format", ["png", "webp", "jpg"], prefs["output_format"])
    time_log = ask("Time-log (TSV path, empty = no log)", prefs.get("time_log", ""), str)

    # 6) Recap
    print("\n--- Recap ---")
    print(f"  src          : {src}{' (BATCH)' if is_batch else ''}")
    print(f"  model        : {model_name}")
    print(f"  factor       : {factor}")
    print(f"  denoise      : {denoise}")
    print(f"  steps        : {steps}")
    print(f"  prompt       : {prompt or '(empty)'}")
    print(f"  seed         : {seed}")
    print(f"  tile         : {tile}")
    print(f"  overlap      : {overlap}")
    print(f"  save_mode    : {save_mode}")
    print(f"  output_dir   : {output_dir}")
    print(f"  output_format: {output_format}")
    print(f"  time_log     : {time_log or '(none)'}\n")

    if not ask_yes_no("Run?", True):
        print("Cancelled.")
        return 0

    new_prefs = {
        "esrgan_dir": app.ESRGAN_DIR,
        "zimage_model": app.BASE_REPO,
        "model": model_name,
        "factor": factor,
        "denoise": denoise,
        "steps": steps,
        "prompt": prompt,
        "seed": seed,
        "tile": tile,
        "overlap": overlap,
        "save_mode": save_mode,
        "output_dir": output_dir,
        "output_format": output_format,
        "time_log": time_log,
    }

    argv = [
        "--cli",
        "-i", src,
        "-m", model_name,
        "--factor", str(factor),
        "--denoise", str(denoise),
        "--steps", str(steps),
        "--prompt", prompt,
        "--seed", str(seed),
        "--tile", str(tile),
        "--overlap", str(overlap),
        "--save-mode", save_mode,
        "--output-dir", output_dir,
        "--output-format", output_format,
    ]
    if time_log:
        argv += ["--time-log", time_log]

    rc = app.cli_main(argv)

    if rc == 0:
        print()
        if ask_yes_no("Save these settings as preferences?", True):
            save_prefs(new_prefs)
    return rc


if __name__ == "__main__":
    sys.exit(main())
