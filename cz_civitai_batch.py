"""crispz-studio — batch CivitAI enrichment (standalone, no torch import → starts fast).

Scans the LoRA and/or checkpoint folders and, for every `.safetensors`, fetches the
missing CivitAI info (preview + trigger words + example prompts) and refreshes the
"newer version available" flag — the same per-model action as the Asset Browser's
🔎 button, but for the whole folder.

Two entry points share the SAME core (`enrich`):
  - the Asset Browser "🔄 Fetch all missing" button (cz_ui runs `enrich` in a thread);
  - this script, meant to be launched from `civitai_index.bat` / `.sh` (and in PARALLEL
    with `--shard i/m` to split the file list across processes).

CLI:
  python cz_civitai_batch.py --kind {loras,models,all} [--force] [--all]
         [--shard i/m] [--loras-dir DIR] [--checkpoints-dir DIR]
         [--api-key KEY] [--sleep 0.5]

Only stdlib + cz_core/cz_civitai. Defensive: a corrupt sidecar, an unknown hash or a
network error never aborts the batch — that file is counted as failed/skipped and the
loop continues.
"""

import os
import sys
import time
import argparse

import cz_core
from cz_core import CONFIG, _prefs, HERE
import cz_civitai

_EXTS = (".safetensors", ".ckpt", ".pt", ".sft")
_BATCH_CFG = CONFIG.get("civitai_batch") if isinstance(CONFIG.get("civitai_batch"), dict) else {}
DEFAULT_SLEEP = float(_BATCH_CFG.get("sleep", 0.5))
DEFAULT_CHECK_UPDATES = bool(_BATCH_CFG.get("check_updates", True))


def resolve_dirs(loras_dir=None, checkpoints_dir=None):
    """Resout (loras_dir, [checkpoints_dirs...]) sans importer cz_pipeline (donc sans
    charger torch). Meme ordre de priorite que cz_pipeline: arg > env > prefs > config >
    defaut <HERE>/loras|checkpoints. Le dossier 'extra' checkpoints est inclus s'il existe."""
    loras = (loras_dir or os.environ.get("LORAS_DIR") or _prefs.get("loras_dir")
             or CONFIG.get("loras_dir") or os.path.join(HERE, "loras"))
    main_ck = (checkpoints_dir or os.environ.get("CHECKPOINTS_DIR")
               or _prefs.get("checkpoints_dir") or CONFIG.get("checkpoints_dir")
               or os.path.join(HERE, "checkpoints"))
    extra_ck = (os.environ.get("CHECKPOINTS_EXTRA_DIR") or _prefs.get("checkpoints_extra_dir")
                or CONFIG.get("checkpoints_extra_dir") or "").strip()
    cks = [main_ck] + ([extra_ck] if extra_ck else [])
    return os.path.abspath(loras), [os.path.abspath(c) for c in cks]


def _list_safetensors(dirs):
    """Liste RECURSIVE des modeles (.safetensors/.ckpt/.pt) dans les dossiers donnes.
    Dedoublonne par nom de fichier (le 1er dossier a la priorite, comme les checkpoints)."""
    out, seen = [], set()
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        for root, sub, files in os.walk(d):
            sub[:] = [x for x in sub if x not in ("_index", ".cache", "recipes")]
            for f in sorted(files):
                if f.lower().endswith(_EXTS) and f not in seen:
                    seen.add(f)
                    out.append(os.path.join(root, f))
    return sorted(out, key=lambda p: os.path.basename(p).lower())


def _apply_shard(files, shard):
    """shard 'i/m' (1<=i<=m) -> sous-ensemble files[i-1::m] (partition disjointe pour
    lancer m process en parallele). shard None/invalide -> liste inchangee."""
    if not shard:
        return files
    try:
        i, m = shard.split("/")
        i, m = int(i), int(m)
        if m >= 1 and 1 <= i <= m:
            return files[i - 1::m]
    except Exception:
        pass
    return files


def collect_files(kind, loras_dir=None, checkpoints_dir=None, shard=None):
    loras, cks = resolve_dirs(loras_dir, checkpoints_dir)
    files = []
    if kind in ("loras", "all"):
        files += _list_safetensors([loras])
    if kind in ("models", "all"):
        files += _list_safetensors(cks)
    return _apply_shard(files, shard)


def _needs_enrich(path, overwrite):
    """True s'il faut (re)telecharger: pas de sidecar civitai.json, OU pas de preview, OU
    overwrite demande."""
    if overwrite:
        return True
    if not cz_civitai.load_civitai_sidecar(path):
        return True
    return not cz_civitai.has_preview(path)


def enrich(files, api_key=None, overwrite=False, only_missing=True, sleep=DEFAULT_SLEEP,
           check_updates=DEFAULT_CHECK_UPDATES, progress=None):
    """Coeur partage (CLI + bouton UI). Pour chaque fichier: enrichit s'il manque des infos
    (ou overwrite), sinon rafraichit seulement le drapeau 'nouvelle version'. progress est
    appele progress(i, n, name, phase, text) — i est 1-base. Renvoie un dict resume."""
    n = len(files)
    summary = {"total": n, "enriched": 0, "skipped": 0, "updated": 0, "failed": 0,
               "warnings": []}

    def _emit(i, name, phase, text):
        if progress:
            try:
                progress(i, n, name, phase, text)
            except Exception:
                pass

    for i, path in enumerate(files, 1):
        name = os.path.basename(path)
        try:
            if only_missing and not _needs_enrich(path, overwrite):
                # deja enrichi -> juste (re)verifier la version, sans re-telecharger
                _emit(i, name, "update", f"{name}: checking version…")
                res = cz_civitai.refresh_update_flag(path, api_key) if check_updates else {}
                summary["skipped"] += 1
                if res.get("update_available"):
                    summary["updated"] += 1
                    summary["warnings"].append(f"{name}: newer version on CivitAI")
                if check_updates:
                    time.sleep(sleep)
                continue

            def _pf(phase, frac, text, _i=i, _name=name):
                _emit(_i, _name, phase, f"{_name}: {text}")

            _emit(i, name, "start", f"{name}: fetching…")
            res = cz_civitai.fetch_civitai_for_model(
                path, api_key=api_key, overwrite=overwrite, progress=_pf,
                check_update=check_updates)
            if res.get("success"):
                summary["enriched"] += 1
                if res.get("update_available"):
                    summary["updated"] += 1
                    summary["warnings"].append(f"{name}: newer version on CivitAI")
            else:
                summary["failed"] += 1
                summary["warnings"].append(f"{name}: {res.get('message', 'failed')}")
            time.sleep(sleep)
        except Exception as e:  # noqa: BLE001 - un fichier ne doit jamais casser le lot
            summary["failed"] += 1
            summary["warnings"].append(f"{name}: {e}")
            cz_core._dbg(f"batch enrich failed for {name}: {e}")
    return summary


def run(kind="all", loras_dir=None, checkpoints_dir=None, shard=None, api_key=None,
        overwrite=False, only_missing=True, sleep=DEFAULT_SLEEP,
        check_updates=DEFAULT_CHECK_UPDATES, progress=None):
    files = collect_files(kind, loras_dir, checkpoints_dir, shard)
    return enrich(files, api_key=api_key, overwrite=overwrite, only_missing=only_missing,
                  sleep=sleep, check_updates=check_updates, progress=progress)


def _cli_progress(i, n, name, phase, text):
    # une ligne reecrite en place: [12/48] modelname: Downloading preview…
    sys.stderr.write(f"\r[{i}/{n}] {text}".ljust(90)[:90])
    sys.stderr.flush()


def main(argv=None):
    # Sorties robustes meme sur une console cp1252 (le .bat peut tourner hors env Pinokio):
    # UTF-8 + remplacement, pour ne jamais planter sur … / — / emojis.
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="Batch CivitAI enrichment for crispz-studio "
                                             "(previews / trigger words / example prompts / "
                                             "new-version warnings).")
    ap.add_argument("--kind", choices=["loras", "models", "all"], default="all")
    ap.add_argument("--force", action="store_true",
                    help="Re-download everything, overwriting existing previews/sidecars.")
    ap.add_argument("--all", dest="all_files", action="store_true",
                    help="Process every file (re-query metadata) instead of only the "
                         "ones missing info. Previews are NOT overwritten unless --force.")
    ap.add_argument("--shard", default=None, metavar="i/m",
                    help="Process only shard i of m (1-based) — run several in parallel.")
    ap.add_argument("--loras-dir", default=None)
    ap.add_argument("--checkpoints-dir", default=None)
    ap.add_argument("--api-key", default=None, help="CivitAI API key (else config/prefs).")
    ap.add_argument("--sleep", type=float, default=DEFAULT_SLEEP,
                    help="Seconds to wait between requests (rate-limit friendly).")
    ap.add_argument("--no-check-updates", action="store_true",
                    help="Skip the newer-version check (faster, one less request/model).")
    a = ap.parse_args(argv)

    files = collect_files(a.kind, a.loras_dir, a.checkpoints_dir, a.shard)
    if not files:
        print(f"[civitai-batch] no models found for kind={a.kind} "
              f"(shard={a.shard or 'all'}). Nothing to do.")
        return 0
    print(f"[civitai-batch] {len(files)} model(s) — kind={a.kind}, shard={a.shard or 'all'}, "
          f"force={a.force}, only_missing={not a.all_files}, sleep={a.sleep}s")
    s = enrich(files, api_key=a.api_key, overwrite=a.force,
               only_missing=(not a.all_files and not a.force), sleep=a.sleep,
               check_updates=(not a.no_check_updates), progress=_cli_progress)
    sys.stderr.write("\n")
    print(f"[civitai-batch] done: enriched={s['enriched']} skipped={s['skipped']} "
          f"updated={s['updated']} failed={s['failed']}")
    for w in s["warnings"][:50]:
        print(f"  - {w}")
    # code de sortie non nul seulement si TOUT a echoue (utile pour un .bat)
    return 1 if (s["failed"] and not s["enriched"] and not s["skipped"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
