"""crispz-studio - Z-Image txt2img + an upscaler/detailer (standalone, without ComfyUI).

A thin entry point. All the code has been cut into cz_* modules:
  cz_core (config/paths/logging/device) · cz_imageio (image I/O) · cz_prompt (styles/
  wildcards) · cz_ollama (describe/improve/compose) · cz_esrgan (Real-ESRGAN) ·
  cz_face (faceswap/restore/BLIP/rembg) · cz_pipeline (the Z-Image core: generation,
  pipelines, LoRAs/checkpoints, offload, guidance) · cz_assetbrowser / cz_assets ·
  cz_ui (build_ui + the handlers) · cz_cli (argparse + the server).

This file only (1) launches the CLI/UI through cz_cli.cli_main and (2) re-exports the few
symbols tools/smoke_test.py reads through `import app`. The mutable runtime state
(LORAS / FACESWAP_RESTORE) is exposed as a live proxy by __getattr__.

To run:  python app.py            (the UI)
         python app.py --help     (the CLI)

"""

import sys

# The modules that hold the mutable runtime state (read live by __getattr__).
import cz_pipeline
import cz_face
import cz_esrgan

# Re-exports for the smoke test and for the `import app` backward compatibility (noqa: symbols unused here).
from cz_core import (  # noqa: F401
    CONFIG, COMPOSE_INSTRUCTION, IMPROVE_INSTRUCTION, DESCRIBE_INSTRUCTION,
    set_log_level,
)
from cz_prompt import STYLES, _apply_styles  # noqa: F401
from cz_imageio import _format_filename, save_image, _read_image_meta  # noqa: F401
from cz_pipeline import (  # noqa: F401
    _reframe_canvas, _gen_meta, set_loras, round_to_multiple,
    generate, txt2img_run, process_one, outpaint, inpaint_run, generate_omni,
)
from cz_face import set_faceswap_restore, _local_caption, _remove_bg  # noqa: F401
from cz_ui import (  # noqa: F401
    build_ui, run, _editor_img, _editor_to_image_mask, _gallery_load, _faceswap,
)
from cz_cli import cli_main, serve_main  # noqa: F401

main = cli_main


# A backward-compatibility facade: every symbol that moved (MUTABLE STATE included:
# LORAS, ESRGAN_DIR, BASE_REPO, FACESWAP_RESTORE, OFFLOAD_MODE, ...) stays reachable live
# through app.NAME. The smoke test (app.LORAS / app.FACESWAP_RESTORE) and
# cli_interactive.py (app.ESRGAN_DIR / app.BASE_REPO / app.set_esrgan_dir ...) depend on
# it. The first module that matches wins.
_PROXY_MODULES = (cz_pipeline, cz_esrgan, cz_face)


def __getattr__(name):
    for _m in _PROXY_MODULES:
        try:
            return getattr(_m, name)
        except AttributeError:
            continue
    raise AttributeError(f"module 'app' has no attribute {name!r}")


if __name__ == "__main__":
    sys.exit(cli_main())
