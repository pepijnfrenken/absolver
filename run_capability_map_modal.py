"""Modal entrypoint for the capability-map analysis.

Loads the model + tokenizer on a Modal GPU (L4), applies the same
prompt-format auto-detection as the pipeline, and writes a JSON report
to a mounted volume (or prints it).

Usage:
    PYTHONPATH=... python3 -m modal run run_capability_map_modal.py --config models/minicpm5-1b.yaml
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import modal

PROJECT_DIR = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch>=2.0", "transformers>=4.30", "pydantic>=2", "pyyaml>=6",
        "numpy>=1.24", "accelerate>=0.20",
    )
    .env({"PYTHONPATH": "/absolver"})
    .add_local_dir(
        str(PROJECT_DIR), remote_path="/absolver",
        ignore=[".venv", ".git", "__pycache__", "*.pyc", ".aiwg",
                ".pytest_cache", ".mypy_cache", ".ruff_cache",
                "abliterated_models", "experiments"],
    )
)

app = modal.App("capability-map")
VOLUME = modal.Volume.from_name("absolver-outputs", create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",
    timeout=1800,
    retries=0,
    secrets=[modal.Secret.from_name("freeinference-token")],
    volumes={"/outputs": VOLUME},
)
def run_capability_map(config_path: str) -> dict:
    """Load model, run the capability map, save JSON report to the volume."""
    import os
    import sys as _sys
    os.chdir("/absolver")
    _sys.path.insert(0, "/absolver")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)

    from config import load_config
    from model_registry import get_model, get_tokenizer
    from prompt_format import detect_prompt_format, format_prompt

    cfg = load_config(config_path)
    model = get_model()
    tok = get_tokenizer()
    fmt = detect_prompt_format(tok, getattr(cfg, "prompt_format", "auto"))
    logger = logging.getLogger("capmap")
    logger.info("Loaded %s | layers detected | prompt_format=%s", cfg.model_id, fmt)

    from capability_map import run_capability_map as run_map
    report = run_map(model, tok, lambda t, p: format_prompt(t, p, fmt), config_path)

    out = Path("/outputs") / f"capability_map_{Path(config_path).stem}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    VOLUME.commit()
    logger.info("Report written to volume: %s", out)

    # Compact summary for the console.
    print("\n=== CAPABILITY MAP SUMMARY ===")
    print(f"Refusal peak: L{report['refusal_peak_layer']}")
    for name, cap in report["capabilities"].items():
        ov = cap["overlap_with_refusal"]
        print(f"  {name:<12} peak=L{ov['capability_peak_layer']:<3} "
              f"corr={ov['correlation']:+.2f} top3_overlap={ov['top3_overlap']:.2f}")
    return report


@app.local_entrypoint()
def main(config: str):
    run_capability_map.remote(config)
