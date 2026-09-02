"""Modal cloud runner for the harness/abl.py toolkit (CPU-only escape hatch).

Usage:
    modal run connectors/harness_modal.py -- collect models/lfm2.5-1.2b-instruct.yaml --model-dir campaigns/... --transcript
    modal run connectors/harness_modal.py -- steer-test models/lfm2.5-1.2b-instruct.yaml --alphas 1.0,2.0,4.0
    ABSOLVER_GPU=A10G modal run connectors/harness_modal.py -- ...

The repo is mounted at /absolver (heavy dirs excluded). HF token from Modal secret
'hf-write-token'. GPU default L4 (24GB).
"""
import os
import sys
from pathlib import Path

import modal  # only imported when run under `modal run`; machines without modal
              # simply won't execute this module's modal-dependent lines.

_PROJECT_DIR = Path(__file__).resolve().parent.parent

app = modal.App("absolver-harness")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch>=2.0",
        "transformers>=4.30",
        "numpy>=1.24",
        "pyyaml>=6",
        "huggingface-hub>=0.20",
        "safetensors>=0.4",
    )
    .env({"PYTHONPATH": "/absolver"})
    .add_local_dir(  # must be LAST build step
        str(_PROJECT_DIR),
        remote_path="/absolver",
        ignore=[".venv", ".git", "__pycache__", "*.pyc", ".aiwg", "abliterated_models", "campaigns", "experiments"],
    )
)

GPU = os.environ.get("ABSOLVER_GPU", "L4")
TIMEOUT = int(os.environ.get("ABSOLVER_TIMEOUT", "3600"))


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT,
    retries=0,
    secrets=[modal.Secret.from_name("hf-write-token")],
)
def run_harness_cmd(argv: list[str]) -> dict:
    import json
    import subprocess
    import time

    sys.path.insert(0, "/absolver")

    started = time.perf_counter()
    cmd = [sys.executable, "-m", "harness.abl"] + argv
    env = dict(os.environ)
    env["PYTHONPATH"] = "/absolver"
    env.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", ""))
    proc = subprocess.run(cmd, cwd="/absolver", env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - started

    return {
        "argv": argv,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-8000:],
        "stderr": proc.stderr[-8000:],
        "elapsed_seconds": round(elapsed, 1),
    }


@app.local_entrypoint()
def main(argv: str = ""):
    import json

    args = argv.split() if argv else []
    if not args:
        print("usage: modal run connectors/harness_modal.py -- '<harness subcommand> [args...]'")
        print("  e.g. modal run connectors/harness_modal.py -- 'list-campaigns'")
        sys.exit(1)
    result = run_harness_cmd.remote(args)
    print(json.dumps(result, indent=2))
