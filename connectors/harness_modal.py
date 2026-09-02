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
        "pydantic>=2",
    )
    .env({"PYTHONPATH": "/absolver"})
    .add_local_dir(  # must be LAST build step
        str(_PROJECT_DIR),
        remote_path="/absolver",
        ignore=[".venv", ".git", "__pycache__", "*.pyc", ".aiwg", "abliterated_models", "campaigns", "experiments", "abl-work"],
    )
)

GPU = os.environ.get("ABSOLVER_GPU", "L4")
TIMEOUT = int(os.environ.get("ABSOLVER_TIMEOUT", "3600"))

# Persistent artifact volume: the harness container is ephemeral and its
# image IGNORES the repo's campaigns/ dir, so an ablated model saved inside
# the container would vanish. ABSOLVER_CAMPAIGNS_ROOT points here; the
# volume is pulled back locally after the run (`modal volume get`).
OUT_VOLUME = modal.Volume.from_name("absolver-phase2", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT,
    retries=0,
    secrets=[modal.Secret.from_name("hf-write-token")],
    volumes={"/out": OUT_VOLUME},
)
def run_harness_cmd(argv: list[str]) -> dict:
    """Run one or more harness subcommands in THIS container.

    ``argv`` may hold several commands separated by ``;`` — each runs in
    sequence in the same container, so a model saved by ``abl`` is present
    for a later ``collect --model-dir`` step (artifacts land in /out, the
    mounted volume, via ABSOLVER_CAMPAIGNS_ROOT).
    """
    import json
    import subprocess
    import time

    sys.path.insert(0, "/absolver")

    segments = [s.strip() for s in " ".join(argv).split(";") if s.strip()]
    env = dict(os.environ)
    env["PYTHONPATH"] = "/absolver"
    env["ABSOLVER_CAMPAIGNS_ROOT"] = "/out"
    env.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", ""))
    # keep progress bars out of the captured output budget so real
    # tracebacks/evidence survive the 12k-char tail
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["TRANSFORMERS_VERBOSITY"] = "error"

    out = []
    overall_started = time.perf_counter()
    for seg in segments:
        seg_argv = seg.split()
        started = time.perf_counter()
        cmd = [sys.executable, "-m", "harness.abl"] + seg_argv
        proc = subprocess.run(cmd, cwd="/absolver", env=env,
                              capture_output=True, text=True)
        out.append({
            "argv": seg_argv,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-12000:],
            "stderr": proc.stderr[-12000:],
            "elapsed_seconds": round(time.perf_counter() - started, 1),
        })
    return {
        "segments": out,
        "total_elapsed_seconds": round(time.perf_counter() - overall_started, 1),
    }


@app.local_entrypoint()
def main(argv: str = ""):
    import json

    args = argv.split() if argv else []
    if not args:
        print("usage: modal run connectors/harness_modal.py -- '<subcmd> [args...][; <subcmd> ...]'")
        print("  e.g. modal run connectors/harness_modal.py -- 'list-campaigns'")
        print("  e.g. modal run connectors/harness_modal.py -- 'abl ...; collect ...'")
        print("fetch artifacts: modal volume get absolver-phase2 <remote-path> <local-path>")
        sys.exit(1)
    result = run_harness_cmd.remote(args)
    print(json.dumps(result, indent=2))
