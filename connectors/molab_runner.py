"""Molab (CoreWeave) session connector for the Absolver pipeline.

Connects to a Marimo kernel running on a Molab GPU session and executes
the abliteration pipeline remotely via the Marimo kernel API.

Usage:
    python connectors/molab_runner.py --url https://sb-xxxxx.sb.molab.run --token xxx --config models/ornith-9b.yaml
    python connectors/molab_runner.py --config models/ornith-9b.yaml   # reads MOLAB_URL, MOLAB_TOKEN from env
"""

import argparse
import base64
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import requests

_PROJECT_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Marimo Kernel API helpers
# ---------------------------------------------------------------------------

def _get_session_id(url: str, token: str) -> Optional[str]:
    """Fetch the active Marimo session ID from the kernel."""
    resp = requests.get(
        f"{url}/api/sessions",
        headers={"Authorization": f"Bearer {token}"},
        timeout=15,
    )
    sessions = resp.json()
    if not sessions:
        print("❌ No active Marimo session found.")
        return None
    return list(sessions.keys())[0]


def _headers(url: str, token: str, session_id: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Marimo-Session-Id": session_id,
    }


def run_code(
    url: str,
    token: str,
    session_id: str,
    code: str,
    timeout: int = 600,
) -> tuple[bool, str]:
    """Execute Python code on the Molab kernel and return (success, output)."""
    hdrs = _headers(url, token, session_id)
    resp = requests.post(
        f"{url}/api/kernel/execute",
        headers=hdrs,
        json={"code": code},
        timeout=timeout,
        stream=True,
    )

    output_parts = []
    for line in resp.text.split("\n"):
        if line.startswith("data: ") and '"data":' in line:
            try:
                data = json.loads(line[6:]).get("data", "")
                if data:
                    output_parts.append(str(data))
            except json.JSONDecodeError:
                pass

    success = '"success": true' in resp.text
    output = "\n".join(output_parts)
    return success, output


def test_connection(url: str, token: str, session_id: str) -> bool:
    """Verify the kernel is alive and responsive."""
    success, output = run_code(
        url, token, session_id,
        "import marimo as mo; mo.status.toast('🔥 Absolver connected via Molab!')",
        timeout=30,
    )
    if success:
        print(f"✅ Connection to {url} OK")
    else:
        print(f"❌ Connection failed: {output[:200]}")
    return success


def install_deps(url: str, token: str, session_id: str) -> bool:
    """Install required Python packages on the remote kernel."""
    deps = [
        "torch>=2.0",
        "transformers>=4.30",
        "langgraph>=0.3",
        "pydantic>=2",
        "pyyaml>=6",
        "huggingface-hub>=0.20",
        "numpy>=1.24",
    ]
    deps_str = " ".join(deps)
    code = (
        "import subprocess, sys; "
        f"subprocess.run([sys.executable, '-m', 'pip', 'install', {deps_str!r}], check=True); "
        "print('✅ All deps installed')"
    )
    success, output = run_code(url, token, session_id, code, timeout=300)
    if success:
        print("✅ Dependencies installed")
    else:
        print(f"❌ Dep install failed: {output[:200]}")
    return success


def upload_project(url: str, token: str, session_id: str) -> bool:
    """Upload the absolution project to the Molab kernel via base64."""
    import zipfile
    import io

    # Zip the project directory (skip venv, pycache, .git)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in _PROJECT_DIR.rglob("*"):
            # Skip unwanted dirs
            skip_parts = {".venv", "__pycache__", ".git", ".aiwg", ".claude", ".gitignore"}
            if any(part in fpath.parts for part in skip_parts):
                continue
            if fpath.is_file() and fpath.suffix in (".py", ".yaml", ".yml", ".md", ".toml", ".txt"):
                arcname = str(fpath.relative_to(_PROJECT_DIR))
                zf.write(fpath, arcname)

    b64 = base64.b64encode(buf.getvalue()).decode()

    code = (
        "import base64, zipfile, io; "
        "import os; "
        f"PROJECT_B64 = {b64!r}; "
        "raw = base64.b64decode(PROJECT_B64); "
        "with zipfile.ZipFile(io.BytesIO(raw)) as zf: "
        "    zf.extractall('/absolver'); "
        "sys.path.insert(0, '/absolver'); "
        "print('✅ Project uploaded to /absolver')"
    )
    success, output = run_code(url, token, session_id, code, timeout=60)
    if success:
        print("✅ Project uploaded to remote kernel")
    else:
        print(f"❌ Upload failed: {output[:200]}")
    return success


def run_pipeline(url: str, token: str, session_id: str, config_path: str) -> dict:
    """Execute the full Absolver pipeline on the Molab kernel and return results."""

    # Read the config YAML and embed it directly to avoid filesystem issues
    config_content = Path(config_path).read_text()

    pipeline_code = f"""
import sys
sys.path.insert(0, '/absolver')

import json
import time
from pathlib import Path

from config import load_config
from graph import build_abliteration_graph

# Save config to a temp location
cfg_path = '/tmp/absolver-config.yaml'
Path(cfg_path).write_text({config_content!r})

cfg = load_config(cfg_path)
graph = build_abliteration_graph()

initial = {{'config': cfg}}
thread_id = '{Path(config_path).stem}'

started = time.perf_counter()
result = graph.invoke(initial, config={{{{'configurable': {{'thread_id': thread_id}}}}}})
elapsed = time.perf_counter() - started

# Extract key metrics
sep = result.get('separation_scores', {{}})
max_sep = max(sep.values()) if sep else 0
refusal = result.get('judge_refusal_rate', result.get('refusal_rate', '?'))
quality = result.get('judge_quality_mean', '?')
verdict = result.get('judge_verdict', result.get('reflexion_final_verdict', 'success'))

output = {{
    'model_id': cfg.model_id,
    'architecture': result.get('architecture'),
    'num_layers': result.get('num_layers'),
    'max_separation': max_sep,
    'refusal_rate': refusal,
    'quality_mean': quality,
    'verdict': verdict,
    'elapsed_seconds': round(elapsed, 1),
    'output_path': result.get('output_path'),
    'hub_push_success': result.get('hub_push_success'),
}}

print(json.dumps(output, indent=2))
"""
    success, output = run_code(url, token, session_id, pipeline_code, timeout=7200)

    # Try to parse JSON result from output
    result_data = {}
    for line in output.split("\n"):
        line = line.strip()
        if line.startswith("{"):
            try:
                result_data = json.loads(line)
            except json.JSONDecodeError:
                pass

    if success:
        print("✅ Pipeline completed on Molab")
    else:
        print(f"❌ Pipeline failed: {output[:500]}")

    return result_data


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Absolver — Molab session runner")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--url", default=None, help="Molab endpoint URL (default: $MOLAB_URL)")
    parser.add_argument("--token", default=None, help="API token (default: $MOLAB_TOKEN)")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-upload", action="store_true", help="Skip project upload")
    args = parser.parse_args()

    url = args.url or os.environ.get("MOLAB_URL")
    token = args.token or os.environ.get("MOLAB_TOKEN")

    if not url or not token:
        print("❌ Provide --url + --token or set MOLAB_URL / MOLAB_TOKEN env vars")
        sys.exit(1)

    # Get session ID
    session_id = _get_session_id(url, token)
    if not session_id:
        sys.exit(1)
    print(f"📡 Session ID: {session_id}")

    # Test connection
    if not test_connection(url, token, session_id):
        sys.exit(1)

    # Install deps
    if not args.skip_deps:
        if not install_deps(url, token, session_id):
            sys.exit(1)

    # Upload project
    if not args.skip_upload:
        if not upload_project(url, token, session_id):
            sys.exit(1)

    # Run pipeline
    result = run_pipeline(url, token, session_id, args.config)

    if result:
        print()
        print("=" * 60)
        print("  Absolver — Molab Results")
        print("=" * 60)
        print(json.dumps(result, indent=2))
        print("=" * 60)


def run_pipeline_molab(config_path: str) -> dict:
    """Programmatic entry: reads env vars for connection, runs pipeline, returns results."""
    url = os.environ.get("MOLAB_URL")
    token = os.environ.get("MOLAB_TOKEN")
    if not url or not token:
        raise RuntimeError("MOLAB_URL and MOLAB_TOKEN are required")

    session_id = _get_session_id(url, token)
    if not session_id:
        raise RuntimeError("No active Molab session found")

    test_connection(url, token, session_id)
    return run_pipeline(url, token, session_id, config_path)


if __name__ == "__main__":
    main()
