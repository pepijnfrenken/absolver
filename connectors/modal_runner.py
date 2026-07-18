"""Modal cloud GPU runner for the Absolver pipeline.

Usage:
    modal run connectors/modal_runner.py --config models/ornith-9b.yaml
    modal run connectors/modal_runner.py --config models/ornith-9b.yaml --gpu A100

Notes:
    - Modal is imported only inside the app factory / entrypoint so that
      importing this module (e.g. from connectors/__init__.py) works on
      machines without modal installed.
    - On Modal cloud or the submit machine, run via ``modal run`` which
      makes the ``modal`` module available at function-call time.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

_PROJECT_DIR = Path(__file__).resolve().parent.parent


# =====================================================================
# Helpers
# =====================================================================


def _print_results(cfg, state: dict, elapsed: float):
    """Pretty-print the pipeline results."""
    arch = state.get("architecture", "?")
    layers = state.get("num_layers", "?")
    sep = state.get("separation_scores", {})
    max_sep = max(sep.values()) if sep else 0
    refusal = state.get("judge_refusal_rate", state.get("refusal_rate", "?"))
    quality = state.get("judge_quality_mean", "?")
    verdict = state.get("judge_verdict", state.get("reflexion_final_verdict", "success"))

    print()
    print("=" * 60)
    print(f"  Absolver — {cfg.model_id}")
    print("=" * 60)
    print(f"  Architecture : {arch}  ({layers} layers)")
    if isinstance(max_sep, float):
        print(f"  Max separation: {max_sep:.1f}")
    if isinstance(refusal, float):
        print(f"  Refusal rate  : {refusal:.1%}")
    if isinstance(quality, float):
        print(f"  Quality mean  : {quality:.2f}")
    print(f"  Verdict       : {verdict}")
    print(f"  Elapsed       : {elapsed:.1f}s")
    print("=" * 60)


# =====================================================================
# Lazy Modal App factory  — only imports modal when called
# =====================================================================


def _build_image():
    import modal

    return (
        modal.Image.debian_slim(python_version="3.11")
        .uv_pip_install(
            "torch>=2.0",
            "transformers>=4.30",
            "langgraph>=0.3",
            "pydantic>=2",
            "pyyaml>=6",
            "huggingface-hub>=0.20",
            "numpy>=1.24",
        )
        .env({"PYTHONPATH": "/absolver"})
    )


def _make_mount(modal):
    return modal.Mount.from_local_dir(_PROJECT_DIR, remote_path="/absolver")


def _create_app() -> object:
    """Create and return the Modal App with its function registered.

    Call this only under ``modal run`` (or wherever ``import modal`` works).
    """
    import modal

    image = _build_image()
    mount = _make_mount(modal)
    gpu = os.environ.get("ABSOLVER_GPU", "L4")
    timeout = int(os.environ.get("ABSOLVER_TIMEOUT", "7200"))

    app = modal.App("absolver-runner")

    @app.function(
        image=image,
        gpu=gpu,
        timeout=timeout,
        retries=0,
        mounts=[mount],
        secrets=[modal.Secret.from_name("huggingface-token", required=False)],
    )
    def run_pipeline_modal_fn(config_path: str) -> dict:
        """Run the Absolver pipeline on Modal GPU."""
        import time

        if "HF_TOKEN" not in os.environ:
            os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
        started = time.perf_counter()

        sys.path.insert(0, "/absolver")
        from config import load_config
        from graph import build_abliteration_graph

        cfg = load_config(config_path)
        graph = build_abliteration_graph()
        initial = {"config": cfg}
        thread_id = Path(config_path).stem
        result = graph.invoke(initial, config={"configurable": {"thread_id": thread_id}})
        elapsed = time.perf_counter() - started

        _print_results(cfg, result, elapsed)

        import torch

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

        output = {
            "model_id": cfg.model_id,
            "architecture": result.get("architecture"),
            "num_layers": result.get("num_layers"),
            "max_separation": max(result.get("separation_scores", {}).values(), default=0),
            "refusal_rate": result.get("judge_refusal_rate", result.get("refusal_rate")),
            "quality_mean": result.get("judge_quality_mean"),
            "verdict": result.get("judge_verdict", result.get("reflexion_final_verdict", "success")),
            "gpu": gpu_name,
            "peak_memory_gb": round(peak_mem / 1e9, 2),
            "elapsed_seconds": round(elapsed, 1),
            "output_path": result.get("output_path"),
            "hub_push_success": result.get("hub_push_success"),
        }
        print()
        print(json.dumps(output, indent=2))
        return output

    @app.local_entrypoint()
    def main():
        import argparse

        parser = argparse.ArgumentParser(description="Absolver — Modal cloud runner")
        parser.add_argument("--config", required=True, help="Path to config YAML")
        parser.add_argument("--gpu", default="L4", help="GPU type: L4, A10G, A100, H100")
        parser.add_argument("--timeout", type=int, default=7200, help="Timeout in seconds")
        args = parser.parse_args()

        os.environ["ABSOLVER_GPU"] = args.gpu
        os.environ["ABSOLVER_TIMEOUT"] = str(args.timeout)

        result = run_pipeline_modal_fn.remote(args.config)
        print("\n✅ Done. Result saved.")

    return app


# =====================================================================
# Public API (no modal import needed to call these)
# =====================================================================


def run_pipeline_modal(config_path: str) -> dict:
    """Entry point for programmatic callers.

    Delegates to the Modal cloud function.  Requires ``modal`` to be
    importable (i.e. called from a machine that has it installed).
    """
    import modal

    app = _create_app()
    # Find the function by name in the app's registry
    for fn in app._function_name_map.values() if hasattr(app, '_function_name_map') else []:
        if hasattr(fn, 'remote'):
            return fn.remote(config_path)
    # Fallback: run the function directly in the local process
    import modal.runner
    return modal.runner.run_function(app, "run_pipeline_modal_fn", args=(config_path,))


# =====================================================================
# CLI (only runs when executed directly via ``modal run``)
# =====================================================================

if __name__ == "__main__":
    _create_app()
