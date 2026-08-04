"""Minimal Modal runner for Absolver on MiniCPM5-1B."""
import modal
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent  # absolver/ directory

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch>=2.0", "transformers>=4.30", "langgraph>=0.3",
        "pydantic>=2", "pyyaml>=6", "huggingface-hub>=0.20", "numpy>=1.24",
        "accelerate",
    )
    .env({"PYTHONPATH": "/absolver"})
    .add_local_dir(
        str(PROJECT_DIR),
        remote_path="/absolver",
        ignore=[
            ".venv", ".git", "__pycache__", "*.pyc",
            ".aiwg", ".claude", "abliterated_*",
            ".mypy_cache", ".pytest_cache", ".ruff_cache",
            ".modalignore", ".gitignore", "experiments",
        ],
    )
)

app = modal.App("absolver-minicpm5")


@app.function(
    image=image,
    gpu="L4",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def run_pipeline(config_path: str = "models/minicpm5-1b.yaml") -> dict:
    import json, os, sys, time
    os.chdir("/absolver")
    sys.path.insert(0, "/absolver")
    from config import load_config
    from graph import build_abliteration_graph

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    cfg = load_config(config_path)
    # Force CUDA device on Modal L4 — disable accelerate device_map which
    # can split small models across CPU/GPU and cause device mismatches.
    cfg.device = None  # tells summon.py to skip device_map
    cfg.low_cpu_mem_usage = False
    graph = build_abliteration_graph()
    started = time.perf_counter()

    result = graph.invoke(
        {"config": cfg},
        config={"configurable": {"thread_id": Path(config_path).stem}},
    )
    elapsed = time.perf_counter() - started

    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    output = {
        "model_id": cfg.model_id,
        "architecture": result.get("architecture"),
        "num_layers": result.get("num_layers"),
        "max_separation": max(result.get("separation_scores", {}).values(), default=0),
        "refusal_rate": result.get("judge_refusal_rate"),
        "quality_mean": result.get("judge_quality_mean"),
        "verdict": result.get("reflexion_final_verdict", "success"),
        "gpu": gpu, "peak_memory_gb": round(peak, 2),
        "elapsed_seconds": round(elapsed, 1),
    }
    print(json.dumps(output, indent=2))
    return output


@app.local_entrypoint()
def main(config: str = "models/minicpm5-1b.yaml"):
    result = run_pipeline.remote(config)
    print(f"\n✅ Done. Refusal: {result.get('refusal_rate', '?')}")
