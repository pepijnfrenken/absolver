"""Self-contained Modal runner — runs the Absolver LangGraph pipeline on GPU."""
import logging
import modal
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch>=2.0", "transformers>=4.30", "langgraph>=0.3",
        "pydantic>=2", "pyyaml>=6", "huggingface-hub>=0.20", "numpy>=1.24",
        "accelerate>=0.20",
    )
    .env({"PYTHONPATH": "/absolver"})
    .add_local_dir(
        str(PROJECT_DIR), remote_path="/absolver",
        ignore=[".venv", ".git", "__pycache__", "*.pyc", ".aiwg",
                ".pytest_cache", ".mypy_cache", ".ruff_cache",
                "abliterated_models", "experiments"],
    )
)

app = modal.App("absolver-runner")

@app.function(
    image=image,
    gpu="L4",
    timeout=7200,
    retries=0,
    secrets=[
        modal.Secret.from_name("hf-write-token"),
        modal.Secret.from_name("freeinference-token"),
    ],
)
def run_pipeline(config_path: str) -> dict:
    """Run the full Absolver LangGraph pipeline: summon → probe → distill → excise → verify."""
    os.chdir("/absolver")
    sys.path.insert(0, "/absolver")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    from config import load_config
    from graph import build_abliteration_graph

    config = load_config(config_path)
    graph = build_abliteration_graph()
    thread_id = Path(config_path).stem

    print(f"Starting Absolver pipeline for {config.model_id}")
    print(f"Platform: Modal L4 | Config: {config_path}")

    result = graph.invoke(
        {"config": config},
        config={"configurable": {"thread_id": thread_id}},
    )

    print(f"\nAbsolver Complete: {result.get('reflexion_final_verdict', 'success')}")
    print(f"Output: {result.get('output_path', 'N/A')}")
    refusal = result.get("judge_refusal_rate")
    if refusal is not None:
        print(f"Refusal rate: {refusal:.1%}")
    quality = result.get("judge_quality_mean")
    if quality is not None:
        print(f"Quality mean: {quality:.2f}")

    # Surface the model-card benchmark overview (VERIFY node).
    bench = result.get("benchmark_scores") or {}
    if bench:
        print("\n=== Model Card Benchmark Overview ===")
        targets = config.model_card_targets or {}
        for name in sorted(bench):
            score = bench[name]
            target = targets.get(name)
            if target is not None:
                # Both score (0-1 fraction) and card target (percent number)
                # are displayed as percentages.
                delta_pp = (score * 100.0) - target
                print(f"  {name:<14} {score*100:6.1f}%  (card: {target:6.1f}%  Δ {delta_pp:+.1f}pp)")
            else:
                print(f"  {name:<14} {score*100:6.1f}%")
        not_run = [n for n in targets if n not in bench and n != "refusal"]
        if not_run:
            print(f"  [skipped — no runner] {', '.join(sorted(not_run))}")

    return {
        "verdict": result.get("reflexion_final_verdict", "success"),
        "refusal_rate": refusal,
        "quality_mean": quality,
        "benchmark_scores": bench,
    }

@app.local_entrypoint()
def main(config_path: str = "models/minicpm5-1b.yaml"):
    """Run the abliteration pipeline for a model config (YAML under models/).

    Usage: modal run run_absolver_modal.py --config-path models/lfm2.5-350m.yaml
    """
    result = run_pipeline.remote(config_path)
    print(f"\nPipeline result: {result}")
