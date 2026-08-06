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

    # Hard cap on TOTAL pipeline invocations. The in-graph ouroboros/reflexion
    # counters can reset when reflexion routes back through probe/distill
    # (their state returns drop ouroboros_count), which caused unbounded
    # excise->verify->judge->reflexion loops (observed: 2h, 10+ verdicts).
    # This outer cap guarantees termination no matter what the graph does.
    import time
    max_invocations = getattr(config, "pipeline_max_invocations", 3)
    result = None
    for invocation in range(1, max_invocations + 1):
        result = graph.invoke(
            {"config": config},
            config={"configurable": {"thread_id": thread_id}},
        )
        verdict = result.get("reflexion_final_verdict") or result.get("judge_verdict") or "success"
        if verdict in ("success", "incompatible", "failed", "pass"):
            # Terminal states — push happens inside rebirth for 'success'.
            if invocation > 1:
                print(f"[invocation {invocation}/{max_invocations}] terminal verdict: {verdict}")
            break
        print(f"[invocation {invocation}/{max_invocations}] non-terminal ({verdict}); re-invoking")
        time.sleep(2)
    else:
        print(f"WARNING: hit {max_invocations} invocation cap without a terminal verdict")

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

@app.function(
    image=image,
    gpu="L4",
    timeout=1800,
    retries=0,
    secrets=[
        modal.Secret.from_name("hf-write-token"),
        modal.Secret.from_name("freeinference-token"),
    ],
)
def evaluate_sweep_candidate(payload: dict) -> dict:
    """Evaluate ONE sweep candidate in its own Modal container.

    Loads the model, applies the candidate's ablation using the provided
    direction tensors, quick-scores refusal/quality/KL, and returns. This is
    the unit of parallelism for the sweep: modal.map fans candidates out
    across containers (~10 concurrent), turning N*20s serial into
    ~max-candidate-time + cold-start wall time.
    """
    import os
    import sys
    import torch
    os.chdir("/absolver")
    sys.path.insert(0, "/absolver")

    from summon import summon_node
    from model_registry import get_model, get_tokenizer
    from sweep import _apply_candidate, _quick_score
    from prompts import DEFAULT_HARMFUL

    model_id = payload["model_id"]
    cand = payload["candidate"]
    probe_cfg = payload["probe_cfg"]

    try:
        # Load the model via the SUMMON node (respects dtype/device/arch).
        summon_node({
            "config": type("C", (), {
                "model_id": model_id,
                "model_arch": "dense",
                "dtype": "bfloat16",
                "device": "auto",
                "trust_remote_code": False,
            })(),
            "model_loaded": False,
            "architecture": None,
            "hidden_size": None,
            "num_layers": None,
        })
        model = get_model()
        tok = get_tokenizer()

        # Rebuild direction tensors from the serialized plain data.
        dirs_plain = payload["directions"]
        directions = {}
        for k, v in dirs_plain.items():
            if isinstance(v, list):
                directions[int(k)] = torch.tensor(v, dtype=torch.float32)
            else:
                directions[int(k)] = v

        _apply_candidate(model, directions, None, cand, None)

        # Quick-score: generate on a handful of harmful prompts. The scorer
        # needs a config-like object for max_seq_len (only); build a minimal
        # one from the host-supplied probe_cfg. Passing cfg=None here used to
        # crash the parallel path with 'NoneType' has no attribute
        # 'max_seq_len', hiding every parallel candidate behind the except.
        prompts = list(DEFAULT_HARMFUL)[: probe_cfg.get("n_verify_prompts", 10)]
        _score_cfg = type("SweepScoreCfg", (), {
            "max_seq_len": probe_cfg.get("max_seq_len", 1024),
        })()
        score = _quick_score(model, tok, _score_cfg, prompts, base_logprobs=None)
        return score
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return {"refusal": 1.0, "quality": 0.0, "kl": None, "error": str(exc)}


@app.local_entrypoint()
def main(config_path: str = "models/minicpm5-1b.yaml"):
    """Run the abliteration pipeline for a model config (YAML under models/).

    Usage: modal run run_absolver_modal.py --config-path models/lfm2.5-350m.yaml
    """
    result = run_pipeline.remote(config_path)
    print(f"\nPipeline result: {result}")
