"""Modal runner: run E03 gates on a pristine model (baseline sanity).

Also runs the gates after applying a known ablation config, so we can see
the gate report delta. Usage:
  modal run run_gates_modal.py                # pristine only
  modal run run_gates_modal.py --ablated     # apply advanced alpha=10 wide + gates
"""
import modal
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.11", force_build=True)
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

app = modal.App("absolver-gates-check")

@app.function(image=image, gpu="L4", timeout=2400, retries=0,
              secrets=[modal.Secret.from_name("hf-write-token"),
                       modal.Secret.from_name("freeinference-token")])
def run_gates(apply_ablation: bool = False) -> str:
    os.chdir("/absolver")
    sys.path.insert(0, "/absolver")
    import logging, json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from config import load_config
    from prompts import DEFAULT_HARMFUL, DEFAULT_HARMLESS
    from gates import run_gates
    from eval_split import build_split

    cfg = load_config("models/qwen2.5-1.5b-instruct.yaml")
    model_id = cfg.model_id
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # Build held-out test split from the pools (35 harmful / 20 harmless)
    h = list(DEFAULT_HARMFUL); g = list(DEFAULT_HARMLESS)
    n = min(len(h), len(g))
    test_size = min(10, n // 2)
    split = build_split(h[:n], g[:n],
                        train_size=n - 2 * test_size,
                        tune_size=test_size, test_size=test_size,
                        seed=cfg.eval_split_seed)
    held_out = list(split.test)

    if apply_ablation:
        # Apply the diag-proven config: advanced alpha=10, layers 20-27,
        # o_proj + down_proj (plain projection, no rescale).
        from excise import _project_2d
        from distill import extract_directions
        from probe import _find_layers, _make_hook, _to_device
        from collections import defaultdict
        layers = _find_layers(model, "dense")
        num_layers = len(layers)
        # collect input-phase acts (harmful vs harmless) for diff_means
        def collect(prompts):
            store = defaultdict(list)
            handles = [layers[i].register_forward_hook(_make_hook(i, store))
                       for i in range(num_layers)]
            try:
                for p in prompts:
                    inp = tok(p, return_tensors="pt", truncation=True, max_length=128)
                    inp = _to_device(inp, "cuda")
                    with torch.no_grad():
                        model(**inp)
            finally:
                for hh in handles:
                    try: hh.remove()
                    except Exception: pass
            return dict(store)
        harm_acts = collect(h[:20]); harm_less = collect(g[:20])
        dirs, scores = extract_directions(
            harm_acts, harm_less, num_layers, model.config.hidden_size,
            "diff_means", 3, "cuda")
        for li in range(20, 28):
            if li not in dirs: continue
            layer = layers[li]
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
                _project_2d(layer.self_attn.o_proj.weight.data, dirs[li].to("cuda"), 10.0)
            ff = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
            if ff is not None and hasattr(ff, "down_proj"):
                _project_2d(ff.down_proj.weight.data, dirs[li].to("cuda"), 10.0)
        tag = "ABLATED(advanced a=10, L20-27, o+down)"
    else:
        tag = "PRISTINE"

    # Minimal benchmark for the capability gate (use mmlu_mini if available)
    benchmark_scores = {}
    try:
        from verify import run_mmlu_mini
        benchmark_scores["mmlu"] = run_mmlu_mini(model, tok, n=20)
    except Exception as exc:
        print("mmlu_mini failed:", exc)

    report = run_gates(model, tok, cfg, prompts=held_out,
                       benchmark_scores=benchmark_scores)
    out = {"tag": tag, "held_out_size": len(held_out), "benchmark_scores": benchmark_scores}
    for k, v in report.items():
        if k in ("_enabled", "eval_pass", "held_out_size"): continue
        out[k] = {"passed": v["passed"], "value": v.get("value"), "detail": v["detail"]}
    out["eval_pass"] = report.get("eval_pass")
    return json.dumps(out, indent=2, default=str)

@app.local_entrypoint()
def main(ablated: bool = False):
    print(run_gates.remote(apply_ablation=ablated))
