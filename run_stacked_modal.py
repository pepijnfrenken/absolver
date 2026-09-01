"""Directly test the stacked ablation at the diag-proven corner.

Applies paired output-phase (primary) + diff_means input-phase (secondary)
at alpha=10, layers 20-27, o_proj+down_proj, then runs the E03 gates on a
held-out split. This isolates whether stacking actually removes refusal
(quick, one candidate — vs the 144-candidate grid).
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

app = modal.App("absolver-stacked-check")


@app.function(image=image, gpu="L4", timeout=2400, retries=0,
              secrets=[modal.Secret.from_name("hf-write-token"),
                       modal.Secret.from_name("freeinference-token")])
def run_stacked() -> str:
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
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=torch.bfloat16).to("cuda")
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    h = list(DEFAULT_HARMFUL); g = list(DEFAULT_HARMLESS)
    n = min(len(h), len(g))
    test_size = min(10, n // 2)
    split = build_split(h[:n], g[:n],
                        train_size=n - 2 * test_size,
                        tune_size=test_size, test_size=test_size,
                        seed=cfg.eval_split_seed)
    held_out = list(split.test)
    train_harm, train_harmless = list(split.train), list(split.tune) + list(split.test)

    # --- collect activations ---
    from probe import _find_layers, _make_hook, _to_device
    from distill import extract_directions
    from collections import defaultdict

    layers = _find_layers(model, "dense")
    num_layers = len(layers)

    def collect(prompts, phase="input"):
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

    # input-phase: harmful vs harmless
    harm_in = collect(train_harm)
    harm_less_in = collect(train_harmless)
    # paired output-phase: use the SAME harmful prompts with an affirmative
    # prefill to get the paired (refusal vs compliant) contrast.
    prefill = cfg.paired_prefill or "Sure, I can help with that."
    paired_refusal = collect(train_harm)
    paired_affirm = []
    # Build the paired affirm set: prefill + prompt
    aff_prompts = [f"{prefill}\n\n{p}" for p in train_harm]
    paired_affirm = collect(aff_prompts)

    # primary = paired output-phase, secondary = diff_means input-phase
    primary, _ = extract_directions(paired_refusal, paired_affirm, num_layers,
                                    model.config.hidden_size, "paired", 3, "cuda")
    secondary, _ = extract_directions(harm_in, harm_less_in, num_layers,
                                      model.config.hidden_size, "diff_means", 3, "cuda")

    # Apply the stacked ablation: layers 20-27, o_proj+down_proj, alpha=10
    from excise import _project_2d
    applied = []
    for li in range(20, 28):
        if li not in primary or li not in secondary:
            continue
        layer = layers[li]
        for wname, W in [("o_proj", getattr(getattr(layer, "self_attn", None), "o_proj", None)),
                         ("down_proj", getattr(getattr(layer, "mlp", None), "down_proj", None))]:
            if W is not None:
                _project_2d(W.weight.data, primary[li].to("cuda"), 10.0)
                _project_2d(W.weight.data, secondary[li].to("cuda"), 10.0)
                applied.append(f"L{li}.{wname}")

    # Gates on the held-out split
    report = run_gates(model, tok, cfg, prompts=held_out, benchmark_scores={})
    out = {"applied_weights": len(applied), "held_out_size": len(held_out),
           "primary_layers": len(primary), "secondary_layers": len(secondary)}
    for k, v in report.items():
        if k in ("_enabled", "eval_pass", "held_out_size"): continue
        out[k] = {"passed": v["passed"], "value": v.get("value"), "detail": v["detail"]}
    out["eval_pass"] = report.get("eval_pass")
    return json.dumps(out, indent=2, default=str)

@app.local_entrypoint()
def main():
    print(run_stacked.remote())
