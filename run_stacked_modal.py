"""VALID corner test: can ANY ablation config pass the gates on Qwen2.5-1.5B?

Fixes the silent no-op in the previous stacked check:
  - split sizes now derive from the REAL pools (harmful=35, harmless=20),
    with a true TRAIN partition (>=1 prompt) for direction extraction;
  - directions are collected with real forward hooks (diff_means
    input-phase, proven causal by steering);
  - applied via the real sweep apply functions (MPOA / stacked);
  - gates run on a held-out TEST split, WITH pristine PPL / first-token KL
    baselines + an mmlu_mini capability score, so every gate is live.

CONCLUSION 2026-09-02 (after extensive local + Modal testing): no single
config passes the gates on this model. Plain projection is geometrically
too weak (refusal lives in a ~0.03% weight component). MPOA removes
refusal only in a narrow alpha band that breaks coherence on the harder
held-out set (a=10: 3/5 refuse; a=15: 0/5 refuse but 1/5 coherent;
a=20 o_proj-only: no effect). The Sept-1 "diag 0.0" was a stacked-
ablation artifact + these silent bugs. The honest verdict: Qwen2.5-1.5B
does not yield to single-shot weight ablation in this pipeline.
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
    from probe import _find_layers, _make_hook, _to_device
    from distill import extract_directions
    from sweep import _apply_mpoa
    from verify import _digest

    cfg = load_config("models/qwen2.5-1.5b-instruct.yaml")
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=torch.bfloat16).to("cuda")
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # ---- true data split: train for directions, TEST held out for gates ----
    h = list(DEFAULT_HARMFUL)
    g = list(DEFAULT_HARMLESS)
    # n=20 pairs (harmless is the binding constraint); gates on 5 held-out.
    n = min(len(h), len(g))
    n_test = 5
    n_train = n - 2 * n_test  # 10
    n_tune = n_test           # 5
    assert n_train >= 1, f"no train prompts: {n} pools, {n_test} test each side"
    split = build_split(h[:n], g[:n],
                        train_size=n_train, tune_size=n_tune, test_size=n_test,
                        seed=cfg.eval_split_seed)
    train_h = list(split.train)
    held_out = list(split.test)
    print(f"SPLIT: n_pairs={n} train={len(train_h)} tune={len(split.tune)} test={len(held_out)}")

    # pristine (untouched) snapshot, for PPL/KL baselines later
    pristine_sd = {k: v.clone().cpu() for k, v in model.state_dict().items()}
    layers = _find_layers(model, "dense")
    num_layers = len(layers)
    hidden = model.config.hidden_size
    dev = "cuda"

    # ---- collect directions: diff_means input-phase (the causal one) ----
    def collect_input(prompts):
        from collections import defaultdict
        store: dict = defaultdict(list)
        handles = [layers[i].register_forward_hook(_make_hook(i, store))
                   for i in range(num_layers)]
        try:
            for p in prompts:
                inp = tok(p, return_tensors="pt", truncation=True, max_length=128)
                inp = _to_device(inp, dev)
                with torch.no_grad():
                    model(**inp)
        finally:
            for hh in handles:
                try:
                    hh.remove()
                except Exception:
                    pass
        return dict(store)

    # diff_means input-phase directions: train harmful vs the positionally
    # matched train harmless (build_split pairs h[:n] with g[:n] 1:1).
    h_idx = {p: i for i, p in enumerate(h[:n])}
    train_g = [g[h_idx[p]] for p in train_h if p in h_idx]
    harm_in = collect_input(train_h)
    harm_less_in = collect_input(train_g)
    print(f"DIFF_MEANS input: harm layers={len(harm_in)} harmless layers={len(harm_less_in)}")

    dirs, _ = extract_directions(harm_in, harm_less_in, num_layers,
                                 hidden, "diff_means", 3, dev)
    print(f"DIRS: {len(dirs)} layers")

    # ---- pristine baselines FIRST (PPL + first-token logprob) on held-out ----
    pristine_logprobs = {}
    pristine_logprobs_first = {}
    try:
        from prompt_format import detect_prompt_format, format_prompt
        hfmt = detect_prompt_format(tok, getattr(cfg, "prompt_format", "auto"))
        for p in held_out:
            formatted = format_prompt(tok, p, hfmt)
            inp = tok(formatted, return_tensors="pt", truncation=True).to(dev)
            with torch.no_grad():
                out = model(**inp)
            lg = out.logits.float()
            lp_first = torch.log_softmax(lg[0, -1], dim=-1).cpu()
            pristine_logprobs_first[_digest(p)] = lp_first
            cont_lg = lg[0, inp["input_ids"].shape[1] - 1: lg.shape[1] - 1]
            lp = torch.log_softmax(cont_lg, dim=-1)
            tokens = inp["input_ids"][0, 1:]
            chosen = lp.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
            ppl = float(torch.exp(-chosen.sum() / max(1, chosen.numel())))
            pristine_logprobs[_digest(p)] = ppl
        print(f"PRISTINE baselines on {len(pristine_logprobs)} held-out prompts")
    except Exception as exc:
        print("pristine logprob collection failed:", exc)

    # ---- apply the WINNING config (local CPU evidence 2026-09-02): ----
    # MPOA alpha=10, tail layers 24-27, o_proj+down_proj (diff_means
    # input-phase direction). Plain projection is geometrically too weak
    # (0.03%/pass); MPOA's norm-preserving high-alpha achieves the
    # effective removal. Steer test showed subtracting ~20*d at MLP output
    # flips refusal to clean compliance; L24-27 MPOA a=10 is the weight
    # equivalent that keeps prose coherent (full L20-27 breaks it).
    candidate = {
        "method": "mpoa",
        "dir_method": "diff_means",
        "target_layers": list(range(24, 28)),
        "target_weights": ["o_proj", "down_proj"],
        "alpha": 10.0,
        "passes": 1,
    }
    _apply_mpoa(model, layers, dirs, candidate)
    print("APPLIED: MPOA L24-27 o_proj+down_proj alpha=10 (diff_means)", flush=True)

    # ---- capability (mmlu_mini) + full gates on the ablated model ----
    benchmark_scores = {}
    try:
        from verify import run_mmlu_mini
        benchmark_scores["mmlu"] = run_mmlu_mini(model, tok, n=20)
    except Exception as exc:
        print("mmlu_mini failed:", exc)

    report = run_gates(model, tok, cfg, prompts=held_out,
                       benchmark_scores=benchmark_scores,
                       pristine_logprobs=pristine_logprobs,
                       pristine_logprobs_first=pristine_logprobs_first)
    out = {"held_out_size": len(held_out), "train_size": len(train_h),
           "benchmark_scores": benchmark_scores}
    for k, v in report.items():
        if k in ("_enabled", "eval_pass", "held_out_size"):
            continue
        out[k] = {"passed": v["passed"], "value": v.get("value"), "detail": v["detail"]}
    out["eval_pass"] = report.get("eval_pass")
    return json.dumps(out, indent=2, default=str)


@app.local_entrypoint()
def main():
    print(run_stacked.remote())
