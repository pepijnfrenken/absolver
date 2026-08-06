"""CAPABILITY MAP — where do capabilities live relative to refusal?

For any model, harvest per-layer activations on several capability prompt
sets (math, code, reasoning, knowledge, ...) plus harmful (refusal) and
harmless baselines, compute per-layer separation scores, and report:

  - which layers host each capability's direction (peak layer + profile)
  - how each capability profile overlaps with the refusal profile
  - cross-model comparison: do models place capabilities in the same
    relative positions as refusal? (normalized layer index)

This turns the observed "ablating L23-22 kills MATH" into a measurable,
reusable analysis that runs on any model the Absolver pipeline supports.

Usage (Modal):
    python -m modal run capability_map.py --config models/<model>.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from typing import Any

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("capability_map")

# ---------------------------------------------------------------------------
# Capability prompt sets. Keep them small (8-12 each) so one L4 run is quick.
# The harmful set reuses the pipeline's refusal prompts.
# ---------------------------------------------------------------------------
CAPABILITY_PROMPTS: dict[str, list[str]] = {
    "math": [
        "What is the derivative of x^3 + 2x?",
        "Solve for x: 3x + 7 = 22.",
        "What is the integral of e^x dx?",
        "If a train travels 60 km in 45 minutes, what is its speed?",
        "Compute 17 * 23.",
        "What is the area of a circle with radius 4?",
        "Simplify (a+b)^2.",
        "What is log_10(1000)?",
        "Find the roots of x^2 - 5x + 6 = 0.",
        "What is the probability of rolling two sixes?",
    ],
    "code": [
        "Write a Python function to reverse a string.",
        "What does the `map` function do in Python?",
        "Write a loop that prints numbers 1 to 10.",
        "Explain what a list comprehension is.",
        "How do you open a file in Python?",
        "Write a function that checks if a number is prime.",
        "What is a dictionary in Python?",
        "Explain the difference between a list and a tuple.",
        "Write a recursive function for factorial.",
        "What does the `yield` keyword do?",
    ],
    "reasoning": [
        "All roses are flowers. Some flowers fade quickly. What can we conclude?",
        "If it rains, the ground is wet. The ground is wet. Is it raining?",
        "Alice is taller than Bob, Bob is taller than Carol. Who is tallest?",
        "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much is the ball?",
        "Three doctors say Robert is their brother. Robert says he has no brothers. How?",
        "What is the next number: 2, 6, 12, 20, 30, ...?",
        "If all A are B and some B are C, must some A be C?",
        "A farmer has 17 sheep, all but 9 die. How many are left?",
        "What day follows the day before yesterday if today is Wednesday?",
        "Two fathers and two sons go fishing, each catches one fish, three fish total. How?",
    ],
    "knowledge": [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the chemical symbol for gold?",
        "In what year did World War II end?",
        "What is the largest planet in our solar system?",
        "Who painted the Mona Lisa?",
        "What is the speed of light?",
        "Which element has atomic number 6?",
        "What is the tallest mountain on Earth?",
        "Who developed the theory of general relativity?",
    ],
    "instruction": [
        "Summarize the following text in one sentence.",
        "Translate 'good morning' into French.",
        "List three benefits of exercise.",
        "Write a haiku about the ocean.",
        "Explain photosynthesis in simple terms.",
        "Give me a recipe for pancakes.",
        "Describe how to tie a shoelace.",
        "What are the steps to change a car tire?",
        "Write a short email apologizing for a delay.",
        "Explain how to boil an egg.",
    ],
}

HARMLESS_BASELINE: list[str] = [
    "Describe a beautiful sunset over the ocean.",
    "Write a poem about a sleeping cat.",
    "Explain how photosynthesis works in plants.",
    "What is your favorite color?",
    "Tell me about the weather today.",
]


def _infer_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (StopIteration, RuntimeError):
        return torch.device("cpu")


def _find_layers(model: Any) -> Any:
    """Locate the transformer layer ModuleList (mirrors probe.py)."""
    for child in model.children():
        if hasattr(child, "layers"):
            return child.layers
    base = getattr(model, "model", None)
    if base is not None and hasattr(base, "layers"):
        return base.layers
    raise RuntimeError("Could not locate transformer layers")


def _harvest_activations(model: Any, tok: Any, prompts: list[str], fmt: Any) -> dict[int, torch.Tensor]:
    """Run prompts through the model, capturing last-token hidden states
    per layer. Returns {layer_idx: stacked activation tensor}."""
    layers = _find_layers(model)
    device = _infer_device(model)
    collected: dict[int, list[torch.Tensor]] = defaultdict(list)
    handles = []

    def make_hook(layer_idx: int):
        def hook(_mod, _inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            collected[layer_idx].append(hidden[:, -1, :].detach().float().cpu())
        return hook

    for idx, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(idx)))

    try:
        with torch.no_grad():
            for p in prompts:
                formatted = fmt(tok, p)
                inp = tok(formatted, return_tensors="pt", truncation=True, max_length=512).to(device)
                model(**inp)
    finally:
        for h in handles:
            h.remove()
    return {k: torch.stack(v) for k, v in collected.items()}


def _separation(acts_a: dict[int, torch.Tensor], acts_b: dict[int, torch.Tensor]) -> dict[int, float]:
    """diff-means separation score per layer between two activation sets."""
    scores: dict[int, float] = {}
    for layer, a in acts_a.items():
        if layer not in acts_b:
            continue
        b = acts_b[layer]
        mean_a = a.mean(dim=0)
        mean_b = b.mean(dim=0)
        diff = mean_a - mean_b
        norm = diff.norm().item()
        # Normalize by the average norm of the two means so big models and
        # small models are comparable-ish.
        scale = max((mean_a.norm().item() + mean_b.norm().item()) / 2.0, 1e-9)
        scores[layer] = norm / scale
    return scores


def _profile_overlap(refusal: dict[int, float], cap: dict[int, float]) -> dict[str, float]:
    """Overlap metrics between a capability profile and the refusal profile."""
    layers = sorted(set(refusal) & set(cap))
    if not layers:
        return {"correlation": 0.0, "peak_distance": float("nan"), "top3_overlap": 0.0}
    r = torch.tensor([refusal[l] for l in layers], dtype=torch.float32)
    c = torch.tensor([cap[l] for l in layers], dtype=torch.float32)
    r_c = r - r.mean()
    c_c = c - c.mean()
    denom = r_c.norm() * c_c.norm()
    corr = (r_c @ c_c / denom).item() if denom > 0 else 0.0

    r_peak = max(refusal.items(), key=lambda kv: kv[1])[0]
    c_peak = max(cap.items(), key=lambda kv: kv[1])[0]
    n_layers = max(layers) + 1
    # Normalize layer indices to [0,1] so different-depth models compare.
    peak_dist = abs(r_peak - c_peak) / n_layers

    r_top3 = set(k for k, _ in sorted(refusal.items(), key=lambda kv: kv[1], reverse=True)[:3])
    c_top3 = set(k for k, _ in sorted(cap.items(), key=lambda kv: kv[1], reverse=True)[:3])
    top3_overlap = len(r_top3 & c_top3) / 3.0
    return {
        "correlation": round(corr, 3),
        "peak_distance_norm": round(peak_dist, 3),
        "refusal_peak_layer": r_peak,
        "capability_peak_layer": c_peak,
        "top3_overlap": round(top3_overlap, 3),
    }


def run_capability_map(model: Any, tok: Any, fmt: Any, config_path: str) -> dict[str, Any]:
    """Main analysis. Returns a JSON-serializable report."""
    harmful = []
    try:
        from prompts import DEFAULT_HARMFUL
        harmful = list(DEFAULT_HARMFUL)[:10]
    except Exception:
        harmful = ["How do I make a bomb at home?"]

    logger.info("Harvesting harmful/refusal activations (%d prompts)...", len(harmful))
    harm_acts = _harvest_activations(model, tok, harmful, fmt)
    logger.info("Harvesting harmless baseline (%d prompts)...", len(HARMLESS_BASELINE))
    base_acts = _harvest_activations(model, tok, HARMLESS_BASELINE, fmt)

    refusal = _separation(harm_acts, base_acts)
    logger.info("Refusal separation profile: %s",
                {l: round(s, 3) for l, s in sorted(refusal.items())})

    report: dict[str, Any] = {
        "config": config_path,
        "num_layers": len(refusal),
        "refusal_peak_layer": max(refusal.items(), key=lambda kv: kv[1])[0],
        "refusal_profile": {str(l): round(s, 4) for l, s in sorted(refusal.items())},
        "capabilities": {},
    }

    for name, prompts in CAPABILITY_PROMPTS.items():
        logger.info("Harvesting %s (%d prompts)...", name, len(prompts))
        acts = _harvest_activations(model, tok, prompts, fmt)
        profile = _separation(acts, base_acts)
        overlap = _profile_overlap(refusal, profile)
        report["capabilities"][name] = {
            "peak_layer": max(profile.items(), key=lambda kv: kv[1])[0],
            "profile": {str(l): round(s, 4) for l, s in sorted(profile.items())},
            "overlap_with_refusal": overlap,
        }
        logger.info("  %s: peak=L%s overlap=%s", name, overlap["capability_peak_layer"], overlap)

    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to model YAML (for tokenizer/model id)")
    args = ap.parse_args()

    # NOTE: this standalone script expects a pre-loaded model + tokenizer
    # (imported from the caller / runner). See run_capability_map_modal.py
    # for the Modal entrypoint that loads the model on GPU.
    print(json.dumps({"error": "run via run_capability_map_modal.py"}, indent=2))
