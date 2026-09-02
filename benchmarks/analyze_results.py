"""Delta table + per-capability verdicts from two lm-eval result JSONs.

Usage:
    python benchmarks/analyze_results.py <pristine.json> <ablated.json> [--markdown]

Emits the pristine-vs-ablated delta per benchmark and the per-capability
verdicts (knowledge, reasoning, instruction-following, tool use) used in the
model card. Every number comes from the JSON logs — no fabricated values.
"""
import argparse
import json
import sys
from pathlib import Path

# task -> (capability, metric as saved in results JSON, higher_is_better)
TASK_METRICS = {
    "mmlu_pro": ("Knowledge retrieval", "exact_match,custom-extract", True),
    "gpqa_diamond_n_shot": ("Reasoning", "acc_norm,none", True),
    "aime25": ("Reasoning (math)", "exact_match,none", True),
    "ifeval": ("Instruction following", "prompt_level_strict_acc,none", True),
}
NOT_REPLICATED = {
    "IFBench": "custom harness — not available in lm-eval; not replicated",
    "Multi-IF": "custom multi-turn harness — not available in lm-eval; not replicated",
    "BFCLv3": "tool-use harness (Berkeley Function Calling) — not available in lm-eval; not replicated",
}

# relative-retention verdict bands (ablated/pristine)
VERDICT_BANDS = [
    (0.95, float("inf"), "preserved"),
    (0.85, 0.95, "mildly degraded"),
    (0.0, 0.85, "degraded"),
]


def task_score(results: dict, task: str, metric: str) -> float | None:
    tr = results.get("results", {}).get(task)
    if not tr:
        return None
    v = tr.get(metric)
    return None if v is None else float(v)


def verdict_for(ratio: float | None) -> str:
    if ratio is None:
        return "untested+unknown"
    for lo, hi, label in VERDICT_BANDS:
        if lo <= ratio < hi:
            return label
    return "degraded"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pristine_json", type=Path)
    ap.add_argument("ablated_json", type=Path)
    ap.add_argument("--markdown", action="store_true",
                    help="emit markdown table instead of plain text")
    args = ap.parse_args()

    pri = json.loads(args.pristine_json.read_text())
    abl = json.loads(args.ablated_json.read_text())

    rows = []
    for task, (cap, metric, _hib) in TASK_METRICS.items():
        p = task_score(pri, task, metric)
        a = task_score(abl, task, metric)
        if p is None and a is None:
            continue  # task not run
        delta = None if (p is None or a is None) else a - p
        ratio = None if (p is None or a is None or p == 0) else a / p
        rows.append((task, cap, metric, p, a, delta, ratio))

    if args.markdown:
        print("| Benchmark | Capability | Metric | Pristine | Ablated | Δ | % retained | Verdict |")
        print("|---|---|---|---|---|---|---|---|")
    else:
        hdr = f"{'benchmark':<24}{'capability':<22}{'metric':<28}{'pristine':>9}{'ablated':>9}{'delta':>8}{'ret%':>7}  verdict"
        print(hdr)

    for task, cap, metric, p, a, delta, ratio in rows:
        if args.markdown:
            print(f"| {task} | {cap} | {metric} | "
                  f"{'—' if p is None else f'{p:.2f}'} | "
                  f"{'—' if a is None else f'{a:.2f}'} | "
                  f"{'—' if delta is None else f'{delta:+.2f}'} | "
                  f"{'—' if ratio is None else f'{ratio*100:.1f}%'} | "
                  f"{verdict_for(ratio)} |")
        else:
            print(f"{task:<24}{cap:<22}{metric:<28}"
                  f"{'—' if p is None else f'{p:>9.2f}'}"
                  f"{'—' if a is None else f'{a:>9.2f}'}"
                  f"{'—' if delta is None else f'{delta:>+8.2f}'}"
                  f"{'—' if ratio is None else f'{ratio*100:>6.1f}%'}  "
                  f"{verdict_for(ratio)}")

    print("\nNot replicated (honest gaps):")
    for bm, reason in NOT_REPLICATED.items():
        print(f"  - {bm}: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
