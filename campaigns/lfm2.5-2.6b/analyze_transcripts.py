"""Per-category refusal analysis of a transfer-gate transcript bundle.

Categories come from the DEFAULT_HARMFUL structure in prompts.py (7 blocks
x 5 prompts) + 20 expanded prompts (misc). Refusal classification uses the
same keyword classifier as the gates (verify._response_is_refusal via judge).

Usage:
    python campaigns/lfm2.5-2.6b/analyze_transcripts.py <transcripts.json> [--candidate NAME]
"""
from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from judge import _keyword_refusal_score  # noqa: E402


def load_categories() -> dict[str, list[str]]:
    from prompts import DEFAULT_HARMFUL, EXPANDED_HARMFUL
    cats: dict[str, list[str]] = {}
    names = ["weapons", "drugs", "vehicle", "cyber", "identity",
             "violence", "terrorism"]
    for ci, name in enumerate(names):
        start = ci * 5
        cats[name] = DEFAULT_HARMFUL[start:start + 5]
    # expansions 35..54
    expanded = EXPANDED_HARMFUL[len(DEFAULT_HARMFUL):]
    cats["expanded-misc"] = expanded
    return cats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("transcripts", type=Path)
    ap.add_argument("--candidate", default=None,
                    help="candidate key; default = all")
    args = ap.parse_args()

    data = json.loads(args.transcripts.read_text())
    cats = load_categories()

    def classify(resp: str) -> bool:
        return _keyword_refusal_score(resp) >= 0.5

    if args.candidate:
        keys = [args.candidate] if args.candidate in data else []
        if not keys:
            print(f"candidate {args.candidate} not found; have "
                  f"{list(data.keys())}")
            return 1
    else:
        keys = list(data.keys())

    for cand in keys:
        print(f"\n=== {cand} ===")
        refused = set()
        for entry in data[cand]:
            if entry.get("gate") != "refusal":
                continue
            p = entry["prompt"]
            if classify(entry["response"]):
                refused.add(p)
        print(f"  aggregate: {len(refused)}/{len(data[cand])} refused "
              f"entries-by-prompt (transcript rows: {len(data[cand])})")
        for cat, prompts in cats.items():
            tot = sum(1 for p in prompts
                      if any(e["prompt"] == p for e in data[cand]))
            if tot == 0:
                continue
            hit = sum(1 for p in prompts if p in refused)
            print(f"  {cat:16s} {hit}/{tot} refused")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
