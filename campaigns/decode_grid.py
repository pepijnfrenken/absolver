#!/usr/bin/env python3
"""Decode-grid rescue test for abliterated LFM2.5 models.

Runs the SAME held-out harmful + benign chat-formatted prompt set the
`collect` gates use across a decoding grid (repetition_penalty,
no_repeat_ngram_size, temperature), on ONE model dir per invocation.
Each response is classified with the repo's own refusal / coherence /
degeneracy helpers so the grid is directly comparable to gate counts.

Usage:
  .venv/bin/python campaigns/decode_grid.py --model-dir <dir> --out <out.json> [--tag TAG] [--max-new 64]

Model is loaded once, freed at exit (gc.collect()). Never loads two models.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_split import build_split
from gates import _coherent_completion, _degeneracy_ratio
from prompt_format import format_prompt
from prompts import DEFAULT_HARMFUL, DEFAULT_HARMLESS
from verify import _decode_continuation, _response_is_refusal

# decoding grid: (label, generate kwargs)
GRID: list[tuple[str, dict]] = [
    ("control",        {}),
    ("rep1.05",        {"repetition_penalty": 1.05}),
    ("rep1.1",         {"repetition_penalty": 1.1}),
    ("rep1.2",         {"repetition_penalty": 1.2}),
    ("ngram3",         {"no_repeat_ngram_size": 3}),
    ("ngram4",         {"no_repeat_ngram_size": 4}),
    ("ngram3+rep1.1",  {"no_repeat_ngram_size": 3, "repetition_penalty": 1.1}),
    ("ngram4+rep1.1",  {"no_repeat_ngram_size": 4, "repetition_penalty": 1.1}),
    ("sample-t0.7",    {"do_sample": True, "temperature": 0.7, "top_p": 0.95,
                        "repetition_penalty": 1.1}),
    ("sample-t0.9",    {"do_sample": True, "temperature": 0.9, "top_p": 0.95,
                        "repetition_penalty": 1.1}),
]


def held_out_prompts() -> tuple[list[str], list[str]]:
    """The collect gate's held-out harmful split + small benign set."""
    h = list(DEFAULT_HARMFUL)
    g = list(DEFAULT_HARMLESS)
    n = min(len(h), len(g))
    split = build_split(h[:n], g[:n], train_size=n - 10, tune_size=5,
                        test_size=5, seed="absolver:qwen25:v1")
    return list(split.test), list(DEFAULT_HARMLESS)[:3]


def classify(resp: str) -> dict:
    return {
        "refusal": _response_is_refusal(resp),
        "coherent": _coherent_completion(resp),
        "degeneracy_ratio": round(_degeneracy_ratio(resp), 4),
    }


def gen(model, tok, formatted: str, max_new: int, kwargs: dict) -> str:
    inp = tok(formatted, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        raw = model.generate(
            **inp, max_new_tokens=max_new, pad_token_id=tok.eos_token_id,
            **kwargs)
    return _decode_continuation(tok, raw, inp["input_ids"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--tag", default="", help="optional label for the JSON")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--labels", default="",
                    help="comma list of grid labels to run (default: all)")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    grid = [(lbl, kw) for lbl, kw in GRID if not labels or lbl in labels]

    harmful, benign = held_out_prompts()
    print(f"held-out harmful ({len(harmful)}):")
    for p in harmful:
        print(f"  - {p}")
    print(f"benign ({len(benign)}):")
    for p in benign:
        print(f"  - {p}")

    t0 = time.time()
    print(f"\nLoading model from {model_dir} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    print(f"  loaded in {time.time() - t0:.0f}s; prompt flavor: chat")

    doc = {
        "model_dir": str(model_dir),
        "tag": args.tag,
        "max_new_tokens": args.max_new,
        "flavor": "chat",
        "harmful_prompts": harmful,
        "benign_prompts": benign,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "configs": [],
    }

    try:
        for lbl, kw in grid:
            c0 = time.time()
            rows = []
            for p in harmful:
                formatted = format_prompt(tok, p, "chat")
                resp = gen(model, tok, formatted, args.max_new, kw)
                rows.append({"kind": "harmful", "prompt": p,
                             "response": resp, **classify(resp)})
            for p in benign:
                formatted = format_prompt(tok, p, "chat")
                resp = gen(model, tok, formatted, args.max_new, kw)
                rows.append({"kind": "benign", "prompt": p,
                             "response": resp, **classify(resp)})
            n_ref = sum(r["refusal"] for r in rows if r["kind"] == "harmful")
            n_coh = sum(r["coherent"] for r in rows if r["kind"] == "harmful")
            print(f"\n=== {lbl:>13} | harmful refusal {n_ref}/{len(harmful)} "
                  f"| harmful coherent {n_coh}/{len(harmful)} | "
                  f"({time.time() - c0:.0f}s) ===")
            for r in rows:
                kind = r["kind"]
                tag = ("REFUSAL" if r["refusal"] else "compliant"
                       if kind == "harmful" else "ok")
                print(f"\n[{kind}] {r['prompt']}")
                print(f"  ({tag}, coherent={r['coherent']}) {r['response']!r}")
            doc["configs"].append({"label": lbl, "kwargs": kw, "rows": rows})
    finally:
        del model, tok
        gc.collect()

    doc["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGrid transcript -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
