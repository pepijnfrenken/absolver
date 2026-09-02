#!/usr/bin/env python3
"""Per-tensor diff between LFM2.5 safetensors files (OOM-safe).

Loads ONE tensor pair at a time (mmap-backed), compares bit-exact and
numerically, frees immediately. Never holds more than two tensors in RAM.

Usage:
  .venv/bin/python campaigns/tensor_diff.py --a <file.safetensors> --b <file.safetensors> [--out diff.json] [--max-rows 60]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

PROJECT_DIR = Path(__file__).resolve().parent.parent


def diff_files(a_path: str, b_path: str, max_rows: int) -> dict:
    with safe_open(a_path, framework="pt") as fa, safe_open(b_path, framework="pt") as fb:
        keys_a = list(fa.keys())
        keys_b = list(fb.keys())
        common = sorted(set(keys_a) & set(keys_b))
        only_a = sorted(set(keys_a) - set(keys_b))
        only_b = sorted(set(keys_b) - set(keys_a))
        rows: list[dict] = []
        touched = 0
        i = 0
        for k in common:
            i += 1
            print(f"  [{i}/{len(common)}] {k}", flush=True)
            ta = fa.get_tensor(k)
            tb = fb.get_tensor(k)
            if ta.shape != tb.shape:
                rows.append({"tensor": k, "status": "SHAPE-MISMATCH",
                             "shape_a": list(ta.shape), "shape_b": list(tb.shape)})
                touched += 1
                continue
            exact = torch.equal(ta, tb)
            if exact:
                rows.append({"tensor": k, "status": "identical",
                             "shape": list(ta.shape)})
            else:
                diff = (ta.float() - tb.float()).abs()
                max_abs = diff.max().item()
                rel = diff.norm().item() / max(1e-12, ta.float().norm().item())
                n_diff = (diff > 0).sum().item()
                rows.append({"tensor": k, "status": "DIFF",
                             "shape": list(ta.shape),
                             "max_abs": round(float(max_abs), 6),
                             "rel_l2": round(float(rel), 6),
                             "n_diff_elems": int(n_diff)})
                touched += 1
            del ta, tb
            try:
                del diff
            except UnboundLocalError:
                pass
        return {
            "a": a_path, "b": b_path,
            "n_keys_a": len(keys_a), "n_keys_b": len(keys_b),
            "common": len(common), "only_a": only_a, "only_b": only_b,
            "n_touched": touched,
            "rows": rows[:max_rows],
            "rows_truncated": len(rows) > max_rows,
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-rows", type=int, default=200)
    args = ap.parse_args()
    doc = diff_files(args.a, args.b, args.max_rows)
    print(f"DIFF {doc['a']}  vs  {doc['b']}")
    print(f"  keys a={doc['n_keys_a']} b={doc['n_keys_b']} "
          f"common={doc['common']} only_a={doc['only_a']} only_b={doc['only_b']}")
    print(f"  tensors differing: {doc['n_touched']}")
    for r in doc["rows"]:
        if r["status"] == "DIFF":
            print(f"  DIFF  {r['tensor']:<72} max_abs={r['max_abs']:<10} "
                  f"rel_l2={r['rel_l2']:<10} n_elems={r['n_diff_elems']}")
    if args.out:
        Path(args.out).write_text(json.dumps(doc, indent=2), encoding="utf-8")
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())