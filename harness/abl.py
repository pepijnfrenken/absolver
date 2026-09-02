#!/usr/bin/env python3
"""Absolver harness — the guided, inspect-first way to run ablations.

NOT an autonomous loop. Each command does ONE thing, prints what it did,
and (for collect) writes a data bundle the campaign README references.
An agent (or human) reads the output, consults the campaign KB, decides
the next single step, and runs it.

Commands:
  inspect <config.yaml>            Load model, print arch + layer profile +
                                   direction separation + bias/weight shape
                                   audit (catches silent-skip landmines).
  directions <config.yaml>         Collect + save per-layer directions to
                                   <campaign>/directions.pt (with metadata).
  abl <config.yaml> --method mpoa --alpha 10 --layers 24,25,26,27 \
      --weights o_proj,down_proj   Apply ONE config to a fresh model,
                                   save ablated weights + a diff manifest.
  collect <config.yaml>            Run gates + behavior + capability map on
                                   the ablated model, write a JSON bundle to
                                   <campaign>/<run-id>/bundle.json.
  list-campaigns                   List campaigns + statuses from YAML.

The output dir convention: campaigns/<model-slug>/<run-id>/ where run-id
is a short timestamp + config tag. The campaign README is the narrative;
these bundles are the raw evidence.

Usage examples:
  python harness/abl.py inspect models/qwen2.5-1.5b-instruct.yaml
  python harness/abl.py directions models/qwen2.5-1.5b-instruct.yaml
  python harness/abl.py abl models/qwen2.5-1.5b-instruct.yaml \
      --method mpoa --alpha 10 --layers 24-27 --weights o_proj,down_proj
  python harness/abl.py collect models/qwen2.5-1.5b-instruct.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _slug(model_id: str) -> str:
    return model_id.replace("/", "-").lower()


def _load_cfg(config_path: str):
    from config import load_config
    return load_config(config_path)


def _load_model_tok(cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = getattr(torch, str(cfg.dtype)) if isinstance(cfg.dtype, str) else cfg.dtype
    trust = getattr(cfg, "trust_remote_code", False)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, torch_dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=trust)
    tok = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=trust)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return model, tok


def _parse_layers(spec: str):
    """Accept '24,25,26,27' or '24-27' or '24'."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


# --------------------------------------------------------------------------- #
# inspect
# --------------------------------------------------------------------------- #

def cmd_inspect(config_path: str) -> int:
    cfg = _load_cfg(config_path)
    model, tok = _load_model_tok(cfg)
    from probe import _find_layers, _make_hook, _to_device
    from distill import extract_directions
    from prompts import DEFAULT_HARMFUL, DEFAULT_HARMLESS

    layers = _find_layers(model, cfg.model_arch)
    num_layers = len(layers)
    hidden = model.config.hidden_size
    dev = "cpu" if not hasattr(model, "device") or str(model.device) == "cpu" else str(model.device)
    print(f"=== {cfg.model_id} ===")
    print(f"arch={cfg.model_arch} layers={num_layers} hidden={hidden}")

    # Bias / weight-shape audit — catches silent-skip landmines
    print("\n--- weight audit (silent-skip landmines) ---")
    for li in [0, num_layers // 2, num_layers - 1]:
        layer = layers[li]
        attn = getattr(layer, "self_attn", None)
        mlp = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
        for wname, mod in [("o_proj", getattr(attn, "o_proj", None)),
                           ("down_proj", getattr(mlp, "down_proj", None))]:
            if mod is None:
                print(f"  L{li} {wname}: MISSING")
                continue
            w = mod.weight
            bias = getattr(mod, "bias", None)
            print(f"  L{li} {wname}: W{tuple(w.shape)} bias={'yes' if bias is not None else 'NO'}")
            if w.dim() == 2 and w.shape[0] != w.shape[1]:
                print(f"      ^ NON-SQUARE: output dim {w.shape[0]} != input {w.shape[1]}")
                print(f"        hidden-space directions ({hidden}) only fit output-dim==hidden weights")

    # Quick direction separation profile (10 prompts each side)
    print("\n--- direction separation (10+10 prompts, diff_means) ---")
    import torch
    from collections import defaultdict
    def collect(prompts):
        store = defaultdict(list)
        handles = [layers[i].register_forward_hook(_make_hook(i, store))
                   for i in range(num_layers)]
        try:
            for p in prompts:
                inp = tok(p, return_tensors="pt", truncation=True, max_length=128)
                with torch.no_grad():
                    model(**inp)
        finally:
            for hh in handles:
                try: hh.remove()
                except Exception: pass
        return dict(store)
    h = list(DEFAULT_HARMFUL)[:10]
    g = list(DEFAULT_HARMLESS)[:10]
    acts_h = collect(h)
    acts_g = collect(g)
    dirs, scores = extract_directions(acts_h, acts_g, num_layers, hidden, "diff_means", 3, dev)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    print(f"  top-8 layers by separation: {[(li, round(s, 2)) for li, s in ranked[:8]]}")
    print(f"  tail-4 (last) layers:       {[(li, round(s, 2)) for li, s in ranked if li >= num_layers - 4]}")
    return 0


# --------------------------------------------------------------------------- #
# directions
# --------------------------------------------------------------------------- #

def cmd_directions(config_path: str) -> int:
    cfg = _load_cfg(config_path)
    model, tok = _load_model_tok(cfg)
    from probe import _find_layers, _make_hook, _to_device
    from distill import extract_directions
    from prompts import DEFAULT_HARMFUL, DEFAULT_HARMLESS

    layers = _find_layers(model, cfg.model_arch)
    num_layers = len(layers)
    hidden = model.config.hidden_size
    dev = "cpu"

    from collections import defaultdict
    def collect(prompts):
        store = defaultdict(list)
        handles = [layers[i].register_forward_hook(_make_hook(i, store))
                   for i in range(num_layers)]
        try:
            for p in prompts:
                inp = tok(p, return_tensors="pt", truncation=True, max_length=128)
                with torch.no_grad():
                    model(**inp)
        finally:
            for hh in handles:
                try: hh.remove()
                except Exception: pass
        return dict(store)

    n = getattr(cfg, "n_probe_prompts", 20)
    h = list(DEFAULT_HARMFUL)[:n]
    g = list(DEFAULT_HARMLESS)[:n]
    # direction method from config
    dm = getattr(cfg, "dir_method", "diff_means")
    acts_h = collect(h)
    acts_g = collect(g)
    dirs, scores = extract_directions(acts_h, acts_g, num_layers, hidden, dm, 3, dev)

    out_dir = PROJECT_DIR / "campaigns" / _slug(cfg.model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "directions.pt"
    import torch
    torch.save({"dirs": {str(k): v.cpu() for k, v in dirs.items()},
                "scores": {str(k): v for k, v in scores.items()},
                "model_id": cfg.model_id, "dir_method": dm,
                "n_prompts": len(h), "hidden": hidden}, path)
    print(f"Saved {len(dirs)} layer directions -> {path}")
    print(f"Top-5 layers: {sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:5]}")
    return 0


# --------------------------------------------------------------------------- #
# abl — apply ONE config to a fresh model
# --------------------------------------------------------------------------- #

def cmd_abl(config_path: str, method: str, alpha: float, layers_spec: str,
            weights_spec: str, dir_method: str | None, out_tag: str | None) -> int:
    cfg = _load_cfg(config_path)
    model, tok = _load_model_tok(cfg)
    import torch
    from probe import _find_layers, _make_hook, _to_device
    from distill import extract_directions
    from sweep import _apply_candidate, _apply_mpoa, _apply_stacked, _apply_bias_vectors, _apply_advanced
    from prompts import DEFAULT_HARMFUL, DEFAULT_HARMLESS
    from excise import _project_2d, _project_2d_mpoa

    layers = _find_layers(model, cfg.model_arch)
    num_layers = len(layers)
    hidden = model.config.hidden_size
    target_layers = _parse_layers(layers_spec)
    weights = [w.strip() for w in weights_spec.split(",")]
    dm = dir_method or getattr(cfg, "dir_method", "diff_means")

    # Collect directions fresh (never reuse stale ones)
    from collections import defaultdict
    def collect(prompts):
        store = defaultdict(list)
        handles = [layers[i].register_forward_hook(_make_hook(i, store))
                   for i in range(num_layers)]
        try:
            for p in prompts:
                inp = tok(p, return_tensors="pt", truncation=True, max_length=128)
                with torch.no_grad():
                    model(**inp)
        finally:
            for hh in handles:
                try: hh.remove()
                except Exception: pass
        return dict(store)

    n = getattr(cfg, "n_probe_prompts", 20)
    h = list(DEFAULT_HARMFUL)[:n]
    g = list(DEFAULT_HARMLESS)[:n]
    acts_h = collect(h)
    acts_g = collect(g)
    dirs, scores = extract_directions(acts_h, acts_g, num_layers, hidden, dm, 3, "cpu")

    candidate = {"method": method, "dir_method": dm,
                 "target_layers": target_layers, "target_weights": weights,
                 "alpha": alpha, "passes": 1}
    _apply_candidate(model, dirs, None, candidate)
    print(f"Applied {method} alpha={alpha} layers={target_layers} weights={weights} dir={dm}")

    # save ablated model
    out_dir = PROJECT_DIR / "campaigns" / _slug(cfg.model_id) / (out_tag or f"abl-{method}-a{alpha}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    manifest = {"model_id": cfg.model_id, "method": method, "dir_method": dm,
                "alpha": alpha, "layers": target_layers, "weights": weights,
                "config": config_path, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved ablated model + manifest -> {out_dir}")
    return 0


# --------------------------------------------------------------------------- #
# collect — run gates + behavior + capability, write a JSON bundle
# --------------------------------------------------------------------------- #

def cmd_collect(config_path: str, model_dir: str | None) -> int:
    cfg = _load_cfg(config_path)
    # load the ablated model if given, else fresh
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = model_dir or cfg.model_id
    if model_dir:
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        tok = AutoTokenizer.from_pretrained(model_dir)
    else:
        model, tok = _load_model_tok(cfg)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    from eval_split import build_split
    from prompts import DEFAULT_HARMFUL, DEFAULT_HARMLESS
    from gates import run_gates
    from verify import run_mmlu_mini, _digest

    # held-out split
    h = list(DEFAULT_HARMFUL); g = list(DEFAULT_HARMLESS)
    n = min(len(h), len(g))
    n_test = 5
    n_train = n - 2 * n_test
    split = build_split(h[:n], g[:n], train_size=n_train, tune_size=n_test,
                        test_size=n_test, seed=cfg.eval_split_seed)
    held_out = list(split.test)

    # capability
    benchmark_scores = {}
    try:
        benchmark_scores["mmlu"] = run_mmlu_mini(model, tok, n=20)
    except Exception as exc:
        print("mmlu_mini failed:", exc)

    report = run_gates(model, tok, cfg, prompts=held_out, benchmark_scores=benchmark_scores)
    out = {"model_id": cfg.model_id if not model_dir else model_id,
           "eval_dir": model_dir,
           "held_out_size": len(held_out), "benchmark_scores": benchmark_scores,
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    for k, v in report.items():
        if k in ("_enabled", "eval_pass", "held_out_size"):
            continue
        out[k] = {"passed": v["passed"], "value": v.get("value"), "detail": v["detail"]}
    out["eval_pass"] = report.get("eval_pass")

    out_dir = PROJECT_DIR / "campaigns" / _slug(cfg.model_id) / "latest-collect"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "bundle.json"
    (out_dir / "bundle.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))
    print(f"\nBundle -> {path}")
    return 0


# --------------------------------------------------------------------------- #
# list-campaigns
# --------------------------------------------------------------------------- #

def cmd_list_campaigns() -> int:
    import yaml
    base = PROJECT_DIR / "campaigns"
    if not base.exists():
        print("no campaigns dir")
        return 0
    print(f"{'Campaign':<38} {'Model':<38} {'Date':<12} {'Status'}")
    print("-" * 110)
    for d in sorted(base.iterdir()):
        if not d.is_dir() or d.name == "templates":
            continue
        readme = d / "README.md"
        if not readme.exists():
            continue
        text = readme.read_text(encoding="utf-8")
        # cheap YAML frontmatter parse
        m = re.match(r"^---\n(.*?)\n---", text, re.S)
        meta = {}
        if m:
            try:
                meta = yaml.safe_load(m.group(1)) or {}
            except Exception:
                pass
        model = meta.get("target_model", "?")
        date = str(meta.get("date", "?"))
        status = str(meta.get("status", "?"))
        print(f"{d.name:<38} {model:<38} {date:<12} {status}")
    return 0


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="Absolver harness (guided, inspect-first)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect", help="load model, print arch + landmine audit + separation profile")
    p.add_argument("config", help="path to models/<x>.yaml")
    p.set_defaults(fn=lambda a: cmd_inspect(a.config))

    p = sub.add_parser("directions", help="collect + save per-layer directions")
    p.add_argument("config")
    p.set_defaults(fn=lambda a: cmd_directions(a.config))

    p = sub.add_parser("abl", help="apply ONE config to a fresh model")
    p.add_argument("config")
    p.add_argument("--method", required=True, help="advanced|mpoa|stacked_ablation|bias_vectors|direct_ablation|steering|lora|projected")
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--layers", required=True, help="e.g. 24,25,26,27 or 24-27")
    p.add_argument("--weights", default="o_proj,down_proj", help="comma list")
    p.add_argument("--dir-method", default=None, help="diff_means|paired|svd|leace")
    p.add_argument("--tag", default=None, help="output subdir tag")
    p.set_defaults(fn=lambda a: cmd_abl(a.config, a.method, a.alpha, a.layers, a.weights, a.dir_method, a.tag))

    p = sub.add_parser("collect", help="run gates + capability on an ablated model dir (or pristine)")
    p.add_argument("config")
    p.add_argument("--model-dir", default=None, help="path to ablated model (default: pristine from config)")
    p.set_defaults(fn=lambda a: cmd_collect(a.config, a.model_dir))

    p = sub.add_parser("list-campaigns", help="list campaign statuses")
    p.set_defaults(fn=lambda a: cmd_list_campaigns())

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
