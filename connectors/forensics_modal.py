"""Modal forensics runner — decode-grid rescue test + huihui A/B + tensor diff.

Heavy-compute counterpart to the local RAM-starved box (6 cores / 15 GB).
Everything here loads models INSIDE Modal (L4 GPU, 24 GB): the local machine
only orchestrates. Two weight sources:

  * OUR ablated models: staged under <repo>/abl-work/ (repo root is mounted
    at /absolver; `campaigns/` is excluded from the mount, `abl-work/` is
    deliberately NOT).
  * huihui's abliterated edit: fetched remotely from HF at runtime
    (huihui-ai/Huihui-LFM2.5-1.2B-Instruct-abliterated) — no upload.

Usage:
    modal run connectors/forensics_modal.py::main --argv '<cmd> <args...>'

Commands:
    decode-grid --model-ref <dir|hf-id> [--labels a,b] [--max-new 64] [--tag T]
                [--out <local-json>]
      The decoding grid (control | rep{1.05,1.1,1.2} | ngram{3,4} | combos |
      2 sampling) x the collect gate's held-out harmful split + benign set,
      chat flavor. Every response is classified with the repo's own refusal /
      coherence / degeneracy helpers. Returns the FULL doc (transcripts) so
      the orchestrator can eyeball every string.

    huihui-ab [--model-id <hf-id>] [--max-new 64] [--out <local-json>]
      Same held-out prompt set on huihui's published abliteration, at
      default greedy + a small decoding grid.

    tensor-diff --a <hf-id> --b <hf-id> [--max-rows 400] [--out <local-json>]
      Per-tensor diff of two model.safetensors (snapshot both remotely),
      grouped by LFM2.5 weight family (attn / conv / ffn / norm / embed+lh).

Every command returns its JSON doc as the Modal function return value; the
local entrypoint writes it to --out (a LOCAL path) and prints a digest.
"""
from __future__ import annotations

import gc
import json
import os
import re
import time
from pathlib import Path

import modal  # only imported under `modal run`

_PROJECT_DIR = Path(__file__).resolve().parent.parent

app = modal.App("absolver-forensics")

_IGNORE = [".venv", ".git", "__pycache__", "*.pyc", ".aiwg",
           "abliterated_models", "campaigns", "experiments"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch>=2.0",
        "transformers>=4.30",
        "numpy>=1.24",
        "pyyaml>=6",
        "huggingface-hub>=0.20",
        "safetensors>=0.4",
        "pydantic>=2",
    )
    .env({"PYTHONPATH": "/absolver"})
    .add_local_dir(  # must be LAST build step
        str(_PROJECT_DIR),
        remote_path="/absolver",
        ignore=_IGNORE,
    )
)

GPU = os.environ.get("ABSOLVER_GPU", "L4")
TIMEOUT = int(os.environ.get("ABSOLVER_TIMEOUT", "3600"))

# The decoding grid: (label, generate kwargs). Mirrors campaigns/decode_grid.py.
GRID: list[tuple[str, dict]] = [
    ("control", {}),
    ("rep1.05", {"repetition_penalty": 1.05}),
    ("rep1.1", {"repetition_penalty": 1.1}),
    ("rep1.2", {"repetition_penalty": 1.2}),
    ("ngram3", {"no_repeat_ngram_size": 3}),
    ("ngram4", {"no_repeat_ngram_size": 4}),
    ("ngram3+rep1.1", {"no_repeat_ngram_size": 3, "repetition_penalty": 1.1}),
    ("ngram4+rep1.1", {"no_repeat_ngram_size": 4, "repetition_penalty": 1.1}),
    ("sample-t0.7", {"do_sample": True, "temperature": 0.7, "top_p": 0.95,
                     "repetition_penalty": 1.1}),
    ("sample-t0.9", {"do_sample": True, "temperature": 0.9, "top_p": 0.95,
                     "repetition_penalty": 1.1}),
]


def _held_out_prompts() -> tuple[list[str], list[str]]:
    """The collect gate's held-out harmful split + small benign set."""
    from eval_split import build_split
    from prompts import DEFAULT_HARMFUL, DEFAULT_HARMLESS
    h = list(DEFAULT_HARMFUL)
    g = list(DEFAULT_HARMLESS)
    n = min(len(h), len(g))
    split = build_split(h[:n], g[:n], train_size=n - 10, tune_size=5,
                        test_size=5, seed="absolver:qwen25:v1")
    return list(split.test), list(DEFAULT_HARMLESS)[:3]


def _classify(resp: str) -> dict:
    from gates import _coherent_completion, _degeneracy_ratio
    from verify import _response_is_refusal
    return {
        "refusal": _response_is_refusal(resp),
        "coherent": _coherent_completion(resp),
        "degeneracy_ratio": round(_degeneracy_ratio(resp), 4),
    }


def _load_model_tok(model_ref: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] loading {model_ref} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_ref, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    if torch.cuda.is_available():
        model = model.to("cuda")
        print(f"  on cuda: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"  loaded in {time.time() - t0:.0f}s (vocab {len(tok)})", flush=True)
    return model, tok


def _gen(model, tok, formatted: str, max_new: int, kwargs: dict, seed: int) -> str:
    import torch
    from verify import _decode_continuation
    if kwargs.get("do_sample"):
        torch.manual_seed(seed)
    inp = tok(formatted, return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inp = {k: v.to("cuda") for k, v in inp.items()}
    with torch.no_grad():
        raw = model.generate(**inp, max_new_tokens=max_new,
                             pad_token_id=tok.eos_token_id, **kwargs)
    return _decode_continuation(tok, raw, inp["input_ids"])


@app.function(image=image, gpu=GPU, timeout=TIMEOUT, retries=0,
              secrets=[modal.Secret.from_name("hf-write-token")])
def run_decode_grid(model_ref: str, labels: str = "", max_new: int = 64,
                    tag: str = "") -> dict:
    model, tok = _load_model_tok(model_ref)
    harmful, benign = _held_out_prompts()
    want = [s.strip() for s in labels.split(",") if s.strip()]
    grid = [(lbl, kw) for lbl, kw in GRID if not want or lbl in want]

    # record the effective flavor per prompt (chat or raw fallback)
    from prompt_format import resolve_flavor
    flavor = resolve_flavor(tok, "chat", "chat")
    print(f"prompt flavor resolved: {flavor}", flush=True)
    print(f"held-out harmful ({len(harmful)}):")
    for p in harmful:
        print(f"  - {p}", flush=True)
    print(f"benign ({len(benign)}):", flush=True)
    for p in benign:
        print(f"  - {p}", flush=True)

    doc = {"model_ref": model_ref, "tag": tag, "max_new_tokens": max_new,
           "flavor": flavor, "harmful_prompts": harmful,
           "benign_prompts": benign,
           "started": time.strftime("%Y-%m-%d %H:%M:%S"),
           "configs": []}
    try:
        for ci, (lbl, kw) in enumerate(grid):
            c0 = time.time()
            rows = []
            for pi, p in enumerate(harmful):
                formatted = _fmt(tok, p)
                resp = _gen(model, tok, formatted, max_new, kw, ci * 100 + pi)
                rows.append({"kind": "harmful", "prompt": p,
                             "response": resp, **(_classify(resp))})
            for pi, p in enumerate(benign):
                formatted = _fmt(tok, p)
                resp = _gen(model, tok, formatted, max_new, kw, ci * 100 + 50 + pi)
                rows.append({"kind": "benign", "prompt": p,
                             "response": resp, **(_classify(resp))})
            n_ref = sum(1 for r in rows if r["kind"] == "harmful" and r["refusal"])
            n_coh = sum(1 for r in rows if r["kind"] == "harmful" and r["coherent"])
            n_clean = sum(1 for r in rows if r["kind"] == "harmful"
                          and not r["refusal"] and r["coherent"]
                          and r["degeneracy_ratio"] < 0.05)
            doc["configs"].append({"label": lbl, "kwargs": kw, "rows": rows,
                                   "n_refusal": n_ref, "n_coherent": n_coh,
                                   "n_clean_compliant": n_clean})
            print(f"[{time.strftime('%H:%M:%S')}] === {lbl:>13} | "
                  f"refusal {n_ref}/{len(harmful)} | coherent {n_coh}/{len(harmful)} "
                  f"| clean {n_clean}/{len(harmful)} | {time.time()-c0:.0f}s ===",
                  flush=True)
            for r in rows:
                kind = r["kind"]
                tag_ = ("REFUSAL" if r["refusal"] else "compliant"
                        if kind == "harmful" else "ok")
                print(f"  [{kind}] ({tag_}, coh={r['coherent']}, "
                      f"deg={r['degeneracy_ratio']}) {r['prompt'][:50]!r}\n"
                      f"      {r['response'][:300]!r}", flush=True)
    finally:
        del model, tok
        gc.collect()
    doc["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return doc


def _fmt(tok, p: str) -> str:
    from prompt_format import format_prompt
    return format_prompt(tok, p, "chat")


@app.function(image=image, gpu=GPU, timeout=TIMEOUT, retries=0,
              secrets=[modal.Secret.from_name("hf-write-token")])
def run_huihui_ab(model_id: str = "huihui-ai/Huihui-LFM2.5-1.2B-Instruct-abliterated",
                  max_new: int = 64) -> dict:
    """Same held-out prompt set vs huihui's published abliteration."""
    model, tok = _load_model_tok(model_id)
    harmful, benign = _held_out_prompts()

    CONFIGS: list[dict] = [
        {"name": "greedy", "do_sample": False, "temperature": None,
         "repetition_penalty": None, "no_repeat_ngram_size": None},
        {"name": "temp0.8", "do_sample": True, "temperature": 0.8,
         "repetition_penalty": None, "no_repeat_ngram_size": None},
        {"name": "rp1.1", "do_sample": True, "temperature": 0.8,
         "repetition_penalty": 1.1, "no_repeat_ngram_size": None},
        {"name": "rp1.2", "do_sample": True, "temperature": 0.8,
         "repetition_penalty": 1.2, "no_repeat_ngram_size": None},
        {"name": "ngram4+rp1.1", "do_sample": True, "temperature": 0.8,
         "repetition_penalty": 1.1, "no_repeat_ngram_size": 4},
    ]
    results: list[dict] = []
    for ci, cfg in enumerate(CONFIGS):
        gen_kw = {"do_sample": cfg["do_sample"]}
        for k in ("temperature", "repetition_penalty", "no_repeat_ngram_size"):
            if cfg[k] is not None:
                gen_kw[k] = cfg[k]
        for pi, p in enumerate(harmful):
            resp = _gen(model, tok, _fmt(tok, p), max_new, gen_kw, ci * 100 + pi)
            results.append({"config": cfg["name"], "kind": "harmful",
                            "prompt": p, "response": resp, **(_classify(resp))})
        for pi, p in enumerate(benign):
            resp = _gen(model, tok, _fmt(tok, p), max_new, gen_kw, ci * 100 + 50 + pi)
            results.append({"config": cfg["name"], "kind": "benign",
                            "prompt": p, "response": resp, **(_classify(resp))})
        n_ref = sum(1 for r in results if r["config"] == cfg["name"]
                    and r["kind"] == "harmful" and r["refusal"])
        n_clean = sum(1 for r in results if r["config"] == cfg["name"]
                      and r["kind"] == "harmful" and not r["refusal"]
                      and r["coherent"] and r["degeneracy_ratio"] < 0.05)
        print(f"[{time.strftime('%H:%M:%S')}] config {cfg['name']:>12} | "
              f"refusal {n_ref}/{len(harmful)} | clean {n_clean}/{len(harmful)}",
              flush=True)
    del model, tok
    gc.collect()
    return {"model_id": model_id, "max_new_tokens": max_new,
            "harmful_prompts": harmful, "benign_prompts": benign,
            "finish": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results}


_GROUP_PATTERNS: list[tuple[str, str]] = [
    ("attn", r"self_attn\."),
    ("conv", r"\.conv\."),
    ("ffn", r"feed_forward\."),
    ("norm", r"norm\."),
    ("embed/lm_head", r"(embed_tokens|lm_head)"),
]


def _group_of(name: str) -> str:
    for group, pat in _GROUP_PATTERNS:
        if re.search(pat, name):
            return group
    return "other"


def _resolve_model_dir(ref: str) -> Path:
    """Resolve a model dir: local path (mounted under /absolver) or HF id."""
    local = Path(ref)
    if local.exists():
        return local
    from huggingface_hub import snapshot_download
    print(f"downloading {ref} ...", flush=True)
    return Path(snapshot_download(ref))


def _safetensors_file(d: Path) -> tuple[Path, list[str]]:
    files = sorted(d.glob("*.safetensors"))
    if not files:
        raise RuntimeError(f"no .safetensors in {d}")
    return files[0], [f.name for f in files]


@app.function(image=image, gpu=GPU, timeout=TIMEOUT, retries=0,
              secrets=[modal.Secret.from_name("hf-write-token")])
def run_tensor_diff(a_id: str, b_id: str, max_rows: int = 400) -> dict:
    """Per-tensor diff of two model.safetensors, grouped by weight family."""
    print(f"resolving {a_id} ...", flush=True)
    a_dir = _resolve_model_dir(a_id)
    print(f"resolving {b_id} ...", flush=True)
    b_dir = _resolve_model_dir(b_id)
    a_file, only_a = _safetensors_file(a_dir)
    b_file, only_b = _safetensors_file(b_dir)
    if a_file.name != b_file.name:
        raise RuntimeError(f"file name mismatch: a={only_a} b={only_b}")

    from safetensors import safe_open
    from collections import defaultdict
    import torch
    rows: list[dict] = []
    groups: dict[str, dict] = defaultdict(lambda: {"n_diff": 0, "n_identical": 0,
                                                   "max_abs": 0.0, "rel_l2": 0.0,
                                                   "touched": []})
    n_common = n_identical = 0
    with safe_open(a_file, framework="pt") as fa, safe_open(b_file, framework="pt") as fb:
        keys_a = list(fa.keys())
        keys_b = list(fb.keys())
        common = sorted(set(keys_a) & set(keys_b))
        only_a = sorted(set(keys_a) - set(keys_b))
        only_b = sorted(set(keys_b) - set(keys_a))
        n_common = len(common)
        for k in common:
            ta = fa.get_tensor(k)
            tb = fb.get_tensor(k)
            g = _group_of(k)
            if ta.shape != tb.shape:
                rows.append({"tensor": k, "group": g, "status": "SHAPE-MISMATCH",
                             "shape_a": list(ta.shape), "shape_b": list(tb.shape)})
                groups[g]["n_diff"] += 1
                groups[g]["touched"].append(k)
                del ta, tb
                continue
            if torch.equal(ta, tb):
                n_identical += 1
                groups[g]["n_identical"] += 1
                rows.append({"tensor": k, "group": g, "status": "identical",
                             "shape": list(ta.shape)})
            else:
                diff = (ta.float() - tb.float()).abs()
                max_abs = float(diff.max().item())
                rel = float(diff.norm().item() /
                            max(1e-12, ta.float().norm().item()))
                n_diff = int((diff > 0).sum().item())
                rows.append({"tensor": k, "group": g, "status": "DIFF",
                             "shape": list(ta.shape), "max_abs": round(max_abs, 6),
                             "rel_l2": round(rel, 6), "n_diff_elems": n_diff})
                groups[g]["n_diff"] += 1
                groups[g]["max_abs"] = max(groups[g]["max_abs"], max_abs)
                groups[g]["rel_l2"] = max(groups[g]["rel_l2"], rel)
                groups[g]["touched"].append(k)
                del diff
            del ta, tb
    group_summary = {g: {k: (v if k != "touched" else len(v))
                         for k, v in d.items()} for g, d in sorted(groups.items())}
    return {
        "a": a_id, "b": b_id,
        "n_keys_a": len(keys_a), "n_keys_b": len(keys_b),
        "common": n_common, "identical": n_identical,
        "differing": n_common - n_identical,
        "only_a": only_a, "only_b": only_b,
        "group_summary": group_summary,
        "rows": rows[:max_rows],
        "rows_truncated": len(rows) > max_rows,
    }


@app.local_entrypoint()
def main(argv: str = ""):
    import argparse

    ap = argparse.ArgumentParser(prog="forensics_modal")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("decode-grid")
    p.add_argument("--model-ref", required=True,
                   help="/absolver/abl-work/<dir> or HF id")
    p.add_argument("--labels", default="")
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--tag", default="")
    p.add_argument("--out", required=True)

    p = sub.add_parser("huihui-ab")
    p.add_argument("--model-id",
                   default="huihui-ai/Huihui-LFM2.5-1.2B-Instruct-abliterated")
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--out", required=True)

    p = sub.add_parser("tensor-diff")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--max-rows", type=int, default=400)
    p.add_argument("--out", required=True)

    args = ap.parse_args(argv.split())
    if args.cmd == "decode-grid":
        doc = run_decode_grid.remote(args.model_ref, args.labels,
                                     args.max_new, args.tag)
    elif args.cmd == "huihui-ab":
        doc = run_huihui_ab.remote(args.model_id, args.max_new)
    else:
        doc = run_tensor_diff.remote(args.a, args.b, args.max_rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWROTE {out} ({out.stat().st_size} bytes)")

    if args.cmd == "decode-grid":
        print("\n== summary ==")
        for cfg in doc["configs"]:
            print(f"{cfg['label']:>14}: refusal {cfg['n_refusal']}/{len(doc['harmful_prompts'])} "
                  f"| coherent {cfg['n_coherent']}/{len(doc['harmful_prompts'])} "
                  f"| clean {cfg['n_clean_compliant']}/{len(doc['harmful_prompts'])}")
    elif args.cmd == "huihui-ab":
        print("\n== summary ==")
        for cfg_name in sorted({r["config"] for r in doc["results"]}):
            rs = [r for r in doc["results"] if r["config"] == cfg_name]
            n_ref = sum(1 for r in rs if r["kind"] == "harmful" and r["refusal"])
            n_clean = sum(1 for r in rs if r["kind"] == "harmful"
                          and not r["refusal"] and r["coherent"]
                          and r["degeneracy_ratio"] < 0.05)
            print(f"{cfg_name:>12}: refusal {n_ref}/{len(doc['harmful_prompts'])} "
                  f"| clean {n_clean}/{len(doc['harmful_prompts'])}")
    else:
        print(f"\n== diff {doc['a']} vs {doc['b']} ==")
        print(f"keys a={doc['n_keys_a']} b={doc['n_keys_b']} common={doc['common']} "
              f"identical={doc['identical']} differing={doc['differing']}")
        for g, s in doc["group_summary"].items():
            print(f"  {g:<12}: diff={s['n_diff']:>3} identical={s['n_identical']:>3} "
                  f"max_abs={s['max_abs']:<9} rel_l2={s['rel_l2']}")
        for r in doc["rows"]:
            if r["status"] == "DIFF":
                print(f"  DIFF {r['group']:<12} {r['tensor'][:60]:<62} "
                      f"max_abs={r['max_abs']:<9} rel_l2={r['rel_l2']:<9} "
                      f"n={r['n_diff_elems']}")