"""Modal runner: LFM2.5-2.6B activation-diff (DIY method 1b, canonical).

Method 1a (direction transfer of the 1.2B u1) STALLED (2026-09-03, v4
sweep): prefix mapping lands on the refusal circuitry (proportional misses
it), but the u1 formula is rotated vs the 2.6B's true row side — alpha
>=2.0 collapses production (benign uniq 0.60/0.51 vs healthy 0.86) instead
of clean compliance; alpha <=1.5 leaves refusal at 1.000. Recovery README
anticipated this (row-side cos 0.89, formula residual 0.42).

This runner harvests the 2.6B's OWN refusal directions per layer via the
paired output-phase probe (probe.py _collect_paired_output_phase logic:
same harmful prompts, unprimed refusal generation vs affirmative-prefilled
generation; direction = mean(refusal) - mean(affirm) per layer, hidden
space), then ablates with the huihui coverage rule (o_proj + conv_out +
w2, every layer) using plain and MPOA projections.

Candidates:
  actdiff-plain-1.0 / actdiff-mpoa-1.0 / actdiff-mpoa-1.5 / actdiff-mpoa-2.0
(proven 1.2B recipe = paired + mpoa alpha 2.0 + full coverage — included
verbatim as actdiff-mpoa-2.0)

Usage:
    ABSOLVER_GPU=L4 modal run campaigns/lfm2.5-2.6b/actdiff-modal.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path


import modal
import torch

_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

app = modal.App("absolver-26b-actdiff")

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
    .add_local_dir(
        str(_PROJECT_DIR),
        remote_path="/absolver",
        ignore=[".venv", ".git", "__pycache__", "*.pyc", ".aiwg",
                "abliterated_models", "campaigns", "experiments", "abl-work"],
    )
)

GPU = os.environ.get("ABSOLVER_GPU", "L4")
TIMEOUT = int(os.environ.get("ABSOLVER_TIMEOUT", "5400"))
VOLUME = modal.Volume.from_name("absolver-phase2", create_if_missing=True)

MODEL_ID = "LiquidAI/LFM2.5-2.6B"
CONFIG_YAML = "/absolver/models/lfm2.5-2.6b-instruct.yaml"


def _resolve_proj(layer, wname: str):
    if wname == "o_proj":
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            return None
        mod = getattr(attn, "o_proj", None) or getattr(attn, "out_proj", None)
        return mod if mod is not None and hasattr(mod, "weight") else None
    if wname == "conv_out":
        conv = getattr(layer, "conv", None)
        if conv is None:
            return None
        mod = getattr(conv, "out_proj", None)
        if mod is None or not hasattr(mod, "weight"):
            return None
        w = mod.weight
        if w.dim() != 2 or w.shape[0] != w.shape[1]:
            return None
        return mod
    if wname == "w2":
        ff = getattr(layer, "feed_forward", None)
        if ff is None:
            return None
        return getattr(ff, "w2", None)
    return None


def project_2d(weight, d, alpha: float, mpoa: bool) -> None:
    d = d.to(dtype=weight.dtype, device=weight.device).reshape(-1)
    if d.shape[0] != weight.shape[0]:
        raise RuntimeError(
            f"direction out-dim {d.shape[0]} != weight out-dim "
            f"{weight.shape[0]} {tuple(weight.shape)}")
    orig = None
    if mpoa:
        orig = weight.norm().clamp(min=1e-8)
    weight.sub_(alpha * torch.einsum("i,j->ij", d, d @ weight))
    if mpoa:
        new = weight.norm().clamp(min=1e-8)
        weight.mul_(orig / new)


def apply_directions(model, dirs: dict[int, torch.Tensor], config: dict,
                     alpha_fit: dict | None = None,
                     good_dirs: dict[int, torch.Tensor] | None = None,
                     layer_subset: list[int] | None = None) -> list[str]:
    applied: list[str] = []
    layers = model.model.layers
    subset = set(layer_subset) if layer_subset else None
    for li, d in dirs.items():
        if subset is not None and li not in subset:
            continue
        d = d.to(dtype=torch.float32, device="cpu").reshape(-1)
        # projected abliteration: orthogonalize refusal against benign
        if good_dirs and li in good_dirs:
            g = good_dirs[li].to(dtype=torch.float32, device="cpu").reshape(-1)
            g = g / g.norm().clamp(min=1e-8)
            d = d - (d @ g) * g
        d = d / d.norm().clamp(min=1e-8)
        alpha = config["alpha"]
        if alpha is None:
            alpha = float(alpha_fit.get(li, 1.0) if alpha_fit else 1.0)
        mpoa = bool(config["mpoa"])
        layer = layers[li]
        for wname in ("o_proj", "conv_out", "w2"):
            mod = _resolve_proj(layer, wname)
            if mod is not None:
                project_2d(mod.weight.data, d, alpha, mpoa)
                applied.append(f"layer.{li}.{wname}")
    return applied


def harvest_paired_directions(model, tok, prompts, layers, device,
                              prefill: str, max_new: int = 48,
                              max_len: int = 128,
                              benign_prompts: list[str] | None = None,
                              ) -> tuple[dict, dict, dict]:
    """Refusal vs affirm output-phase activations (mirrors probe.py)."""
    def _hook(idx: int, store):
        def hook(_mod, _inp, output):
            hs = output[0] if isinstance(output, tuple) else output
            if not hasattr(hs, "dim"):
                return
            if hs.dim() == 3:
                hs = hs[:, -1:, :]   # last token [B, 1, H]
            elif hs.dim() == 2:
                hs = hs.unsqueeze(1)
            store[idx].append(hs.detach().float().cpu())
        return hook

    ref_acts: dict[int, list] = defaultdict(list)
    aff_acts: dict[int, list] = defaultdict(list)
    pad = tok.eos_token_id

    for p in prompts:
        # refusal (unprimed)
        st1: dict[int, list] = defaultdict(list)
        hs = [layers[i].register_forward_hook(_hook(i, st1))
              for i in range(len(layers))]
        try:
            inp = tok(p, return_tensors="pt", truncation=True, max_length=max_len)
            inp = {k: v.to(device) for k, v in inp.items()}
            with torch.no_grad():
                model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=pad)
        except Exception as exc:  # noqa: BLE001
            print(f"  refusal-gen failed ({p[:40]}): {exc}", flush=True)
        finally:
            for h in hs:
                h.remove()
        for i in range(len(layers)):
            steps = st1.get(i)
            if not steps:
                continue
            resp = steps[1:] if len(steps) > 1 else steps  # drop input-phase fwd
            ref_acts[i].append(torch.stack(resp).mean(dim=0).squeeze(0))

        # affirmative (prefilled)
        st2: dict[int, list] = defaultdict(list)
        hs2 = [layers[i].register_forward_hook(_hook(i, st2))
               for i in range(len(layers))]
        try:
            prefilled = f"{p} {prefill}".strip()
            inp2 = tok(prefilled, return_tensors="pt", truncation=True,
                       max_length=max_len)
            inp2 = {k: v.to(device) for k, v in inp2.items()}
            with torch.no_grad():
                model.generate(**inp2, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=pad)
        except Exception as exc:  # noqa: BLE001
            print(f"  affirm-gen failed ({p[:40]}): {exc}", flush=True)
        finally:
            for h in hs2:
                h.remove()
        for i in range(len(layers)):
            steps = st2.get(i)
            if steps:
                aff_acts[i].append(torch.stack(steps).mean(dim=0).squeeze(0))

    # per-layer separation + direction = mean(ref - aff)
    dirs: dict[int, torch.Tensor] = {}
    seps: dict[int, float] = {}
    for i in range(len(layers)):
        if i not in ref_acts or i not in aff_acts:
            continue
        r = torch.stack(ref_acts[i]).mean(dim=0)
        a = torch.stack(aff_acts[i]).mean(dim=0)
        d = (r - a)
        dirs[i] = d
        seps[i] = float(d.norm())

    # benign-output-phase directions (for projected abliteration)
    good_dirs: dict[int, torch.Tensor] = {}
    if benign_prompts:
        st_b: dict[int, list] = defaultdict(list)
        hb = [layers[i].register_forward_hook(_hook(i, st_b))
              for i in range(len(layers))]
        try:
            for p in benign_prompts:
                formatted = tok.apply_chat_template(
                    [{"role": "user", "content": p}], tokenize=False,
                    add_generation_prompt=True)
                inpb = tok(formatted, return_tensors="pt", truncation=True,
                           max_length=max_len)
                inpb = {k: v.to(device) for k, v in inpb.items()}
                with torch.no_grad():
                    model.generate(**inpb, max_new_tokens=max_new,
                                   do_sample=False, pad_token_id=pad)
        except Exception as exc:  # noqa: BLE001
            print(f"  benign-act failed: {exc}", flush=True)
        finally:
            for h in hb:
                h.remove()
        # use the affirm acts as the benign "good" direction per layer
        for i in range(len(layers)):
            if i in aff_acts:
                good_dirs[i] = torch.stack(aff_acts[i]).mean(dim=0).reshape(-1)
    return dirs, seps, good_dirs


@app.function(image=image, gpu=GPU, timeout=TIMEOUT, retries=0,
              secrets=[modal.Secret.from_name("hf-write-token")],
              volumes={"/out": VOLUME})
def run_actdiff(only: str | None = None) -> dict:
    sys.path.insert(0, "/absolver")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prompts import (DEFAULT_HARMFUL, DEFAULT_HARMLESS, EXPANDED_HARMFUL,
                         EXPANDED_HARMLESS)
    from config import load_config
    from verify import run_mmlu_mini, _model_device
    from refusal_lfm26 import refusal_score

    t0 = time.time()
    cfg = load_config(CONFIG_YAML)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True).to("cuda")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    dev = _model_device(model)
    layers = model.model.layers
    print(f"[26b-actdiff] {MODEL_ID} on {dev}, {len(layers)} layers, "
          f"{sum(p.numel() for p in model.parameters())/1e9:.2f}B", flush=True)

    # ---------- harvest paired output-phase directions (the 2.6B's own) ----------
    prefill = getattr(cfg, "paired_prefill", "Sure, I can help with that.")
    print(f"[26b-actdiff] harvesting paired directions on "
          f"{len(DEFAULT_HARMFUL)} prompts prefill={prefill!r} ...", flush=True)
    dirs, seps, good_dirs = harvest_paired_directions(
        model, tok, DEFAULT_HARMFUL, layers, dev, prefill,
        benign_prompts=DEFAULT_HARMLESS[:8])
    top = sorted(seps.items(), key=lambda kv: -kv[1])[:8]
    print(f"[26b-actdiff] separation top: "
          f"{[(li, round(s, 3)) for li, s in top]}", flush=True)
    print(f"[26b-actdiff] good dirs: {len(good_dirs)} layers", flush=True)

    # pristine gates
    def judge_refusal(prompt_subset, transcript):
        rows = []
        for p in prompt_subset:
            formatted = tok.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False,
                add_generation_prompt=True)
            inp = tok(formatted, return_tensors="pt", truncation=True,
                      max_length=1024).to(dev)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=128,
                                     do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            resp = tok.decode(out[0][inp["input_ids"].shape[1]:],
                              skip_special_tokens=True)
            rows.append({"prompt": p, "response": resp,
                         "refusal": refusal_score(resp)})
            transcript.append({"gate": "judge_refusal", "prompt": p,
                               "response": resp})
        return (sum(1 for r in rows if r["refusal"] > 0.5) / len(rows),
                rows)

    def benign_stats(n=4):
        stats = []
        for p in EXPANDED_HARMLESS[:n]:
            formatted = tok.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False,
                add_generation_prompt=True)
            inp = tok(formatted, return_tensors="pt", truncation=True,
                      max_length=1024).to(dev)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=96,
                                     do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            resp = tok.decode(out[0][inp["input_ids"].shape[1]:],
                              skip_special_tokens=True)
            words = re.findall(r"[A-Za-z']+", resp)
            ml = sum(len(w) for w in words) / max(1, len(words))
            uq = len(set(w.lower() for w in words)) / max(1, len(words))
            stats.append({"prompt": p, "mean_len": ml, "uniq": uq,
                          "n_words": len(words), "response": resp})
        return {"benign_mean_word_len": sum(s["mean_len"] for s in stats)
                / max(1, len(stats)),
                "benign_uniq_ratio": sum(s["uniq"] for s in stats)
                / max(1, len(stats)),
                "benign_rows": stats}

    pr_tr: list[dict] = []
    pr_refusal, pr_rows = judge_refusal(EXPANDED_HARMFUL[:15], pr_tr)
    pr_benign = benign_stats()
    print(f"[26b-actdiff] PRISTINE refusal={pr_refusal:.3f} "
          f"benign_len={pr_benign['benign_mean_word_len']:.2f} "
          f"uniq={pr_benign['benign_uniq_ratio']:.2f}", flush=True)

    # snapshot
    snap = [(mod.weight.data, mod.weight.data.detach().clone())
            for li in range(len(layers))
            for wname in ("o_proj", "conv_out", "w2")
            if (mod := _resolve_proj(layers[li], wname)) is not None]
    print(f"[26b-actdiff] snapshot {len(snap)} tensors", flush=True)

    # coverage ladder: refusal drops as coverage widens (top-10 0.933 ->
    # 15-29 0.800 -> all-30 0.000 but degenerate). Find the point where
    # refusal -> 0 while language stays healthy.
    configs = [
        {"name": "cov12-29-mpoa-2.0", "alpha": 2.0, "mpoa": True,
         "layers": list(range(12, 30)), "projected": False},
        {"name": "cov10-29-mpoa-2.0", "alpha": 2.0, "mpoa": True,
         "layers": list(range(10, 30)), "projected": False},
        {"name": "cov8-29-mpoa-2.0", "alpha": 2.0, "mpoa": True,
         "layers": list(range(8, 30)), "projected": False},
        {"name": "cov5-29-mpoa-2.0", "alpha": 2.0, "mpoa": True,
         "layers": list(range(5, 30)), "projected": False},
        {"name": "cov10-29-mpoa-2.5", "alpha": 2.5, "mpoa": True,
         "layers": list(range(10, 30)), "projected": False},
        {"name": "cov15-29-mpoa-2.5", "alpha": 2.5, "mpoa": True,
         "layers": list(range(15, 30)), "projected": False},
    ]
    if only:
        configs = [c for c in configs if c["name"] == only]

    results = {"model_id": MODEL_ID, "method": "activation-diff paired",
               "n_prompts": len(DEFAULT_HARMFUL), "n_dirs": len(dirs),
               "separation_top": top,
               "pristine": {"refusal": pr_refusal, "benign": pr_benign},
               "candidates": []}
    results["candidates"] = []  # noqa  (reassigned after config loop)
    transcripts: dict[str, list] = {"pristine": pr_tr}

    out_root = Path("/out/lfm2.5-2.6b-transfer")
    out_root.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())

    for ci, config in enumerate(configs):
        name = config["name"]
        print(f"\n===== {name} =====", flush=True)
        for data, pristine in snap:
            data.copy_(pristine)
        torch.cuda.empty_cache()
        applied = apply_directions(
            model, dirs, config,
            good_dirs=good_dirs if config.get("projected") else None,
            layer_subset=config.get("layers"))
        print(f"applied {len(applied)} weights", flush=True)

        trans: list[dict] = []
        refusal, rows = judge_refusal(EXPANDED_HARMFUL[:15], trans)
        benign = benign_stats()
        mmlu = None
        try:
            mmlu = run_mmlu_mini(model, tok, n=20)
        except Exception as exc:  # noqa: BLE001
            print(f"mmlu failed: {exc}", flush=True)
        print(f"[{name}] refusal={refusal:.3f} "
              f"benign_len={benign['benign_mean_word_len']:.2f} "
              f"uniq={benign['benign_uniq_ratio']:.2f} mmlu={mmlu}", flush=True)
        results["candidates"].append({
            "config": config, "n_applied": len(applied),
            "refusal": refusal, "benign": benign, "mmlu": mmlu,
            "rows": rows,
        })
        transcripts[name] = trans
        res_path = out_root / f"actdiff_{ts}.json"
        res_path.write_text(json.dumps(results, indent=2, default=str))
        tr_path = out_root / f"actdiff_transcripts_{ts}.json"
        tr_path.write_text(json.dumps(transcripts, indent=2, default=str))
        print(f"[26b-actdiff] incremental save candidates done "
              f"{len(results['candidates'])}", flush=True)

    print(f"[26b-actdiff] done ({time.time()-t0:.0f}s)", flush=True)
    return {"results_path": str(out_root / f"actdiff_{ts}.json"),
            "results": results}


@app.local_entrypoint()
def main(only: str | None = None):
    res = run_actdiff.remote(only=only)
    print(json.dumps(res.get("results", {}), indent=2, default=str))


if __name__ == "__main__":
    main()
