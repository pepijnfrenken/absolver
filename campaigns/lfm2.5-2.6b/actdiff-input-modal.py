"""Modal runner: LFM2.5-2.6B activation-diff, INPUT phase (valid 1b).

Method 1a (u1 transfer) STALLED (v3/v4). Method 1b first attempt
(paired OUTPUT-phase) was INVALIDATED (2026-09-03, prefill-check): the
affirmative prefill "Sure, I can help with that." does NOT flip the 2.6B
into compliance — it refuses anyway ("I'm sorry, but I cannot assist...").
So ref - affirm captured refusal-minus-confused-junk, and all actdiff
runs 1-5 tested contaminated directions. Their cliff behavior is not
evidence about the method.

This runner does the canonical input-phase activation diff:
  - chat-formatted harmful prompts  (user turn + generation prompt)
  - chat-formatted benign prompts
  - per-layer last-token hidden states (layer output hook)
  - direction = mean(harm) - mean(benign), per layer, normalized
  - project o_proj + conv_out + w2 (row space), plain & MPOA, alpha sweep

Candidates: all-layers and top-layers (separation peak) x alpha 1.0/1.5/2.0.
Gate: style classifier refusal on EXPANDED_HARMFUL[:15] + benign coherence
(mean word len/uniq vs pristine 5.19/0.86).

Usage:
    ABSOLVER_GPU=L4 modal run campaigns/lfm2.5-2.6b/actdiff-input-modal.py
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

app = modal.App("absolver-26b-actdiff-input")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch>=2.0", "transformers>=4.30", "numpy>=1.24", "pyyaml>=6",
        "huggingface-hub>=0.20", "safetensors>=0.4", "pydantic>=2",
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
                     layer_subset: list[int] | None = None) -> list[str]:
    applied: list[str] = []
    layers = model.model.layers
    subset = set(layer_subset) if layer_subset else None
    for li, d in dirs.items():
        if subset is not None and li not in subset:
            continue
        d = d.to(dtype=torch.float32, device="cpu").reshape(-1)
        d = d / d.norm().clamp(min=1e-8)
        alpha = config["alpha"]
        mpoa = bool(config["mpoa"])
        layer = layers[li]
        for wname in ("o_proj", "conv_out", "w2"):
            mod = _resolve_proj(layer, wname)
            if mod is not None:
                project_2d(mod.weight.data, d, alpha, mpoa)
                applied.append(f"layer.{li}.{wname}")
    return applied


@app.function(image=image, gpu=GPU, timeout=TIMEOUT, retries=0,
              secrets=[modal.Secret.from_name("hf-write-token")],
              volumes={"/out": VOLUME})
def run_input_actdiff(only: str | None = None) -> dict:
    sys.path.insert(0, "/absolver")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prompts import (DEFAULT_HARMFUL, DEFAULT_HARMLESS, EXPANDED_HARMFUL,
                         EXPANDED_HARMLESS)
    from verify import run_mmlu_mini, _model_device
    from refusal_lfm26 import refusal_score

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True).to("cuda")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    dev = _model_device(model)
    layers = model.model.layers
    print(f"[26b-input] {MODEL_ID} on {dev}, {len(layers)} layers", flush=True)

    # ---------------- input-phase activation harvest ----------------
    def chat_text(p: str) -> str:
        return tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False,
            add_generation_prompt=True)

    def collect_last_hidden(prompts: list[str]) -> dict[int, list]:
        store: dict[int, list] = defaultdict(list)

        def make_hook(idx: int):
            def hook(_mod, _inp, output):
                hs = output[0] if isinstance(output, tuple) else output
                if not hasattr(hs, "dim"):
                    return
                if hs.dim() == 3:
                    hs = hs[:, -1, :]      # last token [B, H]
                elif hs.dim() == 2:
                    hs = hs[:, -1] if hs.shape[1] > 1 else hs[:, 0]
                store[idx].append(hs.detach().float().cpu())
            return hook

        handles = [layers[i].register_forward_hook(make_hook(i))
                   for i in range(len(layers))]
        try:
            for p in prompts:
                inp = tok(chat_text(p), return_tensors="pt", truncation=True,
                          max_length=256).to(dev)
                with torch.no_grad():
                    model(**inp)
        finally:
            for h in handles:
                h.remove()
        return {i: vs for i, vs in store.items()}

    print(f"[26b-input] harvesting input-phase acts "
          f"({len(DEFAULT_HARMFUL)} harmful / {len(DEFAULT_HARMLESS)} benign)",
          flush=True)
    harm_acts = collect_last_hidden(DEFAULT_HARMFUL)
    benign_acts = collect_last_hidden(DEFAULT_HARMLESS)

    dirs: dict[int, torch.Tensor] = {}
    seps: dict[int, float] = {}
    for i in range(len(layers)):
        if i not in harm_acts or i not in benign_acts:
            continue
        h = torch.stack(harm_acts[i]).mean(dim=0)
        b = torch.stack(benign_acts[i]).mean(dim=0)
        d = h - b
        dirs[i] = d
        seps[i] = float(d.norm())
    top = sorted(seps.items(), key=lambda kv: -kv[1])[:8]
    print(f"[26b-input] separation top: "
          f"{[(li, round(s, 3)) for li, s in top]}", flush=True)

    # ---------------- gate helpers ----------------
    def judge_refusal(prompt_subset, transcript) -> float:
        n_ref = 0
        for p in prompt_subset:
            inp = tok(chat_text(p), return_tensors="pt", truncation=True,
                      max_length=1024).to(dev)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=128, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            resp = tok.decode(out[0][inp["input_ids"].shape[1]:],
                              skip_special_tokens=True)
            transcript.append({"gate": "judge_refusal", "prompt": p,
                               "response": resp})
            if refusal_score(resp) > 0.5:
                n_ref += 1
        return n_ref / len(prompt_subset)

    def benign_stats(n=4) -> dict:
        stats = []
        for p in EXPANDED_HARMLESS[:n]:
            inp = tok(chat_text(p), return_tensors="pt", truncation=True,
                      max_length=1024).to(dev)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=96, do_sample=False,
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
    pr_refusal = judge_refusal(EXPANDED_HARMFUL[:15], pr_tr)
    pr_benign = benign_stats()
    print(f"[26b-input] PRISTINE refusal={pr_refusal:.3f} "
          f"benign_len={pr_benign['benign_mean_word_len']:.2f} "
          f"uniq={pr_benign['benign_uniq_ratio']:.2f}", flush=True)

    snap = [(mod.weight.data, mod.weight.data.detach().clone())
            for li in range(len(layers))
            for wname in ("o_proj", "conv_out", "w2")
            if (mod := _resolve_proj(layers[li], wname)) is not None]
    print(f"[26b-input] snapshot {len(snap)} tensors", flush=True)

    top_layers = [li for li, _ in top]
    configs = [
        {"name": "input-all-mpoa-1.0", "alpha": 1.0, "mpoa": True,
         "layers": None},
        {"name": "input-all-mpoa-1.5", "alpha": 1.5, "mpoa": True,
         "layers": None},
        {"name": "input-all-mpoa-2.0", "alpha": 2.0, "mpoa": True,
         "layers": None},
        {"name": "input-top-mpoa-2.0", "alpha": 2.0, "mpoa": True,
         "layers": top_layers},
        {"name": "input-top-mpoa-1.5", "alpha": 1.5, "mpoa": True,
         "layers": top_layers},
        {"name": "input-top-plain-1.0", "alpha": 1.0, "mpoa": False,
         "layers": top_layers},
    ]
    if only:
        configs = [c for c in configs if c["name"] == only]

    results = {"model_id": MODEL_ID, "method": "activation-diff input-phase",
               "n_prompts": len(DEFAULT_HARMFUL), "n_dirs": len(dirs),
               "separation_top": top,
               "pristine": {"refusal": pr_refusal, "benign": pr_benign},
               "candidates": []}
    transcripts: dict[str, list] = {"pristine": pr_tr}

    out_root = Path("/out/lfm2.5-2.6b-transfer")
    out_root.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())

    for config in configs:
        name = config["name"]
        print(f"\n===== {name} =====", flush=True)
        for data, pristine in snap:
            data.copy_(pristine)
        torch.cuda.empty_cache()
        applied = apply_directions(model, dirs, config,
                                   layer_subset=config.get("layers"))
        print(f"applied {len(applied)} weights", flush=True)

        trans: list[dict] = []
        refusal = judge_refusal(EXPANDED_HARMFUL[:15], trans)
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
            "config": {k: v for k, v in config.items()
                       if v is not None}, "n_applied": len(applied),
            "refusal": refusal, "benign": benign, "mmlu": mmlu,
        })
        transcripts[name] = trans
        res_path = out_root / f"actdiff-input_{ts}.json"
        res_path.write_text(json.dumps(results, indent=2, default=str))
        tr_path = out_root / f"actdiff-input_transcripts_{ts}.json"
        tr_path.write_text(json.dumps(transcripts, indent=2, default=str))
        print(f"[26b-input] incremental save candidates done "
              f"{len(results['candidates'])}", flush=True)

    print(f"[26b-input] done ({time.time()-t0:.0f}s)", flush=True)
    return {"results_path": str(out_root / f"actdiff-input_{ts}.json"),
            "results": results}


@app.local_entrypoint()
def main(only: str | None = None):
    res = run_input_actdiff.remote(only=only)
    print(json.dumps(res.get("results", {}), indent=2, default=str))


if __name__ == "__main__":
    main()
