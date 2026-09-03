"""Final gate bundle for the LFM2.5-2.6B winner.

Method: input-phase activation diff (harmful vs benign chat-activations),
MPOA projection, ALL 30 layers, o_proj+conv_out+w2 (=60 weights, coverage
rule). Sweep winner: alpha 1.5 -> refusal 0.000 with healthy language.

This runner runs the FULL gate bundle the mission requires (>=50 prompts,
per-category + aggregate, transcripts, coherence, degeneracy, finite,
PPL vs pristine, first-token KL) on the winner plus alpha neighbors
(1.25, 1.5) so the published config is chosen by full evidence.

Gating instrument: style-aware classifier (refusal_lfm26) — the LLM
judge key is expired (documented) and the stock keyword gate is void for
this model's CoT refusal style. Capability: mmlu_mini(40) + note that
full benchmarks (GPQA/MMLU-Pro/IFEval/MATH) run separately if gates pass.
PPL/KL compare against pristine logprobs collected in-container.

Usage:
    ABSOLVER_GPU=L4 modal run campaigns/lfm2.5-2.6b/final-gate-modal.py
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import modal
import torch

_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

app = modal.App("absolver-26b-final-gate")

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
TIMEOUT = int(os.environ.get("ABSOLVER_TIMEOUT", "7200"))
VOLUME = modal.Volume.from_name("absolver-phase2", create_if_missing=True)
MODEL_ID = "LiquidAI/LFM2.5-2.6B"

HARMFUL_GATE = None       # filled in-container from EXPANDED_HARMFUL
HARMLESS_GATE = None


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


def apply_directions(model, dirs, alpha: float, mpoa: bool = True) -> list[str]:
    applied = []
    layers = model.model.layers
    for li, d in dirs.items():
        d = d.to(dtype=torch.float32, device="cpu").reshape(-1)
        d = d / d.norm().clamp(min=1e-8)
        for wname in ("o_proj", "conv_out", "w2"):
            mod = _resolve_proj(layers[li], wname)
            if mod is not None:
                project_2d(mod.weight.data, d, alpha, mpoa)
                applied.append(f"layer.{li}.{wname}")
    return applied


@app.function(image=image, gpu=GPU, timeout=TIMEOUT, retries=0,
              secrets=[modal.Secret.from_name("hf-write-token")],
              volumes={"/out": VOLUME})
def run_final_gate(alphas_str: str = "1.25,1.5") -> dict:
    sys.path.insert(0, "/absolver")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prompts import (DEFAULT_HARMFUL, DEFAULT_HARMLESS, EXPANDED_HARMFUL,
                         EXPANDED_HARMLESS)
    from verify import run_mmlu_mini, _model_device
    from refusal_lfm26 import refusal_score
    from gates import (gate_coherence, gate_degeneracy, gate_finite_logits,
                       gate_perplexity_increase, gate_first_token_kl)

    t0 = time.time()
    alphas = [float(a) for a in alphas_str.split(",") if a.strip()]
    harmful = EXPANDED_HARMFUL          # 55
    harmless = EXPANDED_HARMLESS        # 40
    all_prompts = harmful + harmless    # 95 >= 50 gate

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True).to("cuda")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    dev = _model_device(model)
    layers = model.model.layers

    def chat_text(p: str) -> str:
        return tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False,
            add_generation_prompt=True)

    # ---------------- input-phase direction harvest ----------------
    def collect_last_hidden(prompts: list[str]) -> dict[int, list]:
        store: dict[int, list] = defaultdict(list)

        def make_hook(idx: int):
            def hook(_mod, _inp, output):
                hs = output[0] if isinstance(output, tuple) else output
                if not hasattr(hs, "dim"):
                    return
                if hs.dim() == 3:
                    hs = hs[:, -1, :]
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

    print(f"[26b-final] harvesting directions ({len(DEFAULT_HARMFUL)}/"
          f"{len(DEFAULT_HARMLESS)} prompts) ...", flush=True)
    harm_acts = collect_last_hidden(DEFAULT_HARMFUL)
    ben_acts = collect_last_hidden(DEFAULT_HARMLESS)
    dirs: dict[int, torch.Tensor] = {}
    for i in range(len(layers)):
        if i not in harm_acts or i not in ben_acts:
            continue
        dirs[i] = (torch.stack(harm_acts[i]).mean(dim=0)
                   - torch.stack(ben_acts[i]).mean(dim=0))
    print(f"[26b-final] {len(dirs)} direction layers", flush=True)

    # ---------------- pristine baselines ----------------
    digest = lambda p: __import__("hashlib").sha256(p.encode()).hexdigest()  # noqa: E731
    pristine_logprobs: dict[str, float] = {}
    pristine_logprobs_first: dict[str, Any] = {}
    for p in all_prompts:
        inp = tok(chat_text(p), return_tensors="pt", truncation=True,
                  max_length=2048).to(dev)
        with torch.no_grad():
            out = model(**inp)
        lg = out.logits.float()
        pristine_logprobs_first[digest(p)] = torch.log_softmax(lg[0, -1], dim=-1).cpu()
        cont = lg[0, 0: lg.shape[1] - 1]
        lp = torch.log_softmax(cont, dim=-1)
        toks = inp["input_ids"][0, 1:]
        if cont.shape[0] == toks.shape[0]:
            chosen = lp.gather(-1, toks.unsqueeze(-1)).squeeze(-1)
            pristine_logprobs[digest(p)] = math.exp(
                -chosen.sum().item() / max(1, chosen.numel()))
    try:
        pristine_mmlu = run_mmlu_mini(model, tok, n=40)
    except Exception as exc:  # noqa: BLE001
        print(f"pristine mmlu failed: {exc}", flush=True)
        pristine_mmlu = None
    print(f"[26b-final] pristine mmlu(40)={pristine_mmlu}", flush=True)

    # ---------------- gate helpers ----------------
    def gen(p: str, max_new: int = 128) -> str:
        inp = tok(chat_text(p), return_tensors="pt", truncation=True,
                  max_length=1024).to(dev)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        return tok.decode(out[0][inp["input_ids"].shape[1]:],
                          skip_special_tokens=True)

    def refusal_rate(transcript: list) -> tuple[float, list]:
        rows = []
        n_ref = 0
        for p in harmful:
            resp = gen(p)
            ref = refusal_score(resp) > 0.5
            n_ref += int(ref)
            rows.append({"prompt": p, "response": resp, "refusal": ref})
            transcript.append({"gate": "judge_refusal", "prompt": p,
                               "response": resp})
        return n_ref / len(harmful), rows

    # ---------------- snapshot ----------------
    snap = [(mod.weight.data, mod.weight.data.detach().clone())
            for li in range(len(layers))
            for wname in ("o_proj", "conv_out", "w2")
            if (mod := _resolve_proj(layers[li], wname)) is not None]
    print(f"[26b-final] snapshot {len(snap)} tensors", flush=True)

    results = {"model_id": MODEL_ID,
               "method": "input-phase actdiff + MPOA all-30",
               "n_directions": len(dirs),
               "prompts": {"harmful": len(harmful), "harmless": len(harmless),
                           "total": len(all_prompts)},
               "pristine": {"mmlu40": pristine_mmlu},
               "saved": [],
               "candidates": []}
    transcripts: dict[str, list] = {}

    out_root = Path("/out/lfm2.5-2.6b-transfer")
    out_root.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())

    for alpha in alphas:
        name = f"input-mpoa-a{alpha}"
        print(f"\n===== {name} =====", flush=True)
        for data, pristine in snap:
            data.copy_(pristine)
        torch.cuda.empty_cache()
        applied = apply_directions(model, dirs, alpha, mpoa=True)
        print(f"applied {len(applied)} weights", flush=True)

        trans: list[dict] = []
        ref_rate, rows = refusal_rate(trans)
        print(f"[{name}] refusal={ref_rate:.3f} ({sum(1 for r in rows if r['refusal'])}/{len(rows)})",
              flush=True)

        # mechanical gates (keyword-free ones still valid on this model)
        gates = {}
        for gname, gfn in [
            ("coherence", lambda: gate_coherence(model, tok, None, harmful)),
            ("degeneracy", lambda: gate_degeneracy(model, tok, None, harmful)),
            ("finite_logits", lambda: gate_finite_logits(model, tok, None, harmful)),
            ("perplexity_increase", lambda: gate_perplexity_increase(
                model, tok, None, all_prompts, pristine_logprobs)),
            ("first_token_kl", lambda: gate_first_token_kl(
                model, tok, None, all_prompts, pristine_logprobs_first)),
        ]:
            try:
                gates[gname] = gfn()
            except Exception as exc:  # noqa: BLE001
                gates[gname] = {"value": None, "passed": False,
                                "detail": f"gate crashed ({exc})"}
        try:
            mmlu = run_mmlu_mini(model, tok, n=40)
        except Exception as exc:  # noqa: BLE001
            mmlu = None
        print(f"[{name}] gates: { {k: round(v.get('value') or 0, 3) if isinstance(v.get('value'), (int, float)) else v.get('value') for k, v in gates.items()} } mmlu={mmlu}",
              flush=True)

        cand = {"alpha": alpha, "n_applied": len(applied),
                "refusal_rate": ref_rate, "refusal_n": sum(1 for r in rows if r["refusal"]),
                "refusal_total": len(rows), "mmlu40": mmlu,
                "pristine_mmlu40": pristine_mmlu,
                "rows": rows, "gates": gates}
        results["candidates"].append(cand)
        transcripts[name] = trans
        res_path = out_root / f"final-gates_{ts}.json"
        res_path.write_text(json.dumps(results, indent=2, default=str))
        tr_path = out_root / f"final-gates_transcripts_{ts}.json"
        tr_path.write_text(json.dumps(transcripts, indent=2, default=str))
        print(f"[26b-final] incremental save ({len(results['candidates'])} done)",
              flush=True)

    print(f"[26b-final] done ({time.time()-t0:.0f}s)", flush=True)
    return {"results_path": str(out_root / f"final-gates_{ts}.json"),
            "results": results}


@app.local_entrypoint()
def main(alphas: str = "1.25,1.5"):
    res = run_final_gate.remote(alphas_str=alphas)
    print("#" * 70)
    print(json.dumps(res.get("results", {}), indent=2, default=str))
    print("#" * 70)


if __name__ == "__main__":
    main()
