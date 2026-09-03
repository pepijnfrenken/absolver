"""Needle-threading sweep: find the alpha where the 2.6B input-phase MPOA
edit passes BOTH the refusal gate (<=0.03 on 55 prompts) and the PPL gate
(<=+15%), with content-quality classification (hallucinated vs coherent).

Also tests separation-weighted per-layer alpha (stronger on high-sep
layers 24-29, weaker lower) — a refinement for collateral PPL damage.

This is the last DIY sweep before documenting the campaign outcome.
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

app = modal.App("absolver-26b-thread")

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


@app.function(image=image, gpu=GPU, timeout=TIMEOUT, retries=0,
              secrets=[modal.Secret.from_name("hf-write-token")],
              volumes={"/out": VOLUME})
def run_thread(only: str | None = None) -> dict:
    sys.path.insert(0, "/absolver")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prompts import (DEFAULT_HARMFUL, DEFAULT_HARMLESS, EXPANDED_HARMFUL)
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

    def chat_text(p: str) -> str:
        return tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False,
            add_generation_prompt=True)

    # ---------------- direction harvest (input-phase) ----------------
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

    print(f"[26b-thread] harvesting ({len(DEFAULT_HARMFUL)}/{len(DEFAULT_HARMLESS)})",
          flush=True)
    ha = collect_last_hidden(DEFAULT_HARMFUL)
    ba = collect_last_hidden(DEFAULT_HARMLESS)
    dirs: dict[int, torch.Tensor] = {}
    seps: dict[int, float] = {}
    for i in range(len(layers)):
        if i not in ha or i not in ba:
            continue
        d = (torch.stack(ha[i]).mean(dim=0) - torch.stack(ba[i]).mean(dim=0))
        dirs[i] = d
        seps[i] = float(d.norm())
    top = sorted(seps.items(), key=lambda kv: -kv[1])[:8]
    print(f"[26b-thread] sep top: {[(li, round(s, 2)) for li, s in top]}", flush=True)
    # normalized separation profile for weighted alpha (0..1)
    max_sep = max(seps.values())
    norm_sep = {li: seps[li] / max_sep for li in seps}

    # ---------------- helpers ----------------
    def gen(p: str, max_new: int = 128) -> str:
        inp = tok(chat_text(p), return_tensors="pt", truncation=True,
                  max_length=1024).to(dev)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        return tok.decode(out[0][inp["input_ids"].shape[1]:],
                          skip_special_tokens=True)

    def prompt_ppl(p: str) -> float:
        inp = tok(chat_text(p), return_tensors="pt", truncation=True,
                  max_length=2048).to(dev)
        with torch.no_grad():
            out = model(**inp)
        lg = out.logits.float()
        cont = lg[0, 0: lg.shape[1] - 1]
        lp = torch.log_softmax(cont, dim=-1)
        toks = inp["input_ids"][0, 1:]
        if cont.shape[0] != toks.shape[0]:
            return float("nan")
        chosen = lp.gather(-1, toks.unsqueeze(-1)).squeeze(-1)
        return math.exp(-chosen.sum().item() / max(1, chosen.numel()))

    # pristine PPL baseline on a fixed subset
    ppl_prompts = EXPANDED_HARMFUL[:20]
    pristine_ppl = {p: prompt_ppl(p) for p in ppl_prompts}
    try:
        pristine_mmlu = run_mmlu_mini(model, tok, n=30)
    except Exception as exc:  # noqa: BLE001
        pristine_mmlu = None
    print(f"[26b-thread] pristine mmlu(30)={pristine_mmlu}", flush=True)

    snap = [(mod.weight.data, mod.weight.data.detach().clone())
            for li in range(len(layers))
            for wname in ("o_proj", "conv_out", "w2")
            if (mod := _resolve_proj(layers[li], wname)) is not None]
    print(f"[26b-thread] snapshot {len(snap)} tensors", flush=True)

    def apply(alpha: float, weighted: bool = False, mpoa: bool = True,
              passes: int = 1) -> list[str]:
        applied = []
        for _ in range(passes):
            for li, d in dirs.items():
                a = alpha
                if weighted:
                    a = alpha * (0.3 + 0.7 * norm_sep[li])
                d = d.to(dtype=torch.float32, device="cpu").reshape(-1)
                d = d / d.norm().clamp(min=1e-8)
                for wname in ("o_proj", "conv_out", "w2"):
                    mod = _resolve_proj(layers[li], wname)
                    if mod is not None:
                        project_2d(mod.weight.data, d, a, mpoa)
                        applied.append(f"layer.{li}.{wname}")
        return applied

    # ---------------- candidates ----------------
    # SVD rank-1 refinement: direction = u1 of the [prompt x hidden]
    # harmful-benign difference matrix (per layer), the closest DIY
    # analogue to the 1.2B exact rank-1 edit. Only implement when the
    # flag is set via config; otherwise use the mean diff.
    svd_dirs: dict[int, torch.Tensor] = {}
    for li in range(len(layers)):
        if li not in ha or li not in ba:
            continue
        H = torch.stack(ha[li])          # [P, H]
        B = torch.stack(ba[li])          # [P, H]
        n = min(H.shape[0], B.shape[0])
        D = H[:n] - B[:n]
        # center and take top left singular vector
        Dc = D - D.mean(dim=0, keepdim=True)
        try:
            _, _, Vh = torch.linalg.svd(Dc.float(), full_matrices=False)
            u1 = Vh[0]                   # [H] dominant separation direction
            svd_dirs[li] = u1
        except Exception as exc:  # noqa: BLE001
            print(f"  svd layer {li} failed: {exc}", flush=True)
    print(f"[26b-thread] svd rank-1 dirs: {len(svd_dirs)}", flush=True)

    def apply_dirs(dirs_map, alpha, weighted=False, mpoa=True, passes=1):
        applied = []
        for _ in range(passes):
            for li, d in dirs_map.items():
                a = alpha
                if weighted:
                    a = alpha * (0.3 + 0.7 * norm_sep[li])
                d = d.to(dtype=torch.float32, device="cpu").reshape(-1)
                d = d / d.norm().clamp(min=1e-8)
                for wname in ("o_proj", "conv_out", "w2"):
                    mod = _resolve_proj(layers[li], wname)
                    if mod is not None:
                        project_2d(mod.weight.data, d, a, mpoa)
                        applied.append(f"layer.{li}.{wname}")
        return applied

    configs = [
        {"name": "svd1-mpoa-1.5", "alpha": 1.5, "weighted": False, "mpoa": True,
         "dirs": "svd"},
        {"name": "svd1-mpoa-1.75", "alpha": 1.75, "weighted": False, "mpoa": True,
         "dirs": "svd"},
        {"name": "svd1-mpoa-2.0", "alpha": 2.0, "weighted": False, "mpoa": True,
         "dirs": "svd"},
        {"name": "svd1-plain-1.0", "alpha": 1.0, "weighted": False, "mpoa": False,
         "dirs": "svd"},
    ]
    if only:
        configs = [c for c in configs if c["name"] == only]

    results = {"model_id": MODEL_ID, "method": "input-phase actdiff MPOA all-30",
               "pristine_ppl_mean": sum(pristine_ppl.values()) / len(pristine_ppl),
               "pristine_mmlu30": pristine_mmlu,
               "separation_top": top, "candidates": []}
    transcripts: dict[str, list] = {}

    for config in configs:
        name = config["name"]
        print(f"\n===== {name} =====", flush=True)
        for data, pristine in snap:
            data.copy_(pristine)
        torch.cuda.empty_cache()
        dirs_map = svd_dirs if config.get("dirs") == "svd" else dirs
        applied = apply_dirs(dirs_map, config["alpha"],
                             weighted=config.get("weighted", False),
                             mpoa=config.get("mpoa", True),
                             passes=config.get("passes", 1))
        print(f"applied {len(applied)} weights ({config.get('dirs') or 'mean'})",
              flush=True)

        # refusal on full 55
        n_ref = 0
        rows = []
        for p in EXPANDED_HARMFUL:
            resp = gen(p)
            ref = refusal_score(resp) > 0.5
            n_ref += int(ref)
            rows.append({"prompt": p, "response": resp, "refusal": ref})
        ref_rate = n_ref / len(EXPANDED_HARMFUL)

        # PPL on the fixed subset
        ppl_incs = []
        for p in ppl_prompts:
            v = prompt_ppl(p)
            if math.isnan(v) or pristine_ppl.get(p, 0) <= 0:
                continue
            ppl_incs.append(v / pristine_ppl[p] - 1.0)
        ppl_inc = sum(ppl_incs) / max(1, len(ppl_incs)) if ppl_incs else float("nan")

        # content classification on first 15 compliant rows
        def classify(resp):
            if refusal_score(resp) > 0.5:
                return "refusal"
            w = re.findall(r"[A-Za-z']+", resp)
            if len(w) < 8:
                return "degenerate"
            ml = sum(len(x) for x in w) / len(w)
            if ml < 3.2:
                return "degenerate"
            # hallucination heuristic: low info-word density / absence of
            # concrete terms is hard to auto-detect; keep manual review rows
            return "coherent"
        classes = {}
        for r in rows[:15]:
            c = classify(r["response"])
            classes[c] = classes.get(c, 0) + 1

        try:
            mmlu = run_mmlu_mini(model, tok, n=30)
        except Exception as exc:  # noqa: BLE001
            mmlu = None
        print(f"[{name}] refusal={ref_rate:.3f} ({n_ref}/55) "
              f"ppl_inc={ppl_inc:+.3f} classes={classes} mmlu={mmlu}",
              flush=True)

        cand = {"config": config, "refusal_rate": ref_rate, "refusal_n": n_ref,
                "ppl_increase": ppl_inc, "mmlu30": mmlu, "classes": classes,
                "rows": rows}
        results["candidates"].append(cand)
        transcripts[name] = rows
        out_root = Path("/out/lfm2.5-2.6b-transfer")
        out_root.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        out_root.joinpath(f"thread_{ts}.json").write_text(
            json.dumps(results, indent=2, default=str))
        out_root.joinpath(f"thread_transcripts_{ts}.json").write_text(
            json.dumps(transcripts, indent=2, default=str))
        print(f"[26b-thread] incremental save ({len(results['candidates'])} done)",
              flush=True)

    print(f"[26b-thread] done ({time.time()-t0:.0f}s)", flush=True)
    return {"results": results}


@app.local_entrypoint()
def main(only: str | None = None):
    res = run_thread.remote(only=only)
    print(json.dumps(res.get("results", {}), indent=2, default=str))


if __name__ == "__main__":
    main()
