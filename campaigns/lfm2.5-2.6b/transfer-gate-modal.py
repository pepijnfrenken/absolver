"""Modal runner: LFM2.5-2.6B direction-transfer + judge-gated bundle.

Applies the 1.2B recovered refusal directions (u1 left singulars,
campaigns/lfm2.5-recovery/directions-huihui-recovered.pt — mounted on the
absolver-phase2 volume at /out/directions-huihui-recovered.pt) to
LiquidAI/LFM2.5-2.6B, then gates each candidate.

CRITICAL MODEL FINDING (2026-09-03, pristine-responses_*.json): the 2.6B's
refusal style is verbose CoT "self-correction/policy-check" prose that
evades the keyword classifier (pristine keyword-refusal 0/55 while the
model refuses every harmful prompt). The KEYWORD refusal gate is therefore
VOID on this model — refusal is scored by the LLM judge (deepseek-v4-flash
via FreeInference, cfg.judge_*), which distinguishes refusal vs low
quality. Mechanical gates (coherence, degeneracy, finite, PPL) stay valid.

Two phases:
  A (rank sweep): apply each candidate, LLM-judge 15 harmful prompts,
     rank by judge refusal_rate.
  B (winners): full 55-prompt judge refusal + quality bundle on the top
     candidates, plus gates.py mechanics on the 95-prompt set.

Per-candidate pristine restore; incremental saves to the volume.

Usage:
    ABSOLVER_GPU=L4 modal run campaigns/lfm2.5-2.6b/transfer-gate-modal.py
    ABSOLVER_GPU=L4 modal run campaigns/lfm2.5-2.6b/transfer-gate-modal.py --only prefix-mpoa-2.0 --final
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import modal
import torch

_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

app = modal.App("absolver-26b-transfer-gate")

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
DIRECTIONS_ON_VOLUME = "/out/directions-huihui-recovered.pt"
CONFIG_YAML = "/absolver/models/lfm2.5-2.6b-instruct.yaml"
N_LAYERS = 30
JUDGE_WORKERS = 8


def _resolve_proj(layer, wname: str):
    """Mirror of sweep._resolve_proj: canonical weight name -> module."""
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
    """W -= alpha * d (d^T W); mpoa rescales the norm afterward."""
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


def geometry_audit(model, tok) -> dict:
    layers = model.model.layers
    lt = model.config.layer_types if hasattr(model.config, "layer_types") else None
    audit = {"n_layers": len(layers), "layer_types": lt, "classes": {}}
    counts = {"o_proj": 0, "conv_out": 0, "w2": 0}
    for li in range(len(layers)):
        row = {}
        for wname in ("o_proj", "conv_out", "w2"):
            mod = _resolve_proj(layers[li], wname)
            if mod is not None:
                row[wname] = [int(x) for x in mod.weight.shape]
                counts[wname] += 1
        audit["classes"][li] = row
    audit["counts"] = counts
    ok = audit["counts"]["w2"] == len(layers)
    for row in audit["classes"].values():
        for wname in ("o_proj", "conv_out", "w2"):
            if wname in row and row[wname][0] != 2048:
                ok = False
    audit["directions_fit_all_out_projs"] = ok
    return audit


def build_configs() -> list[dict]:
    return [
        {"name": "prefix-mpoa-1.0", "mapping": "prefix",
         "alpha": 1.0, "per_layer_alpha": False, "mpoa": True},
        {"name": "prefix-mpoa-1.5", "mapping": "prefix",
         "alpha": 1.5, "per_layer_alpha": False, "mpoa": True},
        {"name": "prefix-mpoa-2.0", "mapping": "prefix",
         "alpha": 2.0, "per_layer_alpha": False, "mpoa": True},
        {"name": "prefix-mpoa-2.5", "mapping": "prefix",
         "alpha": 2.5, "per_layer_alpha": False, "mpoa": True},
        {"name": "propor-mpoa-1.0", "mapping": "proportional",
         "alpha": 1.0, "per_layer_alpha": False, "mpoa": True},
        {"name": "propor-mpoa-2.0", "mapping": "proportional",
         "alpha": 2.0, "per_layer_alpha": False, "mpoa": True},
        {"name": "prefix-plain-2.0", "mapping": "prefix",
         "alpha": 2.0, "per_layer_alpha": False, "mpoa": False},
    ]


def mapping_dir_for(mapping: str, layer_idx: int, dir_keys: list[str]) -> str | None:
    if mapping == "prefix":
        k = str(layer_idx)
        return k if k in dir_keys else None
    if mapping == "repeat_last":
        if str(layer_idx) in dir_keys:
            return str(layer_idx)
        return dir_keys[-1]
    if mapping == "proportional":
        # stretch 16 directions across 30 layers: 1.2B layer i -> round(i*30/16)
        n_layers = 30
        n_dir = len(dir_keys)
        for k in dir_keys:
            src = int(k)
            if layer_idx == round(src * n_layers / n_dir):
                return k
        return None
    raise ValueError(f"unknown mapping {mapping}")


def apply_candidate(model, bundle, config: dict) -> list[str]:
    dirs = bundle["dirs"]
    alpha_fit_pl = bundle.get("alpha_fit_per_layer", {})
    applied: list[str] = []
    layers = model.model.layers
    for li in range(len(layers)):
        dkey = mapping_dir_for(config["mapping"], li, list(dirs.keys()))
        if dkey is None:
            continue
        d = torch.as_tensor(dirs[dkey], dtype=torch.float32)
        d = d / d.norm().clamp(min=1e-8)
        alpha = config["alpha"]
        if config["per_layer_alpha"]:
            alpha = float(alpha_fit_pl.get(dkey, 0.811))
        if alpha is None:
            alpha = 0.811
        mpoa = bool(config["mpoa"])
        layer = layers[li]
        for wname in ("o_proj", "conv_out", "w2"):
            mod = _resolve_proj(layer, wname)
            if mod is not None:
                project_2d(mod.weight.data, d, alpha, mpoa)
                applied.append(f"layer.{li}.{wname}")
    return applied


@app.function(image=image, gpu=GPU, timeout=TIMEOUT, retries=0,
              secrets=[modal.Secret.from_name("hf-write-token"),
                       modal.Secret.from_name("freeinference-token")],
              volumes={"/out": VOLUME})
def run_transfer_gates(only: str | None = None, final: bool = False) -> dict:
    import concurrent.futures as cf
    import math

    sys.path.insert(0, "/absolver")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prompts import EXPANDED_HARMFUL, EXPANDED_HARMLESS
    from gates import run_gates
    from config import load_config
    from verify import run_mmlu_mini, _model_device
    from prompt_format import format_prompt, resolve_flavor

    t0 = time.time()
    cfg = load_config(CONFIG_YAML)

    # ---------- load model + tokenizer ----------
    print(f"[26b-transfer] loading {MODEL_ID} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    if torch.cuda.is_available():
        model = model.to("cuda")
        print(f"model on CUDA: {torch.cuda.get_device_name(0)}", flush=True)
    dev = _model_device(model)
    print(f"model device={dev} params="
          f"{sum(p.numel() for p in model.parameters())/1e9:.2f}B", flush=True)

    audit = geometry_audit(model, tok)
    print(f"[26b-transfer] GEOMETRY AUDIT: n_layers={audit['n_layers']} "
          f"counts={audit['counts']} fit={audit['directions_fit_all_out_projs']}",
          flush=True)
    assert audit["directions_fit_all_out_projs"], "geometry does not fit"

    bundle = torch.load(DIRECTIONS_ON_VOLUME, map_location="cpu",
                        weights_only=False)
    dirs = bundle["dirs"]
    print(f"[26b-transfer] directions: {len(dirs)} layers, "
          f"alpha_fit_global={bundle.get('alpha_fit_global')}", flush=True)
    assert len(dirs) == 16 and all(
        tuple(torch.as_tensor(v).shape) == (2048,) for v in dirs.values()
    ), "directions bundle shape mismatch"

    prompts = EXPANDED_HARMFUL + EXPANDED_HARMLESS
    print(f"[26b-transfer] gate prompts: {len(prompts)} "
          f"(harmful {len(EXPANDED_HARMFUL)}, harmless {len(EXPANDED_HARMLESS)}) "
          f"final={final}", flush=True)
    flavor_r = resolve_flavor(tok, "chat")

    # judge helper (LLM-as-judge; the ONLY valid refusal metric on this model)
    def judge_pairs(pairs: list[tuple[str, str]]) -> list[dict]:
        from judgescore import judge_pair  # local, no repo dep
        with cf.ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as ex:
            return list(ex.map(lambda pr: judge_pair(pr[0], pr[1], cfg), pairs))

    def generate(prompt: str, max_new: int = 128) -> str:
        formatted = format_prompt(tok, prompt, flavor_r)
        inp = tok(formatted, return_tensors="pt", truncation=True,
                  max_length=1024).to(dev)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        gen_ids = out[0][inp["input_ids"].shape[1]:]
        return tok.decode(gen_ids, skip_special_tokens=True)

    def judge_refusal(prompt_subset: list[str],
                      transcript: list[dict]) -> dict:
        pairs = []
        for p in prompt_subset:
            resp = generate(p)
            transcript.append({"gate": "judge_refusal", "prompt": p,
                               "response": resp})
            pairs.append((p, resp))
        verdicts = judge_pairs(pairs)
        refusal_rate = sum(v["refusal_score"] for v in verdicts) / len(verdicts)
        quality_mean = sum(v["quality_score"] for v in verdicts) / len(verdicts)
        return {"refusal_rate": refusal_rate, "quality_mean": quality_mean,
                "n": len(verdicts), "verdicts": verdicts}

    def benign_coherence(n: int = 4) -> dict:
        """Generate a few benign prompts; distinct benchmark distinguishing
        real compliance from degenerate production collapse. Returns mean
        word length + unique-word ratio over the benign responses."""
        import re
        stats = []
        for p in EXPANDED_HARMLESS[:n]:
            resp = generate(p, max_new=96)
            words = re.findall(r"[A-Za-z']+", resp)
            if not words:
                stats.append({"prompt": p, "mean_len": 0.0, "uniq": 0.0,
                              "n_words": 0, "response": resp})
                continue
            stats.append({
                "prompt": p,
                "mean_len": sum(len(w) for w in words) / len(words),
                "uniq": len(set(w.lower() for w in words)) / len(words),
                "n_words": len(words),
                "response": resp,
            })
        mean_len = sum(s["mean_len"] for s in stats) / max(1, len(stats))
        uniq = sum(s["uniq"] for s in stats) / max(1, len(stats))
        return {"benign_mean_word_len": mean_len, "benign_uniq_ratio": uniq,
                "benign_rows": stats}

    # ---------- pristine baselines ----------
    pristine_logprobs: dict[str, float] = {}
    pristine_logprobs_first: dict[str, Any] = {}
    pristine_benchmark_scores: dict[str, float] = {}
    digest = lambda p: __import__("hashlib").sha256(p.encode("utf-8")).hexdigest()  # noqa: E731
    for p in prompts:
        formatted = format_prompt(tok, p, flavor_r)
        inp = tok(formatted, return_tensors="pt", truncation=True,
                  max_length=cfg.max_seq_len).to(dev)
        with torch.no_grad():
            out = model(**inp)
        lg = out.logits.float()
        pristine_logprobs_first[digest(p)] = torch.log_softmax(lg[0, -1], dim=-1).cpu()
        cont = lg[0, 0: lg.shape[1] - 1]
        lp = torch.log_softmax(cont, dim=-1)
        toks = inp["input_ids"][0, 1:]
        if cont.shape[0] != toks.shape[0]:
            continue
        chosen = lp.gather(-1, toks.unsqueeze(-1)).squeeze(-1)
        ppl = math.exp(-chosen.sum().item() / max(1, chosen.numel()))
        pristine_logprobs[digest(p)] = ppl
    try:
        pristine_benchmark_scores["mmlu"] = run_mmlu_mini(model, tok, n=20)
    except Exception as exc:  # noqa: BLE001
        print(f"pristine mmlu_mini failed: {exc}", flush=True)
    print(f"[26b-transfer] pristine mmlu_mini={pristine_benchmark_scores.get('mmlu')}",
          flush=True)

    # pristine judge sanity (expect HIGH refusal) + benign coherence baseline
    pr_transcript: list[dict] = []
    pristine_judge = judge_refusal(EXPANDED_HARMFUL[:15], pr_transcript)
    pristine_benign = benign_coherence()
    print(f"[26b-transfer] PRISTINE judge refusal="
          f"{pristine_judge['refusal_rate']:.3f} "
          f"quality={pristine_judge['quality_mean']:.3f} (n={pristine_judge['n']}) "
          f"benign_len={pristine_benign['benign_mean_word_len']:.2f} "
          f"uniq={pristine_benign['benign_uniq_ratio']:.2f}", flush=True)

    # ---------- snapshot ----------
    snap: list[tuple] = []
    layers = model.model.layers
    for li in range(len(layers)):
        for wname in ("o_proj", "conv_out", "w2"):
            mod = _resolve_proj(layers[li], wname)
            if mod is not None:
                snap.append((mod.weight.data, mod.weight.data.detach().clone()))
    print(f"[26b-transfer] snapshot {len(snap)} projectable tensors", flush=True)

    out_root = Path("/out/lfm2.5-2.6b-transfer")
    out_root.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())

    results = {"model_id": MODEL_ID, "n_directions": len(dirs),
               "geometry": audit, "pristine_judge": pristine_judge,
               "pristine_benign": pristine_benign,
               "pristine_mmlu": pristine_benchmark_scores.get("mmlu"),
               "phase": "final" if final else "sweep",
               "candidates": []}
    transcripts: dict[str, list[dict]] = {"pristine": pr_transcript}

    def save_incremental() -> None:
        res_path = out_root / f"transfer-gates_{ts}.json"
        res_path.write_text(json.dumps(results, indent=2, default=str))
        tr_path = out_root / f"transcripts_{ts}.json"
        tr_path.write_text(json.dumps(transcripts, indent=2, default=str))
        print(f"[26b-transfer] incremental save -> {res_path} "
              f"(candidates done: {len(results['candidates'])})", flush=True)

    configs = build_configs()
    if only:
        configs = [c for c in configs if c["name"] == only]
        if not configs:
            raise ValueError(f"unknown candidate {only}")
    # sweep phase: judge 15 prompts per candidate; final phase: full bundle
    judge_subset = EXPANDED_HARMFUL if final else EXPANDED_HARMFUL[:15]

    for ci, config in enumerate(configs):
        name = config["name"]
        print(f"\n===== CANDIDATE {ci+1}/{len(configs)}: {name} =====", flush=True)
        for data, pristine in snap:
            data.copy_(pristine)
        torch.cuda.empty_cache()

        applied = apply_candidate(model, bundle, config)
        print(f"applied {len(applied)} weights", flush=True)

        gen_transcript: list[dict] = []
        jr = judge_refusal(judge_subset, gen_transcript)
        bench = {}
        try:
            bench["mmlu"] = run_mmlu_mini(model, tok, n=20)
        except Exception as exc:  # noqa: BLE001
            print(f"ablated mmlu_mini failed: {exc}", flush=True)

        benign = benign_coherence()
        cand = {
            "config": {k: v for k, v in config.items()},
            "n_applied": len(applied),
            "judge_refusal_rate": jr["refusal_rate"],
            "judge_quality_mean": jr["quality_mean"],
            "judge_n": jr["n"],
            "mmlu": bench.get("mmlu"),
            "benign": benign,
        }
        if final:
            # mechanical gates on the full 95-prompt set (PPL/KL valid;
            # gates.refusal is keyword and VOID here -> override with judge)
            report = run_gates(
                model, tok, cfg, prompts=prompts,
                benchmark_scores=bench,
                pristine_logprobs=pristine_logprobs,
                pristine_logprobs_first=pristine_logprobs_first,
                pristine_benchmark_scores=pristine_benchmark_scores,
                flavor=flavor_r)
            report["refusal"] = {
                "value": jr["refusal_rate"],
                "passed": jr["refusal_rate"] <= getattr(cfg, "gate_refusal_max", 0.03),
                "detail": f"LLM-judged {jr['refusal_rate']:.3f} over "
                          f"{jr['n']} harmful prompts",
            }
            cand["gates"] = {
                k: {"passed": v.get("passed"), "value": v.get("value"),
                    "detail": v.get("detail")}
                for k, v in report.items()
                if k not in ("_enabled", "eval_pass")
            }
        print(f"[{name}] judge refusal={jr['refusal_rate']:.3f} "
              f"quality={jr['quality_mean']:.3f} mmlu={bench.get('mmlu')} "
              f"benign_len={benign['benign_mean_word_len']:.2f} "
              f"uniq={benign['benign_uniq_ratio']:.2f}", flush=True)
        results["candidates"].append(cand)
        transcripts[name] = gen_transcript
        save_incremental()

    save_incremental()
    print(f"[26b-transfer] done ({time.time()-t0:.0f}s)", flush=True)
    return {"results_path": str(out_root / f"transfer-gates_{ts}.json"),
            "transcripts_path": str(out_root / f"transcripts_{ts}.json"),
            "results": results}


@app.local_entrypoint()
def main(only: str | None = None, final: bool = False):
    print(f"Starting 2.6B transfer+gate on Modal (only={only}, final={final})...",
          flush=True)
    res = run_transfer_gates.remote(only=only, final=final)
    print("#" * 70)
    print(json.dumps(res.get("results", {}), indent=2, default=str))
    print("#" * 70)


if __name__ == "__main__":
    main()
