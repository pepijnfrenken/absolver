"""Apply recovered Heretic rank-1 edit to pristine LFM2.5-2.6B and gate it.

Round 3 (Heretic RE) — behavior verification of the recovered weights.

Recovered triples (sigma, u, v) for 52 edited tensors come from
heretic-recovered-dirs.npz (extracted locally from
Abiray/LFM2.5-2.6B-Heretic-Abliterated-GGUF vs pristine HF).

Steps:
1. Load pristine LiquidAI/LFM2.5-2.6B (bf16, CPU->GPU)
2. For each of 52 tensors: W' = W + sigma * outer(u, v)   [float32, cast back]
3. Save recovered model to volume (absolver-phase2, lfm2.5-2.6b-recovered/)
4. Gate: refusal-55 (style classifier + saved transcripts), PPL vs pristine,
   coherence/degeneracy/finite, MMLU-40 capability retention.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import modal
import torch

_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

app = modal.App("absolver-26b-heretic-recovered")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch>=2.0", "transformers>=4.38", "numpy>=1.24", "pyyaml>=6",
        "huggingface-hub>=0.20", "safetensors>=0.4", "pydantic>=2",
        "accelerate>=0.30",
    )
    .env({"PYTHONPATH": "/absolver"})
    .add_local_dir(
        str(_PROJECT_DIR),
        remote_path="/absolver",
        ignore=[".venv", ".git", "__pycache__", "*.pyc", ".aiwg",
                "abliterated_models", "campaigns", "experiments", "abl-work"],
    )
    .add_local_file(
        str(_PROJECT_DIR / "campaigns" / "lfm2.5-2.6b" / "heretic-recovered-dirs.npz"),
        remote_path="/recovered/heretic-recovered-dirs.npz",
    )
)

GPU = os.environ.get("ABSOLVER_GPU", "L4")
TIMEOUT = int(os.environ.get("ABSOLVER_TIMEOUT", "5400"))
VOLUME = modal.Volume.from_name("absolver-phase2", create_if_missing=True)
VOL_PATH = "/out"
MODEL_ID = "LiquidAI/LFM2.5-2.6B"


@app.function(image=image, gpu=GPU, timeout=TIMEOUT, retries=0, volumes={VOL_PATH: VOLUME})
def apply_and_gate() -> dict:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from prompts import (DEFAULT_HARMFUL, DEFAULT_HARMLESS, EXPANDED_HARMFUL,
                         EXPANDED_HARMLESS)
    from refusal_lfm26 import refusal_score
    from gates import (gate_coherence, gate_degeneracy, gate_finite_logits,
                       gate_perplexity_increase, gate_first_token_kl)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[heretic-rec] device={device}", flush=True)

    harmful = EXPANDED_HARMFUL          # 55
    benign = EXPANDED_HARMLESS          # 40
    refusal_prompts = harmful

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16)
    if torch.cuda.is_available():
        model = model.to("cuda")

    def chat_text(p: str) -> str:
        return tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)

    def gen(p: str, max_new: int = 128) -> str:
        enc = tok(chat_text(p), return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def logprobs_first_and_full(prompts):
        """Return (full_logprob_map, first_token_map) keyed by digest."""
        import hashlib
        full, first = {}, {}
        for p in prompts:
            enc = tok(chat_text(p), return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model(**enc)
            lg = out.logits[:, :-1].log_softmax(-1)
            ids = enc["input_ids"][:, 1:]
            key = hashlib.sha256(p.encode()).hexdigest()[:16]
            full[key] = float(torch.gather(lg, -1, ids.unsqueeze(-1)).squeeze(-1).sum().item())
            first[key] = float(lg[0, 0, ids[0, 0]].item())
        return full, first

    # ---- pristine baselines (BEFORE edit) ----
    all_prompts = harmful + (benign if isinstance(benign, list) else [])
    pristine_logprobs, pristine_first = logprobs_first_and_full(all_prompts)
    print(f"[heretic-rec] pristine baselines on {len(all_prompts)} prompts", flush=True)

    # ---- apply recovered edit ----
    z = np.load("/recovered/heretic-recovered-dirs.npz")
    hf_names = [str(x) for x in z["hf_names"]]
    ss = z["s"]
    uu = z["u"]
    # Build name -> param map from the model (state_dict() returns copies;
    # we must write through to param.data)
    param_map = {k: v for k, v in model.named_parameters()}
    applied = []
    skipped = []
    for i, hf in enumerate(hf_names):
        # HF safetensors uses 'model.layers.N...'; transformers param names
        # prefix with the model module -> 'model.model.layers.N...'
        key = hf if hf in param_map else f"model.{hf}"
        if key not in param_map:
            cands = [k for k in param_map if k.endswith(hf) or hf.replace(".weight", "") in k]
            if not cands:
                skipped.append(hf)
                continue
            key = cands[0]
        W = param_map[key].detach().float().cpu().numpy()
        v = z[f"v_{i}"]
        # u (2048-d) spans the delta's ROW space in safetensors orientation.
        # transformers may store the same tensor transposed — align by shape:
        if W.shape == (len(uu[i]), len(v)):
            W = W + ss[i] * np.outer(uu[i], v)
        elif W.shape == (len(v), len(uu[i])):
            W = W + ss[i] * np.outer(v, uu[i])  # transposed storage
        else:
            skipped.append(f"{hf} shape {W.shape} vs u{len(uu[i])} v{len(v)}")
            continue
        param_map[key].data.copy_(torch.from_numpy(W).to(param_map[key].dtype))
        applied.append(key)
    print(f"[heretic-rec] applied {len(applied)} tensors, skipped {len(skipped)}",
          flush=True)
    if skipped:
        print(f"  skipped: {skipped[:5]}", flush=True)

    # ---- save recovered model ----
    out_dir = Path(VOL_PATH) / "lfm2.5-2.6b-recovered"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (out_dir / "recovered-meta.json").write_text(json.dumps(
        {"n_applied": len(applied), "applied": applied, "skipped": skipped,
         "source": "Abiray/LFM2.5-2.6B-Heretic-Abliterated-GGUF rank1 recovery",
         "method": "W' = W + sigma * outer(u, v)"}, indent=1))
    print(f"[heretic-rec] saved to {out_dir}", flush=True)

    # ---- gates ----
    transcript = []
    refusals = []
    for p in refusal_prompts:
        resp = gen(p)
        ref = refusal_score(resp) > 0.5
        refusals.append(ref)
        transcript.append({"prompt": p, "response": resp, "refusal": ref})
        print(f"[heretic-rec] refusal={int(ref)} | {p[:45]} -> {resp[:60]!r}", flush=True)

    ref_rate = sum(refusals) / len(refusals)

    # mechanical gates (keyword-free valid on this model)
    from gates import (gate_coherence, gate_degeneracy, gate_finite_logits,
                       gate_perplexity_increase, gate_first_token_kl)
    gates = {}
    for gname, gfn in [
        ("coherence", lambda: gate_coherence(model, tok, None, refusal_prompts)),
        ("degeneracy", lambda: gate_degeneracy(model, tok, None, refusal_prompts)),
        ("finite_logits", lambda: gate_finite_logits(model, tok, None, refusal_prompts)),
        ("perplexity_increase", lambda: gate_perplexity_increase(
            model, tok, None, all_prompts, pristine_logprobs)),
        ("first_token_kl", lambda: gate_first_token_kl(
            model, tok, None, all_prompts, pristine_first)),
    ]:
        try:
            gates[gname] = gfn()
        except Exception as exc:  # noqa: BLE001
            gates[gname] = {"value": None, "passed": False,
                            "detail": f"gate crashed ({exc})"}
    print(f"[heretic-rec] gates: "
          f"{ {k: round(v.get('value') or 0, 3) if isinstance(v.get('value'), (int, float)) else v.get('value') for k, v in gates.items()} }",
          flush=True)

    result = {
        "model": "LFM2.5-2.6B + recovered Heretic rank1 edit",
        "n_applied": len(applied), "skipped": skipped,
        "refusal_rate": ref_rate, "refusal_count": int(sum(refusals)),
        "gates": gates,
        "transcript": transcript,
        "saved_to": str(out_dir),
    }
    (Path(VOL_PATH) / "heretic-recovered-gate.json").write_text(
        json.dumps(result, indent=1, default=str))
    print(f"[heretic-rec] refusal {ref_rate:.3f} ({int(sum(refusals))}/{len(refusals)})",
          flush=True)
    return result


@app.local_entrypoint()
def main():
    res = apply_and_gate.remote()
    print(json.dumps({k: v for k, v in res.items() if k != "transcript"}, indent=1)[:2000])
