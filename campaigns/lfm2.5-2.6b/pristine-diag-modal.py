"""Diagnostic: dump verbatim pristine responses of LFM2.5-2.6B on a sample
of harmful + harmless prompts, plus keyword-refusal + judge classification.

Purpose: interpret the surprising 0/55 pristine refusal (quick sweep) —
is the model genuinely non-refusing on these prompts, does its refusal
style evade the keyword classifier, or is generation degenerate?
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal
import torch

_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

app = modal.App("absolver-26b-diag")

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

GPU = "L4"
TIMEOUT = 1800
VOLUME = modal.Volume.from_name("absolver-phase2", create_if_missing=True)
MODEL_ID = "LiquidAI/LFM2.5-2.6B"


@app.function(image=image, gpu=GPU, timeout=TIMEOUT, retries=0,
              secrets=[modal.Secret.from_name("hf-write-token")],
              volumes={"/out": VOLUME})
def dump_responses(limit: int = 12) -> dict:
    sys.path.insert(0, "/absolver")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prompts import DEFAULT_HARMFUL, DEFAULT_HARMLESS
    from verify import _model_device, _response_is_refusal
    from judge import _keyword_refusal_score
    from prompt_format import format_prompt, resolve_flavor

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True).to("cuda")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    fmt = resolve_flavor(tok, "chat")
    dev = _model_device(model)

    def gen(p: str, max_new: int = 128) -> str:
        formatted = format_prompt(tok, p, fmt)
        inp = tok(formatted, return_tensors="pt", truncation=True,
                  max_length=1024).to(dev)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new,
                                 do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        gen_ids = out[0][inp["input_ids"].shape[1]:]
        return tok.decode(gen_ids, skip_special_tokens=True)

    rows = []
    for p in DEFAULT_HARMFUL[:limit]:
        resp = gen(p)
        rows.append({
            "prompt": p,
            "response": resp,
            "kw_score": _keyword_refusal_score(resp),
            "is_refusal": _response_is_refusal(resp),
            "n_words": len(resp.split()),
        })
    for p in DEFAULT_HARMLESS[:4]:
        resp = gen(p)
        rows.append({
            "prompt": p, "response": resp, "benign": True,
            "kw_score": _keyword_refusal_score(resp),
            "n_words": len(resp.split()),
        })

    out_path = Path("/out/lfm2.5-2.6b-transfer")
    out_path.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    fp = out_path / f"pristine-responses_{ts}.json"
    fp.write_text(json.dumps(rows, indent=2))
    print(f"saved {fp}")
    for r in rows:
        tag = "REFUSE" if r.get("is_refusal") else \
              ("BENIGN" if r.get("benign") else "COMPLY")
        print(f"\n--- [{tag}] kw={r['kw_score']:.2f} "
              f"({r.get('n_words', '?')}w) {r['prompt'][:70]}")
        print(f"    {r['response'][:280]}")
    return {"path": str(fp), "n": len(rows)}


@app.local_entrypoint()
def main(limit: int = 12):
    print("dumping pristine responses...")
    res = dump_responses.remote(limit=limit)
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
