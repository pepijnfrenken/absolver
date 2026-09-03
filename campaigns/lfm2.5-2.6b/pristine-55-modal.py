"""Generate pristine LFM2.5-2.6B transcripts over EXPANDED_HARMFUL (55)
for the Round-2 live LLM judge (WS1 control set).

Round 1 never saved pristine transcripts at gate size (only 15-16 samples
for classifier mining). This runner produces the full 55 so the live judge
can validate its instrument: pristine must judge as ~55/55 refusal.

Output: /out/lfm2.5-2.6b-transfer/pristine-55_transcripts_{ts}.json and a
stable copy pristine-55_transcripts.json (overwritten on purpose).

Usage:
    ABSOLVER_GPU=L4 modal run campaigns/lfm2.5-2.6b/pristine-55-modal.py
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

app = modal.App("absolver-26b-pristine-55")

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


@app.function(image=image, gpu=GPU, timeout=TIMEOUT, retries=0,
              volumes={"/out": VOLUME})
def run_pristine_55() -> dict:
    sys.path.insert(0, "/absolver")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prompts import EXPANDED_HARMFUL, EXPANDED_HARMLESS
    from verify import _model_device

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True).to("cuda")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    dev = _model_device(model)

    def chat_text(p: str) -> str:
        return tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False,
            add_generation_prompt=True)

    def gen(p: str, max_new: int = 128) -> str:
        inp = tok(chat_text(p), return_tensors="pt", truncation=True,
                  max_length=1024).to(dev)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        return tok.decode(out[0][inp["input_ids"].shape[1]:],
                          skip_special_tokens=True)

    rows_h = [{"gate": "judge_refusal", "prompt": p, "response": gen(p)}
              for p in EXPANDED_HARMFUL]
    rows_b = [{"gate": "benign_control", "prompt": p, "response": gen(p)}
              for p in EXPANDED_HARMLESS]
    trans = {"pristine": rows_h, "pristine-harness": rows_b}

    out_root = Path("/out/lfm2.5-2.6b-transfer")
    out_root.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    (out_root / f"pristine-55_transcripts_{ts}.json").write_text(
        json.dumps(trans, indent=2))
    (out_root / "pristine-55_transcripts.json").write_text(
        json.dumps(trans, indent=2))
    print(f"[pristine-55] {len(rows_h)} harmful + {len(rows_b)} harmless "
          f"({time.time()-t0:.0f}s)", flush=True)
    return {"n_harmful": len(rows_h), "n_harmless": len(rows_b),
            "path": str(out_root / "pristine-55_transcripts.json")}


@app.local_entrypoint()
def main():
    res = run_pristine_55.remote()
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()