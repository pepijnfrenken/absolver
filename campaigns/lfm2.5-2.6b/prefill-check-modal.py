"""Verify the affirm-prefill condition actually flips the 2.6B into
compliance (validates the paired actdiff directions)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal
import torch

_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

app = modal.App("absolver-26b-prefillcheck")

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

GPU = "L4"
TIMEOUT = 1500
VOLUME = modal.Volume.from_name("absolver-phase2", create_if_missing=True)
MODEL_ID = "LiquidAI/LFM2.5-2.6B"


@app.function(image=image, gpu=GPU, timeout=TIMEOUT, retries=0,
              secrets=[modal.Secret.from_name("hf-write-token")],
              volumes={"/out": VOLUME})
def check() -> dict:
    sys.path.insert(0, "/absolver")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prompts import DEFAULT_HARMFUL

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True).to("cuda")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    def gen(text: str) -> str:
        inp = tok(text, return_tensors="pt", truncation=True, max_length=160).to("cuda")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=48, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)

    rows = []
    for p in DEFAULT_HARMFUL[:5]:
        unprimed = gen(p)
        prefilled = gen(f"{p} Sure, I can help with that.")
        rows.append({"prompt": p, "unprimed": unprimed, "prefilled": prefilled,
                     "unprimed_n": len(unprimed.split()),
                     "prefilled_n": len(prefilled.split())})
        print(f"\n--- {p[:50]}")
        print(f"  UNPRIMED ({len(unprimed.split())}w): {unprimed[:180]}")
        print(f"  PREFILL ({len(prefilled.split())}w): {prefilled[:180]}")

    out_path = Path("/out/lfm2.5-2.6b-transfer")
    out_path.mkdir(parents=True, exist_ok=True)
    fp = out_path / f"prefill-check_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}.json"
    fp.write_text(json.dumps(rows, indent=2))
    return {"path": str(fp), "rows": rows}


@app.local_entrypoint()
def main():
    res = check.remote()
    print(json.dumps(res, indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()
