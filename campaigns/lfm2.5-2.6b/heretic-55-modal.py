"""Round 3 — Heretic LFM2.5-2.6B GGUF adversarial test (WS3).

Tests Abiray's third-party 2.6B abliteration claim against OUR live-judge
gate. Loads the Q8_0 GGUF via llama-cpp-python, generates the 55 harmful +
40 harmless gate prompts, saves transcripts in the campaign's format.

Usage (Modal GPU — CUDA llama-cpp build):
    ABSOLVER_GPU=L4 modal run campaigns/lfm2.5-2.6b/heretic-55-modal.py

Usage (local CPU — needs llama-cpp-python + the GGUF on disk):
    python3 campaigns/lfm2.5-2.6b/heretic-55-modal.py --local
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import modal

_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
_CAMP = Path(__file__).resolve().parent

GGUF_REPO = "Abiray/LFM2.5-2.6B-Heretic-Abliterated-GGUF"
GGUF_FILE = "LFM2.5-2.6B-heretic-Q8_0.gguf"
MAX_NEW = 1536  # Heretic emits visible CoT + outline before detailed content; 512 still truncated mid-instruction. 1536 lets responses complete.
N_CTX = 4096

GPU = os.environ.get("ABSOLVER_GPU", "L4")
app = modal.App("absolver-26b-heretic-55")
volume = modal.Volume.from_name("absolver-phase2", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "llama-cpp-python>=0.3.0", "huggingface-hub>=0.20",
        "numpy>=1.24", "requests>=2.31",
        index_url="https://abetlen.github.io/llama-cpp-python/whl/cu124",
        extra_index_url="https://pypi.org/simple",
    )
    .env({"PYTHONPATH": "/absolver"})
    .add_local_dir(str(_PROJECT_DIR), remote_path="/absolver",
                   ignore=[".venv", ".git", "__pycache__", "*.pyc",
                           ".aiwg", "abliterated_models", "experiments",
                           "abl-work", "campaigns"])
)


@app.function(image=image, gpu=GPU, timeout=10800, retries=0,
              volumes={"/out": volume})
def run_heretic_55() -> dict:
    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download

    sys.path.insert(0, "/absolver")
    from prompts import EXPANDED_HARMFUL, EXPANDED_HARMLESS

    t0 = time.time()
    path = hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE)
    print(f"[heretic-55] GGUF at {path}", flush=True)
    llm = Llama(model_path=path, n_ctx=N_CTX, n_gpu_layers=99,
                verbose=False)

    def gen(p: str) -> str:
        out = llm.create_chat_completion(
            messages=[{"role": "user", "content": p}],
            max_tokens=MAX_NEW, temperature=0.0,
        )
        return out["choices"][0]["message"]["content"]

    include_benign = os.environ.get("HERETIC_HARMLESS", "1") != "0"
    out_root = Path("/out/lfm2.5-2.6b-transfer")
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt = out_root / "heretic-55_checkpoint.json"

    # Checkpoint-aware generation: save after EVERY prompt so a timeout
    # (or crash) never loses completed work — relaunch resumes where it left off.
    full_rows: list[dict] = []
    rows_b: list[dict] = []
    if ckpt.exists():
        try:
            old = json.loads(ckpt.read_text())
            full_rows = [r for r in old.get("heretic-q8", [])
                         if r.get("response") and len(r.get("response", "")) > 800]
            rows_b = [r for r in old.get("heretic-harness", [])
                      if r.get("response")]
            print(f"[heretic-55] checkpoint: {len(full_rows)} harmful done, "
                  f"resuming", flush=True)
        except Exception:
            print("[heretic-55] checkpoint unreadable, starting fresh",
                  flush=True)

    for p in EXPANDED_HARMFUL:
        if any(r.get("prompt") == p for r in full_rows):
            continue
        print(f"[heretic-55] gen ({len(full_rows)+1}/{len(EXPANDED_HARMFUL)}): "
              f"{p[:50]}...", flush=True)
        resp = gen(p)
        full_rows.append({"gate": "judge_refusal", "prompt": p, "response": resp})
        ckpt.write_text(json.dumps(
            {"heretic-q8": full_rows, "heretic-harness": rows_b}, indent=2))

    if include_benign:
        for p in EXPANDED_HARMLESS:
            if any(r.get("prompt") == p for r in rows_b):
                continue
            print(f"[heretic-55] gen benign: {p[:50]}...", flush=True)
            resp = gen(p)
            rows_b.append({"gate": "benign_control", "prompt": p, "response": resp})
            ckpt.write_text(json.dumps(
                {"heretic-q8": full_rows, "heretic-harness": rows_b}, indent=2))

    rows_h = full_rows
    trans = {"heretic-q8": rows_h, "heretic-harness": rows_b}

    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    (out_root / f"heretic-55_transcripts_{ts}.json").write_text(
        json.dumps(trans, indent=2))
    (out_root / "heretic-55_transcripts.json").write_text(
        json.dumps(trans, indent=2))
    print(f"[heretic-55] {len(rows_h)} harmful + {len(rows_b)} harmless "
          f"({time.time()-t0:.0f}s)", flush=True)
    return {"n_harmful": len(rows_h), "n_harmless": len(rows_b),
            "path": "/out/lfm2.5-2.6b-transfer/heretic-55_transcripts.json"}


@app.local_entrypoint()
def main() -> None:
    res = run_heretic_55.remote()
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    # Local CPU fallback path (no Modal)
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true",
                    help="run on local CPU (needs llama-cpp-python installed)")
    args = ap.parse_args()
    if args.local:
        from llama_cpp import Llama

        sys.path.insert(0, str(_PROJECT_DIR))
        from prompts import EXPANDED_HARMFUL, EXPANDED_HARMLESS

        gguf_path = _CAMP / ".." / ".." / "heretic-gguf" / GGUF_FILE
        llm = Llama(model_path=str(gguf_path), n_ctx=N_CTX,
                    n_gpu_layers=0, n_threads=6, verbose=False)

        def gen(p: str) -> str:
            out = llm.create_chat_completion(
                messages=[{"role": "user", "content": p}],
                max_tokens=MAX_NEW, temperature=0.0,
            )
            return out["choices"][0]["message"]["content"]

        t0 = time.time()
        rows_h = [{"gate": "judge_refusal", "prompt": p, "response": gen(p)}
                  for p in EXPANDED_HARMFUL]
        rows_b = [{"gate": "benign_control", "prompt": p, "response": gen(p)}
                  for p in EXPANDED_HARMLESS]
        trans = {"heretic-q8": rows_h, "heretic-harness": rows_b}
        out_path = _CAMP / "heretic-55_transcripts.json"
        out_path.write_text(json.dumps(trans, indent=2))
        print(f"[heretic-55-local] {len(rows_h)}+{len(rows_b)} -> {out_path} "
              f"({time.time()-t0:.0f}s)")
    # else: nothing — `modal run` discovers @app functions directly
