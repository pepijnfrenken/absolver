"""Modal runner: lm-eval benchmark replication, pristine vs ablated LFM2.5-1.2B.

Replicates the runnable subset of LiquidAI's posted benchmark suite
(https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) on BOTH
`LiquidAI/LFM2.5-1.2B-Instruct` (pristine) and
`PinoCookie/LFM2.5-1.2B-Instruct-Abliterated` (ablated) under identical
conditions (same lm-eval version, seed, tasks, gen_kwargs, batch).

Tasks verified present in lm_eval 0.4.13:
  mmlu_pro            — knowledge retrieval (TIGER-Lab/MMLU-Pro, 14 subjects)
  gpqa_diamond_n_shot — reasoning, GPQA-diamond (Idavidrein/gpqa, 5-shot like AA)
  aime25              — math reasoning (math-ai/aime25, 30 problems)
  ifeval              — instruction following (google/IFEval, 541 prompts)
BFCLv3 / Multi-IF / IFBench are NOT in lm-eval (custom harnesses) — reported
as "not replicated" with the reason. No fake numbers.

Results: JSON logs + per-run meta written to the absolver-phase2 volume at
`/out/benchmarks/<tag>/<task>/results_<ts>.json` and a combined
`/out/benchmarks/<tag>/results.json`. Pull single files locally with
`modal volume get absolver-phase2 benchmarks/<tag>/results.json <local>`.

Usage:
  modal run benchmarks/run_lmeval_modal.py --model both --tasks mmlu_pro,gpqa_diamond_n_shot,aime25,ifeval
  modal run benchmarks/run_lmeval_modal.py --model both --tasks mmlu_pro --limit 3000 --tag mmlu_pro_subset
  modal run benchmarks/run_lmeval_modal.py --push-readme /path/README.md --commit "update card"
"""
import json
import os
import sys
import time
from pathlib import Path

import modal  # type: ignore (only imported under `modal run`)

_PROJECT_DIR = Path(__file__).resolve().parent.parent

app = modal.App("absolver-benchmarks")

MODELS = {
    "pristine": "LiquidAI/LFM2.5-1.2B-Instruct",
    "ablated": "PinoCookie/LFM2.5-1.2B-Instruct-Abliterated",
}

# Version pins mirror the proven harness image (transformers 5.14.1 /
# torch 2.13.0 are what the harness containers resolve today; LFM2.5 is a
# native arch in transformers 5.x — no trust_remote_code needed).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "lm_eval==0.4.13",
        "torch==2.13.0",
        "transformers==5.14.1",
        "accelerate",
        "datasets",
        "huggingface-hub",
        "safetensors",
        # lm-eval's ifeval task imports these at module load
        "langdetect",
        "immutabledict",
        "nltk",
    )
    .env(
        {
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

GPU = os.environ.get("ABSOLVER_GPU", "L4")
TIMEOUT = int(os.environ.get("ABSOLVER_TIMEOUT", str(8 * 3600)))
SEED = 1234

OUT_VOLUME = modal.Volume.from_name("absolver-phase2", create_if_missing=True)
HF_SECRET = modal.Secret.from_name("hf-write-token")


def _hf_token() -> str:
    """Write token from the Modal secret (HF_WRITE_TOKEN or HF_TOKEN fallback)."""
    tok = os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN", "")
    if not tok:
        raise RuntimeError("no HF write token in environment")
    return tok


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT,
    retries=2,  # L4 pool preempts sporadically; auto-restart on preemption
    secrets=[HF_SECRET],
    volumes={"/out": OUT_VOLUME},
)
def run_eval(model_id: str, tasks: list[str], tag: str,
             limit: int | None = None, seed: int = SEED,
             num_fewshot: int | None = None,
             batch_size: str = "auto") -> dict:
    """Run the lm-eval suite on ONE model; write JSON logs to the volume."""
    import lm_eval
    from lm_eval import simple_evaluate
    import torch
    import transformers

    out_root = Path(f"/out/benchmarks/{tag}")
    out_root.mkdir(parents=True, exist_ok=True)

    started = time.time()
    results = simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_id},trust_remote_code=True,dtype=bfloat16",
        tasks=tasks,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
        fewshot_random_seed=seed,
        num_fewshot=num_fewshot,
        limit=limit,
        batch_size=batch_size,
        log_samples=False,
        confirm_run_unsafe_code=True,
    )
    elapsed = time.time() - started

    # ---- compact per-task summary ----
    summary = {}
    for task_name in tasks:
        if task_name in results.get("results", {}):
            task_res = results["results"][task_name]
            kept = {k: v for k, v in task_res.items()
                    if k not in ("alias", "samples")}
            summary[task_name] = kept
            print(f"[{tag}] {task_name}: "
                  f"{json.dumps(kept, indent=2, default=str)}", flush=True)

    meta = {
        "model_id": model_id,
        "tag": tag,
        "tasks": tasks,
        "limit": limit,
        "seed": seed,
        "batch_size": results.get("config", {}).get("batch_size"),
        "lm_eval_version": lm_eval.__version__,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "elapsed_sec": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    payload = {
        "meta": meta,
        "results": results.get("results", {}),
        "config": results.get("config", {}),
        "groups": results.get("groups", {}),
        "versions": results.get("versions", {}),
    }
    # default=str: lm-eval config objects contain torch/numpy dtypes that
    # plain json.dumps rejects; scores themselves are always floats/ints
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    per_run = out_root / f"results_{ts}.json"
    per_run.write_text(json.dumps(payload, indent=2, default=str))
    combined = out_root / "results.json"
    combined.write_text(json.dumps(payload, indent=2, default=str))

    print(f"[{tag}] wrote {per_run} and {combined} ({elapsed:.0f}s)", flush=True)
    return {"tag": tag, "model_id": model_id, "summary": summary,
            "elapsed_sec": elapsed, "log_path": str(combined)}


@app.function(
    image=image,
    secrets=[HF_SECRET],
    timeout=900,
    retries=0,
)
def push_readme(readme_content: str, commit_message: str, repo_id: str) -> dict:
    """Push README.md to the HF repo with the write token; verify afterwards."""
    from huggingface_hub import HfApi

    token = _hf_token()
    api = HfApi(token=token)

    url = api.upload_file(
        path_or_fileobj=readme_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    print("uploaded:", url, flush=True)

    # ---- verification: raw README hash + file listing ----
    files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    print("repo files:", sorted(files), flush=True)

    raw_path = api.hf_hub_download(repo_id=repo_id, filename="README.md",
                                   repo_type="model", token=token)
    raw = Path(raw_path).read_text(encoding="utf-8")
    import hashlib
    sent_sha = hashlib.sha256(readme_content.encode("utf-8")).hexdigest()
    got_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    match = sent_sha == got_sha

    info = api.model_info(repo_id=repo_id)
    print(f"README.md sha256 match: {match} (sent={sent_sha[:12]} got={got_sha[:12]})",
          flush=True)
    return {
        "repo_id": repo_id,
        "url": url,
        "files": sorted(files),
        "README_sha256_match": match,
        "sent_sha256": sent_sha,
        "got_sha256": got_sha,
        "last_modified": str(info.lastModified),
    }


@app.local_entrypoint()
def main(model: str = "both",
         tasks: str = "mmlu_pro,gpqa_diamond_n_shot,aime25,ifeval",
         limit: int | None = None,
         num_fewshot: int | None = None,
         batch_size: str = "auto",
         tag: str | None = None,
         push_readme_path: str | None = None,
         commit: str = "model card: benchmark replication + capability-hit analysis",
         repo_id: str = "PinoCookie/LFM2.5-1.2B-Instruct-Abliterated") -> None:
    import json as _json
    import sys as _sys

    if push_readme_path:
        content = Path(push_readme_path).read_text(encoding="utf-8")
        res = push_readme.remote(content, commit, repo_id)
        print(_json.dumps(res, indent=2))
        _sys.exit(0 if res.get("README_sha256_match") else 1)

    if model == "both":
        chosen = list(MODELS)
    else:
        chosen = [m for m in (model.split(",") if "," in model else [model])
                  if m in MODELS]
        if not chosen:
            raise SystemExit(f"--model must be one of {list(MODELS)}")
    task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    outs = []
    for m in chosen:
        # model key always in the tag: two models must never share a path
        mtag = f"{m}_{tag}" if tag else f"{m}_{stamp}"
        outs.append(run_eval.spawn(
            MODELS[m], task_list, mtag, limit=limit, seed=SEED,
            num_fewshot=num_fewshot, batch_size=batch_size))
    for o in outs:
        res = o.get()
        print("#" * 70)
        print(_json.dumps(res, indent=2))
        print("#" * 70)


if __name__ == "__main__":
    main()