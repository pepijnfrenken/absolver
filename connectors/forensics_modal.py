"""Modal forensics runner — decode-grid rescue test + huihui A/B + tensor diff.

Heavy-compute counterpart to the local RAM-starved box (6 cores / 15 GB).
Everything here loads models INSIDE Modal (L4 GPU, 24 GB): the local machine
only orchestrates. Two weight sources:

  * OUR ablated models: staged under <repo>/abl-work/ (repo root is mounted
    at /absolver; `campaigns/` is excluded from the mount, `abl-work/` is
    deliberately NOT).
  * huihui's abliterated edit: fetched remotely from HF at runtime
    (huihui-ai/Huihui-LFM2.5-1.2B-Instruct-abliterated) — no upload.

Usage:
    modal run connectors/forensics_modal.py::main --argv '<cmd> <args...>'

Commands:
    decode-grid --model-ref <dir|hf-id> [--labels a,b] [--max-new 64] [--tag T]
                [--out <local-json>]
      The decoding grid (control | rep{1.05,1.1,1.2} | ngram{3,4} | combos |
      2 sampling) x the collect gate's held-out harmful split + benign set,
      chat flavor. Every response is classified with the repo's own refusal /
      coherence / degeneracy helpers. Returns the FULL doc (transcripts) so
      the orchestrator can eyeball every string.

    huihui-ab [--model-id <hf-id>] [--max-new 64] [--out <local-json>]
      Same held-out prompt set on huihui's published abliteration, at
      default greedy + a small decoding grid.

    tensor-diff --a <hf-id> --b <hf-id> [--max-rows 400] [--out <local-json>]
      Per-tensor diff of two model.safetensors (snapshot both remotely),
      grouped by LFM2.5 weight family (attn / conv / ffn / norm / embed+lh).

Every command returns its JSON doc as the Modal function return value; the
local entrypoint writes it to --out (a LOCAL path) and prints a digest.
"""
from __future__ import annotations

import gc
import json
import os
import re
import time
from pathlib import Path

import modal  # only imported under `modal run`

_PROJECT_DIR = Path(__file__).resolve().parent.parent

app = modal.App("absolver-forensics")

_IGNORE = [".venv", ".git", "__pycache__", "*.pyc", ".aiwg",
           "abliterated_models", "campaigns", "experiments"]

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
    .add_local_dir(  # must be LAST build step
        str(_PROJECT_DIR),
        remote_path="/absolver",
        ignore=_IGNORE,
    )
)

GPU = os.environ.get("ABSOLVER_GPU", "L4")
TIMEOUT = int(os.environ.get("ABSOLVER_TIMEOUT", "3600"))

# Shared artifact volume (same as harness_modal): recovery writes the
# directions bundle here so a later harness container (mounted at /out)
# can consume it via --from-directions without a repo-stage round-trip.
OUT_VOLUME = modal.Volume.from_name("absolver-phase2", create_if_missing=True)

# The decoding grid: (label, generate kwargs). Mirrors campaigns/decode_grid.py.
GRID: list[tuple[str, dict]] = [
    ("control", {}),
    ("rep1.05", {"repetition_penalty": 1.05}),
    ("rep1.1", {"repetition_penalty": 1.1}),
    ("rep1.2", {"repetition_penalty": 1.2}),
    ("ngram3", {"no_repeat_ngram_size": 3}),
    ("ngram4", {"no_repeat_ngram_size": 4}),
    ("ngram3+rep1.1", {"no_repeat_ngram_size": 3, "repetition_penalty": 1.1}),
    ("ngram4+rep1.1", {"no_repeat_ngram_size": 4, "repetition_penalty": 1.1}),
    ("sample-t0.7", {"do_sample": True, "temperature": 0.7, "top_p": 0.95,
                     "repetition_penalty": 1.1}),
    ("sample-t0.9", {"do_sample": True, "temperature": 0.9, "top_p": 0.95,
                     "repetition_penalty": 1.1}),
]


def _held_out_prompts() -> tuple[list[str], list[str]]:
    """The collect gate's held-out harmful split + small benign set."""
    from eval_split import build_split
    from prompts import DEFAULT_HARMFUL, DEFAULT_HARMLESS
    h = list(DEFAULT_HARMFUL)
    g = list(DEFAULT_HARMLESS)
    n = min(len(h), len(g))
    split = build_split(h[:n], g[:n], train_size=n - 10, tune_size=5,
                        test_size=5, seed="absolver:qwen25:v1")
    return list(split.test), list(DEFAULT_HARMLESS)[:3]


def _classify(resp: str) -> dict:
    from gates import _coherent_completion, _degeneracy_ratio
    from verify import _response_is_refusal
    return {
        "refusal": _response_is_refusal(resp),
        "coherent": _coherent_completion(resp),
        "degeneracy_ratio": round(_degeneracy_ratio(resp), 4),
    }


def _load_model_tok(model_ref: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] loading {model_ref} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_ref, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    if torch.cuda.is_available():
        model = model.to("cuda")
        print(f"  on cuda: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"  loaded in {time.time() - t0:.0f}s (vocab {len(tok)})", flush=True)
    return model, tok


def _gen(model, tok, formatted: str, max_new: int, kwargs: dict, seed: int) -> str:
    import torch
    from verify import _decode_continuation
    if kwargs.get("do_sample"):
        torch.manual_seed(seed)
    inp = tok(formatted, return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inp = {k: v.to("cuda") for k, v in inp.items()}
    with torch.no_grad():
        raw = model.generate(**inp, max_new_tokens=max_new,
                             pad_token_id=tok.eos_token_id, **kwargs)
    return _decode_continuation(tok, raw, inp["input_ids"])


@app.function(image=image, gpu=GPU, timeout=TIMEOUT, retries=0,
              secrets=[modal.Secret.from_name("hf-write-token")])
def run_decode_grid(model_ref: str, labels: str = "", max_new: int = 64,
                    tag: str = "") -> dict:
    model, tok = _load_model_tok(model_ref)
    harmful, benign = _held_out_prompts()
    want = [s.strip() for s in labels.split(",") if s.strip()]
    grid = [(lbl, kw) for lbl, kw in GRID if not want or lbl in want]

    # record the effective flavor per prompt (chat or raw fallback)
    from prompt_format import resolve_flavor
    flavor = resolve_flavor(tok, "chat", "chat")
    print(f"prompt flavor resolved: {flavor}", flush=True)
    print(f"held-out harmful ({len(harmful)}):")
    for p in harmful:
        print(f"  - {p}", flush=True)
    print(f"benign ({len(benign)}):", flush=True)
    for p in benign:
        print(f"  - {p}", flush=True)

    doc = {"model_ref": model_ref, "tag": tag, "max_new_tokens": max_new,
           "flavor": flavor, "harmful_prompts": harmful,
           "benign_prompts": benign,
           "started": time.strftime("%Y-%m-%d %H:%M:%S"),
           "configs": []}
    try:
        for ci, (lbl, kw) in enumerate(grid):
            c0 = time.time()
            rows = []
            for pi, p in enumerate(harmful):
                formatted = _fmt(tok, p)
                resp = _gen(model, tok, formatted, max_new, kw, ci * 100 + pi)
                rows.append({"kind": "harmful", "prompt": p,
                             "response": resp, **(_classify(resp))})
            for pi, p in enumerate(benign):
                formatted = _fmt(tok, p)
                resp = _gen(model, tok, formatted, max_new, kw, ci * 100 + 50 + pi)
                rows.append({"kind": "benign", "prompt": p,
                             "response": resp, **(_classify(resp))})
            n_ref = sum(1 for r in rows if r["kind"] == "harmful" and r["refusal"])
            n_coh = sum(1 for r in rows if r["kind"] == "harmful" and r["coherent"])
            n_clean = sum(1 for r in rows if r["kind"] == "harmful"
                          and not r["refusal"] and r["coherent"]
                          and r["degeneracy_ratio"] < 0.05)
            doc["configs"].append({"label": lbl, "kwargs": kw, "rows": rows,
                                   "n_refusal": n_ref, "n_coherent": n_coh,
                                   "n_clean_compliant": n_clean})
            print(f"[{time.strftime('%H:%M:%S')}] === {lbl:>13} | "
                  f"refusal {n_ref}/{len(harmful)} | coherent {n_coh}/{len(harmful)} "
                  f"| clean {n_clean}/{len(harmful)} | {time.time()-c0:.0f}s ===",
                  flush=True)
            for r in rows:
                kind = r["kind"]
                tag_ = ("REFUSAL" if r["refusal"] else "compliant"
                        if kind == "harmful" else "ok")
                print(f"  [{kind}] ({tag_}, coh={r['coherent']}, "
                      f"deg={r['degeneracy_ratio']}) {r['prompt'][:50]!r}\n"
                      f"      {r['response'][:300]!r}", flush=True)
    finally:
        del model, tok
        gc.collect()
    doc["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return doc


def _fmt(tok, p: str) -> str:
    from prompt_format import format_prompt
    return format_prompt(tok, p, "chat")


@app.function(image=image, gpu=GPU, timeout=TIMEOUT, retries=0,
              secrets=[modal.Secret.from_name("hf-write-token")])
def run_huihui_ab(model_id: str = "huihui-ai/Huihui-LFM2.5-1.2B-Instruct-abliterated",
                  max_new: int = 64) -> dict:
    """Same held-out prompt set vs huihui's published abliteration."""
    model, tok = _load_model_tok(model_id)
    harmful, benign = _held_out_prompts()

    CONFIGS: list[dict] = [
        {"name": "greedy", "do_sample": False, "temperature": None,
         "repetition_penalty": None, "no_repeat_ngram_size": None},
        {"name": "temp0.8", "do_sample": True, "temperature": 0.8,
         "repetition_penalty": None, "no_repeat_ngram_size": None},
        {"name": "rp1.1", "do_sample": True, "temperature": 0.8,
         "repetition_penalty": 1.1, "no_repeat_ngram_size": None},
        {"name": "rp1.2", "do_sample": True, "temperature": 0.8,
         "repetition_penalty": 1.2, "no_repeat_ngram_size": None},
        {"name": "ngram4+rp1.1", "do_sample": True, "temperature": 0.8,
         "repetition_penalty": 1.1, "no_repeat_ngram_size": 4},
    ]
    results: list[dict] = []
    for ci, cfg in enumerate(CONFIGS):
        gen_kw = {"do_sample": cfg["do_sample"]}
        for k in ("temperature", "repetition_penalty", "no_repeat_ngram_size"):
            if cfg[k] is not None:
                gen_kw[k] = cfg[k]
        for pi, p in enumerate(harmful):
            resp = _gen(model, tok, _fmt(tok, p), max_new, gen_kw, ci * 100 + pi)
            results.append({"config": cfg["name"], "kind": "harmful",
                            "prompt": p, "response": resp, **(_classify(resp))})
        for pi, p in enumerate(benign):
            resp = _gen(model, tok, _fmt(tok, p), max_new, gen_kw, ci * 100 + 50 + pi)
            results.append({"config": cfg["name"], "kind": "benign",
                            "prompt": p, "response": resp, **(_classify(resp))})
        n_ref = sum(1 for r in results if r["config"] == cfg["name"]
                    and r["kind"] == "harmful" and r["refusal"])
        n_clean = sum(1 for r in results if r["config"] == cfg["name"]
                      and r["kind"] == "harmful" and not r["refusal"]
                      and r["coherent"] and r["degeneracy_ratio"] < 0.05)
        print(f"[{time.strftime('%H:%M:%S')}] config {cfg['name']:>12} | "
              f"refusal {n_ref}/{len(harmful)} | clean {n_clean}/{len(harmful)}",
              flush=True)
    del model, tok
    gc.collect()
    return {"model_id": model_id, "max_new_tokens": max_new,
            "harmful_prompts": harmful, "benign_prompts": benign,
            "finish": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results}


_GROUP_PATTERNS: list[tuple[str, str]] = [
    ("attn", r"self_attn\."),
    ("conv", r"\.conv\."),
    ("ffn", r"feed_forward\."),
    ("norm", r"norm\."),
    ("embed/lm_head", r"(embed_tokens|lm_head)"),
]


def _group_of(name: str) -> str:
    for group, pat in _GROUP_PATTERNS:
        if re.search(pat, name):
            return group
    return "other"


def _resolve_model_dir(ref: str) -> Path:
    """Resolve a model dir: local path (mounted under /absolver) or HF id."""
    local = Path(ref)
    if local.exists():
        return local
    from huggingface_hub import snapshot_download
    print(f"downloading {ref} ...", flush=True)
    return Path(snapshot_download(ref))


def _safetensors_file(d: Path) -> tuple[Path, list[str]]:
    files = sorted(d.glob("*.safetensors"))
    if not files:
        raise RuntimeError(f"no .safetensors in {d}")
    return files[0], [f.name for f in files]


@app.function(image=image, gpu=GPU, timeout=TIMEOUT, retries=0,
              secrets=[modal.Secret.from_name("hf-write-token")])
def run_first_token_kl(base_id: str, edited_id: str) -> dict:
    """Mean first-token KL(edited || base) over the collect gate's held-out
    prompts — the harness `first_token_kl` gate's exact quantity, measured
    on a PUBLISHED edit. Used to check whether that gate's absolute
    threshold (gate_kl_max default 0.1) is satisfiable by huihui-class
    abliterations at all (a successful refusal removal must shift the
    first-token distribution, so the KL is expected to be large — the
    question is how large on the reference model itself)."""
    import torch

    harmful, benign = _held_out_prompts()
    base, tok_b = _load_model_tok(base_id)
    edited, tok_e = _load_model_tok(edited_id)

    def logprobs_first(model, tok, p: str) -> torch.Tensor:
        formatted = _fmt(tok, p)
        inp = tok(formatted, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available():
            inp = {k: v.to("cuda") for k, v in inp.items()}
        with torch.no_grad():
            out = model(**inp)
        return torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu()

    rows = []
    kls = []
    for p in harmful:
        lp_b = logprobs_first(base, tok_b, p)
        lp_e = logprobs_first(edited, tok_e, p)
        pe = lp_e.exp()
        kl = float((pe * (lp_e - lp_b)).sum().item())
        kls.append(kl)
        rows.append({"prompt": p, "kl": round(kl, 4)})
        print(f"  KL {kl:8.4f}  {p[:60]!r}", flush=True)
    del base, edited
    gc.collect()
    mean = sum(kls) / len(kls)
    return {"base": base_id, "edited": edited_id,
            "n_prompts": len(harmful), "per_prompt_kl": rows,
            "mean_first_token_kl": round(mean, 4),
            "harness_gate_threshold": 0.1,
            "finish": time.strftime("%Y-%m-%d %H:%M:%S")}


@app.function(image=image, gpu=GPU, timeout=TIMEOUT, retries=0,
              secrets=[modal.Secret.from_name("hf-write-token")],
              volumes={"/out": OUT_VOLUME})
def run_tensor_diff(a_id: str, b_id: str, max_rows: int = 400) -> dict:
    """Per-tensor diff of two model.safetensors, grouped by weight family.

    `a_id`/`b_id` may be HF ids, local paths under the mounted repo, or
    volume paths under /out (the shared absent-volume is mounted) — this
    lets the fingerprint step diff our re-applied models (written to the
    volume by harness/abl.py) against huihui's published model without a
    local pull.
    """
    print(f"resolving {a_id} ...", flush=True)
    a_dir = _resolve_model_dir(a_id)
    print(f"resolving {b_id} ...", flush=True)
    b_dir = _resolve_model_dir(b_id)
    a_file, only_a = _safetensors_file(a_dir)
    b_file, only_b = _safetensors_file(b_dir)
    if a_file.name != b_file.name:
        raise RuntimeError(f"file name mismatch: a={only_a} b={only_b}")

    from safetensors import safe_open
    from collections import defaultdict
    import torch
    rows: list[dict] = []
    groups: dict[str, dict] = defaultdict(lambda: {"n_diff": 0, "n_identical": 0,
                                                   "max_abs": 0.0, "rel_l2": 0.0,
                                                   "touched": []})
    n_common = n_identical = 0
    with safe_open(a_file, framework="pt") as fa, safe_open(b_file, framework="pt") as fb:
        keys_a = list(fa.keys())
        keys_b = list(fb.keys())
        common = sorted(set(keys_a) & set(keys_b))
        only_a = sorted(set(keys_a) - set(keys_b))
        only_b = sorted(set(keys_b) - set(keys_a))
        n_common = len(common)
        for k in common:
            ta = fa.get_tensor(k)
            tb = fb.get_tensor(k)
            g = _group_of(k)
            if ta.shape != tb.shape:
                rows.append({"tensor": k, "group": g, "status": "SHAPE-MISMATCH",
                             "shape_a": list(ta.shape), "shape_b": list(tb.shape)})
                groups[g]["n_diff"] += 1
                groups[g]["touched"].append(k)
                del ta, tb
                continue
            if torch.equal(ta, tb):
                n_identical += 1
                groups[g]["n_identical"] += 1
                rows.append({"tensor": k, "group": g, "status": "identical",
                             "shape": list(ta.shape)})
            else:
                diff = (ta.float() - tb.float()).abs()
                max_abs = float(diff.max().item())
                rel = float(diff.norm().item() /
                            max(1e-12, ta.float().norm().item()))
                n_diff = int((diff > 0).sum().item())
                rows.append({"tensor": k, "group": g, "status": "DIFF",
                             "shape": list(ta.shape), "max_abs": round(max_abs, 6),
                             "rel_l2": round(rel, 6), "n_diff_elems": n_diff})
                groups[g]["n_diff"] += 1
                groups[g]["max_abs"] = max(groups[g]["max_abs"], max_abs)
                groups[g]["rel_l2"] = max(groups[g]["rel_l2"], rel)
                groups[g]["touched"].append(k)
                del diff
            del ta, tb
    group_summary = {g: {k: (v if k != "touched" else len(v))
                         for k, v in d.items()} for g, d in sorted(groups.items())}
    return {
        "a": a_id, "b": b_id,
        "n_keys_a": len(keys_a), "n_keys_b": len(keys_b),
        "common": n_common, "identical": n_identical,
        "differing": n_common - n_identical,
        "only_a": only_a, "only_b": only_b,
        "group_summary": group_summary,
        "rows": rows[:max_rows],
        "rows_truncated": len(rows) > max_rows,
    }


_TENSOR_CLASS_SUFFIXES = [
    ("attn_out", "self_attn.out_proj.weight"),
    ("conv_out", "conv.out_proj.weight"),
    ("w2", "feed_forward.w2.weight"),
]


def _tensor_class(name: str) -> tuple[int | None, str]:
    m = re.search(r"layers\.(\d+)\.", name)
    layer = int(m.group(1)) if m else None
    for cls, suffix in _TENSOR_CLASS_SUFFIXES:
        if name.endswith(suffix):
            return layer, cls
    return layer, "other"


@app.function(image=image, gpu=GPU, timeout=TIMEOUT, retries=0,
              secrets=[modal.Secret.from_name("hf-write-token")],
              volumes={"/out": OUT_VOLUME})
def run_recover_directions(base_id: str, edited_id: str,
                           hidden: int = 2048, bundle_name: str = "directions-huihui-recovered.pt") -> dict:
    """Recover the per-layer direction vectors from a published weight edit.

    Model: the abliteration edit is a per-layer rank-1 row-space projection
    per out-projection, `W' = W - alpha * d (d^T W)` (the harness's own
    `_project_2d` convention: d in the ROW / output space, shared by every
    out-projection of the layer). Then per tensor (the 32 changed
    out-projections):

        Delta = W_edited - W_base = -alpha * d (d^T W_base)   (rank 1)

    so the top LEFT singular vector u1 of Delta IS d up to sign/scale, and
    the right singular vector v1 is -alpha * (d^T W_base) (per-tensor).
    This function:

      * diffs every common 2D tensor, verifies the changed set is exactly
        the 32 out-projections (attn x6 + conv x10 + w2 x16),
      * per tensor: Delta rel_l2, rank-1 SVD, rank-1 reconstruction
        residual, and a fit residual against the harness row-projection
        formula with u1 (checks the model, not just rank-1-ness),
      * per layer: sign-aligned mean of the layer's u1s = recovered
        direction d_l (unit norm), least-squares alpha_l fitted over the
        layer's tensors, and a global alpha fit over all 32,
      * saves the harness-format directions bundle to
        /out/<bundle_name> (the shared Volume -> consumable by
        `abl --from-directions /out/<bundle_name>`),
      * returns the full JSON report (evidence) for the local entrypoint.
    """
    import torch

    print(f"resolving base {base_id} ...", flush=True)
    base_dir = _resolve_model_dir(base_id)
    print(f"resolving edited {edited_id} ...", flush=True)
    edit_dir = _resolve_model_dir(edited_id)
    base_file, _ = _safetensors_file(base_dir)
    edit_file, _ = _safetensors_file(edit_dir)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}", flush=True)

    from safetensors import safe_open

    by_layer: dict[int, dict[str, str]] = {}   # layer -> class -> tensor key
    other_changed: list[str] = []
    rows: dict[str, dict] = {}
    n_common = n_identical = 0
    with safe_open(base_file, framework="pt") as fb, safe_open(edit_file, framework="pt") as fe:
        common = sorted(set(fb.keys()) & set(fe.keys()))
        n_common = len(common)
        for k in common:
            wb = fb.get_tensor(k)
            we = fe.get_tensor(k)
            if wb.shape != we.shape or wb.dim() != 2:
                continue
            if torch.equal(wb, we):
                n_identical += 1
                continue
            layer, cls = _tensor_class(k)
            if cls == "other":
                other_changed.append(k)
                continue
            Wb = wb.float().to(dev)
            We = we.float().to(dev)
            delta = We - Wb

            U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
            s1 = S[0]
            u1 = U[:, 0]
            v1 = Vh[0, :]
            rank1 = s1 * torch.outer(u1, v1)
            resid = (delta - rank1).norm() / delta.norm().clamp_min(1e-12)
            energy = (s1 ** 2 / (S ** 2).sum()).item()

            # Fit the harness row-projection formula with u1 as d:
            #   -alpha * u1 (u1^T W_base) ~= Delta  =>  alpha_t = -<Delta,proj>/<proj,proj>
            proj = torch.outer(u1, u1 @ Wb)
            num = float((delta * proj).sum().item())
            den = float((proj * proj).sum().item())
            alpha_t = -(num / den) if den > 1e-18 else float("nan")
            fit_resid = ((delta + alpha_t * proj).norm() / delta.norm().clamp_min(1e-12)).item() if den > 1e-18 else float("nan")

            # row-side diagnostics: is v1 parallel to W^T u1 (the canonical
            # -alpha*d*(d^T W) formula) or symmetric in u1? Decides how the
            # recovered components must be RE-APPLIED later.
            wtu = u1 @ Wb
            c_v_wtu = abs(float((v1 * wtu).sum())) / float(v1.norm() * wtu.norm().clamp_min(1e-12))
            c_v_u = abs(float((v1[: min(v1.shape[0], u1.shape[0])] *
                               u1[: min(v1.shape[0], u1.shape[0])]).sum()))
            if v1.shape[0] == u1.shape[0]:
                c_v_u = abs(float((v1 * u1).sum())) / float(v1.norm() * u1.norm().clamp_min(1e-12))
            else:
                c_v_u = float("nan")  # different input dims (w2 vs out_proj) — not comparable
            u1_map_local = u1.cpu()

            rows[k] = {
                "layer": layer, "class": cls, "shape": list(Wb.shape),
                "rel_l2_delta": float((delta.norm() / Wb.norm().clamp_min(1e-12)).item()),
                "sigma1": float(s1.item()),
                "energy_share": energy,
                "rank1_resid": float(resid.item()),
                "alpha_hat_u1": alpha_t,
                "fit_resid_u1": fit_resid,
                "cos_v1_wtu1": float(c_v_wtu),
                "cos_v1_u1": c_v_u,
                "u1": u1_map_local,
                "v1": v1.cpu(),
            }
            by_layer.setdefault(layer, {})[cls] = k
            del delta, U, S, Vh, rank1, proj, Wb, We, wtu
            if dev == "cuda":
                torch.cuda.empty_cache()

    changed = list(rows.keys())
    print(f"common={n_common} identical={n_identical} changed={len(changed)} "
          f"other_changed={other_changed}", flush=True)

    # ---- per-layer direction: sign-aligned mean of the layer's u1s ----
    u1_map: dict[str, torch.Tensor] = {}
    layer_dirs: dict[int, torch.Tensor] = {}
    layer_stats: dict[int, dict] = {}
    for layer in sorted(by_layer):
        keys = list(by_layer[layer].values())
        us = []
        for k in keys:
            U, _, _ = torch.linalg.svd(_delta_of(k, base_file, edit_file, dev),
                                       full_matrices=False)
            u1_map[k] = U[:, 0]
            us.append(U[:, 0])
        # sign-align to the first tensor, then mean + unit normalize
        ref = us[0]
        aligned = [u if float((u * ref).sum()) >= 0 else -u for u in us]
        d = torch.stack(aligned).mean(dim=0)
        d = d / d.norm().clamp_min(1e-12)
        layer_dirs[layer] = d
        cos_pair = [abs(float((us[i] * us[0]).sum())) for i in range(1, len(us))]
        layer_stats[layer] = {"classes": sorted(by_layer[layer]),
                              "n_tensors": len(us),
                              "u1_abs_cos_vs_first": cos_pair}
        print(f"  layer {layer:>2}: {sorted(by_layer[layer])} "
              f"u1|cos|={[round(c, 5) for c in cos_pair]}", flush=True)

    # ---- alpha fits: per-layer and global, against the harness formula ----
    def _fit(dirs: dict[str, torch.Tensor], keys: list[str]) -> tuple[float, float, list[float]]:
        num = den = 0.0
        for k in keys:
            Wb, delta = _load_pair(k, base_file, edit_file, dev)
            proj = torch.outer(dirs[k], dirs[k] @ Wb)
            num += float((delta * proj).sum().item())
            den += float((proj * proj).sum().item())
        alpha = -(num / den) if den > 1e-18 else float("nan")
        per = []
        for k in keys:
            Wb, delta = _load_pair(k, base_file, edit_file, dev)
            proj = torch.outer(dirs[k], dirs[k] @ Wb)
            per.append(((delta + alpha * proj).norm() /
                        delta.norm().clamp_min(1e-12)).item())
        return alpha, den, per

    fit_per_layer: dict[int, dict] = {}
    for layer in sorted(by_layer):
        keys = list(by_layer[layer].values())
        d_layer = {k: layer_dirs[layer] for k in keys}
        alpha_l, den, per = _fit(d_layer, keys)
        fit_per_layer[layer] = {
            "alpha_l": alpha_l, "denominator": den,
            "fit_resid_per_tensor": {k: round(r, 5) for k, r in zip(keys, per)},
        }
        print(f"  layer {layer:>2}: alpha_l={alpha_l:.6f} "
              f"resid={[round(r, 5) for r in per]}", flush=True)

    all_keys = sorted(rows)
    d_global = {k: layer_dirs[rows[k]["layer"]] for k in all_keys}
    alpha_g, den_g, resid_g = _fit(d_global, all_keys)
    print(f"global alpha={alpha_g:.6f} "
          f"resid mean={sum(resid_g)/len(resid_g):.5f} max={max(resid_g):.5f}", flush=True)

    # per-tensor rel_l2 that the global-alpha + per-layer-d application yields
    expected: dict[str, float] = {}
    for k in all_keys:
        lyr = rows[k]["layer"]
        d = layer_dirs[lyr]
        Wb, _ = _load_pair(k, base_file, edit_file, dev)
        proj = torch.outer(d, d @ Wb)
        expected[k] = float((alpha_g * proj).norm().item() /
                            Wb.norm().clamp_min(1e-12).item())
        del proj, Wb

    # ---- save the harness-format directions bundle ----
    # `dirs` = per-layer unit direction (u1) for the harness formula methods
    # (advanced/mpoa: -alpha*d*(d^T W)); `recovered_rank1` = the exact
    # per-tensor rank-1 components for the `recovered` method (W' = W +
    # sigma*u*v^T), which reproduces the edit at max fidelity (the row side
    # of huihui's Delta is NOT W^T*u1 — cos_v1_wtu1 is far from 1 — so the
    # formula methods cannot land closer than the rank-1 residual).
    bundle = {
        "dirs": {str(l): layer_dirs[l].cpu() for l in sorted(layer_dirs)},
        "scores": {},
        "model_id": base_id,
        "dir_method": "recovered-rank1-svd",
        "probe_mode": "weights-svd",
        "prefill": "",
        "flavor": "recovered",
        "n_prompts": 32,
        "hidden": hidden,
        "side": "u1_left_singular",
        "source_edited": edited_id,
        "alpha_fit_global": alpha_g,
        "alpha_fit_per_layer": {str(l): fit_per_layer[l]["alpha_l"] for l in fit_per_layer},
        "per_tensor_rel_l2_expected": {k: round(v, 6) for k, v in expected.items()},
        "per_tensor_rel_l2_huihui": {k: rows[k]["rel_l2_delta"] for k in all_keys},
        "rank1_resid_per_tensor": {k: rows[k]["rank1_resid"] for k in all_keys},
        "recovered_rank1": {k: {"u": rows[k]["u1"], "v": rows[k]["v1"],
                                "sigma": rows[k]["sigma1"]} for k in all_keys},
    }
    import os as _os
    out_path = Path("/out") / bundle_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out_path)
    print(f"SAVED {out_path} ({out_path.stat().st_size} bytes)", flush=True)

    # strip tensors for the JSON report; bundle carries them
    report_rows = {k: {kk: vv for kk, vv in v.items() if kk not in ("u1", "v1")}
                   for k, v in rows.items()}
    return {
        "base": base_id, "edited": edited_id,
        "n_common": n_common, "n_identical": n_identical,
        "n_changed": len(changed), "changed": changed,
        "other_changed": other_changed,
        "layers": sorted(by_layer),
        "per_tensor": report_rows,
        "layer_dirs": {str(l): layer_dirs[l].cpu().tolist() for l in sorted(layer_dirs)},
        "layer_stats": {str(l): v for l, v in layer_stats.items()},
        "alpha_fit_per_layer": fit_per_layer,
        "alpha_fit_global": alpha_g,
        "resid_global_per_tensor": dict(zip(all_keys, resid_g)),
        "rel_l2_expected_with_global_alpha": {k: round(v, 6) for k, v in expected.items()},
        "rel_l2_delta_huihui": {k: rows[k]["rel_l2_delta"] for k in all_keys},
        "bundle_remote": str(out_path),
        "finish": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _delta_of(key: str, base_file, edit_file, dev):
    import torch
    return _load_pair(key, base_file, edit_file, dev)[1]


def _load_pair(key: str, base_file, edit_file, dev):
    """Re-open the safetensors and return (W_base, Delta) for `key`."""
    import torch
    from safetensors import safe_open
    with safe_open(base_file, framework="pt") as fb, safe_open(edit_file, framework="pt") as fe:
        Wb = fb.get_tensor(key).float().to(dev)
        We = fe.get_tensor(key).float().to(dev)
    return Wb, We - Wb


@app.local_entrypoint()
def main(argv: str = ""):
    import argparse

    ap = argparse.ArgumentParser(prog="forensics_modal")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("decode-grid")
    p.add_argument("--model-ref", required=True,
                   help="/absolver/abl-work/<dir> or HF id")
    p.add_argument("--labels", default="")
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--tag", default="")
    p.add_argument("--out", required=True)

    p = sub.add_parser("huihui-ab")
    p.add_argument("--model-id",
                   default="huihui-ai/Huihui-LFM2.5-1.2B-Instruct-abliterated")
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--out", required=True)

    p = sub.add_parser("tensor-diff")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--max-rows", type=int, default=400)
    p.add_argument("--out", required=True)

    p = sub.add_parser("first-token-kl",
                       help="mean first-token KL(edited || base) on the held-out prompts")
    p.add_argument("--base", required=True, help="HF id of the base model")
    p.add_argument("--edited", required=True, help="HF id of the edited model")
    p.add_argument("--out", required=True, help="local JSON report path")

    p = sub.add_parser("recover-directions",
                       help="rank-1 SVD recovery of the per-layer directions "
                            "from a published weight edit (base vs edited)")
    p.add_argument("--base", required=True, help="HF id of the base model")
    p.add_argument("--edited", required=True, help="HF id of the edited model")
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--bundle-name", default="directions-huihui-recovered.pt",
                   help="bundle filename written to the /out Volume")
    p.add_argument("--out", required=True, help="local JSON report path")

    args = ap.parse_args(argv.split())
    if args.cmd == "decode-grid":
        doc = run_decode_grid.remote(args.model_ref, args.labels,
                                     args.max_new, args.tag)
    elif args.cmd == "huihui-ab":
        doc = run_huihui_ab.remote(args.model_id, args.max_new)
    elif args.cmd == "recover-directions":
        doc = run_recover_directions.remote(args.base, args.edited,
                                            args.hidden, args.bundle_name)
    elif args.cmd == "first-token-kl":
        doc = run_first_token_kl.remote(args.base, args.edited)
    else:
        doc = run_tensor_diff.remote(args.a, args.b, args.max_rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWROTE {out} ({out.stat().st_size} bytes)")

    if args.cmd == "decode-grid":
        print("\n== summary ==")
        for cfg in doc["configs"]:
            print(f"{cfg['label']:>14}: refusal {cfg['n_refusal']}/{len(doc['harmful_prompts'])} "
                  f"| coherent {cfg['n_coherent']}/{len(doc['harmful_prompts'])} "
                  f"| clean {cfg['n_clean_compliant']}/{len(doc['harmful_prompts'])}")
    elif args.cmd == "huihui-ab":
        print("\n== summary ==")
        for cfg_name in sorted({r["config"] for r in doc["results"]}):
            rs = [r for r in doc["results"] if r["config"] == cfg_name]
            n_ref = sum(1 for r in rs if r["kind"] == "harmful" and r["refusal"])
            n_clean = sum(1 for r in rs if r["kind"] == "harmful"
                          and not r["refusal"] and r["coherent"]
                          and r["degeneracy_ratio"] < 0.05)
            print(f"{cfg_name:>12}: refusal {n_ref}/{len(doc['harmful_prompts'])} "
                  f"| clean {n_clean}/{len(doc['harmful_prompts'])}")
    elif args.cmd == "recover-directions":
        print("\n== recovery summary ==")
        print(f"changed tensors: {doc['n_changed']} (other_changed={doc['other_changed']})")
        rels = [r["rel_l2_delta"] for r in doc["per_tensor"].values()]
        res1 = [r["rank1_resid"] for r in doc["per_tensor"].values()]
        fitu = [r["fit_resid_u1"] for r in doc["per_tensor"].values()]
        print(f"Delta rel_l2:      min={min(rels):.4f} mean={sum(rels)/len(rels):.4f} max={max(rels):.4f}")
        print(f"rank-1 resid:      min={min(res1):.4f} mean={sum(res1)/len(res1):.4f} max={max(res1):.4f}")
        print(f"u1-formula resid:  min={min(fitu):.4f} mean={sum(fitu)/len(fitu):.4f} max={max(fitu):.4f}")
        print(f"alpha per layer:   {sorted(doc['alpha_fit_per_layer'].items())}")
        print(f"global alpha:      {doc['alpha_fit_global']:.6f}")
        csvw = [r["cos_v1_wtu1"] for r in doc["per_tensor"].values()]
        csvu = [r["cos_v1_u1"] for r in doc["per_tensor"].values() if r["cos_v1_u1"] == r["cos_v1_u1"]]
        print(f"cos(v1, W^T u1):   mean={sum(csvw)/len(csvw):.4f} "
              f"(canonical -a*d*(d^T W) row fit)")
        print(f"cos(v1, u1):       mean={sum(csvu)/len(csvu):.4f} (square tensors only)")
        print(f"bundle written to: {doc['bundle_remote']}")
    elif args.cmd == "first-token-kl":
        print("\n== first-token KL ==")
        for r in doc["per_prompt_kl"]:
            print(f"  KL {r['kl']:8.4f}  {r['prompt'][:60]!r}")
        print(f"mean first-token KL {doc['mean_first_token_kl']:.4f} "
              f"(harness threshold {doc['harness_gate_threshold']})")
    else:
        print(f"\n== diff {doc['a']} vs {doc['b']} ==")
        print(f"keys a={doc['n_keys_a']} b={doc['n_keys_b']} common={doc['common']} "
              f"identical={doc['identical']} differing={doc['differing']}")
        for g, s in doc["group_summary"].items():
            print(f"  {g:<12}: diff={s['n_diff']:>3} identical={s['n_identical']:>3} "
                  f"max_abs={s['max_abs']:<9} rel_l2={s['rel_l2']}")
        for r in doc["rows"]:
            if r["status"] == "DIFF":
                print(f"  DIFF {r['group']:<12} {r['tensor'][:60]:<62} "
                      f"max_abs={r['max_abs']:<9} rel_l2={r['rel_l2']:<9} "
                      f"n={r['n_diff_elems']}")