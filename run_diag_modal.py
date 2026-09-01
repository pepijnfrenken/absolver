"""Modal runner for the direction-quality diagnostic."""
import modal
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.11", force_build=True)
    .uv_pip_install(
        "torch>=2.0", "transformers>=4.30", "langgraph>=0.3",
        "pydantic>=2", "pyyaml>=6", "huggingface-hub>=0.20", "numpy>=1.24",
        "accelerate>=0.20",
    )
    .env({"PYTHONPATH": "/absolver"})
    .add_local_dir(
        str(PROJECT_DIR), remote_path="/absolver",
        ignore=[".venv", ".git", "__pycache__", "*.pyc", ".aiwg",
                ".pytest_cache", ".mypy_cache", ".ruff_cache",
                "abliterated_models", "experiments"],
    )
)

app = modal.App("absolver-diag")

@app.function(
    image=image,
    gpu="L4",
    timeout=3600,
    retries=0,
    secrets=[
        modal.Secret.from_name("hf-write-token"),
        modal.Secret.from_name("freeinference-token"),
    ],
)
def run_diag() -> dict:
    os.chdir("/absolver")
    import sys
    sys.path.insert(0, "/absolver")
    import torch
    from config import load_config
    from summon import summon_node
    from model_registry import get_model, get_tokenizer
    from probe import _collect_paired_output_phase
    from sweep import _apply_candidate, _quick_score
    from prompts import DEFAULT_HARMFUL, DEFAULT_HARMLESS

    cfg = load_config("models/qwen2.5-1.5b-instruct.yaml")
    summon_node({
        "config": cfg, "model_loaded": False, "architecture": None,
        "hidden_size": None, "num_layers": None,
    })
    model = get_model()
    tok = get_tokenizer()
    device = next(model.parameters()).device
    print("device:", device)

    layers = model.model.layers if hasattr(model, "model") else model.layers
    n_layers = len(layers)
    print("n_layers:", n_layers)

    harmful = list(DEFAULT_HARMFUL)[:20]
    harmless = list(DEFAULT_HARMLESS)[:20]
    print("harms:", len(harmful), "harmless:", len(harmless))

    refusal_acts, affirm_acts = _collect_paired_output_phase(
        model, tok, harmful, layers, n_layers, device,
        prefill=cfg.paired_prefill, max_new_tokens=cfg.paired_max_new_tokens)

    norms = {}
    directions = {}
    for i in range(n_layers):
        if i not in refusal_acts or i not in affirm_acts:
            continue
        r = torch.stack(refusal_acts[i]).mean(0).float()
        a = torch.stack(affirm_acts[i]).mean(0).float()
        d = r - a
        norm = d.norm().item()
        norms[i] = norm
        if norm < 1e-8:
            directions[i] = torch.zeros_like(d)
            print(f"layer {i}: ZERO direction")
            continue
        directions[i] = d / norm
        print(f"layer {i}: dir norm={norm:.4f}")

    score_cfg = type("S", (), {"max_seq_len": 1024})()
    base = _quick_score(model, tok, score_cfg, harmful, max_tokens=48)
    print("\nBASELINE refusal:", base["refusal"], "quality:", base["quality"])

    # Check direction-weight alignment: how much does d actually overlap
    # with the o_proj / down_proj row space? If d@W ~ 0, projection is a no-op.
    for li in [26, 27]:
        layer = layers[li]
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
            w = layer.self_attn.o_proj.weight.data.float()
            d = directions[li].float().to(w.device)
            proj_norm = (d @ w).norm().item()
            print(f"layer {li} o_proj: d@W norm={proj_norm:.4f}  (W shape {tuple(w.shape)})")
        ff = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
        if ff is not None and hasattr(ff, "down_proj"):
            w = ff.down_proj.weight.data.float()
            d = directions[li].float().to(w.device)
            proj_norm = (d @ w).norm().item()
            print(f"layer {li} down_proj: d@W norm={proj_norm:.4f}  (W shape {tuple(w.shape)})")

    cand = {"method": "mpoa", "target_layers": [26, 27],
            "target_weights": ["o_proj"], "alpha": 2.0, "passes": 1}
    _apply_candidate(model, directions, None, cand, None)
    after = _quick_score(model, tok, score_cfg, harmful, max_tokens=48)
    print("\nAFTER MPOA alpha=2.0 on [26,27] o_proj:")
    print("  refusal:", after["refusal"], "quality:", after["quality"],
          "kl:", after["kl"])

    # Also try down_proj (MLP) — the refusal direction may live in the MLP
    # rather than attention, especially for dense models.
    cand2 = {"method": "mpoa", "target_layers": [26, 27],
             "target_weights": ["down_proj"], "alpha": 2.0, "passes": 1}
    _apply_candidate(model, directions, None, cand2, None)
    after2 = _quick_score(model, tok, score_cfg, harmful, max_tokens=48)
    print("\nAFTER MPOA alpha=2.0 on [26,27] down_proj (cumulative):")
    print("  refusal:", after2["refusal"], "quality:", after2["quality"],
          "kl:", after2["kl"])

    # PLAIN projection (no MPOA rescale) at alpha=4.0 — the rescale in MPOA
    # may be canceling the removal. Test the raw _project_2d path.
    import inspect
    from excise import _project_2d
    src = inspect.getsource(_project_2d)
    print("\n=== _project_2d source (in container) ===")
    print(src[:600])
    for li in [26, 27]:
        layer = layers[li]
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
            w = layer.self_attn.o_proj.weight.data
            d_vec = directions[li]
            print(f"layer {li}: d shape={tuple(d_vec.shape)} dtype={d_vec.dtype} "
                  f"norm={d_vec.float().norm().item():.4f} "
                  f"min={d_vec.float().min().item():.2e} max={d_vec.float().max().item():.2e}")
            # manual einsum check in float32 (same device)
            wf = w.float()
            dv = d_vec.float().to(wf.device)
            proj = torch.einsum("i,j->ij", dv, dv @ wf)
            print(f"  einsum absmax (f32) = {proj.abs().max().item():.2e}")
            before = w.clone()
            _project_2d(w, d_vec, 4.0)
            diff = (w - before).abs().max().item()
            print(f"  max abs weight delta after plain proj = {diff:.6e}")
    after3 = _quick_score(model, tok, score_cfg, harmful, max_tokens=48)
    print("\nAFTER PLAIN _project_2d alpha=4.0 on [26,27] o_proj (cumulative):")
    print("  refusal:", after3["refusal"], "quality:", after3["quality"],
          "kl:", after3["kl"])

    # STRONG intervention: plain projection at alpha=10.0 on layers 20-27
    # o_proj + down_proj. If THIS doesn't move refusal, the directions are
    # simply not where Qwen's refusal lives.
    print("\n=== STRONG: alpha=10.0, layers 20-27, o_proj + down_proj ===")
    from excise import _project_2d as _p2
    for li in range(20, 28):
        layer = layers[li]
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
            _p2(layer.self_attn.o_proj.weight.data, directions[li], 10.0)
        ff = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
        if ff is not None and hasattr(ff, "down_proj"):
            _p2(ff.down_proj.weight.data, directions[li], 10.0)
    after4 = _quick_score(model, tok, score_cfg, harmful, max_tokens=48)
    print("AFTER STRONG alpha=10.0 layers 20-27 (cumulative):")
    print("  refusal:", after4["refusal"], "quality:", after4["quality"],
          "kl:", after4["kl"])
    # show per-prompt outputs to see WHAT is being refused
    for p in harmful[:5]:
        inp = tok(p, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=48, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        print(f"  PROMPT: {p[:40]}... -> {tok.decode(out[0], skip_special_tokens=True)[:70]!r}")

    # DIFF_MEANS input-phase directions (harmful vs harmless, first forward)
    # — the classic ablation direction, which the shape bug also broke.
    print("\n=== DIFF_MEANS input-phase directions ===")
    from collections import defaultdict
    def _collect_input_phase(prompts_list):
        acts: dict[int, list[torch.Tensor]] = defaultdict(list)
        for p in prompts_list:
            store: dict[int, list[torch.Tensor]] = defaultdict(list)
            handles = [layers[i].register_forward_hook(_make_hook_input(i, store))
                       for i in range(n_layers)]
            try:
                inp = tok(p, return_tensors="pt", truncation=True, max_length=128)
                inp = {k: v.to(device) for k, v in inp.items() if hasattr(v, "to")}
                with torch.no_grad():
                    model(**inp)
            except Exception as exc:
                print("  input-phase forward failed:", exc)
            finally:
                for h in handles:
                    try: h.remove()
                    except Exception: pass
            for i in range(n_layers):
                st = store.get(i)
                if st:
                    acts[i].append(torch.stack(st).mean(0).squeeze(0))
        return dict(acts)

    def _make_hook_input(name, store):
        def hook(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            if hasattr(hs, "dim") and hs.dim() >= 2:
                store[name].append(hs[0, -1].detach().cpu().float() if hs.dim() == 3
                                   else hs[0].detach().cpu().float())
        return hook

    harm_acts = _collect_input_phase(harmful)
    harm_acts_small = _collect_input_phase(harmless)
    dm_directions = {}
    for i in range(n_layers):
        if i not in harm_acts or i not in harm_acts_small:
            continue
        h = torch.stack(harm_acts[i]).mean(0)
        b = torch.stack(harm_acts_small[i]).mean(0)
        d = (h - b).float()
        n = d.norm().item()
        if n < 1e-8:
            continue
        dm_directions[i] = d / n
        if i >= 24:
            print(f"  layer {i} diff_means dir norm={n:.4f}")

    # apply diff_means at alpha=10 on 20-27 o_proj+down_proj
    for li in range(20, 28):
        layer = layers[li]
        if li not in dm_directions:
            continue
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
            _p2(layer.self_attn.o_proj.weight.data, dm_directions[li], 10.0)
        ff = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
        if ff is not None and hasattr(ff, "down_proj"):
            _p2(ff.down_proj.weight.data, dm_directions[li], 10.0)
    after5 = _quick_score(model, tok, score_cfg, harmful, max_tokens=48)
    print("AFTER DIFF_MEANS alpha=10.0 layers 20-27 (cumulative):")
    print("  refusal:", after5["refusal"], "quality:", after5["quality"],
          "kl:", after5["kl"])
    for p in harmful[:5]:
        inp = tok(p, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=48, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        print(f"  PROMPT: {p[:40]}... -> {tok.decode(out[0], skip_special_tokens=True)[:70]!r}")

    return {"norms": norms, "baseline_refusal": base["refusal"],
            "after_refusal": after["refusal"], "after_quality": after["quality"],
            "after_down_refusal": after2["refusal"],
            "after_plain_refusal": after3["refusal"],
            "after_strong_refusal": after4["refusal"],
            "after_diffmeans_refusal": after5["refusal"]}


@app.local_entrypoint()
def main():
    result = run_diag.remote()
    print("\nDIAG RESULT:", result)
