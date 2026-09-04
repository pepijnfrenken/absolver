"""recipe_v2.py — from-scratch abliteration toolkit, v2 lessons applied.

Implements the four fixes learned from the LFM2.5-2.6B campaign's failed
DIY attempts (Round 1/2) + successful Heretic reverse-engineering (Round 3):

1. COVERAGE: include attention out_proj (self_attn.out_proj), not just
   o_proj-alias / conv_out / w2. Heretic's 52-tensor edit used conv 20 +
   w2 25 + attn 7 — the attention channel was one we never touched and it
   matters.
2. GRADUATED ALPHA: Heretic skips shallow layers (0-2 untouched), ramps
   from L3, ~2 tensors/layer, peaks mid-network (~L13-17), tapers. A
   uniform alpha blasts shallow general features -> banal collapse.
3. DIRECTION-QUALITY PRE-PROBE: before spending GPU on a full apply+gate,
   verify the direction is a refusal direction, not content blur. Apply at
   tiny alpha on a FEW mid layers; measure benign PPL/uniq delta. Content-
   blurred directions collapse PPL/uniq even at small alpha; refusal
   directions don't move benign text.
4. WEIGHT-READ DIRECTION (optional): if a reference abliteration exists,
   read the direction from its weights (diff vs base, SVD rank-1) instead
   of estimating from activations — recovery beats discovery. (See
   campaigns/lfm2.5-2.6b/heretic-recovered-dirs.npz for the pattern.)

Reusable: import and call apply_v2(...) with any HF causal LM + tokenizer.
Direction source: matched-pair output-phase harvest (same prompts, refusal
vs affirm-prefilled) from probe.py — the 1.2B-proven recipe.
"""
from __future__ import annotations

from typing import Any, Callable

import torch


# ---------------------------------------------------------------------------
# Tensor resolution: all FOUR output-projection channels
# ---------------------------------------------------------------------------
def _resolve_proj(layer: Any, wname: str):
    """Resolve an output-projection module on a decoder layer.

    wname in {"attn_out", "conv_out", "w2"}. Handles both HF attr layouts
    (o_proj / out_proj / conv.out_proj / feed_forward.w2 / ffn.w2).
    """
    # --- attention output projection (the channel DIY Round 1 MISSED) ---
    if wname == "attn_out":
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            attn = getattr(layer, "attention", None)
        if attn is None:
            return None
        # prefer explicit out_proj (LFM2 / modern); fall back o_proj
        mod = getattr(attn, "out_proj", None)
        if mod is None:
            mod = getattr(attn, "o_proj", None)
        if mod is None:
            mod = getattr(attn, "attn_out", None)
        return mod
    # --- conv output projection ---
    if wname == "conv_out":
        conv = getattr(layer, "conv", None)
        if conv is None:
            return None
        mod = getattr(conv, "out_proj", None)
        if mod is None:
            mod = getattr(conv, "conv_out", None)
        return mod
    # --- feed-forward down projection (w2 / ffn_down / down_proj) ---
    if wname == "w2":
        ff = getattr(layer, "feed_forward", None) or getattr(layer, "ffn", None) \
            or getattr(layer, "mlp", None)
        if ff is None:
            return None
        for cand in ("w2", "down_proj", "ffn_down", "fc2"):
            mod = getattr(ff, cand, None)
            if mod is not None:
                return mod
        return None
    raise ValueError(f"unknown projection class {wname}")


# ---------------------------------------------------------------------------
# Alpha profiles: graduated per-layer strength
# ---------------------------------------------------------------------------
def uniform_alpha(alpha: float) -> Callable[[int], float]:
    return lambda li: alpha


def heretic_style_alpha(alpha: float, skip_shallow: int = 3,
                        peak_mid: float = 0.5, n_layers: int = 30,
                        taper_end: float = 0.7) -> Callable[[int], float]:
    """Heretic-style graduated profile.

    alpha(li) = 0 for li < skip_shallow; ramps linearly to `alpha` by
    li = peak_mid*n_layers; holds; tapers to alpha*taper_end at the last
    layer. Protects shallow general features; concentrates the edit where
    refusal is written (mid-late).
    """
    peak_li = int(peak_mid * n_layers)
    end_li = n_layers - 1
    if end_li <= peak_li:
        return lambda li: alpha if li >= skip_shallow else 0.0

    def _a(li: int) -> float:
        if li < skip_shallow:
            return 0.0
        if li <= peak_li:
            # linear ramp from ~0 at skip_shallow to alpha at peak
            span = max(1, peak_li - skip_shallow)
            return alpha * min(1.0, (li - skip_shallow) / span)
        # taper from alpha down to alpha*taper_end
        span = max(1, end_li - peak_li)
        frac = (li - peak_li) / span
        return alpha * (1.0 - (1.0 - taper_end) * frac)

    return _a


ALPHA_PROFILES: dict[str, Callable[[float], Callable[[int], float]]] = {
    "uniform": lambda a: uniform_alpha(a),
    "heretic": lambda a: heretic_style_alpha(a),
    "skip_shallow_midpeak": lambda a: heretic_style_alpha(a),
}


# ---------------------------------------------------------------------------
# Core apply: full 4-channel coverage + per-layer alpha profile
# ---------------------------------------------------------------------------
def apply_v2(
    model: Any,
    dirs: dict[int, torch.Tensor],
    *,
    alpha: float = 1.0,
    profile: str | Callable[[int], float] = "heretic",
    channels: tuple[str, ...] = ("attn_out", "conv_out", "w2"),
    mpoa: bool = True,
    layer_subset: list[int] | None = None,
    good_dirs: dict[int, torch.Tensor] | None = None,
    verbose: bool = True,
) -> list[str]:
    """Apply refusal-direction projection with v2 lessons.

    dirs: layer-index -> refusal direction (out-dim = hidden). If missing a
    layer, that layer is skipped. Channels default to ALL FOUR (attn_out +
    conv_out + w2); pass a subset to restrict.

    Returns list of applied tensor names.
    """
    # find decoder layers
    decoder = None
    for _name, mod in model.named_children():
        if hasattr(mod, "layers"):
            decoder = mod
            break
    if decoder is None and hasattr(model, "model") and hasattr(model.model, "layers"):
        decoder = model.model
    if decoder is None:
        raise RuntimeError("could not locate .layers on model")
    layers = decoder.layers
    n_layers = len(layers)

    alpha_fn: Callable[[int], float]
    if isinstance(profile, str):
        try:
            alpha_fn = ALPHA_PROFILES[profile](alpha)
        except KeyError:
            raise ValueError(f"unknown profile {profile}; have {list(ALPHA_PROFILES)}") from None
    else:
        alpha_fn = profile

    subset = set(layer_subset) if layer_subset else None
    applied: list[str] = []

    for li, d in dirs.items():
        if subset is not None and li not in subset:
            continue
        if li >= n_layers:
            continue
        a = alpha_fn(li)
        if a <= 0:
            continue
        d = d.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
        # projected abliteration: orthogonalize refusal against benign dir
        if good_dirs is not None and li in good_dirs:
            g = good_dirs[li].detach().to(dtype=torch.float32, device="cpu").reshape(-1)
            gn = g.norm().clamp(min=1e-8)
            g = g / gn
            d = d - (d @ g) * g
        dn = d.norm().clamp(min=1e-8)
        d = d / dn

        layer = layers[li]
        for wname in channels:
            mod = _resolve_proj(layer, wname)
            if mod is None:
                continue
            W = mod.weight.data
            if d.shape[0] != W.shape[0]:
                # direction out-dim mismatch — skip silently (shape varies)
                continue
            orig = None
            if mpoa:
                orig = W.norm().clamp(min=1e-8)
            d_w = d.to(dtype=W.dtype, device=W.device)
            W.sub_(a * torch.einsum("i,j->ij", d_w, d_w @ W))
            if mpoa:
                new = W.norm().clamp(min=1e-8)
                W.mul_((orig / new).to(dtype=W.dtype))
            applied.append(f"layer.{li}.{wname}")
    if verbose:
        print(f"[recipe_v2] applied {len(applied)} tensors "
              f"(profile={profile if isinstance(profile, str) else 'custom'}, "
              f"alpha={alpha}, mpoa={mpoa})", flush=True)
    return applied


# ---------------------------------------------------------------------------
# Direction-quality pre-probe: cheap, before full GPU apply
# ---------------------------------------------------------------------------
@torch.no_grad()
def probe_direction_quality(
    model: Any,
    tok: Any,
    dirs: dict[int, torch.Tensor],
    benign_prompts: list[str],
    *,
    layers_to_test: list[int] | None = None,
    probe_alpha: float = 0.15,
    channels: tuple[str, ...] = ("attn_out", "conv_out", "w2"),
    max_new: int = 64,
) -> dict:
    """Cheap direction sanity probe.

    Applies the direction at SMALL alpha on a FEW layers, then measures
    benign perplexity/coherence change. A content-blurred direction (topic
    proxy) collapses benign PPL/uniq even at small alpha; a real refusal
    direction barely moves benign text.

    Returns {benign_ppl_before, benign_ppl_after, ppl_delta,
             benign_uniq_before, benign_uniq_after, uniq_delta, verdict}.
    verdict: "refusal-like" if benign barely moved; "content-blurred" if
    PPL/uniq collapsed; "inconclusive" otherwise.
    """
    import copy

    decoder = None
    for _name, mod in model.named_children():
        if hasattr(mod, "layers"):
            decoder = mod
            break
    if decoder is None and hasattr(model, "model") and hasattr(model.model, "layers"):
        decoder = model.model
    if decoder is None:
        raise RuntimeError("could not locate .layers on model (probe_direction_quality)")
    layers = decoder.layers
    n_layers = len(layers)

    test_layers = layers_to_test or [int(0.45 * n_layers), int(0.55 * n_layers)]
    test_dirs = {li: dirs[li] for li in test_layers if li in dirs}
    if not test_dirs:
        return {"error": "no test-layer directions"}

    def _benign_stats(m):
        ppls, uniqs = [], []
        for p in benign_prompts[:6]:
            inp = tok(p, return_tensors="pt", truncation=True, max_length=128)
            inp = {k: v.to(next(m.parameters()).device) for k, v in inp.items()}
            out = m(**inp)
            lg = out.logits[:, :-1].log_softmax(-1)
            ids = inp["input_ids"][:, 1:]
            nll = -torch.gather(lg, -1, ids.unsqueeze(-1)).squeeze(-1).mean()
            ppls.append(float(nll.exp()))
            # response uniq proxy: generate short, measure vocab uniqueness
            gen = m.generate(**inp, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
            text = tok.decode(gen[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
            words = text.split()
            uniqs.append(len(set(words)) / max(1, len(words)))
        return sum(ppls) / len(ppls), sum(uniqs) / len(uniqs)

    before_ppl, before_uniq = _benign_stats(model)

    # deep-copy model? too heavy at 2.6B+ on CPU; instead apply on a shallow
    # COPY of the weights for the tested layers and restore after.
    # Simpler: clone the few test-layer projections, apply, measure, restore.
    saved: list[tuple] = []
    for li, d in test_dirs.items():
        layer = layers[li]
        for wname in channels:
            mod = _resolve_proj(layer, wname)
            if mod is None:
                continue
            saved.append((mod.weight.data, mod.weight.data.clone()))
    # apply at probe alpha
    test_model_copy = copy.deepcopy(model) if _model_small(model) else model
    if not _model_small(model):
        # apply + measure + restore in place (no full copy)
        apply_v2(model, test_dirs, alpha=probe_alpha, profile="uniform",
                 channels=channels, mpoa=True, verbose=False)
        after_ppl, after_uniq = _benign_stats(model)
        # restore
        for w, saved_w in saved:
            w.copy_(saved_w)
    else:
        apply_v2(test_model_copy, test_dirs, alpha=probe_alpha,
                 profile="uniform", channels=channels, mpoa=True, verbose=False)
        after_ppl, after_uniq = _benign_stats(test_model_copy)

    ppl_delta = (after_ppl - before_ppl) / max(1e-8, before_ppl)
    uniq_delta = (after_uniq - before_uniq) / max(1e-8, before_uniq)
    if ppl_delta > 0.10 or uniq_delta < -0.15:
        verdict = "content-blurred"
    elif ppl_delta < 0.05 and uniq_delta > -0.05:
        verdict = "refusal-like"
    else:
        verdict = "inconclusive"
    return {
        "benign_ppl_before": before_ppl, "benign_ppl_after": after_ppl,
        "ppl_delta": ppl_delta, "benign_uniq_before": before_uniq,
        "benign_uniq_after": after_uniq, "uniq_delta": uniq_delta,
        "verdict": verdict,
    }


def _model_small(model: Any) -> bool:
    """Heuristic: total params < 2B -> safe to deepcopy for probing."""
    try:
        n = sum(p.numel() for p in model.parameters())
        return n < 2e9
    except Exception:
        return False
