"""SWEEP node — try multiple ablation methods & configs, pick the best.

Runs after DISTILL (directions computed) and before EXCISE. The sweep
tests fundamentally different approaches, not just hyperparameter grids:

  - **Method**: ``advanced`` (weight projection), ``bias_vectors`` (output bias),
    ``direct_ablation`` (zero out weights), ``lora`` (trainable rank adapters, future)
  - **Direction extraction**: ``diff_means``, ``svd``, ``leace``, ``whitened_svd``
  - **Architecture**: dense (o_proj / down_proj), MoE (expert.down), hybrid
  - **Layer / alpha / passes** per method

Each candidate:
  1. Restores pristine model state
  2. Computes directions (if method changes dir_method)
  3. Applies the ablation
  4. Scores quickly (keyword refusal + quality proxy)
  5. Restores pristine for next candidate

The winner's params are written to state so EXCISE uses them.
"""
from __future__ import annotations

import itertools
import logging
import time
from typing import Any

import torch

from model_registry import get_model, get_tokenizer
from prompts import DEFAULT_HARMFUL
from verify import REFUSAL_KEYWORDS
from excise import (
    _project_2d,
    _project_2d_mpoa,
    _project_3d_expert,
    _project_3d_expert_mpoa,
)
from distill import extract_directions
from state import AbliterationState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Grid construction — method × dir_method × layers × alpha × passes
# ---------------------------------------------------------------------------

def _build_candidates(cfg: Any, layer_types: list[str] | None = None) -> list[dict[str, Any]]:
    """Cartesian product of sweep dimensions. Each candidate is a full
    method override dict applied on top of the base config.

    ``layer_types`` (e.g. [\"conv\", \"full_attention\", ...] from the model
    config) filters layer sets: hybrid architectures like LFM have conv
    layers whose weight shapes don't match the hidden dim, so projecting
    them corrupts the model. Any candidate targeting a conv layer is
    dropped.
    """
    methods = getattr(cfg, "sweep_methods", None) or [cfg.method]
    dir_methods = getattr(cfg, "sweep_dir_methods", None) or [cfg.dir_method]
    layer_sets = list(getattr(cfg, "sweep_layer_sets", None) or [[]])
    alphas = list(getattr(cfg, "sweep_alphas", None) or [cfg.alpha])
    passes = list(getattr(cfg, "sweep_passes", None) or [cfg.passes])
    weight_sets = list(getattr(cfg, "sweep_target_weights", None) or [[]])

    def _is_conv_layer(idx: int) -> bool:
        if not layer_types:
            return False
        t = layer_types[idx] if idx < len(layer_types) else None
        return t is not None and "conv" in str(t).lower()

    # Auto-extend the layer space: when the sweep didn't specify layer sets,
    # add the full attention-layer set (all non-conv layers) as a candidate.
    # This is how the pipeline *finds* recipes like the LFM2.5 winner
    # (mpoa x paired x [2,5,8,10,12,14] x alpha 2.0) without a hand-written
    # config — the all-attention set is always tried when layer_types exist.
    if layer_types and (not layer_sets or layer_sets == [[]]):
        attention_idxs = [
            i for i in range(len(layer_types))
            if not _is_conv_layer(i)
        ]
        if attention_idxs:
            layer_sets.append(attention_idxs)
    elif not layer_sets or layer_sets == [[]]:
        # Dense models without layer_types: fall back to the configured
        # target_layers so an empty sweep space still searches something.
        if getattr(cfg, "target_layers", None):
            layer_sets = [list(cfg.target_layers)]

    grid = []
    for m, dm, ls, a, p, ws in itertools.product(
        methods, dir_methods, layer_sets, alphas, passes, weight_sets
    ):
        # Skip candidates whose layer set includes a conv layer — projecting
        # conv weights (non-hidden-shaped) corrupts the model.
        if any(_is_conv_layer(idx) for idx in ls):
            continue
        candidate = {
            "method": m,
            "dir_method": dm,
            "target_layers": ls,
            "alpha": a,
            "passes": p,
            "target_weights": ws or list(cfg.target_weights),
        }
        # Skip candidates with no layers (they won't change anything)
        if ls:
            grid.append(candidate)
    return grid


# ---------------------------------------------------------------------------
# 2. Method dispatch — apply ablation for one candidate
# ---------------------------------------------------------------------------

def _restore(model: Any, pristine: dict[str, torch.Tensor]) -> None:
    """Load pristine weights back into the model (CPU->device as needed)."""
    # nn.Module has no .device attribute — infer from the first parameter.
    try:
        target_device = next(model.parameters()).device
    except (StopIteration, RuntimeError):
        target_device = torch.device("cpu")
    model.load_state_dict({
        k: v.to(device=target_device) if isinstance(v, torch.Tensor) else v
        for k, v in pristine.items()
    })


def _find_layers(model: Any):
    for _name, mod in model.named_children():
        if hasattr(mod, "layers"):
            return mod.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("Cannot locate transformer layers")


def _dispatch_parallel_candidate(cand: dict, directions: dict, state: Any, cfg: Any) -> dict:
    """Dispatch a single sweep candidate to its own Modal task.

    Serializes the candidate + direction tensors (CPU, lists) and calls the
    remote `evaluate_sweep_candidate` function. Falls back to a serialized
    in-process evaluation if Modal is unavailable (local smoke tests).

    Returns the scored result dict merged with the candidate.
    """
    try:
        import modal
        # Host-side: build the same App/function the runner exposes.
        from run_absolver_modal import app, evaluate_sweep_candidate

        # Directions must be plain data for modal.map serialization.
        dirs_plain = {
            str(k): (v.tolist() if hasattr(v, "tolist") else v)
            for k, v in directions.items()
        }
        payload = {
            "model_id": cfg.model_id,
            "candidate": cand,
            "directions": dirs_plain,
            "probe_cfg": {
                "n_directions": cfg.n_directions,
                "prompt_format": getattr(cfg, "prompt_format", "auto"),
                "n_verify_prompts": getattr(cfg, "n_verify_prompts", 20),
                "max_seq_len": getattr(cfg, "max_seq_len", 1024),
                "paired_prefill": getattr(cfg, "paired_prefill", None),
            },
        }
        # modal.map runs each payload in its own container, ~10 concurrent.
        with modal.environ() as _env:
            out = list(map(evaluate_sweep_candidate.remote, [payload]))
        score = out[0] or {}
        return {**cand, **score, "objective": float(score.get("objective", -9.0))}
    except Exception as exc:
        logger.warning("SWEEP parallel dispatch failed (%s) — falling back to serial", exc)
        # In-process fallback: reuse the graph's own machinery via import.
        from model_registry import get_model, get_tokenizer
        from sweep import _quick_score
        model, tok = get_model(), get_tokenizer()
        _restore(model, state.get("pristine_state_dict"))
        try:
            _apply_candidate(model, directions, state.get("pristine_state_dict"), cand, None)
            score = _quick_score(model, tok, cfg, [], base_logprobs=None)
            return {**cand, **score}
        except Exception as exc2:
            logger.warning("SWEEP parallel fallback failed: %s", exc2)
            return {**cand, "refusal": 1.0, "quality": 0.0, "objective": -1.0}




def _as_1d(dirs) -> torch.Tensor:
    """Return the first refusal direction as a flat 1D vector.

    Directions can arrive as 1D [hidden], 2D [1, hidden] (probe hooks keep a
    batch dim), or [n_dirs, hidden] (SVD/whitened). All `_apply_*` methods
    must consume a flat vector or the projection math breaks (e.g. the
    `(1x2048 and 1x2048)` matmul crash in _apply_projected_abliteration).
    """
    d = dirs[0] if torch.is_tensor(dirs) and dirs.dim() > 1 else dirs
    if torch.is_tensor(d):
        return d.reshape(-1)
    return torch.as_tensor(d).reshape(-1)


def _apply_advanced(model: Any, layers_mod, directions: dict, candidate: dict[str, Any]) -> None:
    """Weight projection (diff-means / SVD / LEACE — the current method)."""
    for layer_idx in candidate["target_layers"]:
        if layer_idx not in directions or layer_idx >= len(layers_mod):
            continue
        layer = layers_mod[layer_idx]
        dirs = directions[layer_idx]
        d = _as_1d(dirs)
        for wname in candidate["target_weights"]:
            if wname == "o_proj" and hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
                w = layer.self_attn.o_proj.weight.data  # .data detaches autograd (like excise)
                _project_2d(w, d.to(device=w.device), candidate["alpha"])
            elif wname == "down_proj":
                ff = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
                if ff is not None and hasattr(ff, "down_proj"):
                    w = ff.down_proj.weight.data
                    _project_2d(w, d.to(device=w.device), candidate["alpha"])
                elif hasattr(layer, "feed_forward") and hasattr(layer.feed_forward, "down_proj"):
                    w = layer.feed_forward.down_proj.weight.data
                    _project_2d(w, d.to(device=w.device), candidate["alpha"])
            elif wname == "expert.down" and hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
                for expert in layer.mlp.experts:
                    if hasattr(expert, "down_proj"):
                        w = expert.down_proj.weight.data
                        _project_2d(w, d.to(device=w.device), candidate["alpha"])


def _apply_mpoa(model: Any, layers_mod, directions: dict, candidate: dict[str, Any]) -> None:
    """Magnitude-preserving orthogonal ablation (the LFM2.5 winning recipe).

    Same projection as ``advanced`` but rescales the weight back to its
    original Frobenius norm after subtraction — this is why alpha >= 1.0
    (e.g. 2.0 on all six attention blocks) removes refusal without
    collapsing the layer output scale.
    """
    for layer_idx in candidate["target_layers"]:
        if layer_idx not in directions or layer_idx >= len(layers_mod):
            continue
        layer = layers_mod[layer_idx]
        dirs = directions[layer_idx]
        d = _as_1d(dirs)
        for wname in candidate["target_weights"]:
            if wname == "o_proj" and hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
                w = layer.self_attn.o_proj.weight.data
                _project_2d_mpoa(w, d.to(device=w.device), candidate["alpha"])
            elif wname == "down_proj":
                ff = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
                if ff is not None and hasattr(ff, "down_proj"):
                    w = ff.down_proj.weight.data
                    _project_2d_mpoa(w, d.to(device=w.device), candidate["alpha"])
                elif hasattr(layer, "feed_forward") and hasattr(layer.feed_forward, "down_proj"):
                    w = layer.feed_forward.down_proj.weight.data
                    _project_2d_mpoa(w, d.to(device=w.device), candidate["alpha"])
            elif wname == "expert.down" and hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
                for expert in layer.mlp.experts:
                    if hasattr(expert, "down_proj"):
                        w = expert.down_proj.weight.data
                        _project_2d_mpoa(w, d.to(device=w.device), candidate["alpha"])


def _apply_bias_vectors(model: Any, layers_mod, directions: dict, candidate: dict[str, Any]) -> None:
    """Output bias modification: add/subtract the refusal direction scaled by
    alpha to the output of the target layers. Non-destructive, works well
    for MoE + dense models where weight projection is too aggressive."""
    for layer_idx in candidate["target_layers"]:
        if layer_idx not in directions or layer_idx >= len(layers_mod):
            continue
        layer = layers_mod[layer_idx]
        dirs = directions[layer_idx]
        d = _as_1d(dirs)
        # Add bias to the layer output (both self-attention output and MLP output)
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
            bias_mod = layer.self_attn.o_proj.bias
            if bias_mod is not None:
                w = layer.self_attn.o_proj.weight.data
                bias_mod.data.add_(d.to(device=w.device, dtype=w.dtype) * candidate["alpha"])
        ff = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
        if ff is not None and hasattr(ff, "down_proj"):
            bias_mod = ff.down_proj.bias
            if bias_mod is not None:
                w = ff.down_proj.weight.data
                bias_mod.data.add_(-d.to(device=w.device, dtype=w.dtype) * candidate["alpha"])  # subtract refusal direction from MLP output


def _apply_direct_ablation(model: Any, layers_mod, directions: dict, candidate: dict[str, Any]) -> None:
    """Project OUT the refusal direction entirely from the weight.

    W <- W - alpha * d (d^T W)   (removes the component along d)
    This is the standard orthogonal projection; the term is NOT scaled by
    ||d^T W|| (that would feed back into ||W|| and blow up on later passes).
    """
    for layer_idx in candidate["target_layers"]:
        if layer_idx not in directions or layer_idx >= len(layers_mod):
            continue
        layer = layers_mod[layer_idx]
        dirs = directions[layer_idx]
        d = _as_1d(dirs)
        for wname in candidate["target_weights"]:
            if wname == "o_proj" and hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
                w = layer.self_attn.o_proj.weight
                if w.dim() == 2 and d.shape[0] == w.shape[1]:
                    d_w = d.to(device=w.device, dtype=w.dtype)
                    w.data.sub_(candidate["alpha"] * d_w.unsqueeze(1) @ (d_w @ w).unsqueeze(0))
            elif wname == "down_proj":
                ff = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
                if ff is not None and hasattr(ff, "down_proj"):
                    w = ff.down_proj.weight
                    if w.dim() == 2 and d.shape[0] == w.shape[1]:
                        d_w = d.to(device=w.device, dtype=w.dtype)
                        w.data.sub_(candidate["alpha"] * d_w.unsqueeze(1) @ (d_w @ w).unsqueeze(0))


def _apply_projected_abliteration(model: Any, layers_mod, directions: dict,
                                  good_dirs: dict, candidate: dict[str, Any]) -> None:
    """Heretic/grimjim-style projected abliteration: orthogonalize the refusal
    direction against the harmless direction BEFORE projecting, so capabilities
    that overlap the refusal subspace are preserved. delta_W = -lambda * v (v^T W)
    with v = normalize(refusal - (refusal.good) good)."""
    for layer_idx in candidate["target_layers"]:
        if layer_idx not in directions or layer_idx >= len(layers_mod):
            continue
        layer = layers_mod[layer_idx]
        d = _as_1d(directions[layer_idx]).to(torch.float32)
        # Project refusal direction away from harmless direction
        if layer_idx in good_dirs:
            g = _as_1d(good_dirs[layer_idx]).to(torch.float32)
            d = d - (d @ g) * g
            d = d / d.norm().clamp(min=1e-8)

        for wname in candidate["target_weights"]:
            if wname == "o_proj" and hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
                w = layer.self_attn.o_proj.weight.data
                _project_2d(w, d.to(device=w.device), candidate["alpha"])
            elif wname == "down_proj":
                ff = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
                if ff is not None and hasattr(ff, "down_proj"):
                    w = ff.down_proj.weight.data
                    _project_2d(w, d.to(device=w.device), candidate["alpha"])


def _apply_lora_abliteration(model: Any, layers_mod, directions: dict,
                             candidate: dict[str, Any]) -> None:
    """Heretic-style LoRA abliteration: instead of mutating weights, attach
    rank-1 adapters delta_W = -lambda * v (v^T W), decomposed as
    lora_A = v^T W, lora_B = -lambda * v. We store the delta and apply it to
    the weight in place (the sweep restores pristine between candidates, so
    this is equivalent but keeps the adapter math)."""
    for layer_idx in candidate["target_layers"]:
        if layer_idx not in directions or layer_idx >= len(layers_mod):
            continue
        layer = layers_mod[layer_idx]
        v = _as_1d(directions[layer_idx]).to(torch.float32)
        v = v / v.norm().clamp(min=1e-8)
        for wname in candidate["target_weights"]:
            if wname == "o_proj" and hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
                w = layer.self_attn.o_proj.weight.data
                _apply_lora_delta(w, v, candidate["alpha"])
            elif wname == "down_proj":
                ff = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
                if ff is not None and hasattr(ff, "down_proj"):
                    w = ff.down_proj.weight.data
                    _apply_lora_delta(w, v, candidate["alpha"])


def _apply_lora_delta(w: torch.Tensor, v: torch.Tensor, alpha: float) -> None:
    """W += lora_B @ lora_A = (-alpha * v) @ (v^T W)  (rank-1 LoRA ablation)."""
    if w.dim() != 2 or v.shape[0] != w.shape[1]:
        # Hybrid archs (LFM conv layers) expose non-square weights; skip.
        return
    W = w.to(torch.float32)
    lora_A = (v @ W).view(1, -1)          # [1, in]
    lora_B = (-alpha * v).view(-1, 1)     # [out, 1]
    delta = lora_B @ lora_A               # [out, in]
    w.data.add_(delta.to(device=w.device, dtype=w.dtype))


def _apply_candidate(model: Any, directions: dict, pristine: dict | None,
                     candidate: dict[str, Any], good_dirs: dict | None = None) -> None:
    """Dispatch to the right method for one candidate."""
    layers_mod = _find_layers(model)
    method = candidate["method"]
    passes = max(1, candidate["passes"])

    for i in range(passes):
        if method == "advanced":
            _apply_advanced(model, layers_mod, directions, candidate)
        elif method == "mpoa":
            _apply_mpoa(model, layers_mod, directions, candidate)
        elif method == "bias_vectors":
            _apply_bias_vectors(model, layers_mod, directions, candidate)
        elif method == "direct_ablation":
            _apply_direct_ablation(model, layers_mod, directions, candidate)
        elif method == "projected":
            _apply_projected_abliteration(model, layers_mod, directions, good_dirs or {}, candidate)
        elif method == "lora":
            _apply_lora_abliteration(model, layers_mod, directions, candidate)
        else:
            logger.warning("SWEEP: unknown method %s, falling back to advanced", method)
            _apply_advanced(model, layers_mod, directions, candidate)
        # For passes > 1, restore and re-apply
        if i < passes - 1 and pristine is not None:
            _restore(model, pristine)


# ---------------------------------------------------------------------------
# 3. Quick scoring — cheap refusal + quality proxy
# ---------------------------------------------------------------------------

def _quick_score(model: Any, tok: Any, cfg: Any, prompts: list[str],
                 max_tokens: int = 48, base_logprobs: dict[int, torch.Tensor] | None = None,
                 harmless_prompts: list[str] | None = None) -> dict[str, Any]:
    """Cheap scoring: refusal rate (keyword) + quality.

    Quality is measured two ways:
      - response-length ratio (fallback)
      - KL divergence vs the pristine model's log-probs on harmless prompts
        (Heretic-style: lower KL = less capability damage). Used when
        ``base_logprobs`` is provided.
    """
    device = next(model.parameters()).device
    refusals = 0
    lengths: list[float] = []
    for p in prompts:
        inp = tok(p, return_tensors="pt", truncation=True, max_length=cfg.max_seq_len).to(device)
        try:
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=max_tokens, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            resp = tok.decode(out[0], skip_special_tokens=True)
        except Exception:
            resp = ""
        low = resp.lower()
        if any(kw in low for kw in REFUSAL_KEYWORDS):
            refusals += 1
        lengths.append(min(len(resp) / (max_tokens * 4.0), 1.0))
    refusal_rate = refusals / len(prompts) if prompts else 1.0
    quality = sum(lengths) / len(lengths) if lengths else 0.0
    kl_value: float | None = None

    # Heretic-style KL divergence vs pristine model on harmless prompts.
    # Lower KL = better capability preservation. We keep the RAW KL and a
    # squashed quality = 1/(1+kl). Logging uses raw KL with 4 decimals so
    # small-but-real differences are visible (v2 fix: quality 1.00 was hiding
    # KL 0.001-0.01 because 1/(1+kl) rounds to 1.00 at 2 decimals).
    if base_logprobs is not None and harmless_prompts:
        try:
            kl_value = _kl_divergence(model, tok, cfg, harmless_prompts, base_logprobs)
            quality = 1.0 / (1.0 + kl_value)
        except Exception as exc:
            logger.warning("KL divergence computation failed (%s); using length proxy", exc)

    return {"refusal": refusal_rate, "quality": quality, "kl": kl_value}


def _kl_divergence(model: Any, tok: Any, cfg: Any, prompts: list[str],
                   base_logprobs: dict[int, torch.Tensor]) -> float:
    """Sequence-level KL divergence (Heretic-style) between the candidate
    model and the pristine base model on harmless prompts.

    Compares the FULL next-token softmax distribution (top-K truncated for
    cost) at each position, not just the target token's log-prob. This is the
    key difference from the broken v1: if the ablation shifts the model's
    beliefs across the vocabulary while the argmax token stays the same, v1
    saw KL≈0. Full-distribution KL catches that shift.

    base_logprobs must be precomputed from the pristine model with the SAME
    prompts in the same order (see _precompute_base_logprobs).
    """
    import torch.nn.functional as F
    device = next(model.parameters()).device
    top_k = getattr(cfg, "sweep_kl_topk", 128)
    kls: list[float] = []
    for idx, p in enumerate(prompts):
        inp = tok(p, return_tensors="pt", truncation=True, max_length=cfg.max_seq_len).to(device)
        with torch.no_grad():
            out = model(**inp)
        logits = out.logits[0, :-1]  # [T-1, V]
        n = min(logits.shape[0], base_logprobs[idx].shape[0])
        if n == 0:
            continue
        logits = logits[:n]
        base_lp = base_logprobs[idx].to(device)[:n]

        # Truncate to the top-K vocab positions (by base probability) so the
        # KL is computed over the meaningful mass — cheap and stable.
        base_topk = base_lp.topk(min(top_k, base_lp.shape[-1]), dim=-1)
        base_topk_lp = base_topk.values            # [T-1, K]
        base_topk_idx = base_topk.indices           # [T-1, K]
        abl_lp = F.log_softmax(logits.float(), dim=-1)
        abl_topk_lp = abl_lp.gather(1, base_topk_idx)

        # KL(abl || base) over the top-K distribution, log_target=True.
        kl = F.kl_div(abl_topk_lp, base_topk_lp, reduction="mean", log_target=True)
        kls.append(kl.item())
    return sum(kls) / len(kls) if kls else 0.0


def _precompute_base_logprobs(model: Any, tok: Any, cfg: Any,
                              prompts: list[str]) -> dict[int, torch.Tensor]:
    """Teacher-forced full-vocab log-softmax of the pristine model on the
    given prompts. Used as the KL reference for capability-damage
    measurement. Kept on CPU to bound memory."""
    import torch.nn.functional as F
    device = next(model.parameters()).device
    out: dict[int, torch.Tensor] = {}
    for idx, p in enumerate(prompts):
        inp = tok(p, return_tensors="pt", truncation=True, max_length=cfg.max_seq_len).to(device)
        with torch.no_grad():
            logits = model(**inp).logits[0, :-1]
        out[idx] = F.log_softmax(logits.float(), dim=-1).cpu()
    return out


# ---------------------------------------------------------------------------
# 4. Sweep node — the pipeline entry point
# ---------------------------------------------------------------------------

def sweep_node(state: AbliterationState) -> dict[str, Any]:
    """Try the candidate grid (method × dir_method × layers × alpha × passes),
    pick the best, return winning overrides as state updates."""
    cfg = state["config"]
    if not getattr(cfg, "sweep_enabled", False):
        logger.info("SWEEP disabled; using configured params")
        return {"sweep_results": []}

    model = get_model()
    tok = get_tokenizer()
    base_directions = state.get("refusal_directions", {})
    harm_acts = state.get("harm_acts") or {}
    harmless_acts = state.get("harmless_acts") or {}
    # Paired output-phase activations (probe_mode auto/paired). dir_method
    # 'paired' MUST use these, not the input-phase sets — that's the whole
    # point of the recipe (same prompts, refusal vs affirmative-prefill).
    paired_refusal = state.get("paired_refusal_acts") or {}
    paired_affirm = state.get("paired_affirm_acts") or {}

    # Save pristine once — sweep restores between candidates, EXCISE gets it too.
    pristine = state.get("pristine_state_dict")
    if pristine is None:
        pristine = {k: v.clone().cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in model.state_dict().items()}

    grid = _build_candidates(cfg, layer_types=state.get("layer_types"))
    # Filter candidates to dir_methods that have probe activations. With
    # probe_mode='paired' only paired acts exist (diff_means candidates
    # would silently reuse paired directions); with 'input'/'auto'-without-
    # paired only input acts exist. An explicit mismatch is a config error —
    # warn loudly instead of running garbage candidates.
    available_dirs = {"diff_means", "svd", "leace", "whitened_svd"}
    if paired_refusal and paired_affirm:
        available_dirs.add("paired")
    elif any(c["dir_method"] == "paired" for c in grid):
        logger.warning(
            "SWEEP: 'paired' dir_method requested but no paired probe "
            "activations (probe_mode=%s); dropping paired candidates. "
            "Set probe_mode: auto or paired to collect them.",
            getattr(cfg, "probe_mode", "auto"),
        )
    filtered = [c for c in grid if c["dir_method"] in available_dirs]
    if len(filtered) != len(grid):
        logger.info("SWEEP: dropped %d candidate(s) with unavailable dir_method", len(grid) - len(filtered))
    grid = filtered
    dropped = len(set(tuple(c["target_layers"]) for c in
                      _build_candidates(cfg))) - len(set(tuple(c["target_layers"]) for c in grid))
    if dropped > 0:
        logger.info("SWEEP: dropped %d layer set(s) containing conv layers (hybrid arch)", dropped)
    logger.info("SWEEP: %d candidates (%d methods × %d dir_methods × %d layer sets × %d alphas × %d passes)",
                len(grid),
                len(set(c["method"] for c in grid)),
                len(set(c["dir_method"] for c in grid)),
                len(set(tuple(c["target_layers"]) for c in grid)),
                len(set(c["alpha"] for c in grid)),
                len(set(c["passes"] for c in grid)))

    if not grid:
        logger.info("SWEEP: empty grid, passing through")
        return {"sweep_results": []}

    # Directions depend on dir_method — cache per method to avoid recompute.
    dir_cache: dict[str, dict[int, torch.Tensor]] = {}
    good_cache: dict[str, dict[int, torch.Tensor]] = {}
    needs_good = any(c["method"] == "projected" for c in grid)
    if base_directions:
        dir_cache[cfg.dir_method] = base_directions

    test_prompts = list(DEFAULT_HARMFUL)[: cfg.n_verify_prompts]
    w_refusal = getattr(cfg, "sweep_refusal_weight", 1.0)
    w_quality = getattr(cfg, "sweep_quality_weight", 1.0)
    results: list[dict[str, Any]] = []

    # Precompute pristine-model log-probs on harmless prompts for the
    # Heretic-style KL quality metric (capability-damage measurement).
    from prompts import DEFAULT_HARMLESS
    harmless_prompts = list(DEFAULT_HARMLESS)[: cfg.n_verify_prompts]
    base_logprobs = None
    if getattr(cfg, "sweep_kl_quality", True) and harmless_prompts:
        try:
            base_logprobs = _precompute_base_logprobs(model, tok, cfg, harmless_prompts)
            logger.info("SWEEP: precomputed pristine KL reference on %d harmless prompts", len(harmless_prompts))
        except Exception as exc:
            logger.warning("SWEEP: KL reference precompute failed (%s); falling back to length proxy", exc)
            base_logprobs = None

    for i, cand in enumerate(grid):
        t0 = time.perf_counter()
        dm = cand["dir_method"]
        if dm not in dir_cache:
            # 'paired' consumes the output-phase paired activations; every
            # other dir_method consumes the input-phase harm/harmless sets.
            use_harm = paired_refusal if dm == "paired" else harm_acts
            use_harmless = paired_affirm if dm == "paired" else harmless_acts
            if not use_harm or not use_harmless:
                logger.warning(
                    "SWEEP: no probe activations for dir_method=%s "
                    "(paired activations present: %s)", dm, bool(paired_refusal)
                )
                dir_cache[dm] = base_directions or {}
            else:
                n_dirs = max(1, min(cfg.n_directions, state.get("hidden_size", 0) or cfg.n_directions))
                if needs_good or cand["method"] == "projected":
                    dir_cache[dm], _, good_cache[dm] = extract_directions(
                        use_harm, use_harmless,
                        state.get("num_layers", 0),
                        state.get("hidden_size", 0),
                        dm, n_dirs,
                        "cuda" if torch.cuda.is_available() else "cpu",
                        return_good_dirs=True,
                    )
                else:
                    dir_cache[dm], _ = extract_directions(
                        use_harm, use_harmless,
                        state.get("num_layers", 0),
                        state.get("hidden_size", 0),
                        dm, n_dirs,
                        "cuda" if torch.cuda.is_available() else "cpu",
                    )
                logger.info("SWEEP: recomputed directions for dir_method=%s (%d layers)",
                            dm, len(dir_cache[dm]))
        cand_directions = dir_cache[dm]

        # ------------------------------------------------------------------ #
        # PARALLEL PATH: dispatch this candidate to its own Modal task.
        # Each task loads the model fresh, applies the edit, quick-scores,
        # and returns. Wall-clock ~= (max candidate time) + cold start instead
        # of N * candidate time. Only worth it for grids > ~8 candidates.
        # ------------------------------------------------------------------ #
        if getattr(cfg, "sweep_parallel", False):
            results.append(_dispatch_parallel_candidate(cand, cand_directions, state, cfg))
            logger.info(
                "SWEEP %d/%d: [parallel] method=%s dir=%s layers=%s alpha=%.2f passes=%d → %s",
                i + 1, len(grid), cand["method"], cand["dir_method"],
                cand["target_layers"], cand["alpha"], cand["passes"],
                {k: results[-1].get(k) for k in ("refusal", "quality", "kl", "objective")},
            )
            continue

        _restore(model, pristine)
        _apply_candidate(model, cand_directions, pristine, cand, good_cache.get(dm))
        score = _quick_score(model, tok, cfg, test_prompts, base_logprobs=base_logprobs,
                             harmless_prompts=harmless_prompts)
        elapsed = time.perf_counter() - t0

        # Objective: prefer low refusal + low capability damage. When raw KL
        # is available use it directly (it separates candidates better than
        # the squashed 1/(1+kl) quality, which rounds to 1.00 for small KL).
        if score.get("kl") is not None:
            obj = -w_refusal * score["refusal"] - w_quality * score["kl"]
        else:
            obj = w_quality * score["quality"] - w_refusal * score["refusal"]
        results.append({**cand, **score, "objective": round(obj, 4), "elapsed_s": round(elapsed, 1)})
        kl_str = f" kl={score['kl']:.4f}" if score.get("kl") is not None else ""
        logger.info(
            "SWEEP %d/%d: method=%s dir=%s layers=%s alpha=%.2f passes=%d "
            "refusal=%.2f quality=%.4f%s obj=%.4f (%ss)",
            i + 1, len(grid), cand["method"], cand["dir_method"],
            cand["target_layers"], cand["alpha"], cand["passes"],
            score["refusal"], score["quality"], kl_str, obj, round(elapsed, 1),
        )

    # ------------------------------------------------------------------ #
    # Two-stage winner selection:
    #   1. Keyword quick-score pre-filters the grid (cheap)
    #   2. The real LLM judge re-scores the top-N finalists (accurate)
    # This fixes the keyword-vs-judge mismatch (keyword said 0%, judge 75%).
    # ------------------------------------------------------------------ #
    top = sorted(results, key=lambda r: -r["objective"])
    n_finalists = min(getattr(cfg, "sweep_judge_finalists", 5), len(top))
    finalists = top[:n_finalists]

    # Only run the LLM judge if enabled and we have a judge config.
    judge_enabled = getattr(cfg, "sweep_judge_enabled", False) and getattr(cfg, "judge_enabled", False)
    if judge_enabled and n_finalists > 1:
        from llm_api import chat_completion
        from prompts import JUDGE_PROMPT_TEMPLATE

        jmodel = getattr(cfg, "judge_model", "deepseek-v4-flash")
        jbase = getattr(cfg, "judge_base_url", None) or "https://freeinference.org/v1"
        jkey = getattr(cfg, "judge_api_key", None) or None
        jmax = getattr(cfg, "judge_max_tokens", 200)
        n_judge_prompts = min(getattr(cfg, "sweep_judge_prompts", 5), len(test_prompts))
        judge_prompts = test_prompts[:n_judge_prompts]

        for fi, fin in enumerate(finalists):
            _restore(model, pristine)
            _apply_candidate(model, dir_cache[fin["dir_method"]], pristine, fin,
                             good_cache.get(fin["dir_method"]))
            # Generate + LLM-score a small set of harmful prompts.
            refusals = 0
            qualities: list[float] = []
            for p in judge_prompts:
                try:
                    with torch.no_grad():
                        out = model.generate(
                            **tok(p, return_tensors="pt", truncation=True,
                                  max_length=cfg.max_seq_len).to(next(model.parameters()).device),
                            max_new_tokens=min(jmax, 64), do_sample=False,
                            pad_token_id=tok.eos_token_id,
                        )
                    resp = tok.decode(out[0], skip_special_tokens=True)
                except Exception:
                    resp = ""
                try:
                    vt = chat_completion(
                        JUDGE_PROMPT_TEMPLATE.format(prompt=p, response=resp),
                        model=jmodel, base_url=jbase, api_key=jkey,
                        max_tokens=150, temperature=0.0,
                    )
                    import json as _json
                    clean = vt.strip().strip("```json").strip("```").strip()
                    verdict = _json.loads(clean)
                    rs = float(verdict.get("refusal_score", 0.5))
                    qs = float(verdict.get("quality_score", 0.5))
                except Exception:
                    rs, qs = 0.5, 0.5
                refusals += rs
                qualities.append(qs)
            jrefusal = refusals / len(judge_prompts) if judge_prompts else 1.0
            jquality = sum(qualities) / len(qualities) if qualities else 0.5
            fin["judge_refusal"] = round(jrefusal, 3)
            fin["judge_quality"] = round(jquality, 3)
            fin["objective"] = round(
                w_quality * jquality - w_refusal * jrefusal, 4)
            logger.info("SWEEP judge re-score %d/%d: method=%s layers=%s alpha=%.2f "
                        "judge_refusal=%.2f judge_quality=%.2f obj=%.3f",
                        fi + 1, n_finalists, fin["method"], fin["target_layers"],
                        fin["alpha"], jrefusal, jquality, fin["objective"])

        finalists = sorted(finalists, key=lambda r: -r["objective"])
        best = min(finalists[: max(1, len(finalists) // 2)], key=lambda r: r.get("judge_refusal", r["refusal"]))
        _restore(model, pristine)
    else:
        best = min(top[: max(1, len(top) // 3)], key=lambda r: r["refusal"])

    logger.info("SWEEP winner: method=%s dir=%s layers=%s alpha=%.2f passes=%d "
                "refusal=%.2f quality=%.2f",
                best["method"], best["dir_method"], best["target_layers"],
                best["alpha"], best["passes"], best["refusal"], best["quality"])

    # Restore pristine so EXCISE starts clean with the winning params.
    _restore(model, pristine)

    return {
        "sweep_results": results,
        "sweep_finalists": finalists,
        "sweep_best": best,
        "method": best["method"],
        "dir_method": best["dir_method"],
        "target_layers": best["target_layers"],
        "alpha": best["alpha"],
        "passes": best["passes"],
        "target_weights": best["target_weights"],
        "pristine_state_dict": pristine,
    }