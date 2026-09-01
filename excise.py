"""EXCISE node: refusal-direction weight projection across architectures.

Implements the in-place weight modification that ablates the refusal
direction from a model's o_proj / down_proj / per-expert down weights.
Works for dense, MoE, and diffusion-encoder (text encoder) architectures.
"""
from __future__ import annotations
from model_registry import get_model

import logging
from typing import Any

import torch

from state import AbliterationState

_log = logging.getLogger(__name__)

# Methods EXCISE can actually realize, honestly. MPOA gets the magnitude-
# preserving orthogonal ablation; the rest (plain projection, plus the sweep's
# "advanced" and "direct_ablation" aliases, which are mathematically EXACTLY a
# plain projection) run the plain projection. Everything else — the sweep's
# steering / bias_vectors / lora / projected — has a distinct implementation in
# sweep.py that EXCISE has no portable equivalent for, so EXCISE will NOT
# silently substitute a different algorithm (P1-2): it raises instead. This
# singleton is the single source of truth that SWEEP uses to constrain its
# candidate box and REFLEXION uses to restrict the switch_method candidate pool.
EXCISE_REALIZED_METHODS: frozenset[str] = frozenset(
    {"mpoa", "projection", "advanced", "direct_ablation"}
)


def _decoder_of(model: Any, arch: str) -> Any:
    """Return the module whose ``.layers`` holds the transformer blocks.

    For diffusion pipelines we modify the text encoder, reached at
    ``model.model``. For everything else we grab the first named child
    module that exposes ``.layers``.
    """
    if arch == "diffusion_encoder":
        return model.model
    for _name, mod in model.named_children():
        if hasattr(mod, "layers"):
            return mod
    # Fall back to model.model if it exists (covers some wrapped CausalLMs).
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    raise ValueError(
        "EXCISE: could not locate a decoder module with `.layers` on the "
        f"model of type {type(model).__name__}"
    )


def _device_dtype_of(model: Any):
    """Best-effort device + dtype for the model's parameters."""
    device = "cpu"
    dtype = torch.float32
    try:
        param = next(model.parameters())
        device = param.device
        dtype = param.dtype
    except (StopIteration, RuntimeError):
        pass
    return device, dtype


def _restore_pristine(model: Any, pristine: dict[str, Any] | None, device) -> None:
    """Restore the model's weights from the pristine snapshot.

    Used for transactional rollback (P1-1): if a projection pass raises partway
    through, the model is reverted to its pre-excise state so an exception never
    leaves partially-modified weights behind. The device is derived with
    ``_device_dtype_of`` (NOT ``model.device``, which many HF/device_map models
    do not expose reliably).

    Defensively coerces python-list values back to tensors — pre-fix checkpoints
    serialized tensors as lists, so a restored (un-resumed) snapshot may still
    hold lists.
    """
    if not pristine:
        return
    inplace: dict[str, Any] = {}
    for k, v in pristine.items():
        if isinstance(v, torch.Tensor):
            inplace[k] = v.to(device=device)
        elif isinstance(v, (list, tuple)):
            inplace[k] = torch.tensor(v, device=device)
        else:
            inplace[k] = v
    model.load_state_dict(inplace)


def _project_2d(weight: torch.Tensor, d: torch.Tensor, alpha: float) -> None:
    """In-place refusal-direction subtraction on a 2D weight.

    ``W -= alpha * d (d^T W)`` via the einsum ``i,j->ij``.
    Raises a clear error if the tensor is quantized (4/8-bit) and cannot
    be mutated in place. The original exception is chained so real bugs
    (device/dtype mismatch, shape issues) are visible, not masked.
    """
    d = d.to(dtype=weight.dtype, device=weight.device)
    if weight.dim() != 2:
        return
    if d.dim() > 1:
        # batch dims from probe hooks must be squeezed before projection —
        # otherwise the guard below silently skips and the ablation no-ops
        d = d.reshape(-1)
    if d.shape[0] != weight.shape[1]:
        # Hybrid architectures (LFM conv layers, fused projections) can
        # expose weights whose input dim != hidden. Skip rather than corrupt.
        return
    try:
        weight.sub_(alpha * torch.einsum("i,j->ij", d, d @ weight))
    except RuntimeError as exc:
        if "quantized" in str(exc).lower() or "bitsandbytes" in str(exc).lower():
            raise RuntimeError(
                "EXCISE: in-place weight modification failed — the model appears "
                "to be quantized (4-bit / 8-bit). Refusal-direction projection "
                "requires dequantized (fp16/bf16/fp32) weights. Re-load the model "
                "without `quantize` before running the pipeline."
            ) from exc
        raise RuntimeError(
            f"EXCISE: in-place weight projection failed: {exc} "
            f"(weight {tuple(weight.shape)} {weight.dtype} {weight.device}, "
            f"d {tuple(d.shape)} {d.dtype} {d.device})"
        ) from exc


def _project_2d_mpoa(weight: torch.Tensor, d: torch.Tensor, alpha: float) -> None:
    """Magnitude-preserving orthogonal ablation (MPOA) on a 2D weight.

    ``W -= alpha * d (d^T W)`` like ``_project_2d``, then rescales the whole
    matrix back to its original Frobenius norm. This is the variant used by
    the successful LFM2.5 abliteration on PinoCookie (alpha 2.0, all six
    attention blocks): it removes the refusal direction without collapsing
    the layer's output scale, so high alphas (>= 1.0) stay usable.

    The rescale uses the *global* norm before/after so the projection is
    purely directional (the removed mass is not silently amplified).
    """
    d = d.to(dtype=weight.dtype, device=weight.device)
    if weight.dim() != 2:
        return
    if d.dim() > 1:
        # batch dims from probe hooks must be squeezed — else silent no-op
        d = d.reshape(-1)
    if d.shape[0] != weight.shape[1]:
        return
    try:
        orig_norm = weight.norm().clamp(min=1e-8)
        weight.sub_(alpha * torch.einsum("i,j->ij", d, d @ weight))
        new_norm = weight.norm().clamp(min=1e-8)
        weight.mul_(orig_norm / new_norm)
    except RuntimeError as exc:
        if "quantized" in str(exc).lower() or "bitsandbytes" in str(exc).lower():
            raise RuntimeError(
                "EXCISE: in-place weight modification failed — the model appears "
                "to be quantized (4-bit / 8-bit). MPOA requires dequantized "
                "(fp16/bf16/fp32) weights. Re-load the model without `quantize`."
            ) from exc
        raise RuntimeError(
            f"EXCISE: in-place MPOA projection failed: {exc} "
            f"(weight {tuple(weight.shape)} {weight.dtype} {weight.device}, "
            f"d {tuple(d.shape)} {d.dtype} {d.device})"
        ) from exc


def _project_3d_expert(weight: torch.Tensor, d: torch.Tensor, alpha: float) -> None:
    """In-place projection on a 3D per-expert weight [E, H, D]."""
    try:
        proj = torch.einsum("eij,i->ej", weight, d)
        proj = torch.einsum("i,ej->eij", d, proj)
        weight.sub_(alpha * proj)
    except RuntimeError as exc:
        raise RuntimeError(
            "EXCISE: in-place expert weight modification failed — the model "
            "appears to be quantized (4-bit / 8-bit). Refusal-direction "
            "projection requires dequantized weights. Re-load the model "
            "without `quantize` before running the pipeline."
        ) from exc


def _project_3d_expert_mpoa(weight: torch.Tensor, d: torch.Tensor, alpha: float) -> None:
    """Magnitude-preserving orthogonal ablation on a 3D per-expert weight."""
    try:
        orig_norm = weight.norm().clamp(min=1e-8)
        proj = torch.einsum("eij,i->ej", weight, d)
        proj = torch.einsum("i,ej->eij", d, proj)
        weight.sub_(alpha * proj)
        new_norm = weight.norm().clamp(min=1e-8)
        weight.mul_(orig_norm / new_norm)
    except RuntimeError as exc:
        raise RuntimeError(
            "EXCISE: in-place expert MPOA failed — the model appears to be "
            "quantized (4-bit / 8-bit). MPOA requires dequantized weights."
        ) from exc


def excise_node(state: AbliterationState) -> dict[str, Any]:
    """Project the refusal direction out of the model's weights in place.

    Honors sweep/reflexion-selected overrides: ``method`` (plain projection
    or MPOA), ``alpha``, ``target_layers``, ``target_weights``. Previously
    the node always used the base config's alpha and plain projection, so a
    sweep winner's alpha/method was silently discarded (P0-6).

    Returns a partial state dict with:
      - ``projection_applied``: True
      - ``passes_completed``: incremented
      - ``excise_history``: appended history entries for this pass
      - ``pristine_state_dict``: saved before first excise pass for restoration
    """
    model = get_model()
    arch = state["architecture"]
    directions = state["refusal_directions"]
    target_layers = state.get("target_layers", [])
    target_weights = state.get("target_weights", [])
    # Honor sweep/reflexion overrides; fall back to config values.
    method = state.get("method") or state["config"].method
    alpha = state.get("alpha", state["config"].alpha)
    passes_completed = state.get("passes_completed", 0) + 1

    # P1-1/P1-2: a sweep/reflexion may select a method EXCISE has no distinct
    # implementation for (steering, bias_vectors, lora, projected, ...).
    # Previously it silently ran a plain projection, so reflexion THOUGHT it
    # switched methods when it didn't and a sweep winner's behavior was lost.
    # Now we fail loudly instead: an unrealizable method raises so the
    # discrepancy is never hidden. SWEEP and REFLEXION constrain their own
    # candidate/method pools to EXCISE_REALIZED_METHODS, so this is reached
    # only on a misconfiguration or an external override.
    if method not in EXCISE_REALIZED_METHODS:
        raise ValueError(
            f"EXCISE cannot realize method={method!r}. Implemented methods are "
            f"{sorted(EXCISE_REALIZED_METHODS)}. The sweep/reflexion must limit "
            "its method pool to these; failing loudly rather than substituting "
            "a different ablation algorithm (P1-2)."
        )

    _log.info(
        "EXCISE pass %d: method=%s alpha=%.2f layers=%s weights=%s",
        passes_completed, method, alpha, target_layers, target_weights,
    )

    mpoa = str(method).lower() == "mpoa"

# Save pristine model state before the FIRST excise pass so retries
    # (cumulative weight damage from earlier passes) can restore it later.
    pristine = state.get("pristine_state_dict")
    if pristine is None:
        pristine = {
            k: v.clone().cpu() if isinstance(v, torch.Tensor) else v
            for k, v in model.state_dict().items()
        }

    device, dtype = _device_dtype_of(model)

    # Restore pristine weights before applying this pass's projections.
    # This avoids cumulative damage across retries (P0-5). Device is derived
    # via _device_dtype_of (model.device is unreliable for device_map models).
    if passes_completed > 1:
        _log.info("EXCISE pass %d: restoring pristine weights before projection", passes_completed)
        _restore_pristine(model, pristine, device)

    decoder = _decoder_of(model, arch)

    history: list = list(state.get("excise_history", []))

    # Transactional rollback (P1-1): if any projection raises midway, restore
    # the pristine snapshot before propagating — an exception must never leave
    # partially-modified in-memory weights.
    try:
        for layer_idx in target_layers:
            if layer_idx not in directions:
                continue

            d = directions[layer_idx]
            # Some distill methods return [n_dirs, hidden]; take the strongest.
            # Warn when extra directions are silently dropped (Bug 7).
            if torch.is_tensor(d) and d.dim() > 1:
                # Distill methods can return directions of any rank >= 1
                # ([n_dirs, hidden] from SVD, [n_dirs, c, c] if the upstream
                # SVD got batched by a 3D activation stack, etc.). We want the
                # single strongest direction as a flat [hidden] vector.
                # ``reshape(-1)`` would smear across axes; instead flatten only
                # the leading direction-selection axes and take row 0.
                if d.dim() > 2:
                    _log.warning(
                        "EXCISE: layer %s direction has unexpected rank %d "
                        "(shape=%s); collapsing leading dims. Check PROBE "
                        "activation shapes — a 3D harm_stack batched the SVD.",
                        layer_idx,
                        d.dim(),
                        tuple(d.shape),
                    )
                    # Collapse everything except the last axis into one "dirs"
                    # axis so d[0] reliably yields the feature-space vector.
                    d = d.reshape(-1, d.shape[-1])
                n_directions = d.shape[0]
                if n_directions > 1:
                    _log.warning(
                        "EXCISE: layer %s returned %d refusal directions; "
                        "only the first is projected (the rest are discarded).",
                        layer_idx,
                        n_directions,
                    )
                d = d[0]
            # Normalize once on the model's primary device/dtype. Per-weight
            # device placement is re-applied inside each projection branch
            # (Bug 6: weights may live on different devices, e.g. device_map).
            d = d.to(device=device, dtype=dtype)
            # clamp(min=...) keeps epsilon meaningful under fp16/bf16 where
            # adding 1e-8 to a ~1.0 norm would round away to 1.0 (Bug 5).
            d = d / d.norm().clamp(min=1e-8)

            layer = decoder.layers[layer_idx]
            modified_weights: list = []

            # --- o_proj (attention output) ---
            # Try both self_attn.o_proj and linear_attn.out_proj; Qwen3.5 hybrid
            # layers use either path depending on the layer. Also handle
            # LiquidAI LFM naming (self_attn.out_proj) generically.
            if "o_proj" in target_weights:
                o_modified = False
                o_weights = []
                if hasattr(layer, "self_attn"):
                    for wname in ("o_proj", "out_proj"):
                        if hasattr(layer.self_attn, wname):
                            o_weights.append(getattr(layer.self_attn, wname).weight.data)
                if hasattr(layer, "linear_attn") and hasattr(layer.linear_attn, "out_proj"):
                    o_weights.append(layer.linear_attn.out_proj.weight.data)
                for W in o_weights:
                    if mpoa:
                        _project_2d_mpoa(W, d, alpha)
                    else:
                        _project_2d(W, d, alpha)
                    o_modified = True
                if o_modified:
                    modified_weights.append("o_proj")

            # --- down_proj (MLP) ---
            # Handles both standard llama naming (mlp.down_proj) and LiquidAI
            # LFM naming (mlp.w2) generically.
            if "down_proj" in target_weights:
                down_modified = False
                mlp = getattr(layer, "mlp", None)
                if mlp is not None:
                    for wname in ("down_proj", "w2"):
                        if hasattr(mlp, wname):
                            if mpoa:
                                _project_2d_mpoa(getattr(mlp, wname).weight.data, d, alpha)
                            else:
                                _project_2d(getattr(mlp, wname).weight.data, d, alpha)
                            down_modified = True
                if down_modified:
                    modified_weights.append("down_proj")

            # --- per-expert down weights (MoE) ---
            if "expert.down" in target_weights and hasattr(layer, "experts"):
                for e_name, e_param in layer.experts.named_parameters():
                    if "down" not in e_name.lower():
                        continue
                    if e_param.dim() == 3:
                        # [E, H, D] per-expert weight
                        if mpoa:
                            _project_3d_expert_mpoa(e_param.data, d, alpha)
                        else:
                            _project_3d_expert(e_param.data, d, alpha)
                    elif e_param.dim() == 2:
                        if mpoa:
                            _project_2d_mpoa(e_param.data, d, alpha)
                        else:
                            _project_2d(e_param.data, d, alpha)
                    else:
                        # Unexpected rank — skip defensively rather than crash.
                        continue
                    modified_weights.append(f"expert.down/{e_name}")

            history.append({"layer": layer_idx, "modified_weights": modified_weights})
    except Exception:
        _log.warning(
            "EXCISE pass %d failed midway; restoring pristine weights before "
            "propagating so no partial modification survives.",
            passes_completed,
        )
        _restore_pristine(model, pristine, device)
        raise

    return {
        "projection_applied": True,
        "passes_completed": passes_completed,
        "excise_history": history,
        "pristine_state_dict": pristine,
        "method": method,  # effective method actually realized (P1-1)
    }
