"""EXCISE node: refusal-direction weight projection across architectures.

Implements the in-place weight modification that ablates the refusal
direction from a model's o_proj / down_proj / per-expert down weights.
Works for dense, MoE, and diffusion-encoder (text encoder) architectures.
"""
from __future__ import annotations

import logging
from typing import Any

import torch

from state import AbliterationState

_log = logging.getLogger(__name__)


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


def _project_2d(weight: torch.Tensor, d: torch.Tensor, alpha: float) -> None:
    """In-place refusal-direction subtraction on a 2D weight.

    ``W -= alpha * d (d^T W)`` via the einsum ``i,j->ij``.
    Raises a clear error if the tensor is quantized (4/8-bit) and cannot
    be mutated in place.
    """
    try:
        weight.sub_(alpha * torch.einsum("i,j->ij", d, d @ weight))
    except RuntimeError as exc:
        raise RuntimeError(
            "EXCISE: in-place weight modification failed — the model appears "
            "to be quantized (4-bit / 8-bit). Refusal-direction projection "
            "requires dequantized (fp16/bf16/fp32) weights. Re-load the model "
            "without `quantize` before running the pipeline."
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


def excise_node(state: AbliterationState) -> dict[str, Any]:
    """Project the refusal direction out of the model's weights in place.

    Returns a partial state dict with:
      - ``projection_applied``: True
      - ``passes_completed``: incremented
      - ``excise_history``: appended history entries for this pass
    """
    model = state["model_obj"]
    arch = state["architecture"]
    directions = state["refusal_directions"]
    target_layers = state.get("target_layers", [])
    target_weights = state.get("target_weights", [])
    alpha = state["config"].alpha
    passes_completed = state.get("passes_completed", 0) + 1

    device, dtype = _device_dtype_of(model)
    decoder = _decoder_of(model, arch)

    history: list = list(state.get("excise_history", []))

    for layer_idx in target_layers:
        if layer_idx not in directions:
            continue

        d = directions[layer_idx]
        # Some distill methods return [n_dirs, hidden]; take the strongest.
        # Warn when extra directions are silently dropped (Bug 7).
        if torch.is_tensor(d) and d.dim() > 1:
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
        if (
            "o_proj" in target_weights
            and hasattr(layer, "self_attn")
            and hasattr(layer.self_attn, "o_proj")
        ):
            W = layer.self_attn.o_proj.weight.data
            _project_2d(W, d, alpha)
            modified_weights.append("o_proj")

        # --- down_proj (MLP) ---
        if "down_proj" in target_weights:
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
                W = layer.mlp.down_proj.weight.data
                _project_2d(W, d, alpha)
                modified_weights.append("down_proj")

        # --- per-expert down weights (MoE) ---
        if "expert.down" in target_weights and hasattr(layer, "experts"):
            for e_name, e_param in layer.experts.named_parameters():
                if "down" not in e_name.lower():
                    continue
                if e_param.dim() == 3:
                    # [E, H, D] per-expert weight
                    _project_3d_expert(e_param.data, d, alpha)
                elif e_param.dim() == 2:
                    _project_2d(e_param.data, d, alpha)
                else:
                    # Unexpected rank — skip defensively rather than crash.
                    continue
                modified_weights.append(f"expert.down/{e_name}")

        history.append({"layer": layer_idx, "modified_weights": modified_weights})

    return {
        "projection_applied": True,
        "passes_completed": passes_completed,
        "excise_history": history,
    }
