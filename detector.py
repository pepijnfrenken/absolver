"""Architecture detection for the Absolver abliteration pipeline.

Inspects a loaded HuggingFace model (or the text-encoder submodel of a
diffusion pipeline) and returns a descriptor dict telling downstream nodes
how to treat it (dense vs. MoE vs. diffusion text-encoder, which projection
weights are present, how many layers, etc.).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import torch

# transformers / diffusers are required at runtime but the module must still
# import (e.g. for py_compile) even if they are absent in a stripped env.
try:
    import transformers  # noqa: F401  (imported for side-effect / availability)
except Exception:  # pragma: no cover - env without transformers
    transformers = None  # type: ignore[assignment]

try:
    import diffusers  # noqa: F401
except Exception:  # pragma: no cover - env without diffusers
    diffusers = None  # type: ignore[assignment]


class UnsupportedArchitecture(Exception):
    """Raised when a model does not match any known decoder pattern.

    The message includes a debug dump of the modules that *were* found so the
    user can see how close the model came to matching and what it actually
    looks like.
    """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LAYER_RE = re.compile(r".*layers\.(\d+)$")


def _is_meta(module: Any) -> bool:
    """True if `module` has parameters/buffers that all live on the meta device
    (or that have no real storage), meaning we cannot inspect them directly."""
    try:
        params = list(module.parameters(recurse=False))
        buffers = list(module.buffers(recurse=False))
    except Exception:
        return False
    if not params and not buffers:
        # Fall back to recursion: a parent w/o own params can still hold meta
        # tensors in children. Only consider "leaf-ish" modules here.
        try:
            for p in module.parameters(recurse=True):
                if p.is_meta:
                    return True
        except Exception:
            return False
        return False
    for p in params:
        if getattr(p, "is_meta", False):
            return True
    for b in buffers:
        if getattr(b, "is_meta", False):
            return True
    return False


def _to_cpu(module: Any) -> Any:
    """Move a small inspection module to CPU so we can introspect it.

    Returns the same module after a best-effort `.to('cpu')`. Wrapped so a
    failure (e.g. a QuantizedLinear) never blocks detection — we only *need*
    to inspect sub-module *names* and shapes, not real weights.
    """
    try:
        # to_empty would be ideal but it needs a dispatch; just attempt .to.
        module.to("cpu")
    except Exception:
        pass
    return module


def _safe_getattr(obj: Any, name: str) -> Any:
    """getattr that swallows everything — meta tensors, missing attrs, etc."""
    try:
        return getattr(obj, name)
    except Exception:
        return None


def _inspect_decoder_layer(layer: Any) -> Dict[str, Any]:
    """Probe a single decoder layer for the markers we care about.

    Real HF decoders nest the projection linear layers (e.g. Llama):
        layer.self_attn.o_proj
        layer.mlp.down_proj
    MoE variants additionally expose either:
        layer.block_sparse_moe           (Mixtral)
        layer.experts                    (Qwen-MoE / DeepSeek-MoE)
    We walk the layer's whole named_modules() subtree so we find these
    regardless of the exact nesting depth.
    """
    layer = _to_cpu(layer)

    has_block_sparse_moe = False
    has_experts = False
    has_down_proj = False
    has_o_proj = False
    mlp = _safe_getattr(layer, "mlp")

    try:
        for name, _mod in layer.named_modules():
            if not name:
                continue
            # Use the basename so `self_attn.o_proj`, `mlp.down_proj`, and a
            # bare `o_proj` all match.
            base_name = name.rsplit(".", 1)[-1]
            if base_name == "block_sparse_moe":
                has_block_sparse_moe = True
            elif base_name == "experts":
                has_experts = True
            elif base_name == "down_proj":
                has_down_proj = True
            elif base_name == "o_proj":
                has_o_proj = True
    except Exception:
        # named_modules can fail on exotic quantized/meta modules; fall back
        # to direct attr checks.
        if _safe_getattr(layer, "block_sparse_moe") is not None:
            has_block_sparse_moe = True
        if _safe_getattr(layer, "experts") is not None:
            has_experts = True
        if mlp is not None and _safe_getattr(mlp, "down_proj") is not None:
            has_down_proj = True
        elif _safe_getattr(layer, "down_proj") is not None:
            has_down_proj = True
        if _safe_getattr(layer, "o_proj") is not None:
            has_o_proj = True

    return {
        "has_block_sparse_moe": has_block_sparse_moe,
        "has_experts": has_experts,
        "has_down_proj": has_down_proj,
        "has_o_proj": has_o_proj,
        "mlp": mlp,
    }


def _hidden_size_from_config(model: Any) -> Optional[int]:
    cfg = _safe_getattr(model, "config")
    if cfg is None:
        return None
    for attr in ("hidden_size", "d_model", "n_embd"):
        v = _safe_getattr(cfg, attr)
        if isinstance(v, int):
            return v
    return None


def _count_experts(layer: Any) -> Optional[int]:
    """Best-effort expert count for an MoE layer."""
    experts = _safe_getattr(layer, "experts")
    if experts is None:
        bsm = _safe_getattr(layer, "block_sparse_moe")
        if bsm is not None:
            experts = _safe_getattr(bsm, "experts")
    if experts is None:
        return None
    # nn.ModuleList -> len(); list/tuple -> len(); dict -> len
    try:
        return len(experts)
    except Exception:
        # Some impls expose num_experts directly.
        for attr in ("num_experts", "num_local_experts", "n_experts"):
            v = _safe_getattr(experts, attr)
            if isinstance(v, int):
                return v
        return None

def _layer_types(model: Any, layers: Any) -> Optional[List[str]]:
    """Return per-layer type tags if the model exposes them (e.g. sliding vs
    full attention in Gemma2/Qwen2.5). None if not available."""
    cfg = _safe_getattr(model, "config")
    if cfg is None:
        return None
    sliding = _safe_getattr(cfg, "sliding_window")
    layer_types = _safe_getattr(cfg, "layer_types") or _safe_getattr(
        cfg, "attention_layer_types"
    )
    if isinstance(layer_types, (list, tuple)):
        try:
            return [str(t) for t in layer_types][: len(layers)]
        except Exception:
            return None
    # Heuristic: Gemma2-style alternating sliding/full.
    if sliding is not None:
        try:
            return [
                ("sliding" if (i % 2 == 0) else "full") for i in range(len(layers))
            ]
        except Exception:
            return None
    return None


def _build_target_weights(has_o_proj: bool, has_down_proj: bool, has_experts: bool) -> List[str]:
    tw: List[str] = []
    if has_o_proj:
        tw.append("o_proj")
    if has_down_proj:
        tw.append("down_proj")
    if has_experts:
        tw.append("expert.down")
    if not tw:
        tw = ["o_proj", "down_proj"]
    return tw


def _describe_modules(model: Any, limit: int = 40) -> List[str]:
    """Top-N named_modules dump for error messages."""
    out: List[str] = []
    try:
        for i, (name, _mod) in enumerate(model.named_modules()):
            if i >= limit:
                out.append("...")
                break
            if name:
                out.append(name)
    except Exception as exc:
        out.append(f"<named_modules failed: {exc}>")
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def detect_architecture(model: Any) -> Dict[str, Any]:
    """Inspect `model` and return an architecture descriptor dict.

    Detection order (see plan §5):
      1. ``model.model.layers``           -> decoder (dense or MoE)
      2. layers[0].block_sparse_moe       -> MoE
      3. layers[0].experts                -> MoE
      4. layers[0].mlp.down_proj          -> dense
      5. ``model.text_encoder.model.layers`` -> diffusion_encoder
      6. walk named_modules() for ``.*layers.\\d+`` as last resort

    Raises:
        UnsupportedArchitecture: if none of the patterns match.
    """
    text_encoder_model: Optional[Any] = None

    # ------------------------------------------------------------------
    # 1-4. Standard decoder path: model.model.layers
    # ------------------------------------------------------------------
    base = _safe_getattr(model, "model")
    layers = _safe_getattr(base, "layers") if base is not None else None

    if layers is not None and len(layers) > 0:
        first = layers[0]
        info = _inspect_decoder_layer(first)
        is_moe = info["has_block_sparse_moe"] or info["has_experts"]

        has_o_proj = info["has_o_proj"]
        has_down_proj = info["has_down_proj"]
        has_experts = info["has_experts"]

        if is_moe:
            architecture = "moe"
            num_experts: Optional[int] = _count_experts(first)
        else:
            architecture = "dense"
            num_experts = None

        hidden_size = _hidden_size_from_config(model) or _hidden_size_from_config(base)
        if hidden_size is None:
            # Infer from the down_proj / o_proj in_features.
            for cand in (
                _safe_getattr(first, "o_proj"),
                _safe_getattr(info["mlp"], "down_proj") if info["mlp"] else None,
                _safe_getattr(first, "input_layernorm"),
            ):
                ifs = _safe_getattr(cand, "in_features") if cand is not None else None
                if isinstance(ifs, int):
                    hidden_size = ifs
                    break

        return {
            "architecture": architecture,
            "hidden_size": hidden_size,
            "num_layers": len(layers),
            "num_experts": num_experts,
            "layer_types": _layer_types(model, layers),
            "target_weights": _build_target_weights(
                has_o_proj, has_down_proj, has_experts
            ),
            "has_o_proj": has_o_proj,
            "has_down_proj": has_down_proj,
            "has_experts": has_experts,
            "text_encoder_model": None,
        }

    # ------------------------------------------------------------------
    # 5. Diffusion text-encoder path: model.text_encoder.model.layers
    # ------------------------------------------------------------------
    text_encoder = _safe_getattr(model, "text_encoder")
    te_layers = None
    te_base = None
    if text_encoder is not None:
        te_base = _safe_getattr(text_encoder, "model")
        te_layers = _safe_getattr(te_base, "layers") if te_base is not None else None

    if te_layers is not None and len(te_layers) > 0:
        first = te_layers[0]
        info = _inspect_decoder_layer(first)
        has_o_proj = info["has_o_proj"]
        has_down_proj = info["has_down_proj"]
        has_experts = info["has_experts"]

        # Diffusion text encoders are dense transformers; MoE is not a thing
        # here but we still report what we found.
        num_experts = _count_experts(first) if has_experts else None

        hidden_size = _hidden_size_from_config(text_encoder) or _hidden_size_from_config(
            te_base
        ) or _hidden_size_from_config(model)
        if hidden_size is None:
            for cand in (
                _safe_getattr(first, "o_proj"),
                _safe_getattr(info["mlp"], "down_proj") if info["mlp"] else None,
            ):
                ifs = _safe_getattr(cand, "in_features") if cand is not None else None
                if isinstance(ifs, int):
                    hidden_size = ifs
                    break

        text_encoder_model = text_encoder

        return {
            "architecture": "diffusion_encoder",
            "hidden_size": hidden_size,
            "num_layers": len(te_layers),
            "num_experts": num_experts,
            "layer_types": _layer_types(text_encoder, te_layers),
            "target_weights": _build_target_weights(
                has_o_proj, has_down_proj, has_experts
            ),
            "has_o_proj": has_o_proj,
            "has_down_proj": has_down_proj,
            "has_experts": has_experts,
            "text_encoder_model": text_encoder_model,
        }

    # ------------------------------------------------------------------
    # 6. Last resort: walk named_modules() for .*layers\.\d+
    # ------------------------------------------------------------------
    fallback_layers: Dict[str, Any] = {}
    fallback_prefixes: Dict[str, Any] = {}
    try:
        for name, mod in model.named_modules():
            m = _LAYER_RE.match(name)
            if m:
                prefix = name[: m.start(1)].rstrip(".")
                bucket = fallback_layers.setdefault(prefix, {})
                bucket[int(m.group(1))] = mod
                fallback_prefixes[prefix] = mod
    except Exception:
        pass

    if fallback_layers:
        # Pick the prefix with the most layers — most likely the real decoder.
        prefix = max(fallback_layers.keys(), key=lambda p: len(fallback_layers[p]))
        layer_map = fallback_layers[prefix]
        idxs = sorted(layer_map.keys())
        # Require contiguous-ish numbering starting at 0 to avoid grabbing
        # attention-layer lists etc.
        if idxs and idxs[0] == 0 and len(idxs) >= 2:
            first = layer_map[0]
            info = _inspect_decoder_layer(first)
            has_o_proj = info["has_o_proj"]
            has_down_proj = info["has_down_proj"]
            has_experts = info["has_experts"]
            is_moe = info["has_block_sparse_moe"] or info["has_experts"]

            architecture = "moe" if is_moe else "dense"
            num_experts = _count_experts(first) if is_moe else None

            hidden_size = _hidden_size_from_config(model)
            if hidden_size is None:
                for cand in (
                    _safe_getattr(first, "o_proj"),
                    _safe_getattr(info["mlp"], "down_proj") if info["mlp"] else None,
                ):
                    ifs = _safe_getattr(cand, "in_features") if cand is not None else None
                    if isinstance(ifs, int):
                        hidden_size = ifs
                        break

            return {
                "architecture": architecture,
                "hidden_size": hidden_size,
                "num_layers": len(idxs),
                "num_experts": num_experts,
                "layer_types": None,
                "target_weights": _build_target_weights(
                    has_o_proj, has_down_proj, has_experts
                ),
                "has_o_proj": has_o_proj,
                "has_down_proj": has_down_proj,
                "has_experts": has_experts,
                "text_encoder_model": None,
            }

    # ------------------------------------------------------------------
    # Nothing matched.
    # ------------------------------------------------------------------
    found = _describe_modules(model)
    raise UnsupportedArchitecture(
        "Could not detect architecture for model of type "
        f"{type(model).__name__!r}. "
        "No `model.model.layers`, `text_encoder.model.layers`, or "
        "`.*layers\\d+` module sequence was found. "
        f"Top-level attrs: {sorted(vars(model).keys()) if hasattr(model, '__dict__') else '<n/a>'}. "
        f"First named_modules: {found}"
    )
