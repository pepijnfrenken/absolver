"""PROBE node: harvest per-layer activations via forward hooks.

Registers a forward hook on every decoder layer (or text-encoder layer for
diffusion pipelines) that captures the last-token hidden state, then runs the
harmful and harmless prompt sets through the model under ``torch.no_grad``.
For MoE models, additionally collects router logits via
``output_router_logits=True``.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import torch

from prompts import DEFAULT_HARMFUL, DEFAULT_HARMLESS
from state import AbliterationState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_device(model: Any) -> Any:
    """Return the model's device, falling back to the device of its first
    parameter (or CPU if the model has no parameters)."""
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cpu")


def _find_layers(model: Any, arch: str | None):
    """Locate the ``layers`` ModuleList to hook.

    For diffusion text encoders we go straight to ``model.model.layers``.
    Otherwise we walk the direct children for the first submodule exposing a
    ``.layers`` attribute, with a final fallback to ``model.model.layers``.
    """
    if arch == "diffusion_encoder":
        base = getattr(model, "model", None)
        if base is not None and hasattr(base, "layers"):
            return base.layers

    for _name, mod in model.named_children():
        if hasattr(mod, "layers"):
            return mod.layers

    base = getattr(model, "model", None)
    if base is not None and hasattr(base, "layers"):
        return base.layers

    return None


def _to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    """Move a tokenizer output batch to ``device``."""
    out = {}
    for k, v in batch.items():
        if hasattr(v, "to"):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _make_hook(name: int, store: dict[int, list[torch.Tensor]]):
    """Forward hook capturing the last-token hidden state.

    Handles both 3D outputs ``[batch, seq, hidden]`` (takes ``[:, -1, :]``)
    and 2D outputs ``[batch, hidden]``. Detaches, moves to CPU, casts to
    float32 so downstream DISTILL math is deterministic and device-agnostic.
    """

    def hook(_module, _inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if not hasattr(hs, "dim"):
            return
        try:
            if hs.dim() == 3:
                store[name].append(hs[:, -1, :].detach().cpu().to(torch.float32))
            elif hs.dim() == 2:
                store[name].append(hs.detach().cpu().to(torch.float32))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("hook capture failed on layer %d: %s", name, exc)

    return hook


def _find_moe_layer_indices(layers, num_layers: int) -> list[int]:
    """Return indices of layers that look like MoE experts (have a router /
    block_sparse_moe / experts attribute)."""
    moe_idxs: list[int] = []
    for i in range(num_layers):
        layer = layers[i]
        bsm = getattr(layer, "block_sparse_moe", None)
        if bsm is not None and hasattr(bsm, "gate"):
            moe_idxs.append(i)
            continue
        if getattr(layer, "router", None) is not None:
            moe_idxs.append(i)
            continue
        if getattr(layer, "moe", None) is not None:
            moe_idxs.append(i)
            continue
        if getattr(layer, "experts", None) is not None:
            moe_idxs.append(i)
    return moe_idxs


def _collect_router_logits(
    model: Any,
    tok: Any,
    prompts: list[str],
    device: Any,
    layers,
    num_layers: int,
) -> dict[int, torch.Tensor] | None:
    """Run each prompt with ``output_router_logits=True`` and stack the
    last-token router logit per MoE layer.

    Returns ``None`` if the model has no MoE layers or the forward pass does
    not expose ``router_logits``.
    """
    moe_idxs = _find_moe_layer_indices(layers, num_layers)
    if not moe_idxs:
        return None

    collected: dict[int, list[torch.Tensor]] = defaultdict(list)

    for p in prompts:
        try:
            inp = tok(p, return_tensors="pt", truncation=True, max_length=128)
        except Exception as exc:
            logger.warning("tokenize failed for router pass (%s): %s", p[:40], exc)
            continue
        inp = _to_device(inp, device)
        try:
            with torch.no_grad():
                out = model(**inp, output_router_logits=True)
        except TypeError:
            # Model doesn't accept output_router_logits at all.
            logger.info("model ignores output_router_logits; skipping router harvest")
            return None
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("router-logits forward failed: %s", exc)
            continue

        rl = getattr(out, "router_logits", None)
        if not rl:
            continue

        # rl is typically a tuple/list of [n_tokens, n_experts] tensors, one
        # per MoE layer in module order. Map them onto our detected indices.
        for slot, layer_idx in enumerate(moe_idxs):
            if slot >= len(rl):
                break
            t = rl[slot]
            if not hasattr(t, "dim"):
                continue
            try:
                if t.dim() >= 2:
                    # [n_tokens, n_experts] -> last token
                    t = t[-1]
                collected[layer_idx].append(
                    t.detach().cpu().to(torch.float32)
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("router logit stack failed layer %d: %s", layer_idx, exc)

    if not collected:
        return None

    # Stack per-prompt router logits into [n_prompts, n_experts] per layer.
    stacked: dict[int, torch.Tensor] = {}
    for layer_idx, tensors in collected.items():
        try:
            stacked[layer_idx] = torch.stack(tensors)
        except Exception as exc:
            logger.warning("router stack failed layer %d: %s", layer_idx, exc)
    return stacked or None


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def probe_node(state: AbliterationState) -> dict:
    """Harvest per-layer activations for the harmful and harmless prompt sets.

    Returns a partial state dict with ``harm_acts``, ``harmless_acts``, and
    ``router_logits`` (None unless the model is MoE and exposes router
    logits).
    """
    model = state.get("model_obj")
    tok = state.get("tokenizer")
    arch = state.get("architecture")

    if model is None or tok is None:
        raise RuntimeError("PROBE requires a loaded model and tokenizer")

    layers = _find_layers(model, arch)
    if layers is None:
        raise RuntimeError(
            f"Could not locate decoder layers (arch={arch!r}); "
            "model has no child module with `.layers`"
        )
    num_layers = len(layers)
    device = _infer_device(model)

    harmful = state.get("harmful_prompts") or list(DEFAULT_HARMFUL)
    harmless = state.get("harmless_prompts") or list(DEFAULT_HARMLESS)

    logger.info(
        "PROBE: arch=%s layers=%d device=%s harm=%d harmless=%d",
        arch,
        num_layers,
        device,
        len(harmful),
        len(harmless),
    )

    harm_acts: dict[int, list[torch.Tensor]] = defaultdict(list)
    harmless_acts: dict[int, list[torch.Tensor]] = defaultdict(list)

    def run_set(prompts: list[str], store: dict[int, list[torch.Tensor]]) -> None:
        handles = [
            layers[i].register_forward_hook(_make_hook(i, store))
            for i in range(num_layers)
        ]
        try:
            for p in prompts:
                try:
                    inp = tok(p, return_tensors="pt", truncation=True, max_length=128)
                except Exception as exc:
                    logger.warning("tokenize failed (%s): %s", p[:40], exc)
                    continue
                inp = _to_device(inp, device)
                with torch.no_grad():
                    model(**inp)
        finally:
            for h in handles:
                try:
                    h.remove()
                except Exception:  # pragma: no cover - defensive
                    pass

    run_set(harmful, harm_acts)
    run_set(harmless, harmless_acts)

    router_logits: dict[int, torch.Tensor] | None = None
    if arch == "moe":
        router_logits = _collect_router_logits(
            model, tok, harmful, device, layers, num_layers
        )

    logger.info(
        "PROBE done: harm_layers=%d harmless_layers=%d router=%s",
        len(harm_acts),
        len(harmless_acts),
        "yes" if router_logits else "no",
    )

    return {
        "harm_acts": dict(harm_acts),
        "harmless_acts": dict(harmless_acts),
        "router_logits": router_logits,
    }
