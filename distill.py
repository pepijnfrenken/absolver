"""DISTILL node: per-layer refusal-direction extraction.

Implements four direction-extraction methods (``diff_means``, ``svd``,
``leace``, ``whitened_svd``), ranks layers by separation score, and selects
the target layers to project during EXCISE.
"""
from __future__ import annotations

import logging
from typing import Any

import torch

from config import ModelConfig
from state import AbliterationState

logger = logging.getLogger(__name__)


def _safe_svd(x: torch.Tensor, full_matrices: bool = False):
    """``torch.linalg.svd`` with a CPU fallback for finicky backends (e.g. MPS).

    On fallback the decomposed factors are moved back to the input tensor's
    device so callers never receive leaked CPU tensors.
    """
    try:
        return torch.linalg.svd(x, full_matrices=full_matrices)
    except Exception as exc:  # pragma: no cover - backend-specific
        logger.warning("SVD failed on device %s (%s); retrying on CPU.", x.device, exc)
        U, S, Vt = torch.linalg.svd(x.cpu(), full_matrices=full_matrices)
        return U.to(x.device), S.to(x.device), Vt.to(x.device)


def _diff_means(
    harm_stack: torch.Tensor,
    harmless_mean: torch.Tensor,
    hidden: int,
    n_dirs: int,
    device: str,
):
    """d = mean(harm) - mean(harmless); normalize; score = d.norm()."""
    del n_dirs, hidden
    h = harm_stack.mean(dim=0).to(device)
    b = harmless_mean.to(device)
    d = h - b
    norm = d.norm() + 1e-8
    direction = d / norm
    score = norm.item()
    return direction, score


def _paired(
    harm_stack: torch.Tensor,
    harmless_mean: torch.Tensor,
    hidden: int,
    n_dirs: int,
    device: str,
):
    """Paired output-phase direction (same prompts, refusal vs affirmative).

    Identical math to diff_means (mean refusal activation minus mean
    affirmative-prefill activation) but the *data* comes from the paired
    probe mode: the same harmful prompts in both groups, with the compliant
    group generated from an affirmative-prefilled continuation. This kills
    the topic/difficulty confound that diff_means suffers when the two
    groups use different prompts (see LFM2.5 research notes).

    Score is the L2 norm of the paired mean difference — higher = the
    refusal/compliance contrast is more separated in that layer.
    """
    del n_dirs, hidden
    r = harm_stack.mean(dim=0).to(device)
    a = harmless_mean.to(device)
    d = r - a
    norm = d.norm() + 1e-8
    direction = d / norm
    score = norm.item()
    return direction, score


def _svd(
    harm_stack: torch.Tensor,
    harmless_mean: torch.Tensor,
    hidden: int,
    n_dirs: int,
    device: str,
):
    """Stack per-prompt diffs (harm - harmless_mean); SVD; Vt[:n_dirs]; score=S[0].

    ``Vt`` rows are feature-space directions (right singular vectors), which is
    the correct space for refusal directions. ``U`` columns live in sample
    space and must not be used as the direction here.
    """
    del hidden
    # PROBE should yield per-prompt 1D activations (last token), so
    # harm_stack is [n_samples, hidden]. Defensive: if a hook captured extra
    # leading axes, flatten to 2D — a batched SVD over the last two dims
    # would otherwise produce Vt[:n_dirs] with a spurious extra axis.
    if harm_stack.dim() != 2:
        logger.warning(
            "_svd: harm_stack has %d dims (shape=%s); flattening to 2D. "
            "Check probe.py hook capture shape.",
            harm_stack.dim(),
            tuple(harm_stack.shape),
        )
        harm_stack = harm_stack.reshape(-1, harm_stack.shape[-1])
        harmless_mean = harmless_mean.reshape(-1)
    diffs = harm_stack.to(device) - harmless_mean.to(device).unsqueeze(0)
    diffs = diffs.squeeze()
    _U, S, Vt = _safe_svd(diffs, full_matrices=False)
    direction = Vt[:n_dirs]  # [n_dirs, hidden] in feature space
    score = S[0].item()
    return direction, score


def _leace(
    harm_stack: torch.Tensor,
    harmless_stack: torch.Tensor,
    hidden: int,
    n_dirs: int,
    device: str,
):
    """LEACE: covariance of stacked harm+harmless, lstsq for beta, normalize; score=d@Sigma@d.

    LEACE against a single binary concept yields exactly one direction, so
    ``n_dirs > 1`` is clamped to 1 with a warning rather than fabricating
    orthogonal directions with no concept alignment.
    """
    if n_dirs > 1:
        logger.warning(
            "LEACE produces a single concept direction; clamping n_dirs=%d to 1.", n_dirs
        )
    n_harm = harm_stack.shape[0]
    n_harmless = harmless_stack.shape[0]
    X = torch.cat([harm_stack, harmless_stack], dim=0).to(device)
    y = torch.cat(
        [torch.ones(n_harm, device=device), torch.zeros(n_harmless, device=device)]
    ).to(device)

    X_centered = X - X.mean(dim=0, keepdim=True)
    denom = max(X.shape[0] - 1, 1)
    Sigma = (X_centered.T @ X_centered) / denom
    Sigma_reg = Sigma + 1e-6 * torch.eye(hidden, device=device)

    beta = torch.linalg.lstsq(Sigma_reg, X_centered.T @ y).solution
    norm = beta.norm() + 1e-8
    d = beta / norm
    score = (d @ Sigma_reg @ d).item()
    return d, score


def _whitened_svd(
    harm_stack: torch.Tensor,
    harmless_stack: torch.Tensor,
    hidden: int,
    n_dirs: int,
    device: str,
):
    """Cholesky whitening then SVD on whitened harm acts; Vt[:n_dirs]; score=S[0].

    Whitening applies ``L^{-1}`` (a triangular solve against the Cholesky
    factor), not ``(L L^T)^{-1}``, so rows end up with identity covariance in
    the L^{-1} metric without being over-whitened by the full Sigma^{-1}.
    """
    X_combined = torch.cat([harm_stack, harmless_stack], dim=0).to(device)
    # Population covariance via torch.cov expects [features, samples].
    Sigma = torch.cov(X_combined.T)
    L = torch.linalg.cholesky(Sigma + 1e-6 * torch.eye(hidden, device=device))
    # Whitened harm acts: solve L @ W = harm^T  ->  W = L^{-1} @ harm^T, then transpose.
    whitened = torch.linalg.solve_triangular(
        L, harm_stack.to(device).T, upper=False
    ).T
    _U, S, Vt = _safe_svd(whitened, full_matrices=False)
    direction = Vt[:n_dirs]  # [n_dirs, hidden]
    score = S[0].item()
    return direction, score


def extract_directions(
    harm_acts: dict[int, Any],
    harmless_acts: dict[int, Any],
    num_layers: int,
    hidden: int,
    dir_method: str,
    n_dirs: int,
    device: str,
    return_good_dirs: bool = False,
) -> tuple[dict[int, torch.Tensor], dict[int, float]] | tuple[dict[int, torch.Tensor], dict[int, float], dict[int, torch.Tensor]]:
    """Compute per-layer refusal directions using the given dir_method.

    Returns (directions, separation_scores). Shared by DISTILL and the
    SWEEP node (which recomputes directions per candidate dir_method).

    When ``return_good_dirs=True``, the harmless mean (normalized) is also
    returned per layer — used by projected-abliteration (orthogonalize the
    refusal direction against the harmless direction so capabilities that
    overlap refusal are preserved, per Lai/grimjim + Heretic).
    """
    directions: dict[int, torch.Tensor] = {}
    scores: dict[int, float] = {}
    good_dirs: dict[int, torch.Tensor] = {}

    layer_indices: list[int] = (
        list(range(num_layers)) if num_layers else sorted(set(harm_acts) | set(harmless_acts))
    )

    for i in layer_indices:
        h_list = harm_acts.get(i)
        b_list = harmless_acts.get(i)
        if not h_list or not b_list:
            continue

        harm_stack = torch.stack(h_list).to(torch.float32)
        harmless_stack = torch.stack(b_list).to(torch.float32)
        harmless_mean = harmless_stack.mean(dim=0)

        try:
            if dir_method == "diff_means":
                d, s = _diff_means(harm_stack, harmless_mean, hidden, n_dirs, device)
            elif dir_method == "paired":
                d, s = _paired(harm_stack, harmless_mean, hidden, n_dirs, device)
            elif dir_method == "svd":
                d, s = _svd(harm_stack, harmless_mean, hidden, n_dirs, device)
            elif dir_method == "leace":
                d, s = _leace(harm_stack, harmless_stack, hidden, n_dirs, device)
            elif dir_method == "whitened_svd":
                d, s = _whitened_svd(harm_stack, harmless_stack, hidden, n_dirs, device)
            else:
                logger.warning("Unknown dir_method '%s'; falling back to diff_means.", dir_method)
                d, s = _diff_means(harm_stack, harmless_mean, hidden, n_dirs, device)
        except Exception as exc:
            logger.warning("dir_method '%s' failed on layer %d (%s); skipping.", dir_method, i, exc)
            continue

        directions[i] = d
        scores[i] = float(s)
        if return_good_dirs:
            norm = harmless_mean.norm().clamp(min=1e-8)
            good_dirs[i] = (harmless_mean / norm).to(device)

    if return_good_dirs:
        return directions, scores, good_dirs
    return directions, scores


def distill_node(state: AbliterationState) -> dict:
    """Extract per-layer refusal directions and pick target layers.

    Returns a partial state dict with ``refusal_directions``,
    ``separation_scores``, and ``target_layers``.
    """
    cfg: ModelConfig = state["config"]
    harm_acts: dict[int, Any] = state.get("harm_acts") or {}
    harmless_acts: dict[int, Any] = state.get("harmless_acts") or {}
    num_layers: int = state.get("num_layers", 0)
    hidden: int = state.get("hidden_size", 0)
    # Prefer the device declared on the config; fall back to CUDA detection.
    raw_device = getattr(cfg, "device", "auto")
    device = raw_device if raw_device not in (None, "auto") else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dir_method: str = cfg.dir_method
    n_dirs = max(1, min(cfg.n_directions, hidden)) if hidden else cfg.n_directions

    directions, scores = extract_directions(
        harm_acts, harmless_acts, num_layers, hidden, dir_method, n_dirs, device
    )

    # Rank layers by separation score, descending.
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    if cfg.target_layers:
        target_layers: list[int] = list(cfg.target_layers)
    else:
        # Auto: top-10 layers with score > separation_threshold (default 5).
        threshold = float(cfg.separation_threshold)
        top_n = min(10, len(ranked))
        target_layers = [layer for layer, score in ranked[:top_n] if score > threshold]
        if not target_layers:
            # Fallback: top-5 by score regardless of threshold.
            target_layers = [layer for layer, _score in ranked[:5]]
        if not target_layers:
            logger.warning(
                "All layers have zero separation score; selecting first available layers."
            )

    logger.info(
        "DISTILL: method=%s n_directions=%d layers_scored=%d target_layers=%s",
        dir_method,
        n_dirs,
        len(scores),
        target_layers,
    )

    return {
        "refusal_directions": directions,
        "separation_scores": scores,
        "target_layers": target_layers,
    }
