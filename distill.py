"""DISTILL node: per-layer refusal-direction extraction.

Implements four direction-extraction methods (``diff_means``, ``svd``,
``leace``, ``whitened_svd``), ranks layers by separation score, and selects
the target layers to project during EXCISE.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import torch

from config import ModelConfig
from state import AbliterationState

logger = logging.getLogger(__name__)


def _safe_svd(x: torch.Tensor, full_matrices: bool = False):
    """``torch.linalg.svd`` with a CPU fallback for finicky backends (e.g. MPS)."""
    try:
        return torch.linalg.svd(x, full_matrices=full_matrices)
    except Exception as exc:  # pragma: no cover - backend-specific
        logger.warning("SVD failed on device %s (%s); retrying on CPU.", x.device, exc)
        return torch.linalg.svd(x.cpu(), full_matrices=full_matrices)


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


def _svd(
    harm_stack: torch.Tensor,
    harmless_mean: torch.Tensor,
    hidden: int,
    n_dirs: int,
    device: str,
):
    """Stack per-prompt diffs (harm - harmless_mean); SVD; U[:,:n_dirs].T; score=S[0]."""
    diffs = (harm_stack - harmless_mean.unsqueeze(0)).to(device)
    U, S, _Vt = _safe_svd(diffs, full_matrices=False)
    direction = U[:, :n_dirs].T  # [n_dirs, *] per spec
    score = S[0].item()
    return direction, score


def _leace(
    harm_stack: torch.Tensor,
    harmless_stack: torch.Tensor,
    hidden: int,
    n_dirs: int,
    device: str,
):
    """LEACE: covariance of stacked harm+harmless, lstsq for beta, normalize; score=d@Sigma@d."""
    del n_dirs
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
    """Cholesky whitening then SVD on whitened harm acts; Vt[:n_dirs]; score=S[0]."""
    X_combined = torch.cat([harm_stack, harmless_stack], dim=0).to(device)
    # Population covariance via torch.cov expects [features, samples].
    Sigma = torch.cov(X_combined.T)
    L = torch.linalg.cholesky(Sigma + 1e-6 * torch.eye(hidden, device=device))
    # Whitened harm acts: solve L @ W = harm^T  ->  W = L^{-1} @ harm^T, then transpose.
    whitened = torch.cholesky_solve(harm_stack.to(device).T, L).T
    _U, S, Vt = _safe_svd(whitened, full_matrices=False)
    direction = Vt[:n_dirs]  # [n_dirs, hidden]
    score = S[0].item()
    return direction, score


def distill_node(state: AbliterationState) -> dict:
    """Extract per-layer refusal directions and pick target layers.

    Returns a partial state dict with ``refusal_directions``,
    ``separation_scores``, and ``target_layers``.
    """
    cfg: ModelConfig = state["config"]
    harm_acts: Dict[int, Any] = state.get("harm_acts") or {}
    harmless_acts: Dict[int, Any] = state.get("harmless_acts") or {}
    num_layers: int = state.get("num_layers", 0)
    hidden: int = state.get("hidden_size", 0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dir_method: str = cfg.dir_method
    n_dirs = max(1, min(cfg.n_directions, hidden)) if hidden else cfg.n_directions

    directions: Dict[int, torch.Tensor] = {}
    scores: Dict[int, float] = {}

    layer_indices: List[int] = (
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

    # Rank layers by separation score, descending.
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    if cfg.target_layers:
        target_layers: List[int] = list(cfg.target_layers)
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
