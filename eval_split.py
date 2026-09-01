"""Immutable train/tune/test prompt splits for Absolver evaluations.

Ported from OBLITERATUS's experiment protocol (elder-plinius/OBLITERATUS,
`obliteratus/experiment_protocol.py`, AGPL-3.0 — this is a reimplementation
of the *concept*, not a copy of the code, so it stays MIT-clean):

* Prompt identities are hashed (SHA-256) so the split is deterministic
  without leaking the prompt text into manifests.
* Position-stratified: the ordering of harmful prompts is divided into
  `strata` contiguous blocks, and each block contributes a balanced slice
  to train/tune/test. This prevents one topic cluster from dominating
  a partition.
* The split is FIXED by a seed string; the same dataset + seed always
  yields the same partition.

This closes the sweep/verify leakage hole: the sweep must optimize on the
TRAIN split only, and VERIFY measures on the held-out TEST split. Without
this, the sweep picks a config that memorizes the exact verify prompts
(which is why refusal "0.25" looked stable — it was the same 20 prompts
every time).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

DEFAULT_SEED = "absolver:qwen25:v1"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PromptSplit:
    """Disjoint prompt partitions plus a stable manifest."""

    train: tuple[str, ...]
    tune: tuple[str, ...]
    test: tuple[str, ...]
    manifest: dict = field(default_factory=dict)


def build_split(
    harmful: list[str],
    harmless: list[str],
    *,
    train_size: int = 40,
    tune_size: int = 20,
    test_size: int = 20,
    strata: int = 5,
    seed: str = DEFAULT_SEED,
) -> PromptSplit:
    """Build a stable, position-stratified split of harmful prompts.

    `harmless` is used only for identity checking (paired lists); the split
    is over the harmful prompts, and the same index ordering applies to the
    harmless list for paired use.

    Raises ValueError if the lists are unbalanced or the requested sizes
    don't sum to the harmful list length.
    """
    if len(harmful) != len(harmless):
        # Allow unbalanced pools: pair them up to the shorter length (the
        # harmless list is the pairing reference; harmful extras are dropped
        # from the split but remain usable elsewhere).
        n = min(len(harmful), len(harmless))
        harmful, harmless = harmful[:n], harmless[:n]
    requested = train_size + tune_size + test_size
    if requested != len(harmful):
        raise ValueError(
            f"split sizes total {requested}, but dataset contains {len(harmful)} pairs"
        )
    if strata < 1:
        raise ValueError("strata must be positive")

    # Records: stable id, stratum bucket, key (seed-scrambled order)
    records: list[dict] = []
    seen_ids: set[str] = set()
    for index, (bad, good) in enumerate(zip(harmful, harmless)):
        pair_id = _digest(f"harmful\0{bad}\0harmless\0{good}")
        if pair_id in seen_ids:
            raise ValueError(f"duplicate prompt pair identity at source index {index}")
        seen_ids.add(pair_id)
        records.append(
            {
                "id": pair_id,
                "stratum": min(strata - 1, index * strata // len(harmful)),
                "pair": (bad, good),
                "key": _digest(f"{seed}\0{pair_id}"),
            }
        )

    # Sort within each stratum by key (deterministic pseudo-random order)
    stratified: dict[int, list[dict]] = {s: [] for s in range(strata)}
    for rec in records:
        stratified[rec["stratum"]].append(rec)
    for s in range(strata):
        stratified[s].sort(key=lambda r: r["key"])

    # Fill train/tune/test round-robin across strata so every stratum
    # contributes to every partition.
    counts = {"train": train_size, "tune": tune_size, "test": test_size}
    assigns: dict[str, list[dict]] = {"train": [], "tune": [], "test": []}
    remaining = dict(counts)
    # interleave partitions per stratum: train, tune, test, train, tune, ...
    order = ["train", "tune", "test"]
    for s in range(strata):
        for rec in stratified[s]:
            for part in order:
                if remaining[part] > 0:
                    assigns[part].append(rec)
                    remaining[part] -= 1
                    break

    # If any partition is short (edge case), top up from any leftovers
    # (shouldn't happen when requested == len(harmful), but be safe).
    leftovers = [r for r in records if r not in sum(assigns.values(), [])]
    for part in order:
        while remaining[part] > 0 and leftovers:
            assigns[part].append(leftovers.pop(0))
            remaining[part] -= 1

    def _extract(part: str) -> tuple[str, ...]:
        return tuple(r["pair"][0] for r in assigns[part])

    return PromptSplit(
        train=_extract("train"),
        tune=_extract("tune"),
        test=_extract("test"),
        manifest={
            "seed": seed,
            "strata": strata,
            "sizes": counts,
            "total": len(harmful),
            "train_ids": [r["id"] for r in assigns["train"]],
            "tune_ids": [r["id"] for r in assigns["tune"]],
            "test_ids": [r["id"] for r in assigns["test"]],
        },
    )


def split_for_verify(
    harmful: list[str],
    harmless: list[str],
    *,
    n_verify: int,
    seed: str = DEFAULT_SEED,
) -> PromptSplit:
    """Convenience: build a split sized so the TEST partition is n_verify.

    train gets 2*n_verify, tune gets n_verify, test gets n_verify
    (total = 4*n_verify). This is what VERIFY consumes.
    """
    total = 4 * n_verify
    if len(harmful) < total:
        raise ValueError(
            f"need at least {total} harmful prompts for a {n_verify}-prompt "
            f"held-out test; have {len(harmful)}"
        )
    return build_split(
        harmful, harmless,
        train_size=2 * n_verify, tune_size=n_verify, test_size=n_verify,
        seed=seed,
    )
