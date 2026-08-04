"""Tests for verify.py: benchmark auto-derivation logic."""
from __future__ import annotations

from verify import _derive_requested_benchmarks


def test_derive_uses_model_card_targets_when_verify_benchmarks_empty():
    """When verify_benchmarks is empty, derive from model_card_targets keys in _RUNNER."""
    targets = {"bbh": 71.9, "math": 91.6, "gpqa": 26.3}
    result = _derive_requested_benchmarks(targets, [])
    assert result == ["bbh", "math", "gpqa"], f"Expected ['bbh', 'math', 'gpqa'], got {result}"


def test_derive_uses_explicit_override_when_nonempty():
    """When verify_benchmarks is non-empty, return it verbatim."""
    result = _derive_requested_benchmarks({"bbh": 71.9}, ["arc_easy"])
    assert result == ["arc_easy"], f"Expected ['arc_easy'], got {result}"


def test_derive_ignores_unknown_benchmarks_in_model_card():
    """Unknown model-card keys lacking a runner are filtered out."""
    targets = {"nonexistent_runner": 42.0, "bbh": 71.9}
    result = _derive_requested_benchmarks(targets, [])
    # only bbh is in _RUNNER
    assert result == ["bbh"], f"Expected ['bbh'], got {result}"


def test_derive_uses_mmlu_alias():
    """'mmlu' is in _RUNNER and should be picked up from model_card_targets."""
    targets = {"mmlu": 70.1}
    result = _derive_requested_benchmarks(targets, [])
    assert result == ["mmlu"], f"Expected ['mmlu'], got {result}"