"""Tests for experience.py (ExperienceDB SQLite).

Uses a tmp_path SQLite file for each test so nothing leaks across runs.
Covers the three query paths: exact match, similar-arch match, no match,
plus summary stats and corruption recovery.
"""
from __future__ import annotations

import os
import sqlite3

import pytest

from experience import ExperienceDB


# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #
@pytest.fixture
def db(tmp_path):
    path = tmp_path / "exp.db"
    return ExperienceDB(str(path))


def _record(
    db,
    model_id="m/a",
    architecture="dense",
    hidden_size=4096,
    num_layers=32,
    num_experts=None,
    method_used="advanced",
    dir_method="diff_means",
    alpha=0.5,
    passes=3,
    target_layers=None,
    target_weights=None,
    max_separation=12.0,
    refusal_rate=0.1,
    quality_mean=0.8,
    final_verdict="success",
    reflexion_attempts=0,
    reflexion_history=None,
):
    db.record_attempt(
        model_id=model_id,
        architecture=architecture,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_experts=num_experts,
        method_used=method_used,
        dir_method=dir_method,
        alpha=alpha,
        passes=passes,
        target_layers=target_layers or [10, 11, 12],
        target_weights=target_weights or ["o_proj", "down_proj"],
        max_separation=max_separation,
        refusal_rate=refusal_rate,
        quality_mean=quality_mean,
        final_verdict=final_verdict,
        reflexion_attempts=reflexion_attempts,
        reflexion_history=reflexion_history or [],
    )


# ---------------------------------------------------------------------- #
# Schema + basic insert
# ---------------------------------------------------------------------- #
class TestSchemaAndInsert:
    def test_in_memory_db_initializes(self):
        db = ExperienceDB(":memory:")
        assert db.query_best_method("anything") is None

    def test_record_attempt_creates_row(self, db):
        _record(db, model_id="m/a")
        stats = db.get_summary_stats()
        assert stats["total"] == 1
        assert stats["by_architecture"].get("dense") == 1

    def test_summary_stats_grouping(self, db):
        _record(db, model_id="m/a", architecture="dense", method_used="advanced", final_verdict="success")
        _record(db, model_id="m/b", architecture="moe", method_used="bias_vectors", final_verdict="failed")
        _record(db, model_id="m/c", architecture="moe", method_used="advanced", final_verdict="success")
        stats = db.get_summary_stats()
        assert stats["total"] == 3
        assert stats["by_architecture"]["dense"] == 1
        assert stats["by_architecture"]["moe"] == 2
        assert stats["by_method"]["advanced"] == 2
        assert stats["by_method"]["bias_vectors"] == 1
        assert stats["by_verdict"]["success"] == 2
        assert stats["by_verdict"]["failed"] == 1


# ---------------------------------------------------------------------- #
# Query paths
# ---------------------------------------------------------------------- #
class TestQueryBestMethod:
    def test_exact_match_success_returns_params(self, db):
        _record(
            db,
            model_id="m/known",
            method_used="advanced",
            dir_method="svd",
            alpha=0.42,
            passes=2,
            target_layers=[5, 6, 7],
            final_verdict="success",
        )
        best = db.query_best_method("m/known")
        assert best is not None
        assert best["method"] == "advanced"
        assert best["dir_method"] == "svd"
        assert best["alpha"] == pytest.approx(0.42)
        assert best["passes"] == 2
        assert best["target_layers"] == [5, 6, 7]

    def test_exact_match_failure_returns_none(self, db):
        _record(db, model_id="m/failed", final_verdict="failed")
        assert db.query_best_method("m/failed") is None

    def test_no_match_returns_none(self, db):
        assert db.query_best_method("never/seen") is None


class TestQuerySimilarArch:
    def test_similar_arch_returns_best_quality(self, db):
        _record(
            db,
            model_id="m/a",
            architecture="dense",
            hidden_size=4096,
            quality_mean=0.6,
            final_verdict="success",
            method_used="advanced",
            dir_method="diff_means",
        )
        _record(
            db,
            model_id="m/b",
            architecture="dense",
            hidden_size=4100,
            quality_mean=0.9,  # higher quality -> should win
            final_verdict="success",
            method_used="svd",
            dir_method="svd",
        )
        # Query for hidden_size=4098 -> both in window, m/b should win.
        result = db.query_similar_arch("dense", 4098)
        assert result is not None
        assert result["model_id"] == "m/b"

    def test_similar_arch_out_of_window_returns_none(self, db):
        _record(db, model_id="m/a", architecture="dense", hidden_size=4096)
        # 200 away -> outside +/-100 window.
        assert db.query_similar_arch("dense", 4296) is None

    def test_similar_arch_wrong_arch_returns_none(self, db):
        _record(db, model_id="m/a", architecture="dense", hidden_size=4096)
        assert db.query_similar_arch("moe", 4096) is None


# ---------------------------------------------------------------------- #
# UPSERT behavior
# ---------------------------------------------------------------------- #
class TestUpsertProfile:
    def test_rerecord_updates_profile(self, db):
        _record(db, model_id="m/a", method_used="advanced", final_verdict="success")
        _record(db, model_id="m/a", method_used="bias_vectors", final_verdict="success")
        best = db.query_best_method("m/a")
        assert best["method"] == "bias_vectors"  # latest wins

    def test_failed_attempt_then_success_exposes_success(self, db):
        _record(db, model_id="m/a", final_verdict="failed")
        assert db.query_best_method("m/a") is None
        _record(db, model_id="m/a", final_verdict="success", method_used="advanced")
        best = db.query_best_method("m/a")
        assert best is not None
        assert best["method"] == "advanced"


# ---------------------------------------------------------------------- #
# JSON round-trip
# ---------------------------------------------------------------------- #
class TestJsonRoundTrip:
    def test_list_fields_round_trip(self, db):
        _record(
            db,
            model_id="m/json",
            target_layers=[1, 2, 3],
            target_weights=["o_proj", "down_proj", "expert.down"],
            reflexion_history=[{"attempt": 1, "strategy": "expand_prompts"}],
            reflexion_attempts=1,
        )
        best = db.query_best_method("m/json")
        assert best["target_layers"] == [1, 2, 3]


# ---------------------------------------------------------------------- #
# Corruption recovery
# ---------------------------------------------------------------------- #
class TestCorruptionRecovery:
    def test_garbage_db_gets_backed_up_and_recreated(self, tmp_path):
        path = tmp_path / "bad.db"
        path.write_bytes(b"not a sqlite database at all!!!")
        # Opening should detect corruption, back up, recreate.
        db = ExperienceDB(str(path))
        # Schema should work now.
        _record(db, model_id="m/recover", final_verdict="success")
        assert db.query_best_method("m/recover") is not None
        # A backup of the corrupt file should exist.
        backups = [f for f in os.listdir(tmp_path) if "corrupt" in f.lower()]
        assert len(backups) >= 1
