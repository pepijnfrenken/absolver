"""Experience DB — SQLite-backed record of prior abliteration attempts.

Used by the REFLEXION / SUMMON nodes to auto-select known-good
hyperparameters for a given model architecture.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional


class ExperienceDB:
    """Persistent store of prior abliteration attempts and model profiles."""

    DEFAULT_PATH = "~/.absolver/experience.db"

    def __init__(self, db_path: str = DEFAULT_PATH) -> None:
        # ":memory:" should not be expanduser'd or mkdir'd.
        if db_path == ":memory:":
            self.db_path = ":memory:"
        else:
            self.db_path = str(Path(db_path).expanduser())
            parent = Path(self.db_path).expanduser().parent
            parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        try:
            self._conn = self._open(self.db_path)
            self._init_schema()
        except sqlite3.DatabaseError:
            # Corrupt DB detected on first real SQL — back up and recreate.
            self._recover()

    # ------------------------------------------------------------------ #
    # Connection / schema
    # ------------------------------------------------------------------ #
    @staticmethod
    def _open(db_path: str) -> sqlite3.Connection:
        try:
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:
                # WAL may be unavailable on some backends (e.g. :memory:).
                pass
            return conn
        except sqlite3.DatabaseError:
            # Corrupt on open — back up and recreate.
            ExperienceDB._backup_corrupt(db_path)
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:
                pass
            return conn

    @staticmethod
    def _backup_corrupt(db_path: str) -> None:
        if db_path == ":memory:" or not os.path.exists(db_path):
            return
        ts = int(time.time())
        backup = f"{db_path}.corrupt.{ts}"
        try:
            os.replace(db_path, backup)
        except OSError:
            # Best-effort; if rename fails, just remove so we can recreate.
            try:
                os.remove(db_path)
            except OSError:
                pass

    def _init_schema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY,
                    model_id TEXT,
                    architecture TEXT,
                    hidden_size INT,
                    num_layers INT,
                    num_experts INT,
                    method_used TEXT,
                    dir_method TEXT,
                    alpha REAL,
                    passes INT,
                    target_layers TEXT,
                    target_weights TEXT,
                    max_separation REAL,
                    refusal_rate REAL,
                    quality_mean REAL,
                    final_verdict TEXT,
                    reflexion_attempts INT,
                    reflexion_history TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS model_profiles (
                    model_id TEXT PRIMARY KEY,
                    last_method TEXT,
                    last_dir_method TEXT,
                    last_alpha REAL,
                    last_passes INT,
                    last_target_layers TEXT,
                    last_verdict TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self._conn.commit()

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    def record_attempt(
        self,
        model_id: str,
        architecture: str,
        hidden_size: int,
        num_layers: int,
        method_used: str,
        dir_method: str,
        alpha: float,
        passes: int,
        target_layers: list,
        target_weights: list,
        max_separation: float,
        final_verdict: str,
        reflexion_attempts: int = 0,
        reflexion_history: Optional[list] = None,
        refusal_rate: Optional[float] = None,
        quality_mean: Optional[float] = None,
        num_experts: Optional[int] = None,
    ) -> int:
        """Insert an attempt row and upsert the model profile.

        Returns the new attempt row id.
        """
        target_layers_json = json.dumps(list(target_layers) if target_layers else [])
        target_weights_json = json.dumps(list(target_weights) if target_weights else [])
        reflexion_history_json = json.dumps(
            list(reflexion_history) if reflexion_history else []
        )

        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute(
                    """
                    INSERT INTO attempts (
                        model_id, architecture, hidden_size, num_layers, num_experts,
                        method_used, dir_method, alpha, passes,
                        target_layers, target_weights,
                        max_separation, refusal_rate, quality_mean,
                        final_verdict, reflexion_attempts, reflexion_history
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        model_id,
                        architecture,
                        hidden_size,
                        num_layers,
                        num_experts,
                        method_used,
                        dir_method,
                        alpha,
                        passes,
                        target_layers_json,
                        target_weights_json,
                        max_separation,
                        refusal_rate,
                        quality_mean,
                        final_verdict,
                        reflexion_attempts,
                        reflexion_history_json,
                    ),
                )
                attempt_id = cur.lastrowid

                cur.execute(
                    """
                    INSERT OR REPLACE INTO model_profiles (
                        model_id, last_method, last_dir_method, last_alpha,
                        last_passes, last_target_layers, last_verdict, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (
                        model_id,
                        method_used,
                        dir_method,
                        alpha,
                        passes,
                        target_layers_json,
                        final_verdict,
                    ),
                )
                self._conn.commit()
                return attempt_id or 0
            except sqlite3.DatabaseError:
                self._recover()
                # Retry once after recovery.
                cur = self._conn.cursor()
                cur.execute(
                    """
                    INSERT INTO attempts (
                        model_id, architecture, hidden_size, num_layers, num_experts,
                        method_used, dir_method, alpha, passes,
                        target_layers, target_weights,
                        max_separation, refusal_rate, quality_mean,
                        final_verdict, reflexion_attempts, reflexion_history
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        model_id,
                        architecture,
                        hidden_size,
                        num_layers,
                        num_experts,
                        method_used,
                        dir_method,
                        alpha,
                        passes,
                        target_layers_json,
                        target_weights_json,
                        max_separation,
                        refusal_rate,
                        quality_mean,
                        final_verdict,
                        reflexion_attempts,
                        reflexion_history_json,
                    ),
                )
                attempt_id = cur.lastrowid
                cur.execute(
                    """
                    INSERT OR REPLACE INTO model_profiles (
                        model_id, last_method, last_dir_method, last_alpha,
                        last_passes, last_target_layers, last_verdict, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (
                        model_id,
                        method_used,
                        dir_method,
                        alpha,
                        passes,
                        target_layers_json,
                        final_verdict,
                    ),
                )
                self._conn.commit()
                return attempt_id or 0

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def query_best_method(self, model_id: str) -> Optional[dict]:
        """Return known-good params for an exact model_id, or None.

        Only returns a result when the last recorded verdict was 'success'.
        """
        with self._lock:
            try:
                cur = self._conn.cursor()
                cur.execute(
                    """
                    SELECT last_method, last_dir_method, last_alpha,
                           last_passes, last_target_layers, last_verdict
                    FROM model_profiles
                    WHERE model_id = ?
                    """,
                    (model_id,),
                )
                row = cur.fetchone()
            except sqlite3.DatabaseError:
                self._recover()
                cur = self._conn.cursor()
                cur.execute(
                    """
                    SELECT last_method, last_dir_method, last_alpha,
                           last_passes, last_target_layers, last_verdict
                    FROM model_profiles
                    WHERE model_id = ?
                    """,
                    (model_id,),
                )
                row = cur.fetchone()

        if row is None:
            return None
        if row["last_verdict"] != "success":
            return None

        target_layers = []
        try:
            target_layers = json.loads(row["last_target_layers"] or "[]")
        except (ValueError, TypeError):
            target_layers = []

        return {
            "method": row["last_method"],
            "dir_method": row["last_dir_method"],
            "alpha": row["last_alpha"],
            "passes": row["last_passes"],
            "target_layers": target_layers,
        }

    def query_similar_arch(
        self, architecture: str, hidden_size: int
    ) -> Optional[dict]:
        """Find the best-quality prior attempt for a similar architecture.

        "Similar" = same architecture name and hidden_size within +/-100.
        """
        lo = hidden_size - 100
        hi = hidden_size + 100
        with self._lock:
            try:
                cur = self._conn.cursor()
                cur.execute(
                    """
                    SELECT model_id, architecture, hidden_size, num_layers,
                           num_experts, method_used, dir_method, alpha, passes,
                           target_layers, target_weights, max_separation,
                           refusal_rate, quality_mean, final_verdict,
                           reflexion_attempts, reflexion_history
                    FROM attempts
                    WHERE architecture = ?
                      AND hidden_size BETWEEN ? AND ?
                    ORDER BY quality_mean DESC
                    LIMIT 1
                    """,
                    (architecture, lo, hi),
                )
                row = cur.fetchone()
            except sqlite3.DatabaseError:
                self._recover()
                cur = self._conn.cursor()
                cur.execute(
                    """
                    SELECT model_id, architecture, hidden_size, num_layers,
                           num_experts, method_used, dir_method, alpha, passes,
                           target_layers, target_weights, max_separation,
                           refusal_rate, quality_mean, final_verdict,
                           reflexion_attempts, reflexion_history
                    FROM attempts
                    WHERE architecture = ?
                      AND hidden_size BETWEEN ? AND ?
                    ORDER BY quality_mean DESC
                    LIMIT 1
                    """,
                    (architecture, lo, hi),
                )
                row = cur.fetchone()

        if row is None:
            return None
        return self._row_to_dict(row)

    def get_summary_stats(self) -> dict:
        """Return counts grouped by architecture, method, and verdict."""
        with self._lock:
            try:
                cur = self._conn.cursor()
                cur.execute(
                    """
                    SELECT architecture, method_used, final_verdict, COUNT(*) AS n
                    FROM attempts
                    GROUP BY architecture, method_used, final_verdict
                    """
                )
                rows = cur.fetchall()
            except sqlite3.DatabaseError:
                self._recover()
                cur = self._conn.cursor()
                cur.execute(
                    """
                    SELECT architecture, method_used, final_verdict, COUNT(*) AS n
                    FROM attempts
                    GROUP BY architecture, method_used, final_verdict
                    """
                )
                rows = cur.fetchall()

        by_arch: dict = {}
        by_method: dict = {}
        by_verdict: dict = {}
        for r in rows:
            arch = r["architecture"]
            method = r["method_used"]
            verdict = r["final_verdict"]
            n = r["n"]
            by_arch[arch] = by_arch.get(arch, 0) + n
            by_method[method] = by_method.get(method, 0) + n
            by_verdict[verdict] = by_verdict.get(verdict, 0) + n

        return {
            "by_architecture": by_arch,
            "by_method": by_method,
            "by_verdict": by_verdict,
            "total": sum(r["n"] for r in rows),
        }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        def _loads(value: Any, default):
            if value is None:
                return default
            try:
                return json.loads(value)
            except (ValueError, TypeError):
                return default

        return {
            "model_id": row["model_id"],
            "architecture": row["architecture"],
            "hidden_size": row["hidden_size"],
            "num_layers": row["num_layers"],
            "num_experts": row["num_experts"],
            "method": row["method_used"],
            "dir_method": row["dir_method"],
            "alpha": row["alpha"],
            "passes": row["passes"],
            "target_layers": _loads(row["target_layers"], []),
            "target_weights": _loads(row["target_weights"], []),
            "max_separation": row["max_separation"],
            "refusal_rate": row["refusal_rate"],
            "quality_mean": row["quality_mean"],
            "final_verdict": row["final_verdict"],
            "reflexion_attempts": row["reflexion_attempts"],
            "reflexion_history": _loads(row["reflexion_history"], []),
        }

    def _recover(self) -> None:
        """Close, back up the corrupt file, and reopen + reinit schema."""
        try:
            self._conn.close()
        except sqlite3.Error:
            pass
        if self.db_path != ":memory:":
            self._backup_corrupt(self.db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:
            pass
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY,
                model_id TEXT,
                architecture TEXT,
                hidden_size INT,
                num_layers INT,
                num_experts INT,
                method_used TEXT,
                dir_method TEXT,
                alpha REAL,
                passes INT,
                target_layers TEXT,
                target_weights TEXT,
                max_separation REAL,
                refusal_rate REAL,
                quality_mean REAL,
                final_verdict TEXT,
                reflexion_attempts INT,
                reflexion_history TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS model_profiles (
                model_id TEXT PRIMARY KEY,
                last_method TEXT,
                last_dir_method TEXT,
                last_alpha REAL,
                last_passes INT,
                last_target_layers TEXT,
                last_verdict TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass

    def __enter__(self) -> "ExperienceDB":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
