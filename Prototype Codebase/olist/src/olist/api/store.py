"""Prediction logging and calibration monitoring.

Every score is recorded so that predictions can later be compared against what
actually happened. That comparison is the point: this system multiplies a
probability by an order value to produce a monetary recommendation, so a model
whose *ranking* still works but whose *probabilities* have drifted will keep
producing confident figures that are quietly wrong.

Accuracy monitoring would not catch that. Calibration monitoring does.

SQLite is used because it needs no infrastructure and the write volume here is
trivial. The interface is narrow enough that swapping it for Postgres is a
change to this file alone.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id       TEXT,
    scored_at      TEXT    NOT NULL,
    probability    REAL    NOT NULL,
    risk_level     TEXT    NOT NULL,
    threshold      REAL    NOT NULL,
    flagged        INTEGER NOT NULL,
    confidence     REAL,
    confidence_level TEXT,
    model_version  TEXT    NOT NULL,
    scenario       TEXT,
    top_drivers    TEXT,
    recommendation TEXT,
    -- Filled in later, once the delivery outcome is known. Null until then,
    -- which is what makes the calibration report honest: it can only score
    -- predictions whose outcome has actually been observed.
    observed_late  INTEGER,
    observed_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_pred_order   ON predictions(order_id);
CREATE INDEX IF NOT EXISTS idx_pred_scored  ON predictions(scored_at);
CREATE INDEX IF NOT EXISTS idx_pred_version ON predictions(model_version);
CREATE INDEX IF NOT EXISTS idx_pred_outcome ON predictions(observed_late);
"""


class PredictionStore:
    """Append-only log of scored orders and their eventual outcomes."""

    def __init__(self, path: Path | str = "data/predictions.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- writing ----------------------------------------------------------

    def log(self, *, order_id: str | None, probability: float, risk_level: str,
            threshold: float, confidence: float | None, confidence_level: str | None,
            model_version: str, scenario: str | None = None,
            top_drivers: list[str] | None = None,
            recommendation: str | None = None) -> int:
        """Record one prediction. Returns its row id."""
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO predictions
                   (order_id, scored_at, probability, risk_level, threshold, flagged,
                    confidence, confidence_level, model_version, scenario,
                    top_drivers, recommendation)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (order_id, datetime.now(timezone.utc).isoformat(), float(probability),
                 risk_level, float(threshold), int(probability >= threshold),
                 confidence, confidence_level, model_version, scenario,
                 json.dumps(top_drivers or []), recommendation),
            )
            return int(cur.lastrowid)

    def record_outcome(self, order_id: str, was_late: bool) -> int:
        """Attach the observed outcome to every prediction for this order.

        Called once delivery is resolved. Until this is called the prediction
        cannot contribute to a calibration report, which is deliberate.
        """
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE predictions SET observed_late = ?, observed_at = ? "
                "WHERE order_id = ? AND observed_late IS NULL",
                (int(was_late), datetime.now(timezone.utc).isoformat(), order_id),
            )
            return cur.rowcount

    # -- reading ----------------------------------------------------------

    def recent(self, limit: int = 100, model_version: str | None = None) -> list[dict]:
        query = "SELECT * FROM predictions"
        params: list = []
        if model_version:
            query += " WHERE model_version = ?"
            params.append(model_version)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(query, params)]

    def resolved(self, model_version: str | None = None) -> list[dict]:
        """Predictions whose outcome is known — the only ones that can be scored."""
        query = "SELECT probability, observed_late FROM predictions WHERE observed_late IS NOT NULL"
        params: list = []
        if model_version:
            query += " AND model_version = ?"
            params.append(model_version)
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(query, params)]

    def counts(self) -> dict[str, int]:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            resolved = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE observed_late IS NOT NULL"
            ).fetchone()[0]
        return {"total": total, "resolved": resolved, "pending": total - resolved}


# ==========================================================================
# Calibration
# ==========================================================================


def calibration_report(probabilities: np.ndarray, outcomes: np.ndarray,
                       n_buckets: int = 10, min_per_bucket: int = 20,
                       tolerance: float = 0.10) -> dict:
    """Compare predicted risk against observed outcomes, in buckets.

    A calibrated model tracks the diagonal: orders predicted at 30% should be
    late about 30% of the time. ``max_gap`` is the largest divergence across
    buckets with enough observations to be meaningful.

    ``tolerance`` is the gap at which drift should be acted on. Ten percentage
    points is a starting point, not a derived value; it should be tightened
    once enough resolved predictions exist to establish normal variation.
    """
    probabilities = np.asarray(probabilities, dtype="float64")
    outcomes = np.asarray(outcomes, dtype="float64")

    if len(probabilities) == 0:
        return {"n_orders": 0, "buckets": [], "max_gap": 0.0,
                "mean_absolute_gap": 0.0, "brier_score": None,
                "drifted": False, "tolerance": tolerance}

    edges = np.linspace(0, 1, n_buckets + 1)
    index = np.clip(np.digitize(probabilities, edges[1:-1]), 0, n_buckets - 1)

    buckets, gaps = [], []
    for b in range(n_buckets):
        mask = index == b
        n = int(mask.sum())
        if n < min_per_bucket:
            continue
        predicted = float(probabilities[mask].mean())
        observed = float(outcomes[mask].mean())
        gap = abs(predicted - observed)
        gaps.append(gap)
        buckets.append({
            "band": f"{edges[b]:.0%}-{edges[b + 1]:.0%}",
            "n": n,
            "mean_predicted": predicted,
            "observed_rate": observed,
            "gap": gap,
        })

    max_gap = max(gaps) if gaps else 0.0
    return {
        "n_orders": len(probabilities),
        "buckets": buckets,
        "max_gap": max_gap,
        "mean_absolute_gap": float(np.mean(gaps)) if gaps else 0.0,
        "brier_score": float(np.mean((probabilities - outcomes) ** 2)),
        "drifted": max_gap > tolerance,
        "tolerance": tolerance,
    }
