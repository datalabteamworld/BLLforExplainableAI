"""Collect every empirical number the paper needs, in one run, with the output kept.

    python scripts/collect_paper_evidence.py

Writes two files to reports/:

    paper_evidence.md      readable -- this is the one to send back
    paper_evidence.json    machine-readable, for building the results table

Nothing here retrains or alters the existing pipeline. It calls the same
cleaning, splitting, fitting and evaluation code the project already uses, and
records what comes out. The point is that every figure in the revised paper can
be traced to a file rather than to a recollection.

Requires the real Olist CSVs in data/raw/. Takes roughly 10-25 minutes.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from olist import evaluate as ev  # noqa: E402
from olist import features, pipeline, validate  # noqa: E402
from olist.config import CONFIG, PROCESSED_DIR, RANDOM_STATE, REPORTS_DIR, TARGET, TIME_KEY  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("evidence")

OUT_JSON = REPORTS_DIR / "paper_evidence.json"
OUT_MD = REPORTS_DIR / "paper_evidence.md"

KEEP = ("roc_auc", "average_precision", "brier", "positive_rate",
        "chosen_threshold", "recall_at_chosen", "precision_at_chosen",
        "f1_at_chosen", "recall_at_0.5", "precision_at_0.5", "f1_at_0.5",
        "true_positives", "false_positives", "false_negatives",
        "flag_rate", "cost_per_order", "savings_vs_no_model",
        "oracle_threshold", "recall_at_oracle", "precision_at_oracle")


def _clean(obj):
    """Make numpy types JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    return obj


# ==========================================================================
# 1. Dataset
# ==========================================================================

def load_table(rebuild: bool) -> pd.DataFrame:
    path = PROCESSED_DIR / "orders_features.parquet"
    if rebuild or not path.exists():
        logger.info("Building the modelling table from data/raw/ ...")
        return validate.build_dataset()
    logger.info("Reading %s", path)
    return pd.read_parquet(path)


def describe(df: pd.DataFrame, raw_orders: int | None) -> dict:
    train, test = pipeline.temporal_split(df)
    return {
        "raw_order_rows": raw_orders,
        "modelling_rows": len(df),
        "modelling_columns": int(df.shape[1]),
        "late_rate": float(df[TARGET].mean()),
        "date_min": str(df[TIME_KEY].min().date()),
        "date_max": str(df[TIME_KEY].max().date()),
        "train_end": str(CONFIG.split.train_end),
        "censor_buffer_days": 30,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_late_rate": float(train[TARGET].mean()),
        "test_late_rate": float(test[TARGET].mean()),
        "train_date_min": str(train[TIME_KEY].min().date()),
        "train_date_max": str(train[TIME_KEY].max().date()),
        "test_date_min": str(test[TIME_KEY].min().date()),
        "test_date_max": str(test[TIME_KEY].max().date()),
    }


# ==========================================================================
# 2. Model evaluation, uncalibrated and calibrated
# ==========================================================================

def run_model(df: pd.DataFrame, model_name: str, scenario: str) -> dict:
    """Fit once, evaluate before and after calibration on the same test set."""
    train, test = pipeline.temporal_split(df)
    cols = pipeline.feature_columns(df, scenario)
    y_test = test[TARGET].to_numpy()

    logger.info("  fitting %s / %s on %d rows, %d features",
                model_name, scenario, len(train), len(cols))

    base = pipeline.build_pipeline(train, cols, model_name, y=train[TARGET])
    base.fit(train[cols], train[TARGET])
    proba_raw = base.predict_proba(test[cols])[:, 1]
    # Note: this baseline was fitted on the whole training period, so the
    # holdout slice is in-sample for it; its threshold is optimistic.
    raw = ev.evaluate(y_test, proba_raw, CONFIG.cost, f"{model_name} (uncalibrated)",
                      threshold=ev.holdout_threshold(base, train, cols, CONFIG.cost))

    cal = pipeline.fit_calibrated(
        pipeline.build_pipeline(train, cols, model_name, y=train[TARGET]), train, cols
    )
    proba_cal = cal.predict_proba(test[cols])[:, 1]
    calibrated = ev.evaluate(y_test, proba_cal, CONFIG.cost, f"{model_name} (calibrated)",
                             threshold=ev.holdout_threshold(cal, train, cols, CONFIG.cost))

    return {
        "model": model_name,
        "scenario": scenario,
        "n_features": len(cols),
        "feature_columns": cols,
        "uncalibrated": {k: raw["metrics"][k] for k in KEEP if k in raw["metrics"]},
        "calibrated": {k: calibrated["metrics"][k] for k in KEEP if k in calibrated["metrics"]},
        "brier_improvement": float(raw["metrics"]["brier"] - calibrated["metrics"]["brier"]),
        "_fitted": cal,
        "_train": train,
        "_cols": cols,
    }


def baselines(df: pd.DataFrame) -> dict:
    """Reference points a reviewer will ask for. Cost nothing to compute."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    _, test = pipeline.temporal_split(df)
    y = test[TARGET].to_numpy()
    prev = float(y.mean())
    const = np.full_like(y, prev, dtype=float)
    return {
        "prevalence_in_test": prev,
        "always_predict_prevalence": {
            "roc_auc": 0.5,
            "average_precision": float(average_precision_score(y, const)),
            "brier": float(brier_score_loss(y, const)),
        },
        "always_predict_late": {
            "recall": 1.0,
            "precision": prev,
            "flag_rate": 1.0,
        },
        "note": "A no-skill classifier scores AUC 0.5 and average precision "
                "equal to the positive rate. Any reported figure must be read "
                "against these.",
    }


# ==========================================================================
# 3. Review-score association
# ==========================================================================

def review_effect(df: pd.DataFrame, n_boot: int = 2000) -> dict:
    """Mean difference in review score, late vs on-time, with a bootstrap CI.

    An association measured without controls. Reported as such.
    """
    from scipy import stats

    if "review_score" not in df.columns:
        return {"available": False,
                "reason": "review_score is not in the modelling table"}

    r = df.dropna(subset=["review_score"])
    late = r.loc[r[TARGET] == 1, "review_score"].to_numpy(dtype=float)
    ontime = r.loc[r[TARGET] == 0, "review_score"].to_numpy(dtype=float)
    if not len(late) or not len(ontime):
        return {"available": False, "reason": "one group is empty"}

    diff = float(late.mean() - ontime.mean())
    rng = np.random.default_rng(RANDOM_STATE)
    boot = np.array([
        rng.choice(late, len(late)).mean() - rng.choice(ontime, len(ontime)).mean()
        for _ in range(n_boot)
    ])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    u, p = stats.mannwhitneyu(late, ontime, alternative="less")

    # Rank-biserial correlation: an effect size that does not assume normality.
    rbc = float(1 - (2 * u) / (len(late) * len(ontime)))

    return {
        "available": True,
        "estimator": "difference in group means (late minus on-time), no controls",
        "n_reviewed": int(len(r)),
        "n_late": int(len(late)),
        "n_ontime": int(len(ontime)),
        "mean_late": float(late.mean()),
        "mean_ontime": float(ontime.mean()),
        "mean_difference": diff,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "bootstrap_resamples": n_boot,
        "mannwhitney_u": float(u),
        "p_value": float(p),
        "rank_biserial": rbc,
        "sample": "full modelling table, before the train/test split",
        "caveat": "Association, not a causal estimate. No covariates, no controls.",
    }


# ==========================================================================
# 4. SHAP on the real model
# ==========================================================================

def shap_drivers(fitted, train: pd.DataFrame, cols: list[str],
                 n_explain: int = 300, n_background: int = 200) -> dict:
    """Global driver ranking by mean |SHAP|, from the fitted empirical model."""
    try:
        from olist.bll.explain import explain_orders
    except Exception as exc:  # pragma: no cover
        return {"available": False, "reason": f"import failed: {exc}"}

    try:
        bg = train[cols].sample(min(n_background, len(train)), random_state=RANDOM_STATE)
        sample = train[cols].sample(min(n_explain, len(train)), random_state=7)
        explanations = explain_orders(fitted, sample, background=bg)

        totals: dict[str, float] = {}
        for e in explanations:
            for feat, val in e.contributions.items():
                totals[feat] = totals.get(feat, 0.0) + abs(float(val))
        ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
        n = len(explanations)
        return {
            "available": True,
            "n_explained": n,
            "n_background": len(bg),
            "top_drivers": [{"feature": f, "mean_abs_shap": v / n} for f, v in ranked[:20]],
            "note": "SHAP runs on the inner uncalibrated classifier, so these "
                    "rank drivers but do not additively decompose the "
                    "calibrated probability.",
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


# ==========================================================================
# 5. Report
# ==========================================================================

def _row(name: str, m: dict) -> str:
    g = lambda k: f"{m[k]:.4f}" if k in m and m[k] is not None else "-"
    return (f"| {name} | {g('roc_auc')} | {g('average_precision')} | {g('brier')} | "
            f"{g('chosen_threshold')} | {g('recall_at_chosen')} | {g('precision_at_chosen')} | "
            f"{g('oracle_threshold')} | {g('recall_at_oracle')} | {g('precision_at_oracle')} |")


def write_report(ev_data: dict) -> None:
    d = ev_data["dataset"]
    L: list[str] = []
    a = L.append

    a("# Empirical evidence for the BLL paper\n")
    a(f"Generated {ev_data['run']['timestamp']} · seed {RANDOM_STATE} · "
      f"Python {ev_data['run']['python']}\n")
    a("Every number below came from this run. Nothing is transcribed.\n")

    a("\n## Dataset\n")
    a("| Quantity | Value |")
    a("|---|---|")
    a(f"| Raw order rows | {d['raw_order_rows'] or 'not recomputed'} |")
    a(f"| **Modelling rows** | **{d['modelling_rows']:,}** |")
    a(f"| Columns | {d['modelling_columns']} |")
    a(f"| Late-delivery rate | {d['late_rate']:.4%} |")
    a(f"| Date range | {d['date_min']} to {d['date_max']} |")
    a(f"| Censoring buffer | {d['censor_buffer_days']} days |")
    a(f"| Temporal split at | {d['train_end']} |")
    a(f"| Train | {d['train_rows']:,} rows ({d['train_date_min']} to "
      f"{d['train_date_max']}), late {d['train_late_rate']:.4%} |")
    a(f"| Test | {d['test_rows']:,} rows ({d['test_date_min']} to "
      f"{d['test_date_max']}), late {d['test_late_rate']:.4%} |")

    a("\n## Model performance (held-out temporal test set)\n")
    a("| Model | AUC | Avg precision | Brier | Threshold | Recall | Precision "
      "| Oracle thr. | Oracle recall | Oracle precision |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for r in ev_data["models"]:
        tag = f"{r['model']} / {r['scenario']}"
        a(_row(f"{tag} — uncalibrated", r["uncalibrated"]))
        a(_row(f"{tag} — **calibrated**", r["calibrated"]))
    a("\n*Threshold* is the cost-optimal cutoff chosen on the last 20% of the "
      "training period (the calibration slice), before the test labels are "
      "seen; recall and precision are measured on test at that cutoff. "
      "*Oracle* columns pick the cutoff on the test labels themselves: an "
      "upper bound, not an estimate of deployed performance. The uncalibrated "
      "baselines were fitted on the whole training period, so the holdout "
      "slice is in-sample for them and their threshold is optimistic.\n")

    a("\n### Calibration effect\n")
    a("| Model | Brier before | Brier after | Improvement |")
    a("|---|---|---|---|")
    for r in ev_data["models"]:
        a(f"| {r['model']} / {r['scenario']} | {r['uncalibrated']['brier']:.4f} | "
          f"{r['calibrated']['brier']:.4f} | {r['brier_improvement']:.4f} |")
    a("\nSigmoid calibration is monotone, so AUC and average precision are "
      "preserved; only the Brier score moves.\n")

    b = ev_data["baselines"]
    a("\n## Baselines\n")
    a(f"- Positive rate in test: **{b['prevalence_in_test']:.4%}**")
    a(f"- No-skill AUC: 0.5 · no-skill average precision: "
      f"**{b['always_predict_prevalence']['average_precision']:.4f}** "
      f"· Brier {b['always_predict_prevalence']['brier']:.4f}")
    a(f"- Always-predict-late: recall 1.0, precision "
      f"{b['always_predict_late']['precision']:.4f}, flags 100% of orders\n")

    r = ev_data["review_effect"]
    a("\n## Review score and lateness\n")
    if r.get("available"):
        a(f"- Estimator: {r['estimator']}")
        a(f"- n = {r['n_reviewed']:,} reviewed orders "
          f"({r['n_late']:,} late, {r['n_ontime']:,} on time)")
        a(f"- Mean review score: late {r['mean_late']:.4f}, "
          f"on time {r['mean_ontime']:.4f}")
        a(f"- **Mean difference: {r['mean_difference']:.4f} stars** "
          f"(95% bootstrap CI {r['ci95_low']:.4f} to {r['ci95_high']:.4f}, "
          f"{r['bootstrap_resamples']} resamples)")
        a(f"- Mann-Whitney U = {r['mannwhitney_u']:.1f}, p = {r['p_value']:.3g}")
        a(f"- Rank-biserial correlation: {r['rank_biserial']:.4f}")
        a(f"- {r['caveat']}\n")
    else:
        a(f"- Not computed: {r.get('reason')}\n")

    s = ev_data["shap"]
    a("\n## SHAP — global driver ranking\n")
    if s.get("available"):
        a(f"Mean |SHAP| over {s['n_explained']} orders, "
          f"background {s['n_background']} rows.\n")
        a("| Rank | Feature | Mean \\|SHAP\\| |")
        a("|---|---|---|")
        for i, row in enumerate(s["top_drivers"], 1):
            a(f"| {i} | `{row['feature']}` | {row['mean_abs_shap']:.4f} |")
        a(f"\n{s['note']}\n")
    else:
        a(f"- Not computed: {s.get('reason')}\n")

    c = ev_data["cost_assumptions"]
    a("\n## Cost parameters (assumptions, not measurements)\n")
    a("| Parameter | Value |")
    a("|---|---|")
    for k, v in c.items():
        a(f"| `{k}` | {v} |")
    a("\nThese set the decision threshold and every monetary figure. "
      "They are stipulated, not observed.\n")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    logger.info("Wrote %s", OUT_MD)


# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild the modelling table from data/raw/ first")
    ap.add_argument("--models", nargs="+", default=["logistic_regression", "xgboost"])
    ap.add_argument("--scenarios", nargs="+", default=["post_dispatch", "pre_dispatch"])
    ap.add_argument("--skip-shap", action="store_true")
    args = ap.parse_args()

    df = load_table(args.rebuild)

    raw_orders = None
    try:
        raw = pd.read_csv(ROOT / "data" / "raw" / "olist_orders_dataset.csv")
        raw_orders = len(raw)
    except Exception:
        pass

    out: dict = {
        "run": {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "random_state": RANDOM_STATE,
        },
        "dataset": describe(df, raw_orders),
        "models": [],
        "cost_assumptions": {},
    }

    from olist.bll.brief import BusinessCostModel
    bcm = BusinessCostModel()
    out["cost_assumptions"] = {
        "cost_fn": CONFIG.cost.cost_fn,
        "cost_fp": CONFIG.cost.cost_fp,
        "refund_share": bcm.refund_share,
        "churn_share": bcm.churn_share,
        "support_cost": bcm.support_cost,
        "intervention_cost": bcm.intervention_cost,
        "currency": bcm.currency,
    }

    first_fit = None
    for scenario in args.scenarios:
        for name in args.models:
            try:
                r = run_model(df, name, scenario)
            except Exception as exc:
                logger.warning("  %s / %s failed: %s", name, scenario, exc)
                continue
            if first_fit is None:
                first_fit = (r.pop("_fitted"), r.pop("_train"), r.pop("_cols"))
            else:
                r.pop("_fitted", None); r.pop("_train", None); r.pop("_cols", None)
            out["models"].append(r)

    out["baselines"] = baselines(df)
    out["review_effect"] = review_effect(df)
    out["shap"] = ({"available": False, "reason": "skipped"} if args.skip_shap
                   or first_fit is None else shap_drivers(*first_fit))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(_clean(out), indent=2), encoding="utf-8")
    logger.info("Wrote %s", OUT_JSON)
    write_report(out)

    print("\n" + "=" * 62)
    print("Done. Send back:")
    print(f"  {OUT_MD}")
    print(f"  {OUT_JSON}")
    print("=" * 62)


if __name__ == "__main__":
    main()
