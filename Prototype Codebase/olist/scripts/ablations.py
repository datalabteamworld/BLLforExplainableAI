"""Three ablations the paper needs, and one it needs in order to be defensible.

    python scripts/ablations.py

Writes reports/ablations.md and reports/ablations.json.

Runs logistic regression under the post-dispatch scenario only -- the headline
configuration -- because the question here is not which model wins but what the
reported numbers actually mean.

  A. Calibration, isolated.
     The earlier run compared a model fitted on all of the training period
     against a calibrated model fitted on the first 80% of it, so the Brier
     improvement mixed two changes. Here the *same fitted classifier* is
     evaluated with and without the sigmoid wrapper. Nothing else differs.

  B. Calendar features.
     purchase_month, purchase_quarter and purchase_year top the SHAP ranking.
     Under a temporal split that is a warning sign: purchase_year takes values
     in the test period that never appear in training, so whatever the model
     learned about it cannot transfer. This refits without them and reports
     what it costs.

  C. Permutation importance.
     A model-agnostic cross-check of the SHAP ranking, measured on the held-out
     test set rather than on training rows, and grouped into the categories the
     paper's argument actually turns on.

Takes roughly 10-20 minutes, most of it in C.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from olist import evaluate as ev  # noqa: E402
from olist import pipeline  # noqa: E402
from olist.config import CONFIG, PROCESSED_DIR, RANDOM_STATE, REPORTS_DIR, TARGET, TIME_KEY  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("ablations")

MODEL = "logistic_regression"
SCENARIO = "post_dispatch"
HOLDOUT_FRAC = 0.2

CALENDAR = ["purchase_year", "purchase_quarter", "purchase_month"]

#: Groups for the dispatch-versus-routing argument. Anything unlisted is
#: reported under "other" rather than silently folded into a group.
GROUPS = {
    "calendar / seasonal": [
        "purchase_year", "purchase_quarter", "purchase_month", "purchase_dow",
        "purchase_hour", "purchase_week", "days_from_black_friday",
        "is_black_friday_week", "is_carnival_week", "is_holiday_purchase",
    ],
    "dispatch & promise": [
        "carrier_pickup_days", "carrier_pickup_hours", "shipping_limit_slack_days",
        "estimated_delivery_days", "working_days_in_window",
        "non_working_days_in_window", "holidays_in_window",
        "approval_delay_hours", "seller_late_rate_to_date",
        "seller_history_confidence",
    ],
    "routing / geography": [
        "customer_state", "seller_state", "customer_city", "seller_city",
        "is_same_state", "seller_customer_distance_km", "customer_lat",
        "customer_lng", "seller_lat", "seller_lng",
        "route_late_rate_to_date", "customer_state_late_rate_to_date",
    ],
    "basket / price": [
        "mean_item_price", "max_item_price", "total_price", "total_freight",
        "freight_ratio", "n_items", "n_sellers", "total_weight_g",
        "mean_product_weight_g", "max_product_volume_cm3",
    ],
}

KEEP = ("roc_auc", "average_precision", "brier", "chosen_threshold",
        "recall_at_chosen", "precision_at_chosen", "f1_at_chosen",
        "cost_per_order", "savings_vs_no_model",
        "oracle_threshold", "recall_at_oracle", "precision_at_oracle")


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def split_train(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce fit_calibrated's own date-ordered holdout, exactly."""
    ordered = train.sort_values(TIME_KEY)
    cut = int(len(ordered) * (1 - HOLDOUT_FRAC))
    return ordered.iloc[:cut], ordered.iloc[cut:]


def fit_pair(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> dict:
    """Fit once; evaluate the same classifier raw and sigmoid-wrapped."""
    from sklearn.calibration import CalibratedClassifierCV
    try:
        from sklearn.frozen import FrozenEstimator
    except ImportError:
        FrozenEstimator = None

    fit_part, calib_part = split_train(train)
    y_test = test[TARGET].to_numpy()

    base = pipeline.build_pipeline(fit_part, cols, MODEL, y=fit_part[TARGET])
    base.fit(fit_part[cols], fit_part[TARGET])

    raw = ev.evaluate(y_test, base.predict_proba(test[cols])[:, 1],
                      CONFIG.cost, "uncalibrated",
                      threshold=ev.holdout_threshold(base, train, cols, CONFIG.cost,
                                                     holdout_frac=HOLDOUT_FRAC))

    if FrozenEstimator is not None:
        cal = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    else:
        cal = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    cal.fit(calib_part[cols], calib_part[TARGET])

    wrapped = ev.evaluate(y_test, cal.predict_proba(test[cols])[:, 1],
                          CONFIG.cost, "calibrated",
                          threshold=ev.holdout_threshold(cal, train, cols, CONFIG.cost,
                                                         holdout_frac=HOLDOUT_FRAC))

    return {
        "n_fit": len(fit_part),
        "n_calibration": len(calib_part),
        "calibration_positives": int(calib_part[TARGET].sum()),
        "fit_dates": [str(fit_part[TIME_KEY].min().date()),
                      str(fit_part[TIME_KEY].max().date())],
        "calibration_dates": [str(calib_part[TIME_KEY].min().date()),
                              str(calib_part[TIME_KEY].max().date())],
        "uncalibrated": {k: raw["metrics"][k] for k in KEEP if k in raw["metrics"]},
        "calibrated": {k: wrapped["metrics"][k] for k in KEEP if k in wrapped["metrics"]},
        "_model": cal,
    }


def calendar_range_check(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Do the calendar fields even take values the model has seen before?"""
    out = {}
    for c in CALENDAR:
        if c not in train.columns:
            continue
        tr, te = set(train[c].dropna().unique()), set(test[c].dropna().unique())
        unseen = te - tr
        n_unseen = int(test[c].isin(list(unseen)).sum()) if unseen else 0
        out[c] = {
            "train_values": sorted(float(v) for v in tr)[:12],
            "test_values": sorted(float(v) for v in te)[:12],
            "test_values_unseen_in_training": sorted(float(v) for v in unseen),
            "test_rows_with_unseen_value": n_unseen,
            "share_of_test": round(n_unseen / len(test), 4),
        }
    return out


def permutation(model, test: pd.DataFrame, cols: list[str],
                n_sample: int = 8000, n_repeats: int = 5) -> list[dict]:
    from sklearn.inspection import permutation_importance

    sub = test.sample(min(n_sample, len(test)), random_state=RANDOM_STATE)
    logger.info("  permutation importance on %d test rows, %d repeats "
                "(this is the slow part)", len(sub), n_repeats)
    r = permutation_importance(
        model, sub[cols], sub[TARGET], scoring="roc_auc",
        n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=-1,
    )
    rows = [{"feature": c, "mean_auc_drop": float(m), "std": float(s)}
            for c, m, s in zip(cols, r.importances_mean, r.importances_std)]
    return sorted(rows, key=lambda d: d["mean_auc_drop"], reverse=True)


def group_totals(rows: list[dict], key: str) -> dict:
    lookup = {f: g for g, feats in GROUPS.items() for f in feats}
    totals: dict[str, float] = {g: 0.0 for g in GROUPS}
    totals["other / unmapped"] = 0.0
    unmapped: list[str] = []
    for r in rows:
        g = lookup.get(r["feature"])
        if g is None:
            g = "other / unmapped"
            if abs(r[key]) > 0:
                unmapped.append(r["feature"])
        totals[g] += max(0.0, r[key])
    return {"totals": totals, "unmapped_features": sorted(set(unmapped))}


# ==========================================================================

def report(d: dict) -> str:
    L: list[str] = []
    a = L.append
    a("# Ablations\n")
    a(f"Generated {d['run']['timestamp']} · seed {RANDOM_STATE} · "
      f"{MODEL} / {SCENARIO}\n")

    f = d["full"]
    a("\n## A. Calibration, isolated\n")
    a(f"Same fitted classifier throughout. Fitted on {f['n_fit']:,} orders "
      f"({f['fit_dates'][0]} to {f['fit_dates'][1]}); the sigmoid mapping is "
      f"learned on {f['n_calibration']:,} later orders "
      f"({f['calibration_dates'][0]} to {f['calibration_dates'][1]}, "
      f"{f['calibration_positives']:,} positives). Evaluated on the same "
      f"held-out test set.\n")
    a("| | AUC | Avg precision | Brier | Threshold | Recall | Precision "
      "| Oracle thr. | Oracle recall | Oracle precision |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for lab in ("uncalibrated", "calibrated"):
        m = f[lab]
        a(f"| {lab} | {m['roc_auc']:.4f} | {m['average_precision']:.4f} | "
          f"{m['brier']:.4f} | {m['chosen_threshold']:.4f} | "
          f"{m['recall_at_chosen']:.4f} | {m['precision_at_chosen']:.4f} | "
          f"{m['oracle_threshold']:.4f} | {m['recall_at_oracle']:.4f} | "
          f"{m['precision_at_oracle']:.4f} |")
    a("\nThresholds are chosen on the calibration slice, before the test "
      "labels are seen. Oracle columns choose them on test: an upper bound.\n")
    du = f["uncalibrated"]; dc = f["calibrated"]
    a(f"\n- Brier: **{du['brier']:.4f} → {dc['brier']:.4f}** "
      f"({(1 - dc['brier'] / du['brier']) * 100:.1f}% reduction)")
    a(f"- AUC moved by {abs(dc['roc_auc'] - du['roc_auc']):.6f}, "
      f"average precision by {abs(dc['average_precision'] - du['average_precision']):.6f}")
    a("- Sigmoid scaling is strictly monotone, so ranking metrics should be "
      "unchanged. If they are, this is the clean calibration result the paper "
      "should report.\n")

    a("\n## B. Calendar features\n")
    a("| Feature set | Features | AUC | Avg precision | Brier | Recall |")
    a("|---|---|---|---|---|---|")
    for name, r in d["variants"].items():
        m = r["calibrated"]
        a(f"| {name} | {r['n_features']} | {m['roc_auc']:.4f} | "
          f"{m['average_precision']:.4f} | {m['brier']:.4f} | "
          f"{m['recall_at_chosen']:.4f} |")

    base_auc = d["variants"]["all features"]["calibrated"]["roc_auc"]
    for name, r in d["variants"].items():
        if name == "all features":
            continue
        delta = r["calibrated"]["roc_auc"] - base_auc
        a(f"\n- **{name}**: AUC {delta:+.4f} against the full set.")

    a("\n### Do the calendar values transfer?\n")
    a("| Feature | Test values unseen in training | Test rows affected |")
    a("|---|---|---|")
    for c, v in d["calendar_range"].items():
        unseen = ", ".join(f"{x:g}" for x in v["test_values_unseen_in_training"]) or "none"
        a(f"| `{c}` | {unseen} | {v['test_rows_with_unseen_value']:,} "
          f"({v['share_of_test']:.1%}) |")
    a("\nA feature whose test values never occur in training cannot carry a "
      "learned effect into the test period, whatever attribution it receives.\n")

    a("\n## C. Permutation importance (held-out test set)\n")
    a("Drop in ROC AUC when a single feature is shuffled. Model-agnostic, "
      "measured where it matters.\n")
    a("| Rank | Feature | Mean AUC drop | SD |")
    a("|---|---|---|---|")
    for i, r in enumerate(d["permutation"][:20], 1):
        a(f"| {i} | `{r['feature']}` | {r['mean_auc_drop']:.5f} | {r['std']:.5f} |")

    g = d["permutation_groups"]
    a("\n### Grouped\n")
    a("| Group | Total AUC drop |")
    a("|---|---|")
    for k, v in sorted(g["totals"].items(), key=lambda kv: kv[1], reverse=True):
        a(f"| {k} | {v:.5f} |")
    if g["unmapped_features"]:
        a(f"\nUnmapped: {', '.join('`' + f + '`' for f in g['unmapped_features'][:15])}")
    a("\nThe dispatch-versus-routing claim turns on these two totals, not on "
      "any single feature.\n")

    return "\n".join(L)


def main() -> None:
    path = PROCESSED_DIR / "orders_features.parquet"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run collect_paper_evidence.py --rebuild first.")
    df = pd.read_parquet(path)
    train, test = pipeline.temporal_split(df)
    cols = pipeline.feature_columns(df, SCENARIO)

    out = {"run": {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "model": MODEL, "scenario": SCENARIO,
                   "n_train": len(train), "n_test": len(test)}}

    logger.info("A. calibration, isolated")
    full = fit_pair(train, test, cols)
    model = full.pop("_model")
    out["full"] = full

    logger.info("B. calendar ablations")
    variants = {"all features": {**full, "n_features": len(cols)}}
    for label, drop in [("without purchase_year", ["purchase_year"]),
                        ("without all calendar fields", CALENDAR)]:
        keep = [c for c in cols if c not in drop]
        if len(keep) == len(cols):
            continue
        logger.info("  %s (%d features)", label, len(keep))
        r = fit_pair(train, test, keep)
        r.pop("_model", None)
        variants[label] = {**r, "n_features": len(keep)}
    out["variants"] = variants
    out["calendar_range"] = calendar_range_check(train, test)

    logger.info("C. permutation importance")
    perm = permutation(model, test, cols)
    out["permutation"] = perm
    out["permutation_groups"] = group_totals(perm, "mean_auc_drop")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "ablations.json").write_text(
        json.dumps(_clean(out), indent=2), encoding="utf-8")
    (REPORTS_DIR / "ablations.md").write_text(report(out), encoding="utf-8")

    print("\n" + "=" * 62)
    print("Done. Send back:")
    print(f"  {REPORTS_DIR / 'ablations.md'}")
    print("=" * 62)


if __name__ == "__main__":
    main()
