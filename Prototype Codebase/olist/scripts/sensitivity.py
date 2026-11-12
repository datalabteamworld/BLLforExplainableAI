"""How much of the BLL's recommendation is the model, and how much is the assumptions?

    python scripts/sensitivity.py

Writes reports/sensitivity.md and reports/sensitivity.json.

The decision brief ranks options by expected cost:

    total(option) = risk(option) x exposure(order_value) + action_cost
    exposure      = (refund_share + churn_share) x order_value + support_cost

Only ``risk`` comes from the model. ``refund_share``, ``churn_share``,
``support_cost`` and ``intervention_cost`` are stipulated, and the paper has to
say what happens to the recommendation when they move.

The risks do not depend on the cost parameters, so the model is run once per
intervention and the sweep is then pure arithmetic -- a few seconds, whatever
the grid.

Reports three things:

  1. One-way sweeps. Vary one parameter, hold the rest, record how often the
     recommendation changes.
  2. Break-even intervention cost. For each order, the action cost at which
     intervening stops paying. The distribution of that number is the honest
     answer to "is this recommendation robust?"
  3. Stability. The share of orders whose recommendation is unchanged across
     the whole plausible range.

Takes 2-5 minutes.
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

from olist import currency, pipeline  # noqa: E402
from olist.bll.brief import INTERVENTIONS, BusinessCostModel  # noqa: E402
from olist.config import PROCESSED_DIR, RANDOM_STATE, REPORTS_DIR, TARGET, TIME_KEY  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("sensitivity")

MODEL, SCENARIO = "logistic_regression", "post_dispatch"
N_ORDERS = 500

#: Naira cost parameters. The two shares are currency-neutral and unchanged.
#: The two absolute amounts are stipulated assumptions, so they are stated
#: directly in naira rather than converted from the BRL defaults -- converting
#: an assumption would imply it had been measured in another currency.
BASE = BusinessCostModel(currency="NGN", support_cost=1200.0, intervention_cost=400.0)

#: The BRL defaults, kept for traceability.
BASE_BRL = BusinessCostModel()

#: Ranges a reviewer would consider plausible. Each spans well beyond the
#: default so the sweep shows where the recommendation actually breaks, not
#: merely that it survives a nudge.
SWEEPS = {
    "refund_share": np.round(np.arange(0.00, 1.01, 0.05), 3),
    "churn_share": np.round(np.arange(0.00, 1.01, 0.05), 3),
    "support_cost": np.round(np.arange(0.0, 6001.0, 500.0), 2),
    "intervention_cost": np.round(np.arange(0.0, 4001.0, 200.0), 2),
}


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


def fit(df: pd.DataFrame):
    train, test = pipeline.temporal_split(df)
    cols = pipeline.feature_columns(df, SCENARIO)
    base = pipeline.build_pipeline(train, cols, MODEL, y=train[TARGET])
    model = pipeline.fit_calibrated(base, train, cols)
    return model, test, cols


def option_risks(model, orders: pd.DataFrame, cols: list[str]) -> dict[str, np.ndarray]:
    """Baseline risk plus one risk vector per intervention. One model call each."""
    risks = {"Ship as-is": model.predict_proba(orders[cols])[:, 1]}
    for name, feature, target, _ in INTERVENTIONS:
        if feature not in orders.columns:
            logger.info("  %s: lever %r absent, skipped", name, feature)
            continue
        perturbed = orders.copy()
        if target is None:  # "extend the promise" is relative to the current value
            perturbed[feature] = perturbed[feature].astype(float) + 5
        else:
            perturbed[feature] = float(target)
        risks[name] = model.predict_proba(perturbed[cols])[:, 1]
        logger.info("  %s: mean risk %.4f -> %.4f", name,
                    risks["Ship as-is"].mean(), risks[name].mean())
    return risks


def recommend(risks: dict[str, np.ndarray], values: np.ndarray,
              refund: float, churn: float, support: float,
              action: float) -> tuple[np.ndarray, np.ndarray]:
    """Winning option per order, and the saving against shipping as-is."""
    exposure = (refund + churn) * values + support
    names = list(risks)
    totals = np.stack([
        risks[n] * exposure + (0.0 if n == "Ship as-is" else action)
        for n in names
    ])
    best = totals.argmin(axis=0)
    saving = totals[0] - totals.min(axis=0)
    return np.array(names, dtype=object)[best], saving


def break_even(risks: dict[str, np.ndarray], values: np.ndarray,
               refund: float, churn: float, support: float) -> np.ndarray:
    """Action cost at which the best intervention stops beating ship-as-is.

    Intervening is worth it while  action_cost < (p0 - p_i) x exposure.
    """
    exposure = (refund + churn) * values + support
    p0 = risks["Ship as-is"]
    gains = [(p0 - risks[n]) * exposure for n in risks if n != "Ship as-is"]
    return np.max(np.stack(gains), axis=0) if gains else np.zeros_like(p0)


def main() -> None:
    path = PROCESSED_DIR / "orders_features.parquet"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run collect_paper_evidence.py --rebuild first.")

    df = pd.read_parquet(path)
    logger.info("Fitting %s / %s", MODEL, SCENARIO)
    model, test, cols = fit(df)

    orders = test.sample(min(N_ORDERS, len(test)), random_state=RANDOM_STATE).copy()
    values_brl = (orders["total_price"].astype(float).to_numpy()
                  if "total_price" in orders else np.full(len(orders), 100.0))

    # Real Brazilian order values, converted per order at its purchase year.
    conv = currency.to_ngn(values_brl, orders[TIME_KEY].dt.year)
    values = conv.ngn
    logger.info("Converted %d order values to NGN, effective rate "
                "N%.2f/BRL (%s)", len(values), conv.effective_rate, conv.basis)

    logger.info("Scoring %d test orders under each lever", len(orders))
    risks = option_risks(model, orders, cols)

    base_rec, base_saving = recommend(
        risks, values, BASE.refund_share, BASE.churn_share,
        BASE.support_cost, BASE.intervention_cost)

    out = {
        "run": {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "model": MODEL, "scenario": SCENARIO, "n_orders": len(orders),
                "seed": RANDOM_STATE},
        "defaults": {"refund_share": BASE.refund_share, "churn_share": BASE.churn_share,
                     "support_cost": BASE.support_cost,
                     "intervention_cost": BASE.intervention_cost,
                     "currency": BASE.currency},
        "currency": {
            "presentation": "NGN",
            "source_currency": "BRL",
            "basis": conv.basis,
            "source": conv.source,
            "rates_ngn_per_brl": conv.years_used,
            "effective_rate_ngn_per_brl": conv.effective_rate,
            "orders_outside_rate_table": conv.n_outside_rate_table,
            "disclosure": currency.disclosure(),
        },
        "order_value_ngn": {"mean": float(values.mean()),
                            "median": float(np.median(values)),
                            "p10": float(np.percentile(values, 10)),
                            "p90": float(np.percentile(values, 90))},
        "order_value_brl": {"mean": float(values_brl.mean()),
                            "median": float(np.median(values_brl))},
        "baseline": {
            "mean_risk": float(risks["Ship as-is"].mean()),
            "recommendation_mix": {k: int(v) for k, v in
                                   pd.Series(base_rec).value_counts().items()},
            "mean_saving": float(base_saving.mean()),
            "median_saving": float(np.median(base_saving)),
        },
        "sweeps": {},
    }

    logger.info("Sweeping")
    for param, grid in SWEEPS.items():
        rows = []
        for v in grid:
            kw = {"refund": BASE.refund_share, "churn": BASE.churn_share,
                  "support": BASE.support_cost, "action": BASE.intervention_cost}
            kw[{"refund_share": "refund", "churn_share": "churn",
                "support_cost": "support", "intervention_cost": "action"}[param]] = float(v)
            rec, saving = recommend(risks, values, kw["refund"], kw["churn"],
                                    kw["support"], kw["action"])
            rows.append({
                "value": float(v),
                "changed_share": float((rec != base_rec).mean()),
                "intervene_share": float((rec != "Ship as-is").mean()),
                "mean_saving": float(saving.mean()),
            })
        out["sweeps"][param] = rows

    be = break_even(risks, values, BASE.refund_share, BASE.churn_share, BASE.support_cost)
    out["break_even_action_cost"] = {
        "mean": float(be.mean()), "median": float(np.median(be)),
        "p10": float(np.percentile(be, 10)), "p25": float(np.percentile(be, 25)),
        "p75": float(np.percentile(be, 75)), "p90": float(np.percentile(be, 90)),
        "share_above_default": float((be > BASE.intervention_cost).mean()),
        "default_action_cost": BASE.intervention_cost,
    }

    # Stability: how often does the recommendation survive the whole grid?
    stacked = []
    for param, grid in SWEEPS.items():
        for v in grid:
            kw = {"refund": BASE.refund_share, "churn": BASE.churn_share,
                  "support": BASE.support_cost, "action": BASE.intervention_cost}
            kw[{"refund_share": "refund", "churn_share": "churn",
                "support_cost": "support", "intervention_cost": "action"}[param]] = float(v)
            rec, _ = recommend(risks, values, kw["refund"], kw["churn"],
                               kw["support"], kw["action"])
            stacked.append(rec == base_rec)
    agree = np.stack(stacked).mean(axis=0)
    out["stability"] = {
        "mean_agreement_with_default": float(agree.mean()),
        "share_never_changing": float((agree == 1.0).mean()),
        "share_changing_often": float((agree < 0.75).mean()),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "sensitivity.json").write_text(
        json.dumps(_clean(out), indent=2), encoding="utf-8")

    # ---- report ----
    L: list[str] = []
    a = L.append
    c = BASE.currency  # NGN
    a("# Sensitivity of the recommendation to its cost assumptions\n")
    a(f"Generated {out['run']['timestamp']} · {MODEL} / {SCENARIO} · "
      f"{len(orders)} held-out test orders · seed {RANDOM_STATE}\n")
    a(f"Defaults: refund_share {BASE.refund_share}, churn_share {BASE.churn_share}, "
      f"support_cost {currency.fmt(BASE.support_cost)}, intervention_cost "
      f"{currency.fmt(BASE.intervention_cost)}.\n")
    a(f"\nMonetary values are Olist order values in Brazilian reais, converted "
      f"per order at its purchase year's official annual average rate "
      f"(effective \u20a6{conv.effective_rate:,.2f}/BRL). "
      + "; ".join(f"{y}: \u20a6{r:,.2f}/BRL" for y, r in conv.years_used.items())
      + f". Source: {conv.source}\n")
    a(f"\nMean order value {currency.fmt(values.mean())} "
      f"(R${values_brl.mean():,.2f}). Converted values express Brazilian "
      f"transactions on a naira scale; they are not Nigerian transactions.\n")
    b = out["baseline"]
    a(f"\nAt those defaults: mean risk {b['mean_risk']:.4f}, mean saving "
      f"{currency.fmt(b['mean_saving'])} per order. Recommendation mix: "
      + ", ".join(f"{k} {v}" for k, v in b["recommendation_mix"].items()) + ".\n")

    a("\n## One-way sweeps\n")
    a("`changed` is the share of orders whose recommendation differs from the "
      "one given at the default values.\n")
    for param, rows in out["sweeps"].items():
        a(f"\n### `{param}`\n")
        a("| Value | Changed | Intervene | Mean saving |")
        a("|---|---|---|---|")
        for r in rows:
            a(f"| {r['value']:g} | {r['changed_share']:.1%} | "
              f"{r['intervene_share']:.1%} | {currency.fmt(r['mean_saving'])} |")

    k = out["break_even_action_cost"]
    a("\n## Break-even action cost\n")
    a("The action cost at which intervening stops paying, per order.\n")
    a("| Percentile | Value |")
    a("|---|---|")
    for lab in ("p10", "p25", "median", "p75", "p90"):
        a(f"| {lab} | {currency.fmt(k[lab])} |")
    a(f"\n**{k['share_above_default']:.1%} of orders break even above the "
      f"assumed action cost of {currency.fmt(k['default_action_cost'])}** — that is, "
      f"intervening pays for them. The rest do not, and the brief should not "
      f"recommend intervening on them.\n")

    s = out["stability"]
    a("\n## Stability\n")
    a(f"- Mean agreement with the default recommendation across the whole grid: "
      f"**{s['mean_agreement_with_default']:.1%}**")
    a(f"- Orders whose recommendation never changes: "
      f"**{s['share_never_changing']:.1%}**")
    a(f"- Orders whose recommendation changes in more than a quarter of settings: "
      f"{s['share_changing_often']:.1%}\n")
    a("\nRead this as the honest bound on every monetary claim in the paper: the "
      "model's ranking of risk is empirical, but the decision drawn from it is "
      "a function of four stipulated numbers, and this is how much it moves "
      "when they do.\n")

    (REPORTS_DIR / "sensitivity.md").write_text("\n".join(L), encoding="utf-8")

    print("\n" + "=" * 62)
    print("Done. Send back:")
    print(f"  {REPORTS_DIR / 'sensitivity.md'}")
    print("=" * 62)


if __name__ == "__main__":
    main()
