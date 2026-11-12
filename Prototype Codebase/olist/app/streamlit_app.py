"""Late-delivery risk scorer.

Loads the single pipeline artefact produced by ``python -m olist.train``.
Because preprocessing lives inside that pipeline, this app passes a raw
feature frame straight through -- there is no separate scaler, imputer or
encoder to keep in sync, and no column list to reconstruct by hand.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from olist import calendar_br as cb  # noqa: E402
from olist.config import MODELS_DIR  # noqa: E402

st.set_page_config(page_title="Olist Delivery Risk", page_icon="📦", layout="wide")


@st.cache_resource
def load_artefact(scenario: str = "pre_dispatch") -> dict:
    path = MODELS_DIR / f"model_{scenario}.joblib"
    if not path.exists():
        st.error(f"No model at {path}. Run `python -m olist.train --tune` first.")
        st.stop()
    return joblib.load(path)


@st.cache_resource
def load_calendar() -> pd.DataFrame:
    cal = cb.build_daily_calendar("2016-01-01", "2020-12-31")
    cal["is_non_working"] = 1 - cal["is_working_day"]
    return cal


artefact = load_artefact()
model, threshold, cols = artefact["pipeline"], artefact["threshold"], artefact["feature_columns"]
calendar = load_calendar()

st.title("📦 Olist late-delivery risk")
st.caption(
    f"Model: {artefact['model_name']} · scenario: {artefact['scenario']} · "
    f"decision threshold {threshold:.3f} (cost-optimised, not 0.5) · "
    f"trained {artefact['trained_at'][:10]}"
)

with st.sidebar:
    st.header("Operating point")
    st.markdown(
        "The threshold is chosen to minimise expected cost, given what a missed "
        "late delivery costs relative to an unnecessary intervention. Move it "
        "here to trade recall against alert volume."
    )
    threshold = st.slider("Decision threshold", 0.01, 0.99, float(threshold), 0.01)

st.subheader("Order details")
c1, c2, c3 = st.columns(3)

with c1:
    purchase = st.date_input("Purchase date", pd.Timestamp("2018-06-15"))
    estimated = st.date_input("Promised delivery date", pd.Timestamp("2018-07-05"))
    n_items = st.number_input("Items in order", 1, 25, 1)
    n_sellers = st.number_input("Distinct sellers", 1, 10, 1)

with c2:
    total_price = st.number_input("Order value (R$)", 0.0, 20_000.0, 120.0, 10.0)
    total_freight = st.number_input("Freight (R$)", 0.0, 2_000.0, 18.0, 1.0)
    total_weight = st.number_input("Total weight (g)", 0.0, 100_000.0, 900.0, 100.0)
    distance = st.number_input("Seller→customer distance (km)", 0.0, 5_000.0, 420.0, 10.0)

with c3:
    customer_state = st.selectbox("Customer state", ["SP", "RJ", "MG", "RS", "PR", "BA", "PE", "CE", "AM"])
    seller_state = st.selectbox("Seller state", ["SP", "RJ", "MG", "RS", "PR", "BA", "PE", "CE", "AM"])
    payment_type = st.selectbox("Payment type", ["credit_card", "boleto", "voucher", "debit_card"])
    installments = st.number_input("Max instalments", 1, 24, 1)

if st.button("Score this order", type="primary"):
    purchase_ts = pd.Timestamp(purchase)
    estimated_ts = pd.Timestamp(estimated)
    est_days = (estimated_ts - purchase_ts).days

    if est_days <= 0:
        st.error("The promised date must fall after the purchase date.")
        st.stop()

    starts, ends = pd.Series([purchase_ts]), pd.Series([estimated_ts])
    non_working = float(cb.count_events_in_window(starts, ends, calendar, "is_non_working").iloc[0])

    row = {
        "estimated_delivery_days": est_days,
        "n_items": n_items,
        "n_sellers": n_sellers,
        "is_multi_seller": int(n_sellers > 1),
        "total_price": total_price,
        "total_freight": total_freight,
        "freight_ratio": total_freight / total_price if total_price else None,
        "total_weight_g": total_weight,
        "max_weight_g": total_weight,
        "seller_customer_distance_km": distance,
        "is_same_state": int(customer_state == seller_state),
        "customer_state": customer_state,
        "seller_state": seller_state,
        "payment_type": payment_type,
        "max_installments": installments,
        "purchase_year": purchase_ts.year,
        "purchase_month": purchase_ts.month,
        "purchase_dow": purchase_ts.dayofweek,
        "purchase_hour": 12,
        "purchase_quarter": purchase_ts.quarter,
        "is_weekend_purchase": int(purchase_ts.dayofweek >= 5),
        "non_working_days_in_window": non_working,
        "working_days_in_window": est_days - non_working,
        "non_working_ratio": non_working / est_days,
        "days_from_black_friday": float(
            cb.days_to_nearest_black_friday(pd.Series([purchase_ts]), [purchase_ts.year]).iloc[0]
        ),
    }
    # Any feature not collected by the form is left missing; the pipeline's
    # imputer fills it with the training median.
    X = pd.DataFrame([{c: row.get(c) for c in cols}])
    proba = float(model.predict_proba(X)[0, 1])

    st.divider()
    left, right = st.columns([1, 2])
    left.metric("Late-delivery probability", f"{proba:.1%}")

    if proba >= threshold:
        right.error(
            f"**Flagged.** Above the {threshold:.1%} action threshold. "
            "Escalate to logistics: expedite dispatch or send a proactive delay notice."
        )
    else:
        right.success(f"**Not flagged.** Below the {threshold:.1%} action threshold.")

    st.caption(
        f"{non_working:.0f} of the {est_days} promised days are non-working "
        f"(weekend, national holiday or Carnival), leaving "
        f"{est_days - non_working:.0f} working days of transit."
    )
    st.warning(
        "A probability is not a guarantee. Calibration drifts as the carrier "
        "network changes; re-evaluate against recent outcomes before acting on "
        "these scores at scale.",
        icon="⚠️",
    )
