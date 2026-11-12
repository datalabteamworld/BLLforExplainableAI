"""Portfolio-level risk dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Risk Dashboard", page_icon="📊", layout="wide")

service = st.session_state.get("service")
if service is None:
    st.warning("Open the **Home** page first to load the model.")
    st.stop()

currency = st.session_state.get("currency", "BRL")
history = service.history.copy()
history["risk"] = st.session_state["risk"]
history["risk_level"] = [service.risk_level(p) for p in history["risk"]]

st.title("Risk Dashboard")

# --- filters --------------------------------------------------------------
with st.expander("Filters", expanded=False):
    f1, f2, f3 = st.columns(3)
    states = f1.multiselect("Customer state", sorted(history["customer_state"].dropna().unique()))
    categories = f2.multiselect("Product category", sorted(history["product_category"].dropna().unique())[:40])
    levels = f3.multiselect("Risk level", ["HIGH", "MEDIUM", "LOW"])

view = history
if states:
    view = view[view["customer_state"].isin(states)]
if categories:
    view = view[view["product_category"].isin(categories)]
if levels:
    view = view[view["risk_level"].isin(levels)]

if view.empty:
    st.warning("No orders match these filters.")
    st.stop()

# --- KPIs -----------------------------------------------------------------
counts = view["risk_level"].value_counts()
k = st.columns(5)
k[0].metric("Orders", f"{len(view):,}")
k[1].metric("High risk", f"{counts.get('HIGH', 0):,}")
k[2].metric("Medium risk", f"{counts.get('MEDIUM', 0):,}")
k[3].metric("Low risk", f"{counts.get('LOW', 0):,}")
k[4].metric("Mean risk", f"{view['risk'].mean():.1%}")

costs = service.costs
exposure = view["total_price"] * (costs.refund_share + costs.churn_share) + costs.support_cost
m = st.columns(3)
m[0].metric("Total exposure", f"{currency} {exposure.sum():,.0f}")
m[1].metric("Expected loss", f"{currency} {(view['risk'] * exposure).sum():,.0f}")
m[2].metric("Intervention rate", f"{(view['risk'] >= service.threshold).mean():.1%}")

st.divider()

# --- distribution and time series ----------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Risk distribution")
    # pd.cut yields Interval objects, which Altair cannot serialise.
    counts = (pd.cut(view["risk"], np.arange(0, 1.05, 0.05))
                .value_counts().sort_index().rename("orders"))
    counts.index = [f"{i.left:.0%}" for i in counts.index]
    counts.index.name = "risk band"
    st.bar_chart(counts)

with right:
    st.subheader("Mean risk over time")
    monthly = (view.set_index("order_purchase_timestamp")
                   .resample("ME")["risk"].agg(["mean", "size"]))
    st.line_chart(monthly[monthly["size"] >= 30]["mean"])

# --- breakdowns -----------------------------------------------------------
st.subheader("Where risk concentrates")


def _breakdown(column: str, label: str, min_orders: int = 50, top: int = 12):
    if column not in view.columns:
        st.caption(f"{label}: not available in this dataset.")
        return
    grouped = (view.groupby(column)
                   .agg(mean_risk=("risk", "mean"), orders=("risk", "size"))
                   .query("orders >= @min_orders")
                   .sort_values("mean_risk", ascending=False)
                   .head(top))
    if grouped.empty:
        st.caption(f"{label}: too few orders per group to summarise.")
        return
    st.caption(f"**{label}** (min {min_orders} orders)")
    st.bar_chart(grouped["mean_risk"])


tabs = st.tabs(["Customer state", "Product category", "Seller state", "Payment type"])
with tabs[0]:
    _breakdown("customer_state", "Mean predicted risk by customer state")
with tabs[1]:
    _breakdown("product_category", "Mean predicted risk by product category")
with tabs[2]:
    _breakdown("seller_state", "Mean predicted risk by seller state")
with tabs[3]:
    _breakdown("payment_type", "Mean predicted risk by payment type", min_orders=20)

st.divider()
st.subheader("Predicted risk against observed outcome")
if "is_late_delivery" in view.columns:
    observed = (view.assign(band=pd.cut(view["risk"], np.arange(0, 1.05, 0.1)))
                    .groupby("band", observed=True)
                    .agg(predicted=("risk", "mean"),
                         observed=("is_late_delivery", "mean"),
                         orders=("risk", "size"))
                    .query("orders >= 20"))
    observed.index = [f"{i.left:.0%}" for i in observed.index]
    observed.index.name = "risk band"
    st.line_chart(observed[["predicted", "observed"]])
    st.caption(
        "A calibrated model tracks the diagonal: orders predicted at 30% should be "
        "late about 30% of the time. Divergence indicates calibration drift."
    )
else:
    st.caption("Observed outcomes are not present in this dataset.")
