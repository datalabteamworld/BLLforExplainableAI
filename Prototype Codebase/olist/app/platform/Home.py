"""Olist AI Delivery Risk Intelligence — entry point.

Run with:  streamlit run app/platform/Home.py

Every screen reads from ``olist.bll.PredictionService``; none imports the model
or SHAP directly. Capabilities the Olist data cannot support are rendered as
disabled with a stated reason rather than filled with plausible values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from olist.bll import PredictionService  # noqa: E402
from olist.bll.brief import BusinessCostModel  # noqa: E402

st.set_page_config(page_title="Olist Delivery Risk Intelligence",
                   page_icon="📦", layout="wide")

SEARCH_DIRS = [ROOT / "models", ROOT / "deploy", ROOT, Path.cwd()]
DATA_DIRS = [ROOT / "data" / "processed", ROOT / "deploy", ROOT, Path.cwd()]


def _find(patterns: list[str], dirs: list[Path]) -> Path | None:
    for directory in dirs:
        for pattern in patterns:
            hits = sorted(directory.glob(pattern)) if directory.exists() else []
            if hits:
                return hits[0]
    return None


@st.cache_resource(show_spinner="Loading model…")
def load_service(refund: float, churn: float, support: float,
                 action: float, currency: str) -> PredictionService | None:
    model_path = _find(["model_post_dispatch.joblib", "model_*.joblib"], SEARCH_DIRS)
    if model_path is None:
        return None

    history_path = _find(["orders_features.parquet", "sample_orders.parquet"], DATA_DIRS)
    background_path = _find(["background.parquet"], DATA_DIRS)

    costs = BusinessCostModel(currency=currency, refund_share=refund,
                              churn_share=churn, support_cost=support,
                              intervention_cost=action)
    return PredictionService.from_files(model_path, history_path, background_path, costs=costs)


with st.sidebar:
    st.title("📦 Delivery Risk")
    st.caption("Explainable AI decision support for e-commerce logistics")
    st.divider()

    st.subheader("Business assumptions")
    st.caption(
        "**Assumptions, not measured Olist finances.** Every monetary figure in "
        "this application depends on them, so they are exposed here rather than "
        "buried in the code."
    )
    currency = st.text_input("Currency", "BRL")
    refund = st.slider("Refund exposure (share of order value)", 0.0, 1.0, 0.42, 0.01)
    churn = st.slider("Repeat-purchase risk (share of order value)", 0.0, 1.0, 0.46, 0.01)
    support = st.number_input("Support cost per contact", 0.0, 500.0, 12.0)
    action = st.number_input("Cost of intervening", 0.0, 500.0, 4.0)

service = load_service(refund, churn, support, action, currency)

if service is None:
    st.error(
        "**No trained model found.**\n\n"
        "Expected `model_post_dispatch.joblib` in `models/` or `deploy/`.\n\n"
        "Generate one with:\n"
        "```bash\n"
        "python -m olist.validate --build\n"
        "python -m olist.train --scenario post_dispatch\n"
        "```"
    )
    st.stop()

st.session_state["service"] = service
st.session_state["currency"] = currency

st.title("Olist AI Delivery Risk Intelligence")
st.caption(
    "Predict the risk. Explain the risk. Reduce the risk. Quantify the value of acting."
)

history = service.history
if not len(history):
    st.warning("No historical orders loaded — dashboard and evidence screens are unavailable.")
    st.stop()

if "risk" not in st.session_state:
    with st.spinner("Scoring orders…"):
        st.session_state["risk"] = service.predict_probability(history)
risk = st.session_state["risk"]

flagged = (risk >= service.threshold)
exposure = (history["total_price"] * (refund + churn) + support) if "total_price" in history else pd.Series(dtype=float)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Orders scored", f"{len(history):,}")
c2.metric("High risk (≥50%)", f"{(risk >= 0.5).sum():,}")
c3.metric("Flagged at threshold", f"{flagged.sum():,}", f"{flagged.mean():.1%} of orders")
c4.metric("Mean predicted risk", f"{risk.mean():.1%}")

if len(exposure):
    d1, d2 = st.columns(2)
    d1.metric("Expected loss, flagged orders",
              f"{currency} {(risk[flagged] * exposure[flagged]).sum():,.0f}")
    d2.metric("Total exposure, all orders", f"{currency} {exposure.sum():,.0f}")

st.divider()
st.subheader("Where to go next")

a, b, c = st.columns(3)
a.info("**Risk Dashboard**\n\nPortfolio view: risk distribution, breakdown by state, "
       "seller and category, and financial exposure.")
b.info("**Orders**\n\nOrder-level table with risk, confidence and expected loss. "
       "Select an order to open its full assessment.")
c.info("**Model Insights**\n\nGlobal feature importance, model card, and the "
       "capabilities this dataset does and does not support.")

st.caption(
    f"Model: {service.artefact.get('model_name')} · scenario {service.scenario} · "
    f"decision threshold {service.threshold:.3f} (cost-optimised, not 0.5)"
)
