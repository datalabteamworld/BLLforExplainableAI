"""BLL decision surface — the live form of Figure 5.

Renders one order's decision brief: risk level, plainly worded drivers,
monetised impact, and a ranked table of costed options with a what-if control.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from olist import bll  # noqa: E402
from olist.bll.brief import BusinessCostModel  # noqa: E402
from olist.config import MODELS_DIR, PROCESSED_DIR  # noqa: E402

st.set_page_config(page_title="Delivery risk — BLL", page_icon="📦", layout="wide")


@st.cache_resource
def load(scenario: str):
    path = MODELS_DIR / f"model_{scenario}.joblib"
    if not path.exists():
        st.error(f"No model at {path}. Run `python -m olist.train --scenario {scenario}` first.")
        st.stop()
    return joblib.load(path)


@st.cache_data
def load_orders():
    path = PROCESSED_DIR / "orders_features.parquet"
    if not path.exists():
        st.error("No feature table. Run `python -m olist.validate --build` first.")
        st.stop()
    return pd.read_parquet(path)


with st.sidebar:
    st.header("Configuration")
    scenario = st.selectbox("Scenario", ["post_dispatch", "pre_dispatch"])
    st.caption(
        "Pre-dispatch uses only what is known at order time. Post-dispatch adds "
        "carrier handover timing — more accurate, but only available once the "
        "parcel has left the seller."
    )
    st.divider()
    st.subheader("Cost assumptions")
    st.caption("These are assumptions, not measured Olist finances. Every figure below depends on them.")
    currency = st.text_input("Currency", "BRL")
    refund = st.slider("Refund exposure (share of order value)", 0.0, 1.0, 0.42, 0.01)
    churn = st.slider("Repeat-purchase risk (share of order value)", 0.0, 1.0, 0.46, 0.01)
    support = st.number_input("Support cost per contact", 0.0, 500.0, 12.0)
    action = st.number_input("Cost of intervening", 0.0, 500.0, 4.0)

costs = BusinessCostModel(currency=currency, refund_share=refund, churn_share=churn,
                          support_cost=support, intervention_cost=action)

artefact = load(scenario)
model, cols = artefact["pipeline"], artefact["feature_columns"]
df = load_orders()

split = pd.Timestamp("2018-04-30")
test = df[df["order_purchase_timestamp"] > split].reset_index(drop=True)
background = df[df["order_purchase_timestamp"] <= split][cols].head(300)

if "risk" not in st.session_state or st.session_state.get("scenario") != scenario:
    st.session_state["risk"] = model.predict_proba(test[cols])[:, 1]
    st.session_state["scenario"] = scenario
risk = st.session_state["risk"]

st.title("📦 Delivery risk assessment")
st.caption(
    f"{artefact['model_name']} · {scenario} · decision threshold "
    f"{artefact['threshold']:.3f} · Business Language Layer"
)

mode = st.radio("Choose an order", ["Highest risk", "By index"], horizontal=True)
idx = int(risk.argmax()) if mode == "Highest risk" else st.number_input(
    "Row", 0, len(test) - 1, 0, 1
)

row = test[cols].iloc[[idx]]
order_value = float(test["total_price"].iloc[idx])

with st.spinner("Computing explanation…"):
    explanation = bll.explain_order(model, row, background=background)
    brief = bll.build_brief(model, row, explanation, order_value=order_value,
                            costs=costs, order_id=test["order_id"].iloc[idx])

# --- 1. Interpret --------------------------------------------------------
left, right = st.columns([1, 1])
left.metric("Late-delivery probability", f"{brief.probability:.1%}", brief.risk_level)
right.success(f"**Recommended action**\n\n{brief.recommendation.name}")
if brief.probability >= 0.25:
    st.warning("Likely to arrive after the promised date — intervention recommended.")

# --- 2. Contextualise ----------------------------------------------------
st.subheader("Key contributing factors")
for d in brief.drivers:
    box = st.error if d.raises_risk else st.success
    box(f"**{d.label}** — {d.sentence}  \n`{d.badge}`"
        + ("  ·  `Actionable`" if d.is_lever else ""))

st.subheader("What this means")
for line in brief.consequences:
    st.write("•", line)

# --- 3. Quantify ---------------------------------------------------------
st.subheader("Business impact if late")
for col, (name, amount) in zip(st.columns(len(brief.impact)), brief.impact.items()):
    col.metric(name, costs.format(amount))
st.caption(f"Total exposure on this order: **{costs.format(brief.exposure)}**")

# --- 4. Recommend --------------------------------------------------------
st.subheader("Options")
st.dataframe(
    pd.DataFrame([
        {
            "Option": o.name,
            "Risk": f"{o.risk:.1%}" if o.feasible else "unavailable",
            "Action cost": costs.format(o.action_cost) if o.feasible else "—",
            "Expected loss": costs.format(o.expected_loss) if o.feasible else "—",
            "Recommended": "✓" if o.recommended else "",
        }
        for o in brief.options
    ]),
    hide_index=True, width="stretch",
)
st.info(f"**{brief.headline()}**  Net saving vs shipping as-is: "
        f"**{costs.format(brief.net_saving)}**")

# --- What-if -------------------------------------------------------------
levers = [d.feature for d in brief.drivers if d.is_lever and d.feature in cols]
if levers:
    st.subheader("What if…")
    lever = st.selectbox("Change", levers,
                         format_func=lambda f: bll.glossary.GLOSSARY[f].label)
    current = float(row[lever].iloc[0])
    new = st.slider(f"{bll.glossary.GLOSSARY[lever].label} (now {current:.1f})",
                    0.0, max(current * 3, 10.0), current, 0.5)
    new_risk = bll.counterfactual(model, row, lever, new)
    a, b = st.columns(2)
    a.metric("Risk now", f"{brief.probability:.1%}")
    b.metric("Risk after change", f"{new_risk:.1%}",
             f"{new_risk - brief.probability:+.1%}")

# --- Traceability & caveats ---------------------------------------------
with st.expander("Audit trail — every statement and its attribution"):
    st.dataframe(brief.audit(), hide_index=True, width="stretch")
    st.caption(f"Glossary coverage of shown drivers: {brief.glossary_coverage:.0%}")

for caveat in brief.caveats:
    st.caption(f"⚠️ {caveat}")
