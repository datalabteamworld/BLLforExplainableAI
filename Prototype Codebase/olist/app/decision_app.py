"""Delivery Risk — a Business Language Layer decision surface.

One screen per order, structured to match the four functions of the BLL
framework (Oyeleke et al., 2026): interpret, contextualise, quantify,
recommend.

Design brief. The reader is a logistics operations lead with a queue of orders
and a few seconds per decision, not an analyst. So: one screen, one order, one
recommendation. The visual language is borrowed from the shipping label and
the freight manifest — a monospace utility face for identifiers and figures,
tabular alignment for anything that will be compared, and a single accent
carried only by the risk state. Nothing decorative competes with the number
that drives the decision.

Run with:  streamlit run app/decision_app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", Path.cwd() / "src"):
    if candidate.exists():
        sys.path.insert(0, str(candidate))

from olist.bll import PredictionService, glossary  # noqa: E402
from olist.bll.brief import BusinessCostModel  # noqa: E402

st.set_page_config(page_title="Delivery Risk · BLL", page_icon="📦", layout="wide")

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
# Ink and paper, with one accent that only ever means "risk state". Figures use
# a monospace face so that columns of numbers align optically and an operations
# lead can scan them the way they scan a manifest.

st.markdown("""
<style>
  /* Colours inherit from Streamlit's active theme rather than being fixed,
     so the surface reads correctly in both light and dark mode. The accent
     hues are chosen to hold contrast against either background. */
  :root {
    --ink:      var(--text-color, #16212B);
    --muted:    color-mix(in srgb, var(--text-color, #16212B) 62%, transparent);
    --hairline: color-mix(in srgb, var(--text-color, #16212B) 18%, transparent);
    --surface:  var(--secondary-background-color, #F7F9FA);
    --high:     #E05A4F;
    --medium:   #D89020;
    --low:      #3AA76D;
  }
  .block-container { padding-top: 2.2rem; max-width: 1180px; }

  .eyebrow {
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: .70rem; letter-spacing: .14em; text-transform: uppercase;
    color: var(--muted); margin-bottom: .35rem;
  }
  .figure {
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: 2.9rem; line-height: 1.05; font-weight: 600; color: var(--ink);
    font-variant-numeric: tabular-nums;
  }
  .figure-sm {
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: 1.35rem; font-weight: 600; color: var(--ink);
    font-variant-numeric: tabular-nums;
  }
  .state { font-size: .78rem; font-weight: 700; letter-spacing: .08em; color: var(--muted); }
  .state-high { color: var(--high); }
  .state-medium { color: var(--medium); }
  .state-low { color: var(--low); }

  .panel {
    border: 1px solid var(--hairline); border-radius: 3px;
    padding: 1.05rem 1.2rem; background: var(--surface);
  }

  .driver {
    border-left: 3px solid var(--hairline);
    padding: .5rem 0 .5rem .85rem; margin-bottom: .55rem;
  }
  .driver-up { border-left-color: var(--high); }
  .driver-down { border-left-color: var(--low); }
  .driver-name {
    font-weight: 650; color: var(--ink); font-size: .95rem;
    margin-bottom: .12rem;
  }
  .driver-text { color: var(--muted); font-size: .87rem; line-height: 1.45; }
  .tag {
    font-family: ui-monospace, monospace; font-size: .68rem;
    letter-spacing: .06em; color: var(--muted);
    border: 1px solid var(--hairline); border-radius: 2px;
    padding: .08rem .4rem; margin-right: .3rem;
  }

  .plain-terms {
    border-left: 3px solid var(--ink); background: var(--surface);
    padding: .95rem 1.15rem; font-size: 1.02rem; line-height: 1.6;
    color: var(--ink);
  }
  .silent {
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: .78rem; line-height: 1.7; color: var(--muted);
    background: var(--surface); border: 1px solid var(--hairline);
    border-radius: 3px; padding: 1rem 1.15rem; white-space: pre;
    overflow-x: auto;
  }
  .caption { color: var(--muted); font-size: .8rem; line-height: 1.5; }
  hr { border-color: var(--hairline); }

  /* Figure 7's ladder: each section is one rung of the question sequence a
     decision requires. The numbering encodes that sequence. */
  .rung { display: flex; align-items: baseline; gap: .7rem; margin: .2rem 0 .1rem; }
  .rung-no {
    font-family: ui-monospace, monospace; font-size: .72rem; font-weight: 700;
    color: var(--background-color, #fff); background: var(--ink);
    border-radius: 2px; padding: .1rem .38rem; letter-spacing: .04em;
  }
  .rung-q {
    font-family: ui-monospace, monospace; font-size: .70rem;
    letter-spacing: .13em; text-transform: uppercase; color: var(--muted);
  }
  .rung-owner {
    font-family: ui-monospace, monospace; font-size: .62rem;
    letter-spacing: .1em; color: var(--muted);
    border: 1px solid var(--hairline); border-radius: 2px; padding: .04rem .32rem;
  }

  /* Figure 4's example strip: one order, five registers. */
  .register {
    border: 1px solid var(--hairline); border-radius: 3px;
    padding: .7rem .8rem; height: 100%;
  }
  .register-label {
    font-family: ui-monospace, monospace; font-size: .62rem;
    letter-spacing: .11em; text-transform: uppercase; color: var(--muted);
    margin-bottom: .3rem;
  }
  .register-value {
    font-family: ui-monospace, monospace; font-size: 1.02rem;
    font-weight: 650; color: var(--ink); font-variant-numeric: tabular-nums;
  }
  .register-note { font-size: .74rem; color: var(--muted); line-height: 1.35; }
</style>
""", unsafe_allow_html=True)

STATE_CLASS = {"HIGH": "state-high", "MEDIUM": "state-medium", "LOW": "state-low"}


def rung(number: str, question: str, owner: str = "BLL") -> None:
    """Section header marking one rung of Figure 7's question sequence."""
    st.markdown(
        f'<div class="rung"><span class="rung-no">{number}</span>'
        f'<span class="rung-q">{question}</span>'
        f'<span class="rung-owner">{owner}</span></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

MODEL_DIRS = [ROOT / "models", ROOT / "deploy", ROOT, Path.cwd()]
DATA_DIRS = [ROOT / "data" / "processed", ROOT / "deploy", ROOT, Path.cwd()]


def _find(patterns: list[str], dirs: list[Path]) -> Path | None:
    for directory in dirs:
        if not directory.exists():
            continue
        for pattern in patterns:
            hits = sorted(directory.glob(pattern))
            if hits:
                return hits[0]
    return None


@st.cache_resource(show_spinner="Loading model…")
def load_service(refund, churn, support, action, currency):
    model_path = _find(["model_post_dispatch.joblib", "model_*.joblib"], MODEL_DIRS)
    if model_path is None:
        return None
    return PredictionService.from_files(
        model_path,
        _find(["orders_features.parquet", "sample_orders.parquet"], DATA_DIRS),
        _find(["background.parquet"], DATA_DIRS),
        costs=BusinessCostModel(currency=currency, refund_share=refund,
                                churn_share=churn, support_cost=support,
                                intervention_cost=action),
    )


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<div class="eyebrow">Business Language Layer</div>', unsafe_allow_html=True)
    st.markdown("### Delivery Risk")
    st.caption("Predict the risk. Explain it. Reduce it. Price the decision.")

    st.divider()
    st.markdown("**Cost assumptions**")
    st.caption(
        "Assumptions, not measured Olist finances. Every figure on this page "
        "moves with them, so they are set here rather than hidden in code."
    )
    currency = st.text_input("Currency", "BRL")
    refund = st.slider("Refund exposure", 0.0, 1.0, 0.42, 0.01,
                       help="Freight refund plus goodwill credit, as a share of order value.")
    churn = st.slider("Repeat-purchase risk", 0.0, 1.0, 0.46, 0.01,
                      help="Future revenue at risk from the review-score drop that follows a late delivery.")
    support = st.number_input("Support cost per contact", 0.0, 500.0, 12.0)
    action = st.number_input("Cost of intervening", 0.0, 500.0, 4.0)

service = load_service(refund, churn, support, action, currency)

if service is None:
    st.markdown('<div class="eyebrow">No model loaded</div>', unsafe_allow_html=True)
    st.markdown("### Train a model to begin")
    st.markdown(
        "This screen reads a trained artefact from `models/` or `deploy/`. "
        "Build one with:"
    )
    st.code("python -m olist.validate --build\n"
            "python -m olist.train --scenario post_dispatch", language="bash")
    st.stop()

history = service.history
if not len(history):
    st.error("No historical orders found. Run `python -m olist.validate --build` first.")
    st.stop()

if "risk" not in st.session_state:
    with st.spinner("Scoring orders…"):
        st.session_state["risk"] = service.predict_probability(history)
risk = st.session_state["risk"]

with st.sidebar:
    st.divider()
    st.markdown("**Choose an order**")
    mode = st.radio("Selection", ["Highest risk", "Typical high-risk", "From the queue"],
                    label_visibility="collapsed")

    if mode == "Highest risk":
        pos = int(np.argmax(risk))
    elif mode == "Typical high-risk":
        # The single most extreme order invites "so it is always certain?".
        # The 90th percentile is closer to what an operations lead actually sees.
        target = np.percentile(risk, 90)
        pos = int(np.abs(risk - target).argmin())
    else:
        top = np.argsort(risk)[::-1][:150]
        pos = st.selectbox(
            "Order", list(top), label_visibility="collapsed",
            format_func=lambda i: f"{history['order_id'].iloc[i][:14]} · {risk[i]:.0%}",
        )

    st.divider()
    st.markdown("**Model**")
    for key, value in list(service.model_info().items())[:6]:
        st.caption(f"{key}: **{value}**")

# --------------------------------------------------------------------------
# Assessment
# --------------------------------------------------------------------------

row = history.iloc[[pos]]
order_id = history["order_id"].iloc[pos]
order_value = float(history["total_price"].iloc[pos])

with st.spinner("Reading the model's explanation…"):
    a = service.assess(row, order_value=order_value, order_id=order_id)

brief, costs = a.brief, a.brief.costs
state_class = STATE_CLASS[a.risk_level]

# --- 1. INTERPRET ---------------------------------------------------------

st.markdown('<div class="eyebrow">Delivery risk assessment</div>', unsafe_allow_html=True)
st.markdown(f"## Order `{order_id}`")

route = f"{history.get('seller_state', pd.Series(['—'])).iloc[pos]} → {history['customer_state'].iloc[pos]}"
st.markdown(
    f'<span class="caption">{route} &nbsp;·&nbsp; {history["product_category"].iloc[pos]} '
    f'&nbsp;·&nbsp; {currency} {order_value:,.2f} &nbsp;·&nbsp; promised '
    f'{history["order_estimated_delivery_date"].iloc[pos]:%d %b %Y}</span>',
    unsafe_allow_html=True)

st.write("")
c1, c2, c3 = st.columns([1, 1, 1.6])

with c1:
    st.markdown('<div class="eyebrow">Late-delivery probability</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="figure">{a.probability:.0%}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="state {state_class}">{a.risk_level} RISK</div>', unsafe_allow_html=True)
    st.progress(float(a.probability))

with c2:
    st.markdown('<div class="eyebrow">Model confidence</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="figure">{a.confidence.score:.0%}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="state">{a.confidence.level} · estimate</div>', unsafe_allow_html=True)
    st.progress(float(a.confidence.score))

with c3:
    st.markdown('<div class="eyebrow">Recommended action</div>', unsafe_allow_html=True)
    st.markdown(f"### {brief.recommendation.name}")
    st.markdown(f'<span class="caption">{brief.recommendation.description}</span>',
                unsafe_allow_html=True)

if a.probability >= service.threshold:
    st.warning(f"Above the {service.threshold:.0%} action threshold — likely to arrive "
               f"after the promised date. Intervention recommended.")
else:
    st.info(f"Below the {service.threshold:.0%} action threshold — on track against the promise.")

if a.confidence.is_low:
    with st.expander("Why is this prediction uncertain?", expanded=True):
        for reason in a.confidence.reasons:
            st.markdown(f"- {reason}")
        st.markdown(
            f'<span class="caption">Components — threshold margin '
            f'{a.confidence.threshold_margin:.0%}, input familiarity '
            f'{a.confidence.familiarity:.0%}, evidence density '
            f'{a.confidence.evidence_density:.0%}.</span>', unsafe_allow_html=True)

for warning in a.warnings:
    st.caption(f"⚠️ {warning}")

# --- 2. CONTEXTUALISE -----------------------------------------------------

st.divider()
rung("01", "Why did the model predict this?", "SHAP")
st.markdown("### Key contributing factors")
st.markdown('<span class="caption">The model\'s main reasons, in plain terms.</span>',
            unsafe_allow_html=True)
st.write("")

for d in brief.drivers:
    direction = "up" if d.raises_risk else "down"
    tags = f'<span class="tag">{d.badge}</span>'
    if d is brief.drivers[0]:
        tags += '<span class="tag">Biggest driver</span>'
    if d.is_lever:
        tags += '<span class="tag">Actionable</span>'
    st.markdown(
        f'<div class="driver driver-{direction}">'
        f'<div class="driver-name">{d.label}</div>'
        f'<div class="driver-text">{d.sentence}</div>{tags}</div>',
        unsafe_allow_html=True)

st.write("")
for line in brief.consequences:
    st.markdown(f"- {line}")

# --- 3. QUANTIFY ----------------------------------------------------------

st.divider()
rung("02", "What does it mean for the business?")
st.markdown("### Business impact if late")

impact_cols = st.columns(len(brief.impact))
for col, (name, amount) in zip(impact_cols, brief.impact.items()):
    with col:
        st.markdown(f'<div class="eyebrow">{name}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="figure-sm">{costs.format(amount)}</div>', unsafe_allow_html=True)

st.markdown(
    f'<span class="caption">Three components sum to '
    f'<b>{costs.format(brief.exposure)}</b> — the value at stake on this order.</span>',
    unsafe_allow_html=True)

st.write("")
expected_loss = a.probability * brief.exposure
q1, q2, q3 = st.columns(3)
q1.metric("Loss if we ship as-is", costs.format(expected_loss))
q2.metric("Cost to act", costs.format(brief.recommendation.action_cost))
q3.metric("Net saving", costs.format(brief.net_saving),
          delta=f"{brief.net_saving:+,.0f}")

# The sentence that does the translation work: every figure above, assembled
# into something a non-technical reader takes in at a glance.
best_alternative = min(
    (o for o in brief.options if o.feasible and not o.recommended),
    key=lambda o: o.total_cost, default=None,
)
if brief.net_saving > 0 and best_alternative is not None:
    plain = (
        f"In plain terms: there is a <b>{a.probability:.0%}</b> chance this "
        f"{costs.format(order_value)} order is lost to a late delivery — on average "
        f"<b>{costs.format(expected_loss)}</b>. "
        f"{brief.recommendation.name.lower().capitalize()} costs "
        f"{costs.format(brief.recommendation.action_cost)} and lowers the risk to "
        f"<b>{brief.recommendation.risk:.0%}</b>, so acting now saves about "
        f"<b>{costs.format(brief.net_saving)}</b>."
    )
else:
    plain = (
        f"In plain terms: there is a <b>{a.probability:.0%}</b> chance this "
        f"{costs.format(order_value)} order arrives late, putting about "
        f"<b>{costs.format(expected_loss)}</b> at risk. No available intervention "
        f"costs less than it would save, so standard fulfilment is the better call."
    )
st.markdown(f'<div class="plain-terms">{plain}</div>', unsafe_allow_html=True)

# --- 4. RECOMMEND ---------------------------------------------------------

st.divider()
rung("03", "What outcome should we expect?")
st.markdown("### Comparative reasoning — your options")

st.dataframe(
    pd.DataFrame([{
        "Option": o.name,
        "Risk": f"{o.risk:.0%}" if o.feasible else "unavailable",
        "Cost": costs.format(o.action_cost) if o.feasible else "—",
        "Expected loss": costs.format(o.expected_loss) if o.feasible else "—",
        "": "◄ recommended" if o.recommended else "",
    } for o in brief.options]),
    hide_index=True, width="stretch",
)
st.markdown(
    '<span class="caption">Each option is priced by re-running the model with that '
    'single input changed. These are model-estimated risk changes under a '
    'ceteris-paribus assumption — not causal guarantees — and they do not compose.</span>',
    unsafe_allow_html=True)

# --- Rung 04: what do we do, and what does it cost? -----------------------

st.divider()
rung("04", "What do we do, and what does it cost?")
st.markdown("### Evidence and simulation")

tab_evidence, tab_whatif, tab_target, tab_whynot, tab_translation, tab_audit = st.tabs(
    ["Similar orders", "What if…", "What would it take?", "Why not lower?",
     "Before / after", "Audit trail"]
)

with tab_evidence:
    similar = a.similar
    if similar.n_found and similar.sufficient:
        e1, e2, e3 = st.columns(3)
        e1.metric("Mean similarity", f"{similar.mean_similarity:.0%}")
        e2.metric("Late among similar", f"{similar.late_rate:.0%}")
        e3.metric("Mean delay when late", f"{similar.mean_delay_days:.1f} days")
        st.dataframe(
            pd.DataFrame({
                "Order": similar.orders["order_id"],
                "Similarity": similar.orders["similarity"].map("{:.0%}".format),
                "Outcome": np.where(similar.orders["is_late_delivery"] == 1, "Late", "On time"),
                "Delay (days)": similar.orders["delivery_delay_days"].round(1),
            }),
            hide_index=True, width="stretch",
        )
        st.markdown(
            '<span class="caption">Independent evidence: these orders were matched only '
            'on features knowable before delivery. Matching on the outcome would make the '
            'late rate circular.</span>', unsafe_allow_html=True)
    else:
        st.info(similar.note or "No sufficiently similar historical orders were found.")

with tab_whatif:
    levers = [d.feature for d in brief.drivers
              if d.is_lever and d.feature in service.feature_columns]
    levers = levers or [f for f in ("carrier_pickup_days", "estimated_delivery_days")
                        if f in service.feature_columns]
    if levers:
        lever = st.selectbox("Change", levers,
                             format_func=lambda f: glossary.GLOSSARY[f].label)
        current = float(row[lever].iloc[0])
        new = st.slider(f"{glossary.GLOSSARY[lever].label} — currently {current:.1f}",
                        0.0, float(max(current * 3, 10.0)), current, 0.5)
        new_risk = service.run_counterfactual(row, lever, new)
        w1, w2, w3 = st.columns(3)
        w1.metric("Risk now", f"{a.probability:.0%}")
        w2.metric("Risk after", f"{new_risk:.0%}", f"{new_risk - a.probability:+.0%}")
        w3.metric("Expected loss change",
                  f"{(new_risk - a.probability) * brief.exposure:+,.2f}")

    unavailable = {k: v for k, v in service.available_levers().items() if not v.available}
    if unavailable:
        st.markdown("**Levers this dataset cannot support**")
        for name, capability in unavailable.items():
            st.markdown(f'<span class="caption">· <b>{name}</b> — {capability.reason}</span>',
                        unsafe_allow_html=True)
        st.markdown(
            '<span class="caption">Disabled rather than simulated: offering them would '
            'mean inventing both the control and the resulting figure.</span>',
            unsafe_allow_html=True)

with tab_target:
    st.markdown("**What would it take to get this order into a lower band?**")
    st.markdown(
        '<span class="caption">The reverse of a normal prediction: rather than deriving '
        'an output from given inputs, this searches for the input value that would '
        'produce a desired output.</span>', unsafe_allow_html=True)
    st.write("")

    target_levers = [f for f in service.feature_columns if glossary.is_lever(f)]
    if target_levers:
        t1, t2 = st.columns([2, 1])
        target_lever = t1.selectbox(
            "Lever", target_levers, key="target_lever",
            format_func=lambda f: glossary.GLOSSARY[f].label)
        target_band = t2.selectbox("Target band", ["LOW (under 20%)", "MEDIUM (under 50%)"])
        target_risk = 0.20 if target_band.startswith("LOW") else 0.50

        with st.spinner("Searching…"):
            found = service.find_value_for_target(row, target_lever, target_risk)

        label = glossary.GLOSSARY[target_lever].label
        if found.already_satisfied:
            st.info(found.reason)
        elif found.achievable:
            r1, r2, r3 = st.columns(3)
            r1.metric(f"{label} now", f"{found.current_value:.1f}")
            r2.metric("Would need to be", f"{found.required_value:.1f}",
                      f"{found.required_value - found.current_value:+.1f}")
            r3.metric("Resulting risk", f"{found.achieved_risk:.0%}",
                      f"{found.achieved_risk - a.probability:+.0%}")
            st.success(
                f"Bringing **{label.lower()}** to **{found.required_value:.1f}** would put "
                f"this order in the {target_band.split()[0]} band."
            )
        else:
            st.warning(found.reason)
            if found.best_risk is not None:
                st.markdown(
                    f'<span class="caption">Best achievable with this lever alone: '
                    f'{found.best_risk:.0%} at a value of {found.best_value:.1f}.</span>',
                    unsafe_allow_html=True)
        st.markdown(
            '<span class="caption">Found by bisection over the lever, evaluating the '
            'deployed pipeline at each step, so the returned value is verifiable. It '
            'assumes risk moves monotonically with the lever across the search range.'
            '</span>', unsafe_allow_html=True)
    else:
        st.info("This model has no adjustable levers.")

with tab_whynot:
    st.markdown("**Why is this order not in the low-risk band?**")
    st.markdown(
        '<span class="caption">Not a restatement of the drivers: this names what would '
        'each have to move, and by how much.</span>', unsafe_allow_html=True)
    st.write("")
    with st.spinner("Working out what would have to change…"):
        for line in service.explain_why_not(row, "LOW"):
            st.markdown(f"- {line}")

with tab_translation:
    st.markdown("**The same evidence, two registers.**")
    st.markdown(
        '<span class="caption">Left: what the explainer returns. Right: what the layer '
        'writes from it. Nothing was re-modelled — only the register, the resolution '
        'and the intended reader differ.</span>', unsafe_allow_html=True)
    st.write("")

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="eyebrow">Before · SHAP local explanation</div>',
                    unsafe_allow_html=True)
        contributions = a.explanation.top_drivers(6)
        lines = [f"order {str(order_id)[:16]}",
                 f"model {service.artefact.get('model_name')} · target late_delivery",
                 f"p(late) = {a.probability:.3f}", ""]
        width = max(len(f) for f in contributions.index)
        for feature, value in contributions.items():
            lines.append(f"{feature:<{width}}  {value:+.4f}")
        lines += ["", f"base value E[f(x)] = {a.explanation.base_value:.3f}"]
        st.markdown(f'<div class="silent">{chr(10).join(lines)}</div>', unsafe_allow_html=True)
        st.markdown(
            '<span class="caption"><b>Technically complete. Operationally silent.</b> '
            'Every number is correct. None tells the operations lead whether to ship '
            'the order.</span>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="eyebrow">After · Business Language Layer</div>',
                    unsafe_allow_html=True)
        st.markdown(f"#### {brief.headline()}")
        st.markdown(f'<div class="figure-sm">{costs.format(brief.net_saving)}</div>'
                    f'<span class="caption">expected loss avoided against shipping as-is</span>',
                    unsafe_allow_html=True)
        st.write("")
        for d in brief.drivers[:3]:
            st.markdown(f'<span class="caption">· {d.sentence}</span>', unsafe_allow_html=True)
        st.write("")
        st.markdown(
            f'<span class="caption">Expected loss = p(late) × order value + intervention '
            f'cost. Every figure traces back to the model output.</span>',
            unsafe_allow_html=True)

with tab_audit:
    st.dataframe(brief.audit(), hide_index=True, width="stretch")
    st.markdown(
        f'<span class="caption">Every statement above traces to the feature and '
        f'attribution behind it. Glossary coverage of shown drivers: '
        f'{brief.glossary_coverage:.0%}. A feature with no business mapping is flagged '
        f'rather than given a generic phrase.</span>', unsafe_allow_html=True)

# --- One order, five registers (Figure 4) ---------------------------------

st.divider()
st.markdown('<div class="eyebrow">One order, five registers</div>', unsafe_allow_html=True)
st.markdown(
    '<span class="caption">The same order described five ways. Meaning is preserved '
    'while the language shifts from machine to business.</span>', unsafe_allow_html=True)
st.write("")

top_driver = brief.drivers[0]
registers = [
    ("Technical · model", f"{a.probability:.2f}", "predict_proba(x) — probability of late delivery"),
    ("XAI · mathematical", f"{top_driver.shap_value:+.3f}", f"SHAP({top_driver.feature}) — top driver"),
    ("Business · impact", costs.format(brief.exposure), f"value at risk · {len(brief.options)} options ranked"),
    ("Human · action", brief.recommendation.name, f"saves {costs.format(brief.net_saving)}"),
    ("Outcome · result", "Decision on record", "audit trail retained · promise defended"),
]
for col, (label, value, note) in zip(st.columns(5), registers):
    with col:
        st.markdown(
            f'<div class="register"><div class="register-label">{label}</div>'
            f'<div class="register-value">{value}</div>'
            f'<div class="register-note">{note}</div></div>', unsafe_allow_html=True)

# --- Export ---------------------------------------------------------------

st.divider()
if st.button("Generate decision brief (PDF)", type="primary"):
    from olist.bll import build_report

    context = {
        "Order ID": order_id,
        "Route": route,
        "Product category": history["product_category"].iloc[pos],
        "Purchased": f"{history['order_purchase_timestamp'].iloc[pos]:%Y-%m-%d}",
        "Promised delivery": f"{history['order_estimated_delivery_date'].iloc[pos]:%Y-%m-%d}",
        "Order value": f"{currency} {order_value:,.2f}",
    }
    with st.spinner("Building the brief…"):
        path = Path(tempfile.gettempdir()) / f"decision_brief_{str(order_id)[:12]}.pdf"
        build_report(a, path, order_context=context, model_info=service.model_info())
    st.download_button("Download", path.read_bytes(), file_name=path.name,
                       mime="application/pdf")

st.write("")
for caveat in brief.caveats:
    st.markdown(f'<span class="caption">⚠️ {caveat}</span>', unsafe_allow_html=True)
