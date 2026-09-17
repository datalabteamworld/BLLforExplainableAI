"""PDF decision brief.

Renders an :class:`~olist.bll.service.Assessment` as a shareable document —
the artefact an operations lead forwards to a manager, or files as the record
of why an order was escalated.

Every number comes from the assessment object. Sections whose evidence is
missing are omitted with a stated reason rather than filled with plausible
placeholders.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)

RISK_COLOURS = {
    "HIGH": colors.HexColor("#C0392B"),
    "MEDIUM": colors.HexColor("#D68910"),
    "LOW": colors.HexColor("#1E8449"),
}
INK = colors.HexColor("#1C2833")
MUTED = colors.HexColor("#5D6D7E")
RULE = colors.HexColor("#D5DBDB")


def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], fontSize=19,
                                textColor=INK, spaceAfter=2, alignment=TA_LEFT),
        "sub": ParagraphStyle("s", parent=base["Normal"], fontSize=9.5,
                              textColor=MUTED, spaceAfter=10),
        "h2": ParagraphStyle("h", parent=base["Heading2"], fontSize=12.5,
                             textColor=INK, spaceBefore=14, spaceAfter=5),
        "body": ParagraphStyle("b", parent=base["Normal"], fontSize=9.5,
                               textColor=INK, leading=13.5, spaceAfter=4),
        "small": ParagraphStyle("sm", parent=base["Normal"], fontSize=8,
                                textColor=MUTED, leading=11),
        "big": ParagraphStyle("bg", parent=base["Normal"], fontSize=30,
                              textColor=INK, leading=34),
    }


def _table(data: list[list], widths: list[float], header: bool = True,
           align_right: list[int] | None = None) -> Table:
    style = [
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F4F6F7")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, MUTED),
        ]
    for col in align_right or []:
        style.append(("ALIGN", (col, 0), (col, -1), "RIGHT"))

    table = Table(data, colWidths=widths, hAlign="LEFT")
    table.setStyle(TableStyle(style))
    return table


def build_report(assessment, path: Path, order_context: dict | None = None,
                 model_info: dict | None = None) -> Path:
    """Write the decision brief for one order to ``path``."""
    styles = _styles()
    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title=f"Delivery risk assessment — order {assessment.order_id}",
    )
    brief, costs = assessment.brief, assessment.brief.costs
    story = []

    # --- header -----------------------------------------------------------
    story.append(Paragraph("Delivery Risk Assessment", styles["title"]))
    story.append(Paragraph(
        f"Order {assessment.order_id} &nbsp;·&nbsp; generated "
        f"{pd.Timestamp.now():%Y-%m-%d %H:%M}", styles["sub"]))
    story.append(HRFlowable(width="100%", thickness=1, color=RULE, spaceAfter=10))

    # --- 1. summary -------------------------------------------------------
    risk_colour = RISK_COLOURS.get(assessment.risk_level, INK)
    summary = [[
        Paragraph("Late-delivery probability", styles["small"]),
        Paragraph("Model confidence", styles["small"]),
        Paragraph("Recommended action", styles["small"]),
    ], [
        Paragraph(f'<font color="{risk_colour}">{assessment.probability:.0%}</font>',
                  styles["big"]),
        Paragraph(f"{assessment.confidence.score:.0%}", styles["big"]),
        Paragraph(f"<b>{brief.recommendation.name}</b>", styles["body"]),
    ], [
        Paragraph(f'<font color="{risk_colour}"><b>{assessment.risk_level} RISK</b></font>',
                  styles["body"]),
        Paragraph(f"<b>{assessment.confidence.level} CONFIDENCE</b> (estimate)",
                  styles["body"]),
        Paragraph(brief.recommendation.description, styles["small"]),
    ]]
    story.append(_table(summary, [55 * mm, 55 * mm, 64 * mm], header=False))
    story.append(Spacer(1, 8))

    if assessment.confidence.is_low:
        story.append(Paragraph(
            "<b>Low confidence.</b> The model is uncertain about this order. "
            "Review the contributing factors before taking operational action.",
            styles["body"]))
        for reason in assessment.confidence.reasons:
            story.append(Paragraph(f"• {reason}", styles["small"]))
        story.append(Spacer(1, 6))

    # --- 2. order context -------------------------------------------------
    if order_context:
        story.append(Paragraph("Order summary", styles["h2"]))
        rows = [[Paragraph(f"<b>{k}</b>", styles["body"]),
                 Paragraph(str(v), styles["body"])] for k, v in order_context.items()]
        story.append(_table(rows, [55 * mm, 119 * mm], header=False))

    # --- 3. why this prediction ------------------------------------------
    story.append(Paragraph("Why this prediction?", styles["h2"]))
    story.append(Paragraph(
        "The model's main reasons, in plain terms. Contributions are attributions to "
        "the model's internal risk score (log-odds), so they rank the drivers against "
        "one another; they are not percentage points of the probability shown above.",
        styles["small"]))
    story.append(Spacer(1, 4))

    driver_rows = [["Factor", "Effect", "Contribution"]]
    for d in brief.drivers:
        driver_rows.append([
            Paragraph(f"<b>{d.label}</b><br/><font size=7.5 color='#5D6D7E'>{d.sentence}</font>",
                      styles["small"]),
            "Adds risk" if d.raises_risk else "Lowers risk",
            f"{d.shap_value:+.3f}",
        ])
    story.append(_table(driver_rows, [110 * mm, 32 * mm, 32 * mm], align_right=[2]))

    # --- 4. similar orders -------------------------------------------------
    story.append(Paragraph("Similar historical orders", styles["h2"]))
    similar = assessment.similar
    if similar.n_found and similar.sufficient:
        story.append(Paragraph(
            f"Of the {similar.n_found} most similar past orders "
            f"(mean similarity {similar.mean_similarity:.0%}), "
            f"<b>{similar.late_rate:.0%} arrived late</b>"
            + (f", by {similar.mean_delay_days:.1f} days on average."
               if similar.mean_delay_days else "."),
            styles["body"]))
        rows = [["Order", "Similarity", "Outcome", "Delay (days)"]]
        for _, r in similar.orders.head(6).iterrows():
            late = r.get("is_late_delivery", np.nan)
            delay = r.get("delivery_delay_days", np.nan)
            rows.append([
                str(r.get("order_id", "—"))[:14],
                f"{r['similarity']:.0%}",
                "Late" if late == 1 else "On time",
                f"{delay:.1f}" if pd.notna(delay) and late == 1 else "—",
            ])
        story.append(_table(rows, [58 * mm, 30 * mm, 32 * mm, 32 * mm], align_right=[1, 3]))
    else:
        story.append(Paragraph(
            similar.note or "No sufficiently similar historical orders were found.",
            styles["body"]))

    story.append(PageBreak())

    # --- 5. options and what-if -------------------------------------------
    story.append(Paragraph("Options and expected cost", styles["h2"]))
    option_rows = [["Option", "Risk", "Action cost", "Expected loss", ""]]
    for o in brief.options:
        if not o.feasible:
            option_rows.append([o.name, "unavailable", "—", "—", ""])
            continue
        option_rows.append([
            o.name, f"{o.risk:.0%}", costs.format(o.action_cost),
            costs.format(o.expected_loss), "recommended" if o.recommended else "",
        ])
    story.append(_table(option_rows, [50 * mm, 24 * mm, 30 * mm, 34 * mm, 36 * mm],
                        align_right=[1, 2, 3]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"<b>{brief.headline()}</b> Net saving against shipping as-is: "
        f"<b>{costs.format(brief.net_saving)}</b>.", styles["body"]))
    story.append(Paragraph(
        "Each option is evaluated independently by re-running the model with that "
        "single input changed. The figures are model-estimated risk changes under a "
        "ceteris-paribus assumption, not causal guarantees, and they do not compose.",
        styles["small"]))

    # --- 6. business impact -----------------------------------------------
    story.append(Paragraph("Business impact if late", styles["h2"]))
    impact_rows = [["Component", "Amount"]]
    for name, amount in brief.impact.items():
        impact_rows.append([name, costs.format(amount)])
    impact_rows.append(["Total exposure", costs.format(brief.exposure)])
    impact_rows.append(["Expected loss (risk x exposure)",
                        costs.format(assessment.probability * brief.exposure)])
    story.append(_table(impact_rows, [110 * mm, 64 * mm], align_right=[1]))

    # --- 7. model information ---------------------------------------------
    if model_info:
        story.append(Paragraph("Model information", styles["h2"]))
        rows = [[Paragraph(f"<b>{k}</b>", styles["small"]),
                 Paragraph(str(v), styles["small"])] for k, v in model_info.items()]
        story.append(_table(rows, [70 * mm, 104 * mm], header=False))

    # --- 8. caveats --------------------------------------------------------
    story.append(Paragraph("Limitations", styles["h2"]))
    for caveat in brief.caveats + assessment.warnings:
        story.append(Paragraph(f"• {caveat}", styles["small"]))

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.6, color=RULE, spaceAfter=6))
    story.append(Paragraph(
        "AI-generated prediction for decision support. Predictions are uncertain and "
        "should be reviewed alongside operational context. Monetary figures rest on "
        "configured cost assumptions, not measured finances.", styles["small"]))

    doc.build(story)
    logger.info("Wrote report to %s", path)
    return path
