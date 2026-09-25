import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, KeepTogether,
)

BRAND_NAME = "Heart_Byte"
BRAND_COLOR = colors.HexColor("#0E7C61")

GREEN = colors.HexColor("#1E8449")
GREEN_BG = colors.HexColor("#EAFAF1")
AMBER = colors.HexColor("#B7950B")
AMBER_BG = colors.HexColor("#FEF9E7")
RED = colors.HexColor("#C0392B")
RED_BG = colors.HexColor("#FDEDEC")
GREY = colors.HexColor("#888888")
LINE_GREY = colors.HexColor("#DDDDDD")

RISK_INFO = {
    "Normal Indications": ("No Acoustic Anomalies Flagged", GREEN, GREEN_BG),
    "Acoustic Anomaly": ("Acoustic Pattern Flagged for Review", AMBER, AMBER_BG),
    "Abnormal Rate": ("Heart Rate Pattern Flagged for Review", RED, RED_BG),
    "Irregular Rhythm": ("Rhythm Pattern Flagged for Review", RED, RED_BG),
}

FLAG_STYLE = {
    "ok": ("PASS", GREEN),
    "warning": ("REVIEW", AMBER),
}


def _get_styles():
    """Shared ParagraphStyle set used by both the single and comparison reports."""
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleStyle", parent=styles["Title"], fontSize=17, textColor=colors.white,
            alignment=TA_LEFT, leading=20,
        ),
        "brand": ParagraphStyle(
            "Brand", parent=styles["Normal"], fontSize=10, textColor=colors.white,
            alignment=TA_LEFT, spaceBefore=2,
        ),
        "meta": ParagraphStyle(
            "Meta", parent=styles["Normal"], fontSize=8.5, textColor=colors.white,
            alignment=TA_LEFT, leading=13,
        ),
        "section": ParagraphStyle(
            "Section", parent=styles["Heading2"], fontSize=11.5, textColor=BRAND_COLOR,
            spaceBefore=28, spaceAfter=12, textTransform="uppercase",
        ),
        "rec_label": ParagraphStyle(
            "RecLabel", parent=styles["Normal"], fontSize=8.5, textColor=GREY, fontName="Helvetica-Bold",
        ),
        "rec_value": ParagraphStyle(
            "RecValue", parent=styles["Normal"], fontSize=8.5, textColor=colors.HexColor("#1a1a1a"),
            wordWrap="CJK",
        ),
        "risk_icon": ParagraphStyle(
            "RiskIcon", parent=styles["Normal"], fontSize=22, textColor=colors.white,
            fontName="Helvetica-Bold", alignment=TA_CENTER, leading=24,
        ),
        "risk_icon_word": ParagraphStyle(
            "RiskIconWord", parent=styles["Normal"], fontSize=8, textColor=colors.white,
            fontName="Helvetica-Bold", alignment=TA_CENTER, spaceBefore=3,
        ),
        "risk_heading": ParagraphStyle(
            "RiskHeading", parent=styles["Normal"], fontSize=12.5, fontName="Helvetica-Bold",
            textColor=colors.HexColor("#1a1a1a"),
        ),
        "risk_desc": ParagraphStyle(
            "RiskDesc", parent=styles["Normal"], fontSize=8.5, textColor=colors.HexColor("#444444"),
            spaceBefore=3, leading=12,
        ),
        "finding": ParagraphStyle(
            "Finding", parent=styles["BodyText"], fontSize=10.5, leftIndent=10, spaceAfter=10, leading=15,
        ),
        "disclaimer": ParagraphStyle(
            "Disclaimer", parent=styles["BodyText"], fontSize=7.5, textColor=GREY,
            alignment=TA_CENTER,
        ),
        "caption": ParagraphStyle(
            "Caption", parent=styles["Normal"], fontSize=8, textColor=GREY,
            alignment=TA_CENTER, spaceBefore=10, leading=13,
        ),
        "table_header": ParagraphStyle(
            "TableHeader", parent=styles["Normal"], fontSize=8.5, textColor=colors.white,
            fontName="Helvetica-Bold", alignment=TA_CENTER,
        ),
        "table_cell": ParagraphStyle(
            "TableCell", parent=styles["Normal"], fontSize=8.5, textColor=colors.HexColor("#1a1a1a"),
            alignment=TA_CENTER, wordWrap="CJK",
        ),
    }


def _build_header_bar(styles, subtitle):
    header_data = [[
        Paragraph(f"{BRAND_NAME}", styles["title"]),
        Paragraph(
            f"Report ID: {datetime.now().strftime('HB-%Y%m%d-%H%M%S')}<br/>"
            f"{datetime.now().strftime('%d %B %Y, %I:%M %p')}",
            styles["meta"],
        ),
    ]]
    header_table = Table(header_data, colWidths=[11 * cm, 5.3 * cm])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BRAND_COLOR),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LEFTPADDING", (0, 0), (0, 0), 14),
        ("RIGHTPADDING", (1, 0), (1, 0), 14),
    ]))
    return [header_table, Paragraph(subtitle, styles["brand"]), Spacer(1, 26)]


def _render_signal_chart(time, signal, filtered_signal, envelope, peaks, peak_times, sr=None, s1_indices=None, s2_indices=None):
    """Renders the raw/filtered waveform and envelope/peak plots as a print-friendly PNG."""
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.1), sharex=True)

    axes[0].plot(time, signal, color="#9aa0a6", linewidth=0.6, alpha=0.6, label="Raw Audio (PCG)")
    axes[0].plot(time, filtered_signal, color="#d68910", linewidth=0.8, label="Filtered Signal")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Raw vs. Filtered Acoustic Signal", fontsize=10, fontweight="bold")
    axes[0].legend(loc="upper right", fontsize=7)
    axes[0].grid(alpha=0.25)

    axes[1].plot(time, envelope, color="#1a5276", linewidth=1.0, label="Energy Envelope")
    if sr and s1_indices is not None and s2_indices is not None:
        axes[1].scatter(s1_indices / sr, envelope[s1_indices], color="#d68910", marker="^", s=35, label="S1 (lub)", zorder=5)
        axes[1].scatter(s2_indices / sr, envelope[s2_indices], color="#8e44ad", marker="v", s=35, label="S2 (dub)", zorder=5)
    else:
        axes[1].scatter(peak_times, envelope[peaks], color="#c0392b", marker="x", s=35, label="Detected Beats", zorder=5)
    axes[1].set_ylabel("Envelope")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_title("Envelope Extraction & Heartbeat Peak Detection", fontsize=10, fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=7)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170)
    plt.close(fig)
    buf.seek(0)
    return buf


def _render_comparison_signal_chart(sample_a, sample_b):
    """Two-column version of _render_signal_chart: Sample A and Sample B side by side."""
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.4), sharex="col")

    for col, sample, label in [(0, sample_a, "Sample A"), (1, sample_b, "Sample B")]:
        axes[0, col].plot(sample["time"], sample["signal"], color="#9aa0a6", linewidth=0.5, alpha=0.6, label="Raw")
        axes[0, col].plot(sample["time"], sample["filtered_signal"], color="#d68910", linewidth=0.7, label="Filtered")
        axes[0, col].set_title(f"{label}: Raw vs. Filtered", fontsize=9.5, fontweight="bold")
        axes[0, col].legend(loc="upper right", fontsize=6.5)
        axes[0, col].grid(alpha=0.25)

        axes[1, col].plot(sample["time"], sample["envelope"], color="#1a5276", linewidth=0.9, label="Envelope")
        if sample.get("sample_rate") and sample.get("s1_indices") is not None and sample.get("s2_indices") is not None:
            s1, s2, sr = sample["s1_indices"], sample["s2_indices"], sample["sample_rate"]
            axes[1, col].scatter(s1 / sr, sample["envelope"][s1], color="#d68910", marker="^", s=24, label="S1", zorder=5)
            axes[1, col].scatter(s2 / sr, sample["envelope"][s2], color="#8e44ad", marker="v", s=24, label="S2", zorder=5)
        else:
            axes[1, col].scatter(
                sample["peak_times"], sample["envelope"][sample["peaks"]],
                color="#c0392b", marker="x", s=28, label="Beats", zorder=5,
            )
        axes[1, col].set_title(f"{label}: Envelope & Peaks", fontsize=9.5, fontweight="bold")
        axes[1, col].set_xlabel("Time (seconds)")
        axes[1, col].legend(loc="upper right", fontsize=6.5)
        axes[1, col].grid(alpha=0.25)

    axes[0, 0].set_ylabel("Amplitude")
    axes[1, 0].set_ylabel("Envelope")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170)
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_pdf_report(
    *,
    dataset_name,
    file_name,
    sample_rate,
    duration_sec,
    lowcut,
    highcut,
    bpm,
    hrv,
    num_beats,
    interval_stats,
    status,
    findings,
    checklist,
    disclaimer,
    time,
    signal,
    filtered_signal,
    envelope,
    peaks,
    peak_times,
    s1_indices=None,
    s2_indices=None,
):
    """Builds a professional Heart_Byte cardiac acoustic report PDF and returns it as bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=1.4 * cm, bottomMargin=1.4 * cm,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
        title=f"{BRAND_NAME} - Cardiac Acoustic Report",
    )

    styles = _get_styles()
    elements = list(_build_header_bar(styles, "Cardiac Acoustic Analysis Report"))

    # ---- Recording information ----
    elements.append(Paragraph("Recording Information", styles["section"]))

    def _label(text):
        return Paragraph(text, styles["rec_label"])

    def _value(text):
        return Paragraph(str(text), styles["rec_value"])

    rec_table_data = [
        [_label("Dataset"), _value(dataset_name), _label("Sample Rate"), _value(f"{sample_rate} Hz")],
        [_label("File Name"), _value(file_name), _label("Duration Analyzed"), _value(f"{duration_sec:.1f} s")],
        [_label("Bandpass Filter"), _value(f"{lowcut}–{highcut} Hz"), _label("Beats Detected"), _value(num_beats)],
    ]
    rec_table = Table(rec_table_data, colWidths=[3.2 * cm, 5.3 * cm, 3.6 * cm, 4.2 * cm])
    rec_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7f9f9")),
        ("BOX", (0, 0), (-1, -1), 0.5, LINE_GREY),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, LINE_GREY),
    ]))
    elements.append(rec_table)
    elements.append(Spacer(1, 8))

    # ---- Clinical anomaly indicator ----
    elements.append(Paragraph("Clinical Anomaly Indicator", styles["section"]))
    risk_label, risk_color, risk_bg = RISK_INFO.get(status, ("Review Suggested", AMBER, AMBER_BG))
    is_normal = status == "Normal Indications"
    badge_char = "✓" if is_normal else "!"
    badge_word = "NORMAL" if is_normal else "REVIEW"

    flagged_categories = [item["category"] for item in checklist if item["flag"] == "warning"]
    flagged_text = (
        "All monitored categories are within expected bounds."
        if not flagged_categories
        else "Flagged categories: " + ", ".join(flagged_categories) + "."
    )

    badge_cell = Table(
        [[Paragraph(badge_char, styles["risk_icon"])], [Paragraph(badge_word, styles["risk_icon_word"])]],
        colWidths=[2.6 * cm],
    )
    badge_cell.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
    ]))

    text_cell = [
        Paragraph(risk_label, styles["risk_heading"]),
        Paragraph("Rule-based educational indicator — not a validated clinical risk score.", styles["risk_desc"]),
        Paragraph(flagged_text, styles["risk_desc"]),
    ]

    risk_table = Table([[badge_cell, text_cell]], colWidths=[3.0 * cm, 13.3 * cm])
    risk_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), risk_color),
        ("BACKGROUND", (1, 0), (1, -1), risk_bg),
        ("BOX", (0, 0), (-1, -1), 0.75, risk_color),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 18),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 18),
        ("LEFTPADDING", (1, 0), (1, -1), 14),
        ("RIGHTPADDING", (1, 0), (1, -1), 14),
    ]))
    elements.append(risk_table)
    elements.append(Spacer(1, 8))

    # ---- Key measurements ----
    elements.append(Paragraph("Key Measurements", styles["section"]))
    metrics_data = [
        ["Heart Rate", "Mean Interval", "HRV (SDNN-equiv.)", "Interval Range"],
        [
            f"{bpm:.1f} BPM",
            f"{interval_stats['mean_ms']:.0f} ms",
            f"{hrv:.1f} ms",
            f"{interval_stats['min_ms']:.0f}–{interval_stats['max_ms']:.0f} ms",
        ],
    ]
    metrics_table = Table(metrics_data, colWidths=[4.075 * cm] * 4)
    metrics_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, 0), 8.5),
        ("FONTSIZE", (0, 1), (-1, 1), 12.5),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_COLOR),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 13),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 13),
        ("BOX", (0, 0), (-1, -1), 0.5, LINE_GREY),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE_GREY),
    ]))
    elements.append(metrics_table)
    elements.append(Spacer(1, 8))

    # ---- Physiologist-style checklist ----
    elements.append(Paragraph("Acoustic & Rhythm Checklist", styles["section"]))
    checklist_rows = [["Category", "Measured Value", "Assessment", "Flag"]]
    row_colors = []
    for item in checklist:
        flag_text, flag_color = FLAG_STYLE.get(item["flag"], ("REVIEW", AMBER))
        checklist_rows.append([item["category"], item["value"], item["assessment"], flag_text])
        row_colors.append(flag_color)

    checklist_table = Table(checklist_rows, colWidths=[4.8 * cm, 3.8 * cm, 5.2 * cm, 2.5 * cm])
    checklist_style = [
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4d4d4d")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 11),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 11),
        ("BOX", (0, 0), (-1, -1), 0.5, LINE_GREY),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE_GREY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fbfbfb")]),
    ]
    for i, flag_color in enumerate(row_colors, start=1):
        checklist_style.append(("FONTNAME", (3, i), (3, i), "Helvetica-Bold"))
        checklist_style.append(("TEXTCOLOR", (3, i), (3, i), flag_color))
    checklist_table.setStyle(TableStyle(checklist_style))
    elements.append(checklist_table)

    # ---- Signal visualization ----
    chart_buf = _render_signal_chart(
        time, signal, filtered_signal, envelope, peaks, peak_times,
        sr=sample_rate, s1_indices=s1_indices, s2_indices=s2_indices,
    )
    elements.append(KeepTogether([
        Paragraph("Signal Visualization", styles["section"]),
        Image(chart_buf, width=17.2 * cm, height=11.5 * cm),
        Paragraph(
            "The waveform above is a synthesized ECG-style rendering time-aligned to detected acoustic "
            "beats, for visualization only — it is not derived from electrical cardiac activity.",
            styles["caption"],
        ),
    ]))

    # ---- Impression ----
    elements.append(Paragraph("Impression", styles["section"]))
    for f in findings:
        elements.append(Paragraph(f"•&nbsp;&nbsp;{f}", styles["finding"]))

    elements.append(Spacer(1, 30))
    elements.append(HRFlowable(width="100%", thickness=0.75, color=LINE_GREY, spaceAfter=14))
    elements.append(Paragraph(disclaimer, styles["disclaimer"]))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        f"© {datetime.now().year} {BRAND_NAME} — Automated Educational Signal Processing Report",
        styles["disclaimer"],
    ))

    doc.build(elements)
    buf.seek(0)
    return buf.getvalue()


def generate_comparison_pdf_report(*, sample_a, sample_b, lowcut, highcut, disclaimer):
    """
    Builds a Heart_Byte comparison report PDF for two heartbeat recordings and
    returns it as bytes.

    sample_a / sample_b are dicts with keys: dataset_name, file_name, sample_rate,
    duration_sec, bpm, hrv, num_beats, interval_stats, status, findings, checklist,
    time, signal, filtered_signal, envelope, peaks, peak_times.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=1.4 * cm, bottomMargin=1.4 * cm,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
        title=f"{BRAND_NAME} - Heartbeat Comparison Report",
    )

    styles = _get_styles()
    elements = list(_build_header_bar(styles, "Heartbeat Comparison Report"))

    def _hcell(text):
        return Paragraph(text, styles["table_header"])

    def _cell(text):
        return Paragraph(str(text), styles["table_cell"])

    # ---- Recording + key measurement comparison ----
    elements.append(Paragraph("Recording Comparison", styles["section"]))
    a, b = sample_a, sample_b
    comparison_rows = [
        [_hcell("Parameter"), _hcell("Sample A"), _hcell("Sample B")],
        [_cell("Dataset"), _cell(a["dataset_name"]), _cell(b["dataset_name"])],
        [_cell("File Name"), _cell(a["file_name"]), _cell(b["file_name"])],
        [_cell("Sample Rate"), _cell(f"{a['sample_rate']} Hz"), _cell(f"{b['sample_rate']} Hz")],
        [_cell("Duration Analyzed"), _cell(f"{a['duration_sec']:.1f} s"), _cell(f"{b['duration_sec']:.1f} s")],
        [_cell("Bandpass Filter"), _cell(f"{lowcut}–{highcut} Hz"), _cell(f"{lowcut}–{highcut} Hz")],
        [_cell("Heart Rate"), _cell(f"{a['bpm']:.1f} BPM"), _cell(f"{b['bpm']:.1f} BPM")],
        [_cell("HRV (SDNN-equiv.)"), _cell(f"{a['hrv']:.1f} ms"), _cell(f"{b['hrv']:.1f} ms")],
        [_cell("Detected Beats"), _cell(a["num_beats"]), _cell(b["num_beats"])],
        [_cell("Mean Interval"), _cell(f"{a['interval_stats']['mean_ms']:.0f} ms"), _cell(f"{b['interval_stats']['mean_ms']:.0f} ms")],
        [
            _cell("Interval Range"),
            _cell(f"{a['interval_stats']['min_ms']:.0f}–{a['interval_stats']['max_ms']:.0f} ms"),
            _cell(f"{b['interval_stats']['min_ms']:.0f}–{b['interval_stats']['max_ms']:.0f} ms"),
        ],
        [_cell("Diagnostic Status"), _cell(a["status"]), _cell(b["status"])],
    ]
    comparison_table = Table(comparison_rows, colWidths=[5.5 * cm, 5.7 * cm, 5.7 * cm])
    comparison_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_COLOR),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("BOX", (0, 0), (-1, -1), 0.5, LINE_GREY),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE_GREY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fbfbfb")]),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
    ]))
    elements.append(comparison_table)
    elements.append(Spacer(1, 8))

    # ---- Checklist comparison (both samples share the same 3 fixed categories) ----
    elements.append(Paragraph("Acoustic & Rhythm Checklist Comparison", styles["section"]))
    checklist_rows = [[_hcell("Category"), _hcell("Sample A"), _hcell("Flag"), _hcell("Sample B"), _hcell("Flag")]]
    flag_style_cmds = []
    for i, (item_a, item_b) in enumerate(zip(a["checklist"], b["checklist"]), start=1):
        flag_a_text, flag_a_color = FLAG_STYLE.get(item_a["flag"], ("REVIEW", AMBER))
        flag_b_text, flag_b_color = FLAG_STYLE.get(item_b["flag"], ("REVIEW", AMBER))
        checklist_rows.append([
            _cell(item_a["category"]), _cell(item_a["value"]), _cell(flag_a_text),
            _cell(item_b["value"]), _cell(flag_b_text),
        ])
        flag_style_cmds.append(("TEXTCOLOR", (2, i), (2, i), flag_a_color))
        flag_style_cmds.append(("FONTNAME", (2, i), (2, i), "Helvetica-Bold"))
        flag_style_cmds.append(("TEXTCOLOR", (4, i), (4, i), flag_b_color))
        flag_style_cmds.append(("FONTNAME", (4, i), (4, i), "Helvetica-Bold"))

    checklist_table = Table(checklist_rows, colWidths=[4.3 * cm, 3.4 * cm, 1.9 * cm, 3.4 * cm, 1.9 * cm])
    checklist_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4d4d4d")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("BOX", (0, 0), (-1, -1), 0.5, LINE_GREY),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE_GREY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fbfbfb")]),
    ] + flag_style_cmds
    checklist_table.setStyle(TableStyle(checklist_style))
    elements.append(checklist_table)

    # ---- Signal visualization ----
    chart_buf = _render_comparison_signal_chart(sample_a, sample_b)
    elements.append(KeepTogether([
        Paragraph("Signal Visualization", styles["section"]),
        Image(chart_buf, width=17.2 * cm, height=10.6 * cm),
    ]))

    # ---- Impression, per sample ----
    elements.append(Paragraph("Impression — Sample A", styles["section"]))
    for f in a["findings"]:
        elements.append(Paragraph(f"•&nbsp;&nbsp;{f}", styles["finding"]))
    elements.append(Paragraph("Impression — Sample B", styles["section"]))
    for f in b["findings"]:
        elements.append(Paragraph(f"•&nbsp;&nbsp;{f}", styles["finding"]))

    elements.append(Spacer(1, 30))
    elements.append(HRFlowable(width="100%", thickness=0.75, color=LINE_GREY, spaceAfter=14))
    elements.append(Paragraph(disclaimer, styles["disclaimer"]))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        f"© {datetime.now().year} {BRAND_NAME} — Automated Educational Signal Processing Report",
        styles["disclaimer"],
    ))

    doc.build(elements)
    buf.seek(0)
    return buf.getvalue()
