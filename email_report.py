"""
Daily Technical Profile Match report.
Generates a PDF of the top matching stocks and emails it to kyn100@gmail.com.

Run manually:   python email_report.py
Scheduled via:  Windows Task Scheduler (see setup at bottom of this file)
"""

import json
import logging
import os
import smtplib
from datetime import datetime
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd
import yfinance as yf
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from technical_match import BATCH_SIZE, compute_all_profiles, find_technical_matches
from universe import get_all_tickers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

HERE           = os.path.dirname(os.path.abspath(__file__))
CRITERIA_FILE  = os.path.join(HERE, "saved_criteria.json")
EMAIL_CONFIG   = os.path.join(HERE, "email_config.json")
RECIPIENT      = "kyn100@gmail.com"
TOP_N          = 15


# ── Credentials ───────────────────────────────────────────────────────────────

def _load_cfg() -> dict:
    if os.path.exists(EMAIL_CONFIG):
        with open(EMAIL_CONFIG, encoding="utf-8") as f:
            return json.load(f)
    return {}


# ── Data (no Streamlit cache — standalone script) ────────────────────────────

def _download_ohlcv(tickers: tuple, period: str = "1y") -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}
    ticker_list = list(tickers)
    total_batches = (len(ticker_list) - 1) // BATCH_SIZE + 1
    for i in range(0, len(ticker_list), BATCH_SIZE):
        batch = ticker_list[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        logger.info(f"  Batch {batch_num}/{total_batches} ({len(batch)} tickers)…")
        try:
            raw = yf.download(
                tickers=" ".join(batch),
                period=period, interval="1d",
                group_by="ticker", auto_adjust=True,
                progress=False, threads=True,
            )
            if raw.empty:
                continue
            for t in batch:
                try:
                    df = (
                        raw.xs(t, axis=1, level=0).dropna(how="all")
                        if isinstance(raw.columns, pd.MultiIndex)
                        else raw.dropna(how="all")
                    )
                    if len(df) >= 60:
                        results[t] = df
                except Exception:
                    pass
        except Exception as exc:
            logger.warning(f"Batch {batch_num} error: {exc}")
    return results


# ── PDF ───────────────────────────────────────────────────────────────────────

def _make_pdf(matches: pd.DataFrame, criteria: dict, saved_at: str, path: str) -> None:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        path, pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
    )
    story = []

    # ── Title ─────────────────────────────────────────────────────────────────
    date_str = datetime.now().strftime("%B %d, %Y  %H:%M")
    story.append(Paragraph(f"Technical Profile Match — {date_str}", styles["Title"]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(f"Criteria last updated: {saved_at}", styles["Normal"]))
    story.append(Spacer(1, 0.25 * inch))

    # ── Saved criteria ────────────────────────────────────────────────────────
    story.append(Paragraph("Buy Criteria", styles["Heading2"]))
    story.append(Spacer(1, 0.08 * inch))

    ao_disp  = "Zero-cross ↑" if criteria.get("ao_zero_cross_up")  else f"{criteria.get('ao', 0):.4f}"
    cmf_disp = "Zero-cross ↑" if criteria.get("cmf_zero_cross_up") else f"{criteria.get('cmf', 0):.4f}"

    crit_rows = [
        ["Indicator", "Value", "Note"],
        ["RSI (14)",      f"{criteria['rsi']:.1f}",               "Momentum building"],
        ["52w Position",  f"{criteria['w52_position']:.1%}",      "Recovering from lows"],
        ["AO",            ao_disp,                                 "Crossed into positive" if criteria.get("ao_zero_cross_up") else "Positive" if criteria.get("ao", 0) > 0 else "Negative"],
        ["CMF (20)",      cmf_disp,                                "Money flowing in" if criteria.get("cmf_zero_cross_up") else "Positive" if criteria.get("cmf", 0) > 0 else "Negative"],
        ["MACD",          "Rising" if criteria.get("macd_rising") else "Falling", "Momentum direction"],
        ["vs SMA 20",     f"{criteria['price_vs_sma20']:+.1f}%",  "Short-term trend"],
        ["vs SMA 50",     f"{criteria['price_vs_sma50']:+.1f}%",  "Medium-term trend"],
        ["vs SMA 200",    f"{criteria['price_vs_sma200']:+.1f}%", "Long-term trend"],
        ["ROC 10d",       f"{criteria['roc_10']:+.1f}%",          "Recent momentum"],
        ["Vol Ratio",     f"{criteria['vol_ratio']:.2f}x",        "Recent vs prior avg"],
    ]
    crit_table = Table(crit_rows, colWidths=[1.8 * inch, 1.5 * inch, 3.0 * inch])
    crit_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#f0f9ff"), colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("PADDING",       (0, 0), (-1, -1), 4),
    ]))
    story.append(crit_table)
    story.append(Spacer(1, 0.3 * inch))

    # ── Matches ───────────────────────────────────────────────────────────────
    story.append(Paragraph(f"Top {len(matches)} Matches  (scanned S&P 500 + ETFs)", styles["Heading2"]))
    story.append(Spacer(1, 0.08 * inch))

    headers = ["#", "Ticker", "Score", "RSI", "52w", "AO", "CMF", "vs200", "ROC10", "Vol"]
    match_rows = [headers]
    for _, row in matches.iterrows():
        ao_v  = "↑cross" if row.get("ao_zero_cross_up")  else f"{row['ao']:.3f}"
        cmf_v = "↑cross" if row.get("cmf_zero_cross_up") else f"{row['cmf']:.3f}"
        match_rows.append([
            str(int(row["rank"])),
            row["ticker"],
            f"{row['similarity']:.3f}",
            f"{row['rsi']:.1f}",
            f"{row['w52_position']:.0%}",
            ao_v,
            cmf_v,
            f"{row['price_vs_sma200']:+.1f}%",
            f"{row['roc_10']:+.1f}%",
            f"{row['vol_ratio']:.2f}x",
        ])

    col_w = [0.28, 0.65, 0.55, 0.45, 0.45, 0.65, 0.65, 0.62, 0.62, 0.58]
    match_table = Table(match_rows, colWidths=[w * inch for w in col_w])
    match_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#f0f9ff"), colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("PADDING",       (0, 0), (-1, -1), 4),
        ("ALIGN",         (0, 0), (0, -1), "CENTER"),
        ("ALIGN",         (2, 0), (-1, -1), "RIGHT"),
    ]))
    story.append(match_table)
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph(
        "Not financial advice.  Data: Yahoo Finance.  "
        "Indicators: RSI(14) · AO(5,34) · CMF(20) · MACD(12,26,9) · SMA20/50/200 · ROC10 · 52w range",
        styles["Italic"],
    ))

    doc.build(story)
    logger.info(f"PDF written: {path}")


# ── Email ─────────────────────────────────────────────────────────────────────

def _send_email(pdf_path: str, cfg: dict) -> None:
    gmail_user = cfg.get("gmail_user", "kyn100@gmail.com")
    app_pw     = cfg.get("gmail_app_password", "")
    if not app_pw or app_pw == "REPLACE_WITH_YOUR_APP_PASSWORD":
        raise ValueError(
            "Set gmail_app_password in email_config.json.\n"
            "Get one at: myaccount.google.com → Security → App passwords"
        )

    date_str = datetime.now().strftime("%Y-%m-%d")
    msg = MIMEMultipart()
    msg["From"]    = gmail_user
    msg["To"]      = RECIPIENT
    msg["Subject"] = f"Stock Buy Signals — {date_str}"

    body = (
        f"Hi,\n\n"
        f"Attached is today's Technical Profile Match report ({date_str}).\n\n"
        f"It shows the top stocks and ETFs whose current technical setup best matches "
        f"your saved buy criteria — RSI momentum, AO & CMF zero-crosses, SMA levels, "
        f"and volume.\n\n"
        f"Not financial advice.\n"
    )
    msg.attach(MIMEText(body, "plain"))

    with open(pdf_path, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header(
        "Content-Disposition",
        f'attachment; filename="{os.path.basename(pdf_path)}"',
    )
    msg.attach(part)

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.ehlo()
        server.starttls()
        server.login(gmail_user, app_pw)
        server.sendmail(gmail_user, RECIPIENT, msg.as_string())
    logger.info(f"Email sent → {RECIPIENT}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = _load_cfg()

    if not os.path.exists(CRITERIA_FILE):
        logger.error("saved_criteria.json not found — open the Streamlit app and save criteria first.")
        return

    with open(CRITERIA_FILE, encoding="utf-8") as f:
        saved = json.load(f)
    tp       = saved["profile"]
    saved_at = saved.get("saved_at", "unknown")

    logger.info(f"Loaded criteria (saved {saved_at})")
    logger.info("Fetching universe tickers…")
    universe = tuple(get_all_tickers())

    logger.info(f"Downloading OHLCV for {len(universe)} tickers (~3–5 min)…")
    ohlcv = _download_ohlcv(universe, "1y")

    logger.info("Computing technical profiles…")
    all_profiles = compute_all_profiles(ohlcv)
    logger.info(f"Profiles ready: {len(all_profiles)}")

    matches = find_technical_matches(tp, all_profiles, top_n=TOP_N)
    logger.info(f"Top {len(matches)} matches found.")

    date_str = datetime.now().strftime("%Y-%m-%d")
    pdf_path = os.path.join(HERE, f"report_{date_str}.pdf")
    _make_pdf(matches, tp, saved_at, pdf_path)
    _send_email(pdf_path, cfg)


if __name__ == "__main__":
    main()
