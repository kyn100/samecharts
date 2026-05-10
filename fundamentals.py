"""
Fetches fundamental data, analyst ratings, and price targets from Yahoo Finance.
All results cached 24 hours — fundamentals don't change intraday.
"""

import streamlit as st
import yfinance as yf


def _fmt_large(n) -> str:
    """Format large numbers: 1.23T, 456.7B, 12.3M, etc."""
    if n is None:
        return "N/A"
    n = float(n)
    if abs(n) >= 1e12:
        return f"${n/1e12:.2f}T"
    if abs(n) >= 1e9:
        return f"${n/1e9:.2f}B"
    if abs(n) >= 1e6:
        return f"${n/1e6:.2f}M"
    return f"${n:,.0f}"


def _pct(n) -> str:
    return f"{n*100:.1f}%" if n is not None else "N/A"


def _num(n, fmt=".2f") -> str:
    return f"{n:{fmt}}" if n is not None else "N/A"


@st.cache_data(ttl=86_400, show_spinner=False)
def get_fundamentals(ticker: str) -> dict:
    """Return a flat dict of fundamental + analyst data for one ticker."""
    try:
        info = yf.Ticker(ticker).info

        current = info.get("currentPrice") or info.get("regularMarketPrice")
        target_mean = info.get("targetMeanPrice")
        upside = (target_mean / current - 1) * 100 if current and target_mean else None

        rec_mean = info.get("recommendationMean")          # 1=Strong Buy … 5=Strong Sell
        rec_key  = (info.get("recommendationKey") or "n/a").replace("_", " ").title()

        return {
            # Company
            "name":        info.get("longName", ticker),
            "sector":      info.get("sector", "N/A"),
            "industry":    info.get("industry", "N/A"),
            "country":     info.get("country", "N/A"),
            "employees":   info.get("fullTimeEmployees"),
            "description": info.get("longBusinessSummary", ""),
            "website":     info.get("website", ""),
            # Valuation
            "market_cap":  info.get("marketCap"),
            "trailing_pe": info.get("trailingPE"),
            "forward_pe":  info.get("forwardPE"),
            "pb_ratio":    info.get("priceToBook"),
            "ps_ratio":    info.get("priceToSalesTrailing12Months"),
            "ev_ebitda":   info.get("enterpriseToEbitda"),
            # Per-share
            "trailing_eps": info.get("trailingEps"),
            "forward_eps":  info.get("forwardEps"),
            "dividend_yield": info.get("dividendYield"),
            "beta":        info.get("beta"),
            # Growth & profitability
            "revenue_growth":   info.get("revenueGrowth"),
            "earnings_growth":  info.get("earningsGrowth"),
            "gross_margin":     info.get("grossMargins"),
            "operating_margin": info.get("operatingMargins"),
            "profit_margin":    info.get("profitMargins"),
            "roe":   info.get("returnOnEquity"),
            "roa":   info.get("returnOnAssets"),
            # Balance sheet
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio":  info.get("currentRatio"),
            # Analyst consensus
            "rec_mean":       rec_mean,
            "rec_key":        rec_key,
            "num_analysts":   info.get("numberOfAnalystOpinions"),
            "target_high":    info.get("targetHighPrice"),
            "target_low":     info.get("targetLowPrice"),
            "target_mean":    target_mean,
            "target_median":  info.get("targetMedianPrice"),
            "current_price":  current,
            "upside_mean":    upside,
        }
    except Exception as exc:
        return {"name": ticker, "error": str(exc)}


def rec_label(mean) -> tuple[str, str]:
    """Return (label, hex color) for a recommendation mean score."""
    if mean is None:
        return "N/A", "#64748b"
    if mean <= 1.5:
        return "Strong Buy",  "#16a34a"
    if mean <= 2.5:
        return "Buy",         "#4ade80"
    if mean <= 3.5:
        return "Hold",        "#facc15"
    if mean <= 4.5:
        return "Sell",        "#f97316"
    return "Strong Sell",     "#ef4444"


def render_fundamentals(ticker: str) -> None:
    """Render a full fundamental panel for one ticker inside the current Streamlit container."""
    f = get_fundamentals(ticker)

    if "error" in f:
        st.caption(f"Could not load fundamentals: {f['error']}")
        return

    # ── Company header ────────────────────────────────────────────────────────
    st.markdown(f"**{f['name']}** &nbsp;·&nbsp; {f['sector']} / {f['industry']} &nbsp;·&nbsp; {f['country']}")
    if f["description"]:
        # Show first 3 sentences only
        sentences = f["description"].replace("\n", " ").split(". ")
        brief = ". ".join(sentences[:3]).strip()
        if not brief.endswith("."):
            brief += "."
        st.caption(brief)

    # ── Three metric columns ──────────────────────────────────────────────────
    col_v, col_g, col_a = st.columns(3)

    with col_v:
        st.markdown("**Valuation**")
        st.markdown(f"""
| | |
|---|---|
| Market Cap | {_fmt_large(f['market_cap'])} |
| Trailing P/E | {_num(f['trailing_pe'])} |
| Forward P/E | {_num(f['forward_pe'])} |
| P/B | {_num(f['pb_ratio'])} |
| P/S | {_num(f['ps_ratio'])} |
| EV/EBITDA | {_num(f['ev_ebitda'])} |
| Trailing EPS | {_num(f['trailing_eps'])} |
| Forward EPS | {_num(f['forward_eps'])} |
| Dividend Yield | {_pct(f['dividend_yield'])} |
| Beta | {_num(f['beta'])} |
""")

    with col_g:
        st.markdown("**Growth & Profitability**")
        st.markdown(f"""
| | |
|---|---|
| Revenue Growth | {_pct(f['revenue_growth'])} |
| Earnings Growth | {_pct(f['earnings_growth'])} |
| Gross Margin | {_pct(f['gross_margin'])} |
| Operating Margin | {_pct(f['operating_margin'])} |
| Profit Margin | {_pct(f['profit_margin'])} |
| ROE | {_pct(f['roe'])} |
| ROA | {_pct(f['roa'])} |
| Debt / Equity | {_num(f['debt_to_equity'])} |
| Current Ratio | {_num(f['current_ratio'])} |
| Employees | {f'{f["employees"]:,}' if f['employees'] else 'N/A'} |
""")

    with col_a:
        st.markdown("**Analyst Ratings**")

        label, color = rec_label(f["rec_mean"])
        n = f["num_analysts"] or 0
        st.markdown(
            f"<div style='font-size:1.5rem;font-weight:700;color:{color}'>{label}</div>"
            f"<div style='color:#94a3b8;font-size:.85rem'>{n} analyst{'s' if n!=1 else ''} · mean score {_num(f['rec_mean'])}</div>"
            f"<div style='color:#64748b;font-size:.75rem'>Scale: 1&nbsp;=&nbsp;Strong Buy &nbsp;→&nbsp; 3&nbsp;=&nbsp;Hold &nbsp;→&nbsp; 5&nbsp;=&nbsp;Strong Sell</div>",
            unsafe_allow_html=True,
        )

        # Price target bar
        cur   = f["current_price"]
        t_low = f["target_low"]
        t_hi  = f["target_high"]
        t_mid = f["target_mean"]

        if cur and t_low and t_hi and t_mid:
            st.markdown("")
            lo_pct = (t_low / cur - 1) * 100
            hi_pct = (t_hi  / cur - 1) * 100
            md_pct = (t_mid / cur - 1) * 100
            upside_color = "#4ade80" if md_pct >= 0 else "#f87171"
            st.markdown(f"""
| | |
|---|---|
| Current Price | ${cur:.2f} |
| Target Low | ${t_low:.2f} &nbsp; ({lo_pct:+.1f}%) |
| Target Mean | **${t_mid:.2f}** &nbsp; <span style='color:{upside_color}'>**{md_pct:+.1f}%**</span> |
| Target High | ${t_hi:.2f} &nbsp; ({hi_pct:+.1f}%) |
""", unsafe_allow_html=True)
        else:
            st.caption("No price target data available.")
