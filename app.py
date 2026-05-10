"""
Stock Similarity App
====================
Tab 1 — Chart Shape:     finds stocks whose price chart moved like yours
Tab 2 — Technical Match: finds stocks in the same technical setup as yours

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from universe import get_all_tickers
from similarity import (
    download_universe_prices,
    find_similar,
    normalize_to_100,
    pct_change_total,
)
from technical_match import (
    download_ohlcv,
    extract_profile,
    compute_all_profiles,
    find_technical_matches,
    interpret_profile,
    interpret_sell_profile,
    SELL_PRESET,
    WEIGHTS,
)
from fundamentals import render_fundamentals

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Similarity",
    page_icon="📊",
    layout="wide",
)

PERIODS = {
    "1 Month":  "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year":   "1y",
}

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Chart Shape Similarity", "🔬 Technical Profile Match", "🔴 Sell Setup Scanner"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Chart Shape Similarity
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.title("Chart Shape Similarity")
    st.markdown(
        "Enter a ticker to find stocks and ETFs whose price chart "
        "followed the same pattern over the selected period."
    )

    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    with c1:
        target1 = st.text_input(
            "Ticker", value="MSTR", max_chars=10, key="t1"
        ).strip().upper()
    with c2:
        period_label1 = st.selectbox("Period", list(PERIODS.keys()), index=1, key="p1")
    with c3:
        top_n1 = st.selectbox("Results", [5, 10, 15, 20], index=1, key="n1")
    with c4:
        st.write("")
        st.write("")
        run1 = st.button("Find", type="primary", use_container_width=True, key="run1")

    period1 = PERIODS[period_label1]

    if run1 or "prices1" in st.session_state:
        universe = tuple(get_all_tickers())
        if target1 not in universe:
            universe = universe + (target1,)

        with st.spinner(f"Loading {len(universe)} tickers…"):
            prices = download_universe_prices(universe, period1)

        if prices.empty or target1 not in prices.columns:
            st.error(f"No data for **{target1}**. Check the symbol.")
        else:
            st.session_state["prices1"] = True
            with st.spinner("Calculating similarities…"):
                similar_df = find_similar(target1, prices, top_n=top_n1)

            target_s  = prices[target1].dropna()
            t_norm    = normalize_to_100(target_s)
            t_change  = pct_change_total(target_s)

            # Header
            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Target", target1)
            m2.metric("Price", f"${float(target_s.iloc[-1]):.2f}")
            m3.metric(f"{period_label1} return", f"{t_change:+.1f}%")
            m4.metric("Data days", len(target_s))

            # Overlay chart
            st.subheader(f"Overlay — {target1} vs top {top_n1}")
            fig = go.Figure()
            for _, row in similar_df.iterrows():
                t = row["ticker"]
                if t not in prices.columns:
                    continue
                s = normalize_to_100(prices[t].dropna())
                fig.add_trace(go.Scatter(
                    x=s.index, y=s.values, name=f"{t} ({row['similarity_pct']:+.1f}%)",
                    mode="lines", line=dict(width=1, color="rgba(148,163,184,0.35)"),
                ))
            fig.add_trace(go.Scatter(
                x=t_norm.index, y=t_norm.values, name=f"▶ {target1}",
                mode="lines", line=dict(width=3, color="#38bdf8"),
            ))
            fig.add_hline(y=100, line_dash="dot", line_color="rgba(255,255,255,0.15)")
            fig.update_layout(
                height=360, paper_bgcolor="#1e293b", plot_bgcolor="#0f172a",
                font=dict(color="#94a3b8"),
                legend=dict(orientation="h", y=-0.25, font=dict(size=10)),
                yaxis=dict(title="Base=100", gridcolor="#1e3a5f"),
                xaxis=dict(showgrid=False), hovermode="x unified",
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Rankings table
            st.subheader("Rankings")
            disp = similar_df[["rank","ticker","correlation","similarity_pct"]].copy()
            disp.columns = ["Rank","Ticker","Correlation","Similarity %"]
            st.dataframe(disp, use_container_width=True, hide_index=True,
                height=min(400, (len(disp)+1)*38+10),
                column_config={
                    "Correlation": st.column_config.ProgressColumn(
                        "Correlation", min_value=-1, max_value=1, format="%.4f"),
                    "Similarity %": st.column_config.NumberColumn(format="%.1f%%"),
                })

            # Individual charts grid
            st.divider()
            st.subheader("Individual Comparisons")

            def _comparison_chart(ticker, target, prices):
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.65, 0.35], vertical_spacing=0.06)
                for sym, color, name in [
                    (target, "#38bdf8", f"{target} (target)"),
                    (ticker, "#4ade80", f"{ticker} (match)"),
                ]:
                    if sym not in prices.columns: continue
                    s = normalize_to_100(prices[sym].dropna())
                    fig.add_trace(go.Scatter(x=s.index, y=s.values, name=name,
                        line=dict(color=color, width=2)), row=1, col=1)
                if target in prices.columns and ticker in prices.columns:
                    rt = prices[target].pct_change().dropna() * 100
                    rm = prices[ticker].pct_change().dropna() * 100
                    common = pd.concat([rt, rm], axis=1).dropna()
                    common.columns = [target, ticker]
                    fig.add_trace(go.Bar(x=common.index, y=common[ticker],
                        marker_color=["#4ade80" if v>=0 else "#f87171" for v in common[ticker]],
                        opacity=0.7, showlegend=False), row=2, col=1)
                    fig.add_trace(go.Scatter(x=common.index, y=common[target],
                        line=dict(color="#38bdf8", width=1.5), showlegend=False), row=2, col=1)
                fig.update_layout(height=400, paper_bgcolor="#1e293b", plot_bgcolor="#0f172a",
                    font=dict(color="#94a3b8", size=11),
                    margin=dict(l=0,r=0,t=30,b=0),
                    legend=dict(orientation="h", y=1.1, font=dict(size=10)),
                    hovermode="x unified")
                fig.update_yaxes(gridcolor="#1e3a5f", zerolinecolor="#334155")
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(title_text="Base=100", row=1, col=1)
                fig.update_yaxes(title_text="Daily %", row=2, col=1)
                return fig

            corr_map = dict(zip(similar_df["ticker"], similar_df["correlation"]))
            tlist = similar_df["ticker"].tolist()
            for i in range(0, len(tlist), 2):
                pair = tlist[i:i+2]
                cols = st.columns(2)
                for col, t in zip(cols, pair):
                    with col:
                        ch = pct_change_total(prices[t]) if t in prices.columns else 0
                        st.markdown(f"**{t}** &nbsp; corr: `{corr_map.get(t,0):.4f}` &nbsp;|&nbsp; {period_label1} return: `{ch:+.1f}%`")
                        st.plotly_chart(_comparison_chart(t, target1, prices), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Technical Profile Match
# ══════════════════════════════════════════════════════════════════════════════
import json, os

CRITERIA_FILE = os.path.join(os.path.dirname(__file__), "saved_criteria.json")

def _load_criteria() -> dict | None:
    if os.path.exists(CRITERIA_FILE):
        with open(CRITERIA_FILE, encoding="utf-8") as f:
            return json.load(f)
    return None

def _save_criteria(profile: dict) -> None:
    from datetime import datetime
    data = {
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "profile": {k: v for k, v in profile.items() if k != "ticker"},
    }
    with open(CRITERIA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def _norm_for_radar(p: dict) -> list:
    return [
        p.get("w52_position", 0),
        p.get("rsi", 50) / 100,
        max(0, min(1, p.get("bb_position", 0.5))),
        max(0, min(1, (p.get("price_vs_sma20", 0)  + 40) / 80)),
        max(0, min(1, (p.get("price_vs_sma200", 0) + 80) / 160)),
        max(0, min(1, (p.get("roc_10", 0) + 30) / 60)),
        max(0, min(1, (p.get("ao_norm", 0) + 3) / 6)),
        max(0, min(1, (p.get("cmf", 0) + 1) / 2)),
    ]

RADAR_CATS = ["52w Pos", "RSI", "BB Pos", "vs SMA20", "vs SMA200", "ROC10", "AO", "CMF"]

with tab2:
    st.title("Technical Profile Match")

    saved = _load_criteria()

    # ── Sidebar-style controls inside tab ─────────────────────────────────────
    ctrl_l, ctrl_r = st.columns([3, 1])
    with ctrl_l:
        top_n2 = st.selectbox("Results to show", [10, 15, 20, 30], index=0, key="n2")
    with ctrl_r:
        st.write("")
        run2 = st.button("Scan Now", type="primary", use_container_width=True, key="run2")

    # ── Update criteria expander (no symbol shown in main UI) ─────────────────
    with st.expander("Update Criteria from a New Chart"):
        st.caption("Analyze any ticker to replace the saved criteria with its current technical setup.")
        uc1, uc2 = st.columns([2, 1])
        with uc1:
            update_ticker = st.text_input("Ticker to analyze", max_chars=10, key="ut").strip().upper()
        with uc2:
            st.write("")
            update_btn = st.button("Extract & Save", key="ubtn")

        if update_btn and update_ticker:
            with st.spinner(f"Analyzing {update_ticker}…"):
                tmp_ohlcv = download_ohlcv((update_ticker,), "1y")
            if update_ticker in tmp_ohlcv:
                new_profile = extract_profile(update_ticker, tmp_ohlcv[update_ticker])
                if new_profile:
                    _save_criteria(new_profile)
                    saved = _load_criteria()
                    st.success("Criteria updated and saved.")
                    st.rerun()
            else:
                st.error(f"No data for {update_ticker}.")

    # ── Require saved criteria ────────────────────────────────────────────────
    if not saved:
        st.info("No criteria saved yet. Use **Update Criteria** above to extract a technical setup from any chart.")
    else:
        tp = saved["profile"]

        # ── Saved criteria display ────────────────────────────────────────────
        st.divider()
        st.subheader("Saved Criteria")
        st.caption(f"Last updated: {saved.get('saved_at', 'unknown')}")

        criteria_rows = [
            ("RSI (14)",           f"{tp['rsi']:.1f}",              "Momentum level — target zone"),
            ("52w Position",       f"{tp['w52_position']:.1%}",     "Where in yearly range — recovering from lows"),
            ("Awesome Oscillator", f"{tp['ao']:.4f}",               "Turned positive from negative" if tp.get("ao_zero_cross_up") else "Positive" if tp.get("ao") > 0 else "Negative"),
            ("CMF (20)",           f"{tp['cmf']:.4f}",              "Crossed into positive" if tp.get("cmf_zero_cross_up") else "Positive" if tp.get("cmf") > 0 else "Negative"),
            ("MACD",               "Rising" if tp.get("macd_rising") else "Falling", "Momentum direction"),
            ("Volume Ratio",       f"{tp['vol_ratio']:.2f}x",       "Recent vs prior average"),
            ("vs SMA 20",          f"{tp['price_vs_sma20']:+.1f}%", "Short-term trend"),
            ("vs SMA 50",          f"{tp['price_vs_sma50']:+.1f}%", "Medium-term trend"),
            ("vs SMA 200",         f"{tp['price_vs_sma200']:+.1f}%","Long-term trend — room to recover"),
            ("ROC 10d",            f"{tp['roc_10']:+.1f}%",         "10-day price momentum"),
            ("ROC 20d",            f"{tp['roc_20']:+.1f}%",         "20-day price momentum"),
            ("BB Position",        f"{tp['bb_position']:.3f}",      "0=lower band · 0.5=mid · 1=upper band"),
        ]

        crit_df = pd.DataFrame(criteria_rows, columns=["Indicator", "Value", "Meaning"])
        st.dataframe(crit_df, use_container_width=True, hide_index=True,
                     height=(len(crit_df) + 1) * 38 + 10)

        # Radar chart of saved criteria
        v_crit = _norm_for_radar(tp)
        fig_crit = go.Figure(go.Scatterpolar(
            r=v_crit + [v_crit[0]], theta=RADAR_CATS + [RADAR_CATS[0]],
            fill="toself", fillcolor="rgba(56,189,248,0.12)",
            line=dict(color="#38bdf8", width=2), name="Saved Criteria",
        ))
        fig_crit.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,1], gridcolor="#334155"),
                       angularaxis=dict(gridcolor="#334155")),
            paper_bgcolor="#1e293b", font=dict(color="#94a3b8"), height=340,
            title=dict(text="Criteria Fingerprint", font=dict(color="#38bdf8")),
            margin=dict(l=40, r=40, t=50, b=30),
        )
        st.plotly_chart(fig_crit, use_container_width=True)

        notes = interpret_profile(tp)
        with st.expander("What these criteria mean"):
            for label, desc in notes:
                st.markdown(f"- **{label}**: {desc}")

        # ── Run scan ─────────────────────────────────────────────────────────
        if not run2 and "tech_done" not in st.session_state:
            st.info("Click **Scan Now** to find stocks matching these criteria.")
        else:
            universe2 = tuple(get_all_tickers())
            with st.spinner(f"Downloading 1-year OHLCV for {len(universe2)} tickers…"):
                ohlcv = download_ohlcv(universe2, "1y")

            st.session_state["tech_done"] = True

            with st.spinner("Computing technical profiles…"):
                all_profiles = compute_all_profiles(ohlcv)

            matches = find_technical_matches(tp, all_profiles, top_n=top_n2)

            st.divider()
            st.subheader("Matching Stocks & ETFs")

            if matches.empty:
                st.warning("No matches found.")
            else:
                disp_cols = {
                    "rank":             "Rank",
                    "ticker":           "Ticker",
                    "similarity":       "Similarity",
                    "rsi":              "RSI",
                    "w52_position":     "52w Pos",
                    "ao":               "AO",
                    "ao_zero_cross_up": "AO Cross",
                    "cmf":              "CMF",
                    "cmf_zero_cross_up":"CMF Cross",
                    "vol_ratio":        "Vol Ratio",
                    "price_vs_sma200":  "vs SMA200%",
                    "roc_10":           "ROC 10d%",
                }
                disp = matches[list(disp_cols.keys())].rename(columns=disp_cols)
                st.dataframe(
                    disp, use_container_width=True, hide_index=True,
                    height=min(600, (len(disp)+1)*38+10),
                    column_config={
                        "Similarity":  st.column_config.ProgressColumn("Similarity", min_value=0, max_value=1, format="%.4f"),
                        "RSI":         st.column_config.NumberColumn(format="%.1f"),
                        "52w Pos":     st.column_config.ProgressColumn("52w Pos", min_value=0, max_value=1, format="%.1%"),
                        "AO":          st.column_config.NumberColumn("AO", format="%.4f"),
                        "AO Cross":    st.column_config.CheckboxColumn("AO Cross"),
                        "CMF":         st.column_config.NumberColumn("CMF", format="%.4f"),
                        "CMF Cross":   st.column_config.CheckboxColumn("CMF Cross"),
                        "Vol Ratio":   st.column_config.NumberColumn(format="%.2f"),
                        "vs SMA200%":  st.column_config.NumberColumn(format="%+.1f%%"),
                        "ROC 10d%":    st.column_config.NumberColumn(format="%+.1f%%"),
                    },
                )

                st.divider()
                st.subheader("Match Charts")

                chart_tickers = tuple(matches["ticker"].tolist())
                with st.spinner("Loading charts…"):
                    chart_prices = download_universe_prices(chart_tickers, "3mo")

                match_profiles = {p["ticker"]: p for p in all_profiles if p["ticker"] in matches["ticker"].values}

                for i in range(0, len(matches), 2):
                    pair = matches.iloc[i:i+2]
                    cols = st.columns(2)
                    for col, (_, row) in zip(cols, pair.iterrows()):
                        t = row["ticker"]
                        mp = match_profiles.get(t, {})
                        with col:
                            ao_lbl  = "cross" if row["ao_zero_cross_up"]  else f"{row['ao']:.3f}"
                            cmf_lbl = "cross" if row["cmf_zero_cross_up"] else f"{row['cmf']:.3f}"
                            st.markdown(
                                f"**{t}** &nbsp; similarity: `{row['similarity']:.4f}` &nbsp;|&nbsp; "
                                f"RSI `{row['rsi']:.1f}` &nbsp;|&nbsp; "
                                f"52w `{row['w52_position']:.1%}` &nbsp;|&nbsp; "
                                f"AO `{ao_lbl}` &nbsp;|&nbsp; CMF `{cmf_lbl}`"
                            )

                            if t in chart_prices.columns:
                                s = chart_prices[t].dropna()
                                s_norm = normalize_to_100(s)
                                fig_c = go.Figure(go.Scatter(
                                    x=s_norm.index, y=s_norm.values, name=t,
                                    line=dict(color="#38bdf8", width=2),
                                    fill="toself",
                                    fillcolor="rgba(56,189,248,0.05)",
                                ))
                                fig_c.add_hline(y=100, line_dash="dot", line_color="rgba(255,255,255,0.15)")
                                fig_c.update_layout(
                                    height=200, margin=dict(l=0,r=0,t=10,b=0),
                                    paper_bgcolor="#1e293b", plot_bgcolor="#0f172a",
                                    font=dict(color="#94a3b8", size=10), showlegend=False,
                                    yaxis=dict(gridcolor="#1e3a5f", title="Base=100"),
                                    xaxis=dict(showgrid=False),
                                )
                                st.plotly_chart(fig_c, use_container_width=True)

                            with st.expander(f"Fundamentals & Analyst Data — {t}"):
                                render_fundamentals(t)

        st.caption(
            "Universe: S&P 500 + ETFs · Indicators: RSI(14), AO(5,34), CMF(20), "
            "MACD(12,26,9), BB(20,2), SMA20/50/200, ROC10, 52w position · Not financial advice"
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Sell Setup Scanner
# ══════════════════════════════════════════════════════════════════════════════

SELL_CRITERIA_FILE = os.path.join(os.path.dirname(__file__), "saved_criteria_sell.json")

def _load_sell_criteria() -> dict | None:
    if os.path.exists(SELL_CRITERIA_FILE):
        with open(SELL_CRITERIA_FILE, encoding="utf-8") as f:
            return json.load(f)
    return None

def _save_sell_criteria(profile: dict) -> None:
    from datetime import datetime
    data = {
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "profile": {k: v for k, v in profile.items() if k != "ticker"},
    }
    with open(SELL_CRITERIA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

SELL_COLOR = "#f87171"   # red
SELL_FILL  = "rgba(248,113,113,0.12)"

with tab3:
    st.title("Sell Setup Scanner")
    st.markdown(
        "Scans for stocks where price has **just crossed below the 50-day MA**, "
        "with AO and CMF both freshly entering negative territory — early breakdown signal."
    )

    sc1, sc2 = st.columns([3, 1])
    with sc1:
        top_n3 = st.selectbox("Results to show", [10, 15, 20, 30], index=0, key="n3")
    with sc2:
        st.write("")
        run3 = st.button("Scan Now", type="primary", use_container_width=True, key="run3")

    # ── Update sell criteria from any ticker ──────────────────────────────────
    with st.expander("Update Sell Criteria from a New Chart"):
        st.caption("Analyze a stock that's showing a distribution / topping setup to replace the sell template.")
        uc3a, uc3b = st.columns([2, 1])
        with uc3a:
            update_ticker3 = st.text_input("Ticker to analyze", max_chars=10, key="ut3").strip().upper()
        with uc3b:
            st.write("")
            update_btn3 = st.button("Extract & Save", key="ubtn3")

        if update_btn3 and update_ticker3:
            with st.spinner(f"Analyzing {update_ticker3}…"):
                tmp_ohlcv3 = download_ohlcv((update_ticker3,), "1y")
            if update_ticker3 in tmp_ohlcv3:
                new_sell_profile = extract_profile(update_ticker3, tmp_ohlcv3[update_ticker3])
                if new_sell_profile:
                    _save_sell_criteria(new_sell_profile)
                    st.success("Sell criteria updated and saved.")
                    st.rerun()
            else:
                st.error(f"No data for {update_ticker3}.")

    # ── Use saved criteria or built-in preset ────────────────────────────────
    saved_sell = _load_sell_criteria()
    using_preset = saved_sell is None

    if using_preset:
        tp3 = SELL_PRESET
        st.info("Using built-in sell preset. Run a scan, or extract criteria from any topping stock above.")
        saved_at3 = "Built-in preset"
    else:
        tp3 = saved_sell["profile"]
        saved_at3 = saved_sell.get("saved_at", "unknown")

    # ── Criteria table ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Sell Criteria")
    st.caption(f"Last updated: {saved_at3}")

    ao3_lbl  = ("Crossed below zero" if tp3.get("ao_zero_cross_down")
                 else "Negative" if tp3.get("ao", 0) < 0
                 else "Still positive — watch for cross")
    cmf3_lbl = ("Crossed below zero" if tp3.get("cmf_zero_cross_down")
                 else "Negative" if tp3.get("cmf", 0) < 0
                 else "Still positive — watch for deterioration")
    macd3_lbl = "Falling" if not tp3.get("macd_rising", True) else "Still rising"

    sell_criteria_rows = [
        ("vs SMA 50",          f"{tp3['price_vs_sma50']:+.1f}%",  "Just crossed below SMA50 — key breakdown trigger"),
        ("vs SMA 20",          f"{tp3['price_vs_sma20']:+.1f}%",  "Below SMA20 — short-term trend broken"),
        ("vs SMA 200",         f"{tp3['price_vs_sma200']:+.1f}%", "Still above SMA200 — room to fall further"),
        ("Awesome Oscillator", f"{tp3['ao']:.4f}",                ao3_lbl),
        ("CMF (20)",           f"{tp3['cmf']:.4f}",               cmf3_lbl),
        ("MACD",               macd3_lbl,                         "Momentum direction — turning bearish"),
        ("RSI (14)",           f"{tp3['rsi']:.1f}",               "Dropping, not overbought — early breakdown"),
        ("52w Position",       f"{tp3['w52_position']:.1%}",      "Mid-range — breakdown from trend, not a top"),
        ("Volume Ratio",       f"{tp3['vol_ratio']:.2f}x",        "Elevated volume confirms selling pressure"),
        ("ROC 10d",            f"{tp3['roc_10']:+.1f}%",          "Negative short-term momentum"),
        ("ROC 20d",            f"{tp3['roc_20']:+.1f}%",          "Negative medium-term momentum"),
        ("BB Position",        f"{tp3['bb_position']:.3f}",       "Lower half of band — price weakening"),
    ]

    sell_crit_df = pd.DataFrame(sell_criteria_rows, columns=["Indicator", "Value", "Meaning"])
    st.dataframe(sell_crit_df, use_container_width=True, hide_index=True,
                 height=(len(sell_crit_df) + 1) * 38 + 10)

    # Radar of sell criteria
    v_sell_crit = _norm_for_radar(tp3)
    fig_sell_crit = go.Figure(go.Scatterpolar(
        r=v_sell_crit + [v_sell_crit[0]], theta=RADAR_CATS + [RADAR_CATS[0]],
        fill="toself", fillcolor=SELL_FILL,
        line=dict(color=SELL_COLOR, width=2), name="Sell Criteria",
    ))
    fig_sell_crit.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], gridcolor="#334155"),
                   angularaxis=dict(gridcolor="#334155")),
        paper_bgcolor="#1e293b", font=dict(color="#94a3b8"), height=340,
        title=dict(text="Sell Criteria Fingerprint", font=dict(color=SELL_COLOR)),
        margin=dict(l=40, r=40, t=50, b=30),
    )
    st.plotly_chart(fig_sell_crit, use_container_width=True)

    notes3 = interpret_sell_profile(tp3)
    with st.expander("What these criteria mean"):
        for label, desc in notes3:
            st.markdown(f"- **{label}**: {desc}")

    # ── Run scan ──────────────────────────────────────────────────────────────
    if not run3 and "sell_done" not in st.session_state:
        st.info("Click **Scan Now** to find stocks matching these sell criteria.")
    else:
        universe3 = tuple(get_all_tickers())
        with st.spinner(f"Downloading 1-year OHLCV for {len(universe3)} tickers…"):
            ohlcv3 = download_ohlcv(universe3, "1y")

        st.session_state["sell_done"] = True

        with st.spinner("Computing technical profiles…"):
            all_profiles3 = compute_all_profiles(ohlcv3)

        matches3 = find_technical_matches(tp3, all_profiles3, top_n=top_n3)

        # ── Results table ─────────────────────────────────────────────────────
        st.divider()
        st.subheader("Sell Candidates")

        if matches3.empty:
            st.warning("No matches found.")
        else:
            disp3_cols = {
                "rank":               "Rank",
                "ticker":             "Ticker",
                "similarity":         "Similarity",
                "rsi":                "RSI",
                "w52_position":       "52w Pos",
                "ao":                 "AO",
                "ao_zero_cross_down": "AO ↓Cross",
                "cmf":                "CMF",
                "cmf_zero_cross_down":"CMF ↓Cross",
                "vol_ratio":          "Vol Ratio",
                "price_vs_sma200":    "vs SMA200%",
                "roc_10":             "ROC 10d%",
            }
            disp3 = matches3[list(disp3_cols.keys())].rename(columns=disp3_cols)
            st.dataframe(
                disp3, use_container_width=True, hide_index=True,
                height=min(600, (len(disp3) + 1) * 38 + 10),
                column_config={
                    "Similarity":   st.column_config.ProgressColumn("Similarity", min_value=0, max_value=1, format="%.4f"),
                    "RSI":          st.column_config.NumberColumn(format="%.1f"),
                    "52w Pos":      st.column_config.ProgressColumn("52w Pos", min_value=0, max_value=1, format="%.1%"),
                    "AO":           st.column_config.NumberColumn("AO", format="%.4f"),
                    "AO ↓Cross":    st.column_config.CheckboxColumn("AO ↓Cross"),
                    "CMF":          st.column_config.NumberColumn("CMF", format="%.4f"),
                    "CMF ↓Cross":   st.column_config.CheckboxColumn("CMF ↓Cross"),
                    "Vol Ratio":    st.column_config.NumberColumn(format="%.2f"),
                    "vs SMA200%":   st.column_config.NumberColumn(format="%+.1f%%"),
                    "ROC 10d%":     st.column_config.NumberColumn(format="%+.1f%%"),
                },
            )

            st.divider()
            st.subheader("Match Charts")

            chart_tickers3 = tuple(matches3["ticker"].tolist())
            with st.spinner("Loading charts…"):
                chart_prices3 = download_universe_prices(chart_tickers3, "3mo")

            match_profiles3 = {
                p["ticker"]: p for p in all_profiles3
                if p["ticker"] in matches3["ticker"].values
            }

            for i in range(0, len(matches3), 2):
                pair3 = matches3.iloc[i:i+2]
                cols3 = st.columns(2)
                for col3, (_, row3) in zip(cols3, pair3.iterrows()):
                    t3 = row3["ticker"]
                    mp3 = match_profiles3.get(t3, {})
                    with col3:
                        ao3_match_lbl  = ("↓cross" if row3["ao_zero_cross_down"]
                                          else f"{row3['ao']:.3f}")
                        cmf3_match_lbl = ("↓cross" if row3["cmf_zero_cross_down"]
                                          else f"{row3['cmf']:.3f}")
                        st.markdown(
                            f"**{t3}** &nbsp; similarity: `{row3['similarity']:.4f}` &nbsp;|&nbsp; "
                            f"RSI `{row3['rsi']:.1f}` &nbsp;|&nbsp; "
                            f"52w `{row3['w52_position']:.1%}` &nbsp;|&nbsp; "
                            f"AO `{ao3_match_lbl}` &nbsp;|&nbsp; CMF `{cmf3_match_lbl}`"
                        )

                        if t3 in chart_prices3.columns:
                            s3 = chart_prices3[t3].dropna()
                            s3_norm = normalize_to_100(s3)
                            fig_c3 = go.Figure(go.Scatter(
                                x=s3_norm.index, y=s3_norm.values, name=t3,
                                line=dict(color=SELL_COLOR, width=2),
                                fill="toself",
                                fillcolor="rgba(248,113,113,0.05)",
                            ))
                            fig_c3.add_hline(y=100, line_dash="dot",
                                             line_color="rgba(255,255,255,0.15)")
                            fig_c3.update_layout(
                                height=200, margin=dict(l=0, r=0, t=10, b=0),
                                paper_bgcolor="#1e293b", plot_bgcolor="#0f172a",
                                font=dict(color="#94a3b8", size=10), showlegend=False,
                                yaxis=dict(gridcolor="#1e3a5f", title="Base=100"),
                                xaxis=dict(showgrid=False),
                            )
                            st.plotly_chart(fig_c3, use_container_width=True)

                        with st.expander(f"Fundamentals & Analyst Data — {t3}"):
                            render_fundamentals(t3)

    st.caption(
        "Universe: S&P 500 + ETFs · Not financial advice · "
        "Sell signals identify distribution setups, not guaranteed reversals"
    )
