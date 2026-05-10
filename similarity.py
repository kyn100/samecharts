"""
Chart similarity engine.
Uses Pearson correlation of daily returns to find tickers whose price
charts moved most like a given target ticker over the chosen period.
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

logger = logging.getLogger(__name__)

BATCH_SIZE = 100


@st.cache_data(ttl=3600, show_spinner=False)
def download_universe_prices(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    """
    Download daily close prices for every ticker in the universe.
    Returns a DataFrame indexed by date, one column per ticker.
    Cached for 1 hour — safe to call repeatedly.
    """
    all_closes: dict[str, pd.Series] = {}
    ticker_list = list(tickers)
    total = (len(ticker_list) - 1) // BATCH_SIZE + 1

    for i in range(0, len(ticker_list), BATCH_SIZE):
        batch = ticker_list[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        try:
            raw = yf.download(
                tickers=" ".join(batch),
                period=period,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if raw.empty:
                continue
            for t in batch:
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        s = raw.xs(t, axis=1, level=0)["Close"].dropna()
                    else:
                        s = raw["Close"].dropna()
                    if len(s) >= 10:
                        all_closes[t] = s
                except Exception:
                    pass
        except Exception as exc:
            logger.warning(f"Batch {batch_num}/{total} error: {exc}")

    if not all_closes:
        return pd.DataFrame()

    prices = pd.DataFrame(all_closes)
    prices.index = pd.to_datetime(prices.index)
    return prices


def find_similar(
    target: str,
    prices: pd.DataFrame,
    top_n: int = 20,
    exclude_self: bool = True,
) -> pd.DataFrame:
    """
    Rank every ticker by how visually similar its chart is to `target`.

    Method: correlate normalized price levels (each series rebased to 1.0 at
    its first valid date).  This directly measures chart-shape similarity —
    the same thing your eye does when comparing two charts — rather than
    day-by-day return correlation, which can score high even when the overall
    shapes look nothing alike.
    """
    if target not in prices.columns:
        return pd.DataFrame()

    # Rebase every series to 1.0 at its own first valid close
    normed = prices.apply(
        lambda s: s / s.dropna().iloc[0] if not s.dropna().empty else s
    )

    min_rows = max(10, len(normed) // 2)
    normed = normed.dropna(axis=1, thresh=min_rows)

    if target not in normed.columns:
        return pd.DataFrame()

    target_normed = normed[target].dropna()

    candidates = normed.drop(columns=[target]) if exclude_self else normed

    # Align each candidate to the target's date range before correlating
    scores: dict[str, float] = {}
    for col in candidates.columns:
        s = candidates[col].dropna()
        aligned = pd.concat([target_normed, s], axis=1).dropna()
        if len(aligned) < 10:
            continue
        c = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        if not pd.isna(c):
            scores[col] = c

    corr = pd.Series(scores).sort_values(ascending=False).head(top_n)

    df = pd.DataFrame({
        "ticker": corr.index,
        "correlation": corr.values.round(4),
        "similarity_pct": (corr.values * 100).round(1),
    }).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def normalize_to_100(series: pd.Series) -> pd.Series:
    """Rebase a price series so the first valid value = 100."""
    first = series.dropna().iloc[0] if not series.dropna().empty else 1
    return series / first * 100


def pct_change_total(series: pd.Series) -> float:
    """Total % change from first to last valid value."""
    s = series.dropna()
    if len(s) < 2:
        return 0.0
    return float((s.iloc[-1] / s.iloc[0] - 1) * 100)
