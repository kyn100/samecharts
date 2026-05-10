"""
Technical Profile Matcher
=========================
Extracts a stock's current technical "fingerprint" and finds other stocks
in the universe that are in the same technical situation right now.

ZS example profile (the archetype this was designed for):
  - Recovering from 52-week lows      (w52_position ≈ 0.15)
  - RSI building momentum              (RSI ≈ 60)
  - MACD positive and strengthening
  - Price above SMA20 & SMA50 but still below SMA200  (room to recover)
  - Strong 10/20-day momentum          (ROC10 ≈ +12%)
  - Volume rising
  - Awesome Oscillator turning positive from negative  (zero-line cross)
  - Chaikin Money Flow crossing from negative into positive
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from screener import _rsi, _stochastic, _macd, _bollinger


def _awesome_oscillator(high: pd.Series, low: pd.Series) -> pd.Series:
    """AO = SMA(median price, 5) - SMA(median price, 34)."""
    median = (high + low) / 2
    return median.rolling(5).mean() - median.rolling(34).mean()


def _cmf(high: pd.Series, low: pd.Series, close: pd.Series,
         volume: pd.Series, period: int = 20) -> pd.Series:
    """Chaikin Money Flow = Sum(MFV, period) / Sum(Volume, period)."""
    hl_range = (high - low).replace(0, 1e-10)
    mf_mult = ((close - low) - (high - close)) / hl_range
    mf_vol  = mf_mult * volume
    return mf_vol.rolling(period).sum() / volume.rolling(period).sum()

logger = logging.getLogger(__name__)

BATCH_SIZE = 100

# ── Indicator weights (must sum to 1.0) ───────────────────────────────────────
WEIGHTS = {
    "w52_position":       0.16,   # Where in the yearly range — most distinctive
    "rsi":                0.10,   # Momentum level
    "macd_strength":      0.10,   # MACD magnitude + direction
    "ao_norm":            0.18,   # Awesome Oscillator — zero-line cross is key signal
    "cmf":                0.18,   # Chaikin Money Flow — negative→positive cross
    "vol_ratio":          0.08,   # Volume trend
    "price_vs_sma20":     0.06,   # Short-term trend relationship
    "price_vs_sma50":     0.06,   # Medium-term trend relationship
    "price_vs_sma200":    0.08,   # Long-term trend (recovery room)
    "roc_10":             0.05,   # Recent price momentum
    "bb_position":        0.05,   # Bollinger position
}


# ── OHLCV downloader (cached) ─────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def download_ohlcv(tickers: tuple[str, ...], period: str = "1y") -> dict[str, pd.DataFrame]:
    """Download 1-year OHLCV for all tickers. Returns {ticker: df}."""
    results: dict[str, pd.DataFrame] = {}
    ticker_list = list(tickers)

    for i in range(0, len(ticker_list), BATCH_SIZE):
        batch = ticker_list[i : i + BATCH_SIZE]
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
            logger.warning(f"Batch download error: {exc}")

    return results


# ── Profile extraction ────────────────────────────────────────────────────────

def extract_profile(ticker: str, df: pd.DataFrame) -> Optional[dict]:
    """Compute the technical fingerprint for a single ticker."""
    try:
        close  = df["Close"].astype(float)
        high   = df["High"].astype(float)
        low    = df["Low"].astype(float)
        volume = df["Volume"].astype(float)

        if len(close) < 60:
            return None

        price = float(close.iloc[-1])

        # RSI
        rsi = float(_rsi(close).iloc[-1])

        # Stochastic
        sk, sd = _stochastic(high, low, close)
        stoch_k = float(sk.iloc[-1])

        # MACD
        _, _, hist = _macd(close)
        macd_now  = float(hist.iloc[-1])
        macd_prev = float(hist.iloc[-2])
        # macd_strength: signed magnitude, positive = bullish
        macd_strength = macd_now  # raw value, normalised later relative to price

        # Bollinger Bands
        bu, bm, bl = _bollinger(close)
        band_w = float(bu.iloc[-1]) - float(bl.iloc[-1])
        bb_position = (price - float(bl.iloc[-1])) / band_w if band_w > 0 else 0.5

        # 52-week position
        high_52w = float(close.max())
        low_52w  = float(close.min())
        w52_range = high_52w - low_52w
        w52_position = (price - low_52w) / w52_range if w52_range > 0 else 0.5

        # SMAs
        sma20  = float(close.rolling(20).mean().iloc[-1])
        sma50  = float(close.rolling(50).mean().iloc[-1])
        sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float(close.rolling(len(close)).mean().iloc[-1])

        price_vs_sma20  = (price / sma20  - 1) * 100
        price_vs_sma50  = (price / sma50  - 1) * 100
        price_vs_sma200 = (price / sma200 - 1) * 100

        # Volume ratio (5d vs prior 15d)
        vol_r = float(volume.tail(5).mean()) / (float(volume.iloc[-20:-5].mean()) + 1e-10)

        # ROC
        roc_10 = float((price / close.iloc[-11] - 1) * 100) if len(close) > 11 else 0.0
        roc_20 = float((price / close.iloc[-21] - 1) * 100) if len(close) > 21 else 0.0

        # Awesome Oscillator
        ao_series = _awesome_oscillator(high, low)
        ao_now    = float(ao_series.iloc[-1])
        ao_prev   = float(ao_series.iloc[-2])
        # Normalise AO relative to price so it's comparable across stocks
        ao_norm   = ao_now / max(price, 1) * 100
        ao_rising = ao_now > ao_prev
        # Detect recent zero-line crosses (within last 5 bars)
        ao_recent = ao_series.iloc[-6:-1]
        ao_zero_cross_up   = bool(ao_now > 0 and (ao_recent < 0).any())
        ao_zero_cross_down = bool(ao_now < 0 and (ao_recent > 0).any())

        # Chaikin Money Flow (20-period)
        cmf_series = _cmf(high, low, close, volume, period=20)
        cmf_now    = float(cmf_series.iloc[-1])
        cmf_prev   = float(cmf_series.iloc[-2])
        cmf_rising = cmf_now > cmf_prev
        cmf_zero_cross_up   = bool(cmf_now > 0 and float(cmf_series.iloc[-6:-1].min()) < 0)
        cmf_zero_cross_down = bool(cmf_now < 0 and float(cmf_series.iloc[-6:-1].max()) > 0)

        return {
            "ticker":           ticker,
            "price":            round(price, 2),
            "rsi":              round(rsi, 2),
            "stoch_k":          round(stoch_k, 2),
            "macd_strength":    round(macd_now / max(price, 1) * 1000, 4),
            "macd_rising":      macd_now > macd_prev,
            "bb_position":      round(bb_position, 4),
            "w52_position":     round(w52_position, 4),
            "w52_high":         round(high_52w, 2),
            "w52_low":          round(low_52w, 2),
            "price_vs_sma20":   round(price_vs_sma20, 2),
            "price_vs_sma50":   round(price_vs_sma50, 2),
            "price_vs_sma200":  round(price_vs_sma200, 2),
            "sma20":            round(sma20, 2),
            "sma50":            round(sma50, 2),
            "sma200":           round(sma200, 2),
            "vol_ratio":        round(vol_r, 4),
            "roc_10":           round(roc_10, 2),
            "roc_20":           round(roc_20, 2),
            # New indicators
            "ao":                 round(ao_now, 4),
            "ao_norm":            round(ao_norm, 4),
            "ao_rising":          ao_rising,
            "ao_zero_cross_up":   ao_zero_cross_up,
            "ao_zero_cross_down": ao_zero_cross_down,
            "cmf":                round(cmf_now, 4),
            "cmf_rising":         cmf_rising,
            "cmf_zero_cross_up":  cmf_zero_cross_up,
            "cmf_zero_cross_down":cmf_zero_cross_down,
        }

    except Exception as exc:
        logger.debug(f"{ticker}: profile error — {exc}")
        return None


def compute_all_profiles(ohlcv: dict[str, pd.DataFrame]) -> list[dict]:
    profiles = []
    for ticker, df in ohlcv.items():
        p = extract_profile(ticker, df)
        if p:
            profiles.append(p)
    return profiles


# ── Normalisation helpers ─────────────────────────────────────────────────────

def _norm(value: float, lo: float, hi: float) -> float:
    """Clamp and scale value to [0, 1]."""
    return max(0.0, min(1.0, (value - lo) / (hi - lo + 1e-10)))


def _normalize(profile: dict) -> dict:
    return {
        "w52_position":    _norm(profile["w52_position"],    0.0,   1.0),
        "rsi":             _norm(profile["rsi"],             0.0, 100.0),
        "macd_strength":   _norm(profile["macd_strength"],  -5.0,   5.0),
        "ao_norm":         _norm(profile["ao_norm"],         -3.0,   3.0),
        "cmf":             _norm(profile["cmf"],             -1.0,   1.0),
        "vol_ratio":       _norm(profile["vol_ratio"],        0.3,   2.5),
        "price_vs_sma20":  _norm(profile["price_vs_sma20"], -40.0,  40.0),
        "price_vs_sma50":  _norm(profile["price_vs_sma50"], -60.0,  60.0),
        "price_vs_sma200": _norm(profile["price_vs_sma200"],-80.0,  80.0),
        "roc_10":          _norm(profile["roc_10"],          -30.0,  30.0),
        "bb_position":     _norm(profile["bb_position"],     -0.5,   1.5),
    }


# ── Distance & matching ───────────────────────────────────────────────────────

def _distance(a: dict, b: dict) -> float:
    na, nb = _normalize(a), _normalize(b)
    total = sum(
        WEIGHTS[k] * (na[k] - nb[k]) ** 2
        for k in WEIGHTS
    )
    return total ** 0.5


def find_technical_matches(
    target: dict,
    all_profiles: list[dict],
    top_n: int = 15,
) -> pd.DataFrame:
    target_ticker = target.get("ticker", None)
    rows = []
    for p in all_profiles:
        if target_ticker and p["ticker"] == target_ticker:
            continue
        dist = _distance(target, p)
        similarity = max(0.0, 1.0 - dist / 0.7)
        rows.append({
            "ticker":           p["ticker"],
            "similarity":       round(similarity, 4),
            "price":            p["price"],
            "rsi":              p["rsi"],
            "w52_position":     p["w52_position"],
            "ao":                 p["ao"],
            "ao_rising":          p["ao_rising"],
            "ao_zero_cross_up":   p["ao_zero_cross_up"],
            "ao_zero_cross_down": p.get("ao_zero_cross_down", False),
            "cmf":                p["cmf"],
            "cmf_rising":         p["cmf_rising"],
            "cmf_zero_cross_up":  p["cmf_zero_cross_up"],
            "cmf_zero_cross_down":p.get("cmf_zero_cross_down", False),
            "vol_ratio":        p["vol_ratio"],
            "price_vs_sma20":   p["price_vs_sma20"],
            "price_vs_sma50":   p["price_vs_sma50"],
            "price_vs_sma200":  p["price_vs_sma200"],
            "roc_10":           p["roc_10"],
            "macd_rising":      p["macd_rising"],
        })

    rows.sort(key=lambda r: r["similarity"], reverse=True)
    df = pd.DataFrame(rows[:top_n])
    if not df.empty:
        df.insert(0, "rank", range(1, len(df) + 1))
    return df


# ── Human-readable interpretation ────────────────────────────────────────────

def interpret_profile(p: dict) -> list[tuple[str, str]]:
    """Return a list of (label, description) strings explaining the profile."""
    notes = []

    w52 = p["w52_position"]
    if w52 < 0.25:
        notes.append(("52w Position", f"{w52:.0%} of range — recovering from lows, lots of upside room"))
    elif w52 < 0.50:
        notes.append(("52w Position", f"{w52:.0%} of range — mid-range, balanced"))
    elif w52 < 0.75:
        notes.append(("52w Position", f"{w52:.0%} of range — upper half, momentum driven"))
    else:
        notes.append(("52w Position", f"{w52:.0%} of range — near 52-week highs"))

    rsi = p["rsi"]
    if rsi < 40:
        notes.append(("RSI", f"{rsi:.1f} — oversold territory, potential bounce zone"))
    elif rsi < 55:
        notes.append(("RSI", f"{rsi:.1f} — neutral, balanced momentum"))
    elif rsi < 70:
        notes.append(("RSI", f"{rsi:.1f} — bullish momentum, not yet overbought"))
    else:
        notes.append(("RSI", f"{rsi:.1f} — overbought, proceed with caution"))

    if p["macd_rising"]:
        notes.append(("MACD", "Histogram rising — momentum accelerating"))
    else:
        notes.append(("MACD", "Histogram falling — momentum slowing"))

    s20 = p["price_vs_sma20"]
    s50 = p["price_vs_sma50"]
    s200 = p["price_vs_sma200"]
    if s20 > 0 and s50 > 0 and s200 < 0:
        notes.append(("SMA Trend", f"Above SMA20 ({s20:+.1f}%) & SMA50 ({s50:+.1f}%) but below SMA200 ({s200:+.1f}%) — recovery in progress"))
    elif s20 > 0 and s50 > 0 and s200 > 0:
        notes.append(("SMA Trend", f"Above all three SMAs — confirmed uptrend"))
    elif s20 < 0:
        notes.append(("SMA Trend", f"Below SMA20 ({s20:+.1f}%) — short-term weakness"))

    roc = p["roc_10"]
    if roc > 10:
        notes.append(("Momentum", f"ROC10 {roc:+.1f}% — strong recent surge"))
    elif roc > 3:
        notes.append(("Momentum", f"ROC10 {roc:+.1f}% — healthy positive momentum"))
    elif roc > 0:
        notes.append(("Momentum", f"ROC10 {roc:+.1f}% — mild positive momentum"))
    else:
        notes.append(("Momentum", f"ROC10 {roc:+.1f}% — recent pullback"))

    ao = p.get("ao", 0)
    ao_cross = p.get("ao_zero_cross_up", False)
    ao_rising = p.get("ao_rising", False)
    if ao_cross:
        notes.append(("Awesome Oscillator", f"{ao:.4f} — just crossed zero from below (bullish signal)"))
    elif ao > 0 and ao_rising:
        notes.append(("Awesome Oscillator", f"{ao:.4f} — positive and rising"))
    elif ao > 0:
        notes.append(("Awesome Oscillator", f"{ao:.4f} — positive but losing steam"))
    else:
        notes.append(("Awesome Oscillator", f"{ao:.4f} — still negative"))

    cmf = p.get("cmf", 0)
    cmf_cross = p.get("cmf_zero_cross_up", False)
    cmf_rising = p.get("cmf_rising", False)
    if cmf_cross:
        notes.append(("CMF", f"{cmf:.4f} — just crossed from negative to positive (money flowing in)"))
    elif cmf > 0 and cmf_rising:
        notes.append(("CMF", f"{cmf:.4f} — positive and strengthening"))
    elif cmf > 0:
        notes.append(("CMF", f"{cmf:.4f} — positive but fading"))
    elif cmf_rising:
        notes.append(("CMF", f"{cmf:.4f} — still negative but rising toward zero"))
    else:
        notes.append(("CMF", f"{cmf:.4f} — negative"))

    vol = p.get("vol_ratio", 1.0)
    if vol >= 1.3:
        notes.append(("Volume", f"{vol:.2f}x — significantly above average (confirms move)"))
    elif vol >= 1.05:
        notes.append(("Volume", f"{vol:.2f}x — above average"))
    else:
        notes.append(("Volume", f"{vol:.2f}x — average or below"))

    return notes


# ── Sell-setup preset & interpreter ──────────────────────────────────────────

# Archetype: extended stock showing distribution and momentum breakdown.
# All WEIGHTS keys must be present; booleans are display-only.
SELL_PRESET: dict = {
    # Weighted dimensions (used in distance calculation)
    # Archetype: price just crossed below SMA50, AO and CMF both freshly negative
    "w52_position":    0.48,   # mid-range — not a topping stock, a breaking-down one
    "rsi":             47.0,   # dropping toward oversold, not overbought
    "macd_strength":   -0.6,   # MACD histogram turning negative
    "ao_norm":         -0.25,  # AO just entered negative (fresh zero-cross down)
    "cmf":             -0.07,  # CMF just entered negative (fresh zero-cross down)
    "vol_ratio":        1.20,  # above-average volume confirming breakdown
    "price_vs_sma20":  -3.5,   # below SMA20 — short-term trend broken
    "price_vs_sma50":  -2.5,   # just crossed below SMA50
    "price_vs_sma200":  5.0,   # still above SMA200 — room to fall further
    "roc_10":          -5.0,   # negative 10-day momentum
    "bb_position":      0.30,  # lower half of Bollinger Band
    # Display-only fields
    "ao":               -0.08,
    "ao_rising":        False,
    "ao_zero_cross_up": False,
    "ao_zero_cross_down": True,
    "cmf_rising":       False,
    "cmf_zero_cross_up": False,
    "cmf_zero_cross_down": True,
    "macd_rising":      False,
    "roc_20":          -6.0,
    "stoch_k":          38.0,
}


def interpret_sell_profile(p: dict) -> list[tuple[str, str]]:
    """Return (label, description) explaining a sell-setup profile."""
    notes = []

    s50  = p["price_vs_sma50"]
    s20  = p["price_vs_sma20"]
    s200 = p["price_vs_sma200"]
    if s50 < -1:
        notes.append(("vs SMA50", f"{s50:+.1f}% — price just crossed below SMA50 (key breakdown level)"))
    elif s50 < 0:
        notes.append(("vs SMA50", f"{s50:+.1f}% — testing SMA50 from below"))
    else:
        notes.append(("vs SMA50", f"{s50:+.1f}% — still above SMA50"))

    if s20 < 0:
        notes.append(("vs SMA20", f"{s20:+.1f}% — below SMA20, short-term trend broken"))
    else:
        notes.append(("vs SMA20", f"{s20:+.1f}% — still above SMA20"))

    if s200 > 0:
        notes.append(("vs SMA200", f"{s200:+.1f}% — still above SMA200, room to fall further"))
    else:
        notes.append(("vs SMA200", f"{s200:+.1f}% — below SMA200, downtrend confirmed"))

    ao = p.get("ao", 0)
    ao_cross_dn = p.get("ao_zero_cross_down", False)
    ao_rising   = p.get("ao_rising", True)
    if ao_cross_dn:
        notes.append(("Awesome Oscillator", f"{ao:.4f} — just crossed zero from above (bearish momentum flip)"))
    elif ao < 0 and not ao_rising:
        notes.append(("Awesome Oscillator", f"{ao:.4f} — negative and falling"))
    elif ao < 0:
        notes.append(("Awesome Oscillator", f"{ao:.4f} — freshly negative"))
    else:
        notes.append(("Awesome Oscillator", f"{ao:.4f} — still positive, watch for zero-line cross"))

    cmf = p.get("cmf", 0)
    cmf_cross_dn = p.get("cmf_zero_cross_down", False)
    cmf_rising   = p.get("cmf_rising", True)
    if cmf_cross_dn:
        notes.append(("CMF", f"{cmf:.4f} — just crossed below zero (money flowing out, early distribution)"))
    elif cmf < 0 and not cmf_rising:
        notes.append(("CMF", f"{cmf:.4f} — negative and worsening"))
    elif cmf < 0:
        notes.append(("CMF", f"{cmf:.4f} — freshly negative"))
    else:
        notes.append(("CMF", f"{cmf:.4f} — still positive, watch for deterioration"))

    if not p.get("macd_rising", True):
        notes.append(("MACD", "Histogram falling — momentum breaking down"))
    else:
        notes.append(("MACD", "Histogram still rising — watch for rollover"))

    rsi = p["rsi"]
    if rsi <= 45:
        notes.append(("RSI", f"{rsi:.1f} — weakening, momentum shifting bearish"))
    elif rsi <= 55:
        notes.append(("RSI", f"{rsi:.1f} — mid-range, losing upside momentum"))
    else:
        notes.append(("RSI", f"{rsi:.1f} — still elevated"))

    roc = p["roc_10"]
    if roc < -5:
        notes.append(("Momentum", f"ROC10 {roc:+.1f}% — sharp breakdown underway"))
    elif roc < 0:
        notes.append(("Momentum", f"ROC10 {roc:+.1f}% — negative, breakdown confirmed"))
    else:
        notes.append(("Momentum", f"ROC10 {roc:+.1f}% — still positive but fading"))

    vol = p.get("vol_ratio", 1.0)
    if vol >= 1.3:
        notes.append(("Volume", f"{vol:.2f}x — elevated selling volume (confirms breakdown)"))
    elif vol >= 1.05:
        notes.append(("Volume", f"{vol:.2f}x — slightly above average"))
    else:
        notes.append(("Volume", f"{vol:.2f}x — below average (weak breakdown, watch for confirmation)"))

    return notes
