"""
Downloads OHLCV data in batches and applies multi-stage filters:
  Stage 1 – price, volume (fast, no indicator math)
  Stage 2 – RSI, Stochastic, MACD, Bollinger Bands, trend
Only stocks passing every filter reach the scorer.
"""

import time
import logging
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

import config as cfg

logger = logging.getLogger(__name__)


# ─── Indicator helpers ────────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3
) -> tuple[pd.Series, pd.Series]:
    lo = low.rolling(k).min()
    hi = high.rolling(k).max()
    stoch_k = 100 * (close - lo) / (hi - lo + 1e-10)
    stoch_d = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d


def _macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_f - ema_s
    sig_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - sig_line
    return macd_line, sig_line, hist


def _bollinger(
    close: pd.Series, period: int = 20, std_dev: int = 2
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


# ─── Batch downloader ─────────────────────────────────────────────────────────

def _extract_ticker_df(data: pd.DataFrame, ticker: str, n_tickers: int) -> Optional[pd.DataFrame]:
    """Pull a single ticker's OHLCV from a (possibly multi-level) DataFrame."""
    if data is None or data.empty:
        return None
    try:
        if isinstance(data.columns, pd.MultiIndex):
            # yfinance layout: (ticker, field) — level 0 is ticker
            level = 0 if ticker in data.columns.get_level_values(0) else 1
            df = data.xs(ticker, axis=1, level=level)
        else:
            df = data  # already flat (shouldn't happen with group_by='ticker')
        return df if not df.empty else None
    except (KeyError, TypeError):
        return None


def batch_download(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Download price history for all tickers in batches. Returns {ticker: df}."""
    results: dict[str, pd.DataFrame] = {}
    total_batches = (len(tickers) - 1) // cfg.BATCH_SIZE + 1

    for i in range(0, len(tickers), cfg.BATCH_SIZE):
        batch = tickers[i : i + cfg.BATCH_SIZE]
        batch_num = i // cfg.BATCH_SIZE + 1
        logger.info(
            f"  Downloading batch {batch_num}/{total_batches} ({len(batch)} tickers)…"
        )

        try:
            raw = yf.download(
                tickers=" ".join(batch),
                period=cfg.LOOKBACK_PERIOD,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if raw.empty:
                continue

            for ticker in batch:
                df = _extract_ticker_df(raw, ticker, len(batch))
                if df is not None:
                    df = df.dropna(how="all")
                    if not df.empty:
                        results[ticker] = df

        except Exception as exc:
            logger.warning(f"  Batch {batch_num} error: {exc}")

        if cfg.BATCH_DELAY_SECONDS and i + cfg.BATCH_SIZE < len(tickers):
            time.sleep(cfg.BATCH_DELAY_SECONDS)

    logger.info(f"Download complete: {len(results)}/{len(tickers)} tickers have data")
    return results


# ─── Per-ticker screener ──────────────────────────────────────────────────────

def screen_ticker(ticker: str, df: pd.DataFrame) -> Optional[dict]:
    """
    Return a metrics dict if the ticker passes ALL filters, else None.
    """
    try:
        if len(df) < cfg.MIN_DATA_DAYS:
            return None

        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        volume = df["Volume"].astype(float)

        price = float(close.iloc[-1])
        avg_vol = float(volume.tail(20).mean())

        # ── Stage 1: basic filters (fast) ─────────────────────────────────────
        if not (cfg.MIN_PRICE <= price <= cfg.MAX_PRICE):
            return None
        if avg_vol < cfg.MIN_AVG_VOLUME:
            return None

        # ── Stage 2: technical filters ────────────────────────────────────────

        # 52-week range position (0 = at year low, 1 = at year high)
        high_52w = float(close.max())
        low_52w  = float(close.min())
        range_52w = high_52w - low_52w
        w52_position = (price - low_52w) / range_52w if range_52w > 0 else 0.5
        # A stock below RECOVERY_THRESHOLD is recovering from lows — use relaxed Stoch ceiling
        recovering = w52_position < cfg.RECOVERY_THRESHOLD
        stoch_ceiling = cfg.STOCH_MAX_RECOVERY if recovering else cfg.STOCH_MAX

        # RSI
        rsi_series = _rsi(close, cfg.RSI_PERIOD)
        current_rsi = float(rsi_series.iloc[-1])
        if not (cfg.RSI_MIN <= current_rsi <= cfg.RSI_MAX):
            return None

        # Stochastic
        stoch_k, stoch_d = _stochastic(high, low, close, cfg.STOCH_K_PERIOD, cfg.STOCH_D_PERIOD)
        current_stoch_k = float(stoch_k.iloc[-1])
        current_stoch_d = float(stoch_d.iloc[-1])
        if not (cfg.STOCH_MIN <= current_stoch_k <= stoch_ceiling):
            return None

        # Moving averages – price must be in uptrend above both SMAs
        sma20 = float(close.rolling(cfg.SMA_SHORT).mean().iloc[-1])
        sma50 = float(close.rolling(cfg.SMA_LONG).mean().iloc[-1])
        if price < sma20 or price < sma50:
            return None

        # MACD histogram (current and previous for direction)
        _, _, hist_series = _macd(close, cfg.MACD_FAST, cfg.MACD_SLOW, cfg.MACD_SIGNAL)
        hist_now = float(hist_series.iloc[-1])
        hist_prev = float(hist_series.iloc[-2])

        # Bollinger Bands – reject if at extremes
        bb_upper, bb_mid, bb_lower = _bollinger(close, cfg.BB_PERIOD, cfg.BB_STD)
        band_range = float(bb_upper.iloc[-1] - bb_lower.iloc[-1])
        bb_position = (
            (price - float(bb_lower.iloc[-1])) / band_range
            if band_range > 0 else 0.5
        )
        if bb_position < cfg.BB_EDGE or bb_position > (1 - cfg.BB_EDGE):
            return None

        # Volume trend: recent 5d vs prior 15d
        vol_recent = float(volume.tail(5).mean())
        vol_prior = float(volume.iloc[-20:-5].mean()) if len(volume) >= 20 else vol_recent
        vol_ratio = vol_recent / (vol_prior + 1e-10)

        # 10-day Rate of Change
        roc_10 = (
            float((price / close.iloc[-11] - 1) * 100) if len(close) > 11 else 0.0
        )

        return {
            "ticker": ticker,
            "price": round(price, 2),
            "rsi": round(current_rsi, 2),
            "stoch_k": round(current_stoch_k, 2),
            "stoch_d": round(current_stoch_d, 2),
            "macd_hist": round(hist_now, 5),
            "macd_hist_prev": round(hist_prev, 5),
            "sma20": round(sma20, 2),
            "sma50": round(sma50, 2),
            "bb_position": round(bb_position, 4),
            "avg_volume": int(avg_vol),
            "vol_ratio": round(vol_ratio, 4),
            "roc_10": round(roc_10, 2),
            "w52_high": round(high_52w, 2),
            "w52_low": round(low_52w, 2),
            "w52_position": round(w52_position, 4),
            "recovering": recovering,
        }

    except Exception as exc:
        logger.debug(f"{ticker}: screening error – {exc}")
        return None


def run_screener(ticker_data: dict[str, pd.DataFrame]) -> list[dict]:
    """Screen all downloaded tickers. Returns list of metrics dicts."""
    passed = []
    for ticker, df in ticker_data.items():
        result = screen_ticker(ticker, df)
        if result:
            passed.append(result)
    logger.info(
        f"Screening: {len(passed)} passed out of {len(ticker_data)} downloaded"
    )
    return passed
