# ─── ETF Configuration ────────────────────────────────────────────────────────

# RSI thresholds (filter – hard cutoff)
RSI_PERIOD = 14
RSI_MIN = 30   # < 30 → oversold  → excluded
RSI_MAX = 70   # > 70 → overbought → excluded

# Stochastic thresholds (filter)
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
STOCH_MIN = 20
STOCH_MAX = 80

# 52-week range recovery logic
# If a stock is still deep below its 52-week high (position < RECOVERY_THRESHOLD),
# it is treated as "recovering" and the Stochastic ceiling is relaxed to STOCH_MAX_RECOVERY.
# This avoids falsely flagging stocks that bounced off multi-month lows as "overbought".
RECOVERY_THRESHOLD = 0.40   # below 40% of 52w range → recovery mode
STOCH_MAX_RECOVERY = 95     # relaxed Stochastic ceiling for recovering stocks

# Moving averages
SMA_SHORT = 20
SMA_LONG = 50

# MACD
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Bollinger Bands
BB_PERIOD = 20
BB_STD = 2
BB_EDGE = 0.05   # reject if bb_position < 5% or > 95%

# Basic pre-filters
MIN_PRICE = 5.0          # Avoid penny stocks
MAX_PRICE = 10_000.0
MIN_AVG_VOLUME = 300_000  # Shares/day
MIN_DATA_DAYS = 60        # Minimum history required

# Portfolio
TOP_N = 10
LOOKBACK_PERIOD = "1y"    # 1 year needed for 52-week range calculation

# Download batch size (tickers per yfinance call)
BATCH_SIZE = 100
BATCH_DELAY_SECONDS = 1   # Polite delay between batches

# Scoring weights (must sum to 1.0)
WEIGHTS = {
    "rsi":      0.25,   # Closest to 50 → most balanced
    "macd":     0.20,   # Positive + strengthening histogram
    "trend":    0.20,   # BB position near midline
    "volume":   0.15,   # Increasing relative volume
    "momentum": 0.20,   # Positive 10-day ROC (not extreme)
}

# Output
OUTPUT_DIR = "output"

# Scheduler – 24-hour format, market opens at 09:30 ET; run just after open
DAILY_RUN_TIME = "09:45"
