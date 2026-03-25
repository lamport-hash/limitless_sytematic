"""
SPY / QQQ Intraday Strategy Suite
==================================
1-minute OHLCV data · Index = integer minutes since 2000-01-01 00:00 UTC

Four-call workflow per strategy
--------------------------------
    df      = load_data("spy_1min.csv", ticker="SPY")
    df      = compute_features(df)
    signals = generate_signals_<name>(df, **params)
    result  = compute_metrics(signals, df["close"])

Strategies
----------
  1. ORB              – Opening Range Breakout (configurable window)
  2. Gap & Go         – Gap continuation on first 5-min confirmation
  3. VWAP Reclaim     – VWAP crossover with volume confirmation
  4. Fade Open        – Fade an over-extended open back to VWAP
  5. First 5 Bias     – Trade pullback to OR5 in candle direction
  6. London Momentum  – Pre-market trend → enter at US open
  7. Gap Fill         – Overnight gap reversion to prev close

Metrics returned
----------------
  total_return, annual_return, sharpe_ratio, max_drawdown,
  win_rate, profit_factor, n_trades, avg_trade_pnl_pct + equity plot
"""

from __future__ import annotations

import datetime
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
import ta
import vectorbt as vbt

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
ORIGIN          = pd.Timestamp("2000-01-01", tz="UTC")
ET_TZ           = "America/New_York"
MARKET_OPEN     = datetime.time(9, 30)
MARKET_CLOSE    = datetime.time(16, 0)
PREMARKET_START = datetime.time(4, 0)
EOD_EXIT        = datetime.time(15, 55)   # forced flat before close
# London window expressed in ET (Winter: GMT = ET+5  →  8:00 GMT = 3:00 ET)
LONDON_START_ET = datetime.time(3, 0)
LONDON_END_ET   = datetime.time(4, 30)


# ──────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
def load_data(filepath: str, ticker: str = "SPY") -> pd.DataFrame:
    """
    Load 1-minute OHLCV data whose integer index is *minutes since 2000-01-01*.

    Parameters
    ----------
    filepath : str
        CSV with an integer index column and columns open/high/low/close/volume
        (case-insensitive).
    ticker   : str
        Label stored in df["ticker"].

    Returns
    -------
    pd.DataFrame with DatetimeIndex in America/New_York timezone, sorted.
    """
    df = pd.read_csv(filepath, index_col=0)
    df.columns = df.columns.str.lower().str.strip()

    ts = ORIGIN + pd.to_timedelta(df.index.astype(np.int64), unit="m")
    df.index      = ts.tz_convert(ET_TZ)
    df.index.name = "datetime"

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ticker"] = ticker
    df.sort_index(inplace=True)
    df.dropna(subset=["close"], inplace=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _time_s(df: pd.DataFrame) -> pd.Series:
    """Return a Series of datetime.time values aligned to df.index."""
    return pd.Series(df.index.time, index=df.index, name="time")


def _between(df: pd.DataFrame, start: datetime.time, end: datetime.time) -> pd.Series:
    t = _time_s(df)
    return (t >= start) & (t < end)


def _eod(df: pd.DataFrame, exit_time: datetime.time = EOD_EXIT) -> pd.Series:
    """True from exit_time onward (forces flat before close)."""
    return _time_s(df) >= exit_time


def _daily_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-min bars to daily OHLCV using market-hours bars only."""
    mh = df[_between(df, MARKET_OPEN, MARKET_CLOSE)]
    return (
        mh.resample("1D")
          .agg(open=("open", "first"), high=("high", "max"),
               low=("low", "min"),   close=("close", "last"),
               volume=("volume", "sum"))
          .dropna(how="all")
    )


def _vwap_daily_reset(df: pd.DataFrame) -> pd.Series:
    """Intraday VWAP that resets at midnight every calendar day."""
    tp      = (df["high"] + df["low"] + df["close"]) / 3.0
    tpvol   = tp * df["volume"]
    dates   = pd.Series(df.index.date, index=df.index)
    return tpvol.groupby(dates).cumsum() / df["volume"].groupby(dates).cumsum()


def _opening_range(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """
    For every trading day, compute OHLC of the first *minutes* bars after 09:30.
    Returns a DataFrame indexed by date objects.
    """
    end_h, end_m = divmod(9 * 60 + 30 + minutes, 60)
    or_end  = datetime.time(end_h, end_m)
    window  = df[_between(df, MARKET_OPEN, or_end)].copy()
    window["date"] = window.index.date
    return window.groupby("date").agg(
        or_open=("open", "first"), or_high=("high", "max"),
        or_low=("low",  "min"),   or_close=("close", "last"),
    )


def _map_daily(df: pd.DataFrame, daily_s: pd.Series, name: str) -> pd.Series:
    """
    Broadcast a date-indexed daily Series to every minute bar.
    daily_s must have a DatetimeIndex (e.g. resample("1D") output).
    """
    lookup = {dt.date(): v for dt, v in daily_s.items() if pd.notna(v)}
    dates  = pd.Series(df.index.date, index=df.index)
    return dates.map(lookup).rename(name)


def _sig_dict(name: str, el: pd.Series, xl: pd.Series,
              es: pd.Series, xs: pd.Series) -> Dict:
    """Package signal arrays into the standard dict consumed by compute_metrics."""
    return dict(name=name,
                entries_long=el.fillna(False),   exits_long=xl.fillna(False),
                entries_short=es.fillna(False), exits_short=xs.fillna(False))


# ──────────────────────────────────────────────────────────────────────────────
# 2. COMPUTE FEATURES
# ──────────────────────────────────────────────────────────────────────────────
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich the 1-min DataFrame with every feature required by the strategies.

    New columns added
    -----------------
    Time helpers
        time, date, is_market_hours, is_premarket, is_london_session

    Trend / momentum (computed on the full series including extended hours)
        ema9, ema20, sma50, rsi14, atr14
        bb_upper, bb_lower, bb_mid

    Volume
        vol_sma20  – 20-bar rolling mean
        vol_ratio  – volume / vol_sma20

    Intraday
        vwap       – daily-reset VWAP
        day_open   – price at the 09:30 bar

    Daily context (broadcast to every minute of the day)
        prev_close, prev_high, prev_low
        gap_pct    – (09:30 open − prev_close) / prev_close × 100
        pm_high, pm_low – pre-market session extremes

    Opening ranges (broadcast to every minute of the day)
        or5_open/high/low/close   – first 5 minutes
        or15_open/high/low/close  – first 15 minutes
        or30_open/high/low/close  – first 30 minutes
    """
    df = df.copy()

    # ── time helpers ──────────────────────────────────────────────────────────
    df["time"]               = _time_s(df)
    df["date"]               = pd.Series(df.index.date, index=df.index)
    df["is_market_hours"]    = _between(df, MARKET_OPEN, MARKET_CLOSE)
    df["is_premarket"]       = _between(df, PREMARKET_START, MARKET_OPEN)
    df["is_london_session"]  = _between(df, LONDON_START_ET, LONDON_END_ET)

    # ── indicators ────────────────────────────────────────────────────────────
    df["ema9"]     = ta.trend.ema_indicator(df["close"], window=9)
    df["ema20"]    = ta.trend.ema_indicator(df["close"], window=20)
    df["sma50"]    = ta.trend.sma_indicator(df["close"], window=50)
    df["rsi14"]    = ta.momentum.rsi(df["close"], window=14)
    df["atr14"]    = ta.volatility.average_true_range(
                        df["high"], df["low"], df["close"], window=14)
    _bb = ta.volatility.BollingerBands(df["close"], window=20)
    df["bb_upper"] = _bb.bollinger_hband()
    df["bb_lower"] = _bb.bollinger_lband()
    df["bb_mid"]   = _bb.bollinger_mavg()

    # ── volume ────────────────────────────────────────────────────────────────
    df["vol_sma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = (df["volume"] / df["vol_sma20"]).replace([np.inf, -np.inf], np.nan)

    # ── intraday VWAP ─────────────────────────────────────────────────────────
    df["vwap"] = _vwap_daily_reset(df)

    # ── day open price ────────────────────────────────────────────────────────
    open_bars      = df.loc[df["time"] == MARKET_OPEN, "open"]
    day_open_lkp   = {dt.date(): v for dt, v in open_bars.items()}
    df["day_open"] = df["date"].map(day_open_lkp)

    # ── previous-day OHLC ────────────────────────────────────────────────────
    daily = _daily_ohlcv(df)
    for src, dst in (("close", "prev_close"),
                     ("high",  "prev_high"),
                     ("low",   "prev_low")):
        df[dst] = _map_daily(df, daily[src].shift(1), dst)

    # ── gap % (fixed per day, based on 09:30 open vs prev close) ─────────────
    first_bar_idx = df.index[df["time"] == MARKET_OPEN]
    gap_raw       = pd.Series(np.nan, index=df.index)
    gap_raw.loc[first_bar_idx] = (
        (df.loc[first_bar_idx, "open"] - df.loc[first_bar_idx, "prev_close"])
        / df.loc[first_bar_idx, "prev_close"] * 100
    )
    df["gap_pct"] = gap_raw.groupby(df["date"]).transform("first")

    # ── pre-market range ──────────────────────────────────────────────────────
    pm = (df[df["is_premarket"]]
            .groupby("date")
            .agg(pm_high=("high", "max"), pm_low=("low", "min")))
    df["pm_high"] = df["date"].map(pm["pm_high"].to_dict())
    df["pm_low"]  = df["date"].map(pm["pm_low"].to_dict())

    # ── opening ranges (5, 15, 30 min) ───────────────────────────────────────
    for n in (5, 15, 30):
        or_df = _opening_range(df, minutes=n)
        for raw_col in ("or_open", "or_high", "or_low", "or_close"):
            out_col        = f"or{n}_{raw_col[3:]}"   # e.g. or15_high
            df[out_col]    = df["date"].map(or_df[raw_col].to_dict())

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3-A. Opening Range Breakout (ORB)
# ──────────────────────────────────────────────────────────────────────────────
def generate_signals_orb(
    df: pd.DataFrame,
    or_minutes:    int            = 15,
    vol_filter:    bool           = True,
    entry_cutoff:  datetime.time  = datetime.time(14, 0),
) -> Dict:
    """
    Opening Range Breakout
    ----------------------
    Define the first *or_minutes* after 09:30 as the OR.
    • Long  : 1-min close breaks ABOVE or_high with elevated volume
    • Short : 1-min close breaks BELOW or_low  with elevated volume

    Filters
    -------
    vol_filter   : vol_ratio > 1.2 on the breakout bar
    entry_cutoff : no new entries after this time (default 14:00)

    Exits
    -----
    Long  : close < or_low  OR forced flat at 15:55
    Short : close > or_high OR forced flat at 15:55

    Supported or_minutes: 5, 15, 30 (pre-computed in compute_features).
    Any other value triggers an on-the-fly OR computation.
    """
    if or_minutes in (5, 15, 30):
        or_high = df[f"or{or_minutes}_high"]
        or_low  = df[f"or{or_minutes}_low"]
    else:
        or_df   = _opening_range(df, minutes=or_minutes)
        or_high = df["date"].map(or_df["or_high"].to_dict())
        or_low  = df["date"].map(or_df["or_low"].to_dict())

    end_h, end_m = divmod(9 * 60 + 30 + or_minutes, 60)
    or_end       = datetime.time(end_h, end_m)
    t            = _time_s(df)
    in_window    = (t >= or_end) & (t <= entry_cutoff) & df["is_market_hours"]

    prev_c       = df["close"].shift(1)
    long_brk     = (prev_c <= or_high) & (df["close"] > or_high)
    short_brk    = (prev_c >= or_low)  & (df["close"] < or_low)

    if vol_filter:
        vf        = df["vol_ratio"].fillna(0) > 1.2
        long_brk  &= vf
        short_brk &= vf

    eod = _eod(df)
    return _sig_dict(
        f"ORB_{or_minutes}min",
        el = (long_brk  & in_window),
        xl = (df["close"] < or_low)  | eod,
        es = (short_brk & in_window),
        xs = (df["close"] > or_high) | eod,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3-B. Gap & Go
# ──────────────────────────────────────────────────────────────────────────────
def generate_signals_gap_and_go(
    df: pd.DataFrame,
    min_gap_pct: float = 0.20,
    vol_mult:    float = 1.50,
) -> Dict:
    """
    Gap & Go Continuation
    ----------------------
    After a gap-open, wait for the first 5-min candle to confirm direction,
    then enter at 09:35 if volume is elevated.

    • Long  : gap_pct ≥ +min_gap_pct  AND  first-5min candle is bullish
              AND  vol_ratio > vol_mult at 09:35
    • Short : gap_pct ≤ -min_gap_pct  AND  first-5min candle is bearish
              AND  vol_ratio > vol_mult at 09:35

    Entry  : exactly the 09:35 bar
    Exit   : close crosses back through day_open  OR  EOD
    """
    t        = _time_s(df)
    at_935   = t == datetime.time(9, 35)
    big_vol  = df["vol_ratio"].fillna(0) > vol_mult
    bull5    = df["or5_close"] > df["or5_open"]
    bear5    = df["or5_close"] < df["or5_open"]
    eod      = _eod(df)

    return _sig_dict(
        "Gap_and_Go",
        el = at_935 & (df["gap_pct"] >=  min_gap_pct) & bull5 & big_vol,
        xl = (df["close"] < df["day_open"]) | eod,
        es = at_935 & (df["gap_pct"] <= -min_gap_pct) & bear5 & big_vol,
        xs = (df["close"] > df["day_open"]) | eod,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3-C. VWAP Reclaim / Rejection
# ──────────────────────────────────────────────────────────────────────────────
def generate_signals_vwap_reclaim(
    df: pd.DataFrame,
    vol_filter:  bool          = True,
    entry_cutoff: datetime.time = datetime.time(15, 0),
) -> Dict:
    """
    VWAP Reclaim / Rejection
    -------------------------
    • Long  (reclaim) : prev bar below VWAP  →  curr bar closes above VWAP
    • Short (reject)  : prev bar above VWAP  →  curr bar closes below VWAP

    Volume filter : vol_ratio > 1.0
    Entry window  : 09:30 – entry_cutoff
    Exit          : price crosses VWAP again  OR  EOD
    """
    t   = _time_s(df)
    mh  = df["is_market_hours"] & (t <= entry_cutoff)

    pc  = df["close"].shift(1)
    pv  = df["vwap"].shift(1)

    long_x  = (pc < pv) & (df["close"] > df["vwap"])
    short_x = (pc > pv) & (df["close"] < df["vwap"])

    if vol_filter:
        vf      = df["vol_ratio"].fillna(0) > 1.0
        long_x  &= vf
        short_x &= vf

    eod = _eod(df)
    return _sig_dict(
        "VWAP_Reclaim",
        el = long_x  & mh,
        xl = (df["close"] < df["vwap"]) | eod,
        es = short_x & mh,
        xs = (df["close"] > df["vwap"]) | eod,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3-D. 9:30 Open Fade (Reversal)
# ──────────────────────────────────────────────────────────────────────────────
def generate_signals_fade_open(
    df: pd.DataFrame,
    min_gap_pct:       float = 0.50,
    rsi_overbought:    float = 65.0,
    rsi_oversold:      float = 35.0,
    fade_window_mins:  int   = 10,
) -> Dict:
    """
    9:30 Open Fade
    --------------
    When the market opens into an over-extended condition, fade it back toward VWAP.

    • Short fade : gap_pct ≥ +min_gap_pct  AND  RSI > rsi_overbought
                   AND  current bar is a red (bearish) 1-min candle
    • Long  fade : gap_pct ≤ -min_gap_pct  AND  RSI < rsi_oversold
                   AND  current bar is a green (bullish) 1-min candle

    Entry window : 09:30 – 09:30 + fade_window_mins
    Exit         : close reaches VWAP  OR  EOD
    """
    t      = _time_s(df)
    end_h, end_m = divmod(9 * 60 + 30 + fade_window_mins, 60)
    fade_end     = datetime.time(end_h, end_m)
    in_win       = _between(df, MARKET_OPEN, fade_end) & df["is_market_hours"]

    bull_bar = df["close"] > df["open"]   # green 1-min candle
    bear_bar = df["close"] < df["open"]   # red   1-min candle
    eod      = _eod(df)

    return _sig_dict(
        "Fade_Open",
        el = (df["gap_pct"] <= -min_gap_pct) & (df["rsi14"] < rsi_oversold)  & bull_bar & in_win,
        xl = (df["close"] >= df["vwap"]) | eod,
        es = (df["gap_pct"] >=  min_gap_pct) & (df["rsi14"] > rsi_overbought) & bear_bar & in_win,
        xs = (df["close"] <= df["vwap"]) | eod,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3-E. First 5-Min Candle Directional Bias
# ──────────────────────────────────────────────────────────────────────────────
def generate_signals_first5_bias(
    df: pd.DataFrame,
    pullback_pct:  float         = 0.30,
    entry_cutoff:  datetime.time = datetime.time(11, 0),
) -> Dict:
    """
    First 5-Min Candle Directional Bias
    -------------------------------------
    Use the OR5 candle direction as a session bias, then enter on a
    small pullback to the OR5 boundary.

    • Long  : OR5 bullish  →  price pulls back to within pullback_pct% of or5_high
    • Short : OR5 bearish  →  price bounces  to within pullback_pct% of or5_low

    Entry window : 09:35 – entry_cutoff
    Exit         : price breaks opposite OR5 boundary  OR  EOD
    """
    t      = _time_s(df)
    in_win = (t >= datetime.time(9, 35)) & (t <= entry_cutoff) & df["is_market_hours"]

    bull5 = df["or5_close"] > df["or5_open"]
    bear5 = df["or5_close"] < df["or5_open"]

    band  = pullback_pct / 100.0
    near_high = (
        (df["close"] >= df["or5_high"] * (1.0 - band)) &
        (df["close"] <= df["or5_high"] * (1.0 + band))
    )
    near_low  = (
        (df["close"] >= df["or5_low"] * (1.0 - band)) &
        (df["close"] <= df["or5_low"] * (1.0 + band))
    )

    eod = _eod(df)
    return _sig_dict(
        "First5_Bias",
        el = bull5 & near_high & in_win,
        xl = (df["close"] < df["or5_low"])  | eod,
        es = bear5 & near_low  & in_win,
        xs = (df["close"] > df["or5_high"]) | eod,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3-F. London / Pre-market Momentum
# ──────────────────────────────────────────────────────────────────────────────
def generate_signals_london_momentum(
    df: pd.DataFrame,
    min_move_pct: float = 0.15,
    vol_mult:     float = 1.30,
) -> Dict:
    """
    Pre-market / London Open Momentum
    -----------------------------------
    If SPY/QQQ trends strongly during the London window (03:00–04:30 ET),
    ride the momentum by entering at the US open (09:30).

    Momentum condition
    ------------------
      |close − open| / open × 100 ≥ min_move_pct  during the London session
      AND total London-session volume > vol_mult × avg per-bar volume

    • Long  : London session closes higher
    • Short : London session closes lower

    Entry : the 09:30 bar
    Exit  : price reaches prev_high (long) / prev_low (short)  OR  EOD
    """
    lon = df[df["is_london_session"]].copy()
    lon["date"] = lon.index.date

    lon_stats = lon.groupby("date").agg(
        lon_open    = ("open",     "first"),
        lon_close   = ("close",    "last"),
        lon_totvol  = ("volume",   "sum"),
        lon_avgvol  = ("vol_sma20","mean"),
    )
    lon_stats["lon_move_pct"] = (
        (lon_stats["lon_close"] - lon_stats["lon_open"])
        / lon_stats["lon_open"].replace(0, np.nan) * 100
    )
    lon_stats["lon_vol_ok"] = (
        lon_stats["lon_totvol"] > vol_mult * lon_stats["lon_avgvol"]
    )

    date_col = pd.Series(df.index.date, index=df.index)
    df_w     = df.copy()
    df_w["_lon_move"]   = date_col.map(lon_stats["lon_move_pct"].to_dict())
    df_w["_lon_vol_ok"] = date_col.map(lon_stats["lon_vol_ok"].to_dict())

    t       = _time_s(df_w)
    at_open = t == MARKET_OPEN
    vol_ok  = df_w["_lon_vol_ok"].fillna(False)
    eod     = _eod(df_w)

    return _sig_dict(
        "London_Momentum",
        el = at_open & (df_w["_lon_move"] >=  min_move_pct) & vol_ok,
        xl = (df_w["close"] > df_w["prev_high"]) | eod,
        es = at_open & (df_w["_lon_move"] <= -min_move_pct) & vol_ok,
        xs = (df_w["close"] < df_w["prev_low"])  | eod,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3-G. Overnight Gap Fill
# ──────────────────────────────────────────────────────────────────────────────
def generate_signals_gap_fill(
    df: pd.DataFrame,
    min_gap_pct: float = 0.10,
    max_gap_pct: float = 1.50,
) -> Dict:
    """
    Overnight Gap Fill Reversion
    -----------------------------
    Fade an overnight gap, expecting price to revert to the previous close.

    • Long  : gap_pct ≤ -min_gap_pct  (gap down → buy dip)
    • Short : gap_pct ≥ +min_gap_pct  (gap up   → sell rip)

    Gap size is bounded (min_gap_pct … max_gap_pct) to avoid
    trading into genuine news-driven dislocations.

    Entry : the 09:30 bar
    Exit  : close reaches prev_close (gap filled)  OR  EOD
    """
    t       = _time_s(df)
    at_open = t == MARKET_OPEN
    gap_abs = df["gap_pct"].abs()
    gap_ok  = (gap_abs >= min_gap_pct) & (gap_abs <= max_gap_pct)
    eod     = _eod(df)

    return _sig_dict(
        "Gap_Fill",
        el = at_open & (df["gap_pct"] <= -min_gap_pct) & gap_ok,
        xl = (df["close"] >= df["prev_close"]) | eod,
        es = at_open & (df["gap_pct"] >=  min_gap_pct) & gap_ok,
        xs = (df["close"] <= df["prev_close"]) | eod,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 4. COMPUTE METRICS
# ──────────────────────────────────────────────────────────────────────────────
def compute_metrics(
    signals:   Dict,
    close:     pd.Series,
    name:      Optional[str] = None,
    init_cash: float  = 10_000.0,
    fees:      float  = 0.001,       # 0.1 % per side (round-trip = 0.2 %)
    sl_stop:   float  = 0.010,       # 1 % hard stop-loss
    tp_stop:   Optional[float] = None,  # e.g. 0.02 = 2 % take-profit
    size:      float  = 0.95,        # fraction of cash per trade
    plot:      bool   = True,
) -> Dict:
    """
    Run a vectorbt backtest and return a rich metrics dictionary.

    Parameters
    ----------
    signals   : dict from any generate_signals_* function
    close     : 1-min close price series (df["close"])
    name      : strategy label (defaults to signals["name"])
    init_cash : starting capital in $
    fees      : commission as a fraction of trade value (per side)
    sl_stop   : stop-loss fraction  (0.01 = 1 %)
    tp_stop   : take-profit fraction (None = disabled)
    size      : position size as fraction of available cash
    plot      : show equity curve + drawdown chart

    Returns
    -------
    dict with keys
        total_return, annual_return, sharpe_ratio, max_drawdown,
        win_rate, profit_factor, n_trades, avg_trade_pnl_pct, portfolio
    """
    strat_name = name or signals.get("name", "Strategy")

    def _align(s: pd.Series) -> pd.Series:
        return s.reindex(close.index, fill_value=False).astype(bool)

    el = _align(signals["entries_long"])
    xl = _align(signals["exits_long"])
    es = _align(signals["entries_short"])
    xs = _align(signals["exits_short"])

    # Resolve long/short conflicts on the same bar (long wins)
    conflict = el & es
    es       = es & ~conflict

    # ── build portfolio ───────────────────────────────────────────────────────
    pf_kw = dict(
        close         = close,
        entries       = el,
        exits         = xl,
        short_entries = es,
        short_exits   = xs,
        init_cash     = init_cash,
        fees          = fees,
        size          = size,
        size_type     = "percent",
        sl_stop       = sl_stop,
        freq          = "1min",
        upon_opposite_entry = "close",
    )
    if tp_stop is not None:
        pf_kw["tp_stop"] = tp_stop

    pf = vbt.Portfolio.from_signals(**pf_kw)

    # ── extract stats ─────────────────────────────────────────────────────────
    trades   = pf.trades.records_readable
    n        = len(trades)

    if n == 0:
        print(f"\n[{strat_name}]  ⚠  No trades generated – check filters / data coverage.")
        return dict(total_return=0.0, annual_return=np.nan, sharpe_ratio=np.nan,
                    max_drawdown=0.0, win_rate=np.nan, profit_factor=np.nan,
                    n_trades=0, avg_trade_pnl_pct=np.nan, portfolio=pf)

    wins          = trades.loc[trades["PnL"] > 0, "PnL"]
    losses        = trades.loc[trades["PnL"] < 0, "PnL"].abs()
    win_rate      = len(wins) / n * 100
    profit_factor = wins.sum() / losses.sum() if len(losses) else np.inf

    total_ret  = pf.total_return() * 100
    try:
        ann_ret = pf.annualized_return() * 100
    except Exception:
        ann_ret = np.nan

    sharpe = pf.sharpe_ratio()
    max_dd = pf.max_drawdown() * 100

    ret_col     = "Return" if "Return" in trades.columns else None
    avg_pnl_pct = trades[ret_col].mean() * 100 if ret_col else np.nan

    # ── console summary ───────────────────────────────────────────────────────
    bar = "═" * 54
    print(f"\n{bar}")
    print(f"  Strategy       : {strat_name}")
    print(f"  Period         : {close.index[0].date()} → {close.index[-1].date()}")
    print(f"  Trades         : {n}")
    print(f"  Win Rate       : {win_rate:.1f} %")
    print(f"  Profit Factor  : {profit_factor:.2f}")
    print(f"  Sharpe Ratio   : {sharpe:.2f}")
    print(f"  Total Return   : {total_ret:.2f} %")
    print(f"  Annual Return  : {ann_ret:.2f} %")
    print(f"  Max Drawdown   : {max_dd:.2f} %")
    print(f"  Avg Trade P&L  : {avg_pnl_pct:.3f} %")
    print(f"{bar}\n")

    # ── plot ──────────────────────────────────────────────────────────────────
    if plot:
        try:
            pf.plot(subplots=["cum_returns", "drawdowns", "trades"]).show()
        except Exception:
            pf.plot().show()

    return dict(
        total_return      = total_ret,
        annual_return     = ann_ret,
        sharpe_ratio      = sharpe,
        max_drawdown      = max_dd,
        win_rate          = win_rate,
        profit_factor     = profit_factor,
        n_trades          = n,
        avg_trade_pnl_pct = avg_pnl_pct,
        portfolio         = pf,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 5. BATCH RUNNER  – run every strategy in one call
# ──────────────────────────────────────────────────────────────────────────────
ALL_STRATEGIES: Dict[str, callable] = {
    "orb_5min"         : lambda df: generate_signals_orb(df, or_minutes=5),
    "orb_15min"        : lambda df: generate_signals_orb(df, or_minutes=15),
    "orb_30min"        : lambda df: generate_signals_orb(df, or_minutes=30),
    "gap_and_go"       : generate_signals_gap_and_go,
    "vwap_reclaim"     : generate_signals_vwap_reclaim,
    "fade_open"        : generate_signals_fade_open,
    "first5_bias"      : generate_signals_first5_bias,
    "london_momentum"  : generate_signals_london_momentum,
    "gap_fill"         : generate_signals_gap_fill,
}


def run_all_strategies(
    df: pd.DataFrame,
    strategies: Optional[list] = None,
    **metrics_kwargs,
) -> Dict[str, Dict]:
    """
    Run every strategy (or a subset) and return a dict of metric dicts.

    Parameters
    ----------
    df         : feature-enriched DataFrame from compute_features()
    strategies : list of strategy keys from ALL_STRATEGIES to run;
                 None = run all
    **metrics_kwargs : forwarded to compute_metrics()
                       e.g. sl_stop=0.01, tp_stop=0.02, plot=False

    Returns
    -------
    dict  {strategy_name: metrics_dict}

    Example
    -------
        df      = load_data("spy_1min.csv", "SPY")
        df      = compute_features(df)
        results = run_all_strategies(df, sl_stop=0.01, tp_stop=0.02, plot=False)
        # Show a ranked summary
        summary = compare_strategies(results)
    """
    keys    = strategies or list(ALL_STRATEGIES.keys())
    close   = df["close"]
    results = {}

    for key in keys:
        print(f"\n{'▶' * 3}  {key}  {'◀' * 3}")
        try:
            sig             = ALL_STRATEGIES[key](df)
            results[key]    = compute_metrics(sig, close, name=key, **metrics_kwargs)
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            results[key]    = {"error": str(exc)}

    return results


def compare_strategies(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Build a ranked summary DataFrame from the output of run_all_strategies().

    Ranks strategies by Sharpe ratio (descending).
    """
    rows = []
    for name, r in results.items():
        if "error" in r:
            continue
        rows.append({
            "strategy"       : name,
            "n_trades"       : r["n_trades"],
            "win_rate_%"     : round(r["win_rate"],        1),
            "profit_factor"  : round(r["profit_factor"],   2),
            "sharpe"         : round(r["sharpe_ratio"],     2),
            "total_ret_%"    : round(r["total_return"],     2),
            "annual_ret_%"   : round(r["annual_return"],    2),
            "max_dd_%"       : round(r["max_drawdown"],     2),
            "avg_trade_%"    : round(r["avg_trade_pnl_pct"],3),
        })

    df_out = pd.DataFrame(rows).set_index("strategy")
    return df_out.sort_values("sharpe", ascending=False)


# ──────────────────────────────────────────────────────────────────────────────
# EXAMPLE USAGE (run as script or in a notebook)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Step 1 : Load ─────────────────────────────────────────────────────────
    # df = load_data("spy_1min.csv", ticker="SPY")

    # ── Step 2 : Features ─────────────────────────────────────────────────────
    # df = compute_features(df)

    # ── Step 3a : Single strategy (4-call workflow) ───────────────────────────
    # signals = generate_signals_orb(df, or_minutes=15, vol_filter=True)
    # result  = compute_metrics(signals, df["close"], sl_stop=0.01, tp_stop=0.02)

    # ── Step 3b : Run all strategies at once ──────────────────────────────────
    # results = run_all_strategies(df, sl_stop=0.01, tp_stop=0.02, plot=False)
    # print(compare_strategies(results).to_string())

    # ── Quick smoke-test with synthetic data ──────────────────────────────────
    print("Generating synthetic 1-min data for smoke-test …")
    np.random.seed(42)
    n_days   = 252
    n_bars   = n_days * 390          # 390 market-hours bars per day
    base     = pd.Timestamp("2020-01-02 09:30", tz="America/New_York")
    idx      = pd.bdate_range(base, periods=n_bars, freq="1min", tz="America/New_York")

    close_px = 400 + np.cumsum(np.random.randn(n_bars) * 0.05)
    synth    = pd.DataFrame({
        "open"  : close_px * (1 + np.random.randn(n_bars) * 0.001),
        "high"  : close_px * (1 + np.abs(np.random.randn(n_bars)) * 0.002),
        "low"   : close_px * (1 - np.abs(np.random.randn(n_bars)) * 0.002),
        "close" : close_px,
        "volume": np.random.randint(500_000, 3_000_000, n_bars).astype(float),
    }, index=idx)

    print("Computing features …")
    synth = compute_features(synth)

    print("\nRunning ORB-15 …")
    sigs   = generate_signals_orb(synth, or_minutes=15)
    result = compute_metrics(sigs, synth["close"], plot=False)

    print("\nRunning all strategies …")
    results = run_all_strategies(synth, plot=False)
    print("\n── Strategy Comparison ──────────────────────────────")
    print(compare_strategies(results).to_string())
