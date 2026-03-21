"""
MACD EMA Trend Strategy Implementation.

Implements a trend-following strategy combining:
1. EMA200 trend filter (price must be consistently above/below EMA)
2. MACD signal crossovers with histogram confirmation
3. Multiple exit strategies (fixed SL/TP, trailing ATR, breakeven)
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
from typing import Optional


EMA_LEN = 200
BACKCANDLES_PREV = 5
SW_WINDOW = 4
RR = 1.0


def ema_trend_signal(df: pd.DataFrame, i: int, backcandles_prev: int = BACKCANDLES_PREV) -> int:
    """
    EMA trend signal based on consecutive candles above/below EMA.
    
    Args:
        df: DataFrame with OHLC and EMA column
        i: Current index position
        backcandles_prev: Number of previous candles to check
        
    Returns:
        1: Uptrend (all candles have Open>EMA and Close>EMA)
        -1: Downtrend (all candles have Open<EMA and Close<EMA)
        0: No clear trend
    """
    if i < backcandles_prev:
        return 0
    if np.isnan(df["EMA200"].iloc[i]):
        return 0

    start = i - backcandles_prev
    seg = df.iloc[start:i+1]
    up = ((seg["Open"] > seg["EMA200"]) & (seg["Close"] > seg["EMA200"])).all()
    down = ((seg["Open"] < seg["EMA200"]) & (seg["Close"] < seg["EMA200"])).all()
    return 1 if up else (-1 if down else 0)


def build_features(df: pd.DataFrame, ema_len: int = EMA_LEN, long_only: bool = False) -> pd.DataFrame:
    """
    Build MACD and EMA features for the strategy.
    
    Args:
        df: DataFrame with OHLCV columns (Open, High, Low, Close, Volume)
        ema_len: EMA period length
        long_only: If True, filter out short signals (only long positions)
        
    Returns:
        DataFrame with added indicator columns
    """
    out = df.copy()

    out["EMA200"] = ta.ema(out["Close"], length=ema_len)
    macd = ta.macd(out["Close"], fast=12, slow=26, signal=9)
    out = out.join(macd)

    out["ema_signal"] = 0
    out["ema_signal"] = [ema_trend_signal(out, i, BACKCANDLES_PREV) for i in range(len(out))]

    macd_line = out["MACD_12_26_9"]
    macd_sig = out["MACDs_12_26_9"]
    macd_hist = out["MACDh_12_26_9"]
    macd_line_prev = macd_line.shift(1)
    macd_sig_prev = macd_sig.shift(1)

    hist_thresh = 4e-6
    hist_window = 7
    hist_below_win = macd_hist.rolling(hist_window, min_periods=hist_window).min() < -hist_thresh
    hist_above_win = macd_hist.rolling(hist_window, min_periods=hist_window).max() > hist_thresh

    bull_cross_below0 = (
        hist_below_win &
        (macd_line_prev <= macd_sig_prev) &
        (macd_line > macd_sig) &
        (macd_line < 0) & (macd_sig < 0)
    )

    bear_cross_above0 = (
        hist_above_win &
        (macd_line_prev >= macd_sig_prev) &
        (macd_line < macd_sig) &
        (macd_line > 0) & (macd_sig > 0)
    )

    out["MACD_signal"] = 0
    out.loc[bull_cross_below0, "MACD_signal"] = 1
    if not long_only:
        out.loc[bear_cross_above0, "MACD_signal"] = -1

    out["pre_signal"] = 0
    out.loc[(out["ema_signal"] == 1) & (out["MACD_signal"] == 1), "pre_signal"] = 1
    if not long_only:
        out.loc[(out["ema_signal"] == -1) & (out["MACD_signal"] == -1), "pre_signal"] = -1

    s = out['pre_signal'].astype(int)
    prev_nz = s.replace(0, np.nan).ffill().shift(1)
    flip_first = (s != 0) & prev_nz.notna() & (s != prev_nz)
    out.loc[flip_first, 'pre_signal'] = 0

    if "ATR" not in out.columns:
        out["ATR"] = ta.atr(out["High"], out["Low"], out["Close"], length=14)

    out = out.dropna().copy()
    return out


class MACDEMA_SwingWindow_BE(Strategy):
    """
    MACD EMA strategy with swing window stop loss and breakeven exit.
    
    Stop loss is placed at swing low/high over a window of bars.
    Take profit is set at risk-reward ratio.
    Stop moves to breakeven when price moves 1R in favor.
    """
    rr = RR
    sw_window = SW_WINDOW

    def init(self):
        self._moved_be = False
        self._init_sl_dist = None

    def _move_to_breakeven_if_ready(self):
        if not self.trades:
            return
        tr = self.trades[0]

        if self._init_sl_dist is None and tr.sl is not None:
            self._init_sl_dist = abs(tr.entry_price - tr.sl)

        if not self._init_sl_dist or self._moved_be:
            return

        price = float(self.data.Close[-1])
        if tr.is_long and (price - tr.entry_price) >= self._init_sl_dist:
            tr.sl = tr.entry_price
            self._moved_be = True
        elif tr.is_short and (tr.entry_price - price) >= self._init_sl_dist:
            tr.sl = tr.entry_price
            self._moved_be = True

    def _enough_history(self) -> bool:
        return len(self.data.Close) >= (self.sw_window + 1)

    def next(self):
        close = float(self.data.Close[-1])
        sig = int(self.data.df["pre_signal"].iloc[-1])

        if not self.position:
            self._moved_be = False
            self._init_sl_dist = None

            if not self._enough_history():
                return

            lows = self.data.df["Low"].iloc[-(self.sw_window + 1):]
            highs = self.data.df["High"].iloc[-(self.sw_window + 1):]

            if sig == 1:
                sw_low = float(np.min(lows))
                if not np.isfinite(sw_low) or sw_low >= close:
                    return
                dist = close - sw_low
                sl = sw_low
                tp = close + self.rr * dist
                self.buy(sl=sl, tp=tp)

            elif sig == -1:
                sw_high = float(np.max(highs))
                if not np.isfinite(sw_high) or sw_high <= close:
                    return
                dist = sw_high - close
                sl = sw_high
                tp = close - self.rr * dist
                self.sell(sl=sl, tp=tp)

        else:
            self._move_to_breakeven_if_ready()


class MACDEMA_SwingWindow(Strategy):
    """
    MACD EMA strategy with swing window stop loss (no breakeven).
    
    Stop loss is placed at swing low/high over a window of bars.
    Take profit is set at risk-reward ratio.
    """
    rr = RR
    sw_window = SW_WINDOW

    def init(self):
        pass

    def _enough_history(self) -> bool:
        return len(self.data.Close) >= (self.sw_window + 1)

    def next(self):
        close = float(self.data.Close[-1])
        sig = int(self.data.df["pre_signal"].iloc[-1])

        if self.position:
            return

        if not self._enough_history():
            return

        lows = self.data.df["Low"].iloc[-(self.sw_window + 1):]
        highs = self.data.df["High"].iloc[-(self.sw_window + 1):]

        if sig == 1:
            sw_low = float(np.min(lows))
            if not np.isfinite(sw_low) or sw_low >= close:
                return
            dist = close - sw_low
            self.buy(sl=sw_low, tp=close + self.rr * dist)

        elif sig == -1:
            sw_high = float(np.max(highs))
            if not np.isfinite(sw_high) or sw_high <= close:
                return
            dist = sw_high - close
            self.sell(sl=sw_high, tp=close - self.rr * dist)


class MACDEMA_SwingOrATR(Strategy):
    """
    MACD EMA strategy using max(swing distance, ATR distance) for stop loss.
    
    Uses the larger of:
    - Swing low/high distance
    - ATR * atr_mult
    """
    rr: float = 2.0
    sw_window: int = 5
    atr_mult: float = 2.0
    atr_len: int = 14

    def init(self):
        if "ATR" not in self.data.df.columns:
            df = self.data.df
            df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=self.atr_len)

    def _enough_history(self) -> bool:
        return len(self.data.Close) >= (self.sw_window + 1)

    def next(self):
        close = float(self.data.Close[-1])
        sig = int(self.data.df["pre_signal"].iloc[-1])

        if self.position:
            return
        if not self._enough_history():
            return

        lows = self.data.df["Low"].iloc[-(self.sw_window + 1):]
        highs = self.data.df["High"].iloc[-(self.sw_window + 1):]
        atr = float(self.data.df["ATR"].iloc[-1]) if "ATR" in self.data.df.columns else float("nan")

        atr_dist = self.atr_mult * atr if np.isfinite(atr) else 0.0

        if sig == 1:
            sw_low = float(np.min(lows))
            if not np.isfinite(sw_low) or sw_low >= close:
                return
            swing_dist = close - sw_low
            sl_dist = max(swing_dist, atr_dist)
            if sl_dist <= 0:
                return
            self.buy(sl=close - sl_dist, tp=close + self.rr * sl_dist)

        elif sig == -1:
            sw_high = float(np.max(highs))
            if not np.isfinite(sw_high) or sw_high <= close:
                return
            swing_dist = sw_high - close
            sl_dist = max(swing_dist, atr_dist)
            if sl_dist <= 0:
                return
            self.sell(sl=close + sl_dist, tp=close - self.rr * sl_dist)


class MACDEMA_ATRorBand(Strategy):
    """
    MACD EMA strategy using ATR or percentage band for stop loss.
    
    Uses the larger of:
    - Band percentage (low * (1 - band_pct) for long)
    - ATR * atr_mult
    """
    rr: float = 2.0
    atr_mult: float = 2.0
    atr_len: int = 14
    band_pct: float = 0.01

    def init(self):
        if "ATR" not in self.data.df.columns:
            df = self.data.df
            df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=self.atr_len)

    def next(self):
        if self.position:
            return

        close = float(self.data.Close[-1])
        high = float(self.data.df["High"].iloc[-1])
        low = float(self.data.df["Low"].iloc[-1])
        sig = int(self.data.df["pre_signal"].iloc[-1])

        atr = float(self.data.df["ATR"].iloc[-1]) if "ATR" in self.data.df.columns else float("nan")
        atr_floor = self.atr_mult * atr if np.isfinite(atr) else 0.0

        if sig == 1:
            band_sl_price = low * (1.0 - self.band_pct)
            band_dist = max(0.0, close - band_sl_price)
            sl_dist = max(band_dist, atr_floor)
            if sl_dist <= 0:
                return
            self.buy(sl=close - sl_dist, tp=close + self.rr * sl_dist)

        elif sig == -1:
            band_sl_price = high * (1.0 + self.band_pct)
            band_dist = max(0.0, band_sl_price - close)
            sl_dist = max(band_dist, atr_floor)
            if sl_dist <= 0:
                return
            self.sell(sl=close + sl_dist, tp=close - self.rr * sl_dist)


class MACDEMA_ATRTrail(Strategy):
    """
    MACD EMA strategy with pure ATR-based trailing stop (no fixed TP).
    
    Initial stop distance = ATR * atr_mult
    Long: trail at (highest high since entry - trail_dist), only ratchet upward
    Short: trail at (lowest low since entry + trail_dist), only ratchet downward
    """
    atr_mult: float = 2.0
    atr_len: int = 14
    rr: float = 2.0

    def init(self):
        df = self.data.df
        if "ATR" not in df.columns:
            df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=self.atr_len)

        self._trail_dist = None
        self._peak = None
        self._trough = None

    def _reset_trailing_state(self):
        self._trail_dist = None
        self._peak = None
        self._trough = None

    def next(self):
        df = self.data.df
        close = float(self.data.Close[-1])
        high = float(df["High"].iloc[-1])
        low = float(df["Low"].iloc[-1])
        sig = int(df["pre_signal"].iloc[-1])

        atr = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else float("nan")

        if self.position:
            tr = self.trades[0]

            if (self._trail_dist is None) or not np.isfinite(self._trail_dist):
                self._trail_dist = (self.atr_mult * atr) if np.isfinite(atr) else 0.0

            if tr.is_long:
                self._peak = high if self._peak is None else max(self._peak, high)
                new_sl = self._peak - self._trail_dist
                if tr.sl is None:
                    tr.sl = new_sl
                else:
                    tr.sl = max(tr.sl, new_sl)
            else:
                self._trough = low if self._trough is None else min(self._trough, low)
                new_sl = self._trough + self._trail_dist
                if tr.sl is None:
                    tr.sl = new_sl
                else:
                    tr.sl = min(tr.sl, new_sl)
            return

        self._reset_trailing_state()

        if not np.isfinite(atr) or atr <= 0:
            return

        trail = self.atr_mult * atr

        if sig == 1:
            self._trail_dist = trail
            self._peak = high
            self.buy(sl=close - trail)

        elif sig == -1:
            self._trail_dist = trail
            self._trough = low
            self.sell(size=0.99, sl=close + trail)


def run_backtest(
    df: pd.DataFrame,
    strategy_class=MACDEMA_SwingOrATR,
    cash: float = 100_000,
    commission: float = 0.0002,
    **strategy_kwargs
):
    """
    Run backtest with specified strategy and parameters.
    
    Args:
        df: DataFrame with OHLCV data
        strategy_class: Strategy class to use
        cash: Initial cash
        commission: Commission per trade
        **strategy_kwargs: Strategy-specific parameters
        
    Returns:
        Backtest stats object
    """
    df_features = build_features(df)
    
    bt = Backtest(
        df_features,
        strategy_class,
        cash=cash,
        commission=commission,
        trade_on_close=False,
        hedging=False,
        exclusive_orders=False,
    )
    
    return bt.run()


def split_and_optimize(
    df: pd.DataFrame,
    strategy_class=MACDEMA_SwingOrATR,
    split_ratio: float = 0.5,
    ema_len: int = 200,
    rr_grid=None,
    sw_window_grid=None,
    maximize="Return [%]",
    cash: float = 100_000,
    commission: float = 0.0002,
):
    """
    Split data into train/test and run grid search optimization.
    
    Args:
        df: DataFrame with OHLCV data
        strategy_class: Strategy class to use
        split_ratio: Fraction of data to use for training (0.0-1.0)
        ema_len: EMA period length (configurable)
        rr_grid: List of risk-reward ratios to test
        sw_window_grid: List of swing window sizes to test
        maximize: Metric to maximize
        cash: Initial cash
        commission: Commission per trade
        
    Returns:
        Dictionary with train_stats, test_stats, best_params, heatmap
    """
    if rr_grid is None:
        rr_grid = [1.0, 1.5, 2.0, 2.5, 3.0]
    if sw_window_grid is None:
        sw_window_grid = list(range(3, 9))
    
    n_total = len(df)
    n_split = int(n_total * split_ratio)
    
    df_train = df.iloc[:n_split].copy()
    df_test = df.iloc[n_split:].copy()
    
    print(f"Data split: {n_split} train / {len(df_test)} test bars")
    print(f"Train: {df_train.index[0]} to {df_train.index[-1]}")
    print(f"Test: {df_test.index[0]} to {df_test.index[-1]}")
    
    df_train_features = build_features(df_train, ema_len=ema_len)
    
    bt_train = Backtest(
        df_train_features,
        strategy_class,
        cash=cash,
        commission=commission,
        trade_on_close=False,
        hedging=False,
        exclusive_orders=False,
    )
    
    print(f"\nOptimizing on training data...")
    stats_opt, heatmap = bt_train.optimize(
        rr=rr_grid,
        sw_window=sw_window_grid,
        maximize=maximize,
        return_heatmap=True,
        constraint=lambda p: p.rr > 0 and p.sw_window >= 1,
    )
    
    best_params = {
        'rr': stats_opt._strategy.rr,
        'sw_window': stats_opt._strategy.sw_window,
    }
    
    if hasattr(stats_opt._strategy, 'atr_mult'):
        best_params['atr_mult'] = stats_opt._strategy.atr_mult
    
    print(f"\nBest parameters from training:")
    for key, val in best_params.items():
        print(f"  {key}: {val}")
    
    print(f"\nEvaluating on test data with best parameters...")
    test_stats = run_backtest(
        df_test,
        strategy_class=strategy_class,
        cash=cash,
        commission=commission,
        **best_params
    )
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Train Return [%]:    {stats_opt['Return [%]']:.2f}")
    print(f"Train Sharpe:        {stats_opt['Sharpe Ratio']:.2f}")
    print(f"Train # Trades:      {stats_opt['# Trades']}")
    print(f"Train Win Rate [%]:  {stats_opt['Win Rate [%]']:.2f}")
    print(f"{'='*60}")
    print(f"Test Return [%]:     {test_stats['Return [%]']:.2f}")
    print(f"Test Sharpe:         {test_stats['Sharpe Ratio']:.2f}")
    print(f"Test # Trades:       {test_stats['# Trades']}")
    print(f"Test Win Rate [%]:   {test_stats['Win Rate [%]']:.2f}")
    print(f"Test Max DD [%]:     {test_stats['Max. Drawdown [%]']:.2f}")
    print(f"{'='*60}")
    
    return {
        'train_stats': stats_opt,
        'test_stats': test_stats,
        'best_params': best_params,
        'heatmap': heatmap,
        'df_train': df_train,
        'df_test': df_test,
    }


def optimize_backtest(
    df: pd.DataFrame,
    strategy_class=MACDEMA_SwingOrATR,
    rr_grid=None,
    sw_window_grid=None,
    maximize="Return [%]",
    **kwargs
):
    """
    Optimize strategy parameters and return heatmap.
    
    Args:
        df: DataFrame with OHLCV data
        strategy_class: Strategy class to optimize
        rr_grid: List of risk-reward ratios to test
        sw_window_grid: List of swing window sizes to test
        maximize: Metric to maximize
        **kwargs: Additional parameters
        
    Returns:
        Tuple of (stats, heatmap)
    """
    if rr_grid is None:
        rr_grid = [1.0, 1.5, 2.0, 2.5, 3.0]
    if sw_window_grid is None:
        sw_window_grid = list(range(3, 9))
    
    df_features = build_features(df)
    
    bt = Backtest(
        df_features,
        strategy_class,
        cash=100_000,
        commission=0.0002,
        trade_on_close=False,
        hedging=False,
        exclusive_orders=False,
    )
    
    stats, heatmap = bt.optimize(
        rr=rr_grid,
        sw_window=sw_window_grid,
        maximize=maximize,
        return_heatmap=True,
        constraint=lambda p: p.rr > 0 and p.sw_window >= 1,
    )
    
    return stats, heatmap
