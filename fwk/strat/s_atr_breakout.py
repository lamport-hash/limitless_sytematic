"""
ATR Open Range Breakout Strategy.

Strategy logic:
1. Calculate daily ATR from previous day's HLC
2. Set breakout levels at today's Open ± x*ATR
3. Go long above upper level, short below lower level
4. Small stop (fraction of ATR), exit EOD

Variant features:
- Overnight gap directional bias (positive gap = long bias only)
- Volatility filter (trade only when ATR > ATR_MA)
- Exit at end of day
"""

from typing import Tuple
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy

from core.data_org import get_normalised_file, ProductType
from core.enums import (
    MktDataTFreq,
    ExchangeNAME,
    g_open_col,
    g_high_col,
    g_low_col,
    g_close_col,
    g_volume_col,
    g_index_col,
)
from features.f_atr_breakout import (
    feature_atr_breakout_signal,
    feature_session_info,
)


DEFAULT_SYMBOL = "EURUSD"


def build_features(
    p_df: pd.DataFrame,
    p_atr_len: int = 14,
    p_atr_mult: float = 0.5,
    p_vol_ma_len: int = 20,
    p_use_gap_bias: bool = True,
    p_use_vol_filter: bool = True,
    p_gap_threshold: float = 0.002,
    p_direction: str = "both",
) -> pd.DataFrame:
    """
    Build ATR breakout signal features.
    
    Args:
        p_df: DataFrame with OHLCV columns
        p_atr_len: ATR calculation period
        p_atr_mult: ATR multiplier for breakout levels
        p_vol_ma_len: Volatility MA period
        p_use_gap_bias: Use gap for directional bias
        p_use_vol_filter: Only trade when ATR > ATR_MA
        p_gap_threshold: Minimum gap % for bias
        p_direction: "long", "short", or "both"
    
    Returns:
        DataFrame with added signal columns
    """
    df = p_df.copy()
    
    long_sig, short_sig, upper, lower, atr, vol_ok, gap = feature_atr_breakout_signal(
        df,
        p_atr_len=p_atr_len,
        p_atr_mult=p_atr_mult,
        p_vol_ma_len=p_vol_ma_len,
        p_use_gap_bias=p_use_gap_bias,
        p_use_vol_filter=p_use_vol_filter,
        p_gap_threshold=p_gap_threshold,
    )
    
    is_first, is_last = feature_session_info(df)
    
    df["F_atr_breakout_long_f16"] = long_sig.astype(np.float32)
    df["F_atr_breakout_short_f16"] = short_sig.astype(np.float32)
    df["F_atr_breakout_upper_f32"] = upper.astype(np.float32)
    df["F_atr_breakout_lower_f32"] = lower.astype(np.float32)
    df["F_atr_breakout_atr_f32"] = atr.astype(np.float32)
    df["F_atr_breakout_vol_ok_i8"] = vol_ok.astype(np.int8)
    df["F_atr_breakout_gap_f32"] = gap.astype(np.float32)
    df["F_session_first_i8"] = is_first.astype(np.int8)
    df["F_session_last_i8"] = is_last.astype(np.int8)
    
    df["signal"] = 0
    if p_direction in ("long", "both"):
        df.loc[df["F_atr_breakout_long_f16"] == 1, "signal"] = 1
    if p_direction in ("short", "both"):
        df.loc[df["F_atr_breakout_short_f16"] == 1, "signal"] = -1
    
    return df


def convert_to_ohlcv(p_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert framework columns to standard OHLCV format for backtesting.py.
    """
    ohlcv_df = pd.DataFrame(index=p_df.index)
    
    ohlcv_df["Open"] = p_df[g_open_col]
    ohlcv_df["High"] = p_df[g_high_col]
    ohlcv_df["Low"] = p_df[g_low_col]
    ohlcv_df["Close"] = p_df[g_close_col]
    ohlcv_df["Volume"] = p_df[g_volume_col] if g_volume_col in p_df.columns else 0
    
    ohlcv_df["signal"] = p_df["signal"]
    ohlcv_df["ATR"] = p_df["F_atr_breakout_atr_f32"]
    ohlcv_df["upper_level"] = p_df["F_atr_breakout_upper_f32"]
    ohlcv_df["lower_level"] = p_df["F_atr_breakout_lower_f32"]
    ohlcv_df["is_last_bar"] = p_df["F_session_last_i8"]
    
    if g_index_col in p_df.columns:
        base = pd.Timestamp("2000-01-01")
        ohlcv_df.index = base + pd.to_timedelta(p_df[g_index_col], unit="m")
    
    ohlcv_df = ohlcv_df.dropna(subset=["Open", "High", "Low", "Close"])
    
    return ohlcv_df


class AtrBreakoutStrategy(Strategy):
    """
    ATR Open Range Breakout strategy.
    
    Parameters:
        atr_mult: ATR multiplier for breakout levels (default: 0.5)
        stop_mult: Stop loss in ATR units (default: 0.3)
        exit_eod: Exit position at end of day (default: True)
        direction: "long", "short", or "both"
        use_gap_bias: Use overnight gap for directional bias
        use_vol_filter: Only trade when volatility elevated
    """
    
    atr_mult: float = 0.5
    stop_mult: float = 0.3
    exit_eod: bool = True
    direction: str = "both"
    use_gap_bias: bool = True
    use_vol_filter: bool = True
    
    def init(self):
        self._entry_bar = -1
        self._entry_day = -1
    
    def next(self):
        if self.position:
            if self.exit_eod:
                is_last = int(self.data.df["is_last_bar"].iloc[-1])
                if is_last == 1:
                    self.position.close()
                    return
            return
        
        close = float(self.data.Close[-1])
        sig = int(self.data.df["signal"].iloc[-1])
        atr = float(self.data.df["ATR"].iloc[-1])
        
        if not np.isfinite(atr) or atr <= 0:
            return
        
        sl_dist = self.stop_mult * atr
        
        if sig == 1 and self.direction in ("long", "both"):
            sl = close - sl_dist
            self.buy(sl=sl)
        
        elif sig == -1 and self.direction in ("short", "both"):
            sl = close + sl_dist
            self.sell(sl=sl)


def run_backtest(
    p_df: pd.DataFrame,
    p_cash: float = 100_000,
    p_commission: float = 0.00002,
    **kwargs
) -> Tuple[dict, pd.DataFrame]:
    """
    Run backtest with ATR breakout strategy.
    
    Args:
        p_df: DataFrame with features
        p_cash: Initial cash
        p_commission: Commission per trade
        **kwargs: Strategy parameters
    
    Returns:
        Tuple of (stats, trades)
    """
    ohlcv_df = convert_to_ohlcv(p_df)
    
    bt = Backtest(
        ohlcv_df,
        AtrBreakoutStrategy,
        cash=p_cash,
        commission=p_commission,
        trade_on_close=True,
        hedging=False,
        exclusive_orders=False,
        finalize_trades=True,
    )
    
    stats = bt.run(**kwargs)
    trades = stats._trades if hasattr(stats, '_trades') else pd.DataFrame()
    
    return stats, trades


def main(
    p_df: pd.DataFrame,
    p_atr_len: int = 14,
    p_atr_mult: float = 0.5,
    p_stop_mult: float = 0.3,
    p_vol_ma_len: int = 20,
    p_gap_threshold: float = 0.002,
    p_use_gap_bias: bool = True,
    p_use_vol_filter: bool = True,
    p_direction: str = "both",
    p_exit_eod: bool = True,
    p_cash: float = 100_000,
    p_commission: float = 0.00002,
    p_verbose: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Main entry point: Build features, run backtest.
    """
    if p_verbose:
        print("=" * 80)
        print("ATR Open Range Breakout Strategy")
        print("=" * 80)
    
    df = p_df
    
    if p_verbose:
        print(f"\n2. Building features...")
        print(f"   ATR length: {p_atr_len}")
        print(f"   ATR multiplier: {p_atr_mult}")
        print(f"   Stop multiplier: {p_stop_mult}")
        print(f"   Gap bias: {p_use_gap_bias}")
        print(f"   Vol filter: {p_use_vol_filter}")
    
    df = build_features(
        df,
        p_atr_len=p_atr_len,
        p_atr_mult=p_atr_mult,
        p_vol_ma_len=p_vol_ma_len,
        p_use_gap_bias=p_use_gap_bias,
        p_use_vol_filter=p_use_vol_filter,
        p_gap_threshold=p_gap_threshold,
        p_direction=p_direction,
    )
    
    n_long = df["F_atr_breakout_long_f16"].sum()
    n_short = df["F_atr_breakout_short_f16"].sum()
    n_vol_filtered = (df["F_atr_breakout_vol_ok_i8"] == 1).sum()
    
    if p_verbose:
        print(f"   Long signals: {n_long:.0f}")
        print(f"   Short signals: {n_short:.0f}")
        print(f"   Bars with vol OK: {n_vol_filtered:,}")
    
    if p_verbose:
        print(f"\n3. Running backtest...")
    
    stats, trades = run_backtest(
        df,
        p_cash=p_cash,
        p_commission=p_commission,
        atr_mult=p_atr_mult,
        stop_mult=p_stop_mult,
        direction=p_direction,
        exit_eod=p_exit_eod,
    )
    
    if p_verbose:
        _print_results(stats, trades, p_direction, p_atr_mult, p_stop_mult, p_atr_len)
    
    return df, stats


def _print_results(stats, trades, direction, atr_mult, stop_mult, atr_len):
    """Print backtest results."""
    strategy_return = stats['Return [%]']
    buy_hold_return = stats['Buy & Hold Return [%]']
    outperformance = strategy_return - buy_hold_return
    
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    
    print("\n--- PERFORMANCE ---")
    print(f"Start:                   {stats['Start']}")
    print(f"End:                     {stats['End']}")
    print(f"Duration:                {stats['Duration']}")
    print(f"Return [%]:              {strategy_return:.2f}")
    print(f"Buy & Hold [%]:          {buy_hold_return:.2f}")
    print(f"Outperformance [%]:      {outperformance:+.2f}")
    print(f"Max Drawdown [%]:        {stats['Max. Drawdown [%]']:.2f}")
    
    print("\n--- RISK METRICS ---")
    print(f"Sharpe Ratio:            {stats['Sharpe Ratio']:.2f}")
    print(f"Sortino Ratio:           {stats['Sortino Ratio']:.2f}")
    
    print("\n--- TRADE STATS ---")
    print(f"# Trades:                {stats['# Trades']}")
    print(f"Win Rate [%]:            {stats['Win Rate [%]']:.2f}")
    print(f"Profit Factor:           {stats['Profit Factor']:.2f}")
    print(f"Best Trade [%]:          {stats['Best Trade [%]']:.2f}")
    print(f"Worst Trade [%]:         {stats['Worst Trade [%]']:.2f}")
    print(f"Avg Trade [%]:           {stats['Avg. Trade [%]']:.2f}")
    
    print("\n--- PARAMETERS ---")
    print(f"Direction:               {direction}")
    print(f"ATR mult:                {atr_mult}")
    print(f"Stop mult:               {stop_mult}")
    print(f"ATR length:              {atr_len}")
    
    print("\n" + "=" * 80)
    if outperformance > 0:
        print(f">>> OUTPERFORMED Buy & Hold by {outperformance:.2f}%")
    else:
        print(f">>> UNDERPERFORMED Buy & Hold by {abs(outperformance):.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
