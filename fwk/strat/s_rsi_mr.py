"""
RSI Mean-Reversion Strategy.

Strategy that trades RSI overbought/oversold levels with EMA200 trend filter.
- Long Signal: RSI < oversold (30) AND close > EMA200 (mean-reversion in uptrend)
- Short Signal: RSI > overbought (70) AND close < EMA200 (mean-reversion in downtrend)
"""

from typing import Tuple
import pandas as pd
import numpy as np
# from features.feature_ta_utils import calculate_atr
import pandas as pd
from backtesting import Backtest, Strategy

from core.enums import (
    g_open_col,
    g_high_col,
    g_low_col,
    g_close_col,
    g_volume_col,
    g_index_col,
)

RSI_PERIOD = 14
OVERSOLD = 30
OVERBOUGHT = 70
EMA200_PERIOD = 200


def build_features(
    p_df: pd.DataFrame,
    p_direction: str = "both",
    p_rsi_period: int = RSI_PERIOD,
    p_oversold: int = OVERSOLD,
    p_overbought: int = OVERBOUGHT,
    p_ema200_period: int = EMA200_PERIOD,
    p_atr_len: int = 14,
) -> pd.DataFrame:
    """
    Build RSI mean-reversion signal features and ATR.
    
    Args:
        p_df: DataFrame with framework OHLCV columns
        p_direction: "long", "short", or "both" - which signals to generate
        p_rsi_period: RSI calculation period
        p_oversold: RSI oversold threshold
        p_overbought: RSI overbought threshold
        p_ema200_period: EMA period for trend filter
        p_atr_len: ATR calculation period
        
    Returns:
        DataFrame with added signal columns, RSI, EMA200, and ATR
    """
    df = p_df.copy()
    
    close = df[g_close_col]
    
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=p_rsi_period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=p_rsi_period).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    df["EMA200"] = close.ewm(span=p_ema200_period, adjust=False).mean()
    
    df["ATR"] = ta.atr(
        df[g_high_col],
        df[g_low_col],
        df[g_close_col],
        length=p_atr_len
    )
    
    df["signal"] = 0
    
    if p_direction in ("long", "both"):
        long_mask = (df["RSI"] < p_oversold) & (close > df["EMA200"])
        df.loc[long_mask, "signal"] = 1
    
    if p_direction in ("short", "both"):
        short_mask = (df["RSI"] > p_overbought) & (close < df["EMA200"])
        df.loc[short_mask, "signal"] = -1
    
    return df


def convert_to_ohlcv(p_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert framework columns to standard OHLCV format for backtesting.py.
    
    Args:
        p_df: DataFrame with framework columns (S_open_f32, etc.)
        
    Returns:
        DataFrame with OHLCV columns and datetime index
    """
    ohlcv_df = pd.DataFrame(index=p_df.index)
    
    ohlcv_df["Open"] = p_df[g_open_col]
    ohlcv_df["High"] = p_df[g_high_col]
    ohlcv_df["Low"] = p_df[g_low_col]
    ohlcv_df["Close"] = p_df[g_close_col]
    ohlcv_df["Volume"] = p_df[g_volume_col]
    
    ohlcv_df["signal"] = p_df["signal"]
    ohlcv_df["RSI"] = p_df["RSI"]
    ohlcv_df["EMA200"] = p_df["EMA200"]
    ohlcv_df["ATR"] = p_df["ATR"]
    
    if g_index_col in p_df.columns:
        base = pd.Timestamp("2000-01-01")
        ohlcv_df.index = base + pd.to_timedelta(p_df[g_index_col], unit="m")
    
    ohlcv_df = ohlcv_df.dropna(subset=["Open", "High", "Low", "Close", "ATR", "RSI", "EMA200"])
    
    return ohlcv_df


class RSIMeanReversionStrategy(Strategy):
    """
    RSI Mean-Reversion strategy with ATR-based stop loss and take profit.
    
    Parameters:
        atr_mult: ATR multiplier for stop loss distance
        rr: Risk-reward ratio for take profit
        direction: "long", "short", or "both"
        atr_len: ATR calculation period
    """
    
    atr_mult: float = 2.0
    rr: float = 2.0
    direction: str = "both"
    atr_len: int = 14
    
    def init(self):
        pass
    
    def next(self):
        if self.position:
            return
        
        close = float(self.data.Close[-1])
        sig = int(self.data.df["signal"].iloc[-1])
        atr = float(self.data.df["ATR"].iloc[-1])
        
        if not np.isfinite(atr) or atr <= 0:
            return
        
        sl_dist = self.atr_mult * atr
        
        if sig == 1 and self.direction in ("long", "both"):
            sl = close - sl_dist
            tp = close + self.rr * sl_dist
            self.buy(sl=sl, tp=tp)
        
        elif sig == -1 and self.direction in ("short", "both"):
            sl = close + sl_dist
            tp = close - self.rr * sl_dist
            self.sell(sl=sl, tp=tp)


def run_backtest(
    p_df: pd.DataFrame,
    p_cash: float = 100_000,
    p_commission: float = 0.0002,
    **kwargs
) -> Tuple[dict, pd.DataFrame]:
    """
    Run backtest with RSI Mean-Reversion strategy.
    
    Args:
        p_df: DataFrame with framework columns + signal + RSI + EMA200 + ATR
        p_cash: Initial cash amount
        p_commission: Commission per trade
        **kwargs: Strategy parameters (atr_mult, rr, direction, atr_len)
        
    Returns:
        Tuple of (backtest stats dict, trades DataFrame)
    """
    ohlcv_df = convert_to_ohlcv(p_df)
    
    bt = Backtest(
        ohlcv_df,
        RSIMeanReversionStrategy,
        cash=p_cash,
        commission=p_commission,
        trade_on_close=True,
        hedging=False,
        exclusive_orders=False,
    )
    
    stats = bt.run(**kwargs)
    trades = stats._trades if hasattr(stats, '_trades') else pd.DataFrame()
    
    return stats, trades


def main(
    p_df: pd.DataFrame,
    p_direction: str = "both",
    p_rsi_period: int = RSI_PERIOD,
    p_oversold: int = OVERSOLD,
    p_overbought: int = OVERBOUGHT,
    p_ema200_period: int = EMA200_PERIOD,
    p_atr_mult: float = 2.0,
    p_rr: float = 2.0,
    p_atr_len: int = 14,
    p_cash: float = 100_000,
    p_commission: float = 0.0002,
    p_verbose: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Main entry point: Build features, run backtest.
    
    Args:
        p_df: DataFrame with OHLCV data
        p_direction: "long", "short", or "both"
        p_rsi_period: RSI calculation period
        p_oversold: RSI oversold threshold
        p_overbought: RSI overbought threshold
        p_ema200_period: EMA period for trend filter
        p_atr_mult: ATR multiplier for stop loss
        p_rr: Risk-reward ratio
        p_atr_len: ATR period
        p_cash: Initial cash
        p_commission: Commission per trade
        p_verbose: Print results to console
        
    Returns:
        Tuple of (df with signals, backtest stats dict)
    """
    if p_verbose:
        print("=" * 80)
        print("RSI Mean-Reversion Strategy")
        print("=" * 80)
    
    df = p_df
    
    if p_verbose:
        print(f"\n2. Building features (direction={p_direction})...")
    df = build_features(
        df,
        p_direction=p_direction,
        p_rsi_period=p_rsi_period,
        p_oversold=p_oversold,
        p_overbought=p_overbought,
        p_ema200_period=p_ema200_period,
        p_atr_len=p_atr_len,
    )
    
    n_long = ((df["signal"] == 1)).sum()
    n_short = ((df["signal"] == -1)).sum()
    n_signals = (df["signal"] != 0).sum()
    if p_verbose:
        print(f"   Long signals: {n_long}")
        print(f"   Short signals: {n_short}")
        print(f"   Active signals: {n_signals}")
    
    if p_verbose:
        print(f"\n3. Running backtest...")
        print(f"   RSI period: {p_rsi_period}")
        print(f"   Oversold: {p_oversold}")
        print(f"   Overbought: {p_overbought}")
        print(f"   EMA200 period: {p_ema200_period}")
        print(f"   ATR mult: {p_atr_mult}")
        print(f"   Risk-reward: {p_rr}")
        print(f"   Direction: {p_direction}")
    
    stats, trades = run_backtest(
        df,
        p_cash=p_cash,
        p_commission=p_commission,
        atr_mult=p_atr_mult,
        rr=p_rr,
        direction=p_direction,
        atr_len=p_atr_len,
    )
    
    if p_verbose:
        strategy_return = stats['Return [%]']
        buy_hold_return = stats['Buy & Hold Return [%]']
        outperformance = strategy_return - buy_hold_return
        
        print("\n" + "=" * 80)
        print("BACKTEST RESULTS")
        print("=" * 80)
        
        print("\n--- PERFORMANCE SUMMARY ---")
        print(f"Start Date:            {stats['Start']}")
        print(f"End Date:              {stats['End']}")
        print(f"Duration:              {stats['Duration']}")
        
        print("\n--- RETURNS ---")
        print(f"Strategy Return [%]:       {strategy_return:.2f}")
        print(f"Buy & Hold Return [%]:     {buy_hold_return:.2f}")
        print(f"Outperformance [%]:        {outperformance:+.2f}")
        print(f"Max. Drawdown [%]:         {stats['Max. Drawdown [%]']:.2f}")
        print(f"Exposure Time [%]:         {stats['Exposure Time [%]']:.2f}")
        
        print("\n--- RISK METRICS ---")
        print(f"Sharpe Ratio:              {stats['Sharpe Ratio']:.2f}")
        print(f"Sortino Ratio:             {stats['Sortino Ratio']:.2f}")
        print(f"Calmar Ratio:              {stats['Calmar Ratio']:.2f}")
        
        print("\n--- TRADE STATISTICS ---")
        print(f"# Trades:                  {stats['# Trades']}")
        win_rate = stats['Win Rate [%]']
        print(f"Win Rate [%]:              {win_rate:.2f}")
        
        n_trades = stats['# Trades']
        n_wins = int(n_trades * win_rate / 100)
        n_losses = n_trades - n_wins
        print(f"# Winning Trades:          {n_wins}")
        print(f"# Losing Trades:           {n_losses}")
        
        if len(trades) > 0 and 'ReturnPct' in trades.columns:
            winning_trades = trades[trades['ReturnPct'] > 0]['ReturnPct']
            losing_trades = trades[trades['ReturnPct'] <= 0]['ReturnPct']
            
            avg_win = winning_trades.mean() * 100 if len(winning_trades) > 0 else 0
            avg_loss = losing_trades.mean() * 100 if len(losing_trades) > 0 else 0
            avg_win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            print(f"Avg. Winning Trade [%]:    {avg_win:.2f}")
            print(f"Avg. Losing Trade [%]:     {avg_loss:.2f}")
            print(f"Avg Win / Avg Loss Ratio:  {avg_win_loss_ratio:.2f}")
        
        print(f"Best Trade [%]:            {stats['Best Trade [%]']:.2f}")
        print(f"Worst Trade [%]:           {stats['Worst Trade [%]']:.2f}")
        print(f"Avg. Trade [%]:            {stats['Avg. Trade [%]']:.2f}")
        print(f"Max. Trade Duration:       {stats['Max. Trade Duration']}")
        print(f"Avg. Trade Duration:       {stats['Avg. Trade Duration']}")
        
        print("\n--- PROFIT ANALYSIS ---")
        print(f"Profit Factor:             {stats['Profit Factor']:.2f}")
        print(f"Expectancy [%]:            {stats['Expectancy [%]']:.2f}")
        print(f"SQN (System Quality):      {stats['SQN']:.2f}")
        
        print("\n--- STRATEGY PARAMETERS ---")
        print(f"Direction:                 {p_direction}")
        print(f"RSI Period:                {p_rsi_period}")
        print(f"Oversold:                  {p_oversold}")
        print(f"Overbought:                {p_overbought}")
        print(f"EMA200 Period:             {p_ema200_period}")
        print(f"ATR Multiplier:            {p_atr_mult}")
        print(f"Risk-Reward Ratio:         {p_rr}")
        
        print("\n" + "=" * 80)
        
        if outperformance > 0:
            print(f">>> Strategy OUTPERFORMED Buy & Hold by {outperformance:.2f}%")
        else:
            print(f">>> Strategy UNDERPERFORMED Buy & Hold by {abs(outperformance):.2f}%")
        print("=" * 80)
    
    return df, stats


if __name__ == "__main__":
    main()
