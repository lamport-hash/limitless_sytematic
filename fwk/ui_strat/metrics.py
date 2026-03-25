"""
Performance metrics calculations for backtest results.
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd


MONTH_NAMES = {
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
}


def minutes_to_datetime(minutes_since_2000):
    start_date = datetime(2000, 1, 1)
    return start_date + timedelta(minutes=float(minutes_since_2000))


def calculate_performance_metrics(p_df: pd.DataFrame, risk_free_rate: float = 0.02) -> Dict:
    """Calculate comprehensive performance metrics for a portfolio."""
    p_df = p_df.copy()
    p_df['date'] = p_df['i_minute_i'].apply(minutes_to_datetime)
    
    start_time = p_df.iloc[0]['i_minute_i']
    end_time = p_df.iloc[-1]['i_minute_i']
    start_value = p_df.iloc[0]['port_value']
    end_value = p_df.iloc[-1]['port_value']
    
    minutes_in_year = 365 * 24 * 60
    years = (end_time - start_time) / minutes_in_year
    
    total_return = (end_value / start_value) - 1
    total_return_pct = total_return * 100
    
    cagr = (math.pow(end_value / start_value, 1/years) - 1) if years > 0 else 0
    cagr_pct = cagr * 100
    
    p_df['daily_returns'] = p_df['port_value'].pct_change()
    p_df = p_df.dropna()
    
    p_df['cumulative_max'] = p_df['port_value'].cummax()
    p_df['drawdown'] = (p_df['port_value'] - p_df['cumulative_max']) / p_df['cumulative_max']
    max_drawdown = p_df['drawdown'].min()
    max_drawdown_pct = max_drawdown * 100
    
    max_dd_idx = p_df['drawdown'].idxmin()
    max_dd_date = p_df.loc[max_dd_idx, 'date']
    max_dd_peak_idx = p_df.loc[:max_dd_idx, 'cumulative_max'].idxmax()
    max_dd_peak_date = p_df.loc[max_dd_peak_idx, 'date']
    
    p_df['pnl'] = p_df['port_value'] - start_value
    p_df['pnl_pct'] = (p_df['port_value'] / start_value - 1) * 100
    
    daily_returns = p_df['daily_returns'].dropna() if 'daily_returns' in p_df.columns else pd.Series()
    daily_vol = daily_returns.std() if len(daily_returns) > 0 else 0
    annual_vol = daily_vol * math.sqrt(365) if daily_vol > 0 else 0
    
    excess_returns = daily_returns - risk_free_rate/365 if len(daily_returns) > 0 else pd.Series()
    sharpe_ratio = math.sqrt(365) * excess_returns.mean() / daily_vol if daily_vol > 0 else 0
    
    calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
    
    winning_days = len(daily_returns[daily_returns > 0]) if len(daily_returns) > 0 else 0
    losing_days = len(daily_returns[daily_returns < 0]) if len(daily_returns) > 0 else 0
    total_days = len(daily_returns) if len(daily_returns) > 0 else 0
    win_rate = (winning_days / total_days * 100) if total_days > 0 else 0
    
    gross_profits = daily_returns[daily_returns > 0].sum() if len(daily_returns) > 0 else 0
    gross_losses = abs(daily_returns[daily_returns < 0].sum()) if len(daily_returns) > 0 else 0
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf') if gross_profits > 0 else 0
    
    p_df['is_underwater'] = p_df['drawdown'] < 0
    p_df['prev_cummax'] = p_df['cumulative_max'].shift(1)
    p_df['is_at_high_watermark'] = p_df['port_value'] >= p_df['prev_cummax']
    p_df.loc[p_df.index[0], 'is_at_high_watermark'] = True
    
    recovery_indices = p_df.index[p_df['is_at_high_watermark']].tolist()
    
    if len(recovery_indices) < 2:
        longest_underwater_days = 0
    else:
        recovery_gaps_days = []
        for i in range(1, len(recovery_indices)):
            d1 = p_df.loc[recovery_indices[i], 'date']
            d0 = p_df.loc[recovery_indices[i-1], 'date']
            days_diff = (d1 - d0).days
            recovery_gaps_days.append(days_diff)
        longest_underwater_days = max(recovery_gaps_days) if recovery_gaps_days else 0
    
    gross_profits = daily_returns[daily_returns > 0].sum()
    gross_losses = abs(daily_returns[daily_returns < 0].sum())
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf') if gross_profits > 0 else 0
    
    p_df['is_underwater'] = p_df['drawdown'] < 0
    p_df['prev_cummax'] = p_df['cumulative_max'].shift(1)
    p_df['is_at_high_watermark'] = p_df['port_value'] >= p_df['prev_cummax']
    p_df.loc[p_df.index[0], 'is_at_high_watermark'] = True
    
    recovery_indices = p_df.index[p_df['is_at_high_watermark']].tolist()
    
    if len(recovery_indices) < 2:
        longest_underwater_days = 0
    else:
        recovery_gaps_days = []
        for i in range(1, len(recovery_indices)):
            d1 = p_df.loc[recovery_indices[i], 'date']
            d0 = p_df.loc[recovery_indices[i-1], 'date']
            days_diff = (d1 - d0).days
            recovery_gaps_days.append(days_diff)
        longest_underwater_days = max(recovery_gaps_days) if recovery_gaps_days else 0
    
    p_df['year'] = p_df['date'].dt.year
    p_df['month'] = p_df['date'].dt.month
    
    monthly_returns = p_df.groupby(['year', 'month']).agg({
        'port_value': 'last',
        'date': 'last'
    }).reset_index()
    
    monthly_returns['prev_value'] = monthly_returns.groupby('year')['port_value'].shift(1)
    
    for idx, row in monthly_returns.iterrows():
        if pd.isna(row['prev_value']) and row['month'] == 1:
            prev_year_data = p_df[p_df['date'] < pd.Timestamp(f"{row['year']}-01-01")]
            if len(prev_year_data) > 0:
                monthly_returns.at[idx, 'prev_value'] = prev_year_data.iloc[-1]['port_value']
            else:
                monthly_returns.at[idx, 'prev_value'] = p_df.iloc[0]['port_value']
    
    monthly_returns['monthly_return'] = (monthly_returns['port_value'] / monthly_returns['prev_value'] - 1) * 100
    
    monthly_returns_pivot = monthly_returns.pivot(index='year', columns='month', values='monthly_return')
    monthly_returns_pivot = monthly_returns_pivot.rename(columns=MONTH_NAMES)
    monthly_returns_pivot = monthly_returns_pivot.round(2)
    
    return {
        'start_date': p_df['date'].iloc[0],
        'end_date': p_df['date'].iloc[-1],
        'years': round(years, 2),
        'start_value': round(start_value, 2),
        'end_value': round(end_value, 2),
        'total_return': round(total_return_pct, 2),
        'cagr': round(cagr_pct, 2),
        'max_drawdown': round(max_drawdown_pct, 2),
        'max_drawdown_date': max_dd_date,
        'max_drawdown_peak_date': max_dd_peak_date,
        'annual_volatility': round(annual_vol * 100, 2),
        'sharpe_ratio': round(sharpe_ratio, 2),
        'calmar_ratio': round(calmar_ratio, 2),
        'win_rate': round(win_rate, 2),
        'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 999.99,
        'longest_underwater_days': round(longest_underwater_days, 0),
        'monthly_returns': monthly_returns_pivot,
        'df_with_metrics': p_df
    }


def calculate_asset_metrics(p_df: pd.DataFrame, assets: List[str], orders_df: pd.DataFrame = None, risk_free_rate: float = 0.02) -> Dict:
    """Calculate per-asset performance metrics."""
    asset_metrics = {}
    alloc_cols_in_df = [c for c in p_df.columns if '_alloc' in c]
    close_cols_in_df = [c for c in p_df.columns if '_S_close_f32' in c]
    
    for asset in assets:
        close_col = f"{asset}_S_close_f32"
        alloc_col = f"A_{asset}_alloc"
        
        if close_col not in p_df.columns:
            matching = [c for c in close_cols_in_df if c.startswith(asset + '_')]
            if matching:
                close_col = matching[0]
            else:
                continue
        
        if alloc_col not in p_df.columns:
            matching_alloc = [c for c in alloc_cols_in_df if asset in c]
            if matching_alloc:
                alloc_col = matching_alloc[0]
            else:
                alloc_col = None
        
        prices = p_df[close_col]
        
        if len(prices) < 2 or prices.iloc[0] <= 0:
            continue
        
        total_return = float((prices.iloc[-1] / prices.iloc[0] - 1) * 100)
        
        cummax = prices.cummax()
        drawdown = (prices - cummax) / cummax
        max_dd = float(drawdown.min() * 100)
        
        daily_rets = prices.pct_change().dropna()
        daily_vol = daily_rets.std()
        if daily_vol > 0:
            sharpe = float(math.sqrt(365) * (daily_rets.mean() - risk_free_rate / 365) / daily_vol)
        else:
            sharpe = 0.0
        
        if alloc_col and alloc_col in p_df.columns:
            alloc_series = p_df[alloc_col].fillna(0)
            allocated = int((alloc_series > 0).sum())
            
            if orders_df is not None and len(orders_df) > 0:
                entries = int(((orders_df['etf'] == asset) & (orders_df['direction'] > 0)).sum())
            else:
                entries = int(((alloc_series.shift(1) == 0) & (alloc_series > 0)).sum())
            
            asset_returns = prices.pct_change()
            contribution = asset_returns * alloc_series
            mask = alloc_series > 0
            if mask.any():
                strat_return = float((1 + contribution[mask]).prod() - 1) * 100
            else:
                strat_return = 0.0
        else:
            allocated = 0
            entries = 0
            strat_return = 0.0
        
        asset_metrics[asset] = {
            'total_return': round(total_return, 2),
            'max_drawdown': round(max_dd, 2),
            'sharpe_ratio': round(sharpe, 2),
            'candles_allocated': allocated,
            'strat_return': round(strat_return, 2),
            'entries': entries
        }
    
    total_allocated = sum(am['candles_allocated'] for am in asset_metrics.values())
    for asset in asset_metrics:
        if total_allocated > 0:
            pct = asset_metrics[asset]['candles_allocated'] / total_allocated * 100
        else:
            pct = 0.0
        asset_metrics[asset]['pct_allocated'] = round(pct, 2)
    
    return asset_metrics
