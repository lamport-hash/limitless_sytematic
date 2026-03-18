"""
Dual Momentum Strategy Backtest Runner.

Extracted from notebooks/bkt_dual_mom_clean.ipynb
"""

import os
import uuid
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns
from numba import njit

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strat.strat_backtest import compute_dual_momentum
from backtest.backtest_basket_alloc_based import run_full_backtest
from features.feature_ta_utils import numba_roc_correct_min, numba_rsi

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

CHARTS_DIR = Path(__file__).parent / "static" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

MONTH_NAMES = {
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
}


def extract_assets_from_df(df: pd.DataFrame) -> List[str]:
    """Extract unique asset symbols from DataFrame columns."""
    assets = set()
    for col in df.columns:
        if '_' in col and not col.startswith('F_') and not col.startswith('A_') and not col.startswith('i_'):
            parts = col.split('_')
            if len(parts) > 1:
                asset = parts[0]
                if asset and not asset.startswith('S_'):
                    assets.add(asset)
    return sorted(assets)


def load_parquet_file(filepath: str) -> Tuple[pd.DataFrame, List[str]]:
    """Load parquet file and extract assets."""
    df = pd.read_parquet(filepath)
    assets = extract_assets_from_df(df)
    return df, assets


def generate_roc_features(df: pd.DataFrame, assets: List[str], lookback: int) -> pd.DataFrame:
    """Generate ROC features for the specified lookback period."""
    df = df.copy()
    
    mid_col = "F_mid_f32"
    index_col = "i_minute_i"
    
    for asset in assets:
        mid_price_col = f"{asset}_{mid_col}"
        if mid_price_col not in df.columns:
            continue
            
        col_name = f"{asset}_F_roctrue_{lookback}_{mid_col}_f16"
        if col_name not in df.columns:
            df[col_name] = numba_roc_correct_min(
                df[mid_price_col].to_numpy(),
                df[index_col].to_numpy(),
                lookback * 60
            )
    
    return df


def generate_rsi_features(df: pd.DataFrame, assets: List[str], rsi_period: int) -> pd.DataFrame:
    """Generate RSI features for the specified period using close prices."""
    df = df.copy()
    
    close_col = "S_close_f32"
    
    for asset in assets:
        price_col = f"{asset}_{close_col}"
        if price_col not in df.columns:
            continue
            
        rsi_col_name = f"{asset}_F_rsi_{rsi_period}_{close_col}_f16"
        if rsi_col_name not in df.columns:
            prices = df[price_col].to_numpy()
            rsi_values = numba_rsi(prices, rsi_period)
            df[rsi_col_name] = rsi_values.astype(np.float32)
    
    return df


def minutes_to_datetime(minutes_since_2000):
    start_date = datetime(2000, 1, 1)
    return start_date + timedelta(minutes=float(minutes_since_2000))


def calculate_performance_metrics(p_df: pd.DataFrame, risk_free_rate: float = 0.02) -> Dict:
    """Calculate comprehensive performance metrics for a portfolio."""
    import math
    
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
    
    daily_returns = p_df['daily_returns']
    daily_vol = daily_returns.std()
    annual_vol = daily_vol * math.sqrt(365) if daily_vol > 0 else 0
    
    excess_returns = daily_returns - risk_free_rate/365
    sharpe_ratio = math.sqrt(365) * excess_returns.mean() / daily_vol if daily_vol > 0 else 0
    
    calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
    
    winning_days = len(daily_returns[daily_returns > 0])
    total_days = len(daily_returns)
    win_rate = (winning_days / total_days * 100) if total_days > 0 else 0
    
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
        'monthly_returns': monthly_returns_pivot,
        'df_with_metrics': p_df
    }


def calculate_asset_metrics(p_df: pd.DataFrame, assets: List[str], orders_df: pd.DataFrame = None, risk_free_rate: float = 0.02) -> Dict:
    """Calculate per-asset performance metrics."""
    import math
    
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


def generate_charts(metrics: Dict, run_id: str, assets_to_use: List[str] = None) -> Dict[str, str]:
    """Generate performance charts and save as PNG files."""
    import math
    
    p_df = metrics['df_with_metrics']
    
    chart_files = {}
    
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    asset_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    if assets_to_use:
        for i, asset in enumerate(assets_to_use):
            close_col = f"{asset}_S_close_f32"
            if close_col in p_df.columns:
                asset_prices = p_df[close_col]
                initial_price = asset_prices.iloc[0]
                if initial_price > 0:
                    scaled = (asset_prices / initial_price) * 100
                    color = asset_colors[i % 10]
                    ax1.plot(p_df['date'], scaled, linewidth=1, alpha=0.5, 
                            color=color, linestyle='--', label=f'{asset}')
    
    port_scaled = (p_df['port_value'] / p_df['port_value'].iloc[0]) * 100
    ax1.plot(p_df['date'], port_scaled, linewidth=2.5, color='#2E86AB', label='Portfolio')
    ax1.fill_between(p_df['date'], port_scaled, 100, alpha=0.2, color='#2E86AB')
    
    ax1.set_title('Portfolio vs Assets (Scaled to 100)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value (Starting = 100)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=8, ncol=2)
    ax1.axhline(y=100, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    portfolio_file = f"portfolio_{run_id}.png"
    fig1.savefig(CHARTS_DIR / portfolio_file, dpi=150, bbox_inches='tight')
    chart_files['portfolio'] = portfolio_file
    plt.close(fig1)
    
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.fill_between(p_df['date'], p_df['drawdown'] * 100, 0, 
                     alpha=0.5, color='#F18F01')
    ax2.plot(p_df['date'], p_df['drawdown'] * 100, linewidth=1, color='#F18F01')
    ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)
    
    max_dd_idx = p_df['drawdown'].idxmin()
    ax2.scatter([p_df.loc[max_dd_idx, 'date']], 
                [p_df.loc[max_dd_idx, 'drawdown'] * 100], 
                color='red', s=100, zorder=5, 
                label=f'Max DD: {metrics["max_drawdown"]:.2f}%')
    ax2.legend(loc='lower right')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    drawdown_file = f"drawdown_{run_id}.png"
    fig2.savefig(CHARTS_DIR / drawdown_file, dpi=150, bbox_inches='tight')
    chart_files['drawdown'] = drawdown_file
    plt.close(fig2)
    
    fig3, ax3 = plt.subplots(figsize=(16, 8))
    
    monthly_returns_heatmap = metrics['monthly_returns'].copy()
    for month in MONTH_NAMES.values():
        if month not in monthly_returns_heatmap.columns:
            monthly_returns_heatmap[month] = np.nan
    
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_returns_heatmap = monthly_returns_heatmap[month_order]
    
    annual_returns = []
    for year in monthly_returns_heatmap.index:
        year_data = monthly_returns_heatmap.loc[year]
        valid_returns = year_data.dropna()
        if len(valid_returns) > 0:
            cumulative = 1.0
            for ret in valid_returns:
                cumulative *= (1 + ret / 100)
            annual_return = (cumulative - 1) * 100
            annual_returns.append(round(annual_return, 2))
        else:
            annual_returns.append(np.nan)
    monthly_returns_heatmap['Year'] = annual_returns
    
    heatmap_data = monthly_returns_heatmap.copy()
    annual_col = heatmap_data.pop('Year')
    
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', 
                cmap='RdYlGn', center=0, ax=ax3, 
                cbar_kws={'label': 'Return (%)', 'shrink': 0.8},
                linewidths=0.5, linecolor='gray', annot_kws={'size': 9})
    
    for i, (year, annual_ret) in enumerate(zip(heatmap_data.index, annual_col)):
        if not np.isnan(annual_ret):
            color = 'white' if annual_ret >= 0 else 'white'
            bg_color = '#2E7D32' if annual_ret >= 0 else '#C62828'
            ax3.text(12 + 0.5, i + 0.5, f'{annual_ret:.1f}%', 
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    color=color, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=bg_color, edgecolor='none', alpha=0.9))
    
    ax3.axvline(x=12, color='white', linewidth=2)
    ax3.set_xlim(0, 13)
    ax3.text(12.5, -0.5, 'Annual', ha='center', va='center', fontsize=10, fontweight='bold')
    ax3.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Year')
    
    monthly_file = f"monthly_returns_{run_id}.png"
    fig3.savefig(CHARTS_DIR / monthly_file, dpi=150, bbox_inches='tight')
    chart_files['monthly_returns'] = monthly_file
    plt.close(fig3)
    
    fig4, (ax4_top, ax4_bottom) = plt.subplots(2, 1, figsize=(10, 8))
    
    monthly_returns_flat = metrics['monthly_returns'].values.flatten()
    monthly_returns_clean = monthly_returns_flat[~np.isnan(monthly_returns_flat)]
    
    bins_monthly = np.linspace(-50, 50, 101)
    n, bin_edges, patches = ax4_top.hist(monthly_returns_clean, bins=bins_monthly, 
                                          edgecolor='white', linewidth=0.5, alpha=0.8)
    for patch, left_edge in zip(patches, bin_edges[:-1]):
        if left_edge < 0:
            patch.set_facecolor('#C62828')
        else:
            patch.set_facecolor('#2E7D32')
    
    ax4_top.axvline(x=0, color='white', linestyle='--', linewidth=1.5, alpha=0.7)
    mean_monthly = np.mean(monthly_returns_clean)
    ax4_top.axvline(x=mean_monthly, color='#FFD700', linestyle='-', linewidth=2, 
                    label=f'Mean: {mean_monthly:.2f}%')
    ax4_top.set_title('Monthly Returns Distribution', fontsize=14, fontweight='bold')
    ax4_top.set_xlabel('Return (%)')
    ax4_top.set_ylabel('Frequency')
    ax4_top.set_xlim(-50, 50)
    ax4_top.legend(loc='upper right')
    ax4_top.grid(True, alpha=0.3)
    
    annual_returns_list = []
    for year in monthly_returns_heatmap.index:
        year_data = monthly_returns_heatmap.loc[year]
        valid_returns = year_data.dropna()
        if len(valid_returns) > 0:
            cumulative = 1.0
            for ret in valid_returns:
                if not np.isnan(ret):
                    cumulative *= (1 + ret / 100)
            annual_returns_list.append((cumulative - 1) * 100)
    
    yearly_returns = np.array(annual_returns_list)
    
    bins_yearly = np.linspace(-50, 50, 101)
    n_y, bin_edges_y, patches_y = ax4_bottom.hist(yearly_returns, bins=bins_yearly,
                                                    edgecolor='white', linewidth=0.5, alpha=0.8)
    for patch, left_edge in zip(patches_y, bin_edges_y[:-1]):
        if left_edge < 0:
            patch.set_facecolor('#C62828')
        else:
            patch.set_facecolor('#2E7D32')
    
    ax4_bottom.axvline(x=0, color='white', linestyle='--', linewidth=1.5, alpha=0.7)
    mean_yearly = np.mean(yearly_returns)
    ax4_bottom.axvline(x=mean_yearly, color='#FFD700', linestyle='-', linewidth=2,
                       label=f'Mean: {mean_yearly:.2f}%')
    ax4_bottom.set_title('Yearly Returns Distribution', fontsize=14, fontweight='bold')
    ax4_bottom.set_xlabel('Return (%)')
    ax4_bottom.set_ylabel('Frequency')
    ax4_bottom.set_xlim(-50, 50)
    ax4_bottom.legend(loc='upper right')
    ax4_bottom.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    sharpe_file = f"returns_dist_{run_id}.png"
    fig4.savefig(CHARTS_DIR / sharpe_file, dpi=150, bbox_inches='tight')
    chart_files['sharpe'] = sharpe_file
    plt.close(fig4)
    
    return chart_files


def compute_trades_and_positions(
    orders_df: pd.DataFrame,
    p_df: pd.DataFrame,
    assets: List[str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Compute both closed trades and remaining open positions using FIFO matching.
    
    Returns: (trades_list, positions_list)
    """
    if orders_df is None or len(orders_df) == 0:
        return [], []
    
    price_data = {}
    for asset in assets:
        col = f"{asset}_S_close_f32"
        if col in p_df.columns:
            price_data[asset] = p_df[col].to_numpy()
    
    minute_values = p_df['i_minute_i'].to_numpy() if 'i_minute_i' in p_df.columns else None
    
    trades = []
    positions = []
    
    orders_sorted = orders_df.sort_values('timestamp').reset_index(drop=True)
    
    for asset in assets:
        asset_orders = orders_sorted[orders_sorted['etf'] == asset].copy()
        
        if len(asset_orders) == 0:
            continue
        
        open_lots = []
        
        for _, order in asset_orders.iterrows():
            if order['direction'] > 0:
                open_lots.append({
                    'qty': float(order['size']),
                    'entry_price': float(order['price']),
                    'entry_row': int(order['timestamp'])
                })
            else:
                sell_qty = float(order['size'])
                sell_price = float(order['price'])
                sell_row = int(order['timestamp'])
                
                while sell_qty > 1e-9 and open_lots:
                    lot = open_lots[0]
                    
                    matched_qty = min(lot['qty'], sell_qty)
                    
                    entry_row = lot['entry_row']
                    exit_row = sell_row
                    
                    if minute_values is not None:
                        if entry_row < len(minute_values) and exit_row < len(minute_values):
                            entry_minute = minute_values[entry_row]
                            exit_minute = minute_values[exit_row]
                            entry_dt = minutes_to_datetime(entry_minute)
                            exit_dt = minutes_to_datetime(exit_minute)
                        else:
                            entry_dt = None
                            exit_dt = None
                    else:
                        entry_dt = None
                        exit_dt = None
                    
                    trade_return = ((sell_price - lot['entry_price']) / lot['entry_price']) * 100
                    
                    max_dd = 0.0
                    duration = 0
                    if asset in price_data:
                        prices = price_data[asset]
                        
                        if entry_row < len(prices) and exit_row <= len(prices):
                            trade_prices = prices[entry_row:exit_row + 1]
                            if len(trade_prices) > 0 and lot['entry_price'] > 0:
                                min_price = np.nanmin(trade_prices)
                                max_dd = ((min_price - lot['entry_price']) / lot['entry_price']) * 100
                            duration = exit_row - entry_row
                    
                    trades.append({
                        'asset': asset,
                        'entry_date': entry_dt.strftime('%Y-%m-%d %H:%M') if entry_dt else str(entry_row),
                        'entry_price': float(round(lot['entry_price'], 4)),
                        'entry_qty': float(round(matched_qty, 4)),
                        'exit_date': exit_dt.strftime('%Y-%m-%d %H:%M') if exit_dt else str(exit_row),
                        'exit_price': float(round(sell_price, 4)),
                        'exit_qty': float(round(matched_qty, 4)),
                        'return_pct': float(round(trade_return, 2)),
                        'max_dd_pct': float(round(max_dd, 2)),
                        'duration_bars': int(duration)
                    })
                    
                    lot['qty'] -= matched_qty
                    sell_qty -= matched_qty
                    
                    if lot['qty'] <= 1e-9:
                        open_lots.pop(0)
        
        for lot in open_lots:
            if lot['qty'] > 1e-9:
                col = f"{asset}_S_close_f32"
                current_price = 0.0
                if col in p_df.columns:
                    current_price = float(p_df[col].iloc[-1])
                
                entry_dt = None
                if minute_values is not None and lot['entry_row'] < len(minute_values):
                    entry_minute = minute_values[lot['entry_row']]
                    entry_dt = minutes_to_datetime(entry_minute)
                
                positions.append({
                    'asset': asset,
                    'entry_date': entry_dt.strftime('%Y-%m-%d %H:%M') if entry_dt else str(lot['entry_row']),
                    'entry_price': float(round(lot['entry_price'], 4)),
                    'current_price': float(round(current_price, 4)),
                    'quantity': float(round(lot['qty'], 4)),
                    'market_value': float(round(lot['qty'] * current_price, 2)),
                    'unrealized_pnl_pct': float(round(((current_price - lot['entry_price']) / lot['entry_price']) * 100, 2)) if lot['entry_price'] > 0 else 0.0
                })
    
    trades.sort(key=lambda x: x['entry_date'], reverse=True)
    
    return trades, positions


def compute_current_positions(
    orders_df: pd.DataFrame,
    p_df: pd.DataFrame,
    assets: List[str]
) -> List[Dict[str, Any]]:
    """
    Compute current open positions at the end of the backtest.
    
    Only returns positions for assets that have non-zero allocation at the end.
    Uses FIFO lot tracking for entry date/price.
    """
    if orders_df is None or len(orders_df) == 0 or p_df is None or len(p_df) == 0:
        return []
    
    final_allocs = {}
    for asset in assets:
        col = f"A_{asset}_alloc"
        if col in p_df.columns:
            final_allocs[asset] = float(p_df[col].iloc[-1])
    
    active_assets = [a for a in assets if final_allocs.get(a, 0) > 0]
    
    if not active_assets:
        return []
    
    minute_values = p_df['i_minute_i'].to_numpy() if 'i_minute_i' in p_df.columns else None
    
    positions = []
    
    orders_sorted = orders_df.sort_values('timestamp').reset_index(drop=True)
    
    for asset in active_assets:
        asset_orders = orders_sorted[orders_sorted['etf'] == asset].copy()
        
        if len(asset_orders) == 0:
            continue
        
        open_lots = []
        
        for _, order in asset_orders.iterrows():
            if order['direction'] > 0:
                open_lots.append({
                    'qty': float(order['size']),
                    'entry_price': float(order['price']),
                    'entry_row': int(order['timestamp'])
                })
            else:
                sell_qty = float(order['size'])
                while sell_qty > 1e-9 and open_lots:
                    lot = open_lots[0]
                    matched = min(lot['qty'], sell_qty)
                    lot['qty'] -= matched
                    sell_qty -= matched
                    if lot['qty'] <= 1e-9:
                        open_lots.pop(0)
        
        for lot in open_lots:
            if lot['qty'] > 1e-9:
                col = f"{asset}_S_close_f32"
                current_price = 0.0
                if col in p_df.columns:
                    current_price = float(p_df[col].iloc[-1])
                
                market_value = lot['qty'] * current_price
                
                if market_value < 1.0:
                    continue
                
                entry_dt = None
                if minute_values is not None and lot['entry_row'] < len(minute_values):
                    entry_minute = minute_values[lot['entry_row']]
                    entry_dt = minutes_to_datetime(entry_minute)
                
                positions.append({
                    'asset': asset,
                    'entry_date': entry_dt.strftime('%Y-%m-%d %H:%M') if entry_dt else str(lot['entry_row']),
                    'entry_price': float(round(lot['entry_price'], 4)),
                    'current_price': float(round(current_price, 4)),
                    'quantity': float(round(lot['qty'], 4)),
                    'market_value': float(round(market_value, 2)),
                    'unrealized_pnl_pct': float(round(((current_price - lot['entry_price']) / lot['entry_price']) * 100, 2)) if lot['entry_price'] > 0 else 0.0
                })
    
    return positions


def compute_trades_from_orders(
    orders_df: pd.DataFrame,
    p_df: pd.DataFrame,
    assets: List[str]
) -> List[Dict[str, Any]]:
    trades, _ = compute_trades_and_positions(orders_df, p_df, assets)
    return trades


def run_backtest(
    filepath: str,
    selected_assets: List[str],
    lookback: int,
    default_asset: str,
    top_n: int,
    abs_momentum_threshold: float,
    transaction_cost_pct: float = 0.1,
    min_holding_periods: int = 0,
    switch_threshold_pct: float = 0.0,
    rsi_period: int = 14,
    use_rsi_entry_filter: bool = False,
    rsi_entry_max: float = 30.0,
    use_rsi_entry_queue: bool = False,
    use_rsi_diff_filter: bool = False,
    rsi_diff_threshold: float = 10.0,
    run_id: Optional[str] = None,
) -> Dict:
    """Run the complete dual momentum backtest."""
    if run_id is None:
        run_id = str(uuid.uuid4())[:8]
    
    df, all_assets = load_parquet_file(filepath)
    
    assets_to_use = [a for a in selected_assets if a in all_assets]
    if not assets_to_use:
        raise ValueError("No valid assets found in the selected file")
    
    if default_asset not in assets_to_use:
        raise ValueError(f"Default asset '{default_asset}' not in selected assets")
    
    default_asset_idx = assets_to_use.index(default_asset)
    
    df_with_roc = generate_roc_features(df, assets_to_use, lookback)
    
    if use_rsi_entry_filter or use_rsi_diff_filter:
        df_with_roc = generate_rsi_features(df_with_roc, assets_to_use, rsi_period)
    
    feature_id = f"F_roctrue_{lookback}_F_mid_f32_f16"
    rsi_feature_id = f"F_rsi_{rsi_period}_S_close_f32_f16"
    
    df_allocations = compute_dual_momentum(
        p_df=df_with_roc,
        p_feature_id=feature_id,
        p_default_asset_idx=default_asset_idx,
        p_default_asset=default_asset,
        p_top_n=top_n,
        p_abs_momentum_threshold=abs_momentum_threshold,
        p_asset_list=assets_to_use,
        p_min_holding_periods=min_holding_periods,
        p_switch_threshold_pct=switch_threshold_pct,
        p_use_rsi_entry_filter=use_rsi_entry_filter,
        p_rsi_entry_max=rsi_entry_max,
        p_use_rsi_entry_queue=use_rsi_entry_queue,
        p_use_rsi_diff_filter=use_rsi_diff_filter,
        p_rsi_diff_threshold=rsi_diff_threshold,
        p_rsi_feature_id=rsi_feature_id
    )
    
    p_df_result, orders_df = run_full_backtest(df_allocations, assets_to_use, transaction_cost_pct)
    
    min_idx = max(lookback, 100)
    p_df_result = p_df_result.iloc[min_idx:].copy()
    
    orders_df = orders_df[orders_df['timestamp'] >= min_idx].copy()
    orders_df['timestamp'] = orders_df['timestamp'] - min_idx
    orders_df = orders_df.reset_index(drop=True)
    
    entries_count = int((orders_df['direction'] > 0).sum()) if len(orders_df) > 0 else 0
    exits_count = int((orders_df['direction'] < 0).sum()) if len(orders_df) > 0 else 0
    
    entries_safe = int(((orders_df['direction'] > 0) & (orders_df['etf'] == default_asset)).sum()) if len(orders_df) > 0 else 0
    entries_risky = int(((orders_df['direction'] > 0) & (orders_df['etf'] != default_asset)).sum()) if len(orders_df) > 0 else 0
    exits_safe = int(((orders_df['direction'] < 0) & (orders_df['etf'] == default_asset)).sum()) if len(orders_df) > 0 else 0
    exits_risky = int(((orders_df['direction'] < 0) & (orders_df['etf'] != default_asset)).sum()) if len(orders_df) > 0 else 0
    
    metrics = calculate_performance_metrics(p_df_result)
    
    asset_metrics = calculate_asset_metrics(p_df_result, assets_to_use, orders_df)
    
    chart_files = generate_charts(metrics, run_id, assets_to_use)
    
    monthly_returns_dict = {}
    for year, row in metrics['monthly_returns'].iterrows():
        monthly_returns_dict[int(year)] = {k: (v if not pd.isna(v) else None) for k, v in row.to_dict().items()}
    
    return {
        'run_id': run_id,
        'metrics': {
            'start_date': metrics['start_date'].strftime('%Y-%m-%d %H:%M'),
            'end_date': metrics['end_date'].strftime('%Y-%m-%d %H:%M'),
            'years': metrics['years'],
            'start_value': metrics['start_value'],
            'end_value': metrics['end_value'],
            'total_return': metrics['total_return'],
            'cagr': metrics['cagr'],
            'max_drawdown': metrics['max_drawdown'],
            'max_drawdown_date': metrics['max_drawdown_date'].strftime('%Y-%m-%d'),
            'max_drawdown_peak_date': metrics['max_drawdown_peak_date'].strftime('%Y-%m-%d'),
            'annual_volatility': metrics['annual_volatility'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'calmar_ratio': metrics['calmar_ratio'],
            'win_rate': metrics['win_rate'],
            'monthly_returns': monthly_returns_dict,
        },
        'asset_metrics': asset_metrics,
        'charts': chart_files,
        'orders_count': len(orders_df),
        'entries_count': entries_count,
        'exits_count': exits_count,
        'entries_safe': entries_safe,
        'entries_risky': entries_risky,
        'exits_safe': exits_safe,
        'exits_risky': exits_risky,
        'orders_df': orders_df,
        'p_df': p_df_result,
    }
