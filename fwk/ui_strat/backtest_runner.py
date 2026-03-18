"""
Dual Momentum Strategy Backtest Runner.

Extracted from notebooks/bkt_dual_mom_clean.ipynb
"""

import os
import uuid
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from features.feature_ta_utils import numba_roc_correct_min

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


def generate_charts(metrics: Dict, run_id: str) -> Dict[str, str]:
    """Generate performance charts and save as PNG files."""
    import math
    
    p_df = metrics['df_with_metrics']
    
    chart_files = {}
    
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(p_df['date'], p_df['port_value'], linewidth=2, color='#2E86AB')
    ax1.fill_between(p_df['date'], p_df['port_value'], p_df['port_value'].iloc[0], 
                     alpha=0.3, color='#2E86AB')
    ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
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


def run_backtest(
    filepath: str,
    selected_assets: List[str],
    lookback: int,
    default_asset: str,
    top_n: int,
    abs_momentum_threshold: float,
    transaction_cost_pct: float = 0.1,
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
    
    feature_id = f"F_roctrue_{lookback}_F_mid_f32_f16"
    
    df_allocations = compute_dual_momentum(
        p_df=df_with_roc,
        p_feature_id=feature_id,
        p_default_asset_idx=default_asset_idx,
        p_default_asset=default_asset,
        p_top_n=top_n,
        p_abs_momentum_threshold=abs_momentum_threshold,
        p_asset_list=assets_to_use
    )
    
    p_df_result, orders_df = run_full_backtest(df_allocations, assets_to_use, transaction_cost_pct)
    
    min_idx = max(lookback, 100)
    p_df_result = p_df_result.iloc[min_idx:].copy()
    
    metrics = calculate_performance_metrics(p_df_result)
    
    chart_files = generate_charts(metrics, run_id)
    
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
        'charts': chart_files,
        'orders_count': len(orders_df),
    }
