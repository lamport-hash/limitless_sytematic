"""
Chart generation for backtest results.
"""

import math
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

CHARTS_DIR = Path(__file__).parent / "static" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

MONTH_NAMES = {
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
}


def generate_charts(metrics: Dict, run_id: str, assets_to_use: List[str] = None) -> Dict[str, str]:
    """Generate performance charts and save as PNG files."""
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
