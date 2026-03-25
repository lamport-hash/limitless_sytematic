"""
Optimization Chart Generation.

Generates charts for parameter optimization results.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings('ignore')

CHARTS_DIR = Path(__file__).parent / "static" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_optim_charts(
    results: List[Dict],
    run_id: str,
    lookback_range: List[int],
) -> Dict[str, str]:
    """
    Generate optimization charts showing CAGR and Max DD vs Lookback period.
    
    Returns dict with chart filenames.
    """
    chart_files = {}
    
    valid_results = [r for r in results if r.get('cagr') is not None and r.get('error') is None]
    
    if not valid_results:
        return {'optim': None}
    
    lookbacks = [r['lookback'] for r in valid_results]
    cagrs = [r['cagr'] for r in valid_results]
    max_dds = [r['max_dd'] for r in valid_results]
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    color_cagr = '#2E86AB'
    color_dd = '#E74C3C'
    
    ax1.set_xlabel('Lookback Period (bars)', fontsize=12)
    ax1.set_ylabel('CAGR (%)', color=color_cagr, fontsize=12)
    line1 = ax1.plot(lookbacks, cagrs, color=color_cagr, linewidth=2.5, 
                     marker='o', markersize=4, label='CAGR')
    ax1.tick_params(axis='y', labelcolor=color_cagr)
    ax1.fill_between(lookbacks, cagrs, alpha=0.2, color=color_cagr)
    ax1.grid(True, alpha=0.3)
    
    best_cagr_idx = np.argmax(cagrs)
    best_lookback = lookbacks[best_cagr_idx]
    best_cagr = cagrs[best_cagr_idx]
    ax1.scatter([best_lookback], [best_cagr], color='gold', s=150, 
                zorder=5, edgecolors='black', linewidths=2,
                label=f'Best CAGR: {best_cagr:.1f}% @ {best_lookback}')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Max Drawdown (%)', color=color_dd, fontsize=12)
    line2 = ax2.plot(lookbacks, max_dds, color=color_dd, linewidth=2.5, 
                     linestyle='--', marker='s', markersize=4, label='Max DD')
    ax2.tick_params(axis='y', labelcolor=color_dd)
    ax2.invert_yaxis()
    
    min_dd_idx = np.argmin(max_dds)
    min_dd_lookback = lookbacks[min_dd_idx]
    min_dd = max_dds[min_dd_idx]
    ax2.scatter([min_dd_lookback], [min_dd], color='orange', s=150, 
                zorder=5, edgecolors='black', linewidths=2, marker='D')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    ax1.set_title(f'Optimization: CAGR & Max DD vs Lookback Period\n'
                  f'Range: {lookback_range[0]} - {lookback_range[1]} bars '
                  f'({len(valid_results)} points)', 
                  fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    optim_file = f"optim_{run_id}.png"
    fig.savefig(CHARTS_DIR / optim_file, dpi=150, bbox_inches='tight')
    chart_files['optim'] = optim_file
    plt.close(fig)
    
    return chart_files
