import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tabulate import tabulate
import seaborn as sns
from datetime import datetime, timedelta

# Set style for better looking charts
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Define month names globally so both functions can access it
MONTH_NAMES = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
               7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

def minutes_to_datetime(minutes_since_2000):
    """
    Convert minutes since 2000-01-01 to datetime
    
    Parameters:
    minutes_since_2000: Number of minutes since 2000-01-01 00:00
    
    Returns:
    datetime object
    """
    start_date = datetime(2000, 1, 1)
    return start_date + timedelta(minutes=float(minutes_since_2000))

def calculate_performance_metrics(p_df, risk_free_rate=0.02):
    """
    Calculate comprehensive performance metrics for a portfolio
    
    Parameters:
    p_df: DataFrame with columns 'i_minute_i' (timestamp in minutes since 2000-01-01) and 'port_value'
    risk_free_rate: Annual risk-free rate (default 2%)
    
    Returns:
    Dictionary with all performance metrics
    """
    
    # Convert minutes to datetime
    p_df = p_df.copy()
    p_df['date'] = p_df['i_minute_i'].apply(minutes_to_datetime)
    
    # Basic info
    start_time = p_df.iloc[0].i_minute_i
    end_time = p_df.iloc[-1].i_minute_i
    start_value = p_df.iloc[0].port_value
    end_value = p_df.iloc[-1].port_value
    
    # Time calculations
    minutes_in_year = 365 * 24 * 60
    years = (end_time - start_time) / minutes_in_year
    
    # Returns
    total_return = (end_value / start_value) - 1
    total_return_pct = total_return * 100
    
    # CAGR (Compound Annual Growth Rate)
    cagr = (math.pow(end_value / start_value, 1/years) - 1) if years > 0 else 0
    cagr_pct = cagr * 100
    
    # Calculate daily returns for metrics
    p_df['daily_returns'] = p_df['port_value'].pct_change()
    p_df = p_df.dropna()
    
    # Maximum Drawdown
    p_df['cumulative_max'] = p_df['port_value'].cummax()
    p_df['drawdown'] = (p_df['port_value'] - p_df['cumulative_max']) / p_df['cumulative_max']
    max_drawdown = p_df['drawdown'].min()
    max_drawdown_pct = max_drawdown * 100
    
    # Find max drawdown period
    max_dd_idx = p_df['drawdown'].idxmin()
    max_dd_date = p_df.loc[max_dd_idx, 'date']
    max_dd_peak_idx = p_df[:max_dd_idx]['cumulative_max'].idxmax()
    max_dd_peak_date = p_df.loc[max_dd_peak_idx, 'date']
    
    # Calculate running PnL
    p_df['pnl'] = p_df['port_value'] - start_value
    p_df['pnl_pct'] = (p_df['port_value'] / start_value - 1) * 100
    
    # Volatility (annualized)
    daily_returns = p_df['daily_returns']
    daily_vol = daily_returns.std()
    annual_vol = daily_vol * math.sqrt(365)
    
    # Sharpe Ratio
    excess_returns = daily_returns - risk_free_rate/365
    sharpe_ratio = math.sqrt(365) * excess_returns.mean() / daily_vol if daily_vol > 0 else 0
    
    # Calmar Ratio
    calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Win rate
    winning_days = len(daily_returns[daily_returns > 0])
    total_days = len(daily_returns)
    win_rate = (winning_days / total_days * 100) if total_days > 0 else 0
    
    # Monthly returns matrix
    p_df['year'] = p_df['date'].dt.year
    p_df['month'] = p_df['date'].dt.month
    
    # Calculate monthly returns (using last value of each month)
    monthly_returns = p_df.groupby(['year', 'month']).agg({
        'port_value': 'last',
        'date': 'last'
    }).reset_index()
    
    # Calculate monthly returns percentage
    monthly_returns['prev_value'] = monthly_returns.groupby('year')['port_value'].shift(1)
    
    # For first month of each year, use the last value from previous year
    for idx, row in monthly_returns.iterrows():
        if pd.isna(row['prev_value']) and row['month'] == 1:
            prev_year_data = p_df[p_df['date'] < pd.Timestamp(f"{row['year']}-01-01")]
            if len(prev_year_data) > 0:
                monthly_returns.at[idx, 'prev_value'] = prev_year_data.iloc[-1]['port_value']
            else:
                monthly_returns.at[idx, 'prev_value'] = p_df.iloc[0].port_value
    
    monthly_returns['monthly_return'] = (monthly_returns['port_value'] / monthly_returns['prev_value'] - 1) * 100
    
    # Create monthly returns matrix
    monthly_returns_pivot = monthly_returns.pivot(index='year', columns='month', values='monthly_return')
    
    # Rename months using global MONTH_NAMES
    monthly_returns_pivot = monthly_returns_pivot.rename(columns=MONTH_NAMES)
    
    # Round to 2 decimal places
    monthly_returns_pivot = monthly_returns_pivot.round(2)
    
    # Compile metrics
    metrics = {
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
        'df_with_metrics': p_df  # Store the dataframe with calculated metrics
    }
    
    return metrics

def plot_performance_charts(metrics, save_fig=False, filename='portfolio_performance.png'):
    """
    Create comprehensive performance charts including PnL and Drawdown
    
    Parameters:
    metrics: Dictionary from calculate_performance_metrics()
    save_fig: Whether to save the figure to file
    filename: Name of the file to save
    """
    
    p_df = metrics['df_with_metrics']
    
    # Create figure with subplots - adjust figure size and layout
    fig = plt.figure(figsize=(18, 14))
    
    # Define grid for subplots with more control
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25, 
                          height_ratios=[1, 1, 1.2])
    
    # 1. Portfolio Value Chart (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(p_df['date'], p_df['port_value'], linewidth=2, color='#2E86AB')
    ax1.fill_between(p_df['date'], p_df['port_value'], p_df['port_value'].iloc[0], 
                     alpha=0.3, color='#2E86AB')
    ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. Cumulative PnL (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(p_df['date'], p_df['pnl'], linewidth=2, color='#A23B72')
    ax2.fill_between(p_df['date'], p_df['pnl'], 0, 
                     where=(p_df['pnl'] >= 0), alpha=0.3, color='#2E86AB')
    ax2.fill_between(p_df['date'], p_df['pnl'], 0, 
                     where=(p_df['pnl'] < 0), alpha=0.3, color='#F18F01')
    ax2.set_title('Cumulative P&L', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('P&L ($)')
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Format x-axis dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # 3. Drawdown Chart (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.fill_between(p_df['date'], p_df['drawdown'] * 100, 0, 
                     alpha=0.5, color='#F18F01')
    ax3.plot(p_df['date'], p_df['drawdown'] * 100, linewidth=1, color='#F18F01')
    ax3.set_title('Drawdown', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Drawdown (%)')
    ax3.grid(True, alpha=0.3)
    
    # Highlight max drawdown
    max_dd_idx = p_df['drawdown'].idxmin()
    ax3.scatter(p_df.loc[max_dd_idx, 'date'], 
                p_df.loc[max_dd_idx, 'drawdown'] * 100, 
                color='red', s=100, zorder=5, 
                label=f'Max DD: {metrics["max_drawdown"]:.2f}%')
    ax3.legend(loc='lower right')
    
    # Format x-axis dates
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # 4. Rolling Metrics (middle right)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate rolling Sharpe (30-day)
    rolling_sharpe = p_df['daily_returns'].rolling(30).mean() / p_df['daily_returns'].rolling(30).std() * math.sqrt(365)
    rolling_sharpe = rolling_sharpe.fillna(0)
    
    ax4.plot(p_df['date'], rolling_sharpe, linewidth=2, color='#2E86AB', label='Rolling Sharpe (30d)')
    ax4.axhline(y=metrics['sharpe_ratio'], color='red', linestyle='--', 
                label=f'Avg Sharpe: {metrics["sharpe_ratio"]:.2f}')
    ax4.set_title('Rolling Sharpe Ratio', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Sharpe Ratio')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper left')
    
    # Format x-axis dates
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax4.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    # 5. Monthly Returns Heatmap (bottom) - span both columns
    ax5 = fig.add_subplot(gs[2, :])
    
    # Create heatmap of monthly returns
    monthly_returns_heatmap = metrics['monthly_returns'].copy()
    
    # Ensure all months are present using global MONTH_NAMES
    for month in MONTH_NAMES.values():
        if month not in monthly_returns_heatmap.columns:
            monthly_returns_heatmap[month] = np.nan
    
    # Reorder columns to calendar order
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_returns_heatmap = monthly_returns_heatmap[month_order]
    
    # Create heatmap
    sns.heatmap(monthly_returns_heatmap, annot=True, fmt='.2f', 
                cmap='RdYlGn', center=0, ax=ax5, 
                cbar_kws={'label': 'Return (%)', 'shrink': 0.8},
                linewidths=0.5, linecolor='gray', annot_kws={'size': 10})
    ax5.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Month')
    ax5.set_ylabel('Year')
    
    # Add overall title with more space
    fig.suptitle(f'Portfolio Performance Analysis\nCAGR: {metrics["cagr"]}% | Sharpe: {metrics["sharpe_ratio"]} | Max DD: {metrics["max_drawdown"]}%', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Use fig.tight_layout() instead of plt.tight_layout()
    fig.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust rect to make room for suptitle
    
    if save_fig:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Chart saved as '{filename}'")
    
    plt.show()

def print_performance_summary(metrics):
    """
    Print a nicely formatted performance summary
    """
    
    print("="*70)
    print("📊 PORTFOLIO PERFORMANCE SUMMARY")
    print("="*70)
    
    # Period info
    print("\n📅 PERIOD INFORMATION")
    print("-"*50)
    print(f"Start Date:     {metrics['start_date'].strftime('%Y-%m-%d %H:%M')}")
    print(f"End Date:       {metrics['end_date'].strftime('%Y-%m-%d %H:%M')}")
    print(f"Total Period:   {metrics['years']:.2f} years")
    
    # Value info
    print(f"\n💰 PORTFOLIO VALUES")
    print("-"*50)
    print(f"Start Value:    ${metrics['start_value']:,.2f}")
    print(f"End Value:      ${metrics['end_value']:,.2f}")
    print(f"Total Return:   {metrics['total_return']:+.2f}%")
    print(f"CAGR:           {metrics['cagr']:+.2f}%")
    
    # Risk metrics
    print(f"\n📊 RISK METRICS")
    print("-"*50)
    print(f"Max Drawdown:   {metrics['max_drawdown']:.2f}%")
    print(f"Max DD Peak:    {metrics['max_drawdown_peak_date'].strftime('%Y-%m-%d')}")
    print(f"Max DD Valley:  {metrics['max_drawdown_date'].strftime('%Y-%m-%d')}")
    print(f"Annual Vol:     {metrics['annual_volatility']:.2f}%")
    
    # Performance ratios
    print(f"\n📈 PERFORMANCE RATIOS")
    print("-"*50)
    print(f"Sharpe Ratio:   {metrics['sharpe_ratio']:.2f}")
    print(f"Calmar Ratio:   {metrics['calmar_ratio']:.2f}")
    print(f"Win Rate:       {metrics['win_rate']:.2f}%")
    
    # Monthly returns matrix
    print(f"\n📅 MONTHLY RETURNS MATRIX (%)")
    print("-"*50)
    print(tabulate(metrics['monthly_returns'], 
                   headers='keys', 
                   tablefmt='pretty',
                   floatfmt='.2f',
                   showindex=True))
    
    print("\n" + "="*70)

# Usage
def test():
    print("Calculating performance metrics...")
    print(f"Data range: {p_df.iloc[0].i_minute_i} to {p_df.iloc[-1].i_minute_i} minutes since 2000-01-01")

    metrics = calculate_performance_metrics(p_df[2000:])

    # Print summary
    print_performance_summary(metrics)

    # Plot charts
    print("\nGenerating performance charts...")
    plot_performance_charts(metrics, save_fig=True, filename='portfolio_analysis.png')

    # You can also access individual metrics
    print(f"\n📌 Individual metrics:")
    print(f"   Years: {metrics['years']}")
    print(f"   Total Return: {metrics['total_return']}%")
    print(f"   CAGR: {metrics['cagr']}%")
    print(f"   Sharpe: {metrics['sharpe_ratio']}")
    print(f"   Calmar: {metrics['calmar_ratio']}")
    print(f"   Max DD: {metrics['max_drawdown']}%")
