import numpy as np
import pandas as pd
import vectorbt as vbt
from vectorbt.portfolio import nb
from vectorbt.portfolio.enums import SizeType, Direction
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Your OHLC column mapping
OHLC_COLS = {
    "open": "S_open_f32",
    "high": "S_high_f32",
    "low": "S_low_f32",
    "close": "S_close_f32",
    "volume": "S_volume_f64",
}

@dataclass
class Order:
    timestamp: int
    etf: str
    direction: float
    size: float
    price: float
    allocation: float

@dataclass
class PerformanceMetrics:
    total_return: float
    total_return_pct: float
    pnl_per_bar: float
    pnl_per_month: float
    cum_pnl_final: float
    final_portfolio_value: float
    max_drawdown: float
    max_dd: float
    max_dd_pct: float
    sharpe_ratio: float
    calmar_ratio: float
    profit_factor: float
    win_rate: float
    n_bars: int
    n_positive_bars: int
    n_negative_bars: int
    avg_win: float
    avg_loss: float
    avg_win_loss_ratio: float
    n_trades: int


def prepare_data_for_vectorbt(
    p_df: pd.DataFrame, 
    p_asset_list: List[str] = ['QQQ', 'SPY', 'TLT', 'GLD', 'VWO']
) -> pd.DataFrame:
    """
    Prepare your DataFrame for vectorbt by creating a MultiIndex DataFrame.
    This makes it easier to work with vectorbt's column-based operations.
    """
    # Create a MultiIndex DataFrame with (symbol, field)
    # This is the format vectorbt expects for multiple assets
    close_cols = [f"{etf}_{OHLC_COLS['close']}" for etf in p_asset_list]
    
    # Extract just the close prices for each asset
    close_data = pd.DataFrame(index=p_df.index)
    for i, etf in enumerate(p_asset_list):
        close_data[(etf, 'Close')] = p_df[close_cols[i]]
    
    # Set the MultiIndex columns
    close_data.columns = pd.MultiIndex.from_tuples(close_data.columns, names=['symbol', 'field'])
    
    return close_data


def generate_allocations_vectorbt(
    p_df: pd.DataFrame,
    p_asset_list: List[str] = ['QQQ', 'SPY', 'TLT', 'GLD', 'VWO']
) -> pd.DataFrame:
    """
    Extract allocation columns and format them for vectorbt.
    """
    alloc_cols = [f"A_{etf}_alloc" for etf in p_asset_list]
    
    # Extract allocations
    alloc_data = pd.DataFrame(index=p_df.index)
    for i, etf in enumerate(p_asset_list):
        alloc_data[(etf, 'Allocation')] = p_df[alloc_cols[i]]
    
    alloc_data.columns = pd.MultiIndex.from_tuples(alloc_data.columns, names=['symbol', 'field'])
    
    return alloc_data


def vectorbt_portfolio_backtest(
    p_df: pd.DataFrame,
    p_rebalance_threshold: float = 0.05,
    p_asset_list: List[str] = ['QQQ', 'SPY', 'TLT', 'GLD', 'VWO'],
    p_init_cash: float = 100000.0,
    p_fees: float = 0.001,  # 0.1% trading fee
    p_slippage: float = 0.0005,  # 0.05% slippage
) -> Tuple[vbt.Portfolio, pd.DataFrame, List[Order]]:
    """
    Run portfolio backtest using vectorbt with your allocation data.
    
    This is the FAST way - vectorbt will process all 5 assets simultaneously
    using vectorized operations and Numba compilation.
    """
    # Prepare data for vectorbt
    close_data = prepare_data_for_vectorbt(p_df, p_asset_list)
    alloc_data = generate_allocations_vectorbt(p_df, p_asset_list)
    
    # Get just the close prices as a DataFrame with asset columns
    prices = close_data.xs('Close', axis=1, level='field')
    
    # Get allocations as a DataFrame with asset columns
    allocations = alloc_data.xs('Allocation', axis=1, level='field')
    
    # Calculate allocation changes to identify rebalance dates
    alloc_changes = allocations.diff().abs()
    
    # Create rebalance mask: rebalance when any allocation changes by more than threshold
    # This replicates your logic in generate_orders_from_allocations
    rebalance_mask = (alloc_changes > p_rebalance_threshold).any(axis=1)
    
    # Set first row to rebalance (initial allocation)
    if len(rebalance_mask) > 0:
        rebalance_mask.iloc[0] = True
    
    # Create target allocation DataFrame (only on rebalance days, NaN otherwise)
    target_alloc = allocations.copy()
    target_alloc[~rebalance_mask] = np.nan
    
    # For vectorbt, we need to handle the portfolio as a group
    # We'll use from_orders with targetpercent sizing
    
    # Create orders DataFrame
    # On rebalance days, set target allocation as size
    orders = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    orders[rebalance_mask] = target_alloc[rebalance_mask]
    
    # Run vectorbt portfolio backtest
    # This is the core - vectorbt will handle all the rebalancing logic
    portfolio = vbt.Portfolio.from_orders(
        prices,
        size=orders,
        size_type='targetpercent',  # This is key: allocate to reach target % of portfolio
        direction='longonly',
        fees=p_fees,
        slippage=p_slippage,
        init_cash=p_init_cash,
        freq='1h',  # Hourly data
        group_by=True,  # Treat all assets as one portfolio
        cash_sharing=True,  # Share capital across assets
        call_seq='default'
    )
    
    # Generate orders list (for compatibility with your existing code)
    orders_list = generate_orders_from_trades(portfolio, prices, p_asset_list, p_df.index)
    
    return portfolio, orders, orders_list


def generate_orders_from_trades(
    portfolio: vbt.Portfolio,
    prices: pd.DataFrame,
    p_asset_list: List[str],
    index: pd.Index
) -> List[Order]:
    """
    Convert vectorbt trades to your Order format for compatibility.
    """
    orders_list = []
    
    # Get all trades
    trades = portfolio.trades.records_readable
    
    if trades is None or len(trades) == 0:
        return orders_list
    
    # Group trades by column (asset)
    for col_idx, asset in enumerate(p_asset_list):
        asset_trades = trades[trades['Column'] == col_idx]
        
        for _, trade in asset_trades.iterrows():
            # Entry order
            entry_order = Order(
                timestamp=int(pd.Timestamp(trade['Entry Index']).timestamp() * 1000) if hasattr(trade['Entry Index'], 'timestamp') else int(trade['Entry Index']),
                etf=asset,
                direction=1.0,  # Long entry
                size=trade['Size'],
                price=trade['Entry Price'],
                allocation=0.0  # We don't have allocation info here
            )
            orders_list.append(entry_order)
            
            # Exit order
            exit_order = Order(
                timestamp=int(pd.Timestamp(trade['Exit Index']).timestamp() * 1000) if hasattr(trade['Exit Index'], 'timestamp') else int(trade['Exit Index']),
                etf=asset,
                direction=-1.0,  # Exit
                size=trade['Size'],
                price=trade['Exit Price'],
                allocation=0.0
            )
            orders_list.append(exit_order)
    
    return orders_list


def compute_strategy_performance_vectorbt(
    p_df: pd.DataFrame,
    p_orders_df: pd.DataFrame,
    p_portfolio: vbt.Portfolio,
    p_asset_list: List[str] = ['QQQ', 'SPY', 'TLT', 'GLD', 'VWO']
) -> Tuple[Dict, pd.DataFrame]:
    """
    Compute performance metrics using vectorbt's built-in stats.
    This is much faster and more comprehensive than manual calculation.
    """
    # Get portfolio stats
    stats = p_portfolio.stats()
    
    # Get daily/hourly returns
    returns = p_portfolio.returns()  # This returns a Series
    
    # Get drawdown
    drawdown = p_portfolio.drawdown()
    max_drawdown = drawdown.min()
    
    # Get trades for win rate calculation
    trades = p_portfolio.trades.records_readable
    
    # Calculate win rate
    if trades is not None and len(trades) > 0:
        winning_trades = trades[trades['PnL'] > 0]
        win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
        avg_win = winning_trades['PnL'].mean() if len(winning_trades) > 0 else 0
        avg_loss = trades[trades['PnL'] < 0]['PnL'].mean() if len(trades[trades['PnL'] < 0]) > 0 else 0
        avg_win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        n_trades = len(trades)
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        avg_win_loss_ratio = 0
        n_trades = 0
    
    # Get bar-level P&L
    bar_pnl = p_portfolio.pnl()
    
    # Calculate positive/negative bars
    n_bars = len(bar_pnl)
    n_positive_bars = (bar_pnl > 0).sum()
    n_negative_bars = (bar_pnl < 0).sum()
    avg_pnl_per_bar = bar_pnl.mean()
    
    # Monthly P&L
    monthly_pnl = bar_pnl.resample('M').sum()
    avg_pnl_per_month = monthly_pnl.mean()
    
    # Final portfolio value
    final_value = p_portfolio.value()
    
    # Sharpe ratio
    sharpe = p_portfolio.sharpe_ratio()
    
    # Calmar ratio
    calmar = p_portfolio.calmar_ratio()
    
    # Profit factor
    profit_factor = p_portfolio.profit_factor()
    
    # Create metrics dictionary matching your format
    metrics = {
        "total_return": float(p_portfolio.total_return()),
        "total_return_pct": float(p_portfolio.total_return() * 100),
        "pnl_per_bar": float(avg_pnl_per_bar),
        "pnl_per_month": float(avg_pnl_per_month),
        "cum_pnl_final": float(bar_pnl.sum()),
        "final_portfolio_value": float(final_value),
        "max_drawdown": float(max_drawdown),
        "max_dd_pct": float(max_drawdown * 100),
        "sharpe_ratio": float(sharpe),
        "calmar_ratio": float(calmar),
        "profit_factor": float(profit_factor),
        "win_rate": float(win_rate),
        "n_bars": int(n_bars),
        "n_positive_bars": int(n_positive_bars),
        "n_negative_bars": int(n_negative_bars),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "avg_win_loss_ratio": float(avg_win_loss_ratio),
        "n_trades": int(n_trades),
    }
    
    # Create result DataFrame with bar-level data
    result_df = p_df.copy()
    result_df['pnl_per_bar'] = bar_pnl.values
    result_df['cum_pnl'] = bar_pnl.cumsum().values
    result_df['portfolio_value'] = p_portfolio.value().values
    
    return metrics, result_df


def run_fast_backtest(
    p_df: pd.DataFrame,
    p_rebalance_threshold: float = 0.05,
    p_asset_list: List[str] = ['QQQ', 'SPY', 'TLT', 'GLD', 'VWO'],
    p_init_cash: float = 100000.0,
    p_fees: float = 0.001,
    p_output_dir: Optional[Path] = None,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """
    Complete backtest pipeline using vectorbt for maximum speed.
    
    This function integrates all the components and runs the backtest.
    """
    # Run vectorbt backtest
    print("Running vectorbt backtest...")
    portfolio, orders_df, orders_list = vectorbt_portfolio_backtest(
        p_df=p_df,
        p_rebalance_threshold=p_rebalance_threshold,
        p_asset_list=p_asset_list,
        p_init_cash=p_init_cash,
        p_fees=p_fees
    )
    
    print(f"Backtest completed in {portfolio._vbt.engine_time:.2f} seconds")
    
    # Compute performance metrics
    print("Computing performance metrics...")
    metrics, result_df = compute_strategy_performance_vectorbt(
        p_df=p_df,
        p_orders_df=orders_df,
        p_portfolio=portfolio,
        p_asset_list=p_asset_list
    )
    
    # Generate monthly/yearly returns for diagnostics
    monthly_returns = portfolio.returns().resample('M').apply(lambda x: (1 + x).prod() - 1).reset_index()
    monthly_returns.columns = ['date', 'return']
    
    yearly_returns = portfolio.returns().resample('Y').apply(lambda x: (1 + x).prod() - 1).reset_index()
    yearly_returns.columns = ['date', 'return']
    
    # Save diagnostics if output directory provided
    if p_output_dir:
        save_backtest_diagnostics_vectorbt(
            p_result_df=result_df,
            p_orders_df=orders_df,
            p_monthly_returns=monthly_returns,
            p_yearly_returns=yearly_returns,
            p_metrics=metrics,
            p_portfolio=portfolio,
            p_output_dir=p_output_dir
        )
    
    return metrics, result_df, orders_df


def save_backtest_diagnostics_vectorbt(
    p_result_df: pd.DataFrame,
    p_orders_df: pd.DataFrame,
    p_monthly_returns: pd.DataFrame,
    p_yearly_returns: pd.DataFrame,
    p_metrics: dict,
    p_portfolio: vbt.Portfolio,
    p_output_dir: Path,
) -> None:
    """
    Save comprehensive backtest diagnostics including vectorbt plots.
    """
    p_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save orders (same as before)
    orders_csv = p_output_dir / "orders.csv"
    p_orders_df.to_csv(orders_csv, index=False)
    print(f"Saved orders: {orders_csv}")
    
    # Save result data
    result_parquet = p_output_dir / "backtest_result.parquet"
    p_result_df.to_parquet(result_parquet)
    print(f"Saved backtest result: {result_parquet}")
    
    # Save monthly/yearly returns
    monthly_csv = p_output_dir / "monthly_returns.csv"
    p_monthly_returns.to_csv(monthly_csv, index=False)
    print(f"Saved monthly returns: {monthly_csv}")
    
    yearly_csv = p_output_dir / "yearly_returns.csv"
    p_yearly_returns.to_csv(yearly_csv, index=False)
    print(f"Saved yearly returns: {yearly_csv}")
    
    # Save metrics (enhanced with more details)
    metrics_txt = p_output_dir / "metrics.txt"
    with open(metrics_txt, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("VECTORBT STRATEGY PERFORMANCE METRICS\n")
        f.write("=" * 60 + "\n\n")
        
        for key, value in p_metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.6f}\n")
            else:
                f.write(f"{key}: {value}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("ADDITIONAL VECTORBT METRICS\n")
        f.write("=" * 60 + "\n\n")
        
        # Add vectorbt-specific stats
        stats = p_portfolio.stats()
        for key in stats.index:
            if key not in p_metrics:
                f.write(f"{key}: {stats[key]}\n")
    
    print(f"Saved metrics: {metrics_txt}")
    
    # Create and save performance plot
    try:
        fig = p_portfolio.plot().show()
        fig.write_html(p_output_dir / "performance_plot.html")
        print(f"Saved performance plot: {p_output_dir / 'performance_plot.html'}")
    except:
        print("Could not generate performance plot")


# Example usage
if __name__ == "__main__":
    # Assume p_df is your DataFrame with hourly data and allocation columns
    # p_df = pd.read_parquet("your_data.parquet")
    
    # For demonstration, create sample data
    print("Creating sample data for demonstration...")
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='1H')
    np.random.seed(42)
    
    # Create sample DataFrame with your structure
    asset_list = ['QQQ', 'SPY', 'TLT', 'GLD', 'VWO']
    data = {}
    
    # Add OHLC columns
    for etf in asset_list:
        # Generate random walk for prices
        returns = np.random.randn(len(dates)) * 0.0001
        price = 100 * np.exp(np.cumsum(returns))
        
        data[f"{etf}_{OHLC_COLS['close']}"] = price
        data[f"{etf}_{OHLC_COLS['open']}"] = price * (1 + np.random.randn(len(dates)) * 0.001)
        data[f"{etf}_{OHLC_COLS['high']}"] = price * (1 + np.abs(np.random.randn(len(dates)) * 0.001))
        data[f"{etf}_{OHLC_COLS['low']}"] = price * (1 - np.abs(np.random.randn(len(dates)) * 0.001))
        data[f"{etf}_{OHLC_COLS['volume']}"] = np.random.randint(1000, 10000, len(dates))
        
        # Add allocation columns (simulate some strategy)
        # Simple example: rotate between assets
        alloc = np.zeros(len(dates))
        period = len(dates) // 5
        for i in range(5):
            start_idx = i * period
            end_idx = min((i + 1) * period, len(dates))
            if i < len(asset_list):
                asset_idx = i
                alloc[start_idx:end_idx] = 1.0 if asset_idx == 0 else 0.0
        
        # Make it 50/50 split between two assets
        for i, etf in enumerate(asset_list):
            if i < 2:  # First two assets get 50% each
                data[f"A_{etf}_alloc"] = np.where(alloc == 1, 0.5, 0)
            else:
                data[f"A_{etf}_alloc"] = 0.0
    
    sample_df = pd.DataFrame(data, index=dates)
    
    # Run backtest
    print("\n" + "=" * 60)
    print("RUNNING VECTORBT BACKTEST")
    print("=" * 60)
    
    metrics, result_df, orders_df = run_fast_backtest(
        p_df=sample_df,
        p_rebalance_threshold=0.05,
        p_asset_list=asset_list,
        p_init_cash=100000.0,
        p_fees=0.001,
        p_output_dir=Path("./backtest_results")
    )
    
    # Print key metrics
    print("\n" + "=" * 60)
    print("KEY METRICS")
    print("=" * 60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")