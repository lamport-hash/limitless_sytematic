"""
Portfolio Optimization Module for Trend Detection Parameters
Supports multiple timeframes with automatically scaled parameters
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Callable, Optional, Any, Union
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import warnings
from numba import jit, prange
import json
from datetime import datetime
import os

# Import trend functions from your existing module
from features.targets_trend import (
    add_sma_trends,
    add_zigzag_trends,
    add_regression_trends,
    add_directional_trends,
    add_all_trends
)

from core.enums import g_high_col, g_low_col, g_close_col, g_open_col

warnings.filterwarnings('ignore')

# Global defaults
DEFAULT_COST = 0.005  # 0.5% trading cost
DEFAULT_MAX_LONG = 5
DEFAULT_MAX_SHORT = 2
DEFAULT_POPULATION_SIZE = 50
DEFAULT_GENERATIONS = 20
DEFAULT_MUTATION_RATE = 0.1
DEFAULT_RETURNS_COL = g_close_col


# Timeframe definitions (in minutes)
TIMEFRAME_MINUTES = {
    '1min': 1,
    '5min': 5,
    '15min': 15,
    '30min': 30,
    '1h': 60,
    '4h': 240,
    '1d': 1440,
    '1w': 10080,
    '1M': 43200  # Approximate (30 days)
}

# Base parameter grids for 1-day timeframe
BASE_PARAM_GRIDS = {
    'sma': {
        'long_window': [50, 100, 200],  # days
        'threshold': [0.02, 0.05, 0.08],
        'min_length': [20, 30, 50],  # days
        'min_intensity': [0.10, 0.15, 0.20]
    },
    'zigzag': {
        'threshold': [0.03, 0.05, 0.07],
        'min_length': [15, 20, 30]  # days
    },
    'regression': {
        'window': [30, 50, 100],  # days
        'slope_threshold': [0.0003, 0.0005, 0.001],
        'r2_threshold': [0.2, 0.3, 0.4],
        'min_length': [15, 20, 30]  # days
    },
    'directional': {
        'lookback': [10, 15, 20],  # days
        'min_same_direction': [0.6, 0.65, 0.7],
        'min_length': [10, 15, 20]  # days
    }
}

# Map function names to their default prefixes
FUNCTION_PREFIX_MAP = {
    'add_sma_trends': 'sma',
    'add_zigzag_trends': 'zigzag',
    'add_regression_trends': 'regression',
    'add_directional_trends': 'directional'
}


def scale_parameters_for_timeframe(
    base_params: Dict[str, List],
    from_timeframe: str,
    to_timeframe: str,
    is_time_param: bool = True
) -> Dict[str, List]:
    """
    Scale parameters from one timeframe to another.
    
    Args:
        base_params: Base parameter grid for reference timeframe
        from_timeframe: Source timeframe (e.g., '1d')
        to_timeframe: Target timeframe (e.g., '1h')
        is_time_param: If True, scale time-based parameters proportionally
    
    Returns:
        Scaled parameter grid for target timeframe
    """
    if from_timeframe not in TIMEFRAME_MINUTES or to_timeframe not in TIMEFRAME_MINUTES:
        raise ValueError(f"Unknown timeframe. Available: {list(TIMEFRAME_MINUTES.keys())}")
    
    scale_factor = TIMEFRAME_MINUTES[from_timeframe] / TIMEFRAME_MINUTES[to_timeframe]
    
    scaled_params = {}
    time_based_params = ['long_window', 'min_length', 'window', 'lookback']
    
    for param_name, values in base_params.items():
        if is_time_param and param_name in time_based_params:
            # Scale time-based parameters
            scaled_values = []
            for v in values:
                scaled = int(v * scale_factor)
                # Ensure minimum values
                if param_name in ['min_length']:
                    scaled = max(5, scaled)
                elif param_name in ['window', 'long_window']:
                    scaled = max(10, scaled)
                elif param_name in ['lookback']:
                    scaled = max(3, scaled)
                scaled_values.append(scaled)
            # Remove duplicates while preserving order
            scaled_values = list(dict.fromkeys(scaled_values))
            scaled_params[param_name] = scaled_values
        else:
            # Keep non-time parameters unchanged (thresholds, intensities, etc.)
            scaled_params[param_name] = values.copy()
    
    return scaled_params


def get_param_grid_for_timeframe(
    method: str,
    timeframe: str,
    custom_scaling: Optional[Dict[str, Callable]] = None
) -> Dict[str, List]:
    """
    Get parameter grid for a specific method and timeframe.
    
    Args:
        method: 'sma', 'zigzag', 'regression', or 'directional'
        timeframe: '1min', '1h', '1d', etc.
        custom_scaling: Optional custom scaling functions for specific parameters
    
    Returns:
        Parameter grid for the specified timeframe
    """
    if method not in BASE_PARAM_GRIDS:
        raise ValueError(f"Unknown method: {method}. Available: {list(BASE_PARAM_GRIDS.keys())}")
    
    if timeframe == '1d':
        param_grid = BASE_PARAM_GRIDS[method].copy()
    else:
        # Scale from daily to target timeframe
        param_grid = scale_parameters_for_timeframe(
            BASE_PARAM_GRIDS[method],
            '1d',
            timeframe
        )
    
    # Add prefix for the method
    param_grid['prefix'] = [f"{method}_{timeframe}"]
    
    # Apply custom scaling if provided
    if custom_scaling:
        for param_name, scaling_func in custom_scaling.items():
            if param_name in param_grid:
                param_grid[param_name] = [scaling_func(v, timeframe) for v in param_grid[param_name]]
    
    return param_grid


def get_all_timeframe_param_grids(
    timeframes: List[str],
    methods: Optional[List[str]] = None,
    custom_scaling: Optional[Dict[str, Dict[str, Callable]]] = None
) -> Dict[str, Dict[str, Dict[str, List]]]:
    """
    Get parameter grids for multiple timeframes and methods.
    
    Args:
        timeframes: List of timeframes (e.g., ['1h', '1d', '1w'])
        methods: List of methods (defaults to all methods)
        custom_scaling: Nested dict of custom scaling functions per method per parameter
    
    Returns:
        Nested dictionary: results[method][timeframe] = param_grid
    """
    if methods is None:
        methods = list(BASE_PARAM_GRIDS.keys())
    
    custom_scaling = custom_scaling or {}
    
    results = {}
    for method in methods:
        results[method] = {}
        for tf in timeframes:
            method_scaling = custom_scaling.get(method, {})
            results[method][tf] = get_param_grid_for_timeframe(
                method, tf, method_scaling
            )
    
    return results


@jit(nopython=True)
def compute_portfolio_returns_numba(
    returns_matrix: np.ndarray,
    signals_matrix: np.ndarray,
    costs: float,
    max_long: int,
    max_short: int
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Highly optimized portfolio return computation using Numba.
    """
    n_periods, n_assets = returns_matrix.shape
    
    # Track positions and portfolio value
    positions = np.zeros(n_assets)
    portfolio_value = 1.0
    prev_positions = np.zeros(n_assets)
    
    # Store daily values
    daily_returns = np.zeros(n_periods)
    portfolio_values = np.zeros(n_periods + 1)
    portfolio_values[0] = 1.0
    
    for t in range(n_periods):
        # Get signals for this period
        period_signals = signals_matrix[t]
        
        # Get indices of long and short signals
        long_indices = np.where(period_signals > 0)[0]
        short_indices = np.where(period_signals < 0)[0]
        
        # Limit number of positions
        if len(long_indices) > max_long:
            long_indices = long_indices[:max_long]
        
        if len(short_indices) > max_short:
            short_indices = short_indices[:max_short]
        
        # Build new positions array
        new_positions = np.zeros(n_assets)
        if max_long > 0 and len(long_indices) > 0:
            new_positions[long_indices] = 1.0 / max_long
        if max_short > 0 and len(short_indices) > 0:
            new_positions[short_indices] = -1.0 / max_short
        
        # Calculate trading costs (positions that changed)
        position_changes = np.abs(new_positions - prev_positions)
        trading_costs = np.sum(position_changes) * costs / 2
        
        # Calculate portfolio return for this period
        period_return = 0.0
        for i in range(n_assets):
            period_return += new_positions[i] * returns_matrix[t, i]
        
        # Update portfolio value (apply costs)
        portfolio_value *= (1 + period_return - trading_costs)
        
        # Store values
        daily_returns[t] = period_return - trading_costs
        portfolio_values[t + 1] = portfolio_value
        
        # Update positions for next period
        prev_positions = new_positions
    
    total_return = portfolio_value - 1.0
    return total_return, daily_returns, portfolio_values


def prepare_returns_matrix(
    dfs: List[pd.DataFrame],
    returns_col: str = DEFAULT_RETURNS_COL
) -> np.ndarray:
    """Precompute returns matrix for all assets."""
    n_assets = len(dfs)
    n_periods = len(dfs[0])
    
    returns_matrix = np.zeros((n_periods, n_assets))
    
    for i, df in enumerate(dfs):
        prices = df[returns_col].values
        returns = np.diff(prices) / prices[:-1]
        returns_matrix[1:, i] = returns
    
    return returns_matrix


def get_signals_from_function(
    df: pd.DataFrame,
    trend_function: Callable,
    params: Dict[str, Any]
) -> np.ndarray:
    """Apply trend function and extract signals."""
    df_copy = df.copy()
    
    try:
        df_result, _ = trend_function(df_copy, **params)
    except Exception as e:
        if 'prefix' in params:
            params_no_prefix = {k: v for k, v in params.items() if k != 'prefix'}
            df_result, _ = trend_function(df_copy, **params_no_prefix)
        else:
            raise e
    
    # Determine prefix
    if 'prefix' in params:
        prefix = params['prefix']
    else:
        func_name = trend_function.__name__
        prefix = FUNCTION_PREFIX_MAP.get(func_name, 'trend')
    
    # Find regime column
    possible_cols = [
        f'target_{prefix}_regime',
        'target_sma_regime',
        'target_zigzag_regime',
        'target_regression_regime',
        'target_directional_regime'
    ]
    
    signal_col = None
    for col in possible_cols:
        if col in df_result.columns:
            signal_col = col
            break
    
    if signal_col is None:
        regime_cols = [col for col in df_result.columns if 'regime' in col]
        if regime_cols:
            signal_col = regime_cols[0]
        else:
            raise ValueError(f"Could not find regime column in {df_result.columns}")
    
    return df_result[signal_col].values


def evaluate_parameters(
    params_dict: Dict[str, Any],
    dfs: List[pd.DataFrame],
    trend_function: Callable,
    returns_matrix: np.ndarray,
    max_long: int,
    max_short: int,
    cost: float
) -> Tuple[float, Dict[str, Any]]:
    """Evaluate a single parameter combination."""
    try:
        signals_list = []
        
        for df in dfs:
            signals = get_signals_from_function(df, trend_function, params_dict)
            signals_list.append(signals)
        
        signals_matrix = np.column_stack(signals_list)
        
        total_return, _, _ = compute_portfolio_returns_numba(
            returns_matrix,
            signals_matrix,
            cost,
            max_long,
            max_short
        )
        
        return total_return, params_dict
        
    except Exception as e:
        return -1e9, params_dict


class TrendOptimizer:
    """
    Main optimizer class for trend detection parameters with timeframe support.
    """
    
    def __init__(
        self,
        dfs: List[pd.DataFrame],
        trend_function: Callable,
        timeframe: str,
        returns_col: str = DEFAULT_RETURNS_COL,
        max_long: int = DEFAULT_MAX_LONG,
        max_short: int = DEFAULT_MAX_SHORT,
        cost: float = DEFAULT_COST
    ):
        """
        Initialize the optimizer for a specific timeframe.
        
        Args:
            dfs: List of DataFrames for each asset
            trend_function: One of the add_*_trends functions
            timeframe: Candle timeframe (e.g., '1h', '1d', '1w')
            returns_col: Column to use for return calculation
            max_long: Maximum number of long positions
            max_short: Maximum number of short positions
            cost: Trading cost per transaction
        """
        self.dfs = dfs
        self.trend_function = trend_function
        self.timeframe = timeframe
        self.returns_col = returns_col
        self.max_long = max_long
        self.max_short = max_short
        self.cost = cost
        
        # Validate inputs
        self._validate_dataframes()
        
        # Precompute returns matrix
        self.returns_matrix = prepare_returns_matrix(dfs, returns_col)
        self.n_assets = len(dfs)
        self.n_periods = len(dfs[0])
        
        # Store best results
        self.best_params = None
        self.best_return = -np.inf
        self.optimization_history = []
        
    def _validate_dataframes(self):
        """Validate that all dataframes have the same index."""
        if len(self.dfs) == 0:
            raise ValueError("No dataframes provided")
        
        first_index = self.dfs[0].index
        for i, df in enumerate(self.dfs[1:], 1):
            if not df.index.equals(first_index):
                raise ValueError(f"Dataframe {i} has different index from first dataframe")
    
    def grid_search(
        self,
        param_grid: Dict[str, List],
        n_jobs: int = -1,
        verbose: bool = True
    ) -> Dict:
        """Perform grid search optimization."""
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        combinations = list(product(*param_values))
        total_combinations = len(combinations)
        
        if verbose:
            print(f"Grid search ({self.timeframe}): Testing {total_combinations} combinations")
        
        param_dicts = [dict(zip(param_names, combo)) for combo in combinations]
        results = self._evaluate_parallel(param_dicts, n_jobs, verbose)
        
        best_idx = np.argmax([r[0] for r in results])
        self.best_return, self.best_params = results[best_idx]
        
        return self._create_result_dict(
            method='grid_search',
            n_tested=total_combinations,
            results=results,
            timeframe=self.timeframe
        )
    
    def random_search(
        self,
        param_grid: Dict[str, List],
        n_iter: int = DEFAULT_POPULATION_SIZE,
        n_jobs: int = -1,
        verbose: bool = True
    ) -> Dict:
        """Perform random search optimization."""
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        param_dicts = []
        for _ in range(n_iter):
            combo = [random.choice(values) for values in param_values]
            param_dicts.append(dict(zip(param_names, combo)))
        
        if verbose:
            print(f"Random search ({self.timeframe}): Testing {n_iter} random combinations")
        
        results = self._evaluate_parallel(param_dicts, n_jobs, verbose)
        
        best_idx = np.argmax([r[0] for r in results])
        self.best_return, self.best_params = results[best_idx]
        
        return self._create_result_dict(
            method='random_search',
            n_tested=n_iter,
            results=results,
            timeframe=self.timeframe
        )
    
    def genetic_algorithm(
        self,
        param_grid: Dict[str, List],
        population_size: int = DEFAULT_POPULATION_SIZE,
        generations: int = DEFAULT_GENERATIONS,
        mutation_rate: float = DEFAULT_MUTATION_RATE,
        n_jobs: int = -1,
        verbose: bool = True
    ) -> Dict:
        """Perform genetic algorithm optimization."""
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = [random.choice(values) for values in param_values]
            population.append(individual)
        
        best_overall_return = -np.inf
        best_overall_params = None
        generation_results = []
        
        for gen in range(generations):
            if verbose:
                print(f"Generation {gen + 1}/{generations} ({self.timeframe})")
            
            param_dicts = [dict(zip(param_names, ind)) for ind in population]
            results = self._evaluate_parallel(param_dicts, n_jobs, False)
            results.sort(key=lambda x: x[0], reverse=True)
            
            if results[0][0] > best_overall_return:
                best_overall_return = results[0][0]
                best_overall_params = results[0][1]
            
            generation_results.append({
                'generation': gen + 1,
                'best_return': results[0][0],
                'avg_return': np.mean([r[0] for r in results]),
                'best_params': results[0][1]
            })
            
            # Selection and breeding
            top_count = max(2, population_size // 5)
            top_individuals = [ind for _, ind in results[:top_count]]
            
            new_population = top_individuals.copy()
            
            while len(new_population) < population_size:
                if len(top_individuals) >= 2:
                    parent1, parent2 = random.sample(top_individuals, 2)
                else:
                    parent1 = parent2 = top_individuals[0]
                
                crossover_point = random.randint(1, len(param_names) - 1)
                child = parent1[:crossover_point] + parent2[crossover_point:]
                
                for i in range(len(child)):
                    if random.random() < mutation_rate:
                        child[i] = random.choice(param_values[i])
                
                new_population.append(child)
            
            population = new_population
        
        self.best_return = best_overall_return
        self.best_params = best_overall_params
        
        return self._create_result_dict(
            method='genetic_algorithm',
            n_tested=generations * population_size,
            results=[(r, p) for r, p in results[:10]],
            generation_results=generation_results,
            timeframe=self.timeframe
        )
    
    def _evaluate_parallel(
        self,
        param_dicts: List[Dict],
        n_jobs: int,
        verbose: bool
    ) -> List[Tuple[float, Dict]]:
        """Evaluate multiple parameter combinations in parallel."""
        n_combinations = len(param_dicts)
        results = []
        
        if n_jobs == 1 or n_jobs == 0:
            for i, params in enumerate(param_dicts):
                if verbose and i % max(1, n_combinations // 10) == 0:
                    print(f"  Progress: {i}/{n_combinations}")
                score, _ = evaluate_parameters(
                    params, self.dfs, self.trend_function,
                    self.returns_matrix, self.max_long, self.max_short, self.cost
                )
                results.append((score, params))
        else:
            max_workers = n_jobs if n_jobs > 0 else None
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        evaluate_parameters,
                        params, self.dfs, self.trend_function,
                        self.returns_matrix, self.max_long, self.max_short, self.cost
                    )
                    for params in param_dicts
                ]
                
                for i, future in enumerate(as_completed(futures)):
                    if verbose and i % max(1, n_combinations // 10) == 0:
                        print(f"  Progress: {i+1}/{n_combinations}")
                    score, params = future.result()
                    results.append((score, params))
        
        return results
    
    def _create_result_dict(
        self,
        method: str,
        n_tested: int,
        results: List[Tuple[float, Dict]],
        timeframe: str,
        generation_results: Optional[List] = None
    ) -> Dict:
        """Create standardized result dictionary."""
        scores = [r[0] for r in results if r[0] > -1e8]
        
        if scores:
            stats = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'median': float(np.median(scores)),
                'p25': float(np.percentile(scores, 25)),
                'p75': float(np.percentile(scores, 75))
            }
        else:
            stats = {k: -np.inf for k in ['mean', 'std', 'min', 'max', 'median', 'p25', 'p75']}
        
        result = {
            'timeframe': timeframe,
            'best_params': self.best_params,
            'best_return': float(self.best_return),
            'statistics': stats,
            'n_tested': n_tested,
            'method': method,
            'timestamp': datetime.now().isoformat(),
            'top_results': [
                {'return': float(r[0]), 'params': r[1]}
                for r in sorted(results, key=lambda x: x[0], reverse=True)[:10]
            ]
        }
        
        if generation_results:
            result['generation_history'] = generation_results
        
        return result
    
    def backtest(
        self,
        params: Optional[Dict] = None,
        verbose: bool = True
    ) -> Dict:
        """Backtest a parameter set and return detailed metrics."""
        if params is None:
            if self.best_params is None:
                raise ValueError("No parameters provided and no best_params from optimization")
            params = self.best_params
        
        # Get signals for all assets
        signals_list = []
        for df in self.dfs:
            signals = get_signals_from_function(df, self.trend_function, params)
            signals_list.append(signals)
        
        signals_matrix = np.column_stack(signals_list)
        
        # Compute portfolio returns
        total_return, daily_returns, portfolio_values = compute_portfolio_returns_numba(
            self.returns_matrix,
            signals_matrix,
            self.cost,
            self.max_long,
            self.max_short
        )
        
        # Calculate metrics
        if np.std(daily_returns) > 0:
            sharpe = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns)
        else:
            sharpe = 0
        
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        winning_days = np.sum(daily_returns > 0)
        win_rate = winning_days / len(daily_returns) if len(daily_returns) > 0 else 0
        
        if abs(max_drawdown) > 0:
            calmar = total_return / abs(max_drawdown)
        else:
            calmar = total_return if total_return > 0 else 0
        
        result = {
            'timeframe': self.timeframe,
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'calmar_ratio': float(calmar),
            'avg_daily_return': float(np.mean(daily_returns)),
            'volatility': float(np.std(daily_returns) * np.sqrt(252)),
            'best_day': float(np.max(daily_returns)),
            'worst_day': float(np.min(daily_returns)),
            'positive_days': int(winning_days),
            'negative_days': int(len(daily_returns) - winning_days),
            'total_days': int(len(daily_returns)),
            'portfolio_values': portfolio_values.tolist(),
            'daily_returns': daily_returns.tolist(),
            'params': params
        }
        
        if verbose:
            print("\n" + "="*60)
            print(f"BACKTEST RESULTS - {self.timeframe}")
            print("="*60)
            print(f"Total Return: {result['total_return']:.2%}")
            print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {result['max_drawdown']:.2%}")
            print(f"Win Rate: {result['win_rate']:.2%}")
            print(f"Calmar Ratio: {result['calmar_ratio']:.2f}")
            print(f"Volatility (ann): {result['volatility']:.2%}")
            print("="*60)
        
        return result
    
    def save_results(self, filename: str, results: Dict):
        """Save optimization results to JSON file."""
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            else:
                return obj
        
        with open(filename, 'w') as f:
            json.dump(results, f, default=convert_to_serializable, indent=2)
        
        print(f"Results saved to {filename}")


def optimize_all_timeframes(
    dfs: List[pd.DataFrame],
    timeframes: List[str],
    methods: Optional[List[str]] = None,
    max_long: int = DEFAULT_MAX_LONG,
    max_short: int = DEFAULT_MAX_SHORT,
    cost: float = DEFAULT_COST,
    optimization_method: str = 'random_search',
    n_iter: int = 30,
    n_jobs: int = -1,
    custom_scaling: Optional[Dict[str, Dict[str, Callable]]] = None,
    verbose: bool = True
) -> Dict[str, Dict[str, Dict]]:
    """
    Optimize parameters for multiple timeframes and methods.
    
    Args:
        dfs: List of DataFrames
        timeframes: List of timeframes to optimize
        methods: List of methods to optimize (defaults to all)
        max_long: Maximum long positions
        max_short: Maximum short positions
        cost: Trading cost
        optimization_method: 'grid_search', 'random_search', or 'genetic'
        n_iter: Number of iterations for random search
        n_jobs: Parallel jobs
        custom_scaling: Custom scaling functions per method per parameter
        verbose: Print progress
    
    Returns:
        Nested dictionary: results[timeframe][method] = optimization_result
    """
    if methods is None:
        methods = list(BASE_PARAM_GRIDS.keys())
    
    method_functions = {
        'sma': add_sma_trends,
        'zigzag': add_zigzag_trends,
        'regression': add_regression_trends,
        'directional': add_directional_trends
    }
    
    # Get all parameter grids
    all_param_grids = get_all_timeframe_param_grids(
        timeframes, methods, custom_scaling
    )
    
    results = {}
    
    for timeframe in timeframes:
        print(f"\n{'='*80}")
        print(f"OPTIMIZING FOR TIMEFRAME: {timeframe}")
        print(f"{'='*80}")
        
        results[timeframe] = {}
        
        for method in methods:
            print(f"\n--- Method: {method.upper()} ---")
            
            param_grid = all_param_grids[method][timeframe]
            
            optimizer = TrendOptimizer(
                dfs=dfs,
                trend_function=method_functions[method],
                timeframe=timeframe,
                max_long=max_long,
                max_short=max_short,
                cost=cost
            )
            
            if optimization_method == 'grid_search':
                opt_results = optimizer.grid_search(param_grid, n_jobs=n_jobs, verbose=verbose)
            elif optimization_method == 'random_search':
                opt_results = optimizer.random_search(
                    param_grid, n_iter=n_iter, n_jobs=n_jobs, verbose=verbose
                )
            elif optimization_method == 'genetic':
                opt_results = optimizer.genetic_algorithm(
                    param_grid,
                    population_size=n_iter,
                    generations=20,
                    n_jobs=n_jobs,
                    verbose=verbose
                )
            else:
                raise ValueError(f"Unknown optimization method: {optimization_method}")
            
            # Add backtest results
            backtest_results = optimizer.backtest(verbose=False)
            opt_results['backtest'] = backtest_results
            
            results[timeframe][method] = opt_results
            
            print(f"Best {method} return: {opt_results['best_return']:.2%}")
            print(f"Best params: {opt_results['best_params']}")
    
    # Create summary comparison across timeframes
    print("\n" + "="*80)
    print("TIMEFRAME COMPARISON - BEST RETURNS")
    print("="*80)
    
    comparison_data = []
    for timeframe in timeframes:
        for method in methods:
            if method in results[timeframe]:
                comparison_data.append({
                    'timeframe': timeframe,
                    'method': method,
                    'return': results[timeframe][method]['best_return'],
                    'sharpe': results[timeframe][method]['backtest']['sharpe_ratio'],
                    'drawdown': results[timeframe][method]['backtest']['max_drawdown'],
                    'win_rate': results[timeframe][method]['backtest']['win_rate']
                })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values(['timeframe', 'return'], ascending=[True, False])
    print(comparison_df.to_string(index=False))
    
    return results


# Example usage with multiple timeframes
if __name__ == "__main__":
    # Create sample data for 10 assets at different timeframes
    print("Creating sample data for multiple timeframes...")
    
    # Create daily data
    dates_daily = pd.date_range('2020-01-01', periods=500, freq='D')
    
    # Create hourly data (subset of daily)
    dates_hourly = pd.date_range('2020-01-01', periods=500*24, freq='H')[:500]
    
    # Create weekly data
    dates_weekly = pd.date_range('2020-01-01', periods=200, freq='W')
    
    np.random.seed(42)
    
    def generate_asset_prices(n_periods: int, common_factor: np.ndarray) -> List[pd.DataFrame]:
        """Generate prices for multiple assets."""
        dfs = []
        for asset_id in range(10):
            asset_specific = np.random.randn(n_periods) * 0.02
            prices = 100 * np.exp(common_factor[:n_periods] + asset_specific)
            prices = np.maximum(prices, 10)
            
            #g_high_col, g_low_col, g_close_col
            df = pd.DataFrame({
                g_open_col: prices,
                g_high_col: prices * (1 + np.abs(np.random.randn(n_periods) * 0.01)),
                g_low_col: prices * (1 - np.abs(np.random.randn(n_periods) * 0.01)),
                g_close_col: prices,
                'volume': np.random.randint(1e6, 1e7, n_periods)
            }, index=dates_daily[:n_periods])
            
            dfs.append(df)
        return dfs
    
    # Generate common factor for different timeframes
    common_factor_daily = np.cumsum(np.random.randn(500) * 0.01)
    common_factor_hourly = np.cumsum(np.random.randn(500) * 0.01 / np.sqrt(24))
    common_factor_weekly = np.cumsum(np.random.randn(200) * 0.01 * np.sqrt(7))
    
    # Create datasets for different timeframes
    dfs_dict = {
        '1h': generate_asset_prices(500, common_factor_hourly),
        '1d': generate_asset_prices(500, common_factor_daily),
        '1w': generate_asset_prices(200, common_factor_weekly)
    }
    
    print("Datasets created:")
    for tf, dfs in dfs_dict.items():
        print(f"  {tf}: {len(dfs)} assets, {len(dfs[0])} periods")
    
    # Define custom scaling for specific parameters
    def custom_threshold_scaling(value: float, timeframe: str) -> float:
        """Example custom scaling for threshold based on volatility."""
        if timeframe == '1h':
            return value * 1.5  # Higher threshold for hourly (more noise)
        elif timeframe == '1w':
            return value * 0.7   # Lower threshold for weekly (smoother)
        return value
    
    custom_scaling = {
        'sma': {
            'threshold': custom_threshold_scaling
        },
        'zigzag': {
            'threshold': custom_threshold_scaling
        }
    }
    
    # Run optimization for multiple timeframes
    print("\n" + "="*80)
    print("STARTING MULTI-TIMEFRAME OPTIMIZATION")
    print("="*80)
    
    # We'll optimize for each timeframe separately (since they have different data)
    all_results = {}
    
    for timeframe, dfs in dfs_dict.items():
        print(f"\n{'#'*80}")
        print(f"OPTIMIZING FOR TIMEFRAME: {timeframe}")
        print(f"{'#'*80}")
        
        results = optimize_all_timeframes(
            dfs=dfs,
            timeframes=[timeframe],  # Optimize one timeframe at a time
            methods=['sma', 'zigzag', 'regression', 'directional'],
            max_long=5,
            max_short=2,
            cost=0.005,
            optimization_method='random_search',
            n_iter=20,  # Reduced for demonstration
            n_jobs=-1,
            custom_scaling=custom_scaling,
            verbose=True
        )
        
        all_results[timeframe] = results[timeframe]
    
    # Create final comparison across all timeframes
    print("\n" + "="*80)
    print("FINAL RESULTS - BEST STRATEGY PER TIMEFRAME")
    print("="*80)
    
    final_comparison = []
    for timeframe in all_results:
        for method in all_results[timeframe]:
            result = all_results[timeframe][method]
            final_comparison.append({
                'timeframe': timeframe,
                'method': method,
                'return': f"{result['best_return']:.2%}",
                'sharpe': f"{result['backtest']['sharpe_ratio']:.2f}",
                'drawdown': f"{result['backtest']['max_drawdown']:.2%}",
                'params': str(result['best_params'])
            })
    
    final_df = pd.DataFrame(final_comparison)
    final_df = final_df.sort_values(['timeframe', 'return'], ascending=[True, False])
    print(final_df.to_string(index=False))
    
    # Save all results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"multitimeframe_optimization_{timestamp}.json"
    
    # Convert to serializable format
    serializable_results = {}
    for tf in all_results:
        serializable_results[tf] = {}
        for method in all_results[tf]:
            serializable_results[tf][method] = {
                'best_return': float(all_results[tf][method]['best_return']),
                'best_params': all_results[tf][method]['best_params'],
                'backtest': {
                    'sharpe_ratio': float(all_results[tf][method]['backtest']['sharpe_ratio']),
                    'max_drawdown': float(all_results[tf][method]['backtest']['max_drawdown']),
                    'win_rate': float(all_results[tf][method]['backtest']['win_rate'])
                }
            }
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to {filename}")