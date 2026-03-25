"""
Seasonality analysis service - calculate returns, volatility, RSI and seasonal patterns.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from core.data_org import MktDataTFreq, ProductType, get_normalised_dir
from core.search_data import search_data_paths
from norm.norm_utils import load_normalized_df


def to_python_type(val):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(val, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(val)
    elif isinstance(val, (np.floating, np.float64, np.float32, np.float16)):
        return float(val)
    elif isinstance(val, np.ndarray):
        return [to_python_type(x) for x in val.tolist()]
    elif isinstance(val, (pd.Timestamp,)):
        return str(val)
    elif val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    elif isinstance(val, dict):
        return {k: to_python_type(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [to_python_type(x) for x in val]
    return val


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate period returns from OHLCV data."""
    df = df.copy()
    df['return'] = (df['close'] - df['open']) / df['open'] * 100
    return df


def calculate_volatility(returns: np.ndarray) -> float:
    """Calculate standard deviation of returns."""
    if len(returns) < 2:
        return 0.0
    return float(np.std(returns, ddof=1))


def calculate_rsi(closes: np.ndarray, period: int = 14) -> float:
    """Calculate RSI using Wilder's smoothing method, return average RSI."""
    if len(closes) < period + 1:
        return 50.0
    
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    if avg_loss == 0:
        return 100.0
    
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    rs = avg_gain / avg_loss if avg_loss > 0 else 0
    rsi = 100 - (100 / (1 + rs))
    
    return float(rsi)


def load_asset_data(
    symbol: str,
    freq: str = "candle_1hour",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Load normalized data for a single asset."""
    try:
        freq_enum = MktDataTFreq(freq)
    except ValueError:
        return None
    
    paths = search_data_paths(p_symbol=symbol, p_data_freq=freq_enum)
    
    if not paths:
        return None
    
    all_dfs = []
    for path in paths:
        try:
            df = load_normalized_df(str(path))
            if df is not None and len(df) > 0:
                all_dfs.append(df)
        except Exception:
            continue
    
    if not all_dfs:
        return None
    
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=['i_minute_i'], keep='last')
    combined = combined.sort_values('i_minute_i').reset_index(drop=True)
    
    combined['datetime'] = pd.to_datetime(combined['i_minute_i'] * 60, unit='s', origin='2000-01-01')
    combined = prepare_columns(combined)
    
    if start_date:
        combined = combined[combined['datetime'] >= pd.to_datetime(start_date)]
    if end_date:
        combined = combined[combined['datetime'] <= pd.to_datetime(end_date)]
    
    return combined


def prepare_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names for seasonality analysis."""
    col_mapping = {
        'S_open_f32': 'open',
        'S_high_f32': 'high',
        'S_low_f32': 'low',
        'S_close_f32': 'close',
        'S_volume_f64': 'volume',
    }
    
    for old_col, new_col in col_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
    
    return df


def analyze_intraday(
    df: pd.DataFrame,
    metric: str = "returns",
    interval_minutes: int = 60,
    filter_weekday: Optional[int] = None,
    filter_month: Optional[int] = None,
) -> Dict[str, Any]:
    """Analyze intraday patterns grouped by time interval."""
    df = calculate_returns(df)
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['weekday'] = df['datetime'].dt.weekday
    df['month'] = df['datetime'].dt.month
    
    if filter_weekday is not None:
        df = df[df['weekday'] == filter_weekday]
    if filter_month is not None:
        df = df[df['month'] == filter_month]
    
    df['time_bucket'] = (df['hour'] * 60 + df['minute']) // interval_minutes * interval_minutes
    
    results = []
    for bucket in sorted(df['time_bucket'].unique()):
        bucket_df = df[df['time_bucket'] == bucket]
        hour = bucket // 60
        minute = bucket % 60
        
        if metric == "returns":
            value = float(bucket_df['return'].mean())
        elif metric == "volatility":
            value = calculate_volatility(bucket_df['return'].values)
        elif metric == "rsi":
            value = calculate_rsi(bucket_df['close'].values)
        else:
            value = float(bucket_df['return'].mean())
        
        results.append({
            'time': f"{hour:02d}:{minute:02d}",
            'hour': hour,
            'minute': minute,
            'value': value,
            'count': len(bucket_df),
        })
    
    weekday_profiles = {}
    for wd in range(7):
        wd_df = df[df['weekday'] == wd]
        wd_results = []
        for bucket in sorted(wd_df['time_bucket'].unique()):
            bucket_df = wd_df[wd_df['time_bucket'] == bucket]
            hour = bucket // 60
            minute = bucket % 60
            
            if metric == "returns":
                value = float(bucket_df['return'].mean())
            elif metric == "volatility":
                value = calculate_volatility(bucket_df['return'].values)
            elif metric == "rsi":
                value = calculate_rsi(bucket_df['close'].values)
            else:
                value = float(bucket_df['return'].mean())
            
            wd_results.append({
                'time': f"{hour:02d}:{minute:02d}",
                'value': value,
            })
        weekday_profiles[wd] = wd_results
    
    return {
        'profile': results,
        'weekday_profiles': weekday_profiles,
    }


def analyze_weekday(
    df: pd.DataFrame,
    metric: str = "returns",
) -> List[Dict[str, Any]]:
    """Analyze patterns grouped by day of week."""
    df = calculate_returns(df)
    df['weekday'] = df['datetime'].dt.weekday
    
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    results = []
    
    for wd in range(7):
        wd_df = df[df['weekday'] == wd]
        if len(wd_df) == 0:
            continue
        
        if metric == "returns":
            value = float(wd_df['return'].mean())
            volatility = calculate_volatility(wd_df['return'].values)
            rsi = calculate_rsi(wd_df['close'].values)
        elif metric == "volatility":
            value = calculate_volatility(wd_df['return'].values)
            volatility = value
            rsi = calculate_rsi(wd_df['close'].values)
        elif metric == "rsi":
            rsi = calculate_rsi(wd_df['close'].values)
            value = rsi
            volatility = calculate_volatility(wd_df['return'].values)
        else:
            value = float(wd_df['return'].mean())
            volatility = calculate_volatility(wd_df['return'].values)
            rsi = calculate_rsi(wd_df['close'].values)
        
        win_rate = float((wd_df['return'] > 0).mean() * 100) if len(wd_df) > 0 else 0
        
        results.append({
            'period': weekday_names[wd],
            'period_type': 'weekday',
            'period_value': wd,
            'value': value,
            'avg_return': float(wd_df['return'].mean()),
            'volatility': volatility,
            'rsi': rsi,
            'count': len(wd_df),
            'win_rate': win_rate,
        })
    
    return results


def analyze_month(
    df: pd.DataFrame,
    metric: str = "returns",
) -> List[Dict[str, Any]]:
    """Analyze patterns grouped by month of year."""
    df = calculate_returns(df)
    df['month'] = df['datetime'].dt.month
    
    month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    results = []
    
    for m in range(1, 13):
        m_df = df[df['month'] == m]
        if len(m_df) == 0:
            continue
        
        if metric == "returns":
            value = float(m_df['return'].mean())
            volatility = calculate_volatility(m_df['return'].values)
            rsi = calculate_rsi(m_df['close'].values)
        elif metric == "volatility":
            value = calculate_volatility(m_df['return'].values)
            volatility = value
            rsi = calculate_rsi(m_df['close'].values)
        elif metric == "rsi":
            rsi = calculate_rsi(m_df['close'].values)
            value = rsi
            volatility = calculate_volatility(m_df['return'].values)
        else:
            value = float(m_df['return'].mean())
            volatility = calculate_volatility(m_df['return'].values)
            rsi = calculate_rsi(m_df['close'].values)
        
        win_rate = float((m_df['return'] > 0).mean() * 100) if len(m_df) > 0 else 0
        
        results.append({
            'period': month_names[m - 1],
            'period_type': 'month',
            'period_value': m,
            'value': value,
            'avg_return': float(m_df['return'].mean()),
            'volatility': volatility,
            'rsi': rsi,
            'count': len(m_df),
            'win_rate': win_rate,
        })
    
    return results


def analyze_day_of_month(
    df: pd.DataFrame,
    metric: str = "returns",
) -> List[Dict[str, Any]]:
    """Analyze patterns grouped by day of month (1-31)."""
    df = calculate_returns(df)
    df['day'] = df['datetime'].dt.day
    
    results = []
    
    for d in range(1, 32):
        d_df = df[df['day'] == d]
        if len(d_df) == 0:
            continue
        
        if metric == "returns":
            value = float(d_df['return'].mean())
            volatility = calculate_volatility(d_df['return'].values)
            rsi = calculate_rsi(d_df['close'].values)
        elif metric == "volatility":
            value = calculate_volatility(d_df['return'].values)
            volatility = value
            rsi = calculate_rsi(d_df['close'].values)
        elif metric == "rsi":
            rsi = calculate_rsi(d_df['close'].values)
            value = rsi
            volatility = calculate_volatility(d_df['return'].values)
        else:
            value = float(d_df['return'].mean())
            volatility = calculate_volatility(d_df['return'].values)
            rsi = calculate_rsi(d_df['close'].values)
        
        win_rate = float((d_df['return'] > 0).mean() * 100) if len(d_df) > 0 else 0
        
        results.append({
            'period': f"Day {d}",
            'period_type': 'day_of_month',
            'period_value': d,
            'value': value,
            'avg_return': float(d_df['return'].mean()),
            'volatility': volatility,
            'rsi': rsi,
            'count': len(d_df),
            'win_rate': win_rate,
        })
    
    return results


def calculate_total_returns_index(
    df: pd.DataFrame,
    custom_window: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Calculate cumulative total returns index (indexed at 100 at t=0)."""
    if len(df) == 0:
        return {
            'data': [],
            'start_date': None,
            'end_date': None,
            'start_index': 100.0,
            'end_index': None,
        }
    
    df = df.copy()
    df['return'] = (df['close'] - df['open']) / df['open'] * 100
    
    if custom_window:
        start_hour, start_min = map(int, custom_window['start'].split(':'))
        end_hour, end_min = map(int, custom_window['end'].split(':'))
        
        start_minutes = start_hour * 60 + start_min
        end_minutes = end_hour * 60 + end_min
        
        df['total_minutes'] = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute
        
        if start_minutes <= end_minutes:
            df = df[(df['total_minutes'] >= start_minutes) & (df['total_minutes'] < end_minutes)]
        else:
            df = df[(df['total_minutes'] >= start_minutes) | (df['total_minutes'] < end_minutes)]
    
    if len(df) == 0:
        return {
            'data': [],
            'start_date': None,
            'end_date': None,
            'start_index': 100.0,
            'end_index': None,
        }
    
    df = df.sort_values('datetime').reset_index(drop=True)
    
    df['date'] = df['datetime'].dt.date
    
    daily_df = df.groupby('date').agg({
        'return': 'sum',
        'datetime': 'first',
    }).reset_index()
    
    daily_df = daily_df.sort_values('datetime').reset_index(drop=True)
    daily_df['cumulative_return'] = (1 + daily_df['return'] / 100).cumprod()
    daily_df['total_returns_index'] = 100 * daily_df['cumulative_return']
    
    daily_df['date_str'] = daily_df['datetime'].dt.strftime('%Y-%m-%d')
    
    return {
        'data': [
            {
                'datetime': row['date_str'],
                'value': to_python_type(row['total_returns_index']),
            }
            for _, row in daily_df.iterrows()
        ],
        'start_date': str(daily_df['datetime'].min()),
        'end_date': str(daily_df['datetime'].max()),
        'start_index': 100.0,
        'end_index': to_python_type(daily_df['total_returns_index'].iloc[-1]),
    }


def generate_calendar_heatmap(
    df: pd.DataFrame,
    year: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate calendar heatmap data (GitHub-style)."""
    df = calculate_returns(df)
    df['date'] = df['datetime'].dt.date
    
    if year is None:
        year = int(df['datetime'].dt.year.max())
    
    year_df = df[df['datetime'].dt.year == year]
    
    daily_returns = year_df.groupby('date').agg({
        'return': 'sum',
        'datetime': 'first'
    }).reset_index()
    
    daily_returns['weekday'] = pd.to_datetime(daily_returns['date']).dt.weekday
    daily_returns['week'] = pd.to_datetime(daily_returns['date']).dt.isocalendar().week
    
    heatmap_data = []
    for _, row in daily_returns.iterrows():
        heatmap_data.append({
            'date': str(row['date']),
            'value': to_python_type(row['return']),
            'weekday': int(row['weekday']),
            'week': int(row['week']),
        })
    
    available_years = [int(y) for y in sorted(df['datetime'].dt.year.unique().tolist())]
    
    return {
        'year': year,
        'data': heatmap_data,
        'available_years': available_years,
    }


def calculate_summary_stats(
    df: pd.DataFrame,
    metric: str = "returns",
    filter_weekday: Optional[int] = None,
    filter_month: Optional[int] = None,
) -> Dict[str, Any]:
    """Calculate summary statistics for the selected configuration."""
    df = calculate_returns(df)
    
    if filter_weekday is not None:
        df = df[df['datetime'].dt.weekday == filter_weekday]
    if filter_month is not None:
        df = df[df['datetime'].dt.month == filter_month]
    
    if len(df) == 0:
        return {
            'win_rate': 0,
            'avg_return': 0,
            'volatility': 0,
            'sharpe': 0,
            'max_drawdown': 0,
            'count': 0,
        }
    
    returns = df['return'].values
    cumulative = np.cumsum(returns)
    
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0
    
    avg_return = float(np.mean(returns))
    volatility = calculate_volatility(returns)
    sharpe = avg_return / volatility if volatility > 0 else 0
    win_rate = float((returns > 0).mean() * 100)
    
    return {
        'win_rate': round(win_rate, 2),
        'avg_return': round(avg_return, 4),
        'volatility': round(volatility, 4),
        'sharpe': round(sharpe, 4),
        'max_drawdown': round(max_drawdown, 4),
        'count': len(df),
    }


def analyze_custom_window(
    df: pd.DataFrame,
    start_time: str,
    end_time: str,
    metric: str = "returns",
) -> Dict[str, Any]:
    """Analyze returns within a custom daily time window."""
    df = calculate_returns(df)
    
    start_hour, start_min = map(int, start_time.split(':'))
    end_hour, end_min = map(int, end_time.split(':'))
    
    start_minutes = start_hour * 60 + start_min
    end_minutes = end_hour * 60 + end_min
    
    df['total_minutes'] = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute
    
    if start_minutes <= end_minutes:
        window_df = df[(df['total_minutes'] >= start_minutes) & (df['total_minutes'] < end_minutes)]
    else:
        window_df = df[(df['total_minutes'] >= start_minutes) | (df['total_minutes'] < end_minutes)]
    
    daily_agg = window_df.groupby(window_df['datetime'].dt.date).agg({
        'return': 'sum',
        'open': 'first',
        'close': 'last',
    }).reset_index()
    
    daily_agg.columns = ['date', 'return', 'open', 'close']
    
    if metric == "returns":
        value = float(daily_agg['return'].mean())
    elif metric == "volatility":
        value = calculate_volatility(daily_agg['return'].values)
    elif metric == "rsi":
        value = calculate_rsi(daily_agg['close'].values)
    else:
        value = float(daily_agg['return'].mean())
    
    return {
        'window': f"{start_time} - {end_time}",
        'value': value,
        'daily_count': len(daily_agg),
        'total_periods': len(window_df),
    }


def get_full_analysis(
    symbol: str,
    freq: str = "candle_1hour",
    metric: str = "returns",
    interval_minutes: int = 60,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    filter_weekday: Optional[int] = None,
    filter_month: Optional[int] = None,
    custom_window: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Run full seasonality analysis for an asset."""
    df = load_asset_data(symbol, freq, start_date, end_date)
    
    if df is None or len(df) == 0:
        return {
            'error': f"No data found for {symbol}",
            'symbol': symbol,
        }
    
    df_window = df.copy()
    if custom_window:
        start_hour, start_min = map(int, custom_window['start'].split(':'))
        end_hour, end_min = map(int, custom_window['end'].split(':'))
        
        start_minutes = start_hour * 60 + start_min
        end_minutes = end_hour * 60 + end_min
        
        df_window['total_minutes'] = df_window['datetime'].dt.hour * 60 + df_window['datetime'].dt.minute
        
        if start_minutes <= end_minutes:
            df_window = df_window[(df_window['total_minutes'] >= start_minutes) & (df_window['total_minutes'] < end_minutes)]
        else:
            df_window = df_window[(df_window['total_minutes'] >= start_minutes) | (df_window['total_minutes'] < end_minutes)]
        
        df_window = df_window.drop(columns=['total_minutes'])
    
    intraday = analyze_intraday(
        df_window, metric, interval_minutes, filter_weekday, filter_month
    )
    
    weekday = analyze_weekday(df_window, metric)
    month = analyze_month(df_window, metric)
    day_of_month = analyze_day_of_month(df_window, metric)
    calendar = generate_calendar_heatmap(df)
    summary = calculate_summary_stats(df_window, metric, filter_weekday, filter_month)
    total_returns_index = calculate_total_returns_index(df, custom_window)
    
    result = {
        'symbol': symbol,
        'freq': freq,
        'metric': metric,
        'interval_minutes': interval_minutes,
        'total_rows': len(df_window),
        'date_range': {
            'start': str(df_window['datetime'].min()) if len(df_window) > 0 else None,
            'end': str(df_window['datetime'].max()) if len(df_window) > 0 else None,
        },
        'intraday': intraday,
        'weekday': weekday,
        'month': month,
        'day_of_month': day_of_month,
        'calendar': calendar,
        'summary': summary,
        'total_returns_index': total_returns_index,
    }
    
    if custom_window:
        custom_result = analyze_custom_window(
            df, custom_window['start'], custom_window['end'], metric
        )
        result['custom_window'] = custom_result
    
    return to_python_type(result)


def search_assets_for_seasonality(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Search for assets available for seasonality analysis."""
    from core.search_data import search_data
    
    results = search_data()
    
    assets = {}
    query_lower = query.lower() if query else ''
    
    for r in results:
        key = r.instrument
        if query_lower and query_lower not in key.lower():
            continue
        if key not in assets:
            assets[key] = {
                'symbol': r.instrument,
                'product_type': r.product_type,
                'frequencies': [],
            }
        freq = r.data_freq
        if freq and freq not in assets[key]['frequencies']:
            assets[key]['frequencies'].append(freq)
    
    for key in assets:
        freq_order = {'candle_1min': 0, 'candle_1hour': 1, 'candle_1day': 2}
        assets[key]['frequencies'] = sorted(
            assets[key]['frequencies'],
            key=lambda x: freq_order.get(x, 99)
        )
        if assets[key]['frequencies']:
            assets[key]['default_freq'] = assets[key]['frequencies'][0]
        else:
            assets[key]['default_freq'] = 'candle_1hour'
    
    sorted_assets = sorted(assets.values(), key=lambda x: x['symbol'])
    
    return to_python_type(sorted_assets[:limit])
