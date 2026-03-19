"""
Trade and position tracking for backtest results.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd


def minutes_to_datetime(minutes_since_2000):
    start_date = datetime(2000, 1, 1)
    return start_date + timedelta(minutes=float(minutes_since_2000))


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
    
    has_signal_type = 'signal_type' in orders_sorted.columns
    
    for asset in assets:
        asset_orders = orders_sorted[orders_sorted['etf'] == asset].copy()
        
        if len(asset_orders) == 0:
            continue
        
        open_lots = []
        
        for _, order in asset_orders.iterrows():
            if order['direction'] > 0:
                signal_type = int(order.get('signal_type', 1)) if has_signal_type else 1
                open_lots.append({
                    'qty': float(order['size']),
                    'entry_price': float(order['price']),
                    'entry_row': int(order['timestamp']),
                    'signal_type': signal_type
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
                    side = 'long' if lot['signal_type'] > 0 else ('short' if lot['signal_type'] < 0 else 'unknown')
                    
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
                        'side': side,
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
                
                side = 'long' if lot['signal_type'] > 0 else ('short' if lot['signal_type'] < 0 else 'unknown')
                
                positions.append({
                    'asset': asset,
                    'side': side,
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
    has_signal_type = 'signal_type' in orders_sorted.columns
    
    for asset in active_assets:
        asset_orders = orders_sorted[orders_sorted['etf'] == asset].copy()
        
        if len(asset_orders) == 0:
            continue
        
        open_lots = []
        
        for _, order in asset_orders.iterrows():
            if order['direction'] > 0:
                signal_type = int(order.get('signal_type', 1)) if has_signal_type else 1
                open_lots.append({
                    'qty': float(order['size']),
                    'entry_price': float(order['price']),
                    'entry_row': int(order['timestamp']),
                    'signal_type': signal_type
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
                
                side = 'long' if lot['signal_type'] > 0 else ('short' if lot['signal_type'] < 0 else 'unknown')
                
                positions.append({
                    'asset': asset,
                    'side': side,
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
