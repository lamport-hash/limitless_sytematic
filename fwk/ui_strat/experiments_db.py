"""
Experiments Database Manager.

SQLite database for storing and retrieving backtest experiments.
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_DIR = Path(os.getenv("EXPERIMENTS_DIR", "/app/data/experiments"))
DB_PATH = DB_DIR / "experiments.db"


def _ensure_db_dir():
    DB_DIR.mkdir(parents=True, exist_ok=True)


def _get_connection() -> sqlite3.Connection:
    _ensure_db_dir()
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the experiments database."""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            filename TEXT NOT NULL,
            strategy_type TEXT NOT NULL,
            params TEXT NOT NULL,
            total_return REAL,
            cagr REAL,
            max_dd REAL,
            win_rate REAL,
            orders INTEGER,
            lookback INTEGER,
            top_n INTEGER,
            selected_assets TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_experiment(
    filename: str,
    strategy_type: str,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    selected_assets: List[str],
    name: Optional[str] = None,
) -> int:
    """
    Save an experiment to the database.
    
    Returns the experiment ID.
    """
    init_db()
    conn = _get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO experiments (
            name, filename, strategy_type, params, total_return, cagr, max_dd,
            win_rate, orders, lookback, top_n, selected_assets, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        name,
        filename,
        strategy_type,
        json.dumps(params),
        metrics.get('total_return'),
        metrics.get('cagr'),
        metrics.get('max_drawdown'),
        metrics.get('win_rate'),
        metrics.get('orders_count'),
        params.get('lookback'),
        params.get('top_n'),
        json.dumps(selected_assets),
        datetime.utcnow().isoformat()
    ))
    
    exp_id = cursor.lastrowid or 0
    conn.commit()
    conn.close()
    return exp_id


def list_experiments(limit: int = 100) -> List[Dict[str, Any]]:
    """
    List all saved experiments.
    
    Returns list of experiment records.
    """
    init_db()
    conn = _get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, name, filename, strategy_type, params, total_return, cagr, max_dd,
               win_rate, orders, lookback, top_n, selected_assets, created_at
        FROM experiments
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    experiments = []
    for row in rows:
        exp = {
            'id': row['id'],
            'name': row['name'],
            'filename': row['filename'],
            'strategy_type': row['strategy_type'],
            'params': json.loads(row['params']) if row['params'] else {},
            'total_return': row['total_return'],
            'cagr': row['cagr'],
            'max_dd': row['max_dd'],
            'win_rate': row['win_rate'],
            'orders': row['orders'],
            'lookback': row['lookback'],
            'top_n': row['top_n'],
            'selected_assets': json.loads(row['selected_assets']) if row['selected_assets'] else [],
            'created_at': row['created_at']
        }
        experiments.append(exp)
    
    return experiments


def get_experiment(exp_id: int) -> Optional[Dict[str, Any]]:
    """Get a single experiment by ID."""
    init_db()
    conn = _get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, name, filename, strategy_type, params, total_return, cagr, max_dd,
               win_rate, orders, lookback, top_n, selected_assets, created_at
        FROM experiments
        WHERE id = ?
    """, (exp_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row is None:
        return None
    
    return {
        'id': row['id'],
        'name': row['name'],
        'filename': row['filename'],
        'strategy_type': row['strategy_type'],
        'params': json.loads(row['params']) if row['params'] else {},
        'total_return': row['total_return'],
        'cagr': row['cagr'],
        'max_dd': row['max_dd'],
        'win_rate': row['win_rate'],
        'orders': row['orders'],
        'lookback': row['lookback'],
        'top_n': row['top_n'],
        'selected_assets': json.loads(row['selected_assets']) if row['selected_assets'] else [],
        'created_at': row['created_at']
    }


def delete_experiment(exp_id: int) -> bool:
    """Delete an experiment by ID."""
    init_db()
    conn = _get_connection()
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM experiments WHERE id = ?", (exp_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted
