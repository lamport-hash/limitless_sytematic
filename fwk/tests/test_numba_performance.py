"""
Performance benchmarks for numba-optimized feature calculations.
"""
import pytest
import pandas as pd
import numpy as np
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.feature_ta_utils import (
    numba_ema,
    numba_atr,
    numba_rsi,
    numba_sma,
    numba_rolling_max,
    numba_rolling_min,
    numba_macd,
    numba_crossover_detect,
    calculate_atr,
)


def generate_test_data(n_bars=10000):
    """Generate synthetic OHLCV data for benchmarking."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n_bars) * 0.1)
    high = close + np.abs(np.random.randn(n_bars) * 0.05)
    low = close - np.abs(np.random.randn(n_bars) * 0.05)
    volume = np.random.randint(1000, 10000, n_bars)
    
    return pd.DataFrame({
        'S_high_f32': high,
        'S_low_f32': low,
        'S_close_f32': close,
        'S_volume_f64': volume,
    })


class TestATRPerformance:
    """Benchmark ATR calculations."""
    
    def test_atr_correctness(self):
        """Verify numba_atr produces valid results."""
        df = generate_test_data(1000)
        
        # Warmup JIT
        _ = calculate_atr(df.head(100), period=14)
        
        # Test numba ATR
        atr_numba = calculate_atr(df, period=14)
        
        assert len(atr_numba) == len(df)
        assert np.all(np.isfinite(atr_numba[20:])), "ATR should be finite after warmup"
        assert np.all(atr_numba >= 0), "ATR should be non-negative"
    
    def test_atr_performance(self):
        """Benchmark numba ATR performance."""
        df = generate_test_data(10000)
        
        # Warmup JIT
        _ = calculate_atr(df.head(100), period=14)
        
        # Benchmark numba
        start = time.time()
        for _ in range(10):
            _ = calculate_atr(df, period=14)
        numba_time = time.time() - start
        
        print(f"\nATR Performance (10 iterations): {numba_time:.4f}s")
        print(f"  Avg per iteration: {numba_time/10:.4f}s")
        print(f"  Throughput: {len(df)*10/numba_time:.0f} bars/second")
        
        # Should be very fast
        assert numba_time < 0.1, f"ATR should complete in <0.1s (got {numba_time:.4f}s)"


class TestEMAPerformance:
    """Benchmark EMA calculations."""
    
    def test_ema_correctness(self):
        """Verify numba_ema produces valid results."""
        prices = np.random.randn(1000).cumsum() + 100
        
        # Warmup
        _ = numba_ema(prices[:100], 20)
        
        # Test numba EMA
        ema_numba = numba_ema(prices, 20)
        
        assert len(ema_numba) == len(prices)
        assert np.all(np.isfinite(ema_numba[20:])), "EMA should be finite after warmup"
        
        # Compare with pandas for correctness (skip warmup period)
        ema_pandas = pd.Series(prices).ewm(span=20, adjust=False).mean().values
        np.testing.assert_allclose(
            ema_numba[50:],
            ema_pandas[50:],
            rtol=0.01,
            err_msg="numba_ema differs from pandas ewm"
        )
    
    def test_ema_performance(self):
        """Benchmark numba vs pandas EMA."""
        prices = np.random.randn(10000).cumsum() + 100
        
        # Warmup JIT
        _ = numba_ema(prices[:100], 20)
        
        # Benchmark numba
        start = time.time()
        for _ in range(100):
            _ = numba_ema(prices, 20)
        numba_time = time.time() - start
        
        # Benchmark pandas
        start = time.time()
        for _ in range(100):
            _ = pd.Series(prices).ewm(span=20, adjust=False).mean()
        pandas_time = time.time() - start
        
        speedup = pandas_time / numba_time
        print(f"\nEMA Performance (100 iterations):")
        print(f"  pandas: {pandas_time:.4f}s")
        print(f"  numba: {numba_time:.4f}s")
        print(f"  speedup: {speedup:.2f}x")
        
        assert speedup > 2, f"numba should be at least 2x faster (got {speedup:.2f}x)"


class TestRSIPerformance:
    """Benchmark RSI calculations."""
    
    def test_rsi_correctness(self):
        """Verify numba_rsi produces valid results."""
        prices = np.random.randn(1000).cumsum() + 100
        
        # Warmup
        _ = numba_rsi(prices[:100], 14)
        
        # Test numba RSI
        rsi_numba = numba_rsi(prices, 14)
        
        assert len(rsi_numba) == len(prices)
        assert np.all(rsi_numba >= 0) & np.all(rsi_numba <= 100), "RSI should be in [0, 100]"
        assert np.all(np.isfinite(rsi_numba[20:])), "RSI should be finite after warmup"
    
    def test_rsi_performance(self):
        """Benchmark numba RSI."""
        prices = np.random.randn(10000).cumsum() + 100
        
        # Warmup JIT
        _ = numba_rsi(prices[:100], 14)
        
        # Benchmark numba
        start = time.time()
        for _ in range(100):
            _ = numba_rsi(prices, 14)
        numba_time = time.time() - start
        
        print(f"\nRSI Performance (100 iterations): {numba_time:.4f}s")
        print(f"  Avg per iteration: {numba_time/100:.4f}s")
        
        assert numba_time < 0.5, f"RSI should complete in <0.5s (got {numba_time:.4f}s)"


class TestMACDPerformance:
    """Benchmark MACD calculations."""
    
    def test_macd_correctness(self):
        """Verify numba_macd produces valid results."""
        prices = np.random.randn(1000).cumsum() + 100
        
        # Warmup
        _ = numba_macd(prices[:100], 12, 26, 9)
        
        # Test numba MACD
        macd_line, signal_line, histogram = numba_macd(prices, 12, 26, 9)
        
        assert len(macd_line) == len(prices)
        assert len(signal_line) == len(prices)
        assert len(histogram) == len(prices)
        
        # Verify histogram calculation
        np.testing.assert_allclose(
            histogram[50:],
            (macd_line - signal_line)[50:],
            rtol=0.001,
            err_msg="MACD histogram calculation incorrect"
        )
    
    def test_macd_performance(self):
        """Benchmark numba MACD."""
        prices = np.random.randn(10000).cumsum() + 100
        
        # Warmup JIT
        _ = numba_macd(prices[:100], 12, 26, 9)
        
        # Benchmark numba
        start = time.time()
        for _ in range(100):
            _ = numba_macd(prices, 12, 26, 9)
        numba_time = time.time() - start
        
        print(f"\nMACD Performance (100 iterations): {numba_time:.4f}s")
        print(f"  Avg per iteration: {numba_time/100:.4f}s")
        
        # Should be very fast
        assert numba_time < 0.3, f"MACD should complete in <0.3s (got {numba_time:.4f}s)"


class TestRollingOperations:
    """Benchmark rolling operations."""
    
    def test_rolling_max_correctness(self):
        """Verify numba_rolling_max matches pandas."""
        values = np.random.randn(1000).cumsum() + 100
        
        # Warmup
        _ = numba_rolling_max(values[:100], 20)
        
        # Test numba rolling max
        numba_max = numba_rolling_max(values, 20)
        
        # Compare with pandas (skip warmup period)
        pandas_max = pd.Series(values).rolling(20).max().values
        np.testing.assert_allclose(
            pandas_max[25:],
            numba_max[25:],
            rtol=0.001,
            err_msg="numba_rolling_max differs from pandas rolling"
        )
    
    def test_rolling_min_correctness(self):
        """Verify numba_rolling_min matches pandas."""
        values = np.random.randn(1000).cumsum() + 100
        
        # Warmup
        _ = numba_rolling_min(values[:100], 20)
        
        # Test numba rolling min
        numba_min = numba_rolling_min(values, 20)
        
        # Compare with pandas (skip warmup period)
        pandas_min = pd.Series(values).rolling(20).min().values
        np.testing.assert_allclose(
            pandas_min[25:],
            numba_min[25:],
            rtol=0.001,
            err_msg="numba_rolling_min differs from pandas rolling"
        )
    
    def test_rolling_performance(self):
        """Benchmark rolling operations."""
        values = np.random.randn(10000).cumsum() + 100
        
        # Warmup JIT
        _ = numba_rolling_max(values[:100], 20)
        _ = numba_rolling_min(values[:100], 20)
        _ = numba_sma(values[:100], 20)
        
        # Benchmark numba
        start = time.time()
        for _ in range(50):
            _ = numba_rolling_max(values, 20)
            _ = numba_rolling_min(values, 20)
            _ = numba_sma(values, 20)
        numba_time = time.time() - start
        
        # Benchmark pandas
        start = time.time()
        for _ in range(50):
            _ = pd.Series(values).rolling(20).max()
            _ = pd.Series(values).rolling(20).min()
            _ = pd.Series(values).rolling(20).mean()
        pandas_time = time.time() - start
        
        speedup = pandas_time / numba_time
        print(f"\nRolling Operations (50 iterations):")
        print(f"  pandas: {pandas_time:.4f}s")
        print(f"  numba: {numba_time:.4f}s")
        print(f"  speedup: {speedup:.2f}x")
        
        assert speedup > 1.2, f"numba should be at least 1.2x faster (got {speedup:.2f}x)"


class TestCrossoverDetection:
    """Benchmark crossover detection."""
    
    def test_crossover_correctness(self):
        """Verify crossover detection works correctly."""
        # Create known crossover pattern
        fast = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0])
        slow = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        
        signals = numba_crossover_detect(fast, slow)
        
        # Index 2: fast crosses above slow (1->1)
        # Index 4: fast crosses below slow (-1->-1)
        # Index 6: fast crosses above slow (1->1)
        # Index 8: fast crosses below slow (-1->-1)
        assert signals[2] == 1, f"Should detect bullish crossover at index 2, got {signals[2]}"
        assert signals[4] == -1, f"Should detect bearish crossover at index 4, got {signals[4]}"
        assert signals[6] == 1, f"Should detect bullish crossover at index 6, got {signals[6]}"
        assert signals[8] == -1, f"Should detect bearish crossover at index 8, got {signals[8]}"
    
    def test_crossover_performance(self):
        """Benchmark crossover detection."""
        fast = np.random.randn(10000).cumsum() + 100
        slow = np.random.randn(10000).cumsum() + 100
        
        # Warmup JIT
        _ = numba_crossover_detect(fast[:100], slow[:100])
        
        start = time.time()
        for _ in range(1000):
            _ = numba_crossover_detect(fast, slow)
        numba_time = time.time() - start
        
        print(f"\nCrossover Detection (1000 iterations): {numba_time:.4f}s")
        print(f"  Avg per iteration: {numba_time/1000:.6f}s")
        print(f"  Throughput: {len(fast)*1000/numba_time:.0f} bars/second")
        
        assert numba_time < 0.1, f"Crossover detection should be very fast (got {numba_time:.4f}s)"


class TestStrategyIntegration:
    """End-to-end strategy performance tests."""
    
    def test_ema_crossover_strategy_performance(self):
        """Benchmark full EMA crossover strategy."""
        try:
            from strat.s_ema_crossover import build_features
        except ImportError:
            pytest.skip("s_ema_crossover module not available")
            return
        
        # Generate test data
        df = generate_test_data(5000)
        df['S_open_f32'] = df['S_close_f32'] + np.random.randn(len(df)) * 0.02
        df['i_minute_i'] = range(len(df))
        
        # Warmup
        _ = build_features(df.head(100).copy(), p_fast_period=12, p_slow_period=50)
        
        # Benchmark
        start = time.time()
        for _ in range(10):
            _ = build_features(df.copy(), p_fast_period=12, p_slow_period=50)
        elapsed = time.time() - start
        
        print(f"\nEMA Crossover Strategy (5000 bars, 10 iterations): {elapsed:.4f}s")
        print(f"  Avg per iteration: {elapsed/10:.4f}s")
        print(f"  Throughput: {len(df)*10/elapsed:.0f} bars/second")
        assert elapsed < 1.0, f"Strategy should complete in <1s (got {elapsed:.4f}s)"
    
    def test_macd_strategy_performance(self):
        """Benchmark full MACD strategy."""
        try:
            from strat.s_macd_crossover import build_features
        except ImportError:
            pytest.skip("s_macd_crossover module not available")
            return
        
        df = generate_test_data(5000)
        df['S_open_f32'] = df['S_close_f32'] + np.random.randn(len(df)) * 0.02
        df['i_minute_i'] = range(len(df))
        
        # Warmup
        _ = build_features(df.head(100).copy())
        
        # Benchmark
        start = time.time()
        for _ in range(10):
            _ = build_features(df.copy())
        elapsed = time.time() - start
        
        print(f"\nMACD Strategy (5000 bars, 10 iterations): {elapsed:.4f}s")
        print(f"  Avg per iteration: {elapsed/10:.4f}s")
        print(f"  Throughput: {len(df)*10/elapsed:.0f} bars/second")
        assert elapsed < 1.0, f"Strategy should complete in <1s (got {elapsed:.4f}s)"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
