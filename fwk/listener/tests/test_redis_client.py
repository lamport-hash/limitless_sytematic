import pytest
import fakeredis
from app.listener_redis_client import RedisClient, MktDataType
from core.classes import (
    NormalizedOrderBook,
    NormalizedTrade,
    NormalizedCandle,
)
from datetime import datetime, date


@pytest.fixture
def redis_client():
    """Create RedisClient with fakeredis for testing"""
    fake_redis = fakeredis.FakeRedis()
    client = RedisClient.__new__(RedisClient)
    client.redis = fake_redis
    client.listener_id = "test_listener"
    client.connector_type = "test_connector"
    return client


class TestRedisClient:
    def test_init(self, redis_client):
        assert redis_client.redis is not None
        # Test that we can ping the redis instance
        assert redis_client.redis.ping() is True

    def test_generate_key(self, redis_client):
        key = redis_client._generate_key("binance", "BTC/USDT", 1234567890, "orderbook")
        assert key == "orderbook:binance:BTC/USDT:1234567890"

    def test_generate_ts_key(self, redis_client):
        key = redis_client._generate_ts_key("binance", "BTC/USDT", "orderbook")
        assert key == "orderbook_timestamps:binance:BTC/USDT"

    def test_flush_all(self, redis_client):
        # Store some test data
        redis_client.redis.set("test_key", "test_value")
        assert redis_client.redis.exists("test_key")

        # Flush all
        result = redis_client.flush_all()
        assert result is True
        assert not redis_client.redis.exists("test_key")

    def test_flush_all_key(self, redis_client):
        # Store some test keys with pattern
        redis_client.redis.set("orderbook:key1", "value1")
        redis_client.redis.set("orderbook:key2", "value2")
        redis_client.redis.set("other:key3", "value3")

        # Flush keys with 'orderbook:' pattern
        deleted = redis_client.flush_all_key("orderbook")
        assert deleted == 2
        assert not redis_client.redis.exists("orderbook:key1")
        assert not redis_client.redis.exists("orderbook:key2")
        assert redis_client.redis.exists("other:key3")

    def test_store_and_get_orderbook(self, redis_client):
        orderbook = NormalizedOrderBook(
            bids=[[100.0, 1.0], [99.0, 2.0]],
            asks=[[101.0, 1.0], [102.0, 2.0]],
            timestamp=1234567890000,
        )

        # Store orderbook
        redis_client.store_orderbook("binance", "BTC/USDT", orderbook)

        # Get market data
        retrieved = redis_client.get_market_data(
            "binance", "BTC/USDT", MktDataType.ORDERBOOK
        )
        assert retrieved is not None
        assert retrieved.timestamp == orderbook.timestamp
        assert retrieved.bids == orderbook.bids
        assert retrieved.asks == orderbook.asks

    def test_store_and_get_trades(self, redis_client):
        trades = [
            NormalizedTrade(
                trade_id="123",
                price=100.0,
                amount=1.0,
                timestamp=1234567890000,
                side="buy",
            ),
            NormalizedTrade(
                trade_id="124",
                price=101.0,
                amount=2.0,
                timestamp=1234567891000,
                side="sell",
            ),
        ]

        # Store trades
        redis_client.store_trades("binance", "BTC/USDT", trades)

        # Get market data
        retrieved = redis_client.get_market_data(
            "binance", "BTC/USDT", MktDataType.TRADE, n_last=2
        )
        assert retrieved is not None
        assert len(retrieved) == 2
        assert retrieved[0].trade_id == "124"  # Most recent first
        assert retrieved[1].trade_id == "123"

    def test_store_and_get_candle(self, redis_client):
        candle = NormalizedCandle(
            timestamp=1234567890000,
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000.0,
        )

        # Store candle
        redis_client.store_candle("binance", "BTC/USDT", candle)

        # Set last candle time (since store_candle doesn't do this automatically)
        redis_client.set_last_candle_time("binance", "BTC/USDT", candle.timestamp)

        # Get candle
        retrieved = redis_client.get_candle("binance", "BTC/USDT")
        assert retrieved is not None
        assert retrieved.timestamp == candle.timestamp
        assert retrieved.open == candle.open
        assert retrieved.close == candle.close

    def test_get_candles(self, redis_client):
        candles = [
            NormalizedCandle(
                timestamp=1234567890000,
                open=100.0,
                high=105.0,
                low=95.0,
                close=102.0,
                volume=1000.0,
            ),
            NormalizedCandle(
                timestamp=1234567891000,
                open=102.0,
                high=110.0,
                low=100.0,
                close=108.0,
                volume=1500.0,
            ),
        ]

        # Store candles
        for candle in candles:
            redis_client.store_candle("binance", "BTC/USDT", candle)

        # Get candles
        retrieved = redis_client.get_candles("binance", "BTC/USDT", 2)
        assert retrieved is not None
        assert len(retrieved) == 2
        assert retrieved[0].timestamp == 1234567891000  # Most recent first
        assert retrieved[1].timestamp == 1234567890000

    def test_get_all_symbols(self, redis_client):
        # Store some orderbook data
        orderbook = NormalizedOrderBook(
            bids=[[100.0, 1.0]], asks=[[101.0, 1.0]], timestamp=1234567890000
        )
        redis_client.store_orderbook("binance", "BTC/USDT", orderbook)
        redis_client.store_orderbook("binance", "ETH/USDT", orderbook)

        symbols = redis_client.get_all_symbols()
        assert len(symbols) == 2
        # Convert to set for easier comparison since order might vary
        symbol_set = set(symbols)
        assert ("binance", "BTC/USDT") in symbol_set
        assert ("binance", "ETH/USDT") in symbol_set

    def test_last_candle_time(self, redis_client):
        # Set last candle time
        redis_client.set_last_candle_time("binance", "BTC/USDT", 1234567890000)

        # Get last candle time
        retrieved = redis_client.get_last_candle_time("binance", "BTC/USDT")
        assert retrieved == 1234567890000

    def test_get_daily_candle_counts(self, redis_client):
        # Create candles for different days
        base_time = int(datetime(2023, 1, 1).timestamp() * 1000)
        candles = [
            NormalizedCandle(
                timestamp=base_time, open=100, high=105, low=95, close=102, volume=1000
            ),
            NormalizedCandle(
                timestamp=base_time + 86400000,
                open=102,
                high=110,
                low=100,
                close=108,
                volume=1500,
            ),  # Next day
        ]

        for candle in candles:
            redis_client.store_candle("binance", "BTC/USDT", candle)

        counts = redis_client.get_daily_candle_counts("binance", "BTC/USDT", 2)
        assert len(counts) == 2
        # Should have counts for today and yesterday
        assert all(count >= 0 for count in counts.values())

    def test_get_element_for_date_delete(self, redis_client):
        # Create test data for a specific date
        test_date = date(2023, 1, 1)
        timestamp = int(
            datetime.combine(test_date, datetime.min.time()).timestamp() * 1000
        )

        orderbook = NormalizedOrderBook(
            bids=[[100.0, 1.0]], asks=[[101.0, 1.0]], timestamp=timestamp
        )
        redis_client.store_orderbook("binance", "BTC/USDT", orderbook)

        # Get elements for date
        msg, timestamps = redis_client.get_element_for_date_delete(
            "binance", "BTC/USDT", "orderbook", test_date
        )
        assert len(timestamps) == 1
        assert timestamps[0].decode("utf-8") == str(timestamp)

    def test_analyze_date_range(self, redis_client):
        # Create test data for multiple days
        dates = [date(2023, 1, 1), date(2023, 1, 2)]
        for i, test_date in enumerate(dates):
            timestamp = (
                int(datetime.combine(test_date, datetime.min.time()).timestamp() * 1000)
                + i * 1000
            )
            orderbook = NormalizedOrderBook(
                bids=[[100.0 + i, 1.0]], asks=[[101.0 + i, 1.0]], timestamp=timestamp
            )
            redis_client.store_orderbook("binance", "BTC/USDT", orderbook)

        results = redis_client.analyze_date_range(
            "binance", "BTC/USDT", "orderbook", dates[0], dates[1]
        )
        assert len(results) == 2
        for result in results:
            assert "date" in result
            assert "key_count" in result
            assert "percentage_coverage" in result
