import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
import ccxt.pro as ccxtpro
from app.exchange_ccxt_ws import (
    stream_trades,
    stream_orderbook,
    stream_all,
    stop_event,
)
from app.listener_redis_client import redis_client


class TestStreamFunctions:
    @pytest.mark.asyncio
    async def test_stream_trades(self):
        exchange = ccxtpro.binance()
        mock_redis = Mock()
        mock_redis.store_market_data = Mock()

        data = [
            {
                "id": "1",
                "price": 100.0,
                "amount": 1.0,
                "timestamp": 1640995200000,
                "side": "buy",
            }
        ]

        with patch.object(
            exchange,
            "watch_trades",
            new_callable=AsyncMock,
            side_effect=[data, asyncio.CancelledError()],
        ):
            try:
                await asyncio.wait_for(
                    stream_trades(
                        exchange,
                        "BTCUSDT",
                        mock_redis,
                        "test_listener",
                        "ccxt-ws",
                        "BTCUSDT",
                        "binance",
                    ),
                    timeout=0.1,
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

            mock_redis.store_market_data.assert_called()

    @pytest.mark.asyncio
    async def test_stream_orderbook(self):
        exchange = ccxtpro.binance()
        mock_redis = Mock()
        mock_redis.store_market_data = Mock()

        data = {
            "bids": [[100.0, 1.0]],
            "asks": [[101.0, 1.0]],
            "timestamp": 1640995200000,
        }

        with (
            patch.object(
                exchange,
                "watch_order_book",
                new_callable=AsyncMock,
                side_effect=[data, asyncio.CancelledError()],
            ),
            patch.object(exchange, "milliseconds", return_value=1640995200000),
        ):
            try:
                await asyncio.wait_for(
                    stream_orderbook(
                        exchange,
                        "BTCUSDT",
                        mock_redis,
                        "test_listener",
                        "ccxt-ws",
                        "BTCUSDT",
                        "binance",
                    ),
                    timeout=0.1,
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

            mock_redis.store_market_data.assert_called()


class TestStreamAll:
    @pytest.mark.asyncio
    async def test_stream_all(self):
        symbols = [("binance", "BTCUSDT", "ccxt-ws", "BTCUSDT")]
        task = asyncio.create_task(
            stream_all(
                "binance",
                symbols,
                redis_client,
                "test_listener",
                "ccxt-ws",
                {"trades", "ob"},
                verbose=True,
                store_to_redis=False,
            )
        )
        await asyncio.sleep(10.0)
        stop_event.set()
        await task

        # With real connection, it attempts to connect and stream
