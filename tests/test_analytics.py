"""Tests for market microstructure analytics."""

import pytest

from src.core.analytics import Analytics
from src.core.market_data import MarketDataFeed, TradeRecord
from src.core.order import Order, OrderSide, OrderType
from src.core.orderbook import OrderBook


def build_feed_with_trades(prices: list[float], quantities: list[float] | None = None) -> MarketDataFeed:
    """Create a MarketDataFeed with manually injected trades for testing."""
    book = OrderBook()
    feed = MarketDataFeed(book)

    if quantities is None:
        quantities = [10.0] * len(prices)

    for i, (p, q) in enumerate(zip(prices, quantities)):
        feed.trades.append(TradeRecord(
            timestamp=i,
            price=p,
            quantity=q,
            aggressor_side="BUY" if i % 2 == 0 else "SELL",
        ))
        # Also add a snapshot for each trade
        feed.set_timestep(i)
        # Manually append a snapshot with a mid price equal to the trade price
        from src.core.market_data import L2Snapshot
        feed.snapshots.append(L2Snapshot(
            timestamp=i,
            bids=[(p - 0.5, 100.0)],
            asks=[(p + 0.5, 100.0)],
            mid_price=p,
            spread=1.0,
        ))

    return feed


class TestVWAP:
    """Test VWAP calculation."""

    def test_uniform_vwap(self):
        """VWAP with equal quantities equals simple average."""
        feed = build_feed_with_trades([100.0, 102.0, 104.0], [10.0, 10.0, 10.0])
        analytics = Analytics(feed)
        assert analytics.vwap() == pytest.approx(102.0)

    def test_weighted_vwap(self):
        """VWAP should be volume-weighted."""
        feed = build_feed_with_trades([100.0, 200.0], [90.0, 10.0])
        analytics = Analytics(feed)
        expected = (100.0 * 90.0 + 200.0 * 10.0) / 100.0
        assert analytics.vwap() == pytest.approx(expected)

    def test_empty_vwap(self):
        book = OrderBook()
        feed = MarketDataFeed(book)
        analytics = Analytics(feed)
        assert analytics.vwap() is None


class TestOrderFlowImbalance:
    """Test Order Flow Imbalance."""

    def test_balanced_ofi(self):
        """Alternating buy/sell with equal volume should give OFI near 0."""
        feed = build_feed_with_trades(
            [100.0, 100.0, 100.0, 100.0],
            [10.0, 10.0, 10.0, 10.0],
        )
        analytics = Analytics(feed)
        ofi = analytics.order_flow_imbalance(window=4)
        # Trades alternate BUY/SELL, so buy_vol = sell_vol
        assert ofi == pytest.approx(0.0)

    def test_ofi_all_buys(self):
        """If all trades are buy-initiated, OFI = 1."""
        feed = build_feed_with_trades([100.0, 101.0, 102.0])
        # Override aggressor sides to all BUY
        for t in feed.trades:
            t.aggressor_side = "BUY"
        analytics = Analytics(feed)
        assert analytics.order_flow_imbalance(window=10) == pytest.approx(1.0)

    def test_ofi_empty(self):
        book = OrderBook()
        feed = MarketDataFeed(book)
        analytics = Analytics(feed)
        assert analytics.order_flow_imbalance() is None


class TestKylesLambda:
    """Test Kyle's lambda estimation."""

    def test_kyles_lambda_returns_value(self):
        """With enough trades, Kyle's lambda should return a finite value."""
        prices = [100.0 + 0.1 * i for i in range(60)]
        feed = build_feed_with_trades(prices)
        analytics = Analytics(feed)
        lam = analytics.kyles_lambda(window=50)
        assert lam is not None
        assert isinstance(lam, float)

    def test_kyles_lambda_too_few_trades(self):
        feed = build_feed_with_trades([100.0, 101.0])
        analytics = Analytics(feed)
        assert analytics.kyles_lambda(window=50) is None


class TestBidAskBounce:
    """Test bid-ask bounce calculation."""

    def test_bid_ask_bounce_returns_value(self):
        # Create prices that bounce (negative autocorrelation)
        prices = [100.0, 100.5, 100.0, 100.5, 100.0] * 12
        feed = build_feed_with_trades(prices)
        analytics = Analytics(feed)
        bab = analytics.bid_ask_bounce(window=50)
        assert bab is not None
        # Bouncing pattern should show negative autocorrelation
        assert bab < 0

    def test_bid_ask_bounce_too_few(self):
        feed = build_feed_with_trades([100.0])
        analytics = Analytics(feed)
        assert analytics.bid_ask_bounce(window=50) is None


class TestAnalyticsSummary:
    """Test the summary method."""

    def test_summary_returns_dict(self):
        feed = build_feed_with_trades([100.0 + i for i in range(100)])
        analytics = Analytics(feed)
        summary = analytics.summary()
        assert isinstance(summary, dict)
        assert "vwap" in summary
        assert "kyles_lambda" in summary
        assert "order_flow_imbalance" in summary
        assert "num_trades" in summary
