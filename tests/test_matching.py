"""Tests for the matching engine: price-time priority and order type semantics."""

import pytest

from src.core.order import Order, OrderSide, OrderStatus, OrderType
from src.core.orderbook import OrderBook


class TestPriceTimePriority:
    """Verify price-time priority matching semantics."""

    def test_buy_market_matches_best_ask(self):
        """A market buy should fill at the best (lowest) ask price."""
        book = OrderBook()
        book.submit(Order(side=OrderSide.SELL, order_type=OrderType.LIMIT, price=101.0, quantity=10.0))
        book.submit(Order(side=OrderSide.SELL, order_type=OrderType.LIMIT, price=102.0, quantity=10.0))

        buy = Order(side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=5.0)
        trades = book.submit(buy)

        assert len(trades) == 1
        assert trades[0].price == 101.0
        assert trades[0].quantity == pytest.approx(5.0)

    def test_sell_market_matches_best_bid(self):
        """A market sell should fill at the best (highest) bid price."""
        book = OrderBook()
        book.submit(Order(side=OrderSide.BUY, order_type=OrderType.LIMIT, price=99.0, quantity=10.0))
        book.submit(Order(side=OrderSide.BUY, order_type=OrderType.LIMIT, price=100.0, quantity=10.0))

        sell = Order(side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=5.0)
        trades = book.submit(sell)

        assert len(trades) == 1
        assert trades[0].price == 100.0

    def test_price_priority_buy(self):
        """When buying, fill at lower prices first."""
        book = OrderBook()
        book.submit(Order(side=OrderSide.SELL, order_type=OrderType.LIMIT, price=102.0, quantity=10.0))
        book.submit(Order(side=OrderSide.SELL, order_type=OrderType.LIMIT, price=101.0, quantity=10.0))
        book.submit(Order(side=OrderSide.SELL, order_type=OrderType.LIMIT, price=103.0, quantity=10.0))

        buy = Order(side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=15.0)
        trades = book.submit(buy)

        assert len(trades) == 2
        assert trades[0].price == 101.0
        assert trades[0].quantity == pytest.approx(10.0)
        assert trades[1].price == 102.0
        assert trades[1].quantity == pytest.approx(5.0)

    def test_time_priority_at_same_price(self):
        """At the same price level, earlier orders should fill first (FIFO)."""
        book = OrderBook()
        o1 = Order(side=OrderSide.SELL, order_type=OrderType.LIMIT, price=100.0, quantity=5.0)
        o2 = Order(side=OrderSide.SELL, order_type=OrderType.LIMIT, price=100.0, quantity=5.0)
        book.submit(o1)
        book.submit(o2)

        buy = Order(side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=5.0)
        trades = book.submit(buy)

        # Should fill against o1 (first in queue), not o2
        assert len(trades) == 1
        assert trades[0].resting_order_id == o1.order_id

    def test_partial_fill_leaves_remainder(self):
        """A partial fill should leave the unfilled portion in the book."""
        book = OrderBook()
        ask = Order(side=OrderSide.SELL, order_type=OrderType.LIMIT, price=100.0, quantity=20.0)
        book.submit(ask)

        buy = Order(side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=8.0)
        trades = book.submit(buy)

        assert len(trades) == 1
        assert trades[0].quantity == pytest.approx(8.0)
        assert ask.remaining == pytest.approx(12.0)
        assert ask.status == OrderStatus.PARTIALLY_FILLED
        assert book.best_ask == 100.0

    def test_limit_buy_at_ask_fills_immediately(self):
        """A limit buy at or above the best ask should fill immediately."""
        book = OrderBook()
        book.submit(Order(side=OrderSide.SELL, order_type=OrderType.LIMIT, price=100.0, quantity=10.0))

        buy = Order(side=OrderSide.BUY, order_type=OrderType.LIMIT, price=100.0, quantity=5.0)
        trades = book.submit(buy)

        assert len(trades) == 1
        assert buy.is_filled
        assert buy.status == OrderStatus.FILLED

    def test_limit_buy_below_ask_rests(self):
        """A limit buy below the best ask should rest in the book."""
        book = OrderBook()
        book.submit(Order(side=OrderSide.SELL, order_type=OrderType.LIMIT, price=101.0, quantity=10.0))

        buy = Order(side=OrderSide.BUY, order_type=OrderType.LIMIT, price=100.0, quantity=5.0)
        trades = book.submit(buy)

        assert len(trades) == 0
        assert book.best_bid == 100.0
        assert book.num_orders == 2

    def test_crossing_limit_order_fills_then_rests(self):
        """A limit buy above the ask that isn't fully filled should partially
        execute, then the remainder rests in the book."""
        book = OrderBook()
        book.submit(Order(side=OrderSide.SELL, order_type=OrderType.LIMIT, price=100.0, quantity=5.0))

        buy = Order(side=OrderSide.BUY, order_type=OrderType.LIMIT, price=101.0, quantity=10.0)
        trades = book.submit(buy)

        assert len(trades) == 1
        assert trades[0].quantity == pytest.approx(5.0)
        assert buy.remaining == pytest.approx(5.0)
        # Remainder rests as a bid
        assert book.best_bid == 101.0

    def test_multiple_fills_across_levels(self):
        """A large market order should sweep across multiple price levels."""
        book = OrderBook()
        book.submit(Order(side=OrderSide.SELL, order_type=OrderType.LIMIT, price=100.0, quantity=5.0))
        book.submit(Order(side=OrderSide.SELL, order_type=OrderType.LIMIT, price=101.0, quantity=5.0))
        book.submit(Order(side=OrderSide.SELL, order_type=OrderType.LIMIT, price=102.0, quantity=5.0))

        buy = Order(side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=12.0)
        trades = book.submit(buy)

        assert len(trades) == 3
        assert trades[0].price == 100.0
        assert trades[0].quantity == pytest.approx(5.0)
        assert trades[1].price == 101.0
        assert trades[1].quantity == pytest.approx(5.0)
        assert trades[2].price == 102.0
        assert trades[2].quantity == pytest.approx(2.0)

    def test_unfilled_market_order_cancelled(self):
        """An unfillable market order (empty book) should be cancelled."""
        book = OrderBook()
        buy = Order(side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=10.0)
        trades = book.submit(buy)

        assert len(trades) == 0
        assert buy.status == OrderStatus.CANCELLED


class TestIOCOrders:
    """Test Immediate-Or-Cancel order type."""

    def test_ioc_fills_and_cancels_remainder(self):
        book = OrderBook()
        book.submit(Order(side=OrderSide.SELL, order_type=OrderType.LIMIT, price=100.0, quantity=5.0))

        ioc = Order(side=OrderSide.BUY, order_type=OrderType.IOC, price=100.0, quantity=10.0)
        trades = book.submit(ioc)

        assert len(trades) == 1
        assert trades[0].quantity == pytest.approx(5.0)
        assert ioc.status == OrderStatus.CANCELLED
        # IOC should NOT rest in the book
        assert book.best_bid is None

    def test_ioc_full_fill(self):
        book = OrderBook()
        book.submit(Order(side=OrderSide.SELL, order_type=OrderType.LIMIT, price=100.0, quantity=10.0))

        ioc = Order(side=OrderSide.BUY, order_type=OrderType.IOC, price=100.0, quantity=5.0)
        trades = book.submit(ioc)

        assert len(trades) == 1
        assert ioc.status == OrderStatus.FILLED

    def test_ioc_no_fill(self):
        book = OrderBook()
        ioc = Order(side=OrderSide.BUY, order_type=OrderType.IOC, price=100.0, quantity=10.0)
        trades = book.submit(ioc)

        assert len(trades) == 0
        assert ioc.status == OrderStatus.CANCELLED


class TestFOKOrders:
    """Test Fill-Or-Kill order type."""

    def test_fok_full_fill(self):
        book = OrderBook()
        book.submit(Order(side=OrderSide.SELL, order_type=OrderType.LIMIT, price=100.0, quantity=20.0))

        fok = Order(side=OrderSide.BUY, order_type=OrderType.FOK, price=100.0, quantity=10.0)
        trades = book.submit(fok)

        assert len(trades) == 1
        assert fok.status == OrderStatus.FILLED

    def test_fok_insufficient_liquidity_cancelled(self):
        """FOK order should be cancelled entirely if not enough liquidity."""
        book = OrderBook()
        book.submit(Order(side=OrderSide.SELL, order_type=OrderType.LIMIT, price=100.0, quantity=5.0))

        fok = Order(side=OrderSide.BUY, order_type=OrderType.FOK, price=100.0, quantity=10.0)
        trades = book.submit(fok)

        assert len(trades) == 0
        assert fok.status == OrderStatus.CANCELLED
        # The resting order should still be intact
        assert book.best_ask == 100.0

    def test_fok_empty_book_cancelled(self):
        book = OrderBook()
        fok = Order(side=OrderSide.BUY, order_type=OrderType.FOK, price=100.0, quantity=10.0)
        trades = book.submit(fok)

        assert len(trades) == 0
        assert fok.status == OrderStatus.CANCELLED

    def test_fok_across_levels(self):
        """FOK should succeed if total available across levels is sufficient."""
        book = OrderBook()
        book.submit(Order(side=OrderSide.SELL, order_type=OrderType.LIMIT, price=100.0, quantity=5.0))
        book.submit(Order(side=OrderSide.SELL, order_type=OrderType.LIMIT, price=101.0, quantity=5.0))

        fok = Order(side=OrderSide.BUY, order_type=OrderType.FOK, price=101.0, quantity=10.0)
        trades = book.submit(fok)

        assert len(trades) == 2
        assert fok.status == OrderStatus.FILLED


class TestTradeCallbacks:
    """Test that trade callbacks fire correctly."""

    def test_callback_on_trade(self):
        book = OrderBook()
        trades_received = []
        book.on_trade(lambda t: trades_received.append(t))

        book.submit(Order(side=OrderSide.SELL, order_type=OrderType.LIMIT, price=100.0, quantity=10.0))
        book.submit(Order(side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=5.0))

        assert len(trades_received) == 1
        assert trades_received[0].price == 100.0
        assert trades_received[0].quantity == pytest.approx(5.0)

    def test_agent_ids_in_trade(self):
        book = OrderBook()
        trades_received = []
        book.on_trade(lambda t: trades_received.append(t))

        book.submit(Order(
            side=OrderSide.SELL, order_type=OrderType.LIMIT,
            price=100.0, quantity=10.0, agent_id="seller_1"
        ))
        book.submit(Order(
            side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=5.0, agent_id="buyer_1"
        ))

        assert trades_received[0].buyer_agent_id == "buyer_1"
        assert trades_received[0].seller_agent_id == "seller_1"
