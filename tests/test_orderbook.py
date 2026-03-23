"""Tests for the order book data structure and invariants."""

import pytest

from src.core.order import Order, OrderSide, OrderStatus, OrderType
from src.core.orderbook import OrderBook


class TestOrderBookInvariants:
    """Verify core order book invariants."""

    def test_empty_book(self):
        book = OrderBook()
        assert book.best_bid is None
        assert book.best_ask is None
        assert book.mid_price is None
        assert book.spread is None
        assert book.num_orders == 0

    def test_single_bid(self):
        book = OrderBook()
        order = Order(side=OrderSide.BUY, order_type=OrderType.LIMIT, price=100.0, quantity=10.0)
        book.submit(order)
        assert book.best_bid == 100.0
        assert book.best_ask is None
        assert book.num_orders == 1

    def test_single_ask(self):
        book = OrderBook()
        order = Order(side=OrderSide.SELL, order_type=OrderType.LIMIT, price=101.0, quantity=10.0)
        book.submit(order)
        assert book.best_ask == 101.0
        assert book.best_bid is None

    def test_best_bid_less_than_best_ask(self):
        """Core invariant: best bid must always be strictly less than best ask."""
        book = OrderBook()
        book.submit(Order(side=OrderSide.BUY, order_type=OrderType.LIMIT, price=99.0, quantity=10.0))
        book.submit(Order(side=OrderSide.BUY, order_type=OrderType.LIMIT, price=100.0, quantity=10.0))
        book.submit(Order(side=OrderSide.SELL, order_type=OrderType.LIMIT, price=101.0, quantity=10.0))
        book.submit(Order(side=OrderSide.SELL, order_type=OrderType.LIMIT, price=102.0, quantity=10.0))

        assert book.best_bid == 100.0
        assert book.best_ask == 101.0
        assert book.best_bid < book.best_ask

    def test_bids_sorted_descending(self):
        book = OrderBook()
        for p in [97.0, 99.0, 98.0, 100.0, 96.0]:
            book.submit(Order(side=OrderSide.BUY, order_type=OrderType.LIMIT, price=p, quantity=5.0))

        levels = book.bid_levels(10)
        prices = [p for p, _ in levels]
        assert prices == sorted(prices, reverse=True)

    def test_asks_sorted_ascending(self):
        book = OrderBook()
        for p in [103.0, 101.0, 104.0, 102.0, 105.0]:
            book.submit(Order(side=OrderSide.SELL, order_type=OrderType.LIMIT, price=p, quantity=5.0))

        levels = book.ask_levels(10)
        prices = [p for p, _ in levels]
        assert prices == sorted(prices)

    def test_mid_price(self):
        book = OrderBook()
        book.submit(Order(side=OrderSide.BUY, order_type=OrderType.LIMIT, price=99.0, quantity=10.0))
        book.submit(Order(side=OrderSide.SELL, order_type=OrderType.LIMIT, price=101.0, quantity=10.0))
        assert book.mid_price == 100.0

    def test_spread(self):
        book = OrderBook()
        book.submit(Order(side=OrderSide.BUY, order_type=OrderType.LIMIT, price=99.5, quantity=10.0))
        book.submit(Order(side=OrderSide.SELL, order_type=OrderType.LIMIT, price=100.5, quantity=10.0))
        assert book.spread == pytest.approx(1.0)

    def test_cancel_order(self):
        book = OrderBook()
        order = Order(side=OrderSide.BUY, order_type=OrderType.LIMIT, price=100.0, quantity=10.0)
        book.submit(order)
        assert book.num_orders == 1

        result = book.cancel(order.order_id)
        assert result is True
        assert book.num_orders == 0
        assert book.best_bid is None

    def test_cancel_nonexistent(self):
        book = OrderBook()
        assert book.cancel("nonexistent") is False

    def test_multiple_orders_at_same_price(self):
        book = OrderBook()
        o1 = Order(side=OrderSide.BUY, order_type=OrderType.LIMIT, price=100.0, quantity=5.0)
        o2 = Order(side=OrderSide.BUY, order_type=OrderType.LIMIT, price=100.0, quantity=10.0)
        book.submit(o1)
        book.submit(o2)

        levels = book.bid_levels(10)
        assert len(levels) == 1
        assert levels[0] == (100.0, 15.0)

    def test_bid_ask_volume(self):
        book = OrderBook()
        book.submit(Order(side=OrderSide.BUY, order_type=OrderType.LIMIT, price=99.0, quantity=10.0))
        book.submit(Order(side=OrderSide.BUY, order_type=OrderType.LIMIT, price=98.0, quantity=20.0))
        book.submit(Order(side=OrderSide.SELL, order_type=OrderType.LIMIT, price=101.0, quantity=15.0))

        assert book.bid_volume == pytest.approx(30.0)
        assert book.ask_volume == pytest.approx(15.0)


class TestOrderValidation:
    """Test order creation validation."""

    def test_limit_order_requires_price(self):
        with pytest.raises(ValueError, match="requires a price"):
            Order(side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=10.0)

    def test_positive_quantity(self):
        with pytest.raises(ValueError, match="positive"):
            Order(side=OrderSide.BUY, order_type=OrderType.LIMIT, price=100.0, quantity=-5.0)

    def test_positive_price(self):
        with pytest.raises(ValueError, match="positive"):
            Order(side=OrderSide.BUY, order_type=OrderType.LIMIT, price=-10.0, quantity=5.0)

    def test_market_order_no_price(self):
        order = Order(side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=10.0)
        assert order.price is None

    def test_order_fill(self):
        order = Order(side=OrderSide.BUY, order_type=OrderType.LIMIT, price=100.0, quantity=10.0)
        filled = order.fill(3.0)
        assert filled == pytest.approx(3.0)
        assert order.remaining == pytest.approx(7.0)
        assert order.status == OrderStatus.PARTIALLY_FILLED

    def test_order_fill_complete(self):
        order = Order(side=OrderSide.BUY, order_type=OrderType.LIMIT, price=100.0, quantity=10.0)
        order.fill(10.0)
        assert order.is_filled
        assert order.status == OrderStatus.FILLED

    def test_order_overfill(self):
        order = Order(side=OrderSide.BUY, order_type=OrderType.LIMIT, price=100.0, quantity=10.0)
        filled = order.fill(15.0)
        assert filled == pytest.approx(10.0)
        assert order.is_filled
