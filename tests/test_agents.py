"""Tests for simulation agents."""

import pytest

from src.core.agents import (
    AvellanedaStoikovMM,
    InformedTrader,
    MomentumTrader,
    NoiseTrader,
)
from src.core.order import Order, OrderSide, OrderType
from src.core.orderbook import OrderBook


def make_book_with_spread(mid: float = 100.0, spread: float = 1.0, qty: float = 50.0) -> OrderBook:
    """Create a book with a single bid/ask level around the given mid."""
    book = OrderBook()
    bid = Order(
        side=OrderSide.BUY, order_type=OrderType.LIMIT,
        price=mid - spread / 2, quantity=qty, agent_id="_seed",
    )
    ask = Order(
        side=OrderSide.SELL, order_type=OrderType.LIMIT,
        price=mid + spread / 2, quantity=qty, agent_id="_seed",
    )
    book.submit(bid)
    book.submit(ask)
    return book


class TestAvellanedaStoikovMM:
    """Test the Avellaneda-Stoikov market maker."""

    def test_quotes_both_sides(self):
        mm = AvellanedaStoikovMM(agent_id="mm", gamma=0.1, sigma=0.5, k=1.5, total_time=100)
        book = make_book_with_spread()

        orders = mm.step(0, book)
        assert len(orders) == 2

        sides = {o.side for o in orders}
        assert OrderSide.BUY in sides
        assert OrderSide.SELL in sides

    def test_quotes_are_limit_orders(self):
        mm = AvellanedaStoikovMM(agent_id="mm", gamma=0.1, sigma=0.5, k=1.5, total_time=100)
        book = make_book_with_spread()

        orders = mm.step(0, book)
        for o in orders:
            assert o.order_type == OrderType.LIMIT

    def test_spread_positive(self):
        mm = AvellanedaStoikovMM(agent_id="mm", gamma=0.1, sigma=0.5, k=1.5, total_time=100)
        book = make_book_with_spread()

        orders = mm.step(0, book)
        bid = [o for o in orders if o.side == OrderSide.BUY][0]
        ask = [o for o in orders if o.side == OrderSide.SELL][0]
        assert ask.price > bid.price

    def test_inventory_skew(self):
        """With positive inventory, reservation price shifts down,
        so bid should be lower (more aggressive to sell)."""
        mm = AvellanedaStoikovMM(agent_id="mm", gamma=0.1, sigma=0.5, k=1.5, total_time=100)
        book = make_book_with_spread()

        # No inventory
        orders_neutral = mm.step(0, book)
        bid_neutral = [o for o in orders_neutral if o.side == OrderSide.BUY][0].price

        # With long inventory
        mm2 = AvellanedaStoikovMM(agent_id="mm2", gamma=0.1, sigma=0.5, k=1.5, total_time=100)
        mm2.state.inventory = 50.0
        orders_long = mm2.step(0, book)
        bid_long = [o for o in orders_long if o.side == OrderSide.BUY][0].price

        # With inventory, reservation shifts down, so bid should be lower
        assert bid_long < bid_neutral

    def test_no_quotes_on_empty_book(self):
        mm = AvellanedaStoikovMM(agent_id="mm")
        book = OrderBook()  # empty
        orders = mm.step(0, book)
        assert len(orders) == 0


class TestNoiseTrader:
    """Test the noise trader."""

    def test_produces_orders(self):
        trader = NoiseTrader(agent_id="noise", trade_prob=1.0, seed=42)
        book = make_book_with_spread()

        orders = trader.step(0, book)
        assert len(orders) >= 1

    def test_respects_trade_prob_zero(self):
        trader = NoiseTrader(agent_id="noise", trade_prob=0.0, seed=42)
        book = make_book_with_spread()

        orders = trader.step(0, book)
        assert len(orders) == 0

    def test_order_has_positive_quantity(self):
        trader = NoiseTrader(agent_id="noise", trade_prob=1.0, seed=42)
        book = make_book_with_spread()

        for _ in range(20):
            orders = trader.step(0, book)
            for o in orders:
                assert o.quantity > 0

    def test_order_has_positive_price(self):
        trader = NoiseTrader(agent_id="noise", trade_prob=1.0, seed=42)
        book = make_book_with_spread()

        for _ in range(20):
            orders = trader.step(0, book)
            for o in orders:
                if o.price is not None:
                    assert o.price > 0


class TestMomentumTrader:
    """Test the momentum trader."""

    def test_no_trade_without_history(self):
        trader = MomentumTrader(agent_id="mom", short_window=3, long_window=5)
        book = make_book_with_spread()

        # Not enough history yet
        orders = trader.step(0, book)
        assert len(orders) == 0

    def test_builds_up_history(self):
        trader = MomentumTrader(agent_id="mom", short_window=3, long_window=5, cooldown=0)
        book = make_book_with_spread()

        for t in range(10):
            trader.step(t, book)

        assert len(trader._price_history) > 0


class TestInformedTrader:
    """Test the informed trader."""

    def test_fundamental_evolves(self):
        trader = InformedTrader(
            agent_id="inf", initial_fundamental=100.0,
            fundamental_vol=0.1, seed=42,
        )
        book = make_book_with_spread()

        trader.step(0, book)
        assert len(trader.fundamental_history) == 2  # initial + one step

    def test_trades_on_mispricing(self):
        """When fundamental is far from mid, should submit an order."""
        trader = InformedTrader(
            agent_id="inf", initial_fundamental=105.0,
            fundamental_vol=0.0,  # no randomness
            threshold=0.1,
            seed=42,
        )
        book = make_book_with_spread(mid=100.0)

        orders = trader.step(0, book)
        # fundamental (105) >> mid (100), so should buy
        assert len(orders) == 1
        assert orders[0].side == OrderSide.BUY

    def test_no_trade_within_threshold(self):
        trader = InformedTrader(
            agent_id="inf", initial_fundamental=100.0,
            fundamental_vol=0.0,
            threshold=5.0,
            seed=42,
        )
        book = make_book_with_spread(mid=100.0)

        orders = trader.step(0, book)
        assert len(orders) == 0
