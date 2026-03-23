"""Tests for execution algorithms: TWAP, VWAP, Implementation Shortfall."""

import pytest

from src.core.execution import ISAlgo, TWAPAlgo, VWAPAlgo
from src.core.order import Order, OrderSide, OrderType
from src.core.orderbook import OrderBook


def make_liquid_book(mid: float = 100.0, levels: int = 10, qty_per_level: float = 100.0) -> OrderBook:
    """Create a deeply liquid book for execution testing."""
    book = OrderBook()
    for i in range(levels):
        bid_price = round(mid - 0.1 * (i + 1), 2)
        ask_price = round(mid + 0.1 * (i + 1), 2)
        book.submit(Order(
            side=OrderSide.BUY, order_type=OrderType.LIMIT,
            price=bid_price, quantity=qty_per_level, agent_id="_liq",
        ))
        book.submit(Order(
            side=OrderSide.SELL, order_type=OrderType.LIMIT,
            price=ask_price, quantity=qty_per_level, agent_id="_liq",
        ))
    return book


class TestTWAP:
    """Test Time-Weighted Average Price execution."""

    def test_twap_completes(self):
        """TWAP should fill all slices over the time window."""
        book = make_liquid_book()
        algo = TWAPAlgo(
            side=OrderSide.BUY,
            total_quantity=50.0,
            num_slices=5,
            start_time=0,
            end_time=50,
        )

        for t in range(60):
            orders = algo.step(t, book)
            for order in orders:
                trades = book.submit(order)
                for trade in trades:
                    algo.on_fill(trade)
            # Replenish liquidity
            if book.best_ask is None or len(book.ask_levels(1)) == 0:
                book.submit(Order(
                    side=OrderSide.SELL, order_type=OrderType.LIMIT,
                    price=100.1, quantity=100.0, agent_id="_liq",
                ))

        assert algo.is_done
        report = algo.report()
        assert report.filled_quantity == pytest.approx(50.0, abs=1.0)

    def test_twap_equal_slices(self):
        """TWAP should produce roughly equal child order sizes."""
        book = make_liquid_book()
        algo = TWAPAlgo(
            side=OrderSide.BUY,
            total_quantity=100.0,
            num_slices=10,
            start_time=0,
            end_time=100,
        )

        child_quantities = []
        for t in range(110):
            orders = algo.step(t, book)
            for order in orders:
                child_quantities.append(order.quantity)
                trades = book.submit(order)
                for trade in trades:
                    algo.on_fill(trade)
            if book.best_ask is None:
                book.submit(Order(
                    side=OrderSide.SELL, order_type=OrderType.LIMIT,
                    price=100.1, quantity=200.0, agent_id="_liq",
                ))

        # Each slice should be ~10.0
        for q in child_quantities:
            assert q == pytest.approx(10.0, abs=1.0)

    def test_twap_average_price_near_market(self):
        """TWAP fill price should be close to the market price."""
        book = make_liquid_book(mid=100.0)
        algo = TWAPAlgo(
            side=OrderSide.BUY,
            total_quantity=30.0,
            num_slices=3,
            start_time=0,
            end_time=30,
        )

        for t in range(40):
            orders = algo.step(t, book)
            for order in orders:
                trades = book.submit(order)
                for trade in trades:
                    algo.on_fill(trade)
            if book.best_ask is None:
                book.submit(Order(
                    side=OrderSide.SELL, order_type=OrderType.LIMIT,
                    price=100.1, quantity=200.0, agent_id="_liq",
                ))

        report = algo.report()
        # Average fill should be close to the ask side (~100.1)
        assert 99.0 < report.avg_fill_price < 101.0

    def test_twap_report_slippage(self):
        """TWAP report should compute slippage in basis points."""
        book = make_liquid_book(mid=100.0)
        algo = TWAPAlgo(
            side=OrderSide.BUY,
            total_quantity=20.0,
            num_slices=2,
            start_time=0,
            end_time=20,
        )

        for t in range(30):
            orders = algo.step(t, book)
            for order in orders:
                trades = book.submit(order)
                for trade in trades:
                    algo.on_fill(trade)
            if book.best_ask is None:
                book.submit(Order(
                    side=OrderSide.SELL, order_type=OrderType.LIMIT,
                    price=100.1, quantity=200.0, agent_id="_liq",
                ))

        report = algo.report()
        assert isinstance(report.slippage_bps, float)


class TestVWAP:
    """Test Volume-Weighted Average Price execution."""

    def test_vwap_completes(self):
        book = make_liquid_book()
        algo = VWAPAlgo(
            side=OrderSide.SELL,
            total_quantity=50.0,
            num_slices=5,
            start_time=0,
            end_time=50,
        )

        for t in range(60):
            orders = algo.step(t, book)
            for order in orders:
                trades = book.submit(order)
                for trade in trades:
                    algo.on_fill(trade)
            if book.best_bid is None:
                book.submit(Order(
                    side=OrderSide.BUY, order_type=OrderType.LIMIT,
                    price=99.9, quantity=200.0, agent_id="_liq",
                ))

        assert algo.is_done
        report = algo.report()
        assert report.filled_quantity == pytest.approx(50.0, abs=1.0)

    def test_vwap_u_shaped_profile(self):
        """VWAP default profile should produce larger slices at start and end."""
        book = make_liquid_book()
        algo = VWAPAlgo(
            side=OrderSide.BUY,
            total_quantity=100.0,
            num_slices=5,
            start_time=0,
            end_time=50,
        )

        # The first and last slices should be larger than the middle
        assert algo._slice_sizes[0] > algo._slice_sizes[2]
        assert algo._slice_sizes[-1] > algo._slice_sizes[2]


class TestImplementationShortfall:
    """Test Implementation Shortfall execution."""

    def test_is_completes(self):
        book = make_liquid_book()
        algo = ISAlgo(
            side=OrderSide.BUY,
            total_quantity=50.0,
            num_slices=5,
            start_time=0,
            end_time=50,
            urgency=0.5,
        )

        for t in range(60):
            orders = algo.step(t, book)
            for order in orders:
                trades = book.submit(order)
                for trade in trades:
                    algo.on_fill(trade)
            if book.best_ask is None:
                book.submit(Order(
                    side=OrderSide.SELL, order_type=OrderType.LIMIT,
                    price=100.1, quantity=200.0, agent_id="_liq",
                ))

        assert algo.is_done

    def test_is_report(self):
        book = make_liquid_book()
        algo = ISAlgo(
            side=OrderSide.BUY,
            total_quantity=30.0,
            num_slices=3,
            start_time=0,
            end_time=30,
        )

        for t in range(40):
            orders = algo.step(t, book)
            for order in orders:
                trades = book.submit(order)
                for trade in trades:
                    algo.on_fill(trade)
            if book.best_ask is None:
                book.submit(Order(
                    side=OrderSide.SELL, order_type=OrderType.LIMIT,
                    price=100.1, quantity=200.0, agent_id="_liq",
                ))

        report = algo.report()
        assert report.algo_name == "ISAlgo"
        assert report.filled_quantity > 0
