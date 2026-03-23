"""Execution algorithms: TWAP, VWAP, and Implementation Shortfall.

Parent-order slicing algorithms that break a large order into smaller child
orders over time, minimizing market impact and execution cost.

- TWAP: Time-Weighted Average Price -- equal slices at regular intervals.
- VWAP: Volume-Weighted Average Price -- slices proportional to expected volume.
- Implementation Shortfall: Adaptive urgency based on price drift from arrival.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.core.order import Order, OrderSide, OrderType
from src.core.orderbook import OrderBook, Trade


@dataclass
class ExecutionReport:
    """Summary statistics for a completed execution.

    Attributes:
        algo_name: Name of the algorithm used.
        side: BUY or SELL.
        total_quantity: Total parent order quantity.
        filled_quantity: Actually filled quantity.
        num_child_orders: Number of child orders sent.
        num_fills: Number of individual fills.
        avg_fill_price: Volume-weighted average fill price.
        arrival_price: Mid price at the time the algo started.
        vwap_market: Market VWAP over the execution window.
        slippage_bps: (avg_fill - arrival) / arrival * 10000, sign-adjusted.
        fill_prices: All individual fill prices.
        fill_quantities: All individual fill quantities.
        child_timestamps: Timestamps when child orders were sent.
    """

    algo_name: str
    side: OrderSide
    total_quantity: float
    filled_quantity: float = 0.0
    num_child_orders: int = 0
    num_fills: int = 0
    avg_fill_price: float = 0.0
    arrival_price: float = 0.0
    vwap_market: float = 0.0
    slippage_bps: float = 0.0
    fill_prices: list[float] = field(default_factory=list)
    fill_quantities: list[float] = field(default_factory=list)
    child_timestamps: list[int] = field(default_factory=list)


class ExecutionAlgo:
    """Base class for execution algorithms."""

    def __init__(self, side: OrderSide, total_quantity: float, agent_id: str = "exec") -> None:
        self.side = side
        self.total_quantity = total_quantity
        self.agent_id = agent_id
        self.remaining = total_quantity
        self.arrival_price: Optional[float] = None
        self._fills: list[tuple[float, float]] = []  # (price, qty) pairs
        self._child_timestamps: list[int] = []
        self._done = False

    @property
    def is_done(self) -> bool:
        return self._done or self.remaining <= 1e-12

    def on_fill(self, trade: Trade) -> None:
        """Record a fill from the matching engine."""
        self._fills.append((trade.price, trade.quantity))
        self.remaining -= trade.quantity
        if self.remaining <= 1e-12:
            self.remaining = 0.0
            self._done = True

    def report(self, market_vwap: Optional[float] = None) -> ExecutionReport:
        """Generate execution report."""
        filled_qty = sum(q for _, q in self._fills)
        if filled_qty > 0:
            avg_price = sum(p * q for p, q in self._fills) / filled_qty
        else:
            avg_price = 0.0

        slippage = 0.0
        if self.arrival_price and self.arrival_price > 0 and filled_qty > 0:
            if self.side == OrderSide.BUY:
                slippage = (avg_price - self.arrival_price) / self.arrival_price * 10000
            else:
                slippage = (self.arrival_price - avg_price) / self.arrival_price * 10000

        return ExecutionReport(
            algo_name=self.__class__.__name__,
            side=self.side,
            total_quantity=self.total_quantity,
            filled_quantity=filled_qty,
            num_child_orders=len(self._child_timestamps),
            num_fills=len(self._fills),
            avg_fill_price=avg_price,
            arrival_price=self.arrival_price or 0.0,
            vwap_market=market_vwap or 0.0,
            slippage_bps=slippage,
            fill_prices=[p for p, _ in self._fills],
            fill_quantities=[q for _, q in self._fills],
            child_timestamps=self._child_timestamps,
        )


class TWAPAlgo(ExecutionAlgo):
    """Time-Weighted Average Price execution algorithm.

    Splits the parent order into equal-sized child orders submitted at
    regular time intervals over the execution window.

    Parameters:
        side: BUY or SELL.
        total_quantity: Total quantity to execute.
        num_slices: Number of child orders (time slices).
        start_time: Timestep when execution begins.
        end_time: Timestep when execution must complete.
        agent_id: Agent identifier for the child orders.
    """

    def __init__(
        self,
        side: OrderSide,
        total_quantity: float,
        num_slices: int = 10,
        start_time: int = 0,
        end_time: int = 100,
        agent_id: str = "twap",
    ) -> None:
        super().__init__(side, total_quantity, agent_id)
        self.num_slices = num_slices
        self.start_time = start_time
        self.end_time = end_time

        # Compute schedule: evenly spaced submission times
        interval = max(1, (end_time - start_time) // num_slices)
        self._schedule = [start_time + i * interval for i in range(num_slices)]
        self._slice_size = total_quantity / num_slices
        self._schedule_idx = 0

    def step(self, t: int, book: OrderBook) -> list[Order]:
        """Called each timestep. Returns child orders to submit."""
        if self.is_done:
            return []

        if self.arrival_price is None:
            self.arrival_price = book.mid_price

        orders: list[Order] = []

        while (
            self._schedule_idx < len(self._schedule)
            and t >= self._schedule[self._schedule_idx]
        ):
            qty = min(self._slice_size, self.remaining)
            if qty <= 1e-12:
                break

            # Submit as IOC limit order slightly aggressive
            if self.side == OrderSide.BUY:
                ref_price = book.best_ask
            else:
                ref_price = book.best_bid

            if ref_price is not None:
                order = Order(
                    side=self.side,
                    order_type=OrderType.IOC,
                    price=ref_price,
                    quantity=qty,
                    agent_id=self.agent_id,
                )
                orders.append(order)
                self._child_timestamps.append(t)

            self._schedule_idx += 1

        return orders


class VWAPAlgo(ExecutionAlgo):
    """Volume-Weighted Average Price execution algorithm.

    Distributes the parent order proportional to an expected intraday volume
    profile. The volume profile can be provided or defaults to a U-shaped
    curve (higher volume at open and close).

    Parameters:
        side: BUY or SELL.
        total_quantity: Total quantity to execute.
        num_slices: Number of time buckets.
        start_time: Timestep when execution begins.
        end_time: Timestep when execution must complete.
        volume_profile: Optional array of relative volumes per bucket.
        agent_id: Agent identifier.
    """

    def __init__(
        self,
        side: OrderSide,
        total_quantity: float,
        num_slices: int = 10,
        start_time: int = 0,
        end_time: int = 100,
        volume_profile: Optional[np.ndarray] = None,
        agent_id: str = "vwap",
    ) -> None:
        super().__init__(side, total_quantity, agent_id)
        self.num_slices = num_slices
        self.start_time = start_time
        self.end_time = end_time

        # Default: U-shaped volume profile
        if volume_profile is None:
            x = np.linspace(0, 1, num_slices)
            # Quadratic U-shape: more volume at start and end
            raw = 1.0 + 2.0 * (x - 0.5) ** 2
            volume_profile = raw / raw.sum()
        else:
            volume_profile = volume_profile / volume_profile.sum()

        self._volume_profile = volume_profile

        # Compute schedule
        interval = max(1, (end_time - start_time) // num_slices)
        self._schedule = [start_time + i * interval for i in range(num_slices)]
        self._slice_sizes = [total_quantity * w for w in volume_profile]
        self._schedule_idx = 0

    def step(self, t: int, book: OrderBook) -> list[Order]:
        """Called each timestep. Returns child orders proportional to volume profile."""
        if self.is_done:
            return []

        if self.arrival_price is None:
            self.arrival_price = book.mid_price

        orders: list[Order] = []

        while (
            self._schedule_idx < len(self._schedule)
            and t >= self._schedule[self._schedule_idx]
        ):
            qty = min(self._slice_sizes[self._schedule_idx], self.remaining)
            if qty <= 1e-12:
                self._schedule_idx += 1
                continue

            if self.side == OrderSide.BUY:
                ref_price = book.best_ask
            else:
                ref_price = book.best_bid

            if ref_price is not None:
                order = Order(
                    side=self.side,
                    order_type=OrderType.IOC,
                    price=ref_price,
                    quantity=qty,
                    agent_id=self.agent_id,
                )
                orders.append(order)
                self._child_timestamps.append(t)

            self._schedule_idx += 1

        return orders


class ISAlgo(ExecutionAlgo):
    """Implementation Shortfall (Almgren-Chriss inspired) execution algorithm.

    Adjusts execution urgency based on how far the price has moved from the
    arrival price. If the price moves adversely, the algo accelerates; if
    favorably, it decelerates.

    Parameters:
        side: BUY or SELL.
        total_quantity: Total quantity to execute.
        num_slices: Number of time buckets.
        start_time: Start timestep.
        end_time: End timestep.
        urgency: Base urgency factor (0-1). Higher = more front-loaded.
        agent_id: Agent identifier.
    """

    def __init__(
        self,
        side: OrderSide,
        total_quantity: float,
        num_slices: int = 10,
        start_time: int = 0,
        end_time: int = 100,
        urgency: float = 0.5,
        agent_id: str = "is_algo",
    ) -> None:
        super().__init__(side, total_quantity, agent_id)
        self.num_slices = num_slices
        self.start_time = start_time
        self.end_time = end_time
        self.urgency = urgency

        interval = max(1, (end_time - start_time) // num_slices)
        self._schedule = [start_time + i * interval for i in range(num_slices)]
        self._base_slice = total_quantity / num_slices
        self._schedule_idx = 0

    def step(self, t: int, book: OrderBook) -> list[Order]:
        """Adaptive execution: accelerate on adverse price movement."""
        if self.is_done:
            return []

        mid = book.mid_price
        if self.arrival_price is None:
            self.arrival_price = mid

        orders: list[Order] = []

        while (
            self._schedule_idx < len(self._schedule)
            and t >= self._schedule[self._schedule_idx]
        ):
            # Compute adaptive slice size
            urgency_mult = 1.0
            if mid is not None and self.arrival_price is not None and self.arrival_price > 0:
                drift = (mid - self.arrival_price) / self.arrival_price
                if self.side == OrderSide.BUY:
                    # Price going up = adverse for buyer: increase urgency
                    urgency_mult = 1.0 + self.urgency * drift * 100
                else:
                    # Price going down = adverse for seller
                    urgency_mult = 1.0 - self.urgency * drift * 100

                urgency_mult = max(0.2, min(3.0, urgency_mult))

            qty = min(self._base_slice * urgency_mult, self.remaining)
            if qty <= 1e-12:
                self._schedule_idx += 1
                continue

            if self.side == OrderSide.BUY:
                ref_price = book.best_ask
            else:
                ref_price = book.best_bid

            if ref_price is not None:
                order = Order(
                    side=self.side,
                    order_type=OrderType.IOC,
                    price=ref_price,
                    quantity=qty,
                    agent_id=self.agent_id,
                )
                orders.append(order)
                self._child_timestamps.append(t)

            self._schedule_idx += 1

        return orders
