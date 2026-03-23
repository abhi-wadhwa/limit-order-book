"""Order data structures for the limit order book.

Supports Limit, Market, IOC (Immediate-Or-Cancel), and FOK (Fill-Or-Kill) order types
with price-time priority matching semantics.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class OrderSide(Enum):
    """Side of the order."""

    BUY = auto()
    SELL = auto()


class OrderType(Enum):
    """Order type determines execution semantics.

    - LIMIT: rests in the book at the specified price until filled or cancelled.
    - MARKET: executes immediately at the best available price; unfilled remainder is cancelled.
    - IOC: Immediate-Or-Cancel -- fills what it can immediately, cancels the rest.
    - FOK: Fill-Or-Kill -- fills entirely or not at all (atomic).
    """

    LIMIT = auto()
    MARKET = auto()
    IOC = auto()
    FOK = auto()


class OrderStatus(Enum):
    """Lifecycle status of an order."""

    NEW = auto()
    PARTIALLY_FILLED = auto()
    FILLED = auto()
    CANCELLED = auto()


@dataclass
class Order:
    """Represents a single order in the book.

    Attributes:
        order_id: Unique identifier (auto-generated UUID if not provided).
        side: BUY or SELL.
        order_type: LIMIT, MARKET, IOC, or FOK.
        price: Limit price. Required for LIMIT/IOC/FOK; ignored for MARKET.
        quantity: Total order quantity.
        filled_quantity: Cumulative filled quantity so far.
        status: Current lifecycle status.
        timestamp: Monotonic arrival time (nanoseconds) for price-time priority.
        agent_id: Optional identifier linking the order to a simulation agent.
    """

    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    filled_quantity: float = 0.0
    status: OrderStatus = OrderStatus.NEW
    order_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: int = field(default_factory=lambda: time.monotonic_ns())
    agent_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.order_type != OrderType.MARKET and self.price is None:
            raise ValueError(f"{self.order_type.name} order requires a price")
        if self.quantity <= 0:
            raise ValueError("Order quantity must be positive")
        if self.price is not None and self.price <= 0:
            raise ValueError("Order price must be positive")

    @property
    def remaining(self) -> float:
        """Unfilled quantity."""
        return self.quantity - self.filled_quantity

    @property
    def is_filled(self) -> bool:
        return self.remaining <= 1e-12

    def fill(self, qty: float) -> float:
        """Record a fill of *qty* shares. Returns the actual fill amount."""
        actual = min(qty, self.remaining)
        self.filled_quantity += actual
        if self.is_filled:
            self.status = OrderStatus.FILLED
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIALLY_FILLED
        return actual

    def cancel(self) -> None:
        """Cancel the order."""
        self.status = OrderStatus.CANCELLED

    def __repr__(self) -> str:
        return (
            f"Order({self.order_id}, {self.side.name} {self.order_type.name} "
            f"{self.remaining:.2f}@{self.price}, status={self.status.name})"
        )
