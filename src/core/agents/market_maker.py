"""Avellaneda-Stoikov optimal market maker.

Implements the Avellaneda-Stoikov (2008) model for optimal market making.
The market maker continuously quotes symmetric bid/ask prices around the
reservation price, adjusting spreads based on inventory risk and time horizon.

Key equations:
    Reservation price: r(s, q, t) = s - q * gamma * sigma^2 * (T - t)
    Optimal spread:    delta = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)

where:
    s     = current mid price
    q     = current inventory
    gamma = risk aversion parameter
    sigma = volatility of the underlying
    T     = terminal time
    k     = order arrival intensity parameter
"""

from __future__ import annotations

import math

import numpy as np

from src.core.agents.base import Agent
from src.core.order import Order, OrderSide, OrderType
from src.core.orderbook import OrderBook


class AvellanedaStoikovMM(Agent):
    """Avellaneda-Stoikov market maker that quotes optimal bid/ask levels.

    Parameters:
        agent_id: Unique agent identifier.
        gamma: Risk aversion coefficient (higher = more conservative).
        sigma: Estimated asset volatility.
        k: Order arrival intensity parameter.
        total_time: Total simulation time horizon T.
        order_size: Size of each quote.
        max_inventory: Absolute inventory limit; if exceeded, only quote reducing side.
    """

    def __init__(
        self,
        agent_id: str = "mm_as",
        gamma: float = 0.1,
        sigma: float = 0.5,
        k: float = 1.5,
        total_time: int = 1000,
        order_size: float = 10.0,
        max_inventory: float = 100.0,
    ) -> None:
        super().__init__(agent_id)
        self.gamma = gamma
        self.sigma = sigma
        self.k = k
        self.total_time = total_time
        self.order_size = order_size
        self.max_inventory = max_inventory

    def step(self, t: int, book: OrderBook) -> list[Order]:
        """Cancel old quotes, compute optimal bid/ask, and submit new quotes."""
        # Cancel existing quotes
        self.cancel_all(book)

        mid = book.mid_price
        if mid is None:
            return []

        q = self.state.inventory
        tau = max(self.total_time - t, 1) / self.total_time  # time remaining fraction

        # Reservation price: adjusted mid based on inventory
        reservation = mid - q * self.gamma * (self.sigma ** 2) * tau

        # Optimal spread
        spread = self.gamma * (self.sigma ** 2) * tau + (2.0 / self.gamma) * math.log(
            1.0 + self.gamma / self.k
        )
        half_spread = spread / 2.0

        bid_price = round(reservation - half_spread, 2)
        ask_price = round(reservation + half_spread, 2)

        orders: list[Order] = []

        # Submit bid (buy) if not too long
        if q < self.max_inventory and bid_price > 0:
            bid = Order(
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=bid_price,
                quantity=self.order_size,
                agent_id=self.agent_id,
            )
            orders.append(bid)
            self._active_order_ids.add(bid.order_id)

        # Submit ask (sell) if not too short
        if q > -self.max_inventory and ask_price > 0:
            ask = Order(
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=ask_price,
                quantity=self.order_size,
                agent_id=self.agent_id,
            )
            orders.append(ask)
            self._active_order_ids.add(ask.order_id)

        return orders
