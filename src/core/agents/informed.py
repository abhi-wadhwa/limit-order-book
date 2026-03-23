"""Informed trader agent.

Possesses knowledge of a hidden fundamental value and trades toward it.
Models information asymmetry in the market, similar to the Kyle (1985) framework.
The informed trader observes the gap between the fundamental value and the
current price, trading aggressively when the mispricing is large.
"""

from __future__ import annotations

import numpy as np

from src.core.agents.base import Agent
from src.core.order import Order, OrderSide, OrderType
from src.core.orderbook import OrderBook


class InformedTrader(Agent):
    """Trades toward a hidden fundamental value.

    The fundamental value follows a random walk:
        V(t+1) = V(t) + epsilon,  epsilon ~ N(0, fundamental_vol)

    The informed trader observes V and trades when |mid - V| exceeds a threshold.

    Parameters:
        agent_id: Unique identifier.
        initial_fundamental: Starting fundamental value.
        fundamental_vol: Volatility of the fundamental random walk.
        threshold: Minimum |mid - V| to trigger a trade.
        order_size: Size of each trade.
        aggression: Fraction of mispricing to capture (0-1).
        max_inventory: Absolute inventory limit.
        seed: Random seed.
    """

    def __init__(
        self,
        agent_id: str = "informed",
        initial_fundamental: float = 100.0,
        fundamental_vol: float = 0.1,
        threshold: float = 0.5,
        order_size: float = 10.0,
        aggression: float = 0.5,
        max_inventory: float = 100.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(agent_id)
        self.fundamental: float = initial_fundamental
        self.fundamental_vol = fundamental_vol
        self.threshold = threshold
        self.order_size = order_size
        self.aggression = aggression
        self.max_inventory = max_inventory
        self._rng = np.random.default_rng(seed)
        self.fundamental_history: list[float] = [initial_fundamental]

    def step(self, t: int, book: OrderBook) -> list[Order]:
        # Update fundamental value (random walk)
        self.fundamental += self._rng.normal(0, self.fundamental_vol)
        self.fundamental = max(0.01, self.fundamental)
        self.fundamental_history.append(self.fundamental)

        mid = book.mid_price
        if mid is None:
            return []

        mispricing = self.fundamental - mid

        if abs(mispricing) < self.threshold:
            return []

        orders: list[Order] = []

        if mispricing > 0 and self.state.inventory < self.max_inventory:
            # Fundamental above mid: buy
            # Place a limit order slightly aggressive (inside the spread)
            best_ask = book.best_ask
            if best_ask is not None:
                price = round(mid + abs(mispricing) * self.aggression, 2)
                price = min(price, best_ask)  # Don't cross unreasonably
                order = Order(
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    price=price,
                    quantity=self.order_size,
                    agent_id=self.agent_id,
                )
                orders.append(order)

        elif mispricing < 0 and self.state.inventory > -self.max_inventory:
            # Fundamental below mid: sell
            best_bid = book.best_bid
            if best_bid is not None:
                price = round(mid - abs(mispricing) * self.aggression, 2)
                price = max(price, best_bid)
                order = Order(
                    side=OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    price=price,
                    quantity=self.order_size,
                    agent_id=self.agent_id,
                )
                orders.append(order)

        return orders
