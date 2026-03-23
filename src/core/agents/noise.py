"""Noise trader agent.

Submits random limit orders around the mid price at each timestep with a
configurable probability. Provides background liquidity and creates realistic
price fluctuations in the simulation.
"""

from __future__ import annotations

import numpy as np

from src.core.agents.base import Agent
from src.core.order import Order, OrderSide, OrderType
from src.core.orderbook import OrderBook


class NoiseTrader(Agent):
    """Random-order agent that provides background activity.

    Parameters:
        agent_id: Unique identifier.
        trade_prob: Probability of submitting an order each timestep.
        order_size_mean: Mean order size (lognormal).
        price_noise_std: Std dev of price offset from mid (in price units).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        agent_id: str = "noise",
        trade_prob: float = 0.3,
        order_size_mean: float = 5.0,
        price_noise_std: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(agent_id)
        self.trade_prob = trade_prob
        self.order_size_mean = order_size_mean
        self.price_noise_std = price_noise_std
        self._rng = np.random.default_rng(seed)

    def step(self, t: int, book: OrderBook) -> list[Order]:
        if self._rng.random() > self.trade_prob:
            return []

        mid = book.mid_price
        if mid is None:
            return []

        side = OrderSide.BUY if self._rng.random() < 0.5 else OrderSide.SELL

        # Lognormal order size
        size = max(1.0, self._rng.lognormal(mean=np.log(self.order_size_mean), sigma=0.5))
        size = round(size, 1)

        # Random offset from mid
        offset = self._rng.normal(0, self.price_noise_std)

        if side == OrderSide.BUY:
            price = round(mid - abs(offset), 2)
        else:
            price = round(mid + abs(offset), 2)

        price = max(0.01, price)

        # Mix of limit and market orders
        if self._rng.random() < 0.2:
            order = Order(
                side=side,
                order_type=OrderType.MARKET,
                quantity=size,
                agent_id=self.agent_id,
            )
        else:
            order = Order(
                side=side,
                order_type=OrderType.LIMIT,
                price=price,
                quantity=size,
                agent_id=self.agent_id,
            )

        return [order]
