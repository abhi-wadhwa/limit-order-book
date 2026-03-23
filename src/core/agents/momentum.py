"""Momentum (trend-following) trader agent.

Tracks a short-term and long-term moving average of the mid price.
When the short MA crosses above the long MA, submits a buy order.
When the short MA crosses below the long MA, submits a sell order.
Uses IOC orders to trade immediately at the current price.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from src.core.agents.base import Agent
from src.core.order import Order, OrderSide, OrderType
from src.core.orderbook import OrderBook


class MomentumTrader(Agent):
    """Trend-following agent using dual moving-average crossover.

    Parameters:
        agent_id: Unique identifier.
        short_window: Lookback for the fast moving average.
        long_window: Lookback for the slow moving average.
        order_size: Size of each trade.
        cooldown: Minimum timesteps between trades.
        max_inventory: Absolute inventory limit.
    """

    def __init__(
        self,
        agent_id: str = "momentum",
        short_window: int = 10,
        long_window: int = 30,
        order_size: float = 5.0,
        cooldown: int = 5,
        max_inventory: float = 50.0,
    ) -> None:
        super().__init__(agent_id)
        self.short_window = short_window
        self.long_window = long_window
        self.order_size = order_size
        self.cooldown = cooldown
        self.max_inventory = max_inventory
        self._price_history: deque[float] = deque(maxlen=long_window + 1)
        self._last_trade_t: int = -cooldown

    def step(self, t: int, book: OrderBook) -> list[Order]:
        mid = book.mid_price
        if mid is None:
            return []

        self._price_history.append(mid)

        if len(self._price_history) < self.long_window:
            return []

        if t - self._last_trade_t < self.cooldown:
            return []

        prices = list(self._price_history)
        short_ma = float(np.mean(prices[-self.short_window :]))
        long_ma = float(np.mean(prices[-self.long_window :]))

        orders: list[Order] = []

        if short_ma > long_ma and self.state.inventory < self.max_inventory:
            # Bullish crossover: buy
            order = Order(
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=self.order_size,
                agent_id=self.agent_id,
            )
            orders.append(order)
            self._last_trade_t = t

        elif short_ma < long_ma and self.state.inventory > -self.max_inventory:
            # Bearish crossover: sell
            order = Order(
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=self.order_size,
                agent_id=self.agent_id,
            )
            orders.append(order)
            self._last_trade_t = t

        return orders
