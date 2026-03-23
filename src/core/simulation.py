"""Simulation orchestrator for the agent-based order book simulation.

Coordinates the order book, agents, market data feed, and execution algorithms
through discrete timesteps. Each step:
  1. Agents observe the book and submit orders.
  2. Orders are matched by the engine.
  3. Fills are routed back to agents for P&L tracking.
  4. Market data snapshots are recorded.
"""

from __future__ import annotations

import structlog
from typing import Optional

from src.core.agents.base import Agent
from src.core.analytics import Analytics
from src.core.execution import ExecutionAlgo
from src.core.market_data import MarketDataFeed
from src.core.order import Order, OrderSide, OrderType
from src.core.orderbook import OrderBook, Trade

logger = structlog.get_logger()


class Simulation:
    """Discrete-time simulation engine for the limit order book.

    Parameters:
        agents: List of Agent instances participating in the simulation.
        initial_price: Initial mid price used to seed the book.
        total_steps: Number of simulation timesteps.
        seed_depth: Number of initial limit orders on each side to seed liquidity.
        seed_spread: Half-spread for the seed orders (in price units).
    """

    def __init__(
        self,
        agents: list[Agent],
        initial_price: float = 100.0,
        total_steps: int = 1000,
        seed_depth: int = 5,
        seed_spread: float = 0.5,
    ) -> None:
        self.book = OrderBook()
        self.agents = {a.agent_id: a for a in agents}
        self.initial_price = initial_price
        self.total_steps = total_steps
        self.feed = MarketDataFeed(self.book)
        self.analytics = Analytics(self.feed)
        self._exec_algos: list[ExecutionAlgo] = []
        self._current_step = 0
        self._trade_log: list[Trade] = []

        # Register trade callback for agent fill routing
        self.book.on_trade(self._route_fill)

        # Seed the order book with initial liquidity
        self._seed_book(seed_depth, seed_spread)

    def _seed_book(self, depth: int, spread: float) -> None:
        """Populate the book with initial resting orders to establish a market."""
        for i in range(depth):
            offset = spread * (i + 1)
            bid_price = round(self.initial_price - offset, 2)
            ask_price = round(self.initial_price + offset, 2)
            qty = 50.0

            bid = Order(
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=bid_price,
                quantity=qty,
                agent_id="_seed",
            )
            ask = Order(
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=ask_price,
                quantity=qty,
                agent_id="_seed",
            )
            self.book.submit(bid)
            self.book.submit(ask)

    def add_exec_algo(self, algo: ExecutionAlgo) -> None:
        """Register an execution algorithm to run during the simulation."""
        self._exec_algos.append(algo)

    def step(self) -> int:
        """Execute a single simulation timestep. Returns the current step number."""
        t = self._current_step
        self.feed.set_timestep(t)

        # 1. Agents submit orders
        for agent in self.agents.values():
            orders = agent.step(t, self.book)
            for order in orders:
                self.book.submit(order)

        # 2. Execution algorithms submit child orders
        for algo in self._exec_algos:
            if not algo.is_done:
                child_orders = algo.step(t, self.book)
                for order in child_orders:
                    trades = self.book.submit(order)
                    for trade in trades:
                        algo.on_fill(trade)

        # 3. Record snapshots and agent state
        snap = self.feed.snapshot()
        mid = snap.mid_price

        for agent in self.agents.values():
            agent.record_state(mid)

        self._current_step += 1
        return t

    def run(self, steps: Optional[int] = None) -> None:
        """Run the simulation for the specified number of steps (or all remaining)."""
        target = steps if steps is not None else self.total_steps
        for _ in range(target):
            if self._current_step >= self.total_steps:
                break
            self.step()

        logger.info(
            "simulation_complete",
            total_steps=self._current_step,
            total_trades=len(self.feed.trades),
            num_agents=len(self.agents),
        )

    def _route_fill(self, trade: Trade) -> None:
        """Route trade fills to the appropriate agents."""
        self._trade_log.append(trade)

        # Route to buyer
        if trade.buyer_agent_id and trade.buyer_agent_id in self.agents:
            self.agents[trade.buyer_agent_id].on_fill(trade, OrderSide.BUY)

        # Route to seller
        if trade.seller_agent_id and trade.seller_agent_id in self.agents:
            self.agents[trade.seller_agent_id].on_fill(trade, OrderSide.SELL)

    @property
    def current_step(self) -> int:
        return self._current_step

    def agent_summary(self) -> dict[str, dict]:
        """Return a summary of each agent's state."""
        mid = self.book.mid_price or self.initial_price
        result = {}
        for aid, agent in self.agents.items():
            result[aid] = {
                "type": agent.__class__.__name__,
                "inventory": agent.state.inventory,
                "cash": agent.state.cash,
                "pnl": agent.mark_to_market(mid),
                "num_trades": agent.state.num_trades,
            }
        return result
