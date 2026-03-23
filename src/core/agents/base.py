"""Base agent class for the agent-based simulation.

Every simulation agent subclasses Agent and implements the step() method,
which is called once per simulation timestep. Agents observe market data
and submit/cancel orders on the book.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from src.core.order import Order, OrderSide
from src.core.orderbook import OrderBook, Trade


@dataclass
class AgentState:
    """Mutable state tracked for each agent during the simulation.

    Attributes:
        inventory: Net position (positive = long, negative = short).
        cash: Cumulative cash from trading (sells add, buys subtract).
        pnl_history: PnL at each timestep.
        inventory_history: Inventory at each timestep.
        num_trades: Total number of fills this agent participated in.
    """

    inventory: float = 0.0
    cash: float = 0.0
    pnl_history: list[float] = field(default_factory=list)
    inventory_history: list[float] = field(default_factory=list)
    num_trades: int = 0


class Agent(ABC):
    """Abstract base for all simulation agents.

    Subclasses must implement step(t, book) which is called each timestep.
    The agent may submit or cancel orders on the book, and the simulation
    framework tracks fills automatically via on_fill().
    """

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.state = AgentState()
        self._active_order_ids: set[str] = set()

    @abstractmethod
    def step(self, t: int, book: OrderBook) -> list[Order]:
        """Called each timestep. Return a list of orders to submit.

        Args:
            t: Current simulation timestep.
            book: The order book (read access for market data, submit via return).

        Returns:
            A list of Order objects to submit to the book.
        """
        ...

    def on_fill(self, trade: Trade, side: OrderSide) -> None:
        """Called by the simulation when a fill involves this agent.

        Updates inventory and cash automatically.
        """
        if side == OrderSide.BUY:
            self.state.inventory += trade.quantity
            self.state.cash -= trade.price * trade.quantity
        else:
            self.state.inventory -= trade.quantity
            self.state.cash += trade.price * trade.quantity
        self.state.num_trades += 1

    def mark_to_market(self, mid_price: float) -> float:
        """Calculate mark-to-market PnL: cash + inventory * mid_price."""
        return self.state.cash + self.state.inventory * mid_price

    def record_state(self, mid_price: Optional[float]) -> None:
        """Snapshot current PnL and inventory for history tracking."""
        price = mid_price if mid_price is not None else 0.0
        self.state.pnl_history.append(self.mark_to_market(price))
        self.state.inventory_history.append(self.state.inventory)

    def cancel_all(self, book: OrderBook) -> None:
        """Cancel all active orders for this agent."""
        for oid in list(self._active_order_ids):
            book.cancel(oid)
        self._active_order_ids.clear()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.agent_id}, "
            f"inv={self.state.inventory:.1f}, "
            f"cash={self.state.cash:.2f})"
        )
