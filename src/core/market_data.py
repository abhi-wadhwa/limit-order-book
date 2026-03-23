"""Market data feed: L2 book snapshots and trade stream.

Provides a clean interface for consumers (analytics, visualization, execution algos)
to observe book state without coupling to the internal OrderBook representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.core.orderbook import OrderBook, Trade


@dataclass
class L2Snapshot:
    """Level-2 (price-aggregated) snapshot of the order book.

    Attributes:
        timestamp: Simulation timestep when this snapshot was taken.
        bids: List of (price, total_qty) tuples, descending by price.
        asks: List of (price, total_qty) tuples, ascending by price.
        mid_price: Midpoint of best bid and best ask.
        spread: Best ask minus best bid.
    """

    timestamp: int
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]
    mid_price: Optional[float]
    spread: Optional[float]


@dataclass
class TradeRecord:
    """Enriched trade record for the public trade stream.

    Attributes:
        timestamp: Simulation timestep.
        price: Execution price.
        quantity: Executed quantity.
        aggressor_side: 'BUY' if the taker was a buyer, 'SELL' otherwise.
    """

    timestamp: int
    price: float
    quantity: float
    aggressor_side: str


class MarketDataFeed:
    """Collects and disseminates market data from an OrderBook.

    Maintains a rolling history of L2 snapshots and trades for downstream
    analytics and visualization.
    """

    def __init__(self, book: OrderBook, depth: int = 10) -> None:
        self._book = book
        self._depth = depth
        self.snapshots: list[L2Snapshot] = []
        self.trades: list[TradeRecord] = []
        self._current_timestep: int = 0

        # Register trade listener
        book.on_trade(self._on_trade)

    def set_timestep(self, t: int) -> None:
        """Update the current simulation timestep."""
        self._current_timestep = t

    def snapshot(self) -> L2Snapshot:
        """Take an L2 snapshot of the current book state and record it."""
        snap = L2Snapshot(
            timestamp=self._current_timestep,
            bids=self._book.bid_levels(self._depth),
            asks=self._book.ask_levels(self._depth),
            mid_price=self._book.mid_price,
            spread=self._book.spread,
        )
        self.snapshots.append(snap)
        return snap

    def _on_trade(self, trade: Trade) -> None:
        """Callback invoked by the order book on each fill."""
        # Determine aggressor side: if aggressor is a buyer, we record BUY
        self.trades.append(
            TradeRecord(
                timestamp=self._current_timestep,
                price=trade.price,
                quantity=trade.quantity,
                aggressor_side="BUY" if trade.buyer_agent_id == trade.buyer_agent_id else "SELL",
            )
        )

    def last_trade_price(self) -> Optional[float]:
        """Most recent trade price, or None."""
        return self.trades[-1].price if self.trades else None

    def trade_prices(self) -> list[float]:
        """List of all trade prices in order."""
        return [t.price for t in self.trades]

    def trade_volumes(self) -> list[float]:
        """List of all trade quantities in order."""
        return [t.quantity for t in self.trades]

    def mid_prices(self) -> list[Optional[float]]:
        """List of mid prices from snapshots."""
        return [s.mid_price for s in self.snapshots]

    def spreads(self) -> list[Optional[float]]:
        """List of spreads from snapshots."""
        return [s.spread for s in self.snapshots]
