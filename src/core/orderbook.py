"""Limit Order Book with price-time priority matching engine.

Implements a continuous double-auction order book using SortedDict for O(log n) price-level
operations and deques at each level for O(1) FIFO time-priority within a price.

The matching engine processes incoming orders against resting liquidity, generating
fills (trades) and, for LIMIT orders, inserting any unfilled remainder into the book.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

from sortedcontainers import SortedDict

from src.core.order import Order, OrderSide, OrderStatus, OrderType


@dataclass
class Trade:
    """A single execution (fill) produced by the matching engine.

    Attributes:
        price: Execution price.
        quantity: Executed quantity.
        aggressor_order_id: The incoming (taker) order.
        resting_order_id: The resting (maker) order.
        buyer_agent_id: Agent on the buy side (if applicable).
        seller_agent_id: Agent on the sell side (if applicable).
        timestamp: Timestamp of the trade.
    """

    price: float
    quantity: float
    aggressor_order_id: str
    resting_order_id: str
    buyer_agent_id: Optional[str] = None
    seller_agent_id: Optional[str] = None
    timestamp: int = 0


class OrderBook:
    """Limit Order Book with continuous double-auction matching.

    Price levels are stored in two SortedDicts:
      - bids: sorted in *descending* price order (negated keys)
      - asks: sorted in *ascending* price order

    Each price level holds a deque[Order] for FIFO time-priority.
    """

    def __init__(self) -> None:
        # bids keyed by -price so that iterating gives descending price order
        self._bids: SortedDict = SortedDict()  # key = -price -> deque[Order]
        self._asks: SortedDict = SortedDict()  # key =  price -> deque[Order]
        self._orders: dict[str, Order] = {}  # order_id -> Order (live orders only)
        self._trade_callbacks: list[Callable[[Trade], None]] = []

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def submit(self, order: Order) -> list[Trade]:
        """Submit an order to the book. Returns a list of trades generated."""
        if order.order_type == OrderType.FOK:
            return self._handle_fok(order)

        trades = self._match(order)

        if order.order_type == OrderType.MARKET or order.order_type == OrderType.IOC:
            # Cancel any unfilled remainder
            if not order.is_filled:
                order.cancel()
        elif order.order_type == OrderType.LIMIT:
            if not order.is_filled and order.status != OrderStatus.CANCELLED:
                self._insert(order)

        return trades

    def cancel(self, order_id: str) -> bool:
        """Cancel a resting order by ID. Returns True if successfully cancelled."""
        order = self._orders.pop(order_id, None)
        if order is None:
            return False
        order.cancel()
        # Remove from the price level
        book_side = self._bids if order.side == OrderSide.BUY else self._asks
        key = -order.price if order.side == OrderSide.BUY else order.price
        if key in book_side:
            level: deque = book_side[key]
            try:
                level.remove(order)
            except ValueError:
                pass
            if not level:
                del book_side[key]
        return True

    def on_trade(self, callback: Callable[[Trade], None]) -> None:
        """Register a callback invoked on each trade."""
        self._trade_callbacks.append(callback)

    # ------------------------------------------------------------------ #
    #  Accessors                                                          #
    # ------------------------------------------------------------------ #

    @property
    def best_bid(self) -> Optional[float]:
        if not self._bids:
            return None
        return -self._bids.keys()[0]

    @property
    def best_ask(self) -> Optional[float]:
        if not self._asks:
            return None
        return self._asks.keys()[0]

    @property
    def mid_price(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        return bb or ba

    @property
    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is not None and ba is not None:
            return ba - bb
        return None

    def bid_levels(self, depth: int = 10) -> list[tuple[float, float]]:
        """Return up to *depth* bid levels as (price, total_qty) descending by price."""
        result = []
        for neg_price in self._bids.keys()[:depth]:
            price = -neg_price
            total = sum(o.remaining for o in self._bids[neg_price])
            result.append((price, total))
        return result

    def ask_levels(self, depth: int = 10) -> list[tuple[float, float]]:
        """Return up to *depth* ask levels as (price, total_qty) ascending by price."""
        result = []
        for price_key in self._asks.keys()[:depth]:
            total = sum(o.remaining for o in self._asks[price_key])
            result.append((price_key, total))
        return result

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    @property
    def bid_volume(self) -> float:
        return sum(
            o.remaining for level in self._bids.values() for o in level
        )

    @property
    def ask_volume(self) -> float:
        return sum(
            o.remaining for level in self._asks.values() for o in level
        )

    @property
    def num_orders(self) -> int:
        return len(self._orders)

    # ------------------------------------------------------------------ #
    #  Matching Engine (private)                                          #
    # ------------------------------------------------------------------ #

    def _match(self, incoming: Order) -> list[Trade]:
        """Walk the opposite side of the book and generate fills."""
        trades: list[Trade] = []

        if incoming.side == OrderSide.BUY:
            opposite = self._asks
            price_ok = lambda ask_px: (  # noqa: E731
                incoming.price is None or ask_px <= incoming.price
            )
        else:
            opposite = self._bids
            price_ok = lambda neg_bid: (  # noqa: E731
                incoming.price is None or (-neg_bid) >= incoming.price
            )

        keys_to_remove: list = []

        for key in list(opposite.keys()):
            if incoming.is_filled:
                break
            if not price_ok(key):
                break

            level: deque[Order] = opposite[key]
            while level and not incoming.is_filled:
                resting = level[0]
                fill_qty = min(incoming.remaining, resting.remaining)

                # asks keyed by positive price; bids keyed by -price
                exec_price = key if incoming.side == OrderSide.BUY else -key

                incoming.fill(fill_qty)
                resting.fill(fill_qty)

                if incoming.side == OrderSide.BUY:
                    buyer_id, seller_id = incoming.agent_id, resting.agent_id
                else:
                    buyer_id, seller_id = resting.agent_id, incoming.agent_id

                trade = Trade(
                    price=exec_price if isinstance(exec_price, float) else float(exec_price),
                    quantity=fill_qty,
                    aggressor_order_id=incoming.order_id,
                    resting_order_id=resting.order_id,
                    buyer_agent_id=buyer_id,
                    seller_agent_id=seller_id,
                    timestamp=incoming.timestamp,
                )
                trades.append(trade)
                for cb in self._trade_callbacks:
                    cb(trade)

                if resting.is_filled:
                    level.popleft()
                    self._orders.pop(resting.order_id, None)

            if not level:
                keys_to_remove.append(key)

        for k in keys_to_remove:
            del opposite[k]

        return trades

    def _can_fill_fully(self, incoming: Order) -> bool:
        """Check whether the incoming order can be *completely* filled (for FOK)."""
        remaining = incoming.remaining

        if incoming.side == OrderSide.BUY:
            for ask_key in self._asks.keys():
                if incoming.price is not None and ask_key > incoming.price:
                    break
                for resting in self._asks[ask_key]:
                    remaining -= resting.remaining
                    if remaining <= 1e-12:
                        return True
        else:
            for neg_bid in self._bids.keys():
                bid_price = -neg_bid
                if incoming.price is not None and bid_price < incoming.price:
                    break
                for resting in self._bids[neg_bid]:
                    remaining -= resting.remaining
                    if remaining <= 1e-12:
                        return True
        return remaining <= 1e-12

    def _handle_fok(self, order: Order) -> list[Trade]:
        """Fill-Or-Kill: atomic fill or reject."""
        if self._can_fill_fully(order):
            return self._match(order)
        order.cancel()
        return []

    def _insert(self, order: Order) -> None:
        """Insert a limit order into the book."""
        if order.side == OrderSide.BUY:
            key = -order.price
            book_side = self._bids
        else:
            key = order.price
            book_side = self._asks

        if key not in book_side:
            book_side[key] = deque()
        book_side[key].append(order)
        self._orders[order.order_id] = order

    def __repr__(self) -> str:
        bb = f"{self.best_bid:.2f}" if self.best_bid else "---"
        ba = f"{self.best_ask:.2f}" if self.best_ask else "---"
        return f"OrderBook(bid={bb}, ask={ba}, orders={self.num_orders})"
