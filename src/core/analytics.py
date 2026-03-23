"""Market microstructure analytics.

Computes standard market-quality and information-content metrics:
- Order Flow Imbalance (OFI)
- VWAP (Volume-Weighted Average Price)
- Kyle's Lambda (price impact coefficient)
- Bid-Ask Bounce (negative first-order autocorrelation of returns)
- Realized Spread (post-trade price reversion measure)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.core.market_data import MarketDataFeed


class Analytics:
    """Market microstructure analytics computed from a MarketDataFeed.

    All methods are stateless computations over the feed's accumulated data.
    """

    def __init__(self, feed: MarketDataFeed) -> None:
        self._feed = feed

    def vwap(self) -> Optional[float]:
        """Volume-Weighted Average Price over all trades.

        VWAP = sum(price_i * qty_i) / sum(qty_i)
        """
        trades = self._feed.trades
        if not trades:
            return None
        total_value = sum(t.price * t.quantity for t in trades)
        total_volume = sum(t.quantity for t in trades)
        if total_volume == 0:
            return None
        return total_value / total_volume

    def order_flow_imbalance(self, window: int = 50) -> Optional[float]:
        """Order Flow Imbalance over the last *window* trades.

        OFI = (buy_volume - sell_volume) / (buy_volume + sell_volume)

        Returns a value in [-1, 1]. Positive = net buying pressure.
        """
        trades = self._feed.trades[-window:]
        if not trades:
            return None
        buy_vol = sum(t.quantity for t in trades if t.aggressor_side == "BUY")
        sell_vol = sum(t.quantity for t in trades if t.aggressor_side == "SELL")
        total = buy_vol + sell_vol
        if total == 0:
            return None
        return (buy_vol - sell_vol) / total

    def kyles_lambda(self, window: int = 50) -> Optional[float]:
        """Estimate Kyle's lambda: price impact per unit of signed order flow.

        Runs OLS regression: delta_p = lambda * signed_volume + epsilon

        lambda = Cov(delta_p, signed_vol) / Var(signed_vol)

        Kyle (1985) shows lambda = sigma_v / (2 * sigma_u) in equilibrium,
        measuring the permanent price impact of informed trading.
        """
        trades = self._feed.trades
        if len(trades) < window + 1:
            return None

        recent = trades[-window:]
        prices = np.array([t.price for t in recent])
        # Signed volume: positive for buys, negative for sells
        signed_vols = np.array(
            [t.quantity if t.aggressor_side == "BUY" else -t.quantity for t in recent]
        )

        # Price changes between consecutive trades
        dp = np.diff(prices)
        sv = signed_vols[1:]  # align with dp

        if len(dp) < 2:
            return None

        var_sv = np.var(sv)
        if var_sv < 1e-12:
            return None

        cov = np.cov(dp, sv)[0, 1]
        return float(cov / var_sv)

    def bid_ask_bounce(self, window: int = 50) -> Optional[float]:
        """Bid-ask bounce: first-order autocorrelation of trade-to-trade returns.

        In a pure bid-ask bounce model, returns exhibit negative autocorrelation
        as prices alternate between bid and ask. Values near -0.25 suggest
        the spread is the dominant source of short-term return variance.
        """
        trades = self._feed.trades
        if len(trades) < window + 2:
            return None

        prices = np.array([t.price for t in trades[-window:]])
        returns = np.diff(np.log(prices + 1e-10))

        if len(returns) < 2:
            return None

        # First-order autocorrelation
        r1 = returns[:-1]
        r2 = returns[1:]
        corr_matrix = np.corrcoef(r1, r2)
        return float(corr_matrix[0, 1])

    def realized_spread(self, lag: int = 5) -> Optional[float]:
        """Realized spread: measures how much of the effective spread is
        compensation vs. adverse selection.

        Realized spread at lag k:
            RS = 2 * direction * (trade_price - mid_price_{t+k}) / mid_price

        where direction = +1 for buyer-initiated, -1 for seller-initiated.

        A small realized spread relative to the quoted spread indicates
        high adverse selection (informed trading).
        """
        trades = self._feed.trades
        mids = self._feed.mid_prices()

        if len(trades) < lag + 1 or len(mids) < lag + 1:
            return None

        spreads = []
        for i in range(len(trades) - lag):
            trade = trades[i]
            # Find the mid price 'lag' snapshots ahead
            # Map trade timestamp to snapshot index (approximate)
            future_idx = min(trade.timestamp + lag, len(mids) - 1)
            future_mid = mids[future_idx]
            current_mid = mids[min(trade.timestamp, len(mids) - 1)]

            if future_mid is None or current_mid is None or current_mid == 0:
                continue

            direction = 1.0 if trade.aggressor_side == "BUY" else -1.0
            rs = 2.0 * direction * (trade.price - future_mid) / current_mid
            spreads.append(rs)

        if not spreads:
            return None
        return float(np.mean(spreads))

    def summary(self) -> dict[str, Optional[float]]:
        """Return a dictionary of all analytics metrics."""
        return {
            "vwap": self.vwap(),
            "order_flow_imbalance": self.order_flow_imbalance(),
            "kyles_lambda": self.kyles_lambda(),
            "bid_ask_bounce": self.bid_ask_bounce(),
            "realized_spread": self.realized_spread(),
            "num_trades": len(self._feed.trades),
            "num_snapshots": len(self._feed.snapshots),
        }
