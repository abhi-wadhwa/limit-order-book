"""Demo script: run a full simulation and print results.

Usage:
    python examples/demo.py
"""

from src.core.agents import (
    AvellanedaStoikovMM,
    InformedTrader,
    MomentumTrader,
    NoiseTrader,
)
from src.core.execution import TWAPAlgo, VWAPAlgo
from src.core.order import OrderSide
from src.core.simulation import Simulation


def main() -> None:
    print("=" * 70)
    print("  Limit Order Book Simulator -- Demo")
    print("=" * 70)

    # Configure agents
    agents = [
        AvellanedaStoikovMM(
            agent_id="mm_avellaneda",
            gamma=0.1,
            sigma=0.5,
            k=1.5,
            total_time=500,
            order_size=10.0,
        ),
        MomentumTrader(
            agent_id="momentum_1",
            short_window=10,
            long_window=30,
            order_size=5.0,
        ),
        InformedTrader(
            agent_id="informed_1",
            initial_fundamental=100.0,
            fundamental_vol=0.1,
            threshold=0.5,
            order_size=8.0,
            seed=42,
        ),
        NoiseTrader(agent_id="noise_1", trade_prob=0.3, seed=100),
        NoiseTrader(agent_id="noise_2", trade_prob=0.3, seed=200),
        NoiseTrader(agent_id="noise_3", trade_prob=0.4, seed=300),
    ]

    # Create simulation
    sim = Simulation(
        agents=agents,
        initial_price=100.0,
        total_steps=500,
        seed_depth=5,
        seed_spread=0.5,
    )

    # Add a TWAP execution algorithm
    twap = TWAPAlgo(
        side=OrderSide.BUY,
        total_quantity=100.0,
        num_slices=10,
        start_time=50,
        end_time=400,
    )
    sim.add_exec_algo(twap)

    # Run the simulation
    print("\nRunning 500-step simulation...")
    sim.run()

    # Results
    print(f"\n--- Order Book State ---")
    print(f"  Book: {sim.book}")
    print(f"  Mid price: {sim.book.mid_price:.4f}" if sim.book.mid_price else "  Mid price: N/A")
    print(f"  Spread: {sim.book.spread:.4f}" if sim.book.spread else "  Spread: N/A")
    print(f"  Bid volume: {sim.book.bid_volume:.1f}")
    print(f"  Ask volume: {sim.book.ask_volume:.1f}")

    print(f"\n--- Market Data ---")
    print(f"  Total trades: {len(sim.feed.trades)}")
    print(f"  Total snapshots: {len(sim.feed.snapshots)}")

    # Analytics
    metrics = sim.analytics.summary()
    print(f"\n--- Market Microstructure Metrics ---")
    for key, val in metrics.items():
        if val is not None:
            if isinstance(val, float):
                print(f"  {key}: {val:.6f}")
            else:
                print(f"  {key}: {val}")
        else:
            print(f"  {key}: N/A")

    # Agent summary
    print(f"\n--- Agent Summary ---")
    print(f"  {'Agent':<20} {'Type':<25} {'Inventory':>10} {'PnL':>12} {'Trades':>8}")
    print(f"  {'-'*18:<20} {'-'*23:<25} {'-'*10:>10} {'-'*12:>12} {'-'*8:>8}")
    for aid, info in sim.agent_summary().items():
        print(
            f"  {aid:<20} {info['type']:<25} {info['inventory']:>10.1f} "
            f"{info['pnl']:>12.2f} {info['num_trades']:>8}"
        )

    # TWAP execution report
    twap_report = twap.report(sim.analytics.vwap())
    print(f"\n--- TWAP Execution Report ---")
    print(f"  Total quantity: {twap_report.total_quantity:.1f}")
    print(f"  Filled quantity: {twap_report.filled_quantity:.1f}")
    print(f"  Num child orders: {twap_report.num_child_orders}")
    print(f"  Num fills: {twap_report.num_fills}")
    print(f"  Avg fill price: {twap_report.avg_fill_price:.4f}")
    print(f"  Arrival price: {twap_report.arrival_price:.4f}")
    print(f"  Market VWAP: {twap_report.vwap_market:.4f}")
    print(f"  Slippage (bps): {twap_report.slippage_bps:.2f}")

    print("\n" + "=" * 70)
    print("  Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
