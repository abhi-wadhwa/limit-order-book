"""Streamlit interactive dashboard for the limit order book simulator.

Provides live visualizations:
- Order book depth chart and ladder view
- Price chart with volume bars
- Agent dashboard with PnL and inventory tracking
- Execution algorithm tester with cost analysis
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.core.agents import (
    AvellanedaStoikovMM,
    InformedTrader,
    MomentumTrader,
    NoiseTrader,
)
from src.core.execution import ISAlgo, TWAPAlgo, VWAPAlgo
from src.core.order import OrderSide
from src.core.simulation import Simulation


def create_depth_chart(sim: Simulation) -> go.Figure:
    """Create a depth chart showing cumulative bid/ask volume by price."""
    bids = sim.book.bid_levels(20)
    asks = sim.book.ask_levels(20)

    fig = go.Figure()

    if bids:
        bid_prices = [p for p, _ in bids]
        bid_cum_qty = np.cumsum([q for _, q in bids]).tolist()
        fig.add_trace(go.Scatter(
            x=bid_prices,
            y=bid_cum_qty,
            fill="tozeroy",
            name="Bids",
            line=dict(color="#2ecc71"),
            fillcolor="rgba(46, 204, 113, 0.3)",
        ))

    if asks:
        ask_prices = [p for p, _ in asks]
        ask_cum_qty = np.cumsum([q for _, q in asks]).tolist()
        fig.add_trace(go.Scatter(
            x=ask_prices,
            y=ask_cum_qty,
            fill="tozeroy",
            name="Asks",
            line=dict(color="#e74c3c"),
            fillcolor="rgba(231, 76, 60, 0.3)",
        ))

    fig.update_layout(
        title="Order Book Depth",
        xaxis_title="Price",
        yaxis_title="Cumulative Quantity",
        height=400,
        template="plotly_dark",
    )
    return fig


def create_ladder_view(sim: Simulation) -> tuple[list[dict], list[dict]]:
    """Create ladder-style data for bid and ask sides."""
    bids = sim.book.bid_levels(10)
    asks = sim.book.ask_levels(10)

    bid_data = [{"Price": f"{p:.2f}", "Qty": f"{q:.1f}"} for p, q in bids]
    ask_data = [{"Price": f"{p:.2f}", "Qty": f"{q:.1f}"} for p, q in asks]
    return bid_data, ask_data


def create_price_chart(sim: Simulation) -> go.Figure:
    """Create a price chart with volume bars from the market data feed."""
    mids = sim.feed.mid_prices()
    trade_prices = sim.feed.trade_prices()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=("Mid Price", "Trade Volume"),
    )

    if mids:
        valid_mids = [(i, m) for i, m in enumerate(mids) if m is not None]
        if valid_mids:
            xs, ys = zip(*valid_mids)
            fig.add_trace(go.Scatter(
                x=list(xs), y=list(ys),
                name="Mid Price",
                line=dict(color="#3498db", width=1.5),
            ), row=1, col=1)

    if trade_prices:
        timestamps = [t.timestamp for t in sim.feed.trades]
        volumes = [t.quantity for t in sim.feed.trades]
        colors = ["#2ecc71" if t.aggressor_side == "BUY" else "#e74c3c"
                  for t in sim.feed.trades]

        fig.add_trace(go.Bar(
            x=timestamps,
            y=volumes,
            name="Trade Volume",
            marker_color=colors,
        ), row=2, col=1)

    fig.update_layout(
        height=500,
        template="plotly_dark",
        showlegend=False,
    )
    fig.update_xaxes(title_text="Timestep", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def create_agent_pnl_chart(sim: Simulation) -> go.Figure:
    """Create PnL over time chart for all agents."""
    fig = go.Figure()

    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6",
              "#1abc9c", "#e67e22", "#95a5a6"]

    for i, (aid, agent) in enumerate(sim.agents.items()):
        if agent.state.pnl_history:
            fig.add_trace(go.Scatter(
                y=agent.state.pnl_history,
                name=f"{aid} ({agent.__class__.__name__})",
                line=dict(color=colors[i % len(colors)]),
            ))

    fig.update_layout(
        title="Agent PnL Over Time",
        xaxis_title="Timestep",
        yaxis_title="Mark-to-Market PnL",
        height=400,
        template="plotly_dark",
    )
    return fig


def create_inventory_chart(sim: Simulation) -> go.Figure:
    """Create inventory over time chart for all agents."""
    fig = go.Figure()

    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6",
              "#1abc9c", "#e67e22", "#95a5a6"]

    for i, (aid, agent) in enumerate(sim.agents.items()):
        if agent.state.inventory_history:
            fig.add_trace(go.Scatter(
                y=agent.state.inventory_history,
                name=f"{aid}",
                line=dict(color=colors[i % len(colors)]),
            ))

    fig.update_layout(
        title="Agent Inventory Over Time",
        xaxis_title="Timestep",
        yaxis_title="Net Inventory",
        height=350,
        template="plotly_dark",
    )
    return fig


def create_spread_chart(sim: Simulation) -> go.Figure:
    """Create spread over time chart."""
    spreads = sim.feed.spreads()
    valid = [(i, s) for i, s in enumerate(spreads) if s is not None]

    fig = go.Figure()
    if valid:
        xs, ys = zip(*valid)
        fig.add_trace(go.Scatter(
            x=list(xs), y=list(ys),
            name="Bid-Ask Spread",
            line=dict(color="#f39c12"),
            fill="tozeroy",
            fillcolor="rgba(243, 156, 18, 0.2)",
        ))

    fig.update_layout(
        title="Bid-Ask Spread Over Time",
        xaxis_title="Timestep",
        yaxis_title="Spread",
        height=300,
        template="plotly_dark",
    )
    return fig


def run_simulation(
    steps: int,
    initial_price: float,
    num_noise: int,
    gamma: float,
    sigma: float,
    seed: int,
) -> Simulation:
    """Create and run a simulation with the given parameters."""
    agents = [
        AvellanedaStoikovMM(
            agent_id="mm_avellaneda",
            gamma=gamma,
            sigma=sigma,
            k=1.5,
            total_time=steps,
        ),
        MomentumTrader(agent_id="momentum"),
        InformedTrader(
            agent_id="informed",
            initial_fundamental=initial_price,
            seed=seed,
        ),
    ]
    for i in range(num_noise):
        agents.append(NoiseTrader(agent_id=f"noise_{i}", seed=seed + i))

    sim = Simulation(
        agents=agents,
        initial_price=initial_price,
        total_steps=steps,
    )
    sim.run()
    return sim


def main() -> None:
    st.set_page_config(page_title="Limit Order Book Simulator", layout="wide")

    st.title("Limit Order Book Simulator")
    st.markdown("Agent-based simulation with Avellaneda-Stoikov market making, "
                "momentum traders, informed traders, and execution algorithms.")

    # Sidebar: simulation parameters
    st.sidebar.header("Simulation Parameters")
    steps = st.sidebar.slider("Timesteps", 100, 2000, 500, step=50)
    initial_price = st.sidebar.number_input("Initial Price", 50.0, 500.0, 100.0, step=10.0)
    num_noise = st.sidebar.slider("Noise Traders", 1, 10, 3)
    gamma = st.sidebar.slider("MM Risk Aversion (gamma)", 0.01, 1.0, 0.1, step=0.01)
    sigma = st.sidebar.slider("Volatility (sigma)", 0.1, 2.0, 0.5, step=0.1)
    seed = st.sidebar.number_input("Random Seed", 0, 9999, 42)

    st.sidebar.header("Execution Algorithm")
    exec_enabled = st.sidebar.checkbox("Run Execution Algo", value=False)
    exec_algo_name = st.sidebar.selectbox("Algorithm", ["TWAP", "VWAP", "Impl. Shortfall"])
    exec_side = st.sidebar.selectbox("Side", ["BUY", "SELL"])
    exec_qty = st.sidebar.number_input("Total Quantity", 10.0, 500.0, 100.0, step=10.0)
    exec_slices = st.sidebar.slider("Num Slices", 3, 30, 10)

    if st.sidebar.button("Run Simulation", type="primary"):
        with st.spinner("Running simulation..."):
            sim = run_simulation(steps, initial_price, num_noise, gamma, sigma, int(seed))

            # Optionally run execution algo
            exec_report = None
            if exec_enabled:
                side = OrderSide.BUY if exec_side == "BUY" else OrderSide.SELL
                start_t = steps // 4
                end_t = 3 * steps // 4

                if exec_algo_name == "TWAP":
                    algo = TWAPAlgo(side, exec_qty, exec_slices, start_t, end_t)
                elif exec_algo_name == "VWAP":
                    algo = VWAPAlgo(side, exec_qty, exec_slices, start_t, end_t)
                else:
                    algo = ISAlgo(side, exec_qty, exec_slices, start_t, end_t)

                # Re-run with the algo
                sim2 = run_simulation(steps, initial_price, num_noise, gamma, sigma, int(seed))
                sim2.add_exec_algo(algo)
                sim2.run()
                exec_report = algo.report(sim2.analytics.vwap())
                sim = sim2

        st.session_state["sim"] = sim
        st.session_state["exec_report"] = exec_report

    # Display results
    if "sim" in st.session_state:
        sim = st.session_state["sim"]

        # Top-level metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            mid = sim.book.mid_price
            st.metric("Mid Price", f"{mid:.2f}" if mid else "N/A")
        with col2:
            spread = sim.book.spread
            st.metric("Spread", f"{spread:.4f}" if spread else "N/A")
        with col3:
            st.metric("Total Trades", len(sim.feed.trades))
        with col4:
            vwap = sim.analytics.vwap()
            st.metric("VWAP", f"{vwap:.4f}" if vwap else "N/A")

        # Charts
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Order Book", "Price & Volume", "Agents", "Analytics"]
        )

        with tab1:
            st.plotly_chart(create_depth_chart(sim), use_container_width=True)

            col_bid, col_ask = st.columns(2)
            bid_data, ask_data = create_ladder_view(sim)
            with col_bid:
                st.subheader("Bids")
                if bid_data:
                    st.dataframe(bid_data, use_container_width=True)
                else:
                    st.write("No bids")
            with col_ask:
                st.subheader("Asks")
                if ask_data:
                    st.dataframe(ask_data, use_container_width=True)
                else:
                    st.write("No asks")

        with tab2:
            st.plotly_chart(create_price_chart(sim), use_container_width=True)
            st.plotly_chart(create_spread_chart(sim), use_container_width=True)

        with tab3:
            st.plotly_chart(create_agent_pnl_chart(sim), use_container_width=True)
            st.plotly_chart(create_inventory_chart(sim), use_container_width=True)

            st.subheader("Agent Summary")
            summary = sim.agent_summary()
            summary_rows = []
            for aid, info in summary.items():
                summary_rows.append({
                    "Agent": aid,
                    "Type": info["type"],
                    "Inventory": f"{info['inventory']:.1f}",
                    "PnL": f"{info['pnl']:.2f}",
                    "Trades": info["num_trades"],
                })
            st.dataframe(summary_rows, use_container_width=True)

        with tab4:
            st.subheader("Market Microstructure Metrics")
            metrics = sim.analytics.summary()
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                ofi = metrics.get("order_flow_imbalance")
                st.metric("Order Flow Imbalance",
                          f"{ofi:.4f}" if ofi is not None else "N/A")

                kl = metrics.get("kyles_lambda")
                st.metric("Kyle's Lambda",
                          f"{kl:.6f}" if kl is not None else "N/A")
            with mcol2:
                bab = metrics.get("bid_ask_bounce")
                st.metric("Bid-Ask Bounce (autocorr)",
                          f"{bab:.4f}" if bab is not None else "N/A")

                rs = metrics.get("realized_spread")
                st.metric("Realized Spread",
                          f"{rs:.6f}" if rs is not None else "N/A")

        # Execution algorithm report
        exec_report = st.session_state.get("exec_report")
        if exec_report is not None:
            st.divider()
            st.subheader("Execution Algorithm Report")
            ecol1, ecol2, ecol3, ecol4 = st.columns(4)
            with ecol1:
                st.metric("Algorithm", exec_report.algo_name)
            with ecol2:
                st.metric("Filled / Total",
                          f"{exec_report.filled_quantity:.1f} / {exec_report.total_quantity:.1f}")
            with ecol3:
                st.metric("Avg Fill Price", f"{exec_report.avg_fill_price:.4f}")
            with ecol4:
                st.metric("Slippage (bps)", f"{exec_report.slippage_bps:.2f}")

            st.metric("Arrival Price", f"{exec_report.arrival_price:.4f}")
            st.metric("Market VWAP", f"{exec_report.vwap_market:.4f}")

            if exec_report.fill_prices:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=exec_report.fill_prices,
                    mode="markers+lines",
                    name="Fill Prices",
                    marker=dict(size=6),
                ))
                fig.add_hline(y=exec_report.arrival_price,
                              line_dash="dash", line_color="yellow",
                              annotation_text="Arrival Price")
                fig.add_hline(y=exec_report.avg_fill_price,
                              line_dash="dot", line_color="cyan",
                              annotation_text="Avg Fill")
                fig.update_layout(
                    title="Execution Fill Prices",
                    xaxis_title="Fill #",
                    yaxis_title="Price",
                    height=350,
                    template="plotly_dark",
                )
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Configure parameters in the sidebar and click 'Run Simulation' to begin.")


if __name__ == "__main__":
    main()
