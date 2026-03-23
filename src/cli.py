"""Command-line interface for the limit order book simulator."""

from __future__ import annotations

import typer
import structlog

structlog.configure(
    processors=[
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

app = typer.Typer(
    name="lob",
    help="Limit Order Book Simulator -- matching engine, ABM, execution algos.",
)


@app.command()
def simulate(
    steps: int = typer.Option(500, "--steps", "-s", help="Number of simulation timesteps"),
    price: float = typer.Option(100.0, "--price", "-p", help="Initial mid price"),
    num_noise: int = typer.Option(3, "--noise", "-n", help="Number of noise traders"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
) -> None:
    """Run an agent-based simulation and print summary statistics."""
    from src.core.agents import AvellanedaStoikovMM, InformedTrader, MomentumTrader, NoiseTrader
    from src.core.simulation import Simulation

    agents = [
        AvellanedaStoikovMM(agent_id="mm", gamma=0.1, sigma=0.5, k=1.5, total_time=steps),
        MomentumTrader(agent_id="mom", short_window=10, long_window=30),
        InformedTrader(agent_id="inf", initial_fundamental=price, seed=seed),
    ]
    for i in range(num_noise):
        agents.append(NoiseTrader(agent_id=f"noise_{i}", seed=seed + i))

    sim = Simulation(agents=agents, initial_price=price, total_steps=steps)
    sim.run()

    typer.echo("\n--- Simulation Complete ---")
    typer.echo(f"Steps: {sim.current_step}")
    typer.echo(f"Total trades: {len(sim.feed.trades)}")
    typer.echo(f"Final mid price: {sim.book.mid_price:.4f}" if sim.book.mid_price else "No mid")
    typer.echo(f"Book: {sim.book}")

    metrics = sim.analytics.summary()
    typer.echo("\n--- Market Microstructure Metrics ---")
    for key, val in metrics.items():
        if val is not None:
            if isinstance(val, float):
                typer.echo(f"  {key}: {val:.6f}")
            else:
                typer.echo(f"  {key}: {val}")

    typer.echo("\n--- Agent Summary ---")
    for aid, info in sim.agent_summary().items():
        typer.echo(f"  {aid} ({info['type']}): inv={info['inventory']:.1f}, "
                   f"pnl={info['pnl']:.2f}, trades={info['num_trades']}")


@app.command()
def ui() -> None:
    """Launch the Streamlit interactive dashboard."""
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "src/viz/app.py"],
        check=True,
    )


if __name__ == "__main__":
    app()
