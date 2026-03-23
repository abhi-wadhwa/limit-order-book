# Limit Order Book Simulator

A high-fidelity limit order book (LOB) simulation with a matching engine, agent-based market participants, execution algorithms, and market microstructure analytics.

## Overview

This project implements a complete electronic exchange simulation from scratch:

- **Matching Engine** -- Continuous double-auction with strict price-time priority
- **Order Types** -- Limit, Market, IOC (Immediate-Or-Cancel), FOK (Fill-Or-Kill)
- **Agent-Based Simulation** -- Heterogeneous agents with distinct strategies
- **Execution Algorithms** -- TWAP, VWAP, Implementation Shortfall
- **Market Microstructure Analytics** -- OFI, Kyle's lambda, bid-ask bounce, realized spread
- **Interactive Dashboard** -- Real-time Streamlit visualization with Plotly charts

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Simulation Engine                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│  │  Market   │  │ Momentum │  │  Noise   │  │  Informed    │ │
│  │  Maker    │  │ Trader   │  │  Trader  │  │  Trader      │ │
│  │  (A-S)    │  │          │  │          │  │              │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬───────┘ │
│       │              │              │               │         │
│       └──────────────┴──────┬───────┴───────────────┘         │
│                             ▼                                 │
│                   ┌─────────────────┐                         │
│                   │   Order Book    │◄── Execution Algos      │
│                   │ Matching Engine │    (TWAP, VWAP, IS)     │
│                   └────────┬────────┘                         │
│                            │                                  │
│                   ┌────────▼────────┐                         │
│                   │  Market Data    │                         │
│                   │  Feed (L2)      │                         │
│                   └────────┬────────┘                         │
│                            │                                  │
│                   ┌────────▼────────┐                         │
│                   │   Analytics     │                         │
│                   │ (OFI, Lambda)   │                         │
│                   └─────────────────┘                         │
└──────────────────────────────────────────────────────────────┘
```

## Market Microstructure Theory

### Limit Order Book

A limit order book is a record of outstanding buy and sell orders at various price levels. The LOB operates as a **continuous double auction**: incoming orders are matched against resting orders using **price-time priority** -- the best price is matched first, and among orders at the same price, the earliest arrival is matched first.

### Avellaneda-Stoikov Market Making

The market maker uses the [Avellaneda-Stoikov (2008)](https://doi.org/10.1142/S0219024908004816) optimal quoting framework. The key insight is that a risk-averse market maker adjusts their quotes based on inventory risk and time horizon.

**Reservation price** (inventory-adjusted fair value):

$$r(s, q, t) = s - q \cdot \gamma \sigma^2 (T - t)$$

**Optimal spread** (width of the bid-ask quote):

$$\delta^* = \gamma \sigma^2 (T - t) + \frac{2}{\gamma} \ln\left(1 + \frac{\gamma}{k}\right)$$

Where:
- $s$ = current mid price
- $q$ = inventory (positive = long)
- $\gamma$ = risk aversion parameter
- $\sigma$ = asset volatility
- $T - t$ = time remaining to horizon
- $k$ = order arrival intensity (from the Poisson model of order flow)

The market maker quotes:
- **Bid**: $r - \delta^*/2$
- **Ask**: $r + \delta^*/2$

With large inventory, the reservation price shifts away from mid, causing asymmetric quotes that incentivize inventory-reducing fills.

### Kyle's Lambda

[Kyle (1985)](https://doi.org/10.2307/1913210) models a market with an informed trader, noise traders, and a market maker. The **price impact coefficient** $\lambda$ measures the permanent price impact per unit of signed order flow:

$$\Delta p = \lambda \cdot v + \varepsilon$$

where $v$ is signed trade volume (positive for buys). In Kyle's equilibrium:

$$\lambda = \frac{\sigma_v}{2 \sigma_u}$$

A higher $\lambda$ indicates more information asymmetry -- each unit of order flow moves the price more because the market maker suspects informed trading.

### Bid-Ask Bounce

The **bid-ask bounce** is a well-known microstructure effect where trade-to-trade returns exhibit negative first-order autocorrelation. This occurs because trades alternate between the bid and ask prices, creating an artificial oscillation in observed prices. The autocorrelation coefficient:

$$\rho_1 = \text{Corr}(r_t, r_{t+1})$$

In a pure spread model, $\rho_1 \approx -0.25$.

### Execution Algorithms

**TWAP (Time-Weighted Average Price):**
Splits a parent order into $N$ equal child orders at regular intervals over the execution window $[t_0, T]$:

$$q_i = \frac{Q}{N}, \quad t_i = t_0 + i \cdot \frac{T - t_0}{N}$$

**VWAP (Volume-Weighted Average Price):**
Distributes child orders proportional to the expected volume profile $w_i$:

$$q_i = Q \cdot w_i, \quad \sum_i w_i = 1$$

The default profile is U-shaped (higher participation at open/close), reflecting typical intraday volume patterns.

**Implementation Shortfall (Almgren-Chriss):**
Adapts execution speed based on realized price drift from arrival:

$$q_i = q_{\text{base}} \cdot f(\text{drift})$$

If the price moves adversely, execution accelerates (higher urgency). If it moves favorably, it decelerates.

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/abhi-wadhwa/limit-order-book.git
cd limit-order-book

# Install the package
pip install -e ".[dev]"
```

### Run the Demo

```bash
python examples/demo.py
```

### Run the Interactive Dashboard

```bash
streamlit run src/viz/app.py
```

### Run via CLI

```bash
# Run a simulation from the command line
python -m src.cli simulate --steps 500 --price 100 --noise 3 --seed 42

# Or use the installed entry point
lob simulate --steps 500
```

### Run Tests

```bash
pytest tests/ -v
```

## Project Structure

```
limit-order-book/
├── src/
│   ├── core/
│   │   ├── order.py           # Order data structures (Limit, Market, IOC, FOK)
│   │   ├── orderbook.py       # Order book + matching engine
│   │   ├── market_data.py     # L2 snapshots, trade stream
│   │   ├── agents/
│   │   │   ├── base.py        # Abstract agent class
│   │   │   ├── market_maker.py    # Avellaneda-Stoikov market maker
│   │   │   ├── momentum.py        # Trend-following (dual MA crossover)
│   │   │   ├── noise.py           # Random order flow
│   │   │   └── informed.py        # Trades toward hidden fundamental
│   │   ├── analytics.py       # OFI, VWAP, Kyle's lambda, bid-ask bounce
│   │   ├── execution.py       # TWAP, VWAP, Implementation Shortfall
│   │   └── simulation.py      # Simulation orchestrator
│   ├── viz/
│   │   └── app.py             # Streamlit dashboard
│   └── cli.py                 # Typer CLI
├── tests/
│   ├── test_orderbook.py      # Order book invariants
│   ├── test_matching.py       # Price-time priority matching
│   ├── test_agents.py         # Agent behavior tests
│   ├── test_analytics.py      # Analytics correctness
│   └── test_execution.py      # Execution algorithm tests
├── examples/
│   └── demo.py                # Full simulation demo
├── pyproject.toml
├── Makefile
├── Dockerfile
└── .github/workflows/ci.yml
```

## Key Design Decisions

- **SortedDict** for price levels: O(log n) insertion/deletion, O(1) best-price access
- **Deque** at each price level: O(1) FIFO for time-priority matching
- **Callback-based trade routing**: decoupled agent fill handling via observer pattern
- **Dataclass-based orders**: immutable semantics with fill tracking
- **Discrete-time simulation**: agents act synchronously each timestep for reproducibility

## License

MIT
