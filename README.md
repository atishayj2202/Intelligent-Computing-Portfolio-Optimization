# Institutional Portfolio Construction under Cardinality Friction
**A Drawdown-Penalized Metaheuristic Approach (AOBL-SOS)**

This repository contains the complete, reproducible codebase for our submission to *The Journal of Portfolio Management*.

## Overview
Traditional quadratic programming (Markowitz) fails when non-convex cardinality constraints ($K=30$) and upper bounds (20%) are introduced, resulting in unstable allocations that degrade severely under out-of-sample stress and execution friction. 

This repository implements **AOBL-SOS** (Adaptive Opposition-Based Learning Symbiotic Organisms Search). Unlike standard algorithms, AOBL-SOS:
1. Prevents premature convergence by monitoring swarm stagnation and injecting rank-reversal perturbations.
2. Optimizes a path-dependent objective function that explicitly penalizes maximum drawdown.
3. Produces highly stable allocations (3% monthly turnover) that survive institutional transaction costs.

## Repository Structure
- `data/sp500_daily.csv`: The 179-asset S&P 500 universe returns (2012–2025) used for all out-of-sample backtests.
- `scripts/algorithms.py`: Contains the raw implementations for standard SOS and the proposed AOBL-SOS metaheuristic.
- `scripts/evaluation.py`: Contains the `obj_sharpe_drawdown` fitness function and the `compute_net_metrics` function to explicitly calculate net returns minus transaction costs.
- `scripts/utils.py`: Contains cardinality enforcing constraints and normalization routines.
- `scripts/run_experiments.py`: The central execution script to fully replicate all tables (including Walk-Forward expanding windows) and figures in the paper.
- `results/`: Contains the generated outputs including transaction cost sensitivity, rebalancing frequency analysis, walk forward validation, and cumulative return plots.
- `manuscript/`: Contains the finalized, formatted `.tex` submission, the `.bib` bibliography, and copies of all generated data/graphs for direct compilation.

## Replication
To replicate the exact figures and metrics presented in the paper, execute the following from the root directory:

```bash
python3 -m scripts.run_experiments
```

This will run the full Walk-Forward out-of-sample optimization alongside all other sensitivity tests and deposit the completely reproducible results into the `results/` folder.
