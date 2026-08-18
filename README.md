# Institutional Portfolio Construction under Cardinality Friction

**Authors:** Atishaya Jain, Karan Jangra, Vaibhav Pacherwal  
**Affiliation:** Netaji Subhas University of Technology (NSUT), Delhi  

## Overview
This repository contains the official replication code for the manuscript: **"Institutional Portfolio Construction under Cardinality Friction: A Drawdown-Penalized Metaheuristic Approach."**

The project introduces **AOBL-SOS (Adaptive Opposition-Based Learning Symbiotic Organisms Search)**, a novel parameter-free metaheuristic designed to optimize high-dimensional institutional portfolios under strict, non-convex constraints (budget, non-negativity, $K=30$ cardinality bounds, and $20\%$ individual asset caps).

Unlike traditional mean-variance optimizers which degenerate when faced with cardinality constraints and tail-risk events, this framework directly maximizes a **drawdown-penalized Sharpe ratio**. 

### Key Innovations Addressed in Code:
- **Almgren-Chriss Market Impact Model:** All out-of-sample execution is strictly penalized by the non-linear Almgren-Chriss square-root model to simulate realistic institutional capacity ($100M AUM) and bid-ask friction.
- **Jobson-Korkie (Memmel) Significance:** Formal statistical testing for alpha generation vs. naive $1/N$ and Ledoit-Wolf shrinkage benchmarks.
- **1,000-Path Monte Carlo CVaR:** Block-bootstrap resampling to rigorously quantify 95\% Conditional Value-at-Risk across massive structural permutations.

## Repository Structure
- `data/`: Contains the pre-processed historical return matrices for 134 continuous S&P 500 constituents (2012–2025).
- `manuscript/`: Contains the complete LaTeX source (`paper.tex`) and all generated figures for compilation.
- `scripts/`: Contains the core execution logic:
  - `algorithms.py`: Contains the standard SOS and proposed AOBL-SOS metaheuristic classes.
  - `evaluation_v4.py`: Objective function evaluations (Sharpe, Drawdown) and Almgren-Chriss execution simulation.
  - `utils_v4.py`: Simplex normalization bounds and data loaders.
  - `run_experiments_v4.py`: The master execution script.
- `results/`: Output directory where all compiled CSV tables, Monte Carlo distribution graphs, and statistical $p$-value reports are saved.

## Replication Instructions

To replicate the entire experimental pipeline from scratch, simply execute the master script:

```bash
# 1. Install required dependencies
pip install numpy pandas scipy scikit-learn matplotlib

# 2. Execute the 10-seed stochastic optimization and validation pipeline
PYTHONPATH=. python3 -m scripts.run_experiments_v4
```

This will automatically:
1. Optimize the AOBL-SOS ecosystem across 10 random seeds.
2. Evaluate out-of-sample net returns using the Almgren-Chriss model against 4 distinct benchmarks (1/N, Ledoit-Wolf, Markowitz, Min-Var).
3. Generate the Transaction Cost Sensitivity matrices (5 bps to 25 bps).
4. Run the Jobson-Korkie statistical significance tests.
5. Perform the 1,000-path Monte Carlo bootstrap and generate the CVaR distributions and time-series plots.

Outputs will be safely written to `results/`.

## Dependencies
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `matplotlib`
