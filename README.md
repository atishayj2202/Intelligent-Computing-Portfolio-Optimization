# Institutional Portfolio Construction under Cardinality Friction

**Authors:** Atishaya Jain, Karan Jangra, Vaibhav Pacherwal  
**Affiliation:** Netaji Subhas University of Technology (NSUT), Delhi  

## Overview
This repository contains the official replication suite for the manuscript: **"Institutional Portfolio Construction under Cardinality Friction: A Diversity-Aware Adaptive Metaheuristic Framework."**

The project introduces **AOBL-SOS (Adaptive Opposition-Based Learning Symbiotic Organisms Search)**, a feasibility-preserving metaheuristic designed to optimize high-dimensional institutional portfolios under strict, non-convex constraints ($K=30$ cardinality bounds and $W_{\max}=20\%$ position caps).

### Key Research Methodology Features:
- **Point-in-Time Universe Construction:** Asset selection at each rebalance date uses exclusively past historical availability up to the rebalancing timestamp, eliminating survivorship and look-ahead bias.
- **Expanding Walk-Forward Protocol:** 7 expanding walk-forward windows (2018–2025 OOS) evaluating continuous out-of-sample chained equity curves.
- **Feasibility-Preserving Encoding & Centroid-Based Opposition:** Bounded simplex solution encoding $x=(S, \theta)$ combined with adaptive population centroid reflection.
- **Almgren-Chriss Market Impact Model:** All out-of-sample execution is penalized by the non-linear Almgren-Chriss square-root execution model ($AUM = \$100\text{M}$).
- **Overfitting & Statistical Validation:** CSCV (Combinatorial Symmetric Cross-Validation), Deflated Sharpe Ratio (DSR), Probabilistic Sharpe Ratio (PSR), Minimum Backtest Length (MinBTL), and Fama-French 6-factor alpha regressions.

## Repository Structure
- `data/`: End-of-Day historical return and volume data (`sp500_daily.csv`).
- `manuscript/`: LaTeX source (`paper.tex`), references (`references.bib`), and compiled figure assets.
- `scripts/`:
  - `universe.py`: Point-in-time universe construction & walk-forward window definitions.
  - `utils.py`: Simplex normalization bounds, diversity tracking, and data loaders.
  - `algorithms.py`: Standard SOS and proposed AOBL-SOS metaheuristic implementations.
  - `metaheuristics.py`: GA, PSO, and DE metaheuristic baseline competitors.
  - `ablation.py`: 7-variant algorithmic ablation study module.
  - `evaluation.py`: Objective functions and Almgren-Chriss market impact execution simulation.
  - `pbo_cscv.py`: CSCV, PBO, DSR, PSR, and MinBTL statistical functions.
  - `analysis.py`: Fama-French factor attribution, AUM capacity curves, VIX regime splits, and portfolio stability metrics.
  - `walk_forward.py`: Expanding walk-forward orchestrator across 7 windows.
  - `run_all.py`: Master single entry-point execution script.
- `results/`: Output directory containing generated tables, distributions, and CSV outputs.

## Replication Instructions

To replicate the entire research pipeline from scratch, execute:

```bash
# 1. Install dependencies
pip install numpy pandas scipy scikit-learn matplotlib statsmodels

# 2. Run the complete research suite
PYTHONPATH=. python3 -m scripts.run_all
```

Outputs and plots are automatically written to `results/` and synced to `manuscript/`.
