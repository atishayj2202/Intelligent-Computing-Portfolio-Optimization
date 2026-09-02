# Institutional Portfolio Optimization under Cardinality and Execution Frictions: A Walk-Forward Evaluation of Adaptive Diversity Preservation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Replication: 100% Deterministic](https://img.shields.io/badge/Replication-100%25%20Deterministic-success.svg)](#replication)

This repository contains the complete replication code, empirical data panel, and results for the quantitative finance manuscript:
> **"Institutional Portfolio Optimization under Cardinality and Execution Frictions: A Walk-Forward Evaluation of Adaptive Diversity Preservation"**

---

## 1. Research Overview & Problem Formulation

Institutional equity allocation is governed by strict operational mandates that create an NP-hard Mixed-Integer Nonlinear Programming (MINLP) problem:
*   **Exact Cardinality Constraint**: $\|\mathbf{w}\|_0 = K = 30$ (limiting asset-monitoring and trading overhead).
*   **Position Concentration Cap**: $w_i \le W_{\max} = 20\%$ (mitigating idiosyncratic risk).
*   **Path-Dependent Downside Loss**: $g(\mathbf{w}) = -\text{SR}(\mathbf{w}) + \lambda \cdot |\text{MDD}(\mathbf{w})|$ with $\lambda = 1.0$.

Standard quadratic programming solvers (e.g., SLSQP) cannot natively enforce discrete cardinality bounds, while standard population metaheuristics rapidly suffer from premature diversity collapse along dominant covariance directions.

### Core Methodological Contributions:
1.  **Exact Feasibility-Preserving Projection ($\mathcal{M}$)**: A two-stage continuous-to-discrete mapping from $\mathbb{R}^n \to \Delta^{n-1}(W_{\max})$ that guarantees strict feasibility on the capped cardinality simplex without post-hoc heuristic repair.
2.  **Stagnation-Conditioned Centroid Opposition**: A diversity-preservation restart operator triggered when search stagnates ($S_t \ge 15$) or periodically (every 20 iterations), reflecting the worst 50% of the population across the population centroid with stagnation-scaled intensity $\gamma = 1 + 0.05 \cdot \max(0, S_t - \tau + 1)$.
3.  **Strictly Out-of-Sample Walk-Forward Evaluation**: 7-window expanding walk-forward horizon (2018–2024) with point-in-time universe construction, annually locked targets, and simple-return drift.
4.  **Microstructure Capacity Scaling**: Execution costs evaluated under 10~bps annual formation friction and nonlinear **Almgren–Chriss square-root market impact** scaled from $\$10\text{M}$ to $\$5\text{B}$ AUM.
5.  **Econometric Overfitting Safeguards**: Quantified via Combinatorial Symmetric Cross-Validation ($\text{PBO} = 0.427$), Deflated Sharpe Ratio ($\text{DSR} = 0.956$), stationary block-bootstrap confidence intervals, and Ken French / Carhart 4-factor risk attribution ($R^2 = 0.831$).

---

## 2. Key Empirical Findings (2018–2024 Chained OOS)

Chained out-of-sample performance across all 9 competing allocation algorithms evaluated under identical point-in-time universes, 5-seed median walk-forward selection, and 10~bps annual formation friction:

| Portfolio Strategy | Net Sharpe | Net Ann. Return (CAGR) | Net Ann. Vol. | Max Drawdown | Calmar Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **AOBL-SOS (Adaptive Engine)** | **0.919** | **21.55%** | 21.28% | **-31.75%** | **0.616** |
| Particle Swarm Optimization (PSO) | 0.859 | 20.64% | 21.70% | -33.62% | 0.555 |
| Genetic Algorithm (GA) | 0.809 | 19.28% | 21.36% | -32.45% | 0.533 |
| Symbiotic Organisms Search (SOS Baseline) | 0.792 | 19.06% | 21.53% | -36.05% | 0.473 |
| Ledoit–Wolf Shrinkage (2-Stage SLSQP) | 0.713 | 17.42% | 21.63% | -31.23% | 0.494 |
| Max Sharpe (Markowitz 2-Stage SLSQP) | 0.710 | 17.34% | 21.62% | -31.18% | 0.492 |
| Equal Weight ($1/N$ Full-Universe) | 0.700 | 15.52% | 19.33% | -34.89% | 0.388 |
| Differential Evolution (DE) | 0.661 | 16.01% | 21.20% | -33.27% | 0.421 |
| Minimum Variance (2-Stage SLSQP) | 0.480 | 9.72% | 16.07% | -31.88% | 0.242 |

*Data Source: `results/master_walk_forward_chained.csv`*

---

## 3. Institutional Scalability & Risk Diagnostics

*   **Almgren–Chriss Capacity**: Net Sharpe remains at **0.923 at $\$100\text{M}$** and **0.917 at $\$5\text{B}$ AUM**, with implementation shortfall vs. the $\$10\text{M}$ baseline of only **0.15 percentage points** at $\$5\text{B}$ (`results/aum_capacity_curve.csv`).
*   **Carhart 4-Factor Attribution**: Explains systematic participation ($R^2 = 0.831, N=1{,}761$) with market beta $\beta_{\text{MKT}} = 0.946$, momentum exposure $\beta_{\text{MOM}} = 0.058$, and a growth tilt ($\beta_{\text{HML}} = -0.213$). Annualized conditional residual alpha is $\alpha = 6.53\%$ ($t=2.01, p=0.045$).
*   **Backtest Overfitting Control**: Combinatorial Symmetric Cross-Validation yields $\text{PBO} = 0.427$ and $\text{DSR} = 0.956$ across a 25-configuration parameter grid (`results/pbo_dsr_results.csv`), verifying moderate stability against selection bias.

---

## 4. Repository Structure

```
.
├── data/
│   └── sp500_daily.csv                 # Point-in-time daily panel of simple returns & dollar volumes (2012–2024)
├── manuscript/
│   ├── paper.tex                       # Complete LaTeX manuscript
│   ├── references.bib                  # BibTeX bibliography
│   └── *.png                           # High-resolution vector/raster figures
├── results/
│   ├── master_walk_forward_chained.csv # Primary 2018–2024 chained performance table
│   ├── ablation_study_results.csv      # 6-variant component attribution
│   ├── aum_capacity_curve.csv          # Almgren-Chriss shortfall across $10M–$5B AUM
│   ├── carhart_attribution.csv         # 4-factor risk decomposition
│   ├── pbo_dsr_results.csv             # CSCV PBO and DSR statistics
│   ├── transaction_cost_sensitivity.csv# Formation cost sweep (0 to 50 bps)
│   └── vix_regime_analysis.csv         # Backward-looking volatility regime decomposition
└── scripts/
    ├── algorithms.py                   # AOBL_SOS, SOS, mutualism, commensalism, parasitism, centroid opposition
    ├── metaheuristics.py               # GA, PSO, and DE baseline implementations
    ├── universe.py                     # Point-in-time expanding walk-forward window manager
    ├── utils.py                        # Feasibility mapping M, position capping, and data loaders
    ├── evaluation.py                   # Loss function, Sharpe, CAGR, MDD, Almgren-Chriss slippage
    ├── walk_forward.py                 # Walk-forward optimization and multi-seed chaining engine
    ├── ablation.py                     # 6-variant ablation experiment runner
    ├── pbo_cscv.py                     # Combinatorial Symmetric Cross-Validation (12,870 splits)
    ├── analysis.py                     # AUM capacity, transaction costs, and regime analysis
    ├── diagnostics.py                  # Ken French data fetcher and Carhart HAC regression
    └── run_all.py                      # Master pipeline executing Steps 1 through 7 end-to-end
```

---

## 5. Replication Instructions

### Prerequisites
*   Python 3.10+
*   Standard scientific stack: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `statsmodels`

```bash
# 1. Clone repository
git clone https://github.com/atishayj2202/JPM-Quantitative-Portfolio-Optimization.git
cd JPM-Quantitative-Portfolio-Optimization

# 2. Install dependencies
pip install numpy pandas scipy scikit-learn matplotlib statsmodels requests

# 3. Execute full empirical pipeline
PYTHONPATH=. python3 -m scripts.run_all
```

The pipeline executes sequentially:
1.  **Step 1**: Expanding Walk-Forward Optimization (7 windows $\times$ 9 models $\times$ 5 seeds).
2.  **Step 2**: Component Ablation Study across Variants A–F.
3.  **Step 3**: CSCV PBO and Deflated Sharpe Ratio calculation ($S=16$ blocks, 12,870 combinatorial partitions).
4.  **Step 4**: Almgren–Chriss AUM capacity curve replay ($\$10\text{M}$ to $\$5\text{B}$).
5.  **Step 5**: Proportional formation-cost sensitivity analysis (0 to 50 bps).
6.  **Step 6**: Ken French Carhart four-factor regression with Newey–West HAC standard errors.
7.  **Step 7**: Figure synchronization with `manuscript/`.

---

## 6. Citation

If you use this codebase or methodology in academic research, please cite:

```bibtex
@article{jain2026institutional,
  title={Institutional Portfolio Optimization under Cardinality and Execution Frictions: A Walk-Forward Evaluation of Adaptive Diversity Preservation},
  author={Jain, Atishaya and Jangra, Karan and Pacherwal, Vaibhav and Vats, Rishabh and Kumar, Rajeev},
  journal={Working Paper / Submission Draft},
  year={2026}
}
```

---

## 7. License
This project is licensed under the MIT License.
