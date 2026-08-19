# Institutional Portfolio Construction under Cardinality Friction

**Authors:** Atishaya Jain, Karan Jangra, Vaibhav Pacherwal  
**Affiliation:** Netaji Subhas University of Technology (NSUT), Delhi  

---

## 1. Executive Summary

This repository contains the complete, production-ready research and replication suite for the manuscript:  
**"Institutional Portfolio Construction under Cardinality Friction: A Diversity-Aware Adaptive Metaheuristic Framework."**

The project introduces **AOBL-SOS** (Adaptive Opposition-Based Learning Symbiotic Organisms Search), a feasibility-preserving optimization framework designed to construct large-scale equity portfolios under strict non-convex operational bounds:
- **Maximum Cardinality Bound:** $K=30$ active positions.
- **Maximum Asset Exposure Cap:** $W_{\max}=20\%$ weight constraint per stock.

---

## 2. Key Methodological Features

1. **Point-in-Time Universe Construction:** Eliminates survivorship and look-ahead bias by constructing asset universes dynamically at each rebalance timestamp using exclusively historical data available up to `train_end`.
2. **Expanding Walk-Forward Protocol:** Evaluates out-of-sample performance across 7 expanding walk-forward windows (2018–2024 OOS) on an actual point-in-time S&P 500 equity universe.
3. **Feasibility-Preserving Encoding & Centroid Opposition:** Bounded simplex solution mapping $x=(S, \theta)$ combined with dynamic centroid-based population opposition and stagnation boosting.
4. **Almgren-Chriss Market Impact Execution:** Deducts realistic transaction friction using the non-linear square-root Almgren-Chriss execution model ($AUM = \$100\text{M}$).
5. **Overfitting & Statistical Validation:** CSCV (Combinatorial Symmetric Cross-Validation, $S=16$), Deflated Sharpe Ratio ($\text{DSR} = 0.9522$), Probabilistic Sharpe Ratio ($\text{PSR} = 1.0000$), Implied Hurdle Sharpe ($SR_0 = 0.1951$), and Carhart 4-Factor risk attribution ($\beta_{\text{MKT}}=0.924, \beta_{\text{HML}}=0.291, \beta_{\text{MOM}}=0.638$).

---

## 3. Clean Repository Structure

```
.
├── data/
│   └── sp500_daily.csv                  # Point-in-time S&P 500 daily price & volume data
├── manuscript/
│   ├── paper.tex                         # Complete LaTeX manuscript source
│   ├── references.bib                    # Institutional BibTeX bibliography database
│   ├── *.csv                             # Generated empirical tables synced for LaTeX
│   └── *.png                             # High-resolution vector figures embedded in manuscript
├── results/
│   ├── master_walk_forward_chained.csv   # Master out-of-sample walk-forward performance metrics
│   ├── ablation_study_results.csv        # 6-variant walk-forward ablation results
│   ├── pbo_dsr_results.txt               # PBO, DSR, PSR, and Hurdle Sharpe validation output
│   ├── fama_french_attribution.txt       # Carhart 4-Factor linear regression attribution
│   └── *.png                             # Plot figures (Equity Curves, Drawdown, Diversity)
├── scratch/
│   └── verify_all.py                     # Master calculation & manuscript data verification audit
└── scripts/
    ├── __init__.py                       # Package initialization
    ├── universe.py                       # Point-in-time asset filter & walk-forward window definitions
    ├── utils.py                          # Simplex normalization, diversity tracking, and data loaders
    ├── algorithms.py                     # Standard SOS and proposed AOBL-SOS optimization solvers
    ├── metaheuristics.py                 # Baseline competitor solvers (GA, PSO, DE)
    ├── evaluation.py                     # Objective function & Almgren-Chriss execution model
    ├── ablation.py                       # 6-variant walk-forward ablation module
    ├── pbo_cscv.py                       # CSCV PBO, DSR, PSR, and MinBTL statistical functions
    ├── analysis.py                       # Carhart 4-factor attribution, AUM capacity, VIX regimes
    ├── walk_forward.py                   # Expanding walk-forward orchestrator
    └── run_all.py                        # Master single entry-point pipeline script
```

---

## 4. Replication & Verification Instructions

### Requirements
- Python 3.9+
- Dependencies: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `statsmodels`

```bash
pip install numpy pandas scipy scikit-learn matplotlib statsmodels
```

### Quick Verification & Audit
To verify that all numbers in `manuscript/paper.tex`, result CSVs, and statistical files match with **0 discrepancies**:

```bash
PYTHONPATH=. python3 scratch/verify_all.py
```

### Full Pipeline Execution
To re-run the entire walk-forward optimization, ablation study, CSCV overfitting analysis, and factor attribution from scratch:

```bash
PYTHONPATH=. python3 -m scripts.run_all
```

All results and figures will automatically update in `results/` and sync to `manuscript/`.
