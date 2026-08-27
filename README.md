# Constrained portfolio construction (Quantitative Finance manuscript)

Replication package for the QF submission. Adaptive diversity preservation is the *construction engine*, not a new general-purpose optimizer (that claim is reserved for a separate computational-optimization paper).

## What is honest in this freeze
- Expanding PIT walk-forward, 2018--2024 OOS.
- Metaheuristic seeds: **median OOS rank** among five stochastic runs per window (chained path uses that seed).
- Headline costs: **10 bps** at annual formation only; holdings drift intra-year.
- Almgren--Chriss is used only to replay **locked** weights across AUM.
- Carhart factors: Ken French daily MKT, SMB, HML, MOM.
- Block bootstrap of ΔSharpe vs SOS.
- PBO grid uses 100 iterations; headline uses 150. Both are disclosed.

Headline numbers live in `results/master_walk_forward_chained.csv` after `run_all`.

## Run
```bash
pip install numpy pandas scipy scikit-learn matplotlib statsmodels
PYTHONPATH=. python3 -m scripts.run_all
PYTHONPATH=. python3 scratch/verify_all.py
```

Ken French daily zips are downloaded into `data/` on first factor run.

## QF extras
`manuscript/qf_submission_items.txt` (keywords, highlights, declarations). No other `.txt` result dumps.

## Data
`data/sp500_daily.csv` plus Ken French library files written by `scripts/diagnostics.py`.
