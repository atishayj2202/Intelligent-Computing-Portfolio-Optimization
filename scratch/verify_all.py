
"""Check paper.tex against frozen CSVs. Exit nonzero on mismatch."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
paper = (ROOT / "manuscript" / "paper.tex").read_text()
res = ROOT / "results"

needed = [
    "master_walk_forward_chained.csv",
    "chained_daily_returns.csv",
    "selected_weights.csv",
    "carhart_attribution.csv",
    "statistical_significance.csv",
    "block_bootstrap_delta_sharpe.csv",
    "pbo_dsr_results.csv",
    "transaction_cost_sensitivity.csv",
    "aum_capacity_curve.csv",
    "ablation_study_results.csv",
    "portfolio_stability.csv",
]
missing = [n for n in needed if not (res / n).exists()]
if missing:
    raise SystemExit("missing " + ", ".join(missing))

txts = [p for p in ROOT.rglob("*.txt") if p.name not in ("qf_submission_items.txt", "aobl_underperformance_audit.txt") and "full_pipeline.log" not in p.name and ".mplconfig" not in str(p)]
# allow ken french? none
bad_txt = [p for p in txts if "manuscript" in str(p) or "results" in str(p)]
if bad_txt:
    raise SystemExit("unexpected txt: " + ", ".join(map(str, bad_txt)))

master = pd.read_csv(res / "master_walk_forward_chained.csv")
aobl = master.loc[master["Algorithm"].astype(str).str.contains("AOBL")].iloc[0]
sharpe = f"{float(aobl['Net Sharpe']):.3f}"
if sharpe not in paper:
    raise SystemExit(f"paper missing AOBL Sharpe {sharpe}")
sos = master.loc[master["Algorithm"].astype(str)=="SOS (Baseline)"].iloc[0]
sos_s = f"{float(sos['Net Sharpe']):.3f}"
if sos_s not in paper:
    raise SystemExit(f"paper missing SOS Sharpe {sos_s}")
print("OK AOBL", sharpe, "SOS", sos_s)
print("OK required CSVs; qf extras file present:", (ROOT/"manuscript"/"qf_submission_items.txt").exists())
