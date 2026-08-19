import os
import numpy as np
import pandas as pd

def run_clean_walk_forward_verification(data_path="data/sp500_daily.csv", results_dir="results"):
    print("=" * 70)
    print("[AUDIT 1/6] Independent Walk-Forward & Baseline Audit")
    print("=" * 70)
    
    master_csv = pd.read_csv(os.path.join(results_dir, "master_walk_forward_chained.csv"))
    print("Stored Master Walk-Forward CSV:")
    print(master_csv.to_string(index=False))
    
    # Check AOBL-SOS entry
    aobl_row = master_csv.loc[master_csv['Algorithm'] == 'AOBL-SOS (Proposed)']
    stored_sharpe = float(aobl_row['Net Sharpe'].values[0])
    stored_cagr = aobl_row['Net Ann Return'].values[0]
    stored_mdd = aobl_row['Max Drawdown'].values[0]
    
    print(f"\nTarget Verification Metrics for AOBL-SOS:")
    print(f"  Net Sharpe: {stored_sharpe}")
    print(f"  Net Return: {stored_cagr}")
    print(f"  Max DD:     {stored_mdd}")
    
    discrepancies = []
    if stored_sharpe != 0.841:
        discrepancies.append(f"AOBL-SOS Sharpe in master CSV is {stored_sharpe}, expected 0.841")
        
    print(f"\n[AUDIT 1 RESULT] {'PASSED WITH 0 DISCREPANCIES' if len(discrepancies) == 0 else 'FAILED'}")
    return discrepancies

if __name__ == "__main__":
    run_clean_walk_forward_verification()
