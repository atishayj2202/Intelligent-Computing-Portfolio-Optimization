import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.utils import set_seed, normalize_cap_cardinality, load_data
from scripts.algorithms import AOBL_SOS, SOS
from scripts.evaluation import obj_sharpe_drawdown, compute_net_metrics

def min_variance_portfolio(cov, cap=0.20, K=30):
    n = cov.shape[0]
    def obj(w):
        w_card = normalize_cap_cardinality(w, cap, K)
        return np.dot(w_card.T, np.dot(cov, w_card))
    bounds = [(0, cap) for _ in range(n)]
    init_w = np.ones(n) / n
    res = minimize(obj, init_w, method='L-BFGS-B', bounds=bounds)
    return normalize_cap_cardinality(res.x, cap, K)

def max_sharpe_portfolio(mu, cov, rf_daily, cap=0.20, K=30):
    n = len(mu)
    def obj(w):
        w_card = normalize_cap_cardinality(w, cap, K)
        ret = np.dot(w_card, mu)
        vol = np.sqrt(np.dot(w_card.T, np.dot(cov, w_card)))
        return -(ret - rf_daily) / (vol + 1e-12)
    bounds = [(0, cap) for _ in range(n)]
    init_w = np.ones(n) / n
    res = minimize(obj, init_w, method='L-BFGS-B', bounds=bounds)
    return normalize_cap_cardinality(res.x, cap, K)

def inverse_volatility_portfolio(cov, cap=0.20, K=30):
    vols = np.sqrt(np.diag(cov))
    inv_vols = 1.0 / (vols + 1e-12)
    return normalize_cap_cardinality(inv_vols, cap, K)

def run_jpm_experiments(output_dir="jpm/final_submission/results"):
    os.makedirs(output_dir, exist_ok=True)
    
    set_seed(42)
    data_path = "jpm/final_submission/data/sp500_daily.csv"
    if not os.path.exists(data_path):
        data_path = "../../data/sp500_daily.csv"
        
    tickers, mu, cov, train_ret, test_ret = load_data(data_path, n_stocks=179, seed=42)
    
    cap = 0.20
    K = 30
    rf_annual = 0.02
    rf_daily = rf_annual / 252
    
    print("Optimizing Portfolios...")
    dim = len(mu)
    
    map_func = lambda w: normalize_cap_cardinality(w, cap=cap, K=K)
    obj = lambda w: obj_sharpe_drawdown(w, train_ret.values, rf_annual=rf_annual)
    
    pop = np.random.uniform(0.0, 1.0, (50, dim))
    pop = np.array([map_func(p) for p in pop])
    
    print("  Running AOBL-SOS...")
    _, w_aobl, _ = AOBL_SOS(obj, pop.copy(), map_func, iters=300, is_portfolio=True, 
                            init_obl=True, obl_mode="portfolio_reversal", cap=cap, K=K)
    
    print("  Running Standard SOS...")
    _, w_sos, _ = SOS(obj, pop.copy(), map_func, iters=300, is_portfolio=True)
    
    print("  Computing Min Variance...")
    w_minvar = min_variance_portfolio(cov, cap=cap, K=K)
    
    print("  Computing Max Sharpe (Markowitz)...")
    w_maxsh = max_sharpe_portfolio(mu, cov, rf_daily, cap=cap, K=K)
    
    print("  Computing Risk Parity (Inv-Vol)...")
    w_invvol = inverse_volatility_portfolio(cov, cap=cap, K=K)
    
    w_eq = normalize_cap_cardinality(np.ones(dim), cap=cap, K=K)
    
    portfolios = {
        "AOBL-SOS (Proposed)": w_aobl,
        "Max Sharpe (Markowitz)": w_maxsh,
        "SOS (Baseline)": w_sos,
        "Risk Parity (Inv-Vol)": w_invvol,
        "Min Variance": w_minvar,
        "Equal Weight (1/N)": w_eq
    }
    
    print("\n[1] Evaluating Master Benchmark Table (10 bps Cost, Monthly Rebal)...")
    bench_results = []
    for name, w in portfolios.items():
        res = compute_net_metrics(w, test_ret, cost_bps=10, rebal_freq=21, rf_annual=rf_annual)
        bench_results.append({
            "Portfolio": name,
            "Cost (bps)": 10,
            "Net Ann. Return": f"{res['ann_return']*100:.2f}%",
            "Net Ann. Vol": f"{res['ann_vol']*100:.2f}%",
            "Net Sharpe": f"{res['sharpe']:.3f}",
            "Net Sortino": f"{res['sortino']:.3f}",
            "Max Drawdown": f"{res['max_drawdown']*100:.2f}%",
            "Monthly Turnover": f"{res['avg_turnover']*100:.2f}%"
        })
        
    df_bench = pd.DataFrame(bench_results)
    df_bench.to_csv(os.path.join(output_dir, "master_benchmark_table.csv"), index=False)
    
    print("\n[2] Transaction Cost Sensitivity Analysis (0, 5, 10, 15 bps)...")
    cost_levels = [0, 5, 10, 15]
    tx_results = []
    for name, w in portfolios.items():
        for cost in cost_levels:
            res = compute_net_metrics(w, test_ret, cost_bps=cost, rebal_freq=21, rf_annual=rf_annual)
            tx_results.append({
                "Portfolio": name,
                "Cost_bps": cost,
                "Net_Sharpe": round(res["sharpe"], 3)
            })
            
    df_tx = pd.DataFrame(tx_results)
    df_tx.to_csv(os.path.join(output_dir, "transaction_cost_sensitivity.csv"), index=False)
    
    print("\n[3] Rebalancing Frequency Analysis...")
    rebal_results = []
    freq_map = {"Monthly (21d)": 21, "Quarterly (63d)": 63, "Semi-Annual (126d)": 126}
    
    for freq_name, freq_days in freq_map.items():
        for name, w in portfolios.items():
            res = compute_net_metrics(w, test_ret, cost_bps=10, rebal_freq=freq_days, rf_annual=rf_annual)
            rebal_results.append({
                "Rebalance_Freq": freq_name,
                "Portfolio": name,
                "Net_Sharpe": round(res["sharpe"], 3)
            })
            
    df_rebal = pd.DataFrame(rebal_results)
    df_rebal.to_csv(os.path.join(output_dir, "rebalancing_frequency.csv"), index=False)

    print("\n[4] Generating Cumulative Return Plot...")
    plt.figure(figsize=(12, 6))
    colors = ['#D85A30', '#9C27B0', '#378ADD', '#009688', '#FF9800', '#6D4C41']
    for (name, w), color in zip(portfolios.items(), colors):
        res = compute_net_metrics(w, test_ret, cost_bps=10, rebal_freq=21, rf_annual=rf_annual)
        lw = 2.5 if "AOBL" in name else 1.8
        plt.plot(res["cum_returns"], color=color, lw=lw, label=name)
        
    plt.title('Out-of-Sample Cumulative Net Return (10 bps Transaction Cost)', fontsize=13, fontweight='bold')
    plt.xlabel('Trading Days (2023-2025)')
    plt.ylabel('Net Portfolio Growth (Base = 1.0)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cumulative_net_returns.png"), dpi=200)
    plt.close()
    
    print("Experiments completed. Output files generated in:", output_dir)

if __name__ == "__main__":
    run_jpm_experiments()
