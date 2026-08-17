import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.utils_v4 import set_seed, normalize_cap_cardinality, load_data
from scripts.algorithms import AOBL_SOS, SOS
from scripts.evaluation_v4 import obj_sharpe_drawdown, compute_net_metrics_fixed_bps, compute_net_metrics_almgren_chriss

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

def run_qf_experiments(output_dir="strategies/AOBL-SOS/v4/backtest_results"):
    os.makedirs(output_dir, exist_ok=True)
    
    set_seed(42)
    data_path = "data/sp500_daily_v4.csv"
        
    tickers, mu, cov, train_ret, test_ret, test_vol = load_data(data_path, n_stocks=150, seed=42)
    
    cap = 0.20
    K = 30
    rf_annual = 0.02
    rf_daily = rf_annual / 252
    aum = 100_000_000
    
    print("Optimizing Portfolios (10 Seeds for stochastic metaheuristics)...")
    dim = len(mu)
    map_func = lambda w: normalize_cap_cardinality(w, cap=cap, K=K)
    obj_dd = lambda w: obj_sharpe_drawdown(w, train_ret.values, rf_annual=rf_annual)
    
    n_seeds = 10
    aobl_dd_weights = []
    sos_dd_weights = []
    
    for seed in range(n_seeds):
        set_seed(seed)
        pop = np.random.uniform(0.0, 1.0, (50, dim))
        pop = np.array([map_func(p) for p in pop])
        
        _, w1, _ = AOBL_SOS(obj_dd, pop.copy(), map_func, iters=300, is_portfolio=True, init_obl=True, obl_mode="portfolio_reversal", cap=cap, K=K)
        aobl_dd_weights.append(w1)
        
        _, w3, _ = SOS(obj_dd, pop.copy(), map_func, iters=300, is_portfolio=True)
        sos_dd_weights.append(w3)
        
    w_minvar = min_variance_portfolio(cov, cap=cap, K=K)
    w_maxsh = max_sharpe_portfolio(mu, cov, rf_daily, cap=cap, K=K)
    w_eq = normalize_cap_cardinality(np.ones(dim), cap=cap, K=K)
    
    # ---------------------------------------------------------
    # 1. Almgren-Chriss Master Benchmark Table
    # ---------------------------------------------------------
    print("\n[1] Evaluating Master Benchmark Table (Almgren-Chriss Market Impact)...")
    
    def fmt(m, s, is_pct=False):
        if is_pct:
            return f"{m*100:.2f}% ± {s*100:.2f}%"
        return f"{m:.3f} ± {s:.3f}"

    bench_results = []
    
    # Stochastic Models (Almgren-Chriss)
    for name, w_list in [("AOBL-SOS (Proposed)", aobl_dd_weights), ("SOS (Baseline)", sos_dd_weights)]:
        results = [compute_net_metrics_almgren_chriss(w, test_ret, test_vol, train_ret, aum=aum, rf_annual=rf_annual) for w in w_list]
        m_stds = {k: (np.mean([r[k] for r in results]), np.std([r[k] for r in results])) for k in ['ann_return', 'ann_vol', 'sharpe', 'sortino', 'max_drawdown', 'calmar']}
        bench_results.append({
            "Portfolio": name,
            "Execution": "Almgren-Chriss",
            "Net Ann. Return": fmt(m_stds['ann_return'][0], m_stds['ann_return'][1], True),
            "Net Ann. Vol": fmt(m_stds['ann_vol'][0], m_stds['ann_vol'][1], True),
            "Net Sharpe": fmt(m_stds['sharpe'][0], m_stds['sharpe'][1]),
            "Net Sortino": fmt(m_stds['sortino'][0], m_stds['sortino'][1]),
            "Max Drawdown": fmt(m_stds['max_drawdown'][0], m_stds['max_drawdown'][1], True),
            "Calmar": fmt(m_stds['calmar'][0], m_stds['calmar'][1])
        })

    # Deterministic Models
    det_portfolios = {
        "Max Sharpe (Markowitz)": w_maxsh,
        "Min Variance": w_minvar,
        "Equal Weight (1/N)": w_eq
    }
    for name, w in det_portfolios.items():
        res = compute_net_metrics_almgren_chriss(w, test_ret, test_vol, train_ret, aum=aum, rf_annual=rf_annual)
        bench_results.append({
            "Portfolio": name,
            "Execution": "Almgren-Chriss",
            "Net Ann. Return": f"{res['ann_return']*100:.2f}%",
            "Net Ann. Vol": f"{res['ann_vol']*100:.2f}%",
            "Net Sharpe": f"{res['sharpe']:.3f}",
            "Net Sortino": f"{res['sortino']:.3f}",
            "Max Drawdown": f"{res['max_drawdown']*100:.2f}%",
            "Calmar": f"{res['calmar']:.3f}"
        })
        
    df_bench = pd.DataFrame(bench_results)
    df_bench.to_csv(os.path.join(output_dir, "master_benchmark_ac.csv"), index=False)
    
    # ---------------------------------------------------------
    # 2. Transaction Cost Sensitivity (Fixed BPS)
    # ---------------------------------------------------------
    print("\n[2] Transaction Cost Sensitivity Analysis (5 bps to 25 bps)...")
    sens_results = []
    
    for cost in [5, 10, 15, 20, 25]:
        row = {"Transaction Cost": f"{cost} bps"}
        
        # AOBL
        sharpes_aobl = [compute_net_metrics_fixed_bps(w, test_ret, cost_bps=cost, rf_annual=rf_annual)['sharpe'] for w in aobl_dd_weights]
        row["AOBL-SOS Sharpe"] = f"{np.mean(sharpes_aobl):.2f} ± {np.std(sharpes_aobl):.2f}"
        
        # SOS
        sharpes_sos = [compute_net_metrics_fixed_bps(w, test_ret, cost_bps=cost, rf_annual=rf_annual)['sharpe'] for w in sos_dd_weights]
        row["SOS Sharpe"] = f"{np.mean(sharpes_sos):.2f} ± {np.std(sharpes_sos):.2f}"
        
        # Markowitz
        res_mkw = compute_net_metrics_fixed_bps(w_maxsh, test_ret, cost_bps=cost, rf_annual=rf_annual)
        row["Markowitz Sharpe"] = f"{res_mkw['sharpe']:.2f}"
        
        # MinVar
        res_minvar = compute_net_metrics_fixed_bps(w_minvar, test_ret, cost_bps=cost, rf_annual=rf_annual)
        row["MinVar Sharpe"] = f"{res_minvar['sharpe']:.2f}"
        
        sens_results.append(row)
        
    df_sens = pd.DataFrame(sens_results)
    df_sens.to_csv(os.path.join(output_dir, "transaction_cost_sensitivity.csv"), index=False)
    
    # ---------------------------------------------------------
    # 3. 1000-Path Monte Carlo Block Bootstrap (VaR / CVaR)
    # ---------------------------------------------------------
    print("\n[3] 1000-Path Block Bootstrap Monte Carlo Simulation...")
    # Evaluate Seed 0 of AOBL
    w_aobl = aobl_dd_weights[0]
    res_base = compute_net_metrics_almgren_chriss(w_aobl, test_ret, test_vol, train_ret, aum=aum, rf_annual=rf_annual)
    net_rets = res_base['net_returns']
    
    n_days = len(net_rets)
    n_paths = 1000
    block_size = 10
    
    bootstrap_cvar95 = []
    np.random.seed(42)
    
    for _ in range(n_paths):
        path = []
        while len(path) < n_days:
            start_idx = np.random.randint(0, n_days - block_size)
            path.extend(net_rets[start_idx:start_idx+block_size])
        path = np.array(path[:n_days])
        
        cvar = np.percentile(path, 5)
        bootstrap_cvar95.append(cvar)
        
    mc_cvar_mean = np.mean(bootstrap_cvar95)
    mc_cvar_std = np.std(bootstrap_cvar95)
    
    with open(os.path.join(output_dir, "monte_carlo_cvar.txt"), "w") as f:
        f.write(f"1000-Path Block Bootstrap CVaR 95%: {mc_cvar_mean*100:.2f}% ± {mc_cvar_std*100:.2f}%\n")
        
    plt.figure(figsize=(8, 5))
    plt.hist(np.array(bootstrap_cvar95)*100, bins=50, color='#D85A30', alpha=0.7, edgecolor='black')
    plt.axvline(mc_cvar_mean*100, color='red', linestyle='dashed', linewidth=2, label=f'Mean CVaR: {mc_cvar_mean*100:.2f}%')
    plt.title('Monte Carlo Resampled Out-of-Sample CVaR (95%)')
    plt.xlabel('CVaR 95% (%)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "cpcv_distribution.png"), dpi=200)
    
    print("\nExperiments completed. Output files generated in:", output_dir)

if __name__ == "__main__":
    run_qf_experiments()
