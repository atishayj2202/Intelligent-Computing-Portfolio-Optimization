import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

import scipy.stats as stats
from scripts.universe import get_walk_forward_windows
from scripts.utils import set_seed, normalize_cap_cardinality, load_data_for_window, population_diversity
from scripts.algorithms import AOBL_SOS, SOS
from scripts.metaheuristics import run_ga, run_pso, run_de
from scripts.evaluation import obj_sharpe_drawdown, compute_net_metrics_almgren_chriss

def jobson_korkie_memmel(ret1, ret2):
    """
    Jobson-Korkie test with Memmel's (2003) correction for Sharpe ratio difference significance.
    """
    T = len(ret1)
    mu1, mu2 = np.mean(ret1), np.mean(ret2)
    var1, var2 = np.var(ret1, ddof=1), np.var(ret2, ddof=1)
    covar = np.cov(ret1, ret2)[0, 1]
    
    sh1 = mu1 / np.sqrt(var1 + 1e-12)
    sh2 = mu2 / np.sqrt(var2 + 1e-12)
    
    rho = covar / np.sqrt(var1 * var2 + 1e-12)
    rho = np.clip(rho, -0.9999, 0.9999)
    
    # Memmel (2003) asymptotic variance formula:
    var_diff = (1.0 / T) * (2.0 * (1.0 - rho) + 0.5 * (sh1**2 + sh2**2 - 2.0 * sh1 * sh2 * (rho**2)))
    var_diff = max(1e-12, var_diff)
    
    z = (sh1 - sh2) / np.sqrt(var_diff)
    pval = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return sh1, sh2, z, pval

def min_variance_portfolio(cov, cap=0.20, K=30):
    n = cov.shape[0]
    diag_vars = np.diag(cov)
    K_eff = min(K, n)
    top_k = np.argsort(diag_vars)[:K_eff]
    cov_sub = cov[np.ix_(top_k, top_k)]
    
    def obj(w_sub):
        return float(np.dot(w_sub.T, np.dot(cov_sub, w_sub)))
        
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = [(0, cap) for _ in range(K_eff)]
    init_w = np.ones(K_eff) / K_eff
    res = minimize(obj, init_w, method='SLSQP', bounds=bounds, constraints=cons)
    
    w_full = np.zeros(n)
    w_full[top_k] = res.x
    return normalize_cap_cardinality(w_full, cap, K)

def max_sharpe_portfolio(mu, cov, rf_daily, cap=0.20, K=30):
    n = len(mu)
    vols = np.sqrt(np.maximum(1e-12, np.diag(cov)))
    sharpes = (mu - rf_daily * 252) / vols
    K_eff = min(K, n)
    top_k = np.argsort(sharpes)[-K_eff:]
    
    mu_sub = mu[top_k]
    cov_sub = cov[np.ix_(top_k, top_k)]
    
    def obj(w_sub):
        ret = np.dot(w_sub, mu_sub)
        vol = np.sqrt(max(1e-12, np.dot(w_sub.T, np.dot(cov_sub, w_sub))))
        return -float((ret - rf_daily * 252) / (vol + 1e-12))
        
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = [(0, cap) for _ in range(K_eff)]
    init_w = np.ones(K_eff) / K_eff
    res = minimize(obj, init_w, method='SLSQP', bounds=bounds, constraints=cons)
    
    w_full = np.zeros(n)
    w_full[top_k] = res.x
    return normalize_cap_cardinality(w_full, cap, K)

def ledoit_wolf_portfolio(mu, train_ret, rf_daily, cap=0.20, K=30):
    lw = LedoitWolf().fit(train_ret)
    cov_lw = lw.covariance_
    return max_sharpe_portfolio(mu, cov_lw, rf_daily, cap, K)

def run_expanding_walk_forward(data_path="data/sp500_daily.csv", n_seeds=5, iters=200, output_dir="results"):
    """
    Runs Expanding Walk-Forward Optimization across 7 windows (2018 to 2024 OOS).
    Uses point-in-time universe construction per window to eliminate survivorship bias.
    Chains OOS return streams to evaluate continuous 2018-2025 OOS performance.
    """
    os.makedirs(output_dir, exist_ok=True)
    windows = get_walk_forward_windows()
    
    cap = 0.20
    K = 30
    rf_annual = 0.02
    rf_daily = rf_annual / 252.0
    aum = 100_000_000
    
    algorithm_names = [
        "AOBL-SOS (Proposed)",
        "SOS (Baseline)",
        "GA",
        "PSO",
        "DE",
        "Ledoit-Wolf",
        "Max Sharpe",
        "Min Variance",
        "Equal Weight (1/N)"
    ]
    
    window_results = []
    chained_returns = {alg: [] for alg in algorithm_names}
    
    for win in windows:
        print(f"\n--- Walk-Forward Window {win['window_id']}: Train ({win['train_start']} to {win['train_end']}) -> Test ({win['test_year']}) ---")
        
        tickers, mu, cov, train_ret, test_ret, test_vol = load_data_for_window(
            data_path, win['train_start'], win['train_end'], win['test_start'], win['test_end'], cap=cap, K=K
        )
        
        dim = len(tickers)
        map_func = lambda w: normalize_cap_cardinality(w, cap=cap, K=K)
        obj_dd = lambda w: obj_sharpe_drawdown(w, train_ret.values, rf_annual=rf_annual)
        
        # Optimize across n_seeds
        algo_weights = {alg: [] for alg in algorithm_names}
        
        for seed in range(n_seeds):
            set_seed(seed * 100 + win['window_id'])
            pop = np.random.uniform(0.0, 1.0, (40, dim))
            pop = np.array([map_func(p) for p in pop])
            
            # AOBL-SOS
            _, w_aobl, _ = AOBL_SOS(obj_dd, pop.copy(), map_func, iters=iters, is_portfolio=True, cap=cap, K=K)
            algo_weights["AOBL-SOS (Proposed)"].append(w_aobl)
            
            # SOS
            _, w_sos, _ = SOS(obj_dd, pop.copy(), map_func, iters=iters, is_portfolio=True)
            algo_weights["SOS (Baseline)"].append(w_sos)
            
            # GA
            _, w_ga, _ = run_ga(obj_dd, pop.copy(), map_func, iters=iters)
            algo_weights["GA"].append(w_ga)
            
            # PSO
            _, w_pso, _ = run_pso(obj_dd, pop.copy(), map_func, iters=iters)
            algo_weights["PSO"].append(w_pso)
            
            # DE
            _, w_de, _ = run_de(obj_dd, pop.copy(), map_func, iters=iters)
            algo_weights["DE"].append(w_de)
            
        # Analytical / Deterministic Portfolios
        w_lw = ledoit_wolf_portfolio(mu, train_ret.values, rf_daily, cap=cap, K=K)
        w_maxsh = max_sharpe_portfolio(mu, cov, rf_daily, cap=cap, K=K)
        w_minvar = min_variance_portfolio(cov, cap=cap, K=K)
        w_eq = np.ones(dim) / float(dim) # Canonical DeMiguel et al. (2009) 1/N benchmark
        
        algo_weights["Ledoit-Wolf"] = [w_lw]
        algo_weights["Max Sharpe"] = [w_maxsh]
        algo_weights["Min Variance"] = [w_minvar]
        algo_weights["Equal Weight (1/N)"] = [w_eq]
        
        # Evaluate OOS metrics for each algorithm in this window
        for alg in algorithm_names:
            weights_list = algo_weights[alg]
            metrics_list = [compute_net_metrics_almgren_chriss(w, test_ret, test_vol, train_ret, aum=aum, rf_annual=rf_annual) for w in weights_list]
            
            # Median seed selection for continuous return chaining
            median_idx = int(np.argsort([m['sharpe'] for m in metrics_list])[len(metrics_list)//2])
            best_metrics = metrics_list[median_idx]
            
            chained_returns[alg].extend(best_metrics['net_returns'])
            
            window_results.append({
                'Window': win['window_id'],
                'Test Year': win['test_year'],
                'Algorithm': alg,
                'Net Sharpe': f"{np.median([m['sharpe'] for m in metrics_list]):.3f}",
                'Net Ann Return': f"{np.median([m['ann_return'] for m in metrics_list])*100:.2f}%",
                'Max Drawdown': f"{np.median([m['max_drawdown'] for m in metrics_list])*100:.2f}%"
            })
            
    df_wf = pd.DataFrame(window_results)
    df_wf.to_csv(os.path.join(output_dir, "walk_forward_per_window.csv"), index=False)
    
    # Evaluate aggregate continuous 2018-2025 OOS metrics across full walk-forward horizon
    agg_summary = []
    for alg in algorithm_names:
        rets = np.array(chained_returns[alg])
        total_growth = np.prod(1.0 + rets)
        n_days_total = len(rets)
        cagr = (total_growth ** (252.0 / n_days_total)) - 1.0 if total_growth > 0 else -1.0
        ann_r = cagr - rf_annual
        ann_v = np.std(rets, ddof=1) * np.sqrt(252) + 1e-12
        sh = ann_r / ann_v
        
        cum_ret = np.cumprod(1 + rets)
        running_max = np.maximum.accumulate(cum_ret)
        drawdowns = (cum_ret - running_max) / running_max
        max_dd = np.min(drawdowns)
        calmar = ann_r / abs(max_dd) if max_dd < 0 else np.nan
        
        agg_summary.append({
            'Algorithm': alg,
            'Walk-Forward Horizon': '2018-2025 (Chained OOS)',
            'Net Sharpe': f"{sh:.3f}",
            'Net Ann Return': f"{ann_r*100:.2f}%",
            'Net Ann Vol': f"{ann_v*100:.2f}%",
            'Max Drawdown': f"{max_dd*100:.2f}%",
            'Calmar Ratio': f"{calmar:.3f}"
        })
        
    df_agg = pd.DataFrame(agg_summary)
    df_agg.to_csv(os.path.join(output_dir, "master_walk_forward_chained.csv"), index=False)
    
    return df_wf, df_agg, chained_returns
