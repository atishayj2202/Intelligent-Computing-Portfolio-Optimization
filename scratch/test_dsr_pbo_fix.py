import numpy as np
import pandas as pd
import scipy.stats as stats
from itertools import combinations

from scripts.utils import load_data_for_window, set_seed, normalize_cap_cardinality
from scripts.evaluation import compute_net_metrics_almgren_chriss, obj_sharpe_drawdown
from scripts.algorithms import AOBL_SOS
from scripts.pbo_cscv import compute_pbo_cscv

def test_dsr_pbo():
    data_path = "data/sp500_daily.csv"
    tickers, mu, cov, train_ret, test_ret, test_vol = load_data_for_window(
        data_path, '2012-01-01', '2017-12-31', '2018-01-01', '2024-12-31'
    )
    dim = len(tickers)
    
    # 25-strategy grid
    lambda_grid = [0.0, 0.5, 1.0, 1.5, 2.0]
    k_grid = [15, 20, 25, 30, 35]
    
    grid_returns_list = []
    grid_sharpes = []
    
    for l_val in lambda_grid:
        for k_val in k_grid:
            map_func = lambda w: normalize_cap_cardinality(w, cap=0.20, K=k_val)
            obj_func = lambda w: obj_sharpe_drawdown(w, train_ret.values, lambda_dd=l_val, rf_annual=0.02)
            
            set_seed(42 + int(l_val * 10) + k_val)
            pop = np.random.uniform(0, 1, (40, dim))
            pop = np.array([map_func(p) for p in pop])
            
            _, w_cand, _ = AOBL_SOS(obj_func, pop, map_func, iters=100, is_portfolio=True, cap=0.20, K=k_val)
            res = compute_net_metrics_almgren_chriss(w_cand, test_ret, test_vol, train_ret, aum=100_000_000)
            
            grid_returns_list.append(res['net_returns'])
            grid_sharpes.append(res['sharpe'])
            print(f"Grid (lambda={l_val}, K={k_val}): Net Sharpe = {res['sharpe']:.3f}")
            
    grid_matrix = np.column_stack(grid_returns_list)
    
    # Compute CSCV PBO
    pbo, logits = compute_pbo_cscv(grid_matrix, S=16)
    print(f"\n--- CSCV PBO Result ---")
    print(f"PBO = {pbo:.4f} ({pbo*100:.2f}%)")
    
    # Compute Cross-Strategy Sharpe Variance per López de Prado (2014)
    sr_var_cross_strategy = float(np.var(grid_sharpes, ddof=1))
    print(f"\n--- Cross-Strategy Sharpe Distribution ---")
    print(f"Grid Sharpes: {grid_sharpes}")
    print(f"Mean Grid Sharpe: {np.mean(grid_sharpes):.4f}")
    print(f"Std Dev of Grid Sharpes: {np.std(grid_sharpes, ddof=1):.4f}")
    print(f"Variance of Grid Sharpes (sr_var): {sr_var_cross_strategy:.6f}")
    
    # López de Prado (2014) DSR calculation
    n_trials = 25
    n_samples = len(test_ret)
    realized_sr = 0.841
    
    euler_gamma = 0.5772156649
    e_max_sr = (1 - euler_gamma) * stats.norm.ppf(1 - 1.0 / n_trials) + euler_gamma * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    benchmark_sr_ann = np.sqrt(sr_var_cross_strategy) * e_max_sr
    
    sr_daily = realized_sr / np.sqrt(252.0)
    bench_daily = benchmark_sr_ann / np.sqrt(252.0)
    
    skew = float(stats.skew(grid_returns_list[0]))
    kurt = float(stats.kurtosis(grid_returns_list[0], fisher=False))
    
    sr_std_daily = np.sqrt((1.0 + 0.5 * sr_daily**2 - skew * sr_daily + (kurt - 3.0) / 4.0 * sr_daily**2) / (n_samples - 1.0))
    z = (sr_daily - bench_daily) / sr_std_daily
    dsr = float(stats.norm.cdf(z))
    
    print(f"\n--- López de Prado (2014) DSR Results ---")
    print(f"Expected Max Sharpe (e_max_sr): {e_max_sr:.4f}")
    print(f"Implied Hurdle Sharpe (SR0): {benchmark_sr_ann:.4f}")
    print(f"Realized OOS Sharpe: {realized_sr:.4f}")
    print(f"z-score: {z:.4f}")
    print(f"Deflated Sharpe Ratio (DSR): {dsr:.4f} ({dsr*100:.2f}%)")

if __name__ == "__main__":
    test_dsr_pbo()
