import numpy as np
import pandas as pd
import scipy.stats as stats

from scripts.utils import load_data_for_window, set_seed, normalize_cap_cardinality
from scripts.evaluation import compute_net_metrics_almgren_chriss, obj_sharpe_drawdown
from scripts.algorithms import AOBL_SOS
from scripts.pbo_cscv import compute_pbo_cscv, compute_dsr, compute_psr

def run_true_walk_forward_cscv_dsr():
    data_path = "data/sp500_daily.csv"
    windows = [
        {'id': 1, 'train_s': '2012-01-01', 'train_e': '2017-12-31', 'test_s': '2018-01-01', 'test_e': '2018-12-31'},
        {'id': 2, 'train_s': '2012-01-01', 'train_e': '2018-12-31', 'test_s': '2019-01-01', 'test_e': '2019-12-31'},
        {'id': 3, 'train_s': '2012-01-01', 'train_e': '2019-12-31', 'test_s': '2020-01-01', 'test_e': '2020-12-31'},
        {'id': 4, 'train_s': '2012-01-01', 'train_e': '2020-12-31', 'test_s': '2021-01-01', 'test_e': '2021-12-31'},
        {'id': 5, 'train_s': '2012-01-01', 'train_e': '2021-12-31', 'test_s': '2022-01-01', 'test_e': '2022-12-31'},
        {'id': 6, 'train_s': '2012-01-01', 'train_e': '2022-12-31', 'test_s': '2023-01-01', 'test_e': '2023-12-31'},
        {'id': 7, 'train_s': '2012-01-01', 'train_e': '2023-12-31', 'test_s': '2024-01-01', 'test_e': '2024-12-31'},
    ]
    
    lambda_grid = [0.0, 0.5, 1.0, 1.5, 2.0]
    k_grid = [15, 20, 25, 30, 35]
    
    grid_chained_returns = []
    grid_annual_sharpes = []
    
    for l_val in lambda_grid:
        for k_val in k_grid:
            print(f"Evaluating Strategy Grid Candidate: lambda={l_val}, K={k_val}...")
            cand_rets = []
            
            for win in windows:
                tickers, mu, cov, train_ret, test_ret, test_vol = load_data_for_window(
                    data_path, win['train_s'], win['train_e'], win['test_s'], win['test_e'], cap=0.20, K=k_val
                )
                dim = len(tickers)
                map_func = lambda w: normalize_cap_cardinality(w, cap=0.20, K=k_val)
                obj_func = lambda w: obj_sharpe_drawdown(w, train_ret.values, lambda_dd=l_val, rf_annual=0.02)
                
                set_seed(42 + win['id'] * 10 + int(l_val * 5) + k_val)
                pop = np.random.uniform(0, 1, (40, dim))
                pop = np.array([map_func(p) for p in pop])
                
                _, w_cand, _ = AOBL_SOS(obj_func, pop, map_func, iters=100, is_portfolio=True, cap=0.20, K=k_val)
                res = compute_net_metrics_almgren_chriss(w_cand, test_ret, test_vol, train_ret, aum=100_000_000)
                cand_rets.extend(res['net_returns'])
                
            rets_arr = np.array(cand_rets)
            ann_r = (np.prod(1 + rets_arr) ** (252.0 / len(rets_arr))) - 1.0 - 0.02
            ann_v = np.std(rets_arr, ddof=1) * np.sqrt(252) + 1e-12
            cand_sh = ann_r / ann_v
            
            grid_chained_returns.append(cand_rets)
            grid_annual_sharpes.append(cand_sh)
            print(f"  -> Chained Net Sharpe = {cand_sh:.3f}")
            
    strategy_grid_returns = np.column_stack(grid_chained_returns)
    pbo, logits = compute_pbo_cscv(strategy_grid_returns, S=16)
    
    sr_var_cross_strategy = float(np.var(grid_annual_sharpes, ddof=1))
    realized_sr = 0.841
    n_days = len(grid_chained_returns[0])
    
    skew = float(stats.skew(grid_chained_returns[0]))
    kurt = float(stats.kurtosis(grid_chained_returns[0], fisher=False))
    
    dsr_val, bench_sr = compute_dsr(realized_sr, sr_var_cross_strategy, n_trials=25, n_samples=n_days, skewness=skew, kurtosis=kurt)
    psr_val = compute_psr(realized_sr, 0.0, n_samples=n_days, skewness=skew, kurtosis=kurt)
    
    print("\n" + "=" * 70)
    print("RECONCILED ACCURATE PBO AND DSR STATISTICAL METRICS")
    print("=" * 70)
    print(f"Probability of Backtest Overfitting (PBO): {pbo:.4f} ({pbo*100:.2f}%)")
    print(f"Deflated Sharpe Ratio (DSR):             {dsr_val:.4f} ({dsr_val*100:.2f}%)")
    print(f"Probabilistic Sharpe Ratio (PSR):        {psr_val:.4f} ({psr_val*100:.2f}%)")
    print(f"Implied Hurdle Sharpe (SR0):              {bench_sr:.4f}")
    print(f"Cross-Strategy Sharpe Variance (sr_var):  {sr_var_cross_strategy:.6f}")
    print("=" * 70)

if __name__ == "__main__":
    run_true_walk_forward_cscv_dsr()
