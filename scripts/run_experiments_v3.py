import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.utils import set_seed, normalize_cap_cardinality, load_data
from scripts.algorithms import AOBL_SOS, SOS
from scripts.evaluation import obj_sharpe_drawdown, obj_sharpe_only, compute_net_metrics

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

def run_jpm_experiments(output_dir="manuscript"):
    os.makedirs(output_dir, exist_ok=True)
    
    set_seed(42)
    data_path = "data/sp500_daily.csv"
    if not os.path.exists(data_path):
        data_path = "../../data/sp500_daily.csv"
        
    tickers, mu, cov, train_ret, test_ret = load_data(data_path, n_stocks=150, seed=42)
    
    cap = 0.20
    K = 30
    rf_annual = 0.02
    rf_daily = rf_annual / 252
    
    print("Optimizing Portfolios (10 Seeds for stochastic metaheuristics)...")
    dim = len(mu)
    map_func = lambda w: normalize_cap_cardinality(w, cap=cap, K=K)
    obj_dd = lambda w: obj_sharpe_drawdown(w, train_ret.values, rf_annual=rf_annual)
    obj_mv = lambda w: obj_sharpe_only(w, train_ret.values, rf_annual=rf_annual)
    
    n_seeds = 10
    aobl_dd_weights, aobl_mv_weights = [], []
    sos_dd_weights, sos_mv_weights = [], []
    
    for seed in range(n_seeds):
        set_seed(seed)
        pop = np.random.uniform(0.0, 1.0, (50, dim))
        pop = np.array([map_func(p) for p in pop])
        
        _, w1, _ = AOBL_SOS(obj_dd, pop.copy(), map_func, iters=300, is_portfolio=True, init_obl=True, obl_mode="portfolio_reversal", cap=cap, K=K)
        aobl_dd_weights.append(w1)
        
        _, w2, _ = AOBL_SOS(obj_mv, pop.copy(), map_func, iters=300, is_portfolio=True, init_obl=True, obl_mode="portfolio_reversal", cap=cap, K=K)
        aobl_mv_weights.append(w2)
        
        _, w3, _ = SOS(obj_dd, pop.copy(), map_func, iters=300, is_portfolio=True)
        sos_dd_weights.append(w3)
        
        _, w4, _ = SOS(obj_mv, pop.copy(), map_func, iters=300, is_portfolio=True)
        sos_mv_weights.append(w4)

    w_aobl_dd = np.mean(aobl_dd_weights, axis=0)
    w_aobl_dd = normalize_cap_cardinality(w_aobl_dd, cap=cap, K=K)
    
    print("  Computing Min Variance...")
    w_minvar = min_variance_portfolio(cov, cap=cap, K=K)
    
    print("  Computing Max Sharpe (Markowitz)...")
    w_maxsh = max_sharpe_portfolio(mu, cov, rf_daily, cap=cap, K=K)
    
    print("  Computing Risk Parity (Inv-Vol)...")
    w_invvol = inverse_volatility_portfolio(cov, cap=cap, K=K)
    
    w_eq = normalize_cap_cardinality(np.ones(dim), cap=cap, K=K)
    
    def avg_metrics(weights_list, test_df):
        sharpes, calmars, cvars = [], [], []
        for w in weights_list:
            res = compute_net_metrics(w, test_df, cost_bps=10, rebal_freq=21, rf_annual=rf_annual)
            sharpes.append(res['sharpe'])
            calmars.append(res['calmar'])
            cvars.append(res['cvar_95'])
        return np.mean(sharpes), np.std(sharpes), np.mean(calmars), np.std(calmars), np.mean(cvars), np.std(cvars)
        
    print("\n[Ablation Study] Evaluating 10 Seeds (AOBL/Base x MV/Drawdown)...")
    ablation_results = []
    
    configs = [
        ("Base SOS (MV Obj)", sos_mv_weights),
        ("Base SOS (DD Obj)", sos_dd_weights),
        ("AOBL-SOS (MV Obj)", aobl_mv_weights),
        ("AOBL-SOS (Proposed, DD Obj)", aobl_dd_weights)
    ]
    
    for name, w_list in configs:
        m_sh, s_sh, m_cal, s_cal, m_cvar, s_cvar = avg_metrics(w_list, test_ret)
        ablation_results.append({
            "Variant": name,
            "Net Sharpe (Mean ± Std)": f"{m_sh:.3f} ± {s_sh:.3f}",
            "Calmar (Mean ± Std)": f"{m_cal:.3f} ± {s_cal:.3f}",
            "CVaR 95% (Mean ± Std)": f"{m_cvar*100:.2f}% ± {s_cvar*100:.2f}%"
        })
    df_ablation = pd.DataFrame(ablation_results)
    df_ablation.to_csv(os.path.join(output_dir, "ablation_study.csv"), index=False)
    
    print("\n[1] Evaluating Master Benchmark Table (10 bps Cost, Monthly Rebal)...")
    
    def get_full_metrics_mean_std(weights_list, test_df):
        results = []
        for w in weights_list:
            results.append(compute_net_metrics(w, test_df, cost_bps=10, rebal_freq=21, rf_annual=rf_annual))
        keys = ['ann_return', 'ann_vol', 'sharpe', 'sortino', 'max_drawdown', 'calmar', 'avg_turnover']
        m_stds = {}
        for k in keys:
            vals = [r[k] for r in results]
            m_stds[k] = (np.mean(vals), np.std(vals))
        return m_stds

    def fmt(m, s, is_pct=False):
        if is_pct:
            return f"{m*100:.2f}% ± {s*100:.2f}%"
        return f"{m:.3f} ± {s:.3f}"

    bench_results = []
    # Add stochastic models
    for name, w_list in [("AOBL-SOS (Proposed)", aobl_dd_weights), ("SOS (Baseline)", sos_dd_weights)]:
        m_stds = get_full_metrics_mean_std(w_list, test_ret)
        bench_results.append({
            "Portfolio": name,
            "Cost (bps)": 10,
            "Net Ann. Return": fmt(m_stds['ann_return'][0], m_stds['ann_return'][1], True),
            "Net Ann. Vol": fmt(m_stds['ann_vol'][0], m_stds['ann_vol'][1], True),
            "Net Sharpe": fmt(m_stds['sharpe'][0], m_stds['sharpe'][1]),
            "Net Sortino": fmt(m_stds['sortino'][0], m_stds['sortino'][1]),
            "Max Drawdown": fmt(m_stds['max_drawdown'][0], m_stds['max_drawdown'][1], True),
            "Calmar": fmt(m_stds['calmar'][0], m_stds['calmar'][1]),
            "Monthly Turnover": fmt(m_stds['avg_turnover'][0], m_stds['avg_turnover'][1], True)
        })

    # Add deterministic models
    det_portfolios = {
        "Max Sharpe (Markowitz)": w_maxsh,
        "Risk Parity (Inv-Vol)": w_invvol,
        "Min Variance": w_minvar,
        "Equal Weight (1/N)": w_eq
    }
    for name, w in det_portfolios.items():
        res = compute_net_metrics(w, test_ret, cost_bps=10, rebal_freq=21, rf_annual=rf_annual)
        bench_results.append({
            "Portfolio": name,
            "Cost (bps)": 10,
            "Net Ann. Return": f"{res['ann_return']*100:.2f}%",
            "Net Ann. Vol": f"{res['ann_vol']*100:.2f}%",
            "Net Sharpe": f"{res['sharpe']:.3f}",
            "Net Sortino": f"{res['sortino']:.3f}",
            "Max Drawdown": f"{res['max_drawdown']*100:.2f}%",
            "Calmar": f"{res['calmar']:.3f}",
            "Monthly Turnover": f"{res['avg_turnover']*100:.2f}%"
        })
        
    df_bench = pd.DataFrame(bench_results)
    df_bench.to_csv(os.path.join(output_dir, "master_benchmark_table.csv"), index=False)
    
    print("\n[5] Walk-Forward Regime Robustness Validation...")
    walk_forward_results = []
    
    df_full = pd.read_csv(data_path, index_col=0, parse_dates=True)
    returns_full = df_full.dropna()
    
    regime_windows = [
        ("2018", "2018", "Pre-COVID Bull"),
        ("2019", "2019", "Late-Cycle Expansion"),
        ("2020", "2020", "COVID Crash & Recovery"),
        ("2021", "2021", "Post-COVID Rally"),
        ("2022", "2022", "Rate Hike Drawdown"),
        ("2023", "2025", "Recent Out-of-Sample")
    ]
    
    for test_start, test_end, regime_label in regime_windows:
        test_slice = returns_full.loc[f"{test_start}-01-01":f"{test_end}-12-31"]
        if len(test_slice) == 0:
            continue
        
        row = {"Market Regime": regime_label}
        
        # Stochastic Models
        for name, w_list in [("AOBL-SOS", aobl_dd_weights), ("SOS", sos_dd_weights)]:
            sharpes = [compute_net_metrics(w, test_slice, cost_bps=10, rebal_freq=21, rf_annual=rf_annual)['sharpe'] for w in w_list]
            row[name] = f"{np.mean(sharpes):.2f} ± {np.std(sharpes):.2f}"
            
        # Deterministic Models
        for name, w in [("Markowitz", w_maxsh), ("MinVar", w_minvar), ("Equal-Weight", w_eq)]:
            row[name] = f"{compute_net_metrics(w, test_slice, cost_bps=10, rebal_freq=21, rf_annual=rf_annual)['sharpe']:.2f}"
            
        walk_forward_results.append(row)
    
    df_wf = pd.DataFrame(walk_forward_results)
    df_wf.to_csv(os.path.join(output_dir, "walk_forward_table.csv"), index=False)

    print("\n[6] Generating AOBL-SOS Portfolio Weights Distribution...")
    # Use Seed 0 for the singular plot
    w_aobl = aobl_dd_weights[0]
    w_active = w_aobl[w_aobl > 1e-4]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(w_active)), sorted(w_active, reverse=True), color='#D85A30', edgecolor='black')
    plt.axhline(y=0.20, color='red', linestyle='--', label='20% Upper Cap')
    plt.title(f'AOBL-SOS Active Weights Distribution (K={len(w_active)})', fontsize=13, fontweight='bold')
    plt.xlabel('Active Asset Rank')
    plt.ylabel('Allocation Weight')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "aobl_weights_dist.png"), dpi=200)
    plt.close()

    print("\n[7] Generating Fama-French 3-Factor Attribution...")
    try:
        import pandas_datareader.data as web
        import statsmodels.api as sm
        import warnings
        warnings.filterwarnings('ignore')
        
        port_ret = test_ret.dot(w_aobl)
        start_date = port_ret.index.min().strftime('%Y-%m-%d')
        end_date = port_ret.index.max().strftime('%Y-%m-%d')
        
        ff = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start=start_date, end=end_date)[0]
        ff = ff / 100.0
        
        port_ret.index = pd.to_datetime(port_ret.index).tz_localize(None)
        ff.index = pd.to_datetime(ff.index).tz_localize(None)
        
        df_reg = pd.DataFrame({'Port_Ret': port_ret}).join(ff, how='inner').dropna()
        y = df_reg['Port_Ret'] - df_reg['RF']
        X = df_reg[['Mkt-RF', 'SMB', 'HML']]
        X = sm.add_constant(X)
        
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 1})
        
        res_df = pd.DataFrame({
            'Coefficient': model.params,
            't-stat': model.tvalues,
            'p-value': model.pvalues
        })
        res_df.to_csv(os.path.join(output_dir, "factor_attribution.csv"))
        
        print("  Factor attribution generated successfully.")
    except Exception as e:
        print("  Failed to generate factor attribution:", e)

    print("Experiments completed. Output files generated in:", output_dir)

if __name__ == "__main__":
    run_jpm_experiments()
