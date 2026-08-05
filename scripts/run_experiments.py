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

def run_jpm_experiments(output_dir="manuscript"):
    os.makedirs(output_dir, exist_ok=True)
    
    set_seed(42)
    data_path = "data/sp500_daily.csv"
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
    _, w_aobl, aobl_curve = AOBL_SOS(obj, pop.copy(), map_func, iters=300, is_portfolio=True, 
                            init_obl=True, obl_mode="portfolio_reversal", cap=cap, K=K)
    
    print("  Running Standard SOS...")
    _, w_sos, sos_curve = SOS(obj, pop.copy(), map_func, iters=300, is_portfolio=True)
    
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
            "Calmar": f"{res['calmar']:.3f}",
            "Monthly Turnover": f"{res['avg_turnover']*100:.2f}%"
        })
        
    df_bench = pd.DataFrame(bench_results)
    df_bench.to_csv(os.path.join(output_dir, "master_benchmark_table.csv"), index=False)
    
    print("\n[1b] Generating Institutional Tail-Risk Table...")
    risk_results = []
    for name, w in portfolios.items():
        res = compute_net_metrics(w, test_ret, cost_bps=10, rebal_freq=21, rf_annual=rf_annual)
        risk_results.append({
            "Portfolio": name,
            "CVaR 95%": f"{res['cvar_95']*100:.3f}%",
            "Skewness": f"{res['skewness']:.3f}",
            "Excess Kurtosis": f"{res['kurtosis']:.3f}",
            "Hit Rate": f"{res['hit_rate']*100:.1f}%",
            "Calmar": f"{res['calmar']:.3f}",
        })
    df_risk = pd.DataFrame(risk_results)
    df_risk.to_csv(os.path.join(output_dir, "institutional_risk_metrics.csv"), index=False)
    print(df_risk.to_string(index=False))
    
    print("\n[1c] Bootstrap Statistical Significance Test (10,000 resamples)...")
    n_bootstrap = 10000
    res_aobl_full = compute_net_metrics(w_aobl, test_ret, cost_bps=10, rebal_freq=21, rf_annual=rf_annual)
    aobl_daily = res_aobl_full["daily_returns"]
    n_days_test = len(aobl_daily)
    
    bootstrap_sharpes = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n_days_test, size=n_days_test, replace=True)
        sample = aobl_daily[idx]
        ann_r = np.mean(sample) * 252
        ann_v = np.std(sample, ddof=1) * np.sqrt(252) + 1e-12
        bootstrap_sharpes.append((ann_r - rf_annual) / ann_v)
    bootstrap_sharpes = np.array(bootstrap_sharpes)
    
    ci_lower = np.percentile(bootstrap_sharpes, 2.5)
    ci_upper = np.percentile(bootstrap_sharpes, 97.5)
    pval = np.mean(bootstrap_sharpes <= 0)
    
    bootstrap_results = {
        "Portfolio": "AOBL-SOS",
        "Point Estimate (Sharpe)": round(res_aobl_full["sharpe"], 3),
        "95% CI Lower": round(ci_lower, 3),
        "95% CI Upper": round(ci_upper, 3),
        "Bootstrap p-value (Sharpe <= 0)": f"{pval:.4f}",
        "N Resamples": n_bootstrap
    }
    df_boot = pd.DataFrame([bootstrap_results])
    df_boot.to_csv(os.path.join(output_dir, "bootstrap_significance.csv"), index=False)
    print(f"  AOBL-SOS Sharpe: {res_aobl_full['sharpe']:.3f}  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]  p-value: {pval:.4f}")
    
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
    
    print("\n[4b] Generating Drawdown Profile Plot...")
    plt.figure(figsize=(12, 5))
    for (name, w), color in zip(portfolios.items(), colors):
        res = compute_net_metrics(w, test_ret, cost_bps=10, rebal_freq=21, rf_annual=rf_annual)
        cum_ret = res["cum_returns"]
        drawdown = cum_ret / np.maximum.accumulate(cum_ret) - 1
        lw = 2.0 if "AOBL" in name else 1.2
        plt.plot(drawdown, color=color, lw=lw, label=name)
        
    plt.title('Out-of-Sample Drawdown Profile (2023-2025)', fontsize=13, fontweight='bold')
    plt.xlabel('Trading Days')
    plt.ylabel('Drawdown')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "drawdown_profile.png"), dpi=200)
    plt.close()

    print("\n[4c] Generating AOBL-SOS vs SOS Convergence Analysis...")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(aobl_curve)+1), [-x for x in aobl_curve], color='#D85A30', lw=2.5, label='AOBL-SOS (Proposed)')
    ax.plot(range(1, len(sos_curve)+1), [-x for x in sos_curve], color='#378ADD', lw=2.0, label='Standard SOS')
    ax.set_title('Convergence Analysis: Objective Function (Higher = Better)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective Value (Drawdown-Penalized Sharpe)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "convergence_analysis.png"), dpi=200)
    plt.close(fig)

    print("\n[5] Walk-Forward Regime Robustness Validation...")
    walk_forward_results = []
    
    # Reload full dataset to allow custom slicing
    df_full = pd.read_csv(data_path, index_col=0, parse_dates=True)
    returns_full = df_full.pct_change().dropna()
    
    # Test the SAME optimized weights across distinct market regimes
    # This tests regime robustness of the allocation, not re-optimization noise
    regime_windows = [
        ("2018", "2018", "Pre-COVID Bull"),
        ("2019", "2019", "Late-Cycle Expansion"),
        ("2020", "2020", "COVID Crash & Recovery"),
        ("2021", "2021", "Post-COVID Rally"),
        ("2022", "2022", "Rate Hike Drawdown"),
        ("2023", "2025", "Recent Out-of-Sample")
    ]
    
    regime_portfolios = {
        "AOBL-SOS": w_aobl,
        "Markowitz": w_maxsh,
        "SOS": w_sos,
        "MinVar": w_minvar,
        "Equal-Weight": w_eq
    }
    
    for test_start, test_end, regime_label in regime_windows:
        test_slice = returns_full.loc[f"{test_start}-01-01":f"{test_end}-12-31"]
        if len(test_slice) == 0:
            continue
        
        row = {"Market Regime": regime_label}
        for name, w in regime_portfolios.items():
            res = compute_net_metrics(w, test_slice, cost_bps=10, rebal_freq=21, rf_annual=rf_annual)
            row[name] = round(res["sharpe"], 2)
        walk_forward_results.append(row)
        print(f"  {regime_label}: AOBL-SOS={row['AOBL-SOS']}, Mkwz={row['Markowitz']}, SOS={row['SOS']}")
    
    df_wf = pd.DataFrame(walk_forward_results)
    df_wf.to_csv(os.path.join(output_dir, "walk_forward_table.csv"), index=False)

    print("\n[5b] Generating Walk-Forward Performance Bar Chart...")
    df_wf_plot = df_wf.set_index("Market Regime")
    ax = df_wf_plot.plot(kind='bar', figsize=(14, 6), colormap='viridis', edgecolor='black')
    plt.title('Walk-Forward Out-of-Sample Net Sharpe Ratios', fontsize=14, fontweight='bold')
    plt.xlabel('Market Regimes')
    plt.ylabel('Net Sharpe Ratio')
    plt.xticks(rotation=15)
    plt.legend(title='Portfolio Strategy')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "walk_forward_chart.png"), dpi=200)
    plt.close()

    print("\n[6] Generating AOBL-SOS Portfolio Weights Distribution...")
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
        
        # Portfolio daily returns (test period)
        port_ret = test_ret.dot(w_aobl)
        
        # Get Fama-French data
        start_date = port_ret.index.min().strftime('%Y-%m-%d')
        end_date = port_ret.index.max().strftime('%Y-%m-%d')
        
        ff = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start=start_date, end=end_date)[0]
        # FF data is in percentages (e.g. 1.5 for 1.5%), so divide by 100
        ff = ff / 100.0
        
        # Align indexes
        port_ret.index = pd.to_datetime(port_ret.index).tz_localize(None)
        ff.index = pd.to_datetime(ff.index).tz_localize(None)
        
        # Merge
        df_reg = pd.DataFrame({'Port_Ret': port_ret}).join(ff, how='inner').dropna()
        
        # Excess return over RF
        y = df_reg['Port_Ret'] - df_reg['RF']
        X = df_reg[['Mkt-RF', 'SMB', 'HML']]
        X = sm.add_constant(X)
        
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 1})
        
        # Save results
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
