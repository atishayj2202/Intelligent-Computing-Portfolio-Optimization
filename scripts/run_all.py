import os
import shutil
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.utils import set_seed, load_data_for_window, population_diversity, normalize_cap_cardinality
from scripts.universe import get_walk_forward_windows
from scripts.walk_forward import run_expanding_walk_forward, jobson_korkie_memmel
from scripts.ablation import run_walk_forward_ablation_study
from scripts.algorithms import AOBL_SOS, SOS
from scripts.pbo_cscv import compute_pbo_cscv, compute_dsr, compute_psr, compute_min_btl
from scripts.analysis import (
    run_factor_attribution, 
    run_aum_capacity_analysis, 
    run_vix_regime_analysis, 
    run_portfolio_stability
)
from scripts.evaluation import obj_sharpe_drawdown, compute_net_metrics_almgren_chriss, compute_net_metrics_fixed_bps
from scripts.diagnostics import (
    run_carhart_attribution,
    stationary_block_bootstrap_delta_sharpe,
    run_cost_sensitivity,
    run_capacity_from_weights,
)

def run_master_suite(output_dir="results", manuscript_dir="manuscript"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(manuscript_dir, exist_ok=True)
    
    print("=" * 70)
    print("EXECUTING FULL QUANTITATIVE FINANCE RESEARCH OVERHAUL SUITE")
    print("=" * 70)
    
    set_seed(42)
    data_path = "data/sp500_daily.csv"
    
    # ---------------------------------------------------------
    # 1. Expanding Walk-Forward & Point-in-Time Universe
    # ---------------------------------------------------------
    print("\n[1/7] Running Expanding Walk-Forward Protocol (7 Windows, Point-in-Time Universe)...")
    df_wf_per_win, df_wf_agg, chained_returns = run_expanding_walk_forward(
        data_path=data_path, n_seeds=5, iters=150, output_dir=output_dir
    )
    
    # Jobson-Korkie Significance Tests on Chained Return Streams
    aobl_rets = np.array(chained_returns["AOBL-SOS (Proposed)"])
    sos_rets = np.array(chained_returns["SOS (Baseline)"])
    pso_rets = np.array(chained_returns["PSO"])
    ga_rets = np.array(chained_returns["GA"])
    lw_rets = np.array(chained_returns["Ledoit-Wolf"])
    eq_rets = np.array(chained_returns["Equal Weight (1/N)"])
    
    sh1, sh2, z_sos, p_sos = jobson_korkie_memmel(aobl_rets, sos_rets)
    _, _, z_pso, p_pso = jobson_korkie_memmel(aobl_rets, pso_rets)
    _, _, z_ga, p_ga = jobson_korkie_memmel(aobl_rets, ga_rets)
    _, _, z_lw, p_lw = jobson_korkie_memmel(aobl_rets, lw_rets)
    _, _, z_eq, p_eq = jobson_korkie_memmel(aobl_rets, eq_rets)
    
    pd.DataFrame([
        {"Comparison": "vs SOS", "Z": z_sos, "p_value": p_sos},
        {"Comparison": "vs PSO", "Z": z_pso, "p_value": p_pso},
        {"Comparison": "vs GA", "Z": z_ga, "p_value": p_ga},
        {"Comparison": "vs Ledoit-Wolf", "Z": z_lw, "p_value": p_lw},
        {"Comparison": "vs Equal Weight (1/N)", "Z": z_eq, "p_value": p_eq},
    ]).to_csv(os.path.join(output_dir, "statistical_significance.csv"), index=False)
    boot = stationary_block_bootstrap_delta_sharpe(aobl_rets, sos_rets, block_mean=21, n_boot=2000, rf_annual=0.02)
    pd.DataFrame([boot]).to_csv(os.path.join(output_dir, "block_bootstrap_delta_sharpe.csv"), index=False)
    
    # ---------------------------------------------------------
    # 2. Walk-Forward 7-Variant Ablation Study
    # ---------------------------------------------------------
    print("\n[2/7] Running Walk-Forward 7-Variant Ablation Study...")
    df_ablation = run_walk_forward_ablation_study(data_path=data_path, n_seeds=5, iters=150, output_dir=output_dir)
    
    # ---------------------------------------------------------
    # 3. CSCV / PBO / DSR Overfitting Analysis
    # ---------------------------------------------------------
    print("\n[3/7] Running CSCV / PBO / DSR Overfitting Analysis on Real Strategy Grid...")
    windows = get_walk_forward_windows()
    grid_returns_list = []
    grid_annual_sharpes = []
    
    # Hyperparameter grid: 5 lambdas x 5 Ks = 25 candidate configurations
    lambda_grid = [0.0, 0.5, 1.0, 1.5, 2.0]
    k_grid = [15, 20, 25, 30, 35]
    
    for l_val in lambda_grid:
        for k_val in k_grid:
            cand_chained = []
            for win in windows:
                tickers, mu, cov, train_ret, test_ret, test_vol = load_data_for_window(
                    data_path, win['train_start'], win['train_end'], win['test_start'], win['test_end'], cap=0.20, K=k_val
                )
                dim = len(tickers)
                map_func = lambda w: normalize_cap_cardinality(w, cap=0.20, K=k_val)
                obj_func = lambda w: obj_sharpe_drawdown(w, train_ret.values, rf_annual=0.02, lambda_dd=l_val)
                
                set_seed(42 + win['window_id'])
                pop = np.random.uniform(0, 1, (40, dim))
                pop = np.array([map_func(p) for p in pop])
                
                _, w_cand, _ = AOBL_SOS(obj_func, pop, map_func, iters=100, is_portfolio=True, cap=0.20, K=k_val)
                res = compute_net_metrics_fixed_bps(w_cand, test_ret, cost_bps=10, rebal_freq=None, rf_annual=0.02)
                cand_chained.extend(res['net_returns'])
                
            rets_arr = np.array(cand_chained)
            ann_r = (np.prod(1 + rets_arr) ** (252.0 / len(rets_arr))) - 1.0 - 0.02
            ann_v = np.std(rets_arr, ddof=1) * np.sqrt(252) + 1e-12
            cand_sh = ann_r / ann_v
            grid_annual_sharpes.append(cand_sh)
            grid_returns_list.append(cand_chained)
            
    strategy_grid_returns = np.column_stack(grid_returns_list)
    pbo, logits = compute_pbo_cscv(strategy_grid_returns, S=16)
    
    aobl_series = pd.Series(aobl_rets)
    n_days = len(aobl_rets)
    realized_sr = float(df_wf_agg.loc[df_wf_agg['Algorithm'] == "AOBL-SOS (Proposed)", 'Net Sharpe'].values[0])
    
    # López de Prado (2014) cross-strategy variance across 25 grid candidates
    sr_var = float(np.var(grid_annual_sharpes, ddof=1))
    
    skew = float(stats.skew(aobl_rets))
    kurt = float(stats.kurtosis(aobl_rets, fisher=False))
    
    dsr_val, bench_sr = compute_dsr(realized_sr, sr_var, n_trials=25, n_samples=n_days, skewness=skew, kurtosis=kurt)
    psr_val = compute_psr(realized_sr, 0.0, n_samples=n_days, skewness=skew, kurtosis=kurt)
    min_btl = compute_min_btl(realized_sr, bench_sr, skewness=skew, kurtosis=kurt)
    
    pd.DataFrame([{
        "PBO": pbo,
        "DSR": dsr_val,
        "PSR": psr_val,
        "SR0": bench_sr,
        "MinBTL_years": (np.nan if np.isnan(min_btl) else min_btl),
        "Grid": "lambda x K = 5 x 5",
        "Headline_iters": 150,
        "PBO_grid_iters": 100,
        "PBO_grid_pop": 40,
        "PBO_cost_model": "10bps_annual_formation",
    }]).to_csv(os.path.join(output_dir, "pbo_dsr_results.csv"), index=False)
    pd.DataFrame({"lambda": np.repeat(lambda_grid, len(k_grid)), "K": np.tile(k_grid, len(lambda_grid)), "OOS_Sharpe": grid_annual_sharpes}).to_csv(os.path.join(output_dir, "pbo_grid_sharpes.csv"), index=False)
        
    # ---------------------------------------------------------
    # 4. Factor Attribution & VIX Regime Analysis
    # ---------------------------------------------------------
    print("\n[4/7] Running Ken French/Carhart attribution and volatility-regime analysis...")
    daily = pd.read_csv(os.path.join(output_dir, "chained_daily_returns.csv"), parse_dates=["Date"])
    aobl_dates_series = daily.set_index("Date")["AOBL-SOS (Proposed)"]
    run_carhart_attribution(aobl_dates_series, output_dir=output_dir)
    df_regime = run_vix_regime_analysis(aobl_dates_series, data_path=data_path, output_dir=output_dir)
    # keep train_ret for the diversity plot below
    tickers, mu, cov, train_ret, test_ret, test_vol = load_data_for_window(
        data_path, '2012-01-01', '2017-12-31', '2018-01-01', '2024-12-31'
    )
    
    # ---------------------------------------------------------
    # 5. AUM Capacity & Portfolio Stability
    # ---------------------------------------------------------
    print("\n[5/7] Running AUM Capacity & Portfolio Stability Analysis...")
    dim = len(tickers)
    map_func = lambda w: normalize_cap_cardinality(w, cap=0.20, K=30)
    obj_dd = lambda w: obj_sharpe_drawdown(w, train_ret.values, rf_annual=0.02)
    
    set_seed(42)
    pop = np.random.uniform(0, 1, (40, dim))
    pop = np.array([map_func(p) for p in pop])
    _, w_best, _ = AOBL_SOS(obj_dd, pop, map_func, iters=150, is_portfolio=True)
    weights_csv = os.path.join(output_dir, "selected_weights.csv")
    df_capacity = run_capacity_from_weights(weights_csv, data_path, output_dir=output_dir)
    run_cost_sensitivity(weights_csv, data_path, output_dir=output_dir)
    
    seed_weights = []
    for s in range(20):
        set_seed(s)
        p = np.random.uniform(0, 1, (40, dim))
        p = np.array([map_func(ind) for ind in p])
        _, w_s, _ = AOBL_SOS(obj_dd, p, map_func, iters=150, is_portfolio=True)
        seed_weights.append(w_s)
        
    stability_metrics = run_portfolio_stability(seed_weights, tickers, output_dir=output_dir)
    
    # ---------------------------------------------------------
    # 6. Convergence & Population Diversity Plots
    # ---------------------------------------------------------
    print("\n[6/7] Generating Convergence and Population Diversity Plots...")
    set_seed(42)
    pop_init = np.random.uniform(0, 1, (40, dim))
    pop_init = np.array([map_func(p) for p in pop_init])
    
    _, _, fit_aobl, div_aobl = AOBL_SOS(obj_dd, pop_init.copy(), map_func, iters=150, is_portfolio=True, return_diversity=True)
    _, _, fit_sos, div_sos = SOS(obj_dd, pop_init.copy(), map_func, iters=150, is_portfolio=True, return_diversity=True)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(-np.array(fit_aobl), label='AOBL-SOS', color='#D85A30', linewidth=2)
    plt.plot(-np.array(fit_sos), label='SOS Baseline', color='steelblue', linestyle='--', linewidth=1.5)
    plt.title('Loss Function Convergence $g(\mathbf{w})$')
    plt.xlabel('Iteration')
    plt.ylabel('Negative Loss (Higher is Better)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(div_aobl, label='AOBL-SOS Diversity', color='#D85A30', linewidth=2)
    plt.plot(div_sos, label='SOS Diversity', color='steelblue', linestyle='--', linewidth=1.5)
    plt.title('Population Diversity ($D_t$)')
    plt.xlabel('Iteration')
    plt.ylabel('Pairwise L2 Diversity')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "convergence_and_diversity.png"), dpi=200)
    plt.close()
    
    # ---------------------------------------------------------
    # 7. Copying generated assets to manuscript/
    # ---------------------------------------------------------
    print("\n[7/7] Copying all final CSVs and figures to manuscript/...")
    for item in os.listdir(output_dir):
        s = os.path.join(output_dir, item)
        d = os.path.join(manuscript_dir, item)
        if os.path.isfile(s) and item.endswith((".csv", ".png")):
            shutil.copy2(s, d)
            
    print("\n" + "=" * 70)
    print("SUCCESS: Full Quantitative Research Suite Completed!")
    print(f"Results stored in: {output_dir}/ and synced to: {manuscript_dir}/")
    print("=" * 70)

if __name__ == "__main__":
    run_master_suite()
