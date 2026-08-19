import os
import random
import numpy as np
import pandas as pd

from scripts.universe import get_walk_forward_windows
from scripts.utils import set_seed, normalize_cap_cardinality, load_data_for_window
from scripts.algorithms import SOS, AOBL_SOS, mutualism_step, commensalism_step, parasitism_step, centroid_opposition
from scripts.evaluation import obj_sharpe_drawdown, compute_net_metrics_almgren_chriss

def run_ablation_variant_single(variant_name: str, obj_func, pop: np.ndarray, map_func, iters: int = 150, cap: float = 0.20, K: int = 30):
    """
    Executes a single run of a specific ablation study variant:
    A: Plain SOS
    B: SOS + Static OBL (triggered unconditionally every 20 iterations)
    C: SOS + Random Restart (stagnation triggers random initialization of worst 50%)
    D: SOS + Adaptive Trigger Only (adaptive trigger replaces worst 50% with random uniform)
    E: SOS + Centroid Opposition Only (centroid opposition every 20 iterations unconditionally)
    F: SOS + No Drawdown Penalty (lambda_dd = 0.0)
    G: Full AOBL-SOS
    """
    pop = pop.copy()
    n_pop, dim = pop.shape
    fitness = np.array([obj_func(ind) for ind in pop], dtype=float)
    best_curve = []
    
    if variant_name == "Variant A (Plain SOS)" or variant_name == "Variant F (No Drawdown Penalty)":
        return SOS(obj_func, pop, map_func, iters=iters, is_portfolio=True)
        
    elif variant_name == "Variant G (Full AOBL-SOS)":
        return AOBL_SOS(obj_func, pop, map_func, iters=iters, is_portfolio=True, cap=cap, K=K)
        
    best_val = float(np.min(fitness))
    stagnation = 0
    patience = 15
    
    for t in range(iters):
        best = pop[np.argmin(fitness)]
        
        for i in range(n_pop):
            pop, fitness = mutualism_step(pop, fitness, best, obj_func, map_func, i)
            best = pop[np.argmin(fitness)]
            pop, fitness = commensalism_step(pop, fitness, best, obj_func, map_func, i)
            best = pop[np.argmin(fitness)]
            pop, fitness = parasitism_step(pop, fitness, obj_func, map_func, i, is_portfolio=True)
            
        current_best = float(np.min(fitness))
        best_curve.append(current_best)
        
        if current_best < best_val - 1e-12:
            best_val = current_best
            stagnation = 0
        else:
            stagnation += 1
            
        if variant_name == "Variant B (Static OBL)" and (t + 1) % 20 == 0:
            worst_idx, opp = centroid_opposition(pop, fitness, replace_frac=0.5)
            opp_mapped = np.array([map_func(ind) for ind in opp])
            opp_fit = np.array([obj_func(ind) for ind in opp_mapped], dtype=float)
            improved = opp_fit < fitness[worst_idx]
            pop[worst_idx[improved]] = opp_mapped[improved]
            fitness[worst_idx[improved]] = opp_fit[improved]
            
        elif variant_name == "Variant C (Random Restart)" and stagnation >= patience:
            worst_idx = np.argsort(fitness)[-int(n_pop * 0.5):]
            rand_pop = np.random.uniform(0, 1, size=(len(worst_idx), dim))
            rand_mapped = np.array([map_func(ind) for ind in rand_pop])
            rand_fit = np.array([obj_func(ind) for ind in rand_mapped], dtype=float)
            improved = rand_fit < fitness[worst_idx]
            pop[worst_idx[improved]] = rand_mapped[improved]
            fitness[worst_idx[improved]] = rand_fit[improved]
            stagnation = max(0, patience // 2)
            
        elif variant_name == "Variant D (Adaptive Trigger Only)" and stagnation >= patience:
            p = min(0.95, 0.20 + 0.05 * (stagnation - patience + 1))
            if random.random() < p:
                worst_idx = np.argsort(fitness)[-int(n_pop * 0.5):]
                rand_pop = np.random.uniform(0, 1, size=(len(worst_idx), dim))
                rand_mapped = np.array([map_func(ind) for ind in rand_pop])
                rand_fit = np.array([obj_func(ind) for ind in rand_mapped], dtype=float)
                improved = rand_fit < fitness[worst_idx]
                pop[worst_idx[improved]] = rand_mapped[improved]
                fitness[worst_idx[improved]] = rand_fit[improved]
                stagnation = max(0, patience // 2)
                
        elif variant_name == "Variant E (Centroid Opposition Only)" and (t + 1) % 20 == 0:
            worst_idx, opp = centroid_opposition(pop, fitness, replace_frac=0.5)
            opp_mapped = np.array([map_func(ind) for ind in opp])
            opp_fit = np.array([obj_func(ind) for ind in opp_mapped], dtype=float)
            improved = opp_fit < fitness[worst_idx]
            pop[worst_idx[improved]] = opp_mapped[improved]
            fitness[worst_idx[improved]] = opp_fit[improved]
            
    idx = np.argmin(fitness)
    return float(fitness[idx]), pop[idx], best_curve

def run_walk_forward_ablation_study(data_path="data/sp500_daily.csv", n_seeds=5, iters=150, output_dir="results"):
    """
    Evaluates the 7-Variant Ablation Study across the exact same 7-window Walk-Forward Protocol with n_seeds median selection.
    Ensures ablation results are 100% synchronized with the master walk-forward benchmark.
    """
    windows = get_walk_forward_windows()
    ablation_variants = [
        "Variant A (Plain SOS)",
        "Variant B (Static OBL)",
        "Variant C (Random Restart)",
        "Variant D (Adaptive Trigger Only)",
        "Variant E (Centroid Opposition Only)",
        "Variant F (No Drawdown Penalty)",
        "Variant G (Full AOBL-SOS)"
    ]
    
    cap = 0.20
    K = 30
    rf_annual = 0.02
    aum = 100_000_000
    
    chained_variant_returns = {var: [] for var in ablation_variants}
    
    for win in windows:
        tickers, mu, cov, train_ret, test_ret, test_vol = load_data_for_window(
            data_path, win['train_start'], win['train_end'], win['test_start'], win['test_end'], cap=cap, K=K
        )
        dim = len(tickers)
        map_func = lambda w: normalize_cap_cardinality(w, cap=cap, K=K)
        
        for var_name in ablation_variants:
            metrics_list = []
            
            for seed in range(n_seeds):
                set_seed(seed * 100 + win['window_id'])
                pop = np.random.uniform(0, 1, (40, dim))
                pop = np.array([map_func(p) for p in pop])
                
                # Variant F uses pure Sharpe objective without drawdown penalty for ablation comparison
                if var_name == "Variant F (No Drawdown Penalty)":
                    obj_func = lambda w: obj_sharpe_drawdown(w, train_ret.values, rf_annual=rf_annual, lambda_dd=0.0)
                else:
                    obj_func = lambda w: obj_sharpe_drawdown(w, train_ret.values, rf_annual=rf_annual, lambda_dd=1.0)
                    
                _, w_var, _ = run_ablation_variant_single(var_name, obj_func, pop, map_func, iters=iters, cap=cap, K=K)
                res = compute_net_metrics_almgren_chriss(w_var, test_ret, test_vol, train_ret, aum=aum, rf_annual=rf_annual)
                metrics_list.append(res)
                
            # Median seed selection matching run_expanding_walk_forward
            median_idx = int(np.argsort([m['sharpe'] for m in metrics_list])[len(metrics_list)//2])
            best_res = metrics_list[median_idx]
            chained_variant_returns[var_name].extend(best_res['net_returns'])
            
    ablation_summary = []
    for var_name in ablation_variants:
        rets = np.array(chained_variant_returns[var_name])
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
        
        ablation_summary.append({
            'Ablation Variant': var_name,
            'Walk-Forward Horizon': '2018-2024 (Chained OOS)',
            'Net Sharpe': f"{sh:.3f}",
            'Net Ann Return': f"{ann_r*100:.2f}%",
            'Max Drawdown': f"{max_dd*100:.2f}%",
            'Calmar Ratio': f"{calmar:.3f}"
        })
        
    df_ablation = pd.DataFrame(ablation_summary)
    df_ablation.to_csv(os.path.join(output_dir, "ablation_study_results.csv"), index=False)
    return df_ablation
