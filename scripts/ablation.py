import random
import numpy as np
from scripts.algorithms import SOS, AOBL_SOS, mutualism_step, commensalism_step, parasitism_step, centroid_opposition

def run_ablation_variant(variant_name: str, obj_func, pop: np.ndarray, map_func, iters: int = 300, cap: float = 0.20, K: int = 30):
    """
    Executes a specific ablation study variant:
    A: Plain SOS
    B: SOS + Static OBL (triggered unconditionally every 20 iterations)
    C: SOS + Random Restart (stagnation triggers random initialization of worst 50%)
    D: SOS + Adaptive Trigger Only (adaptive trigger replaces worst 50% with random uniform)
    E: SOS + Centroid Opposition Only (centroid opposition every 20 iterations unconditionally)
    F: SOS + Drawdown Penalty Only (standard SOS on drawdown objective)
    G: Full AOBL-SOS
    """
    pop = pop.copy()
    n_pop, dim = pop.shape
    fitness = np.array([obj_func(ind) for ind in pop], dtype=float)
    best_curve = []
    
    if variant_name == "Variant A (Plain SOS)" or variant_name == "Variant F (Drawdown Only)":
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
