import random
import numpy as np
from scripts.utils import normalize_cap_cardinality, population_diversity

def centroid_opposition(pop: np.ndarray, fitness: np.ndarray, replace_frac: float = 0.5, gamma_base: float = 1.0, stagnation_boost: float = 0.0) -> np.ndarray:
    """
    Centroid-based adaptive opposition operator:
    Calculates population centroid in continuous solution space:
    c = 1/N * \sum x_i
    Reflects worst-performing individuals across centroid:
    x_opp = c + gamma * (c - x_i)
    where gamma adapts based on stagnation severity.
    """
    n = len(pop)
    k_replace = max(1, int(n * replace_frac))
    worst_idx = np.argsort(fitness)[-k_replace:]
    
    centroid = np.mean(pop, axis=0)
    gamma = gamma_base + stagnation_boost
    
    opp = centroid + gamma * (centroid - pop[worst_idx])
    return worst_idx, opp

def mutualism_step(pop: np.ndarray, fitness: np.ndarray, best: np.ndarray, 
                    obj_func, map_func, i: int):
    n = len(pop)
    j = random.randrange(n)
    while j == i:
        j = random.randrange(n)
        
    mutual_vector = (pop[i] + pop[j]) / 2.0
    BF1 = random.choice([1, 2])
    BF2 = random.choice([1, 2])
    
    xi_new = map_func(pop[i] + random.random() * (best - BF1 * mutual_vector))
    xj_new = map_func(pop[j] + random.random() * (best - BF2 * mutual_vector))
    
    f_xi = obj_func(xi_new)
    if f_xi < fitness[i]:
        pop[i] = xi_new
        fitness[i] = f_xi
        
    f_xj = obj_func(xj_new)
    if f_xj < fitness[j]:
        pop[j] = xj_new
        fitness[j] = f_xj
        
    return pop, fitness

def commensalism_step(pop: np.ndarray, fitness: np.ndarray, best: np.ndarray, 
                       obj_func, map_func, i: int):
    n = len(pop)
    j = random.randrange(n)
    while j == i:
        j = random.randrange(n)
        
    xi_new = map_func(pop[i] + random.uniform(-1, 1) * (best - pop[j]))
    
    f_xi = obj_func(xi_new)
    if f_xi < fitness[i]:
        pop[i] = xi_new
        fitness[i] = f_xi
        
    return pop, fitness

def parasitism_step(pop: np.ndarray, fitness: np.ndarray, obj_func, map_func, i: int,
                     lb: float = None, ub: float = None, is_portfolio: bool = False):
    n = len(pop)
    j = random.randrange(n)
    while j == i:
        j = random.randrange(n)
        
    parasite = pop[i].copy()
    dim = len(parasite)
    
    k = max(1, dim // 10)
    idxs = random.sample(range(dim), k)
    
    for idx in idxs:
        if is_portfolio:
            parasite[idx] = random.random()
        else:
            parasite[idx] = random.uniform(lb, ub)
            
    parasite = map_func(parasite)
    
    f_p = obj_func(parasite)
    if f_p < fitness[j]:
        pop[j] = parasite
        fitness[j] = f_p
        
    return pop, fitness

def SOS(obj_func, pop: np.ndarray, map_func, iters: int, 
        lb: float = None, ub: float = None, is_portfolio: bool = False, return_diversity: bool = False):
    pop = pop.copy()
    fitness = np.array([obj_func(ind) for ind in pop], dtype=float)
    best_curve = []
    diversity_curve = []
    
    for t in range(iters):
        best = pop[np.argmin(fitness)]
        
        for i in range(len(pop)):
            pop, fitness = mutualism_step(pop, fitness, best, obj_func, map_func, i)
            best = pop[np.argmin(fitness)]
            pop, fitness = commensalism_step(pop, fitness, best, obj_func, map_func, i)
            best = pop[np.argmin(fitness)]
            pop, fitness = parasitism_step(pop, fitness, obj_func, map_func, i, 
                                           lb=lb, ub=ub, is_portfolio=is_portfolio)
            
        best_curve.append(float(np.min(fitness)))
        if return_diversity:
            diversity_curve.append(population_diversity(pop))
        
    idx = np.argmin(fitness)
    if return_diversity:
        return float(fitness[idx]), pop[idx], best_curve, diversity_curve
    return float(fitness[idx]), pop[idx], best_curve

def AOBL_SOS(
    obj_func,
    pop: np.ndarray,
    map_func,
    iters: int,
    lb: float = None,
    ub: float = None,
    is_portfolio: bool = False,
    replace_frac: float = 0.5,
    patience: int = 15,
    eps: float = 1e-12,
    cap: float = 0.20,
    K: int = 30,
    return_diversity: bool = False
):
    """
    Adaptive Opposition-Based Learning Symbiotic Organisms Search (AOBL-SOS)
    Uses centroid-based opposition with dynamic stagnation triggers.
    Tracks population diversity dynamically.
    """
    pop = pop.copy()
    fitness = np.array([obj_func(ind) for ind in pop], dtype=float)
    
    best_curve = []
    diversity_curve = []
    best_val = float(np.min(fitness))
    stagnation = 0
    
    for t in range(iters):
        best = pop[np.argmin(fitness)]
        
        for i in range(len(pop)):
            pop, fitness = mutualism_step(pop, fitness, best, obj_func, map_func, i)
            best = pop[np.argmin(fitness)]
            pop, fitness = commensalism_step(pop, fitness, best, obj_func, map_func, i)
            best = pop[np.argmin(fitness)]
            pop, fitness = parasitism_step(pop, fitness, obj_func, map_func, i, 
                                           lb=lb, ub=ub, is_portfolio=is_portfolio)
            
        current_best = float(np.min(fitness))
        best_curve.append(current_best)
        
        if return_diversity:
            diversity_curve.append(population_diversity(pop))
        
        if current_best < best_val - eps:
            best_val = current_best
            stagnation = 0
        else:
            stagnation += 1
            
        if stagnation >= patience:
            stagnation_boost = 0.05 * (stagnation - patience + 1)
            worst_idx, opp = centroid_opposition(pop, fitness, replace_frac=replace_frac, stagnation_boost=stagnation_boost)
            opp_mapped = np.array([map_func(ind) for ind in opp])
            opp_fit = np.array([obj_func(ind) for ind in opp_mapped], dtype=float)
            
            improved = opp_fit < fitness[worst_idx]
            pop[worst_idx[improved]] = opp_mapped[improved]
            fitness[worst_idx[improved]] = opp_fit[improved]
            
            stagnation = max(0, patience // 2)
                
    idx = np.argmin(fitness)
    if return_diversity:
        return float(fitness[idx]), pop[idx], best_curve, diversity_curve
    return float(fitness[idx]), pop[idx], best_curve
