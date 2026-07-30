import random
import numpy as np
from scripts.utils import normalize_cap_cardinality

def classic_opposition(pop: np.ndarray, lb: float, ub: float) -> np.ndarray:
    return lb + ub - pop

def quasi_opposition(pop: np.ndarray, lb: float, ub: float) -> np.ndarray:
    m = (lb + ub) / 2.0
    x_op = lb + ub - pop
    r = np.random.uniform(0.0, 1.0, size=pop.shape)
    x_q = m + r * (x_op - m)
    return np.clip(x_q, lb, ub)

def portfolio_classic_opposition(pop: np.ndarray, cap: float = 0.20, K: int = 30) -> np.ndarray:
    opp = 1.0 - pop
    if len(opp.shape) == 1:
        return normalize_cap_cardinality(opp, cap, K)
    return np.array([normalize_cap_cardinality(ind, cap, K) for ind in opp])

def portfolio_reversal_opposition(pop: np.ndarray) -> np.ndarray:
    if len(pop.shape) == 1:
        sort_idx = np.argsort(pop)
        opp = np.zeros_like(pop)
        opp[sort_idx] = pop[sort_idx[::-1]]
        return opp
    
    opp_pop = np.zeros_like(pop)
    for idx, ind in enumerate(pop):
        sort_idx = np.argsort(ind)
        opp_pop[idx][sort_idx] = ind[sort_idx[::-1]]
    return opp_pop

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
        lb: float = None, ub: float = None, is_portfolio: bool = False):
    pop = pop.copy()
    fitness = np.array([obj_func(ind) for ind in pop], dtype=float)
    best_curve = []
    
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
        
    idx = np.argmin(fitness)
    return float(fitness[idx]), pop[idx], best_curve

def apply_obl_replacement(pop: np.ndarray, fitness: np.ndarray, obj_func, 
                          map_func, replace_frac: float, mode: str, 
                          lb: float = None, ub: float = None, cap: float = 0.20, K: int = 30):
    n = len(pop)
    k_replace = max(1, int(n * replace_frac))
    worst_idx = np.argsort(fitness)[-k_replace:]
    
    if mode == "classic":
        opp = classic_opposition(pop[worst_idx], lb, ub)
    elif mode == "quasi":
        opp = quasi_opposition(pop[worst_idx], lb, ub)
    elif mode == "portfolio_classic":
        opp = portfolio_classic_opposition(pop[worst_idx], cap, K)
    elif mode == "portfolio_reversal":
        opp = portfolio_reversal_opposition(pop[worst_idx])
    else:
        raise ValueError(f"Unknown OBL mode: {mode}")
        
    if mode in ["classic", "quasi"]:
        opp = np.clip(opp, lb, ub)
        
    opp_fit = np.array([obj_func(ind) for ind in opp], dtype=float)
    improved = opp_fit < fitness[worst_idx]
    
    pop[worst_idx[improved]] = opp[improved]
    fitness[worst_idx[improved]] = opp_fit[improved]
    return pop, fitness

def AOBL_SOS(
    obj_func,
    pop: np.ndarray,
    map_func,
    iters: int,
    lb: float = None,
    ub: float = None,
    is_portfolio: bool = False,
    init_obl: bool = True,
    obl_mode: str = "portfolio_reversal",
    replace_frac: float = 0.5,
    patience: int = 15,
    eps: float = 1e-12,
    p0: float = 0.20,
    pmax: float = 0.95,
    cap: float = 0.20,
    K: int = 30
):
    pop = pop.copy()
    fitness = np.array([obj_func(ind) for ind in pop], dtype=float)
    
    if init_obl:
        if obl_mode == "classic":
            opp = classic_opposition(pop, lb, ub)
        elif obl_mode == "quasi":
            opp = quasi_opposition(pop, lb, ub)
        elif obl_mode == "portfolio_classic":
            opp = portfolio_classic_opposition(pop, cap, K)
        elif obl_mode == "portfolio_reversal":
            opp = portfolio_reversal_opposition(pop)
            
        if obl_mode in ["classic", "quasi"]:
            opp = np.clip(opp, lb, ub)
            
        combined = np.vstack([pop, opp])
        combined_fit = np.array([obj_func(ind) for ind in combined], dtype=float)
        idx = np.argsort(combined_fit)[:len(pop)]
        pop = combined[idx]
        fitness = combined_fit[idx]
        
    best_curve = []
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
        
        if current_best < best_val - eps:
            best_val = current_best
            stagnation = 0
        else:
            stagnation += 1
            
        if stagnation >= patience:
            p = min(pmax, p0 + 0.05 * (stagnation - patience + 1))
            if random.random() < p:
                pop, fitness = apply_obl_replacement(
                    pop, fitness, obj_func, map_func,
                    replace_frac=replace_frac,
                    mode=obl_mode,
                    lb=lb, ub=ub, cap=cap, K=K
                )
                stagnation = max(0, patience // 2)
                
    idx = np.argmin(fitness)
    return float(fitness[idx]), pop[idx], best_curve
