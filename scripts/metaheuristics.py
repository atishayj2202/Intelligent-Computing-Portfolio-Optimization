import random
import numpy as np

def run_ga(obj_func, pop: np.ndarray, map_func, iters: int, 
           crossover_prob: float = 0.8, mutation_prob: float = 0.2):
    """
    Genetic Algorithm (GA) for portfolio optimization under constraints.
    Uses tournament selection, arithmetic crossover, and Gaussian mutation.
    """
    pop = pop.copy()
    n_pop, dim = pop.shape
    fitness = np.array([obj_func(ind) for ind in pop], dtype=float)
    best_curve = []
    
    for t in range(iters):
        new_pop = []
        
        # Elitism: retain best
        best_idx = np.argmin(fitness)
        new_pop.append(pop[best_idx].copy())
        
        while len(new_pop) < n_pop:
            # Tournament selection
            i1, i2 = random.sample(range(n_pop), 2)
            p1 = pop[i1] if fitness[i1] < fitness[i2] else pop[i2]
            
            j1, j2 = random.sample(range(n_pop), 2)
            p2 = pop[j1] if fitness[j1] < fitness[j2] else pop[j2]
            
            # Crossover
            if random.random() < crossover_prob:
                alpha = random.random()
                c1 = alpha * p1 + (1 - alpha) * p2
                c2 = alpha * p2 + (1 - alpha) * p1
            else:
                c1, c2 = p1.copy(), p2.copy()
                
            # Mutation
            for child in [c1, c2]:
                if len(new_pop) >= n_pop:
                    break
                if random.random() < mutation_prob:
                    child += np.random.normal(0, 0.1, size=dim)
                child = map_func(child)
                new_pop.append(child)
                
        pop = np.array(new_pop[:n_pop])
        fitness = np.array([obj_func(ind) for ind in pop], dtype=float)
        best_curve.append(float(np.min(fitness)))
        
    idx = np.argmin(fitness)
    return float(fitness[idx]), pop[idx], best_curve

def run_pso(obj_func, pop: np.ndarray, map_func, iters: int,
            w_start: float = 0.9, w_end: float = 0.4, c1: float = 1.5, c2: float = 1.5):
    """
    Particle Swarm Optimization (PSO) with inertia weight decay.
    """
    pop = pop.copy()
    n_pop, dim = pop.shape
    velocities = np.zeros_like(pop)
    
    fitness = np.array([obj_func(ind) for ind in pop], dtype=float)
    pbest = pop.copy()
    pbest_fit = fitness.copy()
    
    gbest_idx = np.argmin(pbest_fit)
    gbest = pbest[gbest_idx].copy()
    
    best_curve = []
    
    for t in range(iters):
        w_t = w_start - (w_start - w_end) * (t / max(1, iters - 1))
        
        for i in range(n_pop):
            r1, r2 = np.random.random(dim), np.random.random(dim)
            velocities[i] = (w_t * velocities[i] +
                             c1 * r1 * (pbest[i] - pop[i]) +
                             c2 * r2 * (gbest - pop[i]))
            
            pop[i] = map_func(pop[i] + velocities[i])
            f_i = obj_func(pop[i])
            fitness[i] = f_i
            
            if f_i < pbest_fit[i]:
                pbest[i] = pop[i].copy()
                pbest_fit[i] = f_i
                if f_i < obj_func(gbest):
                    gbest = pop[i].copy()
                    
        best_curve.append(float(np.min(pbest_fit)))
        
    return float(obj_func(gbest)), gbest, best_curve

def run_de(obj_func, pop: np.ndarray, map_func, iters: int,
           F: float = 0.8, CR: float = 0.9):
    """
    Differential Evolution (DE/rand/1/bin).
    """
    pop = pop.copy()
    n_pop, dim = pop.shape
    fitness = np.array([obj_func(ind) for ind in pop], dtype=float)
    best_curve = []
    
    for t in range(iters):
        for i in range(n_pop):
            idxs = [idx for idx in range(n_pop) if idx != i]
            a, b, c = pop[random.sample(idxs, 3)]
            
            mutant = a + F * (b - c)
            trial = np.where(np.random.random(dim) < CR, mutant, pop[i])
            
            # Ensure random mutation along at least one dimension
            j_rand = random.randrange(dim)
            trial[j_rand] = mutant[j_rand]
            
            trial = map_func(trial)
            f_trial = obj_func(trial)
            
            if f_trial < fitness[i]:
                pop[i] = trial
                fitness[i] = f_trial
                
        best_curve.append(float(np.min(fitness)))
        
    idx = np.argmin(fitness)
    return float(fitness[idx]), pop[idx], best_curve
