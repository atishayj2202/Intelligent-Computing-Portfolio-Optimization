import numpy as np
import scipy.stats as stats
from itertools import combinations

def compute_psr(realized_sr: float, benchmark_sr: float, n_samples: int, skewness: float = 0.0, kurtosis: float = 3.0):
    """
    Probabilistic Sharpe Ratio (PSR) per Bailey and López de Prado (2012).
    """
    sr_std = np.sqrt((1 + 0.5 * realized_sr**2 - skewness * realized_sr + (kurtosis - 3) / 4.0 * realized_sr**2) / (n_samples - 1))
    z = (realized_sr - benchmark_sr) / sr_std
    return float(stats.norm.cdf(z))

def compute_dsr(realized_sr: float, sr_var: float, n_trials: int, n_samples: int, skewness: float = 0.0, kurtosis: float = 3.0):
    """
    Deflated Sharpe Ratio (DSR) per Bailey and López de Prado (2014).
    Corrects for multiple testing across n_trials strategy configurations.
    """
    euler_gamma = 0.5772156649
    e_max_sr = (1 - euler_gamma) * stats.norm.ppf(1 - 1.0 / n_trials) + euler_gamma * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    benchmark_sr = np.sqrt(sr_var) * e_max_sr
    
    return compute_psr(realized_sr, benchmark_sr, n_samples, skewness, kurtosis), benchmark_sr

def compute_min_btl(realized_sr: float, benchmark_sr: float, skewness: float = 0.0, kurtosis: float = 3.0, alpha: float = 0.05):
    """
    Minimum Backtest Length (MinBTL) in years required to reject false positive alpha.
    """
    z_alpha = stats.norm.ppf(1 - alpha)
    sr_diff = realized_sr - benchmark_sr
    if sr_diff <= 0:
        return np.nan
    min_days = 1 + (1 + 0.5 * realized_sr**2 - skewness * realized_sr + (kurtosis - 3) / 4.0 * realized_sr**2) * (z_alpha / sr_diff)**2
    return min_days / 252.0

def compute_pbo_cscv(matrix_returns: np.ndarray, S: int = 16):
    """
    Combinatorial Symmetric Cross-Validation (CSCV) for Probability of Backtest Overfitting (PBO).
    matrix_returns: array of shape (T_days, N_strategies)
    S: number of equal partitions (must be even)
    """
    T, N = matrix_returns.shape
    if S % 2 != 0:
        S += 1
        
    block_size = T // S
    trimmed_T = block_size * S
    rets = matrix_returns[:trimmed_T]
    
    # Reshape into S blocks: (S, block_size, N)
    blocks = rets.reshape(S, block_size, N)
    
    all_combos = list(combinations(range(S), S // 2))
    n_combos = len(all_combos)
    
    logits = []
    underperform_count = 0
    
    for train_indices in all_combos:
        test_indices = [i for i in range(S) if i not in train_indices]
        
        train_rets = blocks[list(train_indices)].reshape(-1, N)
        test_rets = blocks[test_indices].reshape(-1, N)
        
        # Annualized Sharpe on IS training split
        is_means = np.mean(train_rets, axis=0) * 252
        is_stds = np.std(train_rets, axis=0, ddof=1) * np.sqrt(252) + 1e-12
        is_sharpes = is_means / is_stds
        
        # Strategy that was optimal in-sample
        best_is_idx = np.argmax(is_sharpes)
        
        # Evaluate all strategies on OOS testing split
        oos_means = np.mean(test_rets, axis=0) * 252
        oos_stds = np.std(test_rets, axis=0, ddof=1) * np.sqrt(252) + 1e-12
        oos_sharpes = oos_means / oos_stds
        
        # Rank of the IS-best strategy in the OOS distribution (1-indexed)
        oos_rank = stats.rankdata(oos_sharpes)[best_is_idx]
        relative_rank = oos_rank / (N + 1.0)
        
        # Check if IS-best underperformed OOS median
        if relative_rank <= 0.5:
            underperform_count += 1
            
        logit = np.log(relative_rank / (1.0 - relative_rank + 1e-12) + 1e-12)
        logits.append(logit)
        
    pbo = underperform_count / float(n_combos)
    return pbo, np.array(logits)
