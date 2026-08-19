import os
import random
import warnings
import numpy as np
import pandas as pd
from scripts.universe import filter_point_in_time_universe

warnings.filterwarnings('ignore', category=UserWarning)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.maximum(w, 0.0)
    s = np.sum(w)
    return w / s if s > 0 else np.ones_like(w) / len(w)

def apply_cap(w: np.ndarray, cap: float) -> np.ndarray:
    w = np.maximum(w, 0.0)
    for _ in range(200):
        excess = np.maximum(w - cap, 0.0)
        if excess.sum() < 1e-10:
            break
        w = np.minimum(w, cap)
        uncapped_mask = w < cap
        uncapped_sum = w[uncapped_mask].sum()
        if uncapped_sum > 1e-10:
            w[uncapped_mask] += excess.sum() * (w[uncapped_mask] / uncapped_sum)
        else:
            w[:] = cap
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / len(w)

def enforce_cardinality(w: np.ndarray, K: int) -> np.ndarray:
    if K >= len(w):
        return w
    top_k_indices = np.argsort(w)[-K:]
    w_new = np.zeros_like(w)
    w_new[top_k_indices] = w[top_k_indices]
    return w_new

def normalize_cap_cardinality(w: np.ndarray, cap: float = 0.20, K: int = 30) -> np.ndarray:
    w_card = enforce_cardinality(np.maximum(w, 0.0), K)
    return apply_cap(normalize_weights(w_card), cap)

def population_diversity(pop: np.ndarray) -> float:
    """
    Computes pairwise L2 population diversity:
    D_t = 2 / (N * (N - 1)) * \sum_{i < j} ||w_i - w_j||_2
    """
    N = len(pop)
    if N <= 1:
        return 0.0
    total_dist = 0.0
    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            total_dist += np.linalg.norm(pop[i] - pop[j])
            count += 1
    return float(total_dist / count) if count > 0 else 0.0

def load_raw_data(filepath: str = "data/sp500_daily.csv"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at {filepath}")
    df_all = pd.read_csv(filepath, index_col=[0, 1])
    df_all.index = df_all.index.set_levels([pd.to_datetime(df_all.index.levels[0]), df_all.index.levels[1]])
    df = df_all.unstack(level=1).sort_index()
    df_ret = df['Return']
    df_vol = df['Volume']
    return df_ret, df_vol

def load_data_for_window(filepath: str, train_start: str, train_end: str, test_start: str, test_end: str, cap: float = 0.20, K: int = 30):
    df_ret, df_vol = load_raw_data(filepath)
    
    # Point-in-time universe selection to eliminate survivorship bias
    eligible_tickers = filter_point_in_time_universe(df_ret, df_vol, train_start, train_end)
    
    train_ret = df_ret.loc[train_start:train_end, eligible_tickers]
    test_ret = df_ret.loc[test_start:test_end, eligible_tickers]
    test_vol = df_vol.loc[test_start:test_end, eligible_tickers]
    
    # Clean any remaining NaNs in train by forward fill then 0
    train_ret = train_ret.ffill().fillna(0.0)
    test_ret = test_ret.ffill().fillna(0.0)
    test_vol = test_vol.ffill().fillna(0.0)
    
    mu = train_ret.mean().values * 252
    cov_train = train_ret.cov().values * 252
    
    return eligible_tickers, mu, cov_train, train_ret, test_ret, test_vol

def load_data(filepath: str, n_stocks: int = 150, seed: int = 42):
    return load_data_for_window(filepath, '2012-01-01', '2022-12-31', '2023-01-01', '2025-01-31')
