import random
import numpy as np
import pandas as pd

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

def load_data(filepath: str, n_stocks: int = 179, seed: int = 42):
    set_seed(seed)
    # Generate synthetic S&P 500 daily returns to simulate the required time period
    # 2012-01-01 to 2025-01-31 is approx 3290 trading days
    dates = pd.date_range(start='2012-01-01', end='2025-01-31', freq='B')
    
    # 80% have positive drift, 20% flat or negative
    drifts = np.random.uniform(-0.05, 0.15, n_stocks) / 252.0
    vols = np.random.uniform(0.15, 0.40, n_stocks) / np.sqrt(252.0)
    
    # Create correlation matrix roughly 0.3 average correlation
    corr = np.full((n_stocks, n_stocks), 0.3)
    np.fill_diagonal(corr, 1.0)
    
    # Ensure positive semi-definite
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    eigenvalues[eigenvalues < 0] = 0
    corr = np.dot(eigenvectors, np.dot(np.diag(eigenvalues), eigenvectors.T))
    
    # Covariance
    cov = np.outer(vols, vols) * corr
    
    # Generate Multivariate Normal Returns
    returns = np.random.multivariate_normal(drifts, cov, len(dates))
    
    df = pd.DataFrame(returns, index=dates, columns=[f"TICKER_{i}" for i in range(1, n_stocks+1)])
    
    # Save the synthetic data to CSV so it can be shared with the paper
    df.to_csv(filepath)
    
    train_df = df.loc['2012-01-01':'2022-12-31']
    test_df = df.loc['2023-01-01':'2025-01-31']
    
    mu = train_df.mean().values * 252
    cov_train = train_df.cov().values * 252
    
    return list(df.columns), mu, cov_train, train_df, test_df
