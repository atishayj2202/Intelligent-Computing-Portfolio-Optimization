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

def load_data(filepath: str, n_stocks: int = 150, seed: int = 42):
    import os
    set_seed(seed)
    
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    else:
        print(f"Data not found at {filepath}. Downloading via yfinance...")
        import yfinance as yf
        
        # A subset of S&P 500 tickers (liquid names)
        # We use a static list to ensure reproducibility and stability
        tickers = [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V',
            'PG', 'UNH', 'HD', 'MA', 'DIS', 'BAC', 'XOM', 'CVX', 'ABBV', 'PEP',
            'KO', 'COST', 'MRK', 'TMO', 'AVGO', 'MCD', 'WMT', 'CSCO', 'ACN', 'ABT',
            'DHR', 'LIN', 'NKE', 'PFE', 'NEE', 'ADBE', 'TXN', 'VZ', 'CMCSA', 'PM',
            'NFLX', 'CRM', 'QCOM', 'ORCL', 'WFC', 'AMD', 'HON', 'UPS', 'INTC', 'LOW',
            'RTX', 'UNP', 'T', 'COP', 'SPGI', 'MDT', 'BMY', 'INTU', 'GS', 'CVS',
            'AMAT', 'BLK', 'SYK', 'BA', 'NOW', 'PLD', 'EL', 'CB', 'GE', 'DE',
            'AXP', 'ISRG', 'MMC', 'ADI', 'LMT', 'BKNG', 'SYY', 'MDLZ', 'TGT', 'GILD',
            'C', 'ZTS', 'MO', 'BDX', 'PNC', 'CSX', 'DUK', 'SO', 'TJX', 'REGN',
            'CL', 'ICE', 'CME', 'BSX', 'WM', 'EW', 'VRTX', 'NOC', 'ATVI', 'FISV',
            'ITW', 'APD', 'EOG', 'MU', 'AON', 'KLAC', 'SHW', 'ETN', 'GPN', 'GD',
            'F', 'GM', 'NSC', 'MCO', 'CHTR', 'HUM', 'SNPS', 'CDNS', 'KMB', 'AEP',
            'ORLY', 'PSA', 'NXPI', 'MCHP', 'SRE', 'ADSK', 'AIG', 'D', 'EMR', 'CTSH',
            'EXC', 'ROST', 'BIIB', 'LRCX', 'KMI', 'PH', 'TEL', 'IDXX', 'MAR', 'SLB'
        ]
        
        # Only take up to n_stocks
        tickers = tickers[:n_stocks]
        
        # Download from Jan 1 2012 to Jan 31 2025
        prices = pd.DataFrame()
        for ticker in tickers:
            try:
                # Need to squeeze because yf 1.2 returns a dataframe with MultiIndex columns for single ticker sometimes, or single level.
                # Just get the 'Close' column.
                hist = yf.download(ticker, start="2012-01-01", end="2025-01-31", progress=False)
                if 'Close' in hist:
                    if isinstance(hist['Close'], pd.DataFrame):
                        prices[ticker] = hist['Close'].iloc[:, 0]
                    else:
                        prices[ticker] = hist['Close']
            except Exception:
                pass
        
        # Calculate daily returns
        df = prices.pct_change()
        
        # Drop first row which is all NaN due to pct_change
        df = df.iloc[1:]
        
        # Drop columns (tickers) that have ANY missing values to prevent NaNs in optimization
        df = df.dropna(axis=1)
        
        # Save for future use
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath)
    
    # Sort index just in case
    df = df.sort_index()
    
    train_df = df.loc['2012-01-01':'2022-12-31']
    test_df = df.loc['2023-01-01':'2025-01-31']
    
    mu = train_df.mean().values * 252
    cov_train = train_df.cov().values * 252
    
    return list(df.columns), mu, cov_train, train_df, test_df
