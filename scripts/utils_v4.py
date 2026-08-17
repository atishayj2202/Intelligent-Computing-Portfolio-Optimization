import os
import random
import numpy as np
import pandas as pd
import yfinance as yf

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
    set_seed(seed)
    
    if os.path.exists(filepath):
        df_all = pd.read_csv(filepath, index_col=[0, 1], parse_dates=True)
        df = df_all.unstack(level=1)
    else:
        print(f"Data not found at {filepath}. Downloading via yfinance...")
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
        tickers = tickers[:n_stocks]
        
        prices_dict = {}
        volumes_dict = {}
        for ticker in tickers:
            try:
                hist = yf.download(ticker, start="2012-01-01", end="2025-01-31", progress=False)
                if 'Close' in hist and 'Volume' in hist:
                    c = hist['Close'].iloc[:, 0] if isinstance(hist['Close'], pd.DataFrame) else hist['Close']
                    v = hist['Volume'].iloc[:, 0] if isinstance(hist['Volume'], pd.DataFrame) else hist['Volume']
                    prices_dict[ticker] = c
                    volumes_dict[ticker] = v * c
            except Exception:
                pass
        
        prices = pd.DataFrame(prices_dict)
        volumes = pd.DataFrame(volumes_dict)
        
        df_ret = prices.pct_change().iloc[1:]
        volumes = volumes.iloc[1:]
        
        df_ret = df_ret.dropna(axis=1)
        valid_tickers = df_ret.columns
        volumes = volumes[valid_tickers]
        
        # Merge into multiindex
        df_ret_stacked = df_ret.stack().rename("Return")
        volumes_stacked = volumes.stack().rename("Volume")
        
        df = pd.concat([df_ret_stacked, volumes_stacked], axis=1)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath)
        df = df.unstack(level=1)
        
    df = df.sort_index()
    df_ret = df['Return']
    df_vol = df['Volume']
    
    train_ret = df_ret.loc['2012-01-01':'2022-12-31']
    test_ret = df_ret.loc['2023-01-01':'2025-01-31']
    test_vol = df_vol.loc['2023-01-01':'2025-01-31']
    
    mu = train_ret.mean().values * 252
    cov_train = train_ret.cov().values * 252
    
    return list(df_ret.columns), mu, cov_train, train_ret, test_ret, test_vol
