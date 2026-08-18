import numpy as np
import pandas as pd

def get_walk_forward_windows():
    """
    Returns 7 expanding walk-forward windows:
    IS: 2012 to Y_end
    OOS: Y_end+1 (1 full year)
    """
    windows = []
    for y_end in range(2017, 2024):
        train_start = "2012-01-01"
        train_end = f"{y_end}-12-31"
        test_start = f"{y_end+1}-01-01"
        test_end = f"{y_end+1}-12-31"
        windows.append({
            'window_id': y_end - 2016,
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'test_year': y_end + 1
        })
    return windows

def filter_point_in_time_universe(df_ret, df_vol, train_start, train_end, min_history_days=252):
    """
    Filters investable assets at rebalance time using ONLY historical data available up to train_end.
    Prevents survivorship bias by ensuring selection relies exclusively on past data.
    """
    train_ret = df_ret.loc[train_start:train_end]
    train_vol = df_vol.loc[train_start:train_end]
    
    # Require asset to have valid returns for at least min_history_days in training window
    valid_counts = train_ret.notnull().sum(axis=0)
    history_mask = valid_counts >= min_history_days
    
    # Require positive average daily trading volume during the training period
    mean_vol = train_vol.mean(axis=0)
    vol_mask = mean_vol > 0
    
    # Require asset to have non-null return on the last available training day (active asset)
    active_mask = train_ret.iloc[-1].notnull()
    
    eligible = history_mask & vol_mask & active_mask
    eligible_tickers = df_ret.columns[eligible].tolist()
    
    return eligible_tickers
