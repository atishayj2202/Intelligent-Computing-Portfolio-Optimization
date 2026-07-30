import numpy as np
import pandas as pd
from typing import Dict, Any

def obj_sharpe_drawdown(w: np.ndarray, train_returns: np.ndarray, rf_annual: float = 0.02) -> float:
    """
    Risk-aware objective function for metaheuristics.
    Evaluates historical path-dependent Max Drawdown and Sharpe Ratio.
    Goal is to MAXIMIZE: Sharpe / (1 + Max_Drawdown)
    We return the negative for minimization.
    """
    port_ret = np.dot(train_returns, w)
    
    ann_ret = np.mean(port_ret) * 252
    ann_vol = np.std(port_ret, ddof=1) * np.sqrt(252) + 1e-12
    sharpe = (ann_ret - rf_annual) / ann_vol
    
    # Calculate Max Drawdown
    cum_returns = np.cumprod(1 + port_ret)
    peaks = np.maximum.accumulate(cum_returns)
    drawdowns = (peaks - cum_returns) / peaks
    max_dd = np.max(drawdowns)
    
    # Penalize portfolios with severe drawdowns
    # If sharpe is negative, we just want to minimize it further
    if sharpe > 0:
        score = sharpe / (1.0 + max_dd)
    else:
        score = sharpe * (1.0 + max_dd)
        
    return -score

def compute_net_metrics(weights: np.ndarray, test_ret_df: pd.DataFrame, 
                        cost_bps: float = 10.0, rebal_freq: int = 21, 
                        rf_annual: float = 0.02) -> Dict[str, Any]:
    """
    Computes out-of-sample portfolio performance taking into account 
    rebalancing turnover and transaction costs.
    """
    n_days = test_ret_df.shape[0]
    c = cost_bps / 10000.0
    
    current_w = weights.copy()
    daily_net_returns = []
    turnover_list = []
    
    for t in range(n_days):
        day_ret = test_ret_df.iloc[t].values
        gross_r = np.dot(current_w, day_ret)
        
        if t > 0 and t % rebal_freq == 0:
            target_w = weights.copy()
            turnover = np.sum(np.abs(target_w - current_w))
            cost = turnover * c
            net_r = gross_r - cost
            turnover_list.append(turnover)
            current_w = target_w.copy()
        else:
            net_r = gross_r
            current_w = current_w * (1 + day_ret)
            s = np.sum(current_w)
            if s > 0:
                current_w = current_w / s
                
        daily_net_returns.append(net_r)
        
    daily_net_returns = np.array(daily_net_returns)
    cum_ret = np.cumprod(1 + daily_net_returns)
    
    ann_return = np.mean(daily_net_returns) * 252
    ann_vol = np.std(daily_net_returns, ddof=1) * np.sqrt(252) + 1e-12
    sharpe = (ann_return - rf_annual) / ann_vol
    
    downside = daily_net_returns[daily_net_returns < 0]
    downside_std = np.std(downside, ddof=1) * np.sqrt(252) if len(downside) > 0 else 1e-12
    sortino = (ann_return - rf_annual) / downside_std
    
    peaks = np.maximum.accumulate(cum_ret)
    drawdowns = (peaks - cum_ret) / peaks
    max_dd = np.max(drawdowns)
    
    avg_turnover = np.mean(turnover_list) if len(turnover_list) > 0 else 0.0
    
    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "avg_turnover": avg_turnover,
        "cum_returns": cum_ret,
        "daily_returns": daily_net_returns
    }
