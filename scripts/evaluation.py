import numpy as np
import pandas as pd

def obj_sharpe_drawdown(w, train_ret, rf_annual=0.0, lambda_dd=1.0):
    """
    Minimization Loss Function:
    g(w) = -SR(w) + lambda * |MDD(w)|
    Minimizing g(w) maximizes risk-adjusted return SR(w) penalized by maximum drawdown |MDD(w)|.
    """
    port_ret = np.dot(train_ret, w)
    
    ann_ret = np.mean(port_ret) * 252 - rf_annual
    ann_vol = np.std(port_ret, ddof=1) * np.sqrt(252) + 1e-12
    sharpe = ann_ret / ann_vol
    
    cum_ret = np.cumprod(1 + port_ret)
    running_max = np.maximum.accumulate(cum_ret)
    drawdowns = (cum_ret - running_max) / running_max
    max_dd = np.min(drawdowns)
    
    penalty = lambda_dd * abs(max_dd)
    # Loss g(w) to MINIMIZE:
    return -sharpe + penalty

def compute_net_metrics_fixed_bps(w, test_ret_df, cost_bps=10, rebal_freq=21, rf_annual=0.0):
    returns = test_ret_df.values
    n_days, n_assets = returns.shape
    
    port_val = 1.0
    val_history = []
    
    current_w = np.copy(w)
    shares = (port_val * current_w) 
    
    # Initial cost
    trade_cost = np.sum(np.abs(current_w)) * (cost_bps / 10000.0)
    port_val -= trade_cost
    
    turnovers = []
    
    for t in range(n_days):
        day_ret = returns[t]
        shares = shares * (1 + day_ret)
        port_val = np.sum(shares)
        
        if (t + 1) % rebal_freq == 0 and t < n_days - 1:
            target_holdings = port_val * w
            trades = np.abs(target_holdings - shares)
            cost = np.sum(trades) * (cost_bps / 10000.0)
            port_val -= cost
            shares = port_val * w
            
            turnovers.append(np.sum(trades) / (port_val + cost))
            
        val_history.append(port_val)
        
    val_history = np.array(val_history)
    net_returns = np.insert(np.diff(val_history) / val_history[:-1], 0, val_history[0]-1)
    
    ann_ret = np.mean(net_returns) * 252 - rf_annual
    ann_vol = np.std(net_returns, ddof=1) * np.sqrt(252) + 1e-12
    sharpe = ann_ret / ann_vol
    
    downside = net_returns[net_returns < 0]
    sortino = (ann_ret) / (np.std(downside, ddof=1) * np.sqrt(252) + 1e-12) if len(downside) > 0 else np.nan
    
    cum_ret = np.cumprod(1 + net_returns)
    running_max = np.maximum.accumulate(cum_ret)
    drawdowns = (cum_ret - running_max) / running_max
    max_dd = np.min(drawdowns)
    
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan
    cvar_95 = np.percentile(net_returns, 5)
    
    return {
        'ann_return': ann_ret,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_dd,
        'calmar': calmar,
        'cvar_95': cvar_95,
        'avg_turnover': np.mean(turnovers) if len(turnovers) > 0 else 0.0,
        'net_returns': net_returns
    }

def compute_net_metrics_almgren_chriss(w, test_ret_df, test_vol_df, train_ret_df, aum=100_000_000, rebal_freq=21, rf_annual=0.02, Y=0.10):
    returns = test_ret_df.values
    adv_dollars = test_vol_df.values
    
    # Fallback to mean ADV if volume is missing
    col_means = np.nanmean(adv_dollars, axis=0)
    inds = np.where(np.isnan(adv_dollars) | (adv_dollars == 0))
    adv_dollars[inds] = np.take(col_means, inds[1])
    
    # Historical daily volatility
    sigma = np.std(train_ret_df.values, axis=0, ddof=1)
    
    n_days, n_assets = returns.shape
    
    port_val = aum
    val_history = []
    
    current_w = np.copy(w)
    shares_val = port_val * current_w
    
    # Initial trades
    trades_dollar = np.abs(shares_val)
    initial_adv = adv_dollars[0]
    
    slippage_frac = Y * sigma * np.sqrt(trades_dollar / (initial_adv + 1e-9))
    slippage_dollar = np.sum(slippage_frac * trades_dollar)
    
    port_val -= slippage_dollar
    shares_val = port_val * current_w
    
    for t in range(n_days):
        day_ret = returns[t]
        shares_val = shares_val * (1 + day_ret)
        port_val = np.sum(shares_val)
        
        if (t + 1) % rebal_freq == 0 and t < n_days - 1:
            target_val = port_val * w
            trades = np.abs(target_val - shares_val)
            
            day_adv = adv_dollars[t]
            slippage_f = Y * sigma * np.sqrt(trades / (day_adv + 1e-9))
            slippage_f = np.clip(slippage_f, 0, 0.05)
            
            cost = np.sum(slippage_f * trades)
            port_val -= cost
            shares_val = port_val * w
            
        val_history.append(port_val)
        
    val_history = np.array(val_history)
    net_returns = np.insert(np.diff(val_history) / val_history[:-1], 0, val_history[0]/aum - 1)
    
    ann_ret = np.mean(net_returns) * 252 - rf_annual
    ann_vol = np.std(net_returns, ddof=1) * np.sqrt(252) + 1e-12
    sharpe = ann_ret / ann_vol
    
    downside = net_returns[net_returns < 0]
    sortino = (ann_ret) / (np.std(downside, ddof=1) * np.sqrt(252) + 1e-12) if len(downside) > 0 else np.nan
    
    cum_ret = np.cumprod(1 + net_returns)
    running_max = np.maximum.accumulate(cum_ret)
    drawdowns = (cum_ret - running_max) / running_max
    max_dd = np.min(drawdowns)
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan
    cvar_95 = np.percentile(net_returns, 5)
    
    return {
        'ann_return': ann_ret,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_dd,
        'calmar': calmar,
        'cvar_95': cvar_95,
        'net_returns': net_returns
    }
