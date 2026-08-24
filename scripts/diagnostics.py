
import io
import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import statsmodels.api as sm

from scripts.universe import get_walk_forward_windows
from scripts.utils import load_data_for_window
from scripts.evaluation import compute_net_metrics_fixed_bps, compute_net_metrics_almgren_chriss, summarize_chained_returns


FRENCH_FF3 = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
FRENCH_MOM = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"


def _parse_french_daily(raw_bytes, skip_rows=4):
    text = raw_bytes.decode("latin-1")
    lines = []
    for line in text.splitlines():
        if line.strip().startswith("Copyright"):
            break
        lines.append(line)
    df = pd.read_csv(io.StringIO("\n".join(lines)), skiprows=skip_rows)
    df = df.dropna(how="all")
    date_col = df.columns[0]
    df = df[df[date_col].astype(str).str.match(r"^\s*\d{8}\s*$", na=False)].copy()
    df.index = pd.to_datetime(df[date_col].astype(str).str.strip(), format="%Y%m%d")
    df = df.drop(columns=[date_col])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Ken French daily files are in percent
    return df / 100.0


def _http_get(url):
    import ssl
    ctx = ssl.create_default_context()
    try:
        return urllib.request.urlopen(url, timeout=60, context=ctx).read()
    except Exception:
        ctx = ssl._create_unverified_context()
        return urllib.request.urlopen(url, timeout=60, context=ctx).read()

def download_ken_french_daily(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    out = os.path.join(data_dir, "ken_french_carhart_daily.csv")
    ff_zip = _http_get(FRENCH_FF3)
    mom_zip = _http_get(FRENCH_MOM)
    with zipfile.ZipFile(io.BytesIO(ff_zip)) as zf:
        ff_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        ff = _parse_french_daily(zf.read(ff_name), skip_rows=4)
    with zipfile.ZipFile(io.BytesIO(mom_zip)) as zf:
        mom_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        mom = _parse_french_daily(zf.read(mom_name), skip_rows=13)
    mom_col = [c for c in mom.columns if "mom" in c.lower().replace(" ", "")][0]
    ff.columns = [c.strip() for c in ff.columns]
    merged = ff.join(mom[[mom_col]].rename(columns={mom_col: "Mom"}), how="inner")
    merged = merged.rename(columns={"Mkt-RF": "MKT", "SMB": "SMB", "HML": "HML", "RF": "RF", "Mom": "MOM"})
    merged.to_csv(out)
    return merged


def load_ken_french_daily(data_dir="data"):
    path = os.path.join(data_dir, "ken_french_carhart_daily.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df
    return download_ken_french_daily(data_dir)


def run_carhart_attribution(portfolio_returns: pd.Series, output_dir="results", data_dir="data"):
    factors = load_ken_french_daily(data_dir)
    aligned = pd.concat([portfolio_returns.rename("RP"), factors], axis=1, join="inner").dropna()
    y = aligned["RP"] - aligned["RF"]
    X = sm.add_constant(aligned[["MKT", "SMB", "HML", "MOM"]])
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    rows = []
    names = ["const", "MKT", "SMB", "HML", "MOM"]
    labels = ["Alpha (daily)", "MKT", "SMB", "HML", "MOM"]
    for name, label in zip(names, labels):
        coef = float(model.params[name])
        tval = float(model.tvalues[name])
        pval = float(model.pvalues[name])
        if name == "const":
            coef_ann = coef * 252.0
            rows.append({
                "Factor": "Alpha (ann.)",
                "Coefficient": coef_ann,
                "t-stat": tval,
                "p-value": pval,
            })
        else:
            rows.append({
                "Factor": label,
                "Coefficient": coef,
                "t-stat": tval,
                "p-value": pval,
            })
    rows.append({"Factor": "Adj. R2", "Coefficient": float(model.rsquared_adj), "t-stat": np.nan, "p-value": np.nan})
    rows.append({"Factor": "N", "Coefficient": float(int(model.nobs)), "t-stat": np.nan, "p-value": np.nan})
    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "carhart_attribution.csv"), index=False)

    # 2022 subsample for the failure-year discussion
    y2022 = aligned.loc["2022-01-01":"2022-12-31"]
    if len(y2022) >= 50:
        y2 = y2022["RP"] - y2022["RF"]
        X2 = sm.add_constant(y2022[["MKT", "SMB", "HML", "MOM"]])
        m2 = sm.OLS(y2, X2).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        sub = pd.DataFrame({
            "Factor": ["Alpha (ann.)", "MKT", "SMB", "HML", "MOM", "Adj. R2"],
            "Coefficient": [m2.params["const"] * 252.0, m2.params["MKT"], m2.params["SMB"], m2.params["HML"], m2.params["MOM"], m2.rsquared_adj],
            "t-stat": [m2.tvalues["const"], m2.tvalues["MKT"], m2.tvalues["SMB"], m2.tvalues["HML"], m2.tvalues["MOM"], np.nan],
            "p-value": [m2.pvalues["const"], m2.pvalues["MKT"], m2.pvalues["SMB"], m2.pvalues["HML"], m2.pvalues["MOM"], np.nan],
        })
        sub.to_csv(os.path.join(output_dir, "carhart_attribution_2022.csv"), index=False)
    return df


def stationary_block_bootstrap_delta_sharpe(r1, r2, block_mean=21, n_boot=2000, rf_annual=0.02, seed=42):
    rng = np.random.default_rng(seed)
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    T = len(r1)
    p = 1.0 / float(block_mean)

    def ann_sharpe(r):
        return ((np.mean(r) * 252) - rf_annual) / (np.std(r, ddof=1) * np.sqrt(252) + 1e-12)

    def sample_indices():
        idx = []
        while len(idx) < T:
            start = int(rng.integers(0, T))
            L = 1
            while rng.random() > p:
                L += 1
            for k in range(L):
                idx.append((start + k) % T)
                if len(idx) >= T:
                    break
        return np.array(idx[:T])

    obs = ann_sharpe(r1) - ann_sharpe(r2)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        ix = sample_indices()
        boots[b] = ann_sharpe(r1[ix]) - ann_sharpe(r2[ix])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    # two-sided percentile p-value around 0
    p_val = 2.0 * min(np.mean(boots >= 0.0), np.mean(boots <= 0.0))
    p_val = min(1.0, p_val)
    return {
        "delta_sharpe": obs,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_bootstrap": float(p_val),
        "n_boot": n_boot,
        "block_mean": block_mean,
    }


def run_cost_sensitivity(selected_weights_csv, data_path, output_dir="results", rf_annual=0.02):
    weights = pd.read_csv(selected_weights_csv)
    aobl = weights[weights["Algorithm"] == "AOBL-SOS (Proposed)"]
    bps_grid = [0, 5, 10, 20, 50]
    windows = get_walk_forward_windows()
    rows = []
    for bps in bps_grid:
        chained = []
        for win in windows:
            tickers, mu, cov, train_ret, test_ret, test_vol = load_data_for_window(
                data_path, win["train_start"], win["train_end"], win["test_start"], win["test_end"], cap=0.20, K=30
            )
            wdf = aobl[aobl["Window"] == win["window_id"]].set_index("Ticker")
            w = np.array([float(wdf.loc[t, "Weight"]) if t in wdf.index else 0.0 for t in tickers])
            s = w.sum()
            if s <= 0:
                raise RuntimeError(f"No AOBL weights for window {win['window_id']}")
            w = w / s
            res = compute_net_metrics_fixed_bps(w, test_ret, cost_bps=bps, rebal_freq=None, rf_annual=rf_annual)
            chained.extend(res["net_returns"])
        summ = summarize_chained_returns(chained, rf_annual=rf_annual)
        rows.append({
            "Cost_bps": bps,
            "Net Sharpe": summ["sharpe"],
            "Net CAGR": summ["ann_return"],
            "Max Drawdown": summ["max_drawdown"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "transaction_cost_sensitivity.csv"), index=False)
    return df


def run_capacity_from_weights(selected_weights_csv, data_path, output_dir="results", rf_annual=0.02):
    weights = pd.read_csv(selected_weights_csv)
    aobl = weights[weights["Algorithm"] == "AOBL-SOS (Proposed)"]
    windows = get_walk_forward_windows()
    payloads = []
    for win in windows:
        tickers, mu, cov, train_ret, test_ret, test_vol = load_data_for_window(
            data_path, win["train_start"], win["train_end"], win["test_start"], win["test_end"], cap=0.20, K=30
        )
        wdf = aobl[aobl["Window"] == win["window_id"]].set_index("Ticker")
        w = np.array([float(wdf.loc[t, "Weight"]) if t in wdf.index else 0.0 for t in tickers])
        w = w / w.sum()
        payloads.append((w, test_ret, test_vol, train_ret))

    aums = [10_000_000, 25_000_000, 50_000_000, 100_000_000, 250_000_000, 500_000_000, 1_000_000_000, 2_500_000_000, 5_000_000_000]
    rows = []
    cagr_10m = None
    for aum in aums:
        chained = []
        for w, test_ret, test_vol, train_ret in payloads:
            res = compute_net_metrics_almgren_chriss(w, test_ret, test_vol, train_ret, aum=aum, rf_annual=rf_annual)
            chained.extend(res["net_returns"])
        summ = summarize_chained_returns(chained, rf_annual=rf_annual)
        if cagr_10m is None:
            cagr_10m = summ["ann_return"]
        shortfall = (cagr_10m - summ["ann_return"]) * 100.0
        label = f"${aum/1e6:.0f}M" if aum < 1e9 else f"${aum/1e9:.1f}B"
        rows.append({
            "AUM": label,
            "AUM_val": aum,
            "Net Sharpe": summ["sharpe"],
            "Net CAGR": summ["ann_return"] * 100.0,
            "Max Drawdown": summ["max_drawdown"] * 100.0,
            "Shortfall_vs_10M_pp": 0.0 if aum == aums[0] else shortfall,
            "Execution_drag_pct_NAV": 100.0 * (1.0 - (1.0 + summ["ann_return"]) / (1.0 + cagr_10m)) if aum != aums[0] else 0.0,
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "aum_capacity_curve.csv"), index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(df["AUM_val"] / 1e6, df["Net Sharpe"], "o-", color="#D85A30", linewidth=2)
    plt.xscale("log")
    plt.title("AUM Capacity Curve (Almgren-Chriss restoration impact)")
    plt.xlabel("Portfolio AUM ($ Millions, Log Scale)")
    plt.ylabel("Net Out-of-Sample Sharpe Ratio")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "aum_capacity_curve.png"), dpi=200)
    plt.close()
    return df
