import pandas as pd
from scripts.utils import load_data

data_path = "data/sp500_daily.csv"
tickers, mu, cov, train_ret, test_ret = load_data(data_path, n_stocks=179, seed=42)

df_full = pd.read_csv(data_path, index_col=0, parse_dates=True)
returns_full = df_full.pct_change().dropna()

test_slice = returns_full.loc["2023-01-01":"2025-12-31"]

print("test_ret shape:", test_ret.shape)
print("test_slice shape:", test_slice.shape)

print("Are they identical?", test_ret.equals(test_slice))
