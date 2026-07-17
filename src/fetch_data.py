"""
features.py
------------
Time-series feature engineering for stock return prediction.

Target: next-day return  r_{t+1} = (Close_{t+1} - Close_t) / Close_t
Inputs: everything must be known at time t (no lookahead leakage).
"""
import numpy as np
import pandas as pd


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def bollinger_bandwidth(series: pd.Series, window: int = 20, n_std: int = 2):
    ma = series.rolling(window).mean()
    sd = series.rolling(window).std()
    upper, lower = ma + n_std * sd, ma - n_std * sd
    return (upper - lower) / ma


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """df must have columns Date, Open, High, Low, Close, Volume (single ticker), sorted by Date."""
    d = df.sort_values("Date").reset_index(drop=True).copy()
    close = d["Close"]

    d["return_1d"] = close.pct_change()
    for lag in [1, 2, 3, 5, 10]:
        d[f"lag_return_{lag}"] = d["return_1d"].shift(lag - 1) if lag == 1 else close.pct_change(lag).shift(1)

    # rolling volatility (realized, annualized) of daily returns
    for w in [5, 10, 20]:
        d[f"volatility_{w}d"] = d["return_1d"].rolling(w).std().shift(1) * np.sqrt(252)

    # moving averages & price-to-MA ratios
    for w in [5, 10, 20, 50]:
        ma = close.rolling(w).mean().shift(1)
        d[f"ma_{w}"] = ma
        d[f"price_to_ma_{w}"] = (close.shift(1) / ma) - 1

    # momentum
    d["momentum_10d"] = (close.shift(1) / close.shift(11)) - 1

    # volume features
    d["volume_change_1d"] = d["Volume"].pct_change().shift(1)
    d["volume_ma_10"] = d["Volume"].rolling(10).mean().shift(1)
    d["volume_ratio"] = (d["Volume"].shift(1) / d["volume_ma_10"])

    # technical indicators (computed on data through t, then shifted so nothing leaks)
    d["rsi_14"] = rsi(close, 14).shift(1)
    macd_line, macd_signal, macd_hist = macd(close)
    d["macd_hist"] = macd_hist.shift(1)
    d["bollinger_bw"] = bollinger_bandwidth(close).shift(1)

    # daily range / gap features
    d["high_low_range"] = ((d["High"] - d["Low"]) / close).shift(1)
    d["overnight_gap"] = ((d["Open"] - close.shift(1)) / close.shift(1))

    # TARGET: next day's return (this is what we predict, using only info through t)
    d["target_next_return"] = d["return_1d"].shift(-1)

    feature_cols = [c for c in d.columns if c not in
                    ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker",
                     "return_1d", "target_next_return"]]

    d = d.dropna(subset=feature_cols + ["target_next_return"]).reset_index(drop=True)
    return d, feature_cols


if __name__ == "__main__":
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "data" / "MSFT.csv"
    df = pd.read_csv(p, parse_dates=["Date"])
    feat_df, cols = build_feature_frame(df)
    print(feat_df[cols].describe().T[["mean", "std", "min", "max"]])
    print(f"\n{len(cols)} features, {len(feat_df)} usable rows")