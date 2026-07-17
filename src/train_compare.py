"""
train_compare.py
-----------------
Trains SVR, Gradient Boosting, and an LSTM on engineered features to predict
next-day stock returns, then compares them on RMSE / MAE / R2.

Uses a strict time-ordered train/val/test split (no shuffling -- this is a
time series, shuffling would leak the future into training).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from features import build_feature_frame

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

SEQ_LEN = 20          # LSTM lookback window (days)
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15        # remaining 0.15 is test


def time_split(n, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return n_train, n_train + n_val


def make_sequences(X, y, seq_len):
    xs, ys = [], []
    for i in range(seq_len, len(X)):
        xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(xs), np.array(ys)


class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=num_layers,
                             batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


def train_lstm(X_train, y_train, X_val, y_val, n_features, epochs=60, lr=1e-3, patience=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMRegressor(n_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    Xt = torch.tensor(X_train, dtype=torch.float32).to(device)
    yt = torch.tensor(y_train, dtype=torch.float32).to(device)
    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
    yv = torch.tensor(y_val, dtype=torch.float32).to(device)

    best_val, best_state, wait = np.inf, None, 0
    history = []
    batch_size = 32

    for epoch in range(epochs):
        model.train()
        perm_starts = list(range(0, len(Xt), batch_size))
        epoch_loss = 0.0
        for start in perm_starts:
            xb, yb = Xt[start:start + batch_size], yt[start:start + batch_size]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(Xt)

        model.eval()
        with torch.no_grad():
            val_pred = model(Xv)
            val_loss = loss_fn(val_pred, yv).item()
        history.append({"epoch": epoch + 1, "train_loss": epoch_loss, "val_loss": val_loss})

        if val_loss < best_val - 1e-8:
            best_val, best_state, wait = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model, history, device


def metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "directional_accuracy": float(np.mean(np.sign(y_true) == np.sign(y_pred))),
    }


def run_for_ticker(ticker: str):
    print(f"\n=== {ticker} ===")
    df = pd.read_csv(DATA_DIR / f"{ticker}.csv", parse_dates=["Date"])
    feat_df, feature_cols = build_feature_frame(df)

    X_raw = feat_df[feature_cols].values
    y_raw = feat_df["target_next_return"].values
    dates = feat_df["Date"].values

    n_train, n_val_end = time_split(len(X_raw))

    scaler_X = StandardScaler().fit(X_raw[:n_train])
    X_scaled = scaler_X.transform(X_raw)

    X_train, y_train = X_scaled[:n_train], y_raw[:n_train]
    X_val, y_val = X_scaled[n_train:n_val_end], y_raw[n_train:n_val_end]
    X_test, y_test = X_scaled[n_val_end:], y_raw[n_val_end:]
    test_dates = dates[n_val_end:]

    results = {}
    predictions = {"dates": pd.to_datetime(test_dates).astype(str).tolist(), "actual": y_test.tolist()}

    # ---------------- SVR ----------------
    t0 = time.time()
    svr = SVR(kernel="rbf", C=1.0, epsilon=0.001, gamma="scale")
    svr.fit(X_train, y_train)
    pred_svr = svr.predict(X_test)
    results["SVR"] = metrics(y_test, pred_svr)
    results["SVR"]["train_time_sec"] = round(time.time() - t0, 2)
    predictions["SVR"] = pred_svr.tolist()
    print("SVR:", results["SVR"])

    # ---------------- Gradient Boosting ----------------
    t0 = time.time()
    gbm = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=0.8, random_state=42,
    )
    gbm.fit(X_train, y_train)
    pred_gbm = gbm.predict(X_test)
    results["GradientBoosting"] = metrics(y_test, pred_gbm)
    results["GradientBoosting"]["train_time_sec"] = round(time.time() - t0, 2)
    predictions["GradientBoosting"] = pred_gbm.tolist()
    print("GBM:", results["GradientBoosting"])

    top_feats = sorted(zip(feature_cols, gbm.feature_importances_), key=lambda x: -x[1])[:10]
    feature_importance = [{"feature": f, "importance": float(i)} for f, i in top_feats]

    # ---------------- LSTM ----------------
    t0 = time.time()
    Xtr_seq, ytr_seq = make_sequences(X_train, y_train, SEQ_LEN)
    Xval_seq, yval_seq = make_sequences(
        np.concatenate([X_train[-SEQ_LEN:], X_val]),
        np.concatenate([y_train[-SEQ_LEN:], y_val]), SEQ_LEN,
    )
    Xtest_seq, ytest_seq = make_sequences(
        np.concatenate([X_val[-SEQ_LEN:], X_test]),
        np.concatenate([y_val[-SEQ_LEN:], y_test]), SEQ_LEN,
    )

    model, history, device = train_lstm(Xtr_seq, ytr_seq, Xval_seq, yval_seq, n_features=X_train.shape[1])
    model.eval()
    with torch.no_grad():
        pred_lstm = model(torch.tensor(Xtest_seq, dtype=torch.float32).to(device)).cpu().numpy()
    results["LSTM"] = metrics(ytest_seq, pred_lstm)
    results["LSTM"]["train_time_sec"] = round(time.time() - t0, 2)
    results["LSTM"]["epochs_trained"] = len(history)
    predictions["LSTM"] = pred_lstm.tolist()
    predictions["dates"] = predictions["dates"][-len(pred_lstm):]
    predictions["actual"] = predictions["actual"][-len(pred_lstm):]
    predictions["SVR"] = predictions["SVR"][-len(pred_lstm):]
    predictions["GradientBoosting"] = predictions["GradientBoosting"][-len(pred_lstm):]
    print("LSTM:", results["LSTM"])

    return {
        "ticker": ticker,
        "metrics": results,
        "predictions": predictions,
        "feature_importance": feature_importance,
        "lstm_history": history,
        "n_train": n_train, "n_val": n_val_end - n_train, "n_test": len(y_test),
        "n_features": len(feature_cols),
    }


def main():
    all_results = {}
    for ticker in ["MSFT", "GOOGL"]:
        all_results[ticker] = run_for_ticker(ticker)

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved combined results -> {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()