"""
train_advanced.py
------------------
Advanced version of train_compare.py. Adds:

1. Walk-forward (expanding-window) validation instead of one fixed split —
   the model is retrained on each new fold and only ever predicts the fold
   immediately after its training window, then the window expands and it
   retrains again. This mimics how the model would actually be used in
   production and avoids over-crediting a single lucky test split.

2. Naive baselines — predict-zero and predict-yesterday's-return — evaluated
   on the exact same out-of-sample points as the real models, so the
   "TOP MODEL" badge means something relative to a dumb benchmark, not just
   relative to the other ML models.

3. Ensembling — a simple average of the three base models, and a Ridge
   stacking meta-model fit on the first part of the out-of-sample period
   and evaluated only on the later part (so the meta-model never sees its
   own test data during fitting).

4. SHAP explainability for Gradient Boosting, as a second, more principled
   view of feature importance alongside the built-in impurity-based one.

Run this instead of / in addition to train_compare.py:

    python train_advanced.py

Writes outputs/results.json in a superset schema of the original — the
dashboard reads this dynamically, so nothing else needs to change.
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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from features import build_feature_frame

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

SEQ_LEN = 20
INITIAL_TRAIN_FRAC = 0.60     # first fold's training window
N_FOLDS = 5                   # number of walk-forward folds covering the remaining data
LSTM_EPOCHS_PER_FOLD = 25     # capped for speed across multiple folds
LSTM_PATIENCE = 5
META_TRAIN_FRAC = 0.60        # fraction of the OOS period used to fit the stacking meta-model


# ---------------------------------------------------------------- LSTM ----
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
        return self.head(out[:, -1, :]).squeeze(-1)


def make_sequences(X, y, seq_len):
    xs, ys = [], []
    for i in range(seq_len, len(X)):
        xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(xs), np.array(ys)


def train_lstm(X_train, y_train, X_val, y_val, n_features, epochs, patience, lr=1e-3):
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
        epoch_loss = 0.0
        for start in range(0, len(Xt), batch_size):
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
            val_loss = loss_fn(model(Xv), yv).item()
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
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "directional_accuracy": float(np.mean(np.sign(y_true) == np.sign(y_pred))),
    }


# ------------------------------------------------------- walk-forward ----
def walk_forward_folds(n, initial_train_frac, n_folds):
    """Yields (train_end, fold_end) indices for an expanding-window walk-forward split."""
    n_initial = int(n * initial_train_frac)
    remaining = n - n_initial
    fold_size = remaining // n_folds
    train_end = n_initial
    for k in range(n_folds):
        fold_end = n if k == n_folds - 1 else train_end + fold_size
        yield train_end, fold_end
        train_end = fold_end


TRANSACTION_COST_BPS = 5  # round-trip cost per position change, in basis points


def run_backtest(dates, y_actual, oos_pred, ens_avg_pred):
    """
    Turns each model's predicted return into a daily long/flat/short position via the
    sign of the prediction, applies a simple transaction cost whenever the position
    changes, and reports the resulting equity curve, annualized Sharpe ratio, max
    drawdown, and win rate -- benchmarked against buy-and-hold.

    This is a simplified, illustrative backtest (no slippage model, no position sizing,
    no borrow cost for shorts) -- its purpose is to show whether a model's directional
    calls translate into any edge once trading frictions are included, not to be a
    production strategy.
    """
    cost = TRANSACTION_COST_BPS / 10000.0
    y = np.asarray(y_actual)
    n = len(y)

    def simulate(pred):
        pred = np.asarray(pred)
        position = np.sign(pred)  # +1 long, -1 short, 0 flat (never exactly 0 in practice)
        prev_position = np.concatenate([[0.0], position[:-1]])
        turnover = np.abs(position - prev_position)
        strategy_return = position * y - turnover * cost

        equity = np.cumprod(1 + strategy_return)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max

        ann_factor = np.sqrt(252)
        sharpe = float(strategy_return.mean() / (strategy_return.std() + 1e-12) * ann_factor)
        traded_days = strategy_return[position != 0]

        return {
            "cumulative_return": float(equity[-1] - 1),
            "sharpe": round(sharpe, 3),
            "max_drawdown": float(drawdown.min()),
            "win_rate": float(np.mean(traded_days > 0)) if len(traded_days) else 0.0,
            "n_trades": int(np.sum(turnover > 0)),
            "equity_curve": equity.tolist(),
        }

    result = {m: simulate(oos_pred[m]) for m in ["SVR", "GradientBoosting", "LSTM"]}
    result["EnsembleAverage"] = simulate(ens_avg_pred)

    # buy-and-hold benchmark: long every day, one entry cost only
    bh_return = y.copy()
    bh_return[0] -= cost
    bh_equity = np.cumprod(1 + bh_return)
    bh_dd = (bh_equity - np.maximum.accumulate(bh_equity)) / np.maximum.accumulate(bh_equity)
    result["BuyAndHold"] = {
        "cumulative_return": float(bh_equity[-1] - 1),
        "sharpe": round(float(bh_return.mean() / (bh_return.std() + 1e-12) * np.sqrt(252)), 3),
        "max_drawdown": float(bh_dd.min()),
        "win_rate": float(np.mean(y > 0)),
        "n_trades": 1,
        "equity_curve": bh_equity.tolist(),
    }

    return {"dates": dates, "transaction_cost_bps": TRANSACTION_COST_BPS, "strategies": result}


def run_for_ticker(ticker: str):
    print(f"\n{'=' * 60}\n{ticker} — walk-forward validation ({N_FOLDS} folds)\n{'=' * 60}")
    df = pd.read_csv(DATA_DIR / f"{ticker}.csv", parse_dates=["Date"])
    feat_df, feature_cols = build_feature_frame(df)

    X_raw = feat_df[feature_cols].values
    y_raw = feat_df["target_next_return"].values
    dates = pd.to_datetime(feat_df["Date"].values)

    n = len(X_raw)
    oos_dates, oos_actual = [], []
    oos_pred = {"SVR": [], "GradientBoosting": [], "LSTM": []}
    lstm_history_last_fold = []
    gbm_last, feature_cols_used = None, feature_cols

    fold_boundaries = list(walk_forward_folds(n, INITIAL_TRAIN_FRAC, N_FOLDS))

    for fold_idx, (train_end, fold_end) in enumerate(fold_boundaries, start=1):
        # carve out a small validation tail from the training window (for LSTM early stopping)
        val_len = max(SEQ_LEN * 3, int((train_end) * 0.10))
        tr_end = train_end - val_len

        scaler_X = StandardScaler().fit(X_raw[:tr_end])
        X_scaled = scaler_X.transform(X_raw)

        X_train, y_train = X_scaled[:tr_end], y_raw[:tr_end]
        X_val, y_val = X_scaled[tr_end:train_end], y_raw[tr_end:train_end]
        X_fold, y_fold = X_scaled[train_end:fold_end], y_raw[train_end:fold_end]
        fold_dates = dates[train_end:fold_end]

        print(f"\n-- Fold {fold_idx}/{N_FOLDS}: train={tr_end}, val={train_end - tr_end}, "
              f"predict={fold_end - train_end} days ({fold_dates[0].date()} to {fold_dates[-1].date()})")

        # SVR
        svr = SVR(kernel="rbf", C=1.0, epsilon=0.001, gamma="scale")
        svr.fit(X_train, y_train)
        pred_svr = svr.predict(X_fold)

        # Gradient Boosting (early-stopped on an internal validation slice to avoid overfitting
        # small, noisy folds -- without this it happily memorizes training noise)
        gbm = GradientBoostingRegressor(
            n_estimators=500, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=42,
            validation_fraction=0.15, n_iter_no_change=15, tol=1e-5,
        )
        gbm.fit(X_train, y_train)
        pred_gbm = gbm.predict(X_fold)
        gbm_last = gbm  # keep the final fold's model for SHAP

        # LSTM
        Xtr_seq, ytr_seq = make_sequences(X_train, y_train, SEQ_LEN)
        Xval_seq, yval_seq = make_sequences(
            np.concatenate([X_train[-SEQ_LEN:], X_val]), np.concatenate([y_train[-SEQ_LEN:], y_val]), SEQ_LEN)
        Xfold_seq, yfold_seq = make_sequences(
            np.concatenate([X_val[-SEQ_LEN:], X_fold]), np.concatenate([y_val[-SEQ_LEN:], y_fold]), SEQ_LEN)

        model, history, device = train_lstm(
            Xtr_seq, ytr_seq, Xval_seq, yval_seq, n_features=X_train.shape[1],
            epochs=LSTM_EPOCHS_PER_FOLD, patience=LSTM_PATIENCE)
        model.eval()
        with torch.no_grad():
            pred_lstm = model(torch.tensor(Xfold_seq, dtype=torch.float32).to(device)).cpu().numpy()
        if fold_idx == N_FOLDS:
            lstm_history_last_fold = history

        # LSTM sequences eat the first SEQ_LEN points of the fold -- align everyone to that shorter length
        cut = len(pred_svr) - len(pred_lstm)
        oos_dates.extend(pd.to_datetime(fold_dates[cut:]).astype(str).tolist())
        oos_actual.extend(y_fold[cut:].tolist())
        oos_pred["SVR"].extend(pred_svr[cut:].tolist())
        oos_pred["GradientBoosting"].extend(pred_gbm[cut:].tolist())
        oos_pred["LSTM"].extend(pred_lstm.tolist())

        fold_rmse_svr = float(np.sqrt(mean_squared_error(y_fold[cut:], pred_svr[cut:])))
        fold_rmse_gbm = float(np.sqrt(mean_squared_error(y_fold[cut:], pred_gbm[cut:])))
        fold_rmse_lstm = float(np.sqrt(mean_squared_error(yfold_seq, pred_lstm)))
        print(f"   fold RMSE — SVR: {fold_rmse_svr:.5f}, GBM: {fold_rmse_gbm:.5f}, LSTM: {fold_rmse_lstm:.5f}")

    # ---------------- aggregate out-of-sample metrics per model ----------------
    y_oos = np.array(oos_actual)
    results = {}
    for m in ["SVR", "GradientBoosting", "LSTM"]:
        results[m] = metrics(y_oos, oos_pred[m])
    print("\nWalk-forward OOS metrics:")
    for m in ["SVR", "GradientBoosting", "LSTM"]:
        print(f"  {m}: {results[m]}")

    # ---------------- naive baselines (same OOS points) ----------------
    naive_zero_pred = np.zeros_like(y_oos)
    naive_persist_pred = np.concatenate([[0.0], y_oos[:-1]])  # predict "yesterday's realized return"
    results["NaiveZero"] = metrics(y_oos, naive_zero_pred)
    results["NaivePersistence"] = metrics(y_oos, naive_persist_pred)
    print(f"  NaiveZero: {results['NaiveZero']}")
    print(f"  NaivePersistence: {results['NaivePersistence']}")

    # ---------------- ensembling ----------------
    ens_avg_pred = np.mean([oos_pred["SVR"], oos_pred["GradientBoosting"], oos_pred["LSTM"]], axis=0)
    results["EnsembleAverage"] = metrics(y_oos, ens_avg_pred)
    print(f"  EnsembleAverage: {results['EnsembleAverage']}")

    n_oos = len(y_oos)
    meta_split = int(n_oos * META_TRAIN_FRAC)
    meta_X = np.column_stack([oos_pred["SVR"], oos_pred["GradientBoosting"], oos_pred["LSTM"]])
    meta_train_X, meta_test_X = meta_X[:meta_split], meta_X[meta_split:]
    meta_train_y, meta_test_y = y_oos[:meta_split], y_oos[meta_split:]

    meta_model = LinearRegression(positive=True)  # positive weights -> interpretable as a blend, no arbitrary shrinkage
    meta_model.fit(meta_train_X, meta_train_y)
    stack_pred_test = meta_model.predict(meta_test_X)
    results["EnsembleStacking"] = metrics(meta_test_y, stack_pred_test)
    results["EnsembleStacking"]["weights"] = {
        "SVR": round(float(meta_model.coef_[0]), 3),
        "GradientBoosting": round(float(meta_model.coef_[1]), 3),
        "LSTM": round(float(meta_model.coef_[2]), 3),
        "intercept": round(float(meta_model.intercept_), 6),
    }
    print(f"  EnsembleStacking (fit on first {META_TRAIN_FRAC:.0%} of OOS, "
          f"tested on remaining {1 - META_TRAIN_FRAC:.0%}): {results['EnsembleStacking']}")

    # ---------------- backtested trading strategy ----------------
    backtest = run_backtest(oos_dates, y_oos, oos_pred, ens_avg_pred)

    # ---------------- feature importance (impurity-based) ----------------
    top_feats = sorted(zip(feature_cols_used, gbm_last.feature_importances_), key=lambda x: -x[1])[:10]
    feature_importance = [{"feature": f, "importance": float(i)} for f, i in top_feats]

    # ---------------- SHAP explainability ----------------
    shap_importance = []
    try:
        import shap
        X_bg = StandardScaler().fit(X_raw[:fold_boundaries[-1][0]]).transform(X_raw[:fold_boundaries[-1][0]])
        sample = X_bg[-min(300, len(X_bg)):]  # sample for speed
        explainer = shap.TreeExplainer(gbm_last)
        shap_values = explainer.shap_values(sample)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_shap = sorted(zip(feature_cols_used, mean_abs_shap), key=lambda x: -x[1])[:10]
        shap_importance = [{"feature": f, "importance": float(v)} for f, v in top_shap]
        print(f"  SHAP importances computed on {len(sample)} samples.")
    except ImportError:
        print("  [!] shap not installed -- skipping SHAP importances. Run: pip install shap")

    print("\nBacktest (sign-of-prediction strategy, 5bps transaction cost):")
    for name, r in backtest["strategies"].items():
        print(f"  {name}: cum_return={r['cumulative_return']:.2%}, sharpe={r['sharpe']}, "
              f"max_dd={r['max_drawdown']:.2%}, win_rate={r['win_rate']:.2%}, trades={r['n_trades']}")

    return {
        "ticker": ticker,
        "validation_scheme": {
            "type": "walk-forward (expanding window)",
            "n_folds": N_FOLDS,
            "initial_train_frac": INITIAL_TRAIN_FRAC,
            "n_oos_points": n_oos,
        },
        "metrics": results,
        "predictions": {
            "dates": oos_dates, "actual": oos_actual,
            "SVR": oos_pred["SVR"], "GradientBoosting": oos_pred["GradientBoosting"], "LSTM": oos_pred["LSTM"],
            "EnsembleAverage": ens_avg_pred.tolist(),
        },
        "backtest": backtest,
        "feature_importance": feature_importance,
        "shap_importance": shap_importance,
        "lstm_history": lstm_history_last_fold,
        "n_train": fold_boundaries[0][0], "n_val": None, "n_test": n_oos,
        "n_features": len(feature_cols_used),
    }


def main():
    all_results = {}
    for ticker in ["MSFT", "GOOGL"]:
        all_results[ticker] = run_for_ticker(ticker)

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'=' * 60}\nSaved combined results -> {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()