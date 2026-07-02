"""
predictive-edge-backtest — runnable demo.

Answers one question the honest way: does a structured dataset contain a
predictive edge that survives a strict out-of-sample walk-forward test?

Runs on a synthetic dataset by default so it works with zero setup. Point it at
your own CSV with --csv PATH --target COLNAME to run the real thing.

    python main.py                       # synthetic demo
    python main.py --csv data.csv --target y

Gracefully degrades: uses XGBoost/LightGBM if installed, else a scikit-learn
gradient-boosting fallback, else a pure-NumPy logistic model — so it always runs.
"""
from __future__ import annotations

import argparse

import numpy as np

from backtest import walk_forward
from leakage_audit import run_full_audit


# --------------------------------------------------------------------------- #
# Models (with graceful fallbacks so the demo always runs)
# --------------------------------------------------------------------------- #
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def baseline_fit_predict(X_tr, y_tr, X_te):
    """Naive baseline: predict the training base rate for every row."""
    rate = float(y_tr.mean())
    return np.full(len(X_te), rate)


def _logistic_fit_predict(X_tr, y_tr, X_te, epochs=300, lr=0.1):
    """Pure-NumPy logistic regression — the last-resort fallback."""
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-9
    Xtr = (X_tr - mu) / sd
    Xte = (X_te - mu) / sd
    w = np.zeros(Xtr.shape[1])
    b = 0.0
    n = len(Xtr)
    for _ in range(epochs):
        p = _sigmoid(Xtr @ w + b)
        g = p - y_tr
        w -= lr * (Xtr.T @ g) / n
        b -= lr * g.mean()
    return _sigmoid(Xte @ w + b)


def make_model():
    """Return the best available gradient-boosted classifier's predict function."""
    try:
        from xgboost import XGBClassifier

        def fp(X_tr, y_tr, X_te):
            m = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                n_jobs=0,
            )
            m.fit(X_tr, y_tr)
            return m.predict_proba(X_te)[:, 1]

        return fp, "XGBoost"
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier

        def fp(X_tr, y_tr, X_te):
            m = LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, verbose=-1)
            m.fit(X_tr, y_tr)
            return m.predict_proba(X_te)[:, 1]

        return fp, "LightGBM"
    except Exception:
        pass
    try:
        from sklearn.ensemble import GradientBoostingClassifier

        def fp(X_tr, y_tr, X_te):
            m = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05)
            m.fit(X_tr, y_tr)
            return m.predict_proba(X_te)[:, 1]

        return fp, "sklearn GradientBoosting"
    except Exception:
        return _logistic_fit_predict, "NumPy logistic (fallback)"


# --------------------------------------------------------------------------- #
# Metric
# --------------------------------------------------------------------------- #
def auc(y_true, y_score):
    """ROC-AUC via the Mann-Whitney U relation. No sklearn dependency."""
    y_true = np.asarray(y_true)
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score), float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def synthetic_dataset(n=4000, n_features=8, signal=0.6, seed=7):
    """Chronologically-ordered data with a weak-but-real signal + noise features."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    weights = np.zeros(n_features)
    weights[:3] = np.array([1.0, -0.7, 0.5]) * signal   # only first 3 carry signal
    logits = X @ weights + rng.normal(scale=1.0, size=n)
    y = (logits > np.median(logits)).astype(int)
    names = [f"feature_{i}" for i in range(n_features)]
    return X, y, names


def load_csv(path, target):
    import pandas as pd

    df = pd.read_csv(path)
    if target not in df.columns:
        raise SystemExit(f"target '{target}' not in columns: {list(df.columns)}")
    y = df[target].to_numpy()
    X = df.drop(columns=[target]).select_dtypes(include="number")
    return X.to_numpy(), y.astype(int), list(X.columns)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Predictive-edge walk-forward backtest")
    ap.add_argument("--csv", help="path to a CSV; omit to run the synthetic demo")
    ap.add_argument("--target", help="target column name (required with --csv)")
    ap.add_argument("--splits", type=int, default=4)
    args = ap.parse_args()

    if args.csv:
        if not args.target:
            raise SystemExit("--target is required with --csv")
        X, y, names = load_csv(args.csv, args.target)
        print(f"Loaded {len(X)} rows x {X.shape[1]} numeric features from {args.csv}")
    else:
        X, y, names = synthetic_dataset()
        print(f"Synthetic demo: {len(X)} rows x {X.shape[1]} features "
              f"(only feature_0..2 carry real signal)")

    print("\n=== 1. Leakage audit ===")
    findings = run_full_audit(X, y, names)
    if findings:
        for f in findings:
            print(f"  [{f.kind}] {f.feature}: {f.note}")
    else:
        print("  No leakage suspects flagged.")

    model_fp, model_name = make_model()
    print(f"\n=== 2. Walk-forward backtest (model: {model_name}) ===")
    report = walk_forward(
        X, y,
        fit_predict=model_fp,
        baseline_fit_predict=baseline_fit_predict,
        score_fn=auc,
        n_splits=args.splits,
    )
    print(report.summary())


if __name__ == "__main__":
    main()
