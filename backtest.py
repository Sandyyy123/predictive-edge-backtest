"""
Walk-forward out-of-sample backtesting engine.

The single most important property of this module: no observation from the
future ever influences a prediction about the past. Each fold trains only on
data strictly earlier than the out-of-sample window it scores.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class FoldResult:
    fold: int
    train_end: int
    test_start: int
    test_end: int
    n_train: int
    n_test: int
    baseline_score: float
    model_score: float

    @property
    def edge(self) -> float:
        return self.model_score - self.baseline_score


@dataclass
class BacktestReport:
    folds: list[FoldResult] = field(default_factory=list)

    @property
    def mean_edge(self) -> float:
        return float(np.mean([f.edge for f in self.folds])) if self.folds else 0.0

    @property
    def positive_folds(self) -> int:
        return sum(1 for f in self.folds if f.edge > 0)

    def verdict(self, min_edge: float = 0.02) -> str:
        if not self.folds:
            return "NO DATA"
        share = self.positive_folds / len(self.folds)
        if self.mean_edge >= min_edge and share >= 0.75:
            return "CONDITIONAL EDGE — worth a phase 2"
        if self.mean_edge >= min_edge:
            return "MARGINAL / REGIME-DEPENDENT edge"
        return "NO EDGE — do not deploy on this data"

    def summary(self) -> str:
        lines = [
            f"{'fold':>4} {'n_train':>8} {'n_test':>8} {'baseline':>9} {'model':>7} {'edge':>7}"
        ]
        for f in self.folds:
            lines.append(
                f"{f.fold:>4} {f.n_train:>8} {f.n_test:>8} "
                f"{f.baseline_score:>9.4f} {f.model_score:>7.4f} {f.edge:>+7.4f}"
            )
        lines.append("-" * 48)
        lines.append(
            f"mean edge = {self.mean_edge:+.4f} | "
            f"positive folds = {self.positive_folds}/{len(self.folds)}"
        )
        lines.append(f"VERDICT: {self.verdict()}")
        return "\n".join(lines)


def walk_forward(
    X: np.ndarray,
    y: np.ndarray,
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    baseline_fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    score_fn: Callable[[np.ndarray, np.ndarray], float],
    n_splits: int = 4,
    min_train_frac: float = 0.4,
) -> BacktestReport:
    """Expanding-window walk-forward backtest.

    Rows are assumed to be in chronological order. Fold k trains on
    [0, train_end) and scores on the following out-of-sample block only.
    """
    n = len(X)
    if n != len(y):
        raise ValueError("X and y length mismatch")

    start = int(n * min_train_frac)
    remaining = n - start
    block = remaining // n_splits
    if block == 0:
        raise ValueError("Not enough rows for the requested number of splits")

    report = BacktestReport()
    for k in range(n_splits):
        train_end = start + k * block
        test_start = train_end
        test_end = n if k == n_splits - 1 else train_end + block

        X_tr, y_tr = X[:train_end], y[:train_end]
        X_te, y_te = X[test_start:test_end], y[test_start:test_end]

        model_pred = fit_predict(X_tr, y_tr, X_te)
        base_pred = baseline_fit_predict(X_tr, y_tr, X_te)

        report.folds.append(
            FoldResult(
                fold=k + 1,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                n_train=len(X_tr),
                n_test=len(X_te),
                baseline_score=score_fn(y_te, base_pred),
                model_score=score_fn(y_te, model_pred),
            )
        )
    return report
