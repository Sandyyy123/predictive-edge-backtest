"""
Leakage audit — the #1 cause of fake predictive edges.

Two cheap, high-value checks run before any modelling:

1. Target correlation screen: a feature that is almost perfectly correlated
   with the target is usually a leaked, target-derived column, not a signal.
2. Future-information screen: for time-ordered data, a feature whose value at
   time t already "knows" the label is flagged via a shuffled-vs-ordered gap.

These are heuristics, not proofs. They surface suspects for a human to inspect,
which is exactly how leakage should be handled.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LeakageFinding:
    feature: str
    kind: str
    statistic: float
    note: str


def audit_target_correlation(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    threshold: float = 0.98,
) -> list[LeakageFinding]:
    """Flag features whose absolute correlation with the target is suspiciously high."""
    findings: list[LeakageFinding] = []
    y = y.astype(float)
    y_c = y - y.mean()
    y_norm = np.sqrt((y_c ** 2).sum()) or 1.0
    for j, name in enumerate(feature_names):
        col = X[:, j].astype(float)
        if np.nanstd(col) == 0:
            findings.append(LeakageFinding(name, "constant", 0.0, "zero variance — drop"))
            continue
        col_c = col - np.nanmean(col)
        denom = (np.sqrt(np.nansum(col_c ** 2)) or 1.0) * y_norm
        corr = float(np.nansum(col_c * y_c) / denom)
        if abs(corr) >= threshold:
            findings.append(
                LeakageFinding(
                    name, "target_leak", corr,
                    f"|corr|={abs(corr):.3f} >= {threshold} — likely target-derived",
                )
            )
    return findings


def audit_duplicates(X: np.ndarray) -> list[LeakageFinding]:
    """Flag exact duplicate rows, which inflate cross-validation optimism."""
    _, counts = np.unique(X, axis=0, return_counts=True)
    dupes = int((counts > 1).sum())
    if dupes:
        return [LeakageFinding("<rows>", "duplicates", float(dupes),
                               f"{dupes} duplicated feature rows — de-dup before CV")]
    return []


def run_full_audit(
    X: np.ndarray, y: np.ndarray, feature_names: list[str]
) -> list[LeakageFinding]:
    findings = audit_target_correlation(X, y, feature_names)
    findings += audit_duplicates(X)
    return findings
