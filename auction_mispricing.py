"""
auction_mispricing.py
=====================
An end-to-end, leak-free initial analysis for detecting MISPRICED auction lots
and estimating forward ROI on structured auction-style data.

This is the auction-specific companion to the generic backtest harness. It is
built to answer one honest question: after removing look-ahead and testing
strictly out-of-sample, can a model identify lots that will earn a positive,
cost-adjusted forward return better than the auction house's own pre-sale
estimate?

Runs with zero setup on a REALISTIC SYNTHETIC auction dataset so the client can
see the full methodology working end-to-end today. Point it at real data with
--csv PATH once the schema is confirmed.

    python auction_mispricing.py                 # synthetic demo
    python auction_mispricing.py --csv lots.csv  # your data (see column map below)

Every variable is documented in VARIABLE_DICTIONARY below and printed at runtime.
"""
from __future__ import annotations

import argparse
import numpy as np

from backtest import walk_forward
from leakage_audit import run_full_audit


# --------------------------------------------------------------------------- #
# VARIABLE DICTIONARY  (printed for the client at runtime)
# --------------------------------------------------------------------------- #
VARIABLE_DICTIONARY = {
    # ---- identifiers / time ----
    "sale_date": ("time", "Date the lot sold. Drives the strict time-ordered "
                  "train/test split. Nothing dated after a lot's sale may inform "
                  "its prediction."),
    # ---- catalogue features known AT sale time (safe inputs) ----
    "artist_tier": ("feature", "Ordinal reputation tier of the maker/artist "
                    "(1=blue-chip .. 4=emerging). High-cardinality artist IDs are "
                    "target-encoded using ONLY past sales (as-of the sale date)."),
    "medium": ("feature", "Category of the object (oil, print, sculpture, watch, "
               "etc.). Strong hedonic driver of price level."),
    "area_cm2": ("feature", "Physical size. In art, price scales sub-linearly "
                 "with area; a classic hedonic feature."),
    "provenance_score": ("feature", "Strength of ownership history / exhibition "
                         "record (0..1). Higher provenance lifts realized price."),
    "condition_score": ("feature", "Condition/quality grade (0..1). Directly "
                        "affects desirability and resale."),
    "rarity": ("feature", "Scarcity proxy (0..1): edition size, uniqueness."),
    "presale_estimate": ("baseline", "The auction house's pre-sale mid-estimate. "
                         "This IS the market-implied fair value and is the hard "
                         "baseline the model must beat. Known at sale time."),
    "hammer_price": ("context", "Price actually paid at this sale = the purchase "
                     "price for an ROI calculation. Known at sale time."),
    # ---- future outcome (used ONLY to build the label, never as a feature) ----
    "future_value": ("outcome", "A later realized resale price / valuation. Used "
                     "to construct the target. NEVER an input feature."),
    # ---- cost terms (net-return realism) ----
    "buyers_premium": ("cost", "Buyer's premium paid on purchase (~20-26% at "
                       "major houses). Subtracted so ROI is net, not gross."),
    "sellers_fee": ("cost", "Seller commission on the future sale. Subtracted."),
    # ---- derived target ----
    "target_log_roi": ("target", "log((future_value*(1-sellers_fee)) / "
                       "(hammer_price*(1+buyers_premium))). Net forward log-ROI "
                       "over the holding horizon. The regression target."),
    "target_is_mispriced": ("target", "1 if net forward log-ROI clears a cost-of-"
                            "capital hurdle. A PURE outcome label (it never uses the "
                            "estimate gap), so beating the market-implied baseline is "
                            "an honest test. The decision (classification) target."),
}


def print_variable_dictionary():
    print("\n=== VARIABLE DICTIONARY ===")
    order = ["time", "feature", "baseline", "context", "outcome", "cost", "target"]
    for kind in order:
        rows = [(k, v[1]) for k, v in VARIABLE_DICTIONARY.items() if v[0] == kind]
        if not rows:
            continue
        print(f"\n[{kind.upper()}]")
        for name, desc in rows:
            print(f"  - {name}: {desc}")


# --------------------------------------------------------------------------- #
# Realistic synthetic auction dataset
# --------------------------------------------------------------------------- #
def synthetic_auctions(n=5000, seed=11, hurdle=0.05):
    """Build a chronologically-ordered auction dataset with a weak-but-real
    relationship between catalogue quality and forward return, plus realistic
    noise, costs, and a pre-sale estimate that is a good (but beatable) baseline.
    """
    rng = np.random.default_rng(seed)

    artist_tier = rng.integers(1, 5, n)                    # 1 best .. 4 emerging
    medium = rng.integers(0, 4, n)
    area = rng.lognormal(mean=7.0, sigma=0.6, size=n)      # cm^2
    provenance = rng.beta(2, 3, n)
    condition = rng.beta(5, 2, n)
    rarity = rng.beta(2, 5, n)

    # "True" fair value (log scale) from hedonic drivers
    fair = (
        11.0
        - 0.55 * artist_tier
        + 0.15 * medium
        + 0.35 * np.log(area)
        + 1.1 * provenance
        + 0.8 * condition
        + 1.4 * rarity
    )
    # Pre-sale estimate = market's read of fair value. It captures the hedonic
    # LEVEL well, but by design it does NOT anticipate forward drift.
    presale = np.exp(fair + rng.normal(0, 0.18, n))
    # Hammer price = estimate perturbed by auction-day randomness. So the
    # estimate gap (presale - hammer) is essentially day-noise: a fair market
    # baseline that should NOT predict forward return.
    hammer = np.exp(np.log(presale) + rng.normal(0, 0.22, n))

    # Forward value: appreciation is driven by provenance + rarity (real, known
    # catalogue features the pre-sale estimate under-weights), on top of a base
    # drift, plus large idiosyncratic noise. Costs are a high bar, so only
    # well-selected lots clear them net-of-fees -> a realistic, honest edge.
    forward_drift = -0.05 + 0.30 * provenance + 0.55 * rarity - 0.06 * (artist_tier - 1) / 3
    future = hammer * np.exp(forward_drift + rng.normal(0, 0.30, n))

    buyers_premium = np.full(n, 0.25)
    sellers_fee = np.full(n, 0.10)

    net_cost = hammer * (1 + buyers_premium)
    net_proceeds = future * (1 - sellers_fee)
    log_roi = np.log(net_proceeds / net_cost)
    # PURE outcome label: did the net forward ROI clear the hurdle? No estimate
    # gap in the definition, so beating the market-implied baseline is honest.
    is_mispriced = (log_roi > hurdle).astype(int)

    feature_names = ["artist_tier", "medium", "area_cm2", "provenance_score",
                     "condition_score", "rarity", "presale_estimate", "hammer_price"]
    X = np.column_stack([artist_tier, medium, area, provenance, condition, rarity,
                         np.log(presale), np.log(hammer)])
    meta = {
        "presale": presale, "hammer": hammer, "future": future,
        "log_roi": log_roi, "buyers_premium": buyers_premium, "sellers_fee": sellers_fee,
    }
    return X, is_mispriced, log_roi, feature_names, meta


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def baseline_estimate_only(X, meta, feature_names):
    """Market-implied baseline: rank lots purely by how far the hammer sits below
    the pre-sale estimate (the market's own fair-value signal). No ML."""
    hammer = meta["hammer"]
    presale = meta["presale"]
    return (presale - hammer) / presale     # bigger gap => 'cheaper' vs estimate


def make_gbm():
    try:
        from xgboost import XGBClassifier

        def fp(Xtr, ytr, Xte):
            m = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.04,
                              subsample=0.8, colsample_bytree=0.8,
                              eval_metric="logloss", n_jobs=0)
            m.fit(Xtr, ytr)
            return m.predict_proba(Xte)[:, 1]
        return fp, "XGBoost"
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier

        def fp(Xtr, ytr, Xte):
            m = LGBMClassifier(n_estimators=250, max_depth=4, learning_rate=0.04, verbose=-1)
            m.fit(Xtr, ytr)
            return m.predict_proba(Xte)[:, 1]
        return fp, "LightGBM"
    except Exception:
        pass
    from sklearn.ensemble import GradientBoostingClassifier

    def fp(Xtr, ytr, Xte):
        m = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05)
        m.fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]
    return fp, "sklearn GradientBoosting"


def auc(y_true, y_score):
    y_true = np.asarray(y_true)
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score), float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    n_pos = y_true.sum(); n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def top_decile_net_roi(y_score, log_roi):
    """Economic significance: mean net log-ROI of the top-10% highest-scored lots."""
    k = max(1, len(y_score) // 10)
    idx = np.argsort(y_score)[-k:]
    return float(np.mean(log_roi[idx]))


# --------------------------------------------------------------------------- #
def permutation_test(X, y, model_fp, score_fn, n_perm=30, seed=0):
    """Label-permutation null: shuffle labels, re-run the SAME OOS protocol, and
    report where the model's real out-of-sample AUC sits in the null. Under a
    true null (no signal) OOS AUC clusters at 0.5; a real edge sits in the tail."""
    rng = np.random.default_rng(seed)

    def model_oos_auc(y_):
        rep = walk_forward(X, y_, fit_predict=model_fp,
                           baseline_fit_predict=lambda a, b, c: np.full(len(c), b.mean()),
                           score_fn=score_fn, n_splits=4)
        return float(np.mean([f.model_score for f in rep.folds]))

    real = model_oos_auc(y)
    null = np.array([model_oos_auc(rng.permutation(y)) for _ in range(n_perm)])
    p = float((np.sum(null >= real) + 1) / (n_perm + 1))
    return real, p, null


def selection_bias_demo(seed=11, hurdle=0.05):
    """Demonstrate the #1 killer of a phantom edge: survivorship / selection bias,
    HONESTLY - including why it is only partially fixable from observed data.

    Only a minority of lots ever RESELL, and winners resell disproportionately.
    Critically, that selection is driven by the OUTCOME (the realized return),
    which is not knowable at the decision point -> the missingness is MNAR
    (missing not at random). A propensity model can only be fit on OBSERVED
    features, so inverse-propensity weighting corrects the part of selection that
    features can explain, and UNDER-corrects the outcome-driven part. Full
    correction needs a Heckman-type selection model with a return proxy /
    instrument. We show all three numbers so the client sees the honest gap.
    Grounded in Korteweg, Kraussl & Verwijmeren (RFS 2016): correcting selection
    cut art returns ~8.7% -> 6.3% and Sharpe ~0.27 -> 0.11.
    """
    X, _, log_roi, names, meta = synthetic_auctions(n=8000, seed=seed, hurdle=hurdle)

    # TRUE resale process is MNAR: it depends on the realized return (winners
    # resell) plus an observable feature (rarity). The return term is what makes
    # this hard - it is not in the feature set available at the decision point.
    rng = np.random.default_rng(seed + 1)
    rarity = X[:, 5]
    p_true = 1.0 / (1.0 + np.exp(-(-1.4 + 2.2 * log_roi + 1.0 * rarity)))
    resold = (rng.uniform(size=len(log_roi)) < p_true).astype(int)

    resold_frac = resold.mean()
    naive = log_roi[resold == 1].mean()                 # what a naive study sees
    true_pop = log_roi.mean()                            # the real population mean

    # HONEST correction: fit P(resold) on OBSERVED FEATURES ONLY (never the
    # outcome). This is all a real analyst can do.
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, resold)
    p_hat = clf.predict_proba(X)[:, 1]
    w = 1.0 / np.clip(p_hat[resold == 1], 0.02, 1.0)
    ipw_features = np.average(log_roi[resold == 1], weights=w)

    print("\n=== 5. Selection / survivorship bias (the #1 killer) ===")
    print(f"  Resold fraction (the observable subset)      = {resold_frac:.1%}")
    print(f"  Naive mean net log-ROI on resold-only         = {naive:+.4f}  <- biased UP")
    print(f"  IPW (features-only propensity) corrected      = {ipw_features:+.4f}")
    print(f"  True population mean net log-ROI              = {true_pop:+.4f}")
    print(f"  Bias still remaining after features-only IPW  = {ipw_features - true_pop:+.4f}")
    print("  Honest read: selection here is on the OUTCOME (MNAR), so a")
    print("  features-only propensity under-corrects. Report uncorrected AND")
    print("  corrected both, and use a Heckman selection model (return proxy /")
    print("  instrument) for the outcome-driven part. Never train/score only on")
    print("  resold assets and never claim survivorship bias is cleanly solved.")


def main():
    ap = argparse.ArgumentParser(description="Auction mispricing / forward-ROI analysis")
    ap.add_argument("--csv")
    ap.add_argument("--splits", type=int, default=4)
    args = ap.parse_args()

    print_variable_dictionary()

    if args.csv:
        raise SystemExit("CSV mode: map your columns to the VARIABLE DICTIONARY first "
                         "(see column names above), then wire load here.")
    X, y, log_roi, names, meta = synthetic_auctions()
    print(f"\nDataset: {len(X)} lots x {X.shape[1]} features "
          f"(synthetic, chronologically ordered). Mispriced rate = {y.mean():.1%}")

    print("\n=== 1. Leakage audit (run before any modelling) ===")
    findings = run_full_audit(X, y, names)
    print("  No leakage suspects flagged." if not findings else
          "\n".join(f"  [{f.kind}] {f.feature}: {f.note}" for f in findings))

    print("\n=== 2. Baselines vs ML (walk-forward, out-of-sample) ===")
    model_fp, model_name = make_gbm()

    # Baseline A: market-implied (estimate gap). Evaluated OOS the same way.
    def estimate_baseline_fp(Xtr, ytr, Xte):
        # rank by 'cheapness vs estimate' using columns log(presale), log(hammer)
        presale_log, hammer_log = Xte[:, 6], Xte[:, 7]
        return presale_log - hammer_log

    rep = walk_forward(
        X, y,
        fit_predict=model_fp,
        baseline_fit_predict=estimate_baseline_fp,   # <-- the HARD baseline
        score_fn=auc, n_splits=args.splits,
    )
    print(f"  Model: {model_name}   Baseline: market-implied (pre-sale estimate gap)")
    print(rep.summary())

    print("\n=== 3. Economic significance (top-decile net forward log-ROI) ===")
    # single OOS split for a readable economic read
    cut = int(len(X) * 0.7)
    scores = model_fp(X[:cut], y[:cut], X[cut:])
    base_scores = estimate_baseline_fp(None, None, X[cut:])
    print(f"  Model top-decile net log-ROI   = {top_decile_net_roi(scores, log_roi[cut:]):+.4f}")
    print(f"  Baseline top-decile net log-ROI= {top_decile_net_roi(base_scores, log_roi[cut:]):+.4f}")
    print(f"  All-lots mean net log-ROI      = {np.mean(log_roi[cut:]):+.4f}")

    print("\n=== 4. Is it luck? Label-permutation null test ===")
    real, p, null = permutation_test(X, y, model_fp, auc, n_perm=30)
    print(f"  Real model OOS AUC = {real:.4f} | null mean = {null.mean():.4f} "
          f"| p-value = {p:.3f}")
    print("  Interpretation:", "edge unlikely to be luck (p<0.05)" if p < 0.05
          else "edge NOT distinguishable from luck - report as NO EDGE")

    selection_bias_demo()


if __name__ == "__main__":
    main()
