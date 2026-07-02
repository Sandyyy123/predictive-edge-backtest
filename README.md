# predictive-edge-backtest

A compact, honest research harness for one question: **does a historical
structured dataset contain a predictive edge that survives a strict
out-of-sample backtest — or is it noise?**

Built for proof-of-concept quant/ML research where the deliverable is a *verdict
with evidence*, not a dashboard and not an LLM agent. A well-evidenced "no edge"
is a valid, valuable result — it stops you deploying noise on real capital.

## Why it is trustworthy

- **No look-ahead.** `backtest.py` uses expanding-window walk-forward: fold *k*
  trains only on data strictly earlier than the out-of-sample window it scores.
- **Leakage audit first.** `leakage_audit.py` screens every feature for
  target-derived contamination and duplicate rows *before* any model is fit —
  the #1 cause of fake edges.
- **Baseline as the bar.** Every ML model is scored against a naive base-rate
  baseline, fold by fold. Edge = model − baseline, and it must persist across
  folds, not spike in one lucky period.
- **Honest verdict.** `CONDITIONAL EDGE` / `MARGINAL` / `NO EDGE`, derived from
  mean edge and the share of positive folds — not cherry-picked.

## Run it

```bash
pip install -r requirements.txt

# synthetic demo — works with zero setup
python main.py

# your own data
python main.py --csv your_data.csv --target your_label_column
```

The demo builds a chronologically-ordered dataset where only 3 of 8 features
carry real signal, then recovers a stable out-of-sample edge:

```
fold  n_train   n_test  baseline   model    edge
   1     1600      600    0.5044  0.7736 +0.2691
   2     2200      600    0.4878  0.7926 +0.3047
   3     2800      600    0.5169  0.7689 +0.2520
   4     3400      600    0.5155  0.7570 +0.2415
------------------------------------------------
mean edge = +0.2668 | positive folds = 4/4
VERDICT: CONDITIONAL EDGE — worth a phase 2
```

## Graceful degradation

`main.py` uses **XGBoost** if available, then **LightGBM**, then scikit-learn's
`GradientBoostingClassifier`, and finally a pure-NumPy logistic model — so the
harness always runs, even in a bare environment. AUC is computed via the
Mann-Whitney relation with no hard sklearn dependency.

## Auction mispricing demo (`auction_mispricing.py`)

An auction-specific initial analysis: detect **mispriced lots / forward ROI** on
structured auction data (art, watches, wine, cars, collectibles). It ships a
fully-documented variable dictionary and a realistic synthetic auction dataset so
the whole methodology runs today.

```bash
python auction_mispricing.py
```

The honest result it produces on the synthetic data:

```
Baseline (market-implied pre-sale estimate gap): AUC ~0.50  -> no forward-return skill
Model (XGBoost, walk-forward OOS):               AUC ~0.61  -> beats baseline 4/4 folds

Economic significance (net of 25% buyer premium + 10% seller fee):
  All-lots mean net log-ROI       = -0.13   (the average lot LOSES money after costs)
  Baseline top-decile net log-ROI = -0.13   (estimate gap does not help)
  Model top-decile net log-ROI    = +0.02   (the model finds the minority that clears costs)

Label-permutation null test: real AUC 0.61 vs null 0.50, p = 0.032
```

The point it makes: costs are a high bar, most lots are not worth buying, and the
value of the model is *selection* — surfacing the few lots whose forward return
survives fees. A model that only beat a random baseline would be worthless here.

## Files

| File | Role |
|------|------|
| `backtest.py` | Walk-forward engine, fold accounting, verdict logic |
| `leakage_audit.py` | Target-leak / duplicate screens run before modelling |
| `main.py` | Generic entry point: load → audit → backtest → verdict |
| `auction_mispricing.py` | Auction-specific analysis + variable dictionary + costs + permutation test |

## Scope

This is the phase-1 harness: clean, define target, baseline, GBM, out-of-sample
backtest, feature drivers, verdict. Phase 2 (cost-adjusted returns, expanded
features/history, deployment hardening) builds on top of it.

---

Dr. Sandeep Grover — PhD in Data Science. Predictive modelling and statistical
validation on structured data.
