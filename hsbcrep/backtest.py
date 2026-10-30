"""HSBC-style rolling backtest (Ciceri et al., arXiv:2509.17715, Sec. 3.3), plus a purged alternative.

Rolling protocol, as in the paper: events are processed in time order in test blocks. For each block, the training
set is the `n_train` most recent events that end at least `blind` before the block's first event. Each model family
is tuned by grid-search CV on the training set; the family with the best in-sample CV AUC predicts the block.
Pooled out-of-sample AUC and the median of per-day AUCs are reported.

Model grids follow Table A.3 of the paper, thinned where noted so the full study runs on a laptop.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:                                    # fall back to sklearn's histogram GBM
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAVE_XGB = False

N_JOBS = 1


def families(full=False):
    """Model families and grids. full=True uses the paper's grids; the default thins them (about 10x faster)."""
    lr = (make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000, solver="lbfgs")),
          {"logisticregression__C": list(np.logspace(-4, 4, 9 if full else 5))})
    rf = (RandomForestClassifier(random_state=0, n_jobs=1),
          {"criterion": ["gini", "entropy", "log_loss"] if full else ["gini"],
           "n_estimators": list(range(80, 181, 20)) if full else [100], "min_samples_leaf": [1, 5]})
    if HAVE_XGB:
        gb = (XGBClassifier(eval_metric="logloss", verbosity=0, n_jobs=1, random_state=0),
              {"max_depth": [3, 5, 7, 9, 11] if full else [3, 5], "n_estimators": list(range(80, 181, 20)) if full else [100],
               "learning_rate": [0.15, 0.1, 0.05, 0.01] if full else [0.1, 0.05]})
    else:
        gb = (HistGradientBoostingClassifier(random_state=0),
              {"max_depth": [3, 5], "learning_rate": [0.1, 0.05], "max_iter": [100]})
    nn = (make_pipeline(StandardScaler(), MLPClassifier(max_iter=10000, random_state=0)),
          {"mlpclassifier__hidden_layer_sizes": [(64,), (64, 32)] if not full else [(32,), (64,), (128,), (64, 32), (128, 64)],
           "mlpclassifier__activation": ["relu", "tanh"] if not full else ["relu", "logistic", "tanh"],
           "mlpclassifier__alpha": [1e-3, 1e-1]})
    return {"LR": lr, "XGB": gb, "RF": rf, "NN": nn}


def _fit_best(X, y, fams, seed):
    best, best_auc, best_name = None, -1, None
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for name, (est, grid) in fams.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gs = GridSearchCV(est, grid, cv=cv, scoring="roc_auc", n_jobs=N_JOBS).fit(X, y)
        if gs.best_score_ > best_auc:
            best, best_auc, best_name = gs.best_estimator_, gs.best_score_, name
    return best, best_name, best_auc


def rolling(X, y, t_ms, day, n_train=200, block=25, blind_days=0.0, fams=None, seed=0, only=None):
    """HSBC rolling backtest. X rows must be sorted by time. Returns per-event predictions."""
    X, y, t_ms = np.asarray(X, float), np.asarray(y, int), np.asarray(t_ms, np.int64)
    order = np.argsort(t_ms, kind="stable")
    assert (order == np.arange(len(order))).all(), "sort events by time first"
    fams = fams or families()
    if only:
        fams = {k: fams[k] for k in only}
    blind_ms = int(blind_days * 86_400_000)
    rows = []
    start = n_train
    for b0 in range(0, len(y), block):
        b1 = min(b0 + block, len(y))
        cut = t_ms[b0] - blind_ms
        avail = np.flatnonzero(t_ms < cut) if blind_ms > 0 else np.arange(b0)
        if len(avail) < n_train or len(np.unique(y[avail[-n_train:]])) < 2:
            continue
        tr = avail[-n_train:]
        m, name, cv_auc = _fit_best(X[tr], y[tr], fams, seed)
        p = m.predict_proba(X[b0:b1])[:, 1]
        for k, i in enumerate(range(b0, b1)):
            rows.append({"i": i, "p": p[k], "y": y[i], "day": day[i], "model": name, "cv_auc": cv_auc})
    return pd.DataFrame(rows)


def summarize(pred):
    """Pooled AUC and median per-day AUC (HSBC report medians over windows)."""
    if len(pred) == 0 or pred.y.nunique() < 2:
        return {"pooled_auc": np.nan, "median_day_auc": np.nan, "n": len(pred)}
    days = [roc_auc_score(g.y, g.p) for _, g in pred.groupby("day") if g.y.nunique() == 2]
    return {"pooled_auc": roc_auc_score(pred.y, pred.p), "median_day_auc": float(np.median(days)) if days else np.nan,
            "n": len(pred)}


def purged_cv(X, y, t_ms, n_folds=5, embargo_days=1.0, fams=None, seed=0):
    """Leak-safe check: contiguous time folds; training excludes an embargo band around each test fold."""
    X, y, t_ms = np.asarray(X, float), np.asarray(y, int), np.asarray(t_ms, np.int64)
    fams = fams or families()
    edges = np.linspace(0, len(y), n_folds + 1).astype(int)
    emb = int(embargo_days * 86_400_000)
    out = np.full(len(y), np.nan)
    for f in range(n_folds):
        te = np.arange(edges[f], edges[f + 1])
        lo, hi = t_ms[te[0]] - emb, t_ms[te[-1]] + emb
        tr = np.flatnonzero((t_ms < lo) | (t_ms > hi))
        m, _, _ = _fit_best(X[tr], y[tr], fams, seed)
        out[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, out)
