"""Tasks 5-8: out-of-time validation, model comparison, class imbalance, scoring."""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .config import SEED
from .features import CATEGORICAL


# --------------------------------------------------------------------------- models
class RecencyRule:
    """Rule-based baseline: flag churn when the member has not bought for >= k months.
    The score (months since last purchase) is used for PR-AUC; k is tuned on validation."""
    name = "Rule: recency >= k months"

    def fit(self, X, y):
        best = max(range(1, 7), key=lambda k: f1_score(y, (X["RECENCY_MONTHS"] >= k).astype(int)))
        self.k_ = best
        return self

    def predict_proba(self, X):
        s = (X["RECENCY_MONTHS"].clip(0, 12) / 12).values
        return np.c_[1 - s, s]


def _split_cols(features):
    cat = [c for c in features if c in CATEGORICAL]
    num = [c for c in features if c not in CATEGORICAL]
    return num, cat


def make_logreg(features):
    num, cat = _split_cols(features)
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), cat),
    ])
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000))])


def make_rf(features):
    num, cat = _split_cols(features)
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), num),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat),
    ])
    return Pipeline([("pre", pre),
                     ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                                    class_weight="balanced_subsample",
                                                    n_jobs=-1, random_state=SEED))])


class LGBM:
    """LightGBM with native categorical handling. `balanced` toggles class weighting."""

    def __init__(self, features, balanced=False):
        self.features, self.balanced = features, balanced
        self.model = LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15,
                                    min_child_samples=30, subsample=0.8, subsample_freq=1,
                                    colsample_bytree=0.8, reg_lambda=1.0,
                                    class_weight="balanced" if balanced else None,
                                    random_state=SEED, verbose=-1)

    def _prep(self, X):
        X = X[self.features].copy()
        for c in X.columns:
            if c in CATEGORICAL:
                X[c] = pd.Categorical(X[c], categories=self.cats_[c])
        return X

    def fit(self, X, y):
        self.cats_ = {c: sorted(X[c].dropna().unique()) for c in self.features if c in CATEGORICAL}
        self.model.fit(self._prep(X), y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self._prep(X))


def model_zoo(features):
    return {
        "Rule baseline (recency)": lambda: RecencyRule(),
        "Logistic regression (balanced)": lambda: make_logreg(features),
        "Random forest (balanced)": lambda: make_rf(features),
        "LightGBM (balanced weights)": lambda: LGBM(features, balanced=True),
        "LightGBM": lambda: LGBM(features, balanced=False),
    }


# --------------------------------------------------------------------------- metrics
def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    i = int(np.nanargmax(f1[:-1]))
    return float(thr[i])


def scores(y, p, thr):
    yhat = (p >= thr).astype(int)
    return {
        "n": len(y), "churn_rate": float(np.mean(y)),
        "accuracy": accuracy_score(y, yhat), "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat), "f1": f1_score(y, yhat),
        "pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p), "threshold": thr,
    }


def top_k_lift(y, p, frac=0.1):
    n = max(1, int(len(y) * frac))
    idx = np.argsort(-p)[:n]
    return float(np.mean(np.asarray(y)[idx]) / np.mean(y))


def fit_predict(factory, train, test, features):
    m = factory()
    if isinstance(m, RecencyRule):
        m.fit(train, train["CHURNED"])
    else:
        m.fit(train[features], train["CHURNED"])
    return m, m.predict_proba(test[features] if not isinstance(m, RecencyRule) else test)[:, 1]


def threshold_for(m, p_valid, valid):
    if isinstance(m, RecencyRule):
        return m.k_ / 12 - 1e-9
    return best_f1_threshold(valid["CHURNED"], p_valid)
