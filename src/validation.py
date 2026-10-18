"""Preprocessing pipelines (fit on train fold only) + validation helpers."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import (f1_score, roc_auc_score, precision_score,
                             recall_score, confusion_matrix)


def make_preprocess(num_cols, cat_cols, scale_numeric=False):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    transformers = [
        ("num", Pipeline(num_steps), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ]
    return ColumnTransformer(transformers)


def combined_score(f1, auc):
    return 0.6 * f1 + 0.4 * auc


def fold_metrics(y_true, proba):
    pred = (proba >= 0.5).astype(int)  # ONLY permitted threshold
    return {
        "f1": f1_score(y_true, pred, zero_division=0),
        "auc": roc_auc_score(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "cm": confusion_matrix(y_true, pred).tolist(),
    }


def cross_validate(model_fn, X, y, num_cols, cat_cols, n_splits=5, seed=42, groups=None):
    """model_fn: callable returning fresh unfitted estimator. Stratified or GroupKFold."""
    import numpy as np
    results = []
    if groups is None:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = skf.split(X, y)
    else:
        gkf = GroupKFold(n_splits=n_splits)
        splits = gkf.split(X, y, groups)
    oof = np.zeros(len(X))
    for tr_idx, va_idx in splits:
        pre = make_preprocess(num_cols, cat_cols,
                              scale_numeric=model_fn.__name__ == "make_logreg")
        clf = model_fn()
        from sklearn.pipeline import Pipeline as PL
        pipe = PL([("pre", pre), ("clf", clf)])
        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        proba = pipe.predict_proba(X.iloc[va_idx])[:, 1]
        oof[va_idx] = proba
        m = fold_metrics(y.iloc[va_idx].values, proba)
        m["combined"] = combined_score(m["f1"], m["auc"])
        results.append(m)
    overall = fold_metrics(y.values, oof)
    overall["combined"] = combined_score(overall["f1"], overall["auc"])
    return results, overall, oof
