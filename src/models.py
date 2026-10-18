"""Model zoo. All use class_weight/balanced settings; threshold stays 0.5."""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from .config import SEED


def make_logreg():
    return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)


def make_rf():
    return RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                  class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=SEED)


def make_et():
    return ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2,
                                class_weight="balanced_subsample",
                                n_jobs=-1, random_state=SEED)


def make_hgb():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                         max_leaf_nodes=31, l2_regularization=1.0,
                                         class_weight="balanced", random_state=SEED)


def make_xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.04,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                         scale_pos_weight=1.0, n_jobs=-1, random_state=SEED,
                         tree_method="hist", eval_metric="logloss")


def make_lgbm():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=800, num_leaves=63, learning_rate=0.03,
                         feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=3,
                         lambda_l2=2.0, class_weight="balanced",
                         n_jobs=-1, random_state=SEED, verbose=-1)


MODEL_FNS = {
    "logreg": make_logreg,
    "rf": make_rf,
    "extratrees": make_et,
    "histgb": make_hgb,
    "xgboost": make_xgb,
    "lightgbm": make_lgbm,
}
