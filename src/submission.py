"""Submission builder with automatic verification."""
import pandas as pd
from .config import ID_COL, TARGET_F1, TARGET_RAUC


def build_submission(test_ids: pd.Series, proba, sample_path=None) -> pd.DataFrame:
    import numpy as np
    proba = np.clip(np.asarray(proba, dtype=float), 0.0, 1.0)
    sub = pd.DataFrame({ID_COL: test_ids.values, TARGET_RAUC: proba})
    sub[TARGET_F1] = (sub[TARGET_RAUC] >= 0.5).astype(int)  # ONLY allowed threshold
    sub = sub[[ID_COL, TARGET_F1, TARGET_RAUC]]
    return sub


def verify_submission(sub: pd.DataFrame, test: pd.DataFrame, sample_path=None):
    errors = []
    if list(sub.columns) != [ID_COL, TARGET_F1, TARGET_RAUC]:
        errors.append(f"columns must be exactly {[ID_COL, TARGET_F1, TARGET_RAUC]}, got {list(sub.columns)}")
    if len(sub) != len(test):
        errors.append(f"row count {len(sub)} != test rows {len(test)}")
    if sub[ID_COL].duplicated().sum() > 0:
        errors.append("duplicate IDs in submission")
    if set(sub[ID_COL]) != set(test[ID_COL]):
        errors.append("submission IDs do not match test IDs exactly")
    if sub[TARGET_F1].isnull().sum() or sub[TARGET_RAUC].isnull().sum():
        errors.append("missing predictions")
    if not set(sub[TARGET_F1].unique()).issubset({0, 1}):
        errors.append("TargetF1 must contain only 0/1")
    if ((sub[TARGET_RAUC] < 0) | (sub[TARGET_RAUC] > 1)).any():
        errors.append("TargetRAUC out of [0,1]")
    expected_f1 = (sub[TARGET_RAUC] >= 0.5).astype(int)
    if not (expected_f1 == sub[TARGET_F1]).all():
        errors.append("TargetF1 does not correspond to TargetRAUC at threshold 0.5")
    if sample_path is not None:
        sample = pd.read_csv(sample_path)
        if list(sample.columns) != list(sub.columns):
            errors.append("column mismatch vs SampleSubmission.csv")
    return errors
