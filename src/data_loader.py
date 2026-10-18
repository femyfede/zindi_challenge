"""Data loading + merge. Join key = ID (verified: 100% coverage, deathdate consistent)."""
import pandas as pd
from .config import ID_COL, TARGET, TRAIN_PATH, TEST_PATH, CLIMATE_PATH


def load_data(train_path=TRAIN_PATH, test_path=TEST_PATH, climate_path=CLIMATE_PATH):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    climate = pd.read_csv(climate_path)
    return train, test, climate


def merge_climate(base: pd.DataFrame, climate: pd.DataFrame) -> pd.DataFrame:
    """Left-join climate features on ID. Drops duplicate deathdate from climate side."""
    clim = climate.drop(columns=["deathdate"], errors="ignore")
    # hot_days_30d is constant 0 -> drop to avoid zero-variance column
    clim = clim.drop(columns=["hot_days_30d"], errors="ignore")
    merged = base.merge(clim, on=ID_COL, how="left", validate="one_to_one")
    return merged


def get_xy(df: pd.DataFrame, feature_cols):
    X = df[feature_cols].copy()
    y = df[TARGET].copy() if TARGET in df.columns else None
    return X, y
