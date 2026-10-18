"""Target encoding utilities with proper in-fold fitting."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def target_encode_oof(X, y, cat_col, n_splits=5, smooth=30, seed=42, noise=0.0):
    """Out-of-fold target encoding for a single categorical column.
    Returns OOF encoded values + global mean for test-time use."""
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    global_mean = y.mean()
    for ti, vi in skf.split(X, y):
        m = pd.DataFrame({"cat": X[cat_col].iloc[ti].values, "y": y[ti]})
        agg = m.groupby("cat")["y"].agg(["mean", "count"])
        enc = ((agg["mean"] * agg["count"] + global_mean * smooth) / (agg["count"] + smooth)).to_dict()
        oof[vi] = X[cat_col].iloc[vi].map(enc).fillna(global_mean).values
    if noise > 0:
        oof = oof * (1 + noise * np.random.randn(len(oof)))
    return oof, global_mean


def target_encode_full(X_train, y_train, X_test, cat_col, smooth=30):
    """Fit encoding on full train, apply to test."""
    global_mean = y_train.mean()
    m = pd.DataFrame({"cat": X_train[cat_col].values, "y": y_train.values})
    agg = m.groupby("cat")["y"].agg(["mean", "count"])
    enc = ((agg["mean"] * agg["count"] + global_mean * smooth) / (agg["count"] + smooth)).to_dict()
    train_enc = X_train[cat_col].map(enc).fillna(global_mean).values
    test_enc = X_test[cat_col].map(enc).fillna(global_mean).values
    return train_enc, test_enc, enc, global_mean


def spatial_features(df, n_clusters=8, seed=42):
    """Add spatial cluster features: distance to cluster centers, cluster ID."""
    from sklearn.cluster import KMeans
    coords = df[["latitude", "longitude"]].values
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(coords)
    df = df.copy()
    df["geo_cluster"] = km.labels_
    centers = km.cluster_centers_
    for c in range(n_clusters):
        df[f"dist_cluster_{c}"] = np.sqrt(
            (df["latitude"] - centers[c, 0]) ** 2 + (df["longitude"] - centers[c, 1]) ** 2)
    # nearest center distance
    df["dist_nearest_cluster"] = df[[f"dist_cluster_{c}" for c in range(n_clusters)]].min(axis=1)
    return df