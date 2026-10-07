"""
Model construction, training, and inference utilities.
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

from src.config import (
    SEED, XGB_PARAMS, BEST_K, CLUSTER_SAMPLE,
    NON_FEATURE_COLS, EXTERNAL_PATTERN_COLS, DROP_FEATURE_COLS,
    PATTERN_BLOCKLIST_TOKENS, STRUCTURAL_RENAME,
)


# ── XGBoost ───────────────────────────────────────────────────────────────────

def build_xgb_model(y_train: np.ndarray, feature_cols: list[str]) -> xgb.XGBClassifier:
    """Return a configured XGBClassifier with scale_pos_weight set from y_train."""
    neg = int((y_train == 0).sum())
    pos = int(y_train.sum())
    params = dict(XGB_PARAMS)
    params["scale_pos_weight"]    = float(np.sqrt(neg / max(pos, 1)))
    params["monotone_constraints"] = tuple(0 for _ in feature_cols)
    params["n_jobs"]               = max(1, os.cpu_count() - 2)
    return xgb.XGBClassifier(**params)


# ── Feature matrix builder ────────────────────────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame,
    train_idx: np.ndarray,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Enforce the experiment rule:
      TRUE Pattern TXT membership → create A/B masks → DROP TXT cols → build X.

    Returns (X, feature_cols).
    """
    df = df.rename(columns=STRUCTURAL_RENAME)

    if "txt_patterned" not in df.columns:
        raise AssertionError("TRUE TXT label 'txt_patterned' is missing.")

    df_model = df.drop(
        columns=[c for c in EXTERNAL_PATTERN_COLS if c in df.columns],
        errors="ignore",
    ).copy()

    feature_cols = []
    for c in df_model.columns:
        if c in NON_FEATURE_COLS or c in DROP_FEATURE_COLS:
            continue
        if not pd.api.types.is_numeric_dtype(df_model[c]):
            continue
        feature_cols.append(c)

    # Drop train-constant features
    train_var  = df_model.iloc[train_idx][feature_cols].var()
    constant   = train_var[train_var <= 1e-12].index.tolist()
    feature_cols = [c for c in feature_cols if c not in constant]

    X = (
        df_model[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype("float32")
    )

    if not np.isfinite(X.to_numpy()).all():
        raise AssertionError("Non-finite values remain in X.")

    bad_external = [
        c for c in feature_cols
        if any(t in c.lower() for t in PATTERN_BLOCKLIST_TOKENS)
    ]
    if bad_external:
        raise AssertionError(f"Pattern-derived feature names remain: {bad_external}")

    return X, feature_cols


# ── Clustering ────────────────────────────────────────────────────────────────

CLUSTER_FEATURES = [
    "amount_paid_log", "amount_received",
    "sender_prev_count", "sender_unique_out", "sender_diversity",
    "sender_network_expansion", "sender_activity_100",
    "sender_delta_sec", "sender_delta_log",
    "receiver_prev_count", "receiver_unique_in", "receiver_diversity",
    "receiver_network_expansion", "receiver_delta_sec", "receiver_delta_log",
    "pair_prev_count", "pair_velocity_relative", "pair_activity_20",
    "pair_activity_vs_sender_20_100", "pair_delta_sec",
    "reverse_pair_prev_count", "reciprocal_intensity",
    "new_pair_existing_sender", "sender_currency_novelty",
    "receiver_currency_novelty", "sender_format_novelty",
    "sender_out_diversity_ratio", "corridor_share_receiver",
    "corridor_established_anomaly", "corridor_share_x_anomaly",
    "hub_mix_20", "hub_mix_100",
]


def fit_cluster_model(
    X: pd.DataFrame,
    train_idx: np.ndarray,
    k: int = BEST_K,
    sample: int = CLUSTER_SAMPLE,
    seed: int = SEED,
) -> tuple[StandardScaler, KMeans, list[str]]:
    """Fit scaler + K-Means on TRAIN data only. Returns (scaler, model, features)."""
    available = [c for c in CLUSTER_FEATURES if c in X.columns]
    X_train   = X.iloc[train_idx][available].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")

    scaler = StandardScaler()
    scaler.fit(X_train)

    rng   = np.random.RandomState(seed)
    n_samp = min(sample, len(X_train))
    idx   = rng.choice(len(X_train), size=n_samp, replace=False)
    Z     = scaler.transform(X_train.iloc[idx])

    km = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300)
    km.fit(Z)
    return scaler, km, available


def add_cluster_features(
    X: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    scaler: StandardScaler,
    km: KMeans,
    cluster_features: list[str],
    feature_cols: list[str],
) -> None:
    """Add cluster_id and cluster_distance to X in-place and update feature_cols."""
    def _assign(idxs, label):
        part = X.iloc[idxs][cluster_features].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
        Z    = scaler.transform(part)
        lbls = km.predict(Z)
        dists = km.transform(Z)[np.arange(len(lbls)), lbls]
        return pd.DataFrame({"cluster_id": lbls.astype("int8"),
                              "cluster_distance": dists.astype("float32")},
                            index=X.index[idxs])

    X["cluster_id"]       = np.nan
    X["cluster_distance"] = np.nan
    for idxs, name in [(train_idx, "TRAIN"), (val_idx, "VAL"), (test_idx, "TEST")]:
        res = _assign(idxs, name)
        X.loc[res.index, "cluster_id"]       = res["cluster_id"].values
        X.loc[res.index, "cluster_distance"] = res["cluster_distance"].values

    X["cluster_id"]       = X["cluster_id"].astype("int8")
    X["cluster_distance"] = X["cluster_distance"].astype("float32")

    for c in ["cluster_id", "cluster_distance"]:
        if c not in feature_cols:
            feature_cols.append(c)
