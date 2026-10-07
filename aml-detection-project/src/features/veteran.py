"""
V3 veteran-account and established-behaviour anomaly features.

Adds features that describe unusual behaviour relative to a sender's
own long-term history — addresses V2 recall failure for accounts with
100+ prior transactions.
"""

import numpy as np
import pandas as pd


def build_veteran_features(df: pd.DataFrame) -> None:
    """Add veteran-account features in-place. df must already have V2 features."""
    sender_prev = df["sender_prev_count"].to_numpy(dtype="float32")
    veteran     = (sender_prev >= 100).astype(np.float32)

    df["sender_veteran_flag"]  = veteran.astype("int8")
    df["sender_history_log"]   = np.log1p(sender_prev).astype("float32")

    sa20 = df["sender_activity_20"].to_numpy(dtype="float32")
    df["sender_activity_shift_20_history"] = ((sa20 + 1.0) / (sender_prev / 25.0 + 1.0)).clip(0, 100).astype("float32")
    df["sender_activity_share_20_all"]     = ((sa20 + 1.0) / (sender_prev + 1.0)).clip(0, 1).astype("float32")

    su = df["sender_unique_out"].to_numpy(dtype="float32")
    df["sender_diversity_per_history"]  = ((su + 1.0) / (sender_prev + 1.0)).clip(0, 1).astype("float32")
    df["sender_diversity_x_veteran"]    = (su * veteran).astype("float32")

    pair_is_new = df["pair_is_new"].to_numpy(dtype="float32")
    df["new_pair_x_veteran"]                   = (pair_is_new * veteran).astype("float32")
    df["new_pair_existing_sender_x_veteran"]   = (df["new_pair_existing_sender"].to_numpy(dtype="float32") * veteran).astype("float32")

    pair_prev = df["pair_prev_count"].to_numpy(dtype="float32")
    df["pair_share_of_sender_history"] = ((pair_prev + 1.0) / (sender_prev + 1.0)).clip(0, 1).astype("float32")
    df["pair_velocity_x_veteran"]      = (df["pair_velocity_relative"].to_numpy(dtype="float32") * (1.0 + veteran)).clip(-100, 100).astype("float32")
    df["pair_activity_x_veteran"]      = (df["pair_activity_vs_sender_20_100"].to_numpy(dtype="float32") * (1.0 + veteran)).clip(-100, 100).astype("float32")

    amount_paid      = df["amount_paid"].to_numpy(dtype="float32")
    sender_mean_20   = df["sender_paid_mean_20"].to_numpy(dtype="float32")
    sender_mean_100  = df["sender_paid_mean_100"].to_numpy(dtype="float32")
    df["sender_amount_vs_mean_20"]    = ((amount_paid + 1.0) / (sender_mean_20  + 1.0)).clip(0, 1000).astype("float32")
    df["sender_amount_vs_mean_100"]   = ((amount_paid + 1.0) / (sender_mean_100 + 1.0)).clip(0, 1000).astype("float32")
    df["sender_amount_regime_change"] = ((sender_mean_20 + 1.0) / (sender_mean_100 + 1.0)).clip(0, 100).astype("float32")

    df["network_expansion_x_veteran"]        = (df["sender_network_expansion"].to_numpy(dtype="float32")   * (1.0 + veteran)).clip(-100, 100).astype("float32")
    df["sender_diversity_ratio_x_veteran"]   = (df["sender_out_diversity_ratio"].to_numpy(dtype="float32") * (1.0 + veteran)).clip(-100, 100).astype("float32")
    df["veteran_currency_novelty"]           = (veteran * df["sender_currency_novelty"].to_numpy(dtype="float32")).astype("float32")
    df["veteran_format_novelty"]             = (veteran * df["sender_format_novelty"].to_numpy(dtype="float32")).astype("float32")
    df["veteran_novelty_score"]              = (veteran * (
        df["new_pair_existing_sender"].to_numpy(dtype="float32") +
        df["sender_currency_novelty"].to_numpy(dtype="float32") +
        df["sender_format_novelty"].to_numpy(dtype="float32")
    )).astype("float32")

    pair_velocity     = np.clip(df["pair_velocity_relative"].to_numpy(dtype="float32"), 0, 1e6)
    network_expansion = np.clip(df["sender_network_expansion"].to_numpy(dtype="float32"), 0, 1e6)
    amount_regime     = np.clip(df["sender_amount_regime_change"].to_numpy(dtype="float32"), 0, 1e6)
    df["veteran_behavior_shift"] = (veteran * (
        np.log1p(pair_velocity) + np.log1p(network_expansion) + np.log1p(amount_regime)
    )).clip(-100, 100).astype("float32")
