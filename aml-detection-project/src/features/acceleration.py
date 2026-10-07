"""V2 behavioural acceleration and regime-change features."""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.features.helpers import safe_ratio


def build_acceleration_features(df: pd.DataFrame) -> None:
    """Add short-vs-long-term ratio features in-place."""
    specs = [
        ("sender_activity_20",     "sender_activity_100",   "sender_activity_accel_20_100"),
        ("receiver_activity_20",   "receiver_activity_100", "receiver_activity_accel_20_100"),
        ("pair_activity_20",       "sender_activity_100",   "pair_activity_vs_sender_20_100"),
        ("sender_paid_mean_20",    "sender_paid_mean_100",  "sender_paid_accel_20_100"),
        ("sender_received_mean_20","sender_received_mean_100","sender_received_accel_20_100"),
        ("receiver_paid_mean_20",  "receiver_paid_mean_100","receiver_paid_accel_20_100"),
        ("receiver_received_mean_20","receiver_received_mean_100","receiver_received_accel_20_100"),
    ]
    for num, den, out in tqdm(specs, desc="Behaviour acceleration", unit="feature"):
        df[out] = safe_ratio(df[num].to_numpy(), np.maximum(df[den].to_numpy(), 1e-6))

    df["sender_amount_regime_change"]   = safe_ratio(df.sender_paid_mean_20.to_numpy()      + 1, df.sender_paid_mean_500.to_numpy()      + 1)
    df["receiver_amount_regime_change"] = safe_ratio(df.receiver_received_mean_20.to_numpy()+ 1, df.receiver_received_mean_500.to_numpy()+ 1)

    df["sender_network_expansion"]   = safe_ratio(df.sender_unique_out.to_numpy()   + 1, df.sender_activity_100.to_numpy()   / 25 + 1)
    df["receiver_network_expansion"] = safe_ratio(df.receiver_unique_in.to_numpy()  + 1, df.receiver_activity_100.to_numpy() / 25 + 1)

    df["paid_vs_sender_mean_20_v3"]        = safe_ratio(df.amount_paid.to_numpy(),     np.maximum(df.sender_paid_mean_20.to_numpy(),       1e-6))
    df["received_vs_sender_mean_20_v3"]    = safe_ratio(df.amount_received.to_numpy(), np.maximum(df.sender_received_mean_20.to_numpy(),    1e-6))
    df["paid_vs_receiver_mean_100_v3"]     = safe_ratio(df.amount_paid.to_numpy(),     np.maximum(df.receiver_paid_mean_100.to_numpy(),     1e-6))
    df["received_vs_receiver_mean_100_v3"] = safe_ratio(df.amount_received.to_numpy(), np.maximum(df.receiver_received_mean_100.to_numpy(), 1e-6))
