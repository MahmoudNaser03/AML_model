"""V2 reciprocal / reverse-transaction features."""

import numpy as np
import pandas as pd

from src.features.helpers import safe_ratio


def build_reciprocal_features(df: pd.DataFrame) -> None:
    """Add reverse-pair timing and amount features in-place."""
    reverse_key = ["to_bank", "to_account", "from_bank", "from_account"]

    prev_reverse_received = df.groupby(reverse_key, sort=False, observed=True)["amount_received"].shift(1)
    prev_reverse_paid     = df.groupby(reverse_key, sort=False, observed=True)["amount_paid"].shift(1)
    prev_reverse_ts       = df.groupby(reverse_key, sort=False, observed=True)["ts"].shift(1)

    df["reverse_pair_delta_sec_v3"] = (
        (df.ts - prev_reverse_ts).dt.total_seconds().fillna(1e9).clip(0, 1e9).astype("float32")
    )
    df["reverse_pair_delta_log_v3"] = np.log1p(df.reverse_pair_delta_sec_v3).astype("float32")

    df["reverse_received_ratio"] = safe_ratio(
        df.amount_received.to_numpy(),
        np.maximum(prev_reverse_received.fillna(0).to_numpy(), 1e-6),
    )
    df["reverse_paid_ratio"] = safe_ratio(
        df.amount_paid.to_numpy(),
        np.maximum(prev_reverse_paid.fillna(0).to_numpy(), 1e-6),
    )
    df["reverse_amount_gap"] = (
        np.abs(df.amount_received.to_numpy() - prev_reverse_paid.fillna(0).to_numpy())
    ).clip(0, 1e9).astype("float32")
