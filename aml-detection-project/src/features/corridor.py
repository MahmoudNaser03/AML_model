"""Corridor and causal graph features."""

import numpy as np
import pandas as pd

from src.features.helpers import add_group_cumcount, safe_ratio
from src.features.history import CORRIDOR


def build_corridor_features(df: pd.DataFrame) -> None:
    """Add corridor-level and derived features in-place."""
    add_group_cumcount(df, CORRIDOR, "corridor_prev_count")

    df["corridor_share_sender"]    = safe_ratio(df.corridor_prev_count.to_numpy(), df.sender_prev_count.to_numpy()   + 1)
    df["corridor_share_receiver"]  = safe_ratio(df.corridor_prev_count.to_numpy(), df.receiver_prev_count.to_numpy() + 1)
    df["corridor_established_anomaly"] = (1 / np.sqrt(df.corridor_prev_count.to_numpy() + 1)).astype("float32")
    df["corridor_share_x_anomaly"] = (df.corridor_share_sender * df.corridor_established_anomaly).astype("float32")
