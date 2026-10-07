"""V2 network / fan-in / fan-out features."""

import numpy as np
import pandas as pd

from src.features.helpers import safe_ratio


def build_network_features(df: pd.DataFrame) -> None:
    """Add hub, fan-in/fan-out, and diversity features in-place."""
    df["fanin_burst_20"]  = df.receiver_activity_20.astype("float32")
    df["fanout_burst_20"] = df.sender_activity_20.astype("float32")

    df["fanout_diversity_100"] = safe_ratio(df.sender_unique_out.to_numpy(),   df.sender_activity_100.to_numpy()   + 1)
    df["fanin_diversity_100"]  = safe_ratio(df.receiver_unique_in.to_numpy(),  df.receiver_activity_100.to_numpy() + 1)

    df["hub_mix_20"]  = (df.fanin_burst_20  * df.fanout_burst_20).astype("float32")
    df["hub_mix_100"] = np.sqrt(
        np.maximum(df.sender_prev_count, 0) * np.maximum(df.receiver_prev_count, 0)
    ).astype("float32")

    df["sender_out_diversity_ratio"]  = safe_ratio(df.sender_unique_out.to_numpy(),  df.sender_prev_count.to_numpy()   + 1)
    df["receiver_in_diversity_ratio"] = safe_ratio(df.receiver_unique_in.to_numpy(), df.receiver_prev_count.to_numpy() + 1)
