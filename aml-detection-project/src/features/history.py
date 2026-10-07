"""Causal sender / receiver / pair / reciprocal history features."""

import numpy as np
import pandas as pd

from src.features.helpers import add_group_cumcount, safe_ratio


# Public group-key definitions (used by downstream modules)
S        = ["from_bank", "from_account"]
R        = ["to_bank",   "to_account"]
P        = ["from_bank", "from_account", "to_bank",   "to_account"]
RP       = ["to_bank",   "to_account",   "from_bank", "from_account"]
CORRIDOR = ["from_bank", "to_bank", "payment_currency", "receiving_currency"]


def build_history_features(df: pd.DataFrame) -> None:
    """Add causal account/pair/reciprocal history features in-place."""
    add_group_cumcount(df, S,        "sender_prev_count")
    add_group_cumcount(df, R,        "receiver_prev_count")
    add_group_cumcount(df, P,        "pair_prev_count")
    add_group_cumcount(df, RP,       "reverse_pair_prev_count")
    add_group_cumcount(df, CORRIDOR, "currency_pair_prev_count")

    for keys, prefix in [(S, "sender"), (R, "receiver"), (P, "pair")]:
        prev = df.groupby(keys, sort=False, observed=True)["ts"].shift(1)
        df[f"{prefix}_delta_sec"] = (
            (df.ts - prev).dt.total_seconds()
            .fillna(1e9).clip(0, 1e9).astype("float32")
        )
        df[f"{prefix}_delta_log"] = np.log1p(df[f"{prefix}_delta_sec"]).astype("float32")

    df["pair_velocity_relative"] = (
        df.pair_prev_count / np.maximum(df.sender_prev_count + 1, 1)
    ).astype("float32")

    df["reciprocal_score"]     = (df.reverse_pair_prev_count > 0).astype("int8")
    df["reciprocal_intensity"] = (
        df.reverse_pair_prev_count / np.maximum(df.pair_prev_count + 1, 1)
    ).astype("float32")
