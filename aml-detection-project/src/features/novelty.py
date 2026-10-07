"""V2 novelty and first-seen features."""

import numpy as np
import pandas as pd

from src.features.helpers import safe_ratio
from src.features.history import S, R, P, RP, CORRIDOR


def build_novelty_features(df: pd.DataFrame) -> None:
    """Add first-seen and novelty features in-place (df must be chronologically sorted)."""
    df["sender_is_new"]        = (~df.duplicated(S)).astype("int8")
    df["receiver_is_new"]      = (~df.duplicated(R)).astype("int8")
    df["pair_is_new"]          = (~df.duplicated(P)).astype("int8")
    df["reverse_pair_is_new"]  = (~df.duplicated(RP)).astype("int8")
    df["corridor_is_new"]      = (~df.duplicated(CORRIDOR)).astype("int8")

    df["sender_format_novelty"]   = (~df.duplicated(S + ["payment_format"])).astype("int8")
    df["sender_currency_novelty"] = (~df.duplicated(S + ["payment_currency"])).astype("int8")
    df["receiver_currency_novelty"] = (~df.duplicated(R + ["receiving_currency"])).astype("int8")

    sender_new_edge   = df["pair_is_new"].astype("int8")
    receiver_new_edge = df["reverse_pair_is_new"].astype("int8")

    df["sender_unique_out"] = (
        df.groupby(S, sort=False, observed=True)[sender_new_edge.name].cumsum()
    ).astype("float32")
    df["receiver_unique_in"] = (
        df.groupby(R, sort=False, observed=True)[receiver_new_edge.name].cumsum()
    ).astype("float32")

    # cumsum includes current new edge — subtract for causal count
    df["sender_unique_out"]  = (df["sender_unique_out"]  - sender_new_edge).clip(lower=0).astype("float32")
    df["receiver_unique_in"] = (df["receiver_unique_in"] - receiver_new_edge).clip(lower=0).astype("float32")

    df["sender_diversity"]   = safe_ratio(df.sender_unique_out,  np.maximum(df.sender_prev_count   + 1, 1))
    df["receiver_diversity"] = safe_ratio(df.receiver_unique_in, np.maximum(df.receiver_prev_count + 1, 1))

    df["new_sender_new_pair"]        = ((df.sender_is_new == 1) & (df.pair_is_new == 1)).astype("int8")
    df["new_pair_existing_sender"]   = ((df.sender_is_new == 0) & (df.pair_is_new == 1)).astype("int8")
    df["new_pair_cross_currency"]    = ((df.pair_is_new   == 1) & (df.cross_currency == 1)).astype("int8")
    df["new_pair_reverse_seen"]      = ((df.pair_is_new   == 1) & (df.reverse_pair_prev_count > 0)).astype("int8")
