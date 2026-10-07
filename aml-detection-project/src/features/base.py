"""Base transaction-level and local-anomaly features."""

import numpy as np
import pandas as pd

from src.features.helpers import safe_ratio


def build_base_features(df: pd.DataFrame) -> None:
    """Add base features to df in-place."""
    df["hour"]          = df.ts.dt.hour.astype("int8")
    df["minute"]        = df.ts.dt.minute.astype("int8")
    df["minute_of_day"] = (df.ts.dt.hour * 60 + df.ts.dt.minute).astype("int16")
    df["day_of_month"]  = df.ts.dt.day.astype("int8")
    df["month"]         = df.ts.dt.month.astype("int8")

    paid     = df.amount_paid.to_numpy(dtype=np.float64)
    received = df.amount_received.to_numpy(dtype=np.float64)

    df["amount_received_log"] = np.log1p(np.maximum(received, 0)).astype("float32")
    df["amount_paid_log"]     = np.log1p(np.maximum(paid,     0)).astype("float32")

    df["flow_gap"]           = (received - paid).astype("float32")
    df["flow_abs_diff"]      = np.abs(received - paid).astype("float32")
    df["flow_relative_diff"] = (
        np.abs(received - paid) /
        np.maximum(np.maximum(np.abs(received), np.abs(paid)), 1e-6)
    ).clip(0, 1e6).astype("float32")

    df["received_paid_ratio"] = safe_ratio(received, np.maximum(np.abs(paid),     1e-6))
    df["paid_received_ratio"] = safe_ratio(paid,     np.maximum(np.abs(received), 1e-6))

    df["amount_max"]         = np.maximum(paid, received).astype("float32")
    df["amount_min"]         = np.minimum(paid, received).astype("float32")
    df["amount_range_ratio"] = safe_ratio(df["amount_max"], np.maximum(df["amount_min"], 1e-6))

    df["same_bank"]     = (df.from_bank.astype("string").to_numpy() ==
                           df.to_bank.astype("string").to_numpy()).astype("int8")
    df["cross_currency"] = (df.payment_currency.astype("string").to_numpy() !=
                             df.receiving_currency.astype("string").to_numpy()).astype("int8")

    df["paid_under_1k"]    = (paid < 1_000).astype("int8")
    df["paid_below_500"]   = (paid <   500).astype("int8")
    df["paid_below_50000"] = (paid < 50_000).astype("int8")

    hour = df["hour"].to_numpy(dtype=np.float32)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype("float32")
