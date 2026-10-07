"""
Vectorised causal rolling behavioural features.

Sorts each group once and uses vectorised cumulative sums —
no per-row Python loop, no look-ahead.
"""

import gc
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.features.helpers import safe_ratio
from src.features.history import S, R, P


def causal_group_rolling_mean(
    frame: pd.DataFrame,
    keys: list[str],
    value_col: str,
    windows: list[int],
    prefix: str,
) -> None:
    """Causal group-level rolling mean for each window size (in-place)."""
    codes  = frame.groupby(keys, sort=False, observed=True).ngroup().to_numpy()
    values = frame[value_col].to_numpy(dtype=np.float64)
    n      = len(frame)

    order  = np.argsort(codes, kind="stable")
    sg, sv = codes[order], values[order]

    starts = np.r_[0, np.flatnonzero(sg[1:] != sg[:-1]) + 1]
    ends   = np.r_[starts[1:], n]
    counts = ends - starts
    group_start = np.repeat(starts, counts)
    pos         = np.arange(n, dtype=np.int64) - group_start

    cs_global   = np.cumsum(sv, dtype=np.float64)
    group_totals = np.add.reduceat(sv, starts)
    offsets      = np.r_[0, np.cumsum(group_totals[:-1])]
    cs           = cs_global - np.repeat(offsets, counts)
    prev_cs      = cs - sv

    for w in tqdm(windows, desc=f"{prefix}: rolling windows", unit="window"):
        left_pos  = pos - w - 1
        left_idx  = group_start + left_pos
        left_cs   = np.zeros(n, dtype=np.float64)
        mask      = left_pos >= 0
        left_cs[mask] = cs[left_idx[mask]]

        window_sum   = prev_cs - left_cs
        window_count = np.minimum(pos, w).astype(np.float64)

        result_sorted = np.divide(
            window_sum, window_count,
            out=np.zeros(n, dtype=np.float64),
            where=window_count > 0,
        )
        result = np.empty(n, dtype=np.float32)
        result[order] = result_sorted.astype(np.float32)
        frame[f"{prefix}_mean_{w}"] = result

        del result_sorted, result, window_sum, left_cs
        gc.collect()


def causal_group_count(
    frame: pd.DataFrame,
    keys: list[str],
    windows: list[int],
    prefix: str,
) -> None:
    """Causal group-level rolling transaction count for each window size (in-place)."""
    codes = frame.groupby(keys, sort=False, observed=True).ngroup().to_numpy()
    n     = len(frame)
    order = np.argsort(codes, kind="stable")
    sg    = codes[order]

    starts      = np.r_[0, np.flatnonzero(sg[1:] != sg[:-1]) + 1]
    ends        = np.r_[starts[1:], n]
    counts      = ends - starts
    group_start = np.repeat(starts, counts)
    pos         = np.arange(n, dtype=np.int64) - group_start

    for w in tqdm(windows, desc=f"{prefix}: activity windows", unit="window"):
        result_sorted = np.minimum(pos, w).astype(np.float32)
        result        = np.empty(n, dtype=np.float32)
        result[order] = result_sorted
        frame[f"{prefix}_{w}"] = result


def build_rolling_features(df: pd.DataFrame) -> None:
    """Build all rolling amount means and activity counts (in-place)."""
    for value_col, prefix, keys in tqdm(
        [
            ("amount_paid",     "sender_paid",      S),
            ("amount_received", "sender_received",  S),
            ("amount_paid",     "receiver_paid",    R),
            ("amount_received", "receiver_received", R),
        ],
        desc="Rolling amount families", unit="family",
    ):
        causal_group_rolling_mean(df, keys, value_col, [20, 100, 500], prefix)

    for keys, prefix in [(S, "sender_activity"), (R, "receiver_activity"), (P, "pair_activity")]:
        causal_group_count(df, keys, [20, 100], prefix)

    ratio_specs = [
        ("paid_vs_sender_mean_20",       "amount_paid",     "sender_paid_mean_20"),
        ("paid_vs_sender_mean_100",      "amount_paid",     "sender_paid_mean_100"),
        ("received_vs_sender_mean_20",   "amount_received", "sender_received_mean_20"),
        ("received_vs_sender_mean_100",  "amount_received", "sender_received_mean_100"),
        ("paid_vs_receiver_mean_100",    "amount_paid",     "receiver_paid_mean_100"),
        ("received_vs_receiver_mean_100","amount_received", "receiver_received_mean_100"),
    ]
    for out_col, num_col, den_col in tqdm(ratio_specs, desc="Current-vs-history ratios", unit="feature"):
        df[out_col] = safe_ratio(df[num_col].to_numpy(), np.maximum(df[den_col].to_numpy(), 1e-6))

    gc.collect()
