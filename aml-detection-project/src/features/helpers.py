"""Low-level causal feature utilities shared across all feature modules."""

import numpy as np
import pandas as pd


def add_group_cumcount(frame: pd.DataFrame, keys: list[str], name: str) -> None:
    """Add cumulative within-group count (causal, excludes current row)."""
    frame[name] = (
        frame.groupby(keys, sort=False, observed=True)
        .cumcount()
        .astype("float32")
    )


def safe_ratio(a, b, clip: float = 1e6) -> np.ndarray:
    """Element-wise a/b; returns 0 where |b| < 1e-12; clips to ±clip."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = np.divide(a, b, out=np.zeros_like(a), where=np.abs(b) > 1e-12)
    return np.clip(out, -clip, clip).astype("float32")
