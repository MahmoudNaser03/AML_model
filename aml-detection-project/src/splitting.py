"""
Pattern-component-aware chronological train / val / test split.

Pattern transactions belonging to the same laundering attempt stay together.
Unpatterned transactions are assigned chronologically by row position.
"""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.config import TRAIN_FRAC, VAL_FRAC


def _build_connected_components(pattern_map: pd.DataFrame) -> dict[int, int]:
    """Return {pattern_id: root_pattern_id} via Union-Find."""
    tx_to_patterns = (
        pattern_map.groupby("row_id")["pattern_id"]
        .agg(lambda s: tuple(sorted(set(map(int, s)))))
    )

    parent: dict[int, int] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for pats in tqdm(tx_to_patterns.values,
                     desc="Building pattern components", unit="transaction"):
        if len(pats) > 1:
            first = pats[0]
            for p in pats[1:]:
                union(first, p)
        else:
            find(pats[0])

    return {p: find(p) for p in parent}


def build_split(df: pd.DataFrame, pattern_map: pd.DataFrame,
                train_frac: float = TRAIN_FRAC,
                val_frac:   float = VAL_FRAC) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assign each row a split label ('train', 'val', 'test') and return
    (df_with_split, pattern_map_with_component).

    Guarantees no pattern component crosses split boundaries.
    """
    d = df.copy()
    d["split"] = "unassigned"

    comp_map = _build_connected_components(pattern_map)

    pm2 = pattern_map.copy()
    pm2["pattern_component"] = pm2["pattern_id"].map(comp_map)

    components = (
        pm2.groupby("pattern_component")["row_id"]
        .apply(lambda s: np.array(sorted(set(s)), dtype=np.int64))
    )

    n = len(d)
    train_end = int(n * train_frac)
    val_end   = int(n * (train_frac + val_frac))

    component_order = sorted(
        components.index,
        key=lambda c: int(d.loc[d.row_id.isin(components.loc[c]), "row_id"].min()),
    )

    for comp in tqdm(component_order,
                     desc="Assigning pattern components to splits", unit="component"):
        rows      = components.loc[comp]
        first_pos = int(rows.min())
        split     = "train" if first_pos < train_end else ("val" if first_pos < val_end else "test")
        d.loc[d.row_id.isin(rows), "split"] = split

    remaining = d.loc[d["split"].eq("unassigned"), ["row_id", "ts"]].sort_values(["ts", "row_id"])
    m        = len(remaining)
    r_train  = int(m * train_frac)
    r_val    = int(m * val_frac)

    d.loc[remaining.iloc[:r_train].row_id,              "split"] = "train"
    d.loc[remaining.iloc[r_train:r_train + r_val].row_id, "split"] = "val"
    d.loc[remaining.iloc[r_train + r_val:].row_id,       "split"] = "test"

    if d["split"].eq("unassigned").any():
        raise AssertionError("Unassigned rows remain after split.")

    split_counts = (
        pm2.merge(d[["row_id", "split"]], on="row_id", how="left")
        .groupby("pattern_component")["split"].nunique()
    )
    if split_counts.max() != 1:
        raise AssertionError("Pattern component crosses train/val/test boundary.")

    return d, pm2
