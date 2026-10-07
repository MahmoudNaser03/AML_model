"""
Add TRUE Pattern TXT membership columns to the dataframe.

These are experiment metadata only — never model features.
"""

import pandas as pd


def add_pattern_meta(df: pd.DataFrame, pattern_map: pd.DataFrame) -> pd.DataFrame:
    """Merge pattern membership counts from pattern_map into df. Returns updated df."""
    pm = (
        pattern_map.groupby("row_id")
        .agg(
            txt_pattern_memberships=("pattern_id",   "nunique"),
            txt_pattern_type_nunique=("pattern_type", "nunique"),
        )
    )
    df = df.merge(pm, left_on="row_id", right_index=True, how="left")
    df["txt_pattern_memberships"]  = df["txt_pattern_memberships"].fillna(0).astype("int16")
    df["txt_pattern_type_nunique"] = df["txt_pattern_type_nunique"].fillna(0).astype("int8")
    df["txt_patterned"]            = df["txt_pattern_memberships"].gt(0).astype("int8")

    mapped = df["txt_pattern_memberships"].gt(0)
    if mapped.any():
        rate = float(df.loc[mapped, "target"].mean())
        if rate < 0.99:
            raise AssertionError(
                f"Pattern TXT mapping inconsistent with target labels (rate={rate:.6%})."
            )
    return df
