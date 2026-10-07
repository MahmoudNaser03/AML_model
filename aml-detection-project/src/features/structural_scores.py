"""
Transaction-derived continuous structural pattern scores (V3).

No Pattern TXT information, target labels, pattern IDs, or memberships
are used here.  All scores are causal transaction-derived features.
"""

import numpy as np
import pandas as pd


def build_structural_scores(
    df: pd.DataFrame,
    train_idx: np.ndarray,
) -> None:
    """Add eight structural pattern scores and three aggregate columns in-place."""
    n = len(df)

    def arr(name, default=0.0):
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(default).to_numpy(dtype="float32")
        return np.full(n, default, dtype="float32")

    train_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True

    def qscale(x, q=0.95):
        ref = x[train_mask]
        ref = ref[np.isfinite(ref)]
        cap = float(np.nanquantile(ref, q)) if len(ref) else 1.0
        if not np.isfinite(cap) or cap <= 1e-9:
            cap = 1.0
        return np.clip(x / cap, 0.0, 1.0).astype("float32")

    def high(x):
        return qscale(np.log1p(np.clip(x, 0, None)))

    def low(x):
        return (1.0 - high(x)).astype("float32")

    sp, rp, pp = arr("sender_prev_count"), arr("receiver_prev_count"), arr("pair_prev_count")
    rev          = arr("reverse_pair_prev_count")
    sa20, ra20   = arr("sender_activity_20"),  arr("receiver_activity_20")
    sa100, ra100 = arr("sender_activity_100"), arr("receiver_activity_100")
    sd, rd       = arr("sender_diversity"),    arr("receiver_diversity")
    sn, rn       = arr("sender_network_expansion"), arr("receiver_network_expansion")
    pv, pn       = arr("pair_velocity_relative"), arr("pair_is_new")
    npe          = arr("new_pair_existing_sender")
    rec          = arr("reciprocal_intensity")
    gap          = arr("reverse_amount_gap")
    rpr, rrr     = arr("reverse_paid_ratio"), arr("reverse_received_ratio")
    pdsec, sdsec, rdsec = arr("pair_delta_sec"), arr("sender_delta_sec"), arr("receiver_delta_sec")
    su, ru       = arr("sender_unique_out"), arr("receiver_unique_in")

    sb       = np.clip((sa20 + 1) / (sa100 / 5 + 1), 0, 10)
    rb       = np.clip((ra20 + 1) / (ra100 / 5 + 1), 0, 10)
    new_pair = np.clip(0.70 * pn + 0.30 * npe, 0, 1).astype("float32")

    g        = gap[train_mask]; g = g[np.isfinite(g)]
    gscale   = max(float(np.nanmedian(np.abs(g))) if len(g) else 1.0, 1e-6)
    amount_similarity = np.exp(-np.clip(gap, 0, 1e6) / gscale).astype("float32")

    fanout  = .22*high(sa20) + .20*high(su) + .18*high(sd) + .18*high(sn) + .12*qscale(sb) + .10*new_pair
    fanin   = .22*high(ra20) + .20*high(ru) + .18*high(rd) + .18*high(rn) + .12*qscale(rb) + .10*new_pair
    cycle   = .30*qscale(rev) + .25*np.clip(rec/3, 0, 1) + .20*low(pdsec) + .15*amount_similarity + .10*np.clip((rpr+rrr)/2, 0, 1)
    stack   = .35*high(pp) + .25*qscale(pv) + .20*high(sp) + .10*qscale(sa20) + .10*low(pdsec)
    bipart  = .22*high(su) + .22*high(ru) + .16*high(sa20) + .16*high(ra20) + .12*high(sd) + .12*high(rd)
    sg_sc   = .25*high(su) + .20*high(sn) + .15*high(sa20) + .15*high(ra20) + .15*high(rd) + .10*new_pair
    gs_sc   = .25*high(ru) + .20*high(rn) + .15*high(ra20) + .15*high(sa20) + .15*high(sd) + .10*new_pair
    rand_sc = .40*new_pair + .25*low(pp) + .15*qscale(sdsec) + .10*qscale(rdsec) + .10*qscale(sn)

    score_map = {
        "struct_fanout_score":         fanout,
        "struct_fanin_score":          fanin,
        "struct_random_score":         rand_sc,
        "struct_cycle_score":          cycle,
        "struct_stack_score":          stack,
        "struct_bipartite_score":      bipart,
        "struct_scatter_gather_score": sg_sc,
        "struct_gather_scatter_score": gs_sc,
    }
    for name, values in score_map.items():
        df[name] = np.clip(np.nan_to_num(values, nan=0, posinf=1, neginf=0), 0, 1).astype("float32")

    M = df[list(score_map)].to_numpy(dtype="float32")
    df["struct_pattern_max_score"]    = M.max(axis=1).astype("float32")
    df["struct_pattern_mean_score"]   = M.mean(axis=1).astype("float32")
    df["struct_pattern_active_count"] = (M >= 0.60).sum(axis=1).astype("int8")
