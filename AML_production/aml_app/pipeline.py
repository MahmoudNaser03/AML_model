"""Versioned batch feature pipeline shared by research and serving.

No target/pattern labels, learned statistics, or future rows enter features.
Equal timestamps use the source-file order. Each submitted batch starts cold.

The feature-building flow is:
1. normalize and chronologically order the submitted transactions;
2. encode accounts and directed sender/receiver pairs within the batch;
3. create transaction-level amount, payment, and time features;
4. add causal sender, receiver, pair, and reverse-flow history; and
5. restore the caller-facing row index and enforce the artifact's feature list.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_VERSION = "causal-batch-v2"
RAW_COLUMNS = ["timestamp", "from_bank", "from_account", "to_bank", "to_account",
               "amount_received", "receiving_currency", "amount_paid",
               "payment_currency", "payment_format"]
ALIASES = dict(zip(["Timestamp", "From Bank", "Account", "To Bank", "Account.1",
                   "Amount Received", "Receiving Currency", "Amount Paid",
                   "Payment Currency", "Payment Format"], RAW_COLUMNS))
TEXT_COLUMNS = [c for c in RAW_COLUMNS if c not in {"timestamp", "amount_paid", "amount_received"}]


def source_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def read_transactions(source, *, nrows=None) -> pd.DataFrame:
    """Preserve leading zeros in bank/account IDs, in both CSV and notebook use."""
    types = {c: "string" for c in TEXT_COLUMNS}
    types.update({k: "string" for k, v in ALIASES.items() if v in TEXT_COLUMNS})
    return pd.read_csv(source, dtype=types, nrows=nrows, keep_default_na=False)


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate raw fields and return a chronological, model-safe copy.

    ``_input_order`` records where each row appeared in the submitted batch. It
    provides a deterministic tie-break for equal timestamps and later becomes
    the feature frame's index, so callers can match scores to their input rows.
    """
    d = frame.rename(columns=ALIASES)
    if d.columns.duplicated().any():
        raise ValueError("Duplicate input columns or conflicting aliases.")
    missing = set(RAW_COLUMNS) - set(d.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    # Extra input columns, including labels and supplied feature values, never enter scoring.
    d = d.loc[:, RAW_COLUMNS].copy()
    try:
        d["ts"] = pd.to_datetime(d.pop("timestamp"), utc=True, format="mixed", errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("timestamp must contain valid datetimes.") from exc
    if d.ts.isna().any():
        raise ValueError("timestamp cannot be empty.")
    for name in TEXT_COLUMNS:
        s = d[name].astype("string").str.strip()
        if s.isna().any() or s.eq("").any() or s.str.len().gt(256).any():
            raise ValueError(f"{name} must contain nonempty identifiers of at most 256 characters.")
        d[name] = s.astype("category")
    for name in ("amount_paid", "amount_received"):
        try:
            d[name] = pd.to_numeric(d[name], errors="raise").astype("float64")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{name} must contain finite, nonnegative numbers.") from exc
        if not np.isfinite(d[name]).all() or d[name].lt(0).any():
            raise ValueError(f"{name} must contain finite, nonnegative numbers.")
    d["_input_order"] = np.arange(len(d), dtype=np.int64)
    return d.sort_values(["ts", "_input_order"], kind="stable").reset_index(drop=True)


def ratio(a, b):
    """Compute a bounded ratio while protecting against zero denominators."""
    return np.clip(np.asarray(a) / np.maximum(np.asarray(b), 1e-6), -1e6, 1e6)


def _group_features(g, times, paid, received, prefix, out):
    """Add causal history features for one entity role.

    ``g`` contains the integer account code for each chronological row. Every
    result describes only transactions earlier in the batch: prior counts,
    elapsed time since the previous transaction, rolling activity, and prior
    amount averages. ``prefix`` distinguishes sender and receiver features.
    """
    n = len(g)
    ids = pd.Series(g)

    # cumcount is the number of earlier rows for this account; the current row
    # is deliberately excluded from every historical feature below.
    counts = ids.groupby(ids, sort=False).cumcount().to_numpy()
    out[prefix + "_prev_count"] = counts

    # Accounts with no previous transaction receive a large, finite sentinel
    # before log scaling rather than an infinity or missing model input.
    prior = pd.Series(times).groupby(ids, sort=False).shift().to_numpy()
    out[prefix + "_delta_log"] = np.log1p(np.where(np.isnan(prior), 1e9, times - prior))

    # Group identical accounts together while preserving chronological order
    # inside each account. ``pos`` is the zero-based position within a group.
    order = np.argsort(g, kind="stable")
    gs = g[order]
    pos = counts[order]
    absolute = np.arange(n)

    # A gap wider than the full batch separates each account's time range. This
    # lets searchsorted calculate true elapsed-time windows across one array.
    width = float(times.max() - times.min() + 86401)
    keys = gs.astype("float64") * width + times[order] - times.min()
    for seconds, suffix in [(3600, "1h"), (86400, "24h")]:
        left = np.searchsorted(keys, keys - seconds, side="left")
        # Never allow the search window to cross into the preceding account.
        left = np.maximum(left, absolute - pos)
        value = np.empty(n, dtype="float32")
        value[order] = absolute - left
        out[f"{prefix}_count_{suffix}"] = value

    # Calculate the mean of up to the previous 20/100 amounts. Cumulative sums
    # keep this linear in the number of rows instead of scanning every window.
    for values, kind in [(paid, "paid"), (received, "received")]:
        sv = values[order]
        # pandas cumsum maintains a separate compensated accumulator per group.
        cumulative = pd.Series(sv).groupby(gs, sort=False).cumsum().to_numpy()
        previous = np.where(pos > 0, cumulative[np.maximum(absolute - 1, 0)], 0)
        for window in (20, 100):
            left_index = np.maximum(absolute - window - 1, 0)
            left_sum = np.where(pos > window, cumulative[left_index], 0)
            avg = (previous - left_sum) / np.maximum(np.minimum(pos, window), 1)
            value = np.empty(n, dtype="float32")
            value[order] = np.maximum(avg, 0)
            out[f"{prefix}_{kind}_mean_{window}"] = value
        out[f"{prefix}_{kind}_relative_20"] = ratio(values + 1, out[f"{prefix}_{kind}_mean_20"] + 1)
    return counts


def build_features(frame: pd.DataFrame, feature_columns=None, **_unused):
    """Build causal float32 model features for one independent batch.

    Rows are processed chronologically, and historical features can see only
    earlier rows in this call. The returned normalized frame stays in that
    chronological order. The feature frame uses ``_input_order`` as its index
    so downstream predictions can be joined back to the submitted rows.

    When ``feature_columns`` is provided, it is treated as the model artifact's
    ordered contract: unknown names fail explicitly and valid columns are
    returned in exactly that order.
    """

    # Stage 1: validate, normalize, and establish deterministic event order.
    d = normalize(frame)
    if d.empty:
        raise ValueError("At least one transaction is required.")
    n = len(d)
    paid, received = d.amount_paid.to_numpy(), d.amount_received.to_numpy()

    # Convert timestamps to seconds. pandas may store them at microsecond or
    # nanosecond resolution depending on the installed version/input data.
    times = d.ts.astype("int64").to_numpy() / (1e9 if d.ts.dtype.unit == "ns" else 1e6)

    # Stage 2: assign one batch-local code to each exact (bank, account) tuple.
    # Using tuples prevents collisions such as ("1", "23") vs ("12", "3").
    accounts = pd.MultiIndex.from_arrays([
        pd.concat([d.from_bank.astype("string"), d.to_bank.astype("string")], ignore_index=True),
        pd.concat([d.from_account.astype("string"), d.to_account.astype("string")], ignore_index=True)])
    codes, uniques = pd.factorize(accounts, sort=False)
    sender, receiver = codes[:n], codes[n:]
    base = np.int64(len(uniques))

    # Pair IDs preserve direction. ``reverse`` identifies receiver -> sender,
    # which is later used to detect return-flow behavior.
    pairs = sender.astype("int64") * base + receiver
    reverse = receiver.astype("int64") * base + sender
    hour = d.ts.dt.hour.to_numpy() + d.ts.dt.minute.to_numpy() / 60

    # Stage 3: features that depend only on the current transaction.
    out = {
        "amount_paid_log": np.log1p(paid), "amount_received_log": np.log1p(received),
        "flow_gap_log": np.sign(received - paid) * np.log1p(abs(received - paid)),
        "flow_relative_gap": ratio(abs(received - paid), np.maximum(paid, received)),
        "received_paid_ratio": ratio(received + 1, paid + 1),
        "same_bank": d.from_bank.astype("string").eq(d.to_bank.astype("string")).to_numpy(),
        "same_account": sender == receiver,
        "cross_currency": d.payment_currency.astype("string").ne(d.receiving_currency.astype("string")).to_numpy(),
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
        "paid_round_100": (paid > 0) & (np.remainder(paid, 100) == 0),
        "paid_round_1000": (paid > 0) & (np.remainder(paid, 1000) == 0),
    }
    # Explicit categories avoid learned/future vocabulary and arbitrary ordinal IDs.
    for fmt in ["ACH", "Cheque", "Credit Card", "Cash", "Reinvestment", "Wire", "Bitcoin"]:
        out["format_" + fmt.lower().replace(" ", "_")] = d.payment_format.astype("string").str.casefold().eq(fmt.casefold()).to_numpy()

    # Stage 4: causal histories for each account role and directed account pair.
    sp = _group_features(sender, times, paid, received, "sender", out)
    rp = _group_features(receiver, times, paid, received, "receiver", out)
    pair_series = pd.Series(pairs)
    pp = pair_series.groupby(pair_series, sort=False).cumcount().to_numpy()
    out["pair_prev_count"] = pp
    out["pair_share_sender"] = ratio(pp, sp + 1)
    out["pair_share_receiver"] = ratio(pp, rp + 1)

    # A pair is new only on its first chronological occurrence. Summing those
    # first occurrences per account counts distinct prior counterparties.
    new = pp == 0
    for ids, prefix, count in [(sender, "sender", sp), (receiver, "receiver", rp)]:
        unique = pd.Series(new.astype("int32")).groupby(ids, sort=False).cumsum().to_numpy() - new
        out[prefix + "_unique_counterparties"] = unique
        out[prefix + "_diversity"] = ratio(unique, count + 1)
        out[prefix + "_is_new"] = count == 0
        out[prefix + "_burst_ratio"] = ratio(out[prefix + "_count_1h"] + 1, out[prefix + "_count_24h"] / 24 + 1)
    out["pair_is_new"] = new
    out["new_pair_existing_sender"] = new & (sp > 0)

    # These flags capture first-seen account/category combinations. duplicated
    # considers earlier chronological rows and marks the first occurrence new.
    for cols, name in [(["from_bank", "from_account", "payment_format"], "sender_format_new"),
                       (["from_bank", "from_account", "payment_currency"], "sender_currency_new"),
                       (["to_bank", "to_account", "receiving_currency"], "receiver_currency_new")]:
        out[name] = ~d.duplicated(cols).to_numpy()
    # Query the last earlier transaction with the opposite directed pair ID.
    # merge_asof searches on chronological row position and forbids matching the
    # current position, keeping the lookup strictly causal.
    position = np.arange(n, dtype="int64")
    right = pd.DataFrame({"position": position, "pair": pairs, "previous": position})
    left = pd.DataFrame({"position": position, "pair": reverse})
    previous = pd.merge_asof(left, right, on="position", by="pair", allow_exact_matches=False)["previous"].to_numpy()
    seen = np.isfinite(previous)
    idx = np.where(seen, previous, 0).astype("int64")
    out["reverse_prev_count"] = np.where(seen, pp[idx] + 1, 0)
    out["reverse_seen"] = seen
    out["reverse_delta_log"] = np.log1p(np.where(seen, times - times[idx], 1e9))
    out["reverse_amount_similarity"] = np.where(seen, np.exp(-abs(received - paid[idx]) / np.maximum(received, 1)), 0)
    # Sender's previous incoming activity describes possible rapid onward
    # transfers: money arrives at an account and is soon sent elsewhere.
    right = pd.DataFrame({"position": position, "account": receiver, "previous": position})
    left = pd.DataFrame({"position": position, "account": sender})
    previous = pd.merge_asof(left, right, on="position", by="account", allow_exact_matches=False)["previous"].to_numpy()
    seen = np.isfinite(previous)
    idx = np.where(seen, previous, 0).astype("int64")
    out["sender_prior_incoming_count"] = np.where(seen, rp[idx] + 1, 0)
    out["sender_incoming_delta_log"] = np.log1p(np.where(seen, times - times[idx], 1e9))
    out["onward_amount_ratio"] = np.where(seen, ratio(paid + 1, received[idx] + 1), 0)
    out["sender_in_out_mix"] = np.log1p(out["sender_prior_incoming_count"] * np.sqrt(sp + 1))

    # Stage 5: create one consistently typed matrix, restore input row identity,
    # and finally apply the exact ordered feature contract requested by a model.
    X = pd.DataFrame({k: np.asarray(v, dtype="float32") for k, v in out.items()}, index=d._input_order.to_numpy())
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Nonfinite engineered features.")
    if feature_columns is not None:
        missing = set(feature_columns) - set(X.columns)
        if missing:
            raise ValueError("Unsupported artifact features: " + ", ".join(sorted(missing)))
        X = X.loc[:, feature_columns]
    return X, d
