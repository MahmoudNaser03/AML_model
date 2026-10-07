"""Train-cutoff entity-risk features for the version-3 AML ensemble.

The registry stores only fixed-size hashes and aggregate counts. It contains no
transaction rows and never learns from a scoring batch.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ENTITY_FEATURE_VERSION = "train-entity-risk-v1"
ENTITY_GROUPS = ("sender", "receiver", "either_sender", "either_receiver", "pair", "from_bank", "to_bank")
ENTITY_FEATURE_COLUMNS = [
    f"{group}_{suffix}"
    for group in ENTITY_GROUPS
    for suffix in ("train_count_log", "positive_count_log", "known_positive", "risk_alpha2", "risk_alpha10")
]


def source_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _digest(parts) -> bytes:
    hasher = hashlib.blake2b(digest_size=16, person=b"aml-entity-v1")
    if not isinstance(parts, tuple):
        parts = (parts,)
    for part in parts:
        value = str(part).encode("utf-8")
        hasher.update(len(value).to_bytes(4, "big"))
        hasher.update(value)
    return hasher.digest()


def _keys(values):
    return np.fromiter((_digest(value) for value in values), dtype="S16", count=len(values))


def _identity_codes(d):
    n = len(d)
    identities = pd.MultiIndex.from_arrays([
        pd.concat([d.from_bank.astype("string"), d.to_bank.astype("string")], ignore_index=True),
        pd.concat([d.from_account.astype("string"), d.to_account.astype("string")], ignore_index=True),
    ])
    codes, unique_accounts = pd.factorize(identities, sort=False)
    sender, receiver = codes[:n], codes[n:]
    base = len(unique_accounts)
    pair_codes, unique_pairs = pd.factorize(sender.astype("int64") * base + receiver, sort=False)
    from_bank_codes, unique_from_banks = pd.factorize(d.from_bank.astype("string"), sort=False)
    to_bank_codes, unique_to_banks = pd.factorize(d.to_bank.astype("string"), sort=False)
    account_keys = _keys(list(unique_accounts))
    pair_values = [(*unique_accounts[int(value // base)], *unique_accounts[int(value % base)]) for value in unique_pairs]
    return {
        "sender": (sender, account_keys),
        "receiver": (receiver, account_keys),
        "pair": (pair_codes, _keys(pair_values)),
        "from_bank": (from_bank_codes, _keys(list(unique_from_banks))),
        "to_bank": (to_bank_codes, _keys(list(unique_to_banks))),
    }


def _values(count, positive, prior):
    return np.column_stack([
        np.log1p(count),
        np.log1p(positive),
        positive > 0,
        (positive + 2 * prior) / (count + 2),
        (positive + 10 * prior) / (count + 10),
    ]).astype("float32")


class EntityRegistry:
    def __init__(self, path):
        archive = np.load(path, allow_pickle=False)
        self.prior = float(archive["global_prior"][0])
        self.tables = {
            name: (archive[f"{name}_keys"], archive[f"{name}_count"], archive[f"{name}_positive"])
            for name in ("sender", "receiver", "either", "pair", "from_bank", "to_bank")
        }

    def lookup(self, name, keys):
        known_keys, known_count, known_positive = self.tables[name]
        position = np.searchsorted(known_keys, keys)
        valid = position < len(known_keys)
        safe = np.minimum(position, max(len(known_keys) - 1, 0))
        matched = valid & (known_keys[safe] == keys) if len(known_keys) else np.zeros(len(keys), dtype=bool)
        count = np.zeros(len(keys), dtype="float64")
        positive = np.zeros(len(keys), dtype="float64")
        if matched.any():
            count[matched] = known_count[safe[matched]]
            positive[matched] = known_positive[safe[matched]]
        return count, positive


def fit_registry(d: pd.DataFrame, y, destination):
    """Create training leave-one-out features and persist aggregate hash tables."""
    y = np.asarray(y, dtype="int8")
    if len(d) != len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("Entity registry requires one binary training label per row.")
    encoded = _identity_codes(d)
    prior = float(y.mean())
    tables = {}

    def aggregate(name, codes, keys, *, second_codes=None):
        size = len(keys)
        if second_codes is None:
            count = np.bincount(codes, minlength=size)
            positive = np.bincount(codes, weights=y, minlength=size)
            row_count = count[codes] - 1
            row_positive = positive[codes] - y
        else:
            count = np.bincount(np.r_[codes, second_codes], minlength=size)
            positive = np.bincount(np.r_[codes, second_codes], weights=np.r_[y, y], minlength=size)
            same = codes == second_codes
            row_count = count[codes] - 1 - same
            row_positive = positive[codes] - y - y * same
        order = np.argsort(keys)
        tables[f"{name}_keys"] = keys[order]
        tables[f"{name}_count"] = count[order].astype("uint32")
        tables[f"{name}_positive"] = positive[order].astype("uint32")
        return _values(row_count, row_positive, prior)

    sender, account_keys = encoded["sender"]
    receiver, _ = encoded["receiver"]
    blocks = [
        aggregate("sender", sender, account_keys),
        aggregate("receiver", receiver, account_keys),
        aggregate("either", sender, account_keys, second_codes=receiver),
        # The same combined-role table is queried for the receiving endpoint.
        None,
    ]
    either_count = np.bincount(np.r_[sender, receiver], minlength=len(account_keys))
    either_positive = np.bincount(np.r_[sender, receiver], weights=np.r_[y, y], minlength=len(account_keys))
    same = sender == receiver
    blocks[3] = _values(
        either_count[receiver] - 1 - same,
        either_positive[receiver] - y - y * same,
        prior,
    )
    for name in ("pair", "from_bank", "to_bank"):
        blocks.append(aggregate(name, *encoded[name]))
    np.savez_compressed(destination, global_prior=np.array([prior]), **tables)
    return pd.DataFrame(np.column_stack(blocks), columns=ENTITY_FEATURE_COLUMNS, index=d._input_order.to_numpy())


def build_entity_features(d: pd.DataFrame, registry: EntityRegistry):
    """Look up immutable training aggregates for normalized scoring rows."""
    encoded = _identity_codes(d)

    def block(table, key_name):
        codes, unique_keys = encoded[key_name]
        count, positive = registry.lookup(table, unique_keys)
        return _values(count[codes], positive[codes], registry.prior)

    blocks = [
        block("sender", "sender"),
        block("receiver", "receiver"),
        block("either", "sender"),
        block("either", "receiver"),
        block("pair", "pair"),
        block("from_bank", "from_bank"),
        block("to_bank", "to_bank"),
    ]
    return pd.DataFrame(np.column_stack(blocks), columns=ENTITY_FEATURE_COLUMNS, index=d._input_order.to_numpy())
