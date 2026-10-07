"""Public shared feature contract. Implementation lives in pipeline.py."""
from .pipeline import (
    ALIASES as SOURCE_COLUMN_ALIASES,
    FEATURE_VERSION,
    RAW_COLUMNS,
    build_features,
    normalize as normalize_transactions,
    read_transactions,
)

__all__ = [
    "FEATURE_VERSION",
    "RAW_COLUMNS",
    "SOURCE_COLUMN_ALIASES",
    "advance_history",
    "build_features",
    "normalize_transactions",
    "read_transactions",
]


def advance_history(*args, **kwargs):
    raise NotImplementedError("Version 2 uses independent batches; persistent history requires separate validation and retraining.")
