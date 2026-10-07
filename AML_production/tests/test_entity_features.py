import numpy as np
import pandas as pd
import pytest

from aml_app.entity_features import (
    ENTITY_FEATURE_COLUMNS,
    EntityRegistry,
    build_entity_features,
    fit_registry,
)
from aml_app.pipeline import build_features
from aml_app.service import combine_scores


def _rows():
    return pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "from_bank": "01",
                "from_account": "secret-a",
                "to_bank": "02",
                "to_account": "secret-b",
                "amount_paid": 10,
                "amount_received": 10,
                "payment_currency": "USD",
                "receiving_currency": "USD",
                "payment_format": "ACH",
            },
            {
                "timestamp": "2026-01-01T00:01:00Z",
                "from_bank": "01",
                "from_account": "secret-a",
                "to_bank": "03",
                "to_account": "secret-c",
                "amount_paid": 20,
                "amount_received": 20,
                "payment_currency": "USD",
                "receiving_currency": "USD",
                "payment_format": "ACH",
            },
        ]
    )


def test_registry_is_train_only_hashed_aggregate(tmp_path):
    _, normalized = build_features(_rows())
    path = tmp_path / "registry.npz"
    training = fit_registry(normalized, np.array([1, 0]), path)
    registry = EntityRegistry(path)
    future = build_entity_features(normalized.iloc[[1]], registry)

    assert training.columns.tolist() == ENTITY_FEATURE_COLUMNS
    assert training.iloc[0]["sender_positive_count_log"] == 0
    assert future.iloc[0]["sender_positive_count_log"] == pytest.approx(np.log1p(1))
    assert b"secret-a" not in path.read_bytes()
    assert np.isfinite(future.to_numpy()).all()


def test_geometric_three_policy():
    policy = {
        "rule": "geometric_three",
        "weights": {"model_a": 0.75, "model_b": 0.20, "baseline_b": 0.05},
    }
    result = combine_scores(np.array([0.4]), np.array([0.2]), policy, np.array([0.1]))
    assert result[0] == np.exp(
        0.75 * np.log(0.4) + 0.20 * np.log(0.2) + 0.05 * np.log(0.1)
    )
