import numpy as np
import pandas as pd
import pytest

from aml_app.pipeline import build_features, read_transactions


def transactions():
    rows = []
    for stamp, sender, receiver, amount in [
        ("2026-01-01T00:00:00Z", "001", "002", 10),
        ("2026-01-01T00:10:00Z", "001", "002", 20),
        ("2026-01-01T00:20:00Z", "002", "001", 30),
        ("2026-01-01T02:00:00Z", "003", "002", 40),
        ("2026-01-01T02:10:00Z", "001", "002", 50),
    ]:
        rows.append(
            {
                "timestamp": stamp,
                "from_bank": "01",
                "from_account": sender,
                "to_bank": "01",
                "to_account": receiver,
                "amount_paid": float(amount),
                "amount_received": float(amount),
                "payment_currency": "USD",
                "receiving_currency": "USD",
                "payment_format": "ACH",
            }
        )
    return pd.DataFrame(rows)


def test_directional_history_and_unique_counterparties():
    x, _ = build_features(transactions())
    np.testing.assert_array_equal(x.reverse_prev_count, [0, 0, 2, 0, 1])
    np.testing.assert_array_equal(x.receiver_unique_counterparties, [0, 1, 0, 1, 2])
    np.testing.assert_array_equal(x.sender_paid_mean_20, [0, 10, 0, 0, 15])
    np.testing.assert_array_equal(x.sender_count_1h, [0, 1, 0, 0, 0])
    assert x.loc[2, "sender_prior_incoming_count"] == 2


def test_future_rows_and_labels_cannot_change_past_features():
    raw = transactions()
    short, _ = build_features(raw.iloc[:3])
    raw["Is Laundering"] = [1, 1, 1, 1, 1]
    raw["sender_prev_count"] = 999
    long, _ = build_features(raw)
    pd.testing.assert_frame_equal(short, long.iloc[:3])


def test_csv_and_json_parity_with_leading_zero_ids():
    import io

    raw = transactions()
    csv = read_transactions(io.StringIO(raw.to_csv(index=False)))
    x1, _ = build_features(raw)
    x2, _ = build_features(csv)
    pd.testing.assert_frame_equal(x1, x2)
    assert csv.from_account.iloc[0] == "001"


def test_input_order_and_equal_timestamp_tie_break():
    raw = transactions().iloc[[4, 0, 1, 3, 2]].reset_index(drop=True)
    x, d = build_features(raw)
    assert d._input_order.tolist() == [1, 2, 4, 3, 0]
    assert x.loc[0, "sender_prev_count"] == 2
    same_time = transactions().iloc[:2].copy()
    same_time["timestamp"] = "2026-01-01T00:00:00Z"
    x, _ = build_features(same_time)
    assert x.sender_prev_count.tolist() == [0, 1]


@pytest.mark.parametrize(
    "column,value",
    [
        ("amount_paid", -1),
        ("amount_received", np.inf),
        ("amount_paid", np.nan),
        ("timestamp", ""),
        ("timestamp", "bad"),
        ("from_account", " "),
    ],
)
def test_invalid_inputs_are_rejected(column, value):
    raw = transactions()
    raw.loc[0, column] = value
    with pytest.raises(ValueError):
        build_features(raw)


def test_missing_feature_is_not_silently_filled():
    with pytest.raises(ValueError, match="Unsupported"):
        build_features(transactions(), ["invented"])


def test_alias_conflict_is_rejected():
    raw = transactions()
    raw["From Bank"] = "other"
    with pytest.raises(ValueError, match="Duplicate"):
        build_features(raw)
