import pandas as pd
import pytest

from aml_app.evaluation import (
    binary_labels,
    evaluate_alerts,
    evaluate_score_threshold,
    suggested_label_column,
)


def test_suggested_label_column_recognizes_dataset_header():
    columns = ["Timestamp", "Amount Paid", "Is Laundering"]
    assert suggested_label_column(columns) == "Is Laundering"
    assert suggested_label_column(["Timestamp", "Amount Paid"]) is None


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([0, 1, 0], [0, 1, 0]),
        (["false", "true", "no"], [0, 1, 0]),
        (["negative", "laundering", "positive"], [0, 1, 1]),
    ],
)
def test_binary_labels_accepts_explicit_binary_values(values, expected):
    assert binary_labels(values).tolist() == expected


def test_binary_labels_rejects_missing_or_multiclass_values():
    with pytest.raises(ValueError, match="binary"):
        binary_labels([0, "", 2])


def test_evaluate_alerts_aligns_by_record_id_and_calculates_confusion_counts():
    scored = pd.DataFrame(
        {
            "record_id": [2, 0, 3, 1],
            "alert": [True, True, False, False],
            "risk_score": [0.9, 0.8, 0.2, 0.1],
        }
    )
    comparison, metrics = evaluate_alerts(scored, [1, 1, 1, 0])

    assert comparison["actual_label"].tolist() == [1, 1, 0, 1]
    assert comparison["outcome"].tolist() == [
        "True positive",
        "True positive",
        "True negative",
        "False negative",
    ]
    assert metrics == {
        "rows": 4,
        "actual_positives": 3,
        "tp": 2,
        "fp": 0,
        "fn": 1,
        "tn": 1,
        "precision": 1.0,
        "recall": pytest.approx(2 / 3),
        "specificity": 1.0,
        "f1": pytest.approx(0.8),
        "accuracy": 0.75,
    }


def test_evaluate_alerts_rejects_misaligned_records():
    scored = pd.DataFrame({"record_id": [0, 0], "alert": [True, False]})
    with pytest.raises(ValueError, match="one-to-one"):
        evaluate_alerts(scored, [0, 1])


def test_evaluate_alerts_reports_zero_f1_when_every_positive_is_missed():
    scored = pd.DataFrame({"record_id": [0, 1], "alert": [False, False]})
    _, metrics = evaluate_alerts(scored, [1, 0])
    assert metrics["precision"] is None
    assert metrics["recall"] == 0
    assert metrics["f1"] == 0


def test_lower_score_threshold_can_trade_precision_for_recall():
    scored = pd.DataFrame(
        {
            "record_id": [0, 1, 2, 3],
            "risk_score": [0.9, 0.7, 0.6, 0.1],
            "alert": [True, False, False, False],
        }
    )
    labels = [1, 0, 1, 0]
    comparison, metrics = evaluate_score_threshold(scored, labels, 0.5)

    assert comparison["alert"].tolist() == [True, False, False, False]
    assert comparison["evaluation_alert"].tolist() == [True, True, True, False]
    assert metrics["recall"] == 1
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["threshold"] == 0.5
