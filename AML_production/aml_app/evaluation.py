"""Ground-truth comparison helpers for scored dashboard batches."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

LABEL_COLUMN_NAMES = (
    "islaundering",
    "islaundered",
    "actuallabel",
    "groundtruth",
    "label",
    "target",
)


def suggested_label_column(columns) -> str | None:
    """Return the first commonly named label column, without guessing otherwise."""
    normalized = {
        re.sub(r"[^a-z0-9]", "", str(column).lower()): str(column)
        for column in columns
    }
    for candidate in LABEL_COLUMN_NAMES:
        match = normalized.get(candidate.lower())
        if match is not None:
            return match
    return None


def binary_labels(values) -> pd.Series:
    """Validate and normalize common binary ground-truth representations."""
    labels = pd.Series(values).reset_index(drop=True)
    if labels.empty:
        raise ValueError("The label column is empty.")

    numeric = pd.to_numeric(labels, errors="coerce")
    if numeric.notna().all() and numeric.isin([0, 1]).all():
        return numeric.astype("int8")

    text = labels.astype("string").str.strip().str.lower()
    mapping = {
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
        "positive": 1,
        "negative": 0,
        "laundering": 1,
        "not laundering": 0,
    }
    mapped = text.map(mapping)
    if mapped.isna().any():
        raise ValueError("Labels must be binary: 0/1, true/false, yes/no, or positive/negative.")
    return mapped.astype("int8")


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _evaluate_predictions(scored: pd.DataFrame, labels, predicted) -> tuple[pd.DataFrame, dict]:
    """Align labels and predictions to scored records and calculate results."""
    if "record_id" not in scored:
        raise ValueError("Scored records are missing record IDs.")
    predicted = pd.Series(predicted).astype(bool).reset_index(drop=True)
    if len(predicted) != len(scored):
        raise ValueError("Predictions do not align one-to-one with scored records.")
    actual = binary_labels(labels)
    record_ids = pd.to_numeric(scored["record_id"], errors="coerce")
    if (
        record_ids.isna().any()
        or not np.equal(record_ids, np.floor(record_ids)).all()
        or record_ids.duplicated().any()
        or record_ids.lt(0).any()
        or record_ids.ge(len(actual)).any()
        or len(record_ids) != len(actual)
    ):
        raise ValueError("Scored records do not align one-to-one with the uploaded labels.")

    aligned_actual = actual.iloc[record_ids.astype(int).to_numpy()].reset_index(drop=True)
    positive = aligned_actual.astype(bool)
    tp = int((predicted & positive).sum())
    fp = int((predicted & ~positive).sum())
    fn = int((~predicted & positive).sum())
    tn = int((~predicted & ~positive).sum())
    precision = _rate(tp, tp + fp)
    recall = _rate(tp, tp + fn)
    specificity = _rate(tn, tn + fp)
    accuracy = _rate(tp + tn, len(actual))
    f1 = _rate(2 * tp, 2 * tp + fp + fn)

    comparison = scored.reset_index(drop=True).copy()
    comparison["actual_label"] = aligned_actual
    comparison["outcome"] = np.select(
        [predicted & positive, predicted & ~positive, ~predicted & positive],
        ["True positive", "False positive", "False negative"],
        default="True negative",
    )
    metrics = {
        "rows": len(actual),
        "actual_positives": int(positive.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
    }
    return comparison, metrics


def evaluate_alerts(scored: pd.DataFrame, labels) -> tuple[pd.DataFrame, dict]:
    """Compare deployed alert decisions with uploaded ground truth."""
    if "alert" not in scored:
        raise ValueError("Scored records are missing alert decisions.")
    return _evaluate_predictions(scored, labels, scored["alert"])


def evaluate_score_threshold(scored: pd.DataFrame, labels, threshold: float) -> tuple[pd.DataFrame, dict]:
    """Preview ground-truth results at a candidate score threshold."""
    if "risk_score" not in scored:
        raise ValueError("Scored records are missing risk scores.")
    try:
        threshold = float(threshold)
        scores = pd.to_numeric(scored["risk_score"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Risk scores and the evaluation threshold must be numeric.") from exc
    if not 0 <= threshold <= 1 or not np.isfinite(threshold) or not np.isfinite(scores).all():
        raise ValueError("Risk scores and the evaluation threshold must be finite values from 0 to 1.")

    predicted = scores.ge(threshold)
    comparison, metrics = _evaluate_predictions(scored, labels, predicted)
    comparison["evaluation_alert"] = predicted.to_numpy()
    metrics["threshold"] = threshold
    return comparison, metrics
