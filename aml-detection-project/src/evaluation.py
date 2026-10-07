"""
Evaluation helpers: operating-point calibration, metrics, fusion search.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, precision_score, recall_score, f1_score,
)


# ── Single-model calibration ─────────────────────────────────────────────────

def recall_at_floor(y_true: np.ndarray, scores: np.ndarray, precision_floor: float) -> float:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    valid = np.where(precision[:-1] >= precision_floor)[0]
    return float(recall[:-1][valid].max()) if len(valid) else np.nan


def threshold_at_precision_floor(
    y_true: np.ndarray, scores: np.ndarray, precision_floor: float
) -> tuple[float, float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    valid = np.where(precision[:-1] >= precision_floor)[0]
    if len(valid) == 0:
        return np.nan, np.nan, np.nan
    best = valid[np.argmax(recall[:-1][valid])]
    return float(thresholds[best]), float(precision[best]), float(recall[best])


# ── Binary classification metrics ─────────────────────────────────────────────

def eval_binary(y_true: np.ndarray, pred: np.ndarray, score: np.ndarray, name: str) -> dict:
    return {
        "model":     name,
        "AP":        average_precision_score(y_true, score),
        "ROC_AUC":   roc_auc_score(y_true, score),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall":    recall_score(y_true, pred, zero_division=0),
        "F1":        f1_score(y_true, pred, zero_division=0),
        "alerts":    int(pred.sum()),
    }


# ── Fusion search ──────────────────────────────────────────────────────────────

def _best_at_floor(y_true: np.ndarray, score: np.ndarray, floor: float) -> dict | None:
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    ok = np.flatnonzero(precision[:-1] >= floor)
    if len(ok) == 0:
        return None
    best_recall = recall[ok].max()
    cand = ok[recall[ok] >= best_recall - 1e-12]
    i    = cand[np.argmax(precision[cand])]
    return {
        "floor":     float(floor),
        "threshold": float(thresholds[i]),
        "precision": float(precision[i]),
        "recall":    float(recall[i]),
        "f1":        float(2 * precision[i] * recall[i] / max(precision[i] + recall[i], 1e-12)),
    }


def fusion_search(
    pA_val: np.ndarray,
    pB_val: np.ndarray,
    y_val:  np.ndarray,
    floors: list[float] | None = None,
) -> pd.DataFrame:
    """
    Search weighted and non-linear fusion rules on validation.
    Returns a DataFrame with all evaluated operating points.
    """
    if floors is None:
        floors = [0.10,0.12,0.15, 0.20, 0.25, 0.30]

    rows = []

    for w in np.linspace(0.0, 1.0, 21):
        scores = w * pA_val + (1.0 - w) * pB_val
        for floor in floors:
            result = _best_at_floor(y_val, scores, floor)
            if result:
                rows.append({"family": "weighted", "rule": "weighted",
                             "w_pattern": float(w), "w_unpatterned": float(1.0 - w), **result})

    candidates = {
        "max":                    np.maximum(pA_val, pB_val),
        "geometric":              np.sqrt(np.clip(pA_val, 0, 1) * np.clip(pB_val, 0, 1)),
        "union":                  (1 - (1 - pA_val) * (1 - pB_val)).clip(0, 1),
        "pattern_plus_residual":  (pA_val + (1 - pA_val) * pB_val).clip(0, 1),
    }
    for name, scores in candidates.items():
        for floor in floors:
            result = _best_at_floor(y_val, scores, floor)
            if result:
                rows.append({"family": "nonlinear", "rule": name,
                             "w_pattern": np.nan, "w_unpatterned": np.nan, **result})

    return pd.DataFrame(rows)


def select_best_fusion(fusion_table: pd.DataFrame, floor: float = 0.15) -> pd.Series:
    feasible = fusion_table[
        (fusion_table["floor"] == floor) &
        (fusion_table["precision"] >= floor)
    ].copy()
    if feasible.empty:
        raise RuntimeError(f"No fusion operating point reaches the {floor:.0%} precision floor.")
    return feasible.sort_values(["recall", "precision"], ascending=[False, False]).iloc[0]
