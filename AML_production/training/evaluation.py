"""Threshold selection always uses development data at natural prevalence."""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from aml_app.service import combine_scores


def operating_point(y, scores, floor=0.15):
    y, scores = np.asarray(y), np.asarray(scores)
    if not 0 < floor <= 1 or not np.isfinite(scores).all():
        raise ValueError("Invalid precision floor or scores.")
    if not y.sum():
        raise ValueError("The development split needs positive examples.")
    p, r, t = precision_recall_curve(y, scores)
    candidates = np.flatnonzero(p[:-1] >= floor)
    if not len(candidates):
        return {"threshold": float(np.nextafter(scores.max(), np.inf)), "precision": 0., "recall": 0., "feasible": False}
    best = max(candidates, key=lambda i: (r[i], p[i]))
    return {"threshold": float(t[best]), "precision": float(p[best]), "recall": float(r[best]), "feasible": True}


def metrics(y, scores, threshold):
    y, scores = np.asarray(y, dtype=bool), np.asarray(scores)
    alert = scores >= threshold
    tp = int((alert & y).sum()); fp = int((alert & ~y).sum())
    fn = int((~alert & y).sum()); tn = int((~alert & ~y).sum())
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    return {"rows": len(y), "positives": int(y.sum()), "alerts": int(alert.sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "average_precision": float(average_precision_score(y, scores)),
            "roc_auc": float(roc_auc_score(y, scores)) if 0 < y.sum() < len(y) else None,
            "alert_rate": float(alert.mean()), "threshold": float(threshold)}


def policy_scores(a, b, policy):
    return combine_scores(a, b, policy)
