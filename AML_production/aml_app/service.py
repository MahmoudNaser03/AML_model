"""Artifact-bound scoring, deployed decision policy, and native explanations."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .artifacts import ArtifactError, validate_bundle
from .entity_features import EntityRegistry, build_entity_features
from .observability import get_tracer, record_span_error
from .pipeline import build_features


def combine_scores(a, b, policy, baseline=None):
    rule = policy["rule"]
    if rule == "model_a": return a
    if rule == "model_b": return b
    if rule == "max": return np.maximum(a, b)
    if rule == "weighted": return policy["weight_a"] * a + (1 - policy["weight_a"]) * b
    if rule == "geometric_three":
        if baseline is None:
            raise ArtifactError("Three-model policy requires a baseline score.")
        weights = policy["weights"]
        return np.exp(
            weights["model_a"] * np.log(np.maximum(a, 1e-12))
            + weights["model_b"] * np.log(np.maximum(b, 1e-12))
            + weights["baseline_b"] * np.log(np.maximum(baseline, 1e-12))
        )
    raise ArtifactError("Unsupported decision policy.")


class AMLScorer:
    def __init__(self, artifact_dir, *, allow_development=False):
        self.dir = Path(artifact_dir)
        self.manifest = validate_bundle(self.dir, allow_development=allow_development)
        self.version = self.manifest["model_version"]
        self.features = pd.read_csv(self.dir / "feature_columns.csv")["feature"].tolist()
        self.schema = self.manifest["schema_version"]
        self.base_features = (
            pd.read_csv(self.dir / "base_feature_columns.csv")["feature"].tolist()
            if self.schema == 3 else self.features
        )
        self.policy = json.loads((self.dir / "decision_policy.json").read_text())
        self.thresholds = json.loads((self.dir / "thresholds.json").read_text())
        self.reference = json.loads((self.dir / "reference.json").read_text())
        self.a, self.b = xgb.XGBClassifier(), xgb.XGBClassifier()
        for model, filename in [(self.a, "model_A_general_aml.json"), (self.b, "model_B_pattern_specialist.json")]:
            model.load_model(self.dir / filename)
            model.set_params(n_jobs=int(os.getenv("AML_MODEL_THREADS", "1")))
            if model.n_features_in_ != len(self.features) or model.get_booster().feature_names != self.features:
                raise ArtifactError("Model and feature contract do not agree.")
        self.baseline = None
        self.entity_registry = None
        if self.schema == 3:
            self.baseline = xgb.XGBClassifier()
            self.baseline.load_model(self.dir / "model_baseline_b.json")
            self.baseline.set_params(n_jobs=int(os.getenv("AML_MODEL_THREADS", "1")))
            if self.baseline.n_features_in_ != len(self.base_features):
                raise ArtifactError("Baseline model and base feature contract do not agree.")
            self.entity_registry = EntityRegistry(self.dir / "entity_registry.npz")
        if len(self.features) != len(set(self.features)) or len(self.features) != self.manifest["feature_count"]:
            raise ArtifactError("Invalid feature contract.")
        if self.policy.get("rule") not in {"model_a", "model_b", "max", "weighted", "geometric_three"}:
            raise ArtifactError("Unsupported decision policy.")
        if not np.isfinite(self.policy["threshold"]) or not 0 < self.policy["threshold"] <= 1:
            raise ArtifactError("Invalid decision threshold.")
        if self.policy["rule"] == "weighted" and not 0 <= self.policy.get("weight_a", -1) <= 1:
            raise ArtifactError("Invalid fusion weight.")
        if self.policy["rule"] == "geometric_three":
            weights = self.policy.get("weights", {})
            if set(weights) != {"model_a", "model_b", "baseline_b"} or not all(
                0 <= value <= 1 for value in weights.values()
            ) or not np.isclose(sum(weights.values()), 1):
                raise ArtifactError("Invalid three-model fusion weights.")

    def _score_matrix(self, X, normalized):
        tracer = get_tracer("aml.scorer")
        with tracer.start_as_current_span("aml.model_a.inference"):
            a = self.a.predict_proba(X)[:, 1]
        with tracer.start_as_current_span("aml.model_b.inference"):
            b = self.b.predict_proba(X)[:, 1]
        baseline = None
        if self.baseline is not None:
            with tracer.start_as_current_span("aml.baseline_b.inference"):
                baseline = self.baseline.predict_proba(X.loc[:, self.base_features])[:, 1]
        score = combine_scores(a, b, self.policy, baseline)
        threshold = self.policy["threshold"]
        out = normalized.copy()
        out["model_a_score"], out["model_b_score"], out["risk_score"] = a, b, score
        if baseline is not None:
            out["baseline_b_score"] = baseline
        out["model_a_alert"] = a >= self.thresholds["OP_THRESHOLD_A"]
        out["model_b_alert"] = b >= self.thresholds["OP_THRESHOLD_B"]
        out["alert"] = score >= threshold
        out["near_alert"] = (~out.alert) & (score >= threshold * .9)
        out["risk_level"] = np.select([out.alert, out.near_alert, score >= threshold * .5],
                                         ["Alert", "Near threshold", "Review"], default="Routine")
        out["decision_threshold"] = threshold
        out["model_version"] = self.version
        return out.sort_values("_input_order", kind="stable")

    def score_with_observability(self, rows):
        started = time.perf_counter()
        tracer = get_tracer("aml.scorer")
        with tracer.start_as_current_span("aml.score") as span:
            span.set_attribute("aml.model.version", self.version)
            span.set_attribute("aml.input.rows", len(rows))
            try:
                with tracer.start_as_current_span("aml.feature_engineering"):
                    X, normalized = build_features(rows, self.base_features)
                    if self.entity_registry is not None:
                        entity = build_entity_features(normalized, self.entity_registry)
                        X = pd.concat([X, entity], axis=1).loc[:, self.features]
                feature_seconds = time.perf_counter() - started
                inference_started = time.perf_counter()
                out = self._score_matrix(X, normalized)
                timings = {"feature_engineering": feature_seconds, "model_inference": time.perf_counter() - inference_started,
                           "total_scoring": time.perf_counter() - started}
                return out, X, timings
            except Exception as exc:
                record_span_error(exc)
                raise

    def score_with_matrix(self, rows):
        result, X, _ = self.score_with_observability(rows)
        return result, X

    def score(self, rows):
        return self.score_with_observability(rows)[0]

    def drift(self, X):
        """Population stability index; batch drift is a diagnostic, not model accuracy."""
        if len(X) < 200:
            return {"status": "insufficient_sample", "minimum_rows": 200}
        values = {}
        for feature, ref in self.reference.items():
            edges = np.r_[-np.inf, ref["interior_edges"], np.inf]
            current = np.histogram(X[feature].to_numpy(), bins=edges)[0] / len(X)
            current, expected = np.maximum(current, 1e-6), np.maximum(ref["fractions"], 1e-6)
            values[feature] = float(np.sum((current - expected) * np.log(current / expected)))
        top = sorted(values.items(), key=lambda item: item[1], reverse=True)[:5]
        return {"status": "computed", "max_psi": max(values.values(), default=0),
                "features_over_0_2": sum(v > .2 for v in values.values()), "top_features": dict(top),
                "note": "Compared with training batches; different history-window sizes can also cause drift."}

    def explain_matrix_row(self, X, input_order, model="A"):
        if model not in {"A", "B", "BASELINE"}: raise ValueError("Unsupported explanation model.")
        if input_order not in X.index: raise ValueError("Transaction is absent from the scored batch.")
        if model == "A":
            clf, features = self.a, self.features
        elif model == "B":
            clf, features = self.b, self.features
        elif self.baseline is not None:
            clf, features = self.baseline, self.base_features
        else:
            raise ValueError("Baseline model is unavailable.")
        booster = clf.get_booster()
        # Respect early stopping: explanation and inference must use the same trees.
        end = clf.best_iteration + 1 if "best_iteration" in booster.attributes() else 0
        selected = X.loc[[input_order], features]
        contrib = booster.predict(xgb.DMatrix(selected, feature_names=features),
                                  pred_contribs=True, iteration_range=(0, end))[0, :-1]
        row = selected.iloc[0]
        return [{"model": model, "feature": features[i], "value": float(row.iloc[i]),
                 "impact": float(contrib[i]), "direction": "increases risk" if contrib[i] >= 0 else "reduces risk"}
                for i in np.argsort(abs(contrib))[::-1]]

    def explain_decision(self, X, input_order, decision):
        rule = self.policy["rule"]
        if rule == "model_a": models = ["A"]
        elif rule == "model_b": models = ["B"]
        elif rule == "max": models = ["A" if decision["model_a_score"] >= decision["model_b_score"] else "B"]
        elif rule == "geometric_three": models = ["A", "B", "BASELINE"]
        else: models = ["A", "B"]
        explanation = [item for model in models for item in self.explain_matrix_row(X, input_order, model)]
        return {"explanation_model": " + ".join("Model " + model for model in models),
                "decision_policy": self.policy, "explanation": explanation,
                "explanation_note": "Contributions are per-model log-odds, not probability changes. Fusion explanations are shown separately by model."}
