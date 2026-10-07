"""Train and export the recall-oriented entity-risk ensemble.

The raw data directory is read in place and is never copied into the release.
Candidate selection uses tune, the threshold uses calibration, and test is only
reported after both are fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from aml_app.entity_features import (
    ENTITY_FEATURE_COLUMNS,
    ENTITY_FEATURE_VERSION,
    EntityRegistry,
    build_entity_features,
    fit_registry,
)
from aml_app.entity_features import (
    source_hash as entity_source_hash,
)
from aml_app.pipeline import FEATURE_VERSION, normalize, read_transactions, source_hash
from training.evaluation import metrics, operating_point
from training.train import sha256, temporal_splits, write_json


def geometric(scores, weights):
    return np.exp(sum(weight * np.log(np.maximum(score, 1e-12)) for score, weight in zip(scores, weights)))


def candidate(depth, jobs, seed):
    return xgb.XGBClassifier(
        n_estimators=1400,
        max_depth=depth,
        learning_rate=.025,
        min_child_weight=5,
        reg_lambda=10,
        reg_alpha=.1,
        subsample=.9,
        colsample_bytree=.9,
        tree_method="hist",
        n_jobs=jobs,
        random_state=seed,
        objective="binary:logistic",
        eval_metric="aucpr",
        early_stopping_rounds=100,
    )


def run(args):
    base = args.base
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Output directory must be empty; releases are immutable.")
    identity = json.loads((base / "data_identity.json").read_text())
    if sha256(args.data) != identity["transactions_sha256"]:
        raise ValueError("Source data does not match the validated causal-v2 baseline.")
    raw = read_transactions(args.data)
    if "Is Laundering" not in raw or not raw["Is Laundering"].isin([0, 1]).all():
        raise ValueError("Training data requires binary Is Laundering labels.")
    normalized = normalize(raw)
    order = normalized._input_order.to_numpy()
    y_all = raw["Is Laundering"].to_numpy(dtype="int8")[order]
    splits = temporal_splits(normalized.ts)
    names = ["train", "tune", "calibration", "test"]
    y = {name: np.load(base / f"{name}_y.npy") for name in names}
    patterned = {name: np.load(base / f"{name}_patterned.npy").astype(bool) for name in names}
    for name in names:
        if not np.array_equal(y_all[splits[name]], y[name]):
            raise ValueError(f"{name} labels do not match the baseline split.")

    registry_path = output / "entity_registry.npz"
    entity = {}
    started = time.perf_counter()
    train_rows = normalized.iloc[splits["train"]]
    entity["train"] = fit_registry(train_rows, y["train"], registry_path).to_numpy()
    registry = EntityRegistry(registry_path)
    for name in names[1:]:
        entity[name] = build_entity_features(normalized.iloc[splits[name]], registry).to_numpy()
    print(f"Entity registry and matrices: {time.perf_counter() - started:.1f}s", flush=True)

    base_columns = json.loads((base / "features.json").read_text())
    feature_columns = [*base_columns, *ENTITY_FEATURE_COLUMNS]
    X = {name: np.load(base / f"{name}_X.npy", mmap_mode="r") for name in names}
    rng = np.random.default_rng(args.seed)
    negative = rng.permutation(np.flatnonzero(y["train"] == 0))[:args.max_negatives]
    populations = {
        "A": np.sort(np.r_[negative, np.flatnonzero(y["train"] == 1)]),
        "B": np.sort(np.r_[negative, np.flatnonzero(patterned["train"])]),
    }
    tune_matrix = np.column_stack([X["tune"], entity["tune"]])
    models = {}
    tune_scores = {}
    trials = []
    for name in ["A", "B"]:
        indices = populations[name]
        model = candidate(3, args.jobs, args.seed)
        started = time.perf_counter()
        model.fit(
            np.column_stack([X["train"][indices], entity["train"][indices]]),
            y["train"][indices],
            eval_set=[(tune_matrix, y["tune"])],
            verbose=False,
        )
        tune_scores[name] = model.predict_proba(tune_matrix)[:, 1]
        point = operating_point(y["tune"], tune_scores[name], args.tune_precision_floor)
        row = {
            "name": name,
            "best_iteration": model.best_iteration,
            "seconds": time.perf_counter() - started,
            **metrics(y["tune"], tune_scores[name], point["threshold"]),
        }
        trials.append(row)
        models[name] = model
        print(json.dumps(row), flush=True)

    baseline = xgb.XGBClassifier()
    baseline.load_model(base / "model_B_pattern_specialist.json")
    baseline.set_params(n_jobs=args.jobs)
    baseline_tune = baseline.predict_proba(X["tune"])[:, 1]
    policies = [
        {"weights": {"baseline_b": .05, "model_a": .75, "model_b": .20}},
        {"weights": {"baseline_b": .25, "model_a": .75, "model_b": 0}},
        {"weights": {"baseline_b": 0, "model_a": 1, "model_b": 0}},
        {"weights": {"baseline_b": 0, "model_a": 0, "model_b": 1}},
    ]
    for policy in policies:
        weights = policy["weights"]
        score = geometric(
            [baseline_tune, tune_scores["A"], tune_scores["B"]],
            [weights["baseline_b"], weights["model_a"], weights["model_b"]],
        )
        point = operating_point(y["tune"], score, args.tune_precision_floor)
        policy.update(point)
    selected = max(policies, key=lambda value: (value["feasible"], value["recall"], value["precision"]))
    weights = selected["weights"]
    policy = {
        "rule": "geometric_three",
        "weights": weights,
        "score_is_calibrated_probability": False,
        "selection_split": "tune",
        "threshold_split": "calibration",
        "tune_precision_floor": args.tune_precision_floor,
    }

    scores = {}
    component_scores = {}
    for name in ["calibration", "test"]:
        matrix = np.column_stack([X[name], entity[name]])
        component_scores[name] = {
            "baseline": baseline.predict_proba(X[name])[:, 1],
            "A": models["A"].predict_proba(matrix)[:, 1],
            "B": models["B"].predict_proba(matrix)[:, 1],
        }
        scores[name] = geometric(
            [component_scores[name]["baseline"], component_scores[name]["A"], component_scores[name]["B"]],
            [weights["baseline_b"], weights["model_a"], weights["model_b"]],
        )
    baseline_evaluation = json.loads((base / "evaluation.json").read_text())
    precision_target = baseline_evaluation["test"]["precision"]
    locked = operating_point(y["calibration"], scores["calibration"], precision_target)
    policy.update({"threshold": locked["threshold"], "precision_floor": precision_target})
    evaluation = {
        "development_only": False,
        "calibration": metrics(y["calibration"], scores["calibration"], locked["threshold"]),
        "test": metrics(y["test"], scores["test"], locked["threshold"]),
        "baseline_test": baseline_evaluation["test"],
        "selected_models": {"A": "entity_general_d3", "B": "entity_pattern_specialist_d3"},
        "decision_policy": policy,
        "comparison_note": "Both releases use the identical corrected temporal split; the threshold was locked on calibration.",
    }
    thresholds = {
        "OP_THRESHOLD_A": operating_point(y["calibration"], component_scores["calibration"]["A"], precision_target)["threshold"],
        "OP_THRESHOLD_B": operating_point(y["calibration"], component_scores["calibration"]["B"], precision_target)["threshold"],
        "OP_THRESHOLD_BASELINE_B": baseline_evaluation["decision_policy"]["threshold"],
    }

    for name, filename in [("A", "model_A_general_aml.json"), ("B", "model_B_pattern_specialist.json")]:
        models[name].get_booster().feature_names = feature_columns
        models[name].save_model(output / filename)
    shutil.copy2(base / "model_B_pattern_specialist.json", output / "model_baseline_b.json")
    shutil.copy2(base / "reference.json", output / "reference.json")
    shutil.copy2(base / "splits.json", output / "splits.json")
    pd.DataFrame({"feature": base_columns}).to_csv(output / "base_feature_columns.csv", index=False)
    pd.DataFrame({"feature": feature_columns}).to_csv(output / "feature_columns.csv", index=False)
    write_json(output / "thresholds.json", thresholds)
    write_json(output / "decision_policy.json", policy)
    write_json(output / "evaluation.json", evaluation)
    write_json(output / "trials.json", trials)
    write_json(output / "policy_search.json", policies)
    files = [path.name for path in output.iterdir()]
    manifest = {
        "schema_version": 3,
        "model_version": output.name,
        "feature_version": FEATURE_VERSION,
        "feature_source_sha256": source_hash(),
        "entity_feature_version": ENTITY_FEATURE_VERSION,
        "entity_feature_source_sha256": entity_source_hash(),
        "feature_count": len(feature_columns),
        "history_mode": "independent_batch_plus_fixed_train_registry",
        "data": identity,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "xgboost", "scikit-learn"]
        },
        "python": platform.python_version(),
        "development_only": False,
        "files": {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in files},
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(evaluation, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("../aml-detection-project/data/raw/HI-Small_Trans.csv"))
    parser.add_argument("--base", type=Path, default=Path("artifacts/causal-v2"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/entity-v3"))
    parser.add_argument("--tune-precision-floor", type=float, default=.15)
    parser.add_argument("--max-negatives", type=int, default=250_000)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
