"""Train, select, calibrate, test, and export one immutable AML candidate bundle.

Run `python -m training.train --help`. The original bundle is never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from aml_app.pipeline import ALIASES, FEATURE_VERSION, RAW_COLUMNS, build_features, normalize, read_transactions, source_hash
from .evaluation import metrics, operating_point, policy_scores


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""): h.update(part)
    return h.hexdigest()


def pattern_membership(raw, path):
    """Map only labelled positives; pattern metadata never influences time splits."""
    def key(row):
        fields = list(row[:10])
        fields[0] = pd.Timestamp(fields[0]).isoformat()
        for i in [5, 7]: fields[i] = f"{float(fields[i]):.6f}"
        return tuple(str(x).strip() for x in fields)
    signatures = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith(("BEGIN", "END")): continue
            fields = next(csv.reader([line]))
            if len(fields) >= 10: signatures.add(key(fields))
    raw = raw.rename(columns=ALIASES)
    y = raw["Is Laundering"].to_numpy(dtype="int8")
    result = np.zeros(len(raw), dtype=bool)
    positions = np.flatnonzero(y)
    for i, row in zip(positions, raw.iloc[positions][RAW_COLUMNS].itertuples(index=False, name=None)):
        result[i] = key(row) in signatures
    if not result.sum(): raise ValueError("No pattern positives matched; verify dataset/identifier formatting.")
    return result


def temporal_splits(timestamps):
    """60/10/10/20 time split: fit, tune, calibration, final test; no timestamp overlap."""
    t = pd.to_datetime(timestamps, utc=True, format="mixed").to_numpy()
    if not len(t) or np.any(t[1:] < t[:-1]): raise ValueError("Sort data chronologically first.")
    ends = [0]
    for fraction in [.6, .7, .8]:
        ends.append(int(np.searchsorted(t, t[int(len(t) * fraction)], side="left")))
    ends.append(len(t))
    if any(b <= a for a, b in zip(ends, ends[1:])):
        raise ValueError("Need enough distinct timestamps for four chronological splits.")
    return dict(zip(["train", "tune", "calibration", "test"], [slice(a, b) for a, b in zip(ends, ends[1:])]))


def prepare(args, root):
    print("Reading transactions (identifiers preserved as strings)", flush=True)
    raw = read_transactions(args.data, nrows=args.max_rows)
    if "Is Laundering" not in raw: raise ValueError("Training data requires Is Laundering.")
    if not raw["Is Laundering"].isin([0, 1]).all(): raise ValueError("Labels must be binary.")
    patterned = pattern_membership(raw, args.patterns)
    d = normalize(raw)
    order = d._input_order.to_numpy()
    y = raw["Is Laundering"].to_numpy(dtype="int8")[order]
    patterned = patterned[order]
    del raw; gc.collect()
    splits = temporal_splits(d.ts)
    summary = {}
    columns = None
    for name, selection in splits.items():
        start = time.perf_counter()
        part = d.iloc[selection].drop(columns="_input_order").rename(columns={"ts": "timestamp"})
        print(f"Building {name} features: {len(part):,} rows", flush=True)
        X, normalized = build_features(part)
        columns = X.columns.tolist()
        np.save(root / f"{name}_X.npy", X.to_numpy())
        np.save(root / f"{name}_y.npy", y[selection])
        np.save(root / f"{name}_patterned.npy", patterned[selection])
        if name == "train":
            # Training-only drift reference, for batch-level distribution checks.
            reference = {}
            for column in columns:
                values = X[column].to_numpy()
                interior = np.unique(np.quantile(values, np.linspace(0.1, 0.9, 9)))
                edges = np.r_[-np.inf, interior, np.inf]
                counts = np.histogram(values, bins=edges)[0]
                reference[column] = {"interior_edges": interior.tolist(), "fractions": (counts / counts.sum()).tolist()}
            write_json(root / "reference.json", reference)
        summary[name] = {"rows": len(part), "positives": int(y[selection].sum()),
                         "patterned_positives": int(patterned[selection].sum()),
                         "first_timestamp": part.timestamp.min().isoformat(), "last_timestamp": part.timestamp.max().isoformat(),
                         "feature_seconds": round(time.perf_counter() - start, 3), "history": "independent_batch"}
        print(summary[name], flush=True)
        del part, X, normalized; gc.collect()
    write_json(root / "features.json", columns)
    write_json(root / "splits.json", summary)
    write_json(root / "data_identity.json", {"transactions_sha256": sha256(args.data), "patterns_sha256": sha256(args.patterns),
                                            "max_rows": args.max_rows, "feature_source_sha256": source_hash()})


def validate_cache_identity(identity, args):
    expected = {
        "transactions_sha256": sha256(args.data),
        "patterns_sha256": sha256(args.patterns),
        "max_rows": args.max_rows,
        "feature_source_sha256": source_hash(),
    }
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("Feature cache does not match data, row limit, or feature source; use a new output directory.")


def run(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists(): raise ValueError("Output already contains a completed bundle; choose a new directory.")
    if args.prepare_only or not (root / "features.json").exists(): prepare(args, root)
    if args.prepare_only: return
    identity = json.loads((root / "data_identity.json").read_text())
    validate_cache_identity(identity, args)
    columns = json.loads((root / "features.json").read_text())
    X = {name: np.load(root / f"{name}_X.npy", mmap_mode="r") for name in ["train", "tune", "calibration", "test"]}
    y = {name: np.load(root / f"{name}_y.npy") for name in X}
    patterned = np.load(root / "train_patterned.npy")
    for name in y:
        if y[name].sum() < 10: raise ValueError(f"{name} has fewer than 10 positives; use more data.")
    rng = np.random.default_rng(args.seed)
    negatives = np.flatnonzero(y["train"] == 0)
    sample = rng.choice(negatives, min(args.max_negatives, len(negatives)), replace=False)
    positive = np.flatnonzero(y["train"])
    normal_train = np.sort(np.r_[sample, positive])
    specialist_train = np.sort(np.r_[sample, np.flatnonzero(patterned)])
    # One fixed sample is shared by all candidates; validation/test remain complete.
    configs = [(4, 1), (6, 1), (6, 10), (8, 10)] if not args.quick else [(4, 1), (6, 10)]
    trials = []; best = {}; run_id = None
    if args.mlflow_uri:
        import mlflow
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment("aml-causal-batch-v2")
        active = mlflow.start_run(run_name=root.name)
        run_id = active.info.run_id
        mlflow.log_params({"feature_version": FEATURE_VERSION, "precision_floor": args.precision_floor,
                           "max_negative_training_rows": args.max_negatives, "seed": args.seed,
                           "development_only": bool(args.max_rows), "feature_count": len(columns)})
        mlflow.set_tags({"data_sha256": identity["transactions_sha256"], "feature_source_sha256": source_hash()})
    try:
        for model_name, indices in [("A", normal_train), ("B", specialist_train)]:
            for depth, weight in configs:
                name = f"{model_name}_d{depth}_w{weight}"
                print(f"Training {name}: {len(indices):,} sampled rows", flush=True)
                clf = xgb.XGBClassifier(n_estimators=args.trees, max_depth=depth, learning_rate=.07,
                       min_child_weight=5, reg_lambda=10, reg_alpha=.1, subsample=.85, colsample_bytree=.9,
                       scale_pos_weight=weight, tree_method="hist", n_jobs=args.jobs, random_state=args.seed,
                       objective="binary:logistic", eval_metric="aucpr", early_stopping_rounds=40)
                started = time.perf_counter()
                clf.fit(X["train"][indices], y["train"][indices], eval_set=[(X["tune"], y["tune"])], verbose=False)
                scores = clf.predict_proba(X["tune"])[:, 1]
                op = operating_point(y["tune"], scores, args.precision_floor)
                result = {"name": name, "model": model_name, "depth": depth, "weight": weight,
                          "best_iteration": clf.best_iteration, "seconds": time.perf_counter() - started,
                          **metrics(y["tune"], scores, op["threshold"]), "feasible": op["feasible"]}
                trials.append(result); write_json(root / "trials.json", trials)
                print(json.dumps(result), flush=True)
                if args.mlflow_uri:
                    for key in ["average_precision", "precision", "recall", "alert_rate", "seconds"]:
                        mlflow.log_metric(f"tune.{name}.{key}", result[key])
                rank = (op["feasible"], result["recall"], result["average_precision"])
                if model_name not in best or rank > best[model_name][0]:
                    best[model_name] = (rank, clf, scores, name)
                del clf; gc.collect()
        a, b = best["A"][2], best["B"][2]
        policies = [{"rule": "model_a"}, {"rule": "model_b"}, {"rule": "max"}]
        policies += [{"rule": "weighted", "weight_a": w} for w in [.25, .5, .75]]
        evaluated = []
        for policy in policies:
            scores = policy_scores(a, b, policy)
            op = operating_point(y["tune"], scores, args.precision_floor)
            evaluated.append({**policy, **op, "average_precision": float(metrics(y["tune"], scores, op["threshold"])["average_precision"])})
        chosen = max(evaluated, key=lambda p: (p["feasible"], p["recall"], p["average_precision"]))
        policy = {k: chosen[k] for k in ["rule", "weight_a"] if k in chosen}
        write_json(root / "policy_search.json", evaluated)
        calibration_a = best["A"][1].predict_proba(X["calibration"])[:, 1]
        calibration_b = best["B"][1].predict_proba(X["calibration"])[:, 1]
        calibration_scores = policy_scores(calibration_a, calibration_b, policy)
        op = operating_point(y["calibration"], calibration_scores, args.precision_floor)
        if not op["feasible"]: raise ValueError("Selected policy cannot meet precision floor on calibration data.")
        policy.update({"threshold": op["threshold"], "precision_floor": args.precision_floor,
                       "score_is_calibrated_probability": False, "selection_split": "tune", "threshold_split": "calibration"})
        write_json(root / "decision_policy.json", policy)
        floors = []
        for floor in [.1, .15, .2, .3, .4, .5, .6, .7, .8, .9]:
            floors.append({"precision_floor": floor, **operating_point(y["calibration"], calibration_scores, floor)})
        # Only now consult the untouched test once, with the rule and thresholds locked.
        test_a = best["A"][1].predict_proba(X["test"])[:, 1]
        test_b = best["B"][1].predict_proba(X["test"])[:, 1]
        test_scores = policy_scores(test_a, test_b, policy)
        thresholds = {key: operating_point(y["calibration"], values, args.precision_floor)["threshold"]
                      for key, values in [("OP_THRESHOLD_A", calibration_a), ("OP_THRESHOLD_B", calibration_b)]}
        results = {"development_only": bool(args.max_rows), "calibration": metrics(y["calibration"], calibration_scores, op["threshold"]),
                   "test": metrics(y["test"], test_scores, op["threshold"]),
                   "model_a_test": metrics(y["test"], test_a, thresholds["OP_THRESHOLD_A"]),
                   "model_b_test": metrics(y["test"], test_b, thresholds["OP_THRESHOLD_B"]),
                   "selected_models": {k: v[3] for k, v in best.items()}, "decision_policy": policy,
                   "comparison_note": "Original notebook metrics use different split/feature semantics and are not a controlled comparison."}
        for row in floors:
            row["test_at_locked_threshold"] = metrics(y["test"], test_scores, row["threshold"])
        write_json(root / "precision_recall_tradeoffs.json", floors)
        write_json(root / "evaluation.json", results)
        write_json(root / "thresholds.json", thresholds)
        pd.DataFrame({"feature": columns}).to_csv(root / "feature_columns.csv", index=False)
        for key, filename in [("A", "model_A_general_aml.json"), ("B", "model_B_pattern_specialist.json")]:
            model = best[key][1]
            model.get_booster().feature_names = columns
            model.save_model(root / filename)
        versions = {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "xgboost", "scikit-learn"]}
        files = ["model_A_general_aml.json", "model_B_pattern_specialist.json", "feature_columns.csv", "thresholds.json",
                 "decision_policy.json", "reference.json", "evaluation.json", "splits.json", "precision_recall_tradeoffs.json"]
        manifest = {"schema_version": 2, "model_version": root.name, "feature_version": FEATURE_VERSION,
                    "feature_source_sha256": source_hash(), "feature_count": len(columns), "clustering": False,
                    "history_mode": "independent_batch", "data": identity, "versions": versions,
                    "python": platform.python_version(), "mlflow_run_id": run_id,
                    "development_only": bool(args.max_rows), "files": {f: sha256(root / f) for f in files}}
        write_json(root / "manifest.json", manifest)
        if args.mlflow_uri:
            for split in ["calibration", "test"]:
                for key in ["precision", "recall", "average_precision", "alert_rate", "f1"]:
                    mlflow.log_metric(f"{split}.{key}", results[split][key])
            for name in files + ["manifest.json", "trials.json", "policy_search.json"]:
                mlflow.log_artifact(str(root / name), artifact_path="bundle")
        print(json.dumps(results, indent=2), flush=True)
        print(f"Candidate bundle complete: {root}", flush=True)
    finally:
        if args.mlflow_uri:
            mlflow.end_run(status="FAILED" if sys.exc_info()[0] else "FINISHED")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("../aml-detection-project/data/raw/HI-Small_Trans.csv"))
    parser.add_argument("--patterns", type=Path, default=Path("../aml-detection-project/data/external/HI-Small_Patterns.txt"))
    parser.add_argument("--output", default="artifacts/causal-v2")
    parser.add_argument("--precision-floor", type=float, default=.15)
    parser.add_argument("--max-negatives", type=int, default=250_000)
    parser.add_argument("--trees", type=int, default=400)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None, help="Development only; blocks production promotion.")
    parser.add_argument("--quick", action="store_true", help="Two candidates per detector instead of four.")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--mlflow-uri", default="sqlite:///mlflow.db")
    args = parser.parse_args()
    if not 0 < args.precision_floor <= 1 or args.max_negatives < 1 or args.trees < 1 or args.jobs < 1:
        parser.error("Invalid training limits or precision floor.")
    run(args)


if __name__ == "__main__": main()
