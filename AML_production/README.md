# AML Risk Intelligence

An offline AML investigation service with a shared training/serving feature pipeline, validated XGBoost decision policy, Ray Serve autoscaling, and operational monitoring.

## Current measured result

The selected entity-risk ensemble catches **81.03%** of positives at **21.35% precision** on the corrected chronological test split. Average precision is **0.7013**. On the identical split, the previous release caught 72.47% at 20.30% precision. The new policy catches 154 additional positives while also improving precision.

These are full test-population results from the synthetic IBM HI-Small dataset. The old notebook used different split/feature semantics and is not a controlled comparison. Start with the [end-to-end guide](docs/END_TO_END.md), then see the detailed [results](docs/RESULTS.md).

## Start locally

Python 3.11+; the current complete bundle must be present in `production_artifacts/`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn aml_app.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
source .venv/bin/activate
streamlit run dashboard.py
```

For the full local stack:

```bash
docker compose up --build -d
```

- Analyst UI: http://localhost:8501
- API docs: http://localhost:8000/docs
- Ray: http://localhost:8265
- Replica monitor: http://localhost:8502
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090

Published ports bind to localhost. Configure optional local access keys/passwords using [.env.example](.env.example). Grafana retains its password in an existing volume.

## Research and retraining

Install `requirements-training.txt`, then run:

```bash
python -m training.train \
  --data ../aml-detection-project/data/raw/HI-Small_Trans.csv \
  --patterns ../aml-detection-project/data/external/HI-Small_Patterns.txt \
  --output artifacts/new-release \
  --precision-floor 0.15 --trees 400 --jobs 2

python -m training.train_entity_v3 \
  --data ../aml-detection-project/data/raw/HI-Small_Trans.csv \
  --base artifacts/new-release \
  --output artifacts/entity-v3 --jobs 2
```

Use a new output directory for each experiment. The notebook calls this same training code. It defaults to displaying the completed experiment; training is opt-in through its `RUN_TRAINING` flag.

For a research-only second-stage experiment, open
[`notebooks/false_positive_pruning_experiment.ipynb`](notebooks/false_positive_pruning_experiment.ipynb).
It first clusters the existing alert queue, adds cluster membership and
centroid distances to a supervised false-positive ranker, locks the pruning
choice on calibration, and shows the precision/recall trade-off before and
after pruning. It does not modify or promote the production model bundle.

The pipeline:

1. Uses strict chronological fit/tune/calibration/test partitions with no shared timestamps.
2. Generates 63 causal batch features from the same implementation used by serving.
3. Samples normal training rows only; preserves complete validation/test populations.
4. Compares general and patterned-specialist XGBoost candidates.
5. Chooses a model/fusion rule on tune, locks its threshold on calibration, then evaluates test.
6. The v3 stage adds 35 train-cutoff entity-risk features, represented at serving by a checksummed registry of 128-bit hashes and aggregate counts.
7. Records trial metrics, provenance, hashes, and a complete artifact bundle.

The current winner is a geometric fusion of the prior specialist, an entity-aware general model, and an entity-aware patterned specialist. Component scores remain diagnostic; only the exported policy determines the final alert. See the [recall model review](docs/RECALL_MODEL_REVIEW.md) for rejected alternatives and limitations.

Read [the feature audit and research references](docs/FEATURE_AUDIT.md) before changing feature definitions.

## API and data contract

CSV accepts these normalized columns or their original HI-Small equivalents:

```text
timestamp,from_bank,from_account,to_bank,to_account,amount_received,
receiving_currency,amount_paid,payment_currency,payment_format
```

Identifiers remain strings, including leading zeros. Amounts must be finite and nonnegative. Blank identifiers and invalid timestamps are rejected. Extra CSV fields, including labels and precomputed features, never enter the model.

In the dashboard's CSV tab, enable **Compare alerts with actual labels** and select a binary ground-truth column (the HI-Small `Is Laundering` column is detected automatically) to see TP, FP, FN, TN, precision, recall, specificity, F1, and accuracy. An evaluation-threshold slider previews the recall/precision trade-off without changing the deployed decision policy. Use a representative evaluation file containing both positive and negative labels before selecting a production threshold. Labels are used only in dashboard evaluation and are not saved in dashboard history. The deployed entity registry is frozen at the training cutoff; uploaded labels never update it or enter model features. The local `data/` directory remains excluded from Git, Docker build contexts, and release manifests.

| Route | Purpose |
|---|---|
| `GET /live` | Process liveness |
| `GET /health` | Model readiness; 503 on incompatible/missing artifacts |
| `GET /model` | Release, deployed policy and validation results |
| `GET /schema` | Input contract and limits |
| `POST /predict` | One JSON transaction plus policy-specific explanation |
| `POST /predict/csv` | Multipart CSV field `file`; scores, decisions and batch drift |
| `GET /predict/csv/{batch_id}/explain/{record_id}` | Explanation for a cached alert |
| `GET /metrics` | Aggregated Prometheus metrics |

A CSV is an independent investigation. Features use earlier rows in that CSV only; previous uploads are not model history. Equal timestamps preserve source order. Single transactions start cold. Scores rank transactions and are **not calibrated laundering probabilities**.

The default CSV limits are 50,000 rows and 20 MiB. Explanations follow the deployed rule and use the same early-stopped trees as scoring. Their feature contributions are in each model's log-odds units.

## Release and verify

```bash
python -m pytest tests -q
python -m training.promote artifacts/entity-v3
docker compose up -d --force-recreate ray-serve
python -m scripts.check_stack
python -m scripts.verify_autoscaling
```

Promotion verifies hashes, feature version and minimum precision/recall gates, rejects development-only artifacts, and retains a timestamped backup. Model artifacts are ignored by Git; distribute the entire manifest-listed bundle through the recorded MLflow run or your artifact repository.

## Monitoring and experiment tracking

```bash
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000 --workers 1
```

MLflow contains the trial search, final metrics, provenance and artifact bundle. Grafana combines Prometheus metrics, Tempo traces and Loki logs. Alert rules cover failed readiness, errors, latency and unusual alert volume. Feature PSI detects distribution differences; it does not measure precision or recall.

Dashboard persistence is off by default. Enabling `AML_PERSIST_HISTORY=true` uses SQLite with 30-day retention; it does not append new schemas to the old CSV. A shared SQLite explanation cache supports replicas on the local Ray node.

See [operations](docs/OPERATIONS.md) for configuration, cache behavior, tracing queries, rollback and deployment limits. External deployment still needs institution-specific access, TLS, data governance and model validation.

## Layout

```text
aml_app/pipeline.py           Shared, versioned feature implementation
aml_app/entity_features.py    Hashed, train-cutoff aggregate entity features
aml_app/features.py           Public feature interface
aml_app/service.py            Artifact validation, policy scoring, explanations
aml_app/main.py               API, input limits, access control, metrics
aml_app/cache.py              Shared local explanation cache
training/                    Train, evaluate, export and promote
tests/                       Feature, evaluation, artifact and API regressions
scripts/                     Live-stack and autoscaling verification
observability/               Prometheus, Grafana, OpenTelemetry, Tempo, Loki
docs/                        Results, feature audit and operations
production_artifacts/         Promoted immutable bundle (not versioned in Git)
```
