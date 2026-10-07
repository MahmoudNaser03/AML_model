# Operations

## Run and inspect

Run `docker compose up --build -d`. The API's `/health` returns 503 until a complete, compatible bundle loads. `/live` is process liveness only. All published ports bind to localhost.

| Service | Local URL | Purpose |
|---|---|---|
| API | http://localhost:8000/docs | Schema, single and CSV scoring |
| Analyst dashboard | http://localhost:8501 | Investigations, explanations, evaluation report |
| Ray | http://localhost:8265 | Replicas, resources, scheduling |
| Replica monitor | http://localhost:8502 | Received/completed scoring requests |
| Prometheus | http://localhost:9090 | Metrics, targets, alert rules |
| Grafana | http://localhost:3000 | Metrics, traces and logs |
| MLflow (separate command) | http://localhost:5000 | Experiment trials and artifacts |

Start MLflow with `.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000 --workers 1`. Training defaults to that local database and records parameters, trial metrics, final test metrics, dataset/feature hashes and the full bundle.

## Useful monitoring

Prometheus `/metrics` aggregates worker metrics using a shared, run-specific multiprocess directory. It does not randomly sample one replica's counters. Ray's own metrics are scraped separately using Ray service discovery. Graceful replica shutdown removes live gauge files. A hard-killed worker may require a deployment restart to clear a stale live gauge; cumulative counters intentionally survive replica removal within a deployment run.

Grafana AML API Overview includes error fraction, readiness, scoring throughput, alert fraction, stage latency, explanation latency/cache misses and batch feature PSI. Ray Serve Replica Behavior shows desired/running replicas, queue pressure and per-replica activity. Prometheus evaluates five rules for API availability, unavailable artifacts, scoring failures, slow scoring and large alert-volume changes. Rules are visible in Prometheus; outbound paging/Alertmanager routing is not configured.

For one request, use Tempo:

```traceql
{ resource.service.name = "aml-model-api" && name = "aml.predict.csv" }
```

For correlated logs, use Loki:

```logql
{service_name="aml-model-api"} |= "score_completed"
```

Add the request ID from the response header to narrow the search. Logs carry trace/span IDs. Application telemetry excludes raw rows and feature values; exception recording retains the exception type rather than potentially sensitive submitted values.

PSI compares sufficiently large submitted batches with training feature distributions. Small batches return `insufficient_sample`. A shorter history window itself changes features: PSI is not proof of accuracy loss. Actual recall/precision require later ground-truth labels.

## Batch behavior and resource limits

Each CSV is an independent chronological batch. The API accepts at most 50,000 rows and 20 MiB of CSV content. CPU-heavy CSV processing runs off the async event loop. A single transaction starts cold. Per-model diagnostic alert flags do not override the deployed decision policy.

Ray starts with one replica, targets one ongoing request per replica, accepts up to two, and caps desired replicas at ten. Each requests 0.5 CPU and each XGBoost scorer uses one thread. Available CPUs/RAM constrain actual replica count. A 512-MiB Ray object store fits the configured shared memory allocation.

Alert explanation features are cached in SQLite for 15 minutes, with ten batches/64 MiB of compressed payload capacity. Replicas on this local Ray node share the cache; an explanation can be served by a different replica. Entries are bound to model version. Cache eviction/restarts can require rescoring. This local SQLite design is not a distributed cache for multiple machines.

## Access and data retention

Set `AML_API_KEY` for authenticated scoring, schema, model and monitor routes; the dashboard and load generator forward it. `DEPLOYMENT_ENVIRONMENT=production` refuses protected requests/readiness when no key is configured. This is shared-key access, not per-analyst roles. Health and metrics are operational endpoints kept on the local network.

Dashboard disk history is **off by default**. Set `AML_PERSIST_HISTORY=true` to enable concurrent SQLite history, with 30-day retention when saving/loading and a 100k-row chart view. Session history is retained only within that browser session. Use a TLS/authentication gateway and an approved data policy before exposing the stack outside localhost.

## Release and rollback

1. Train into a new artifact directory. Inspect the notebook, recorded precision/recall and policy choice.
2. Run `python -m pytest tests -q` and validate a candidate with `AMLScorer(path)`.
3. Run `python -m training.promote artifacts/<release>`. Default acceptance gates are recall >= 0.70 and precision >= 0.15 on the locked test report. Development-only, incompatible or damaged bundles fail.
4. Recreate the service with `docker compose up -d --force-recreate ray-serve`; model files are loaded at startup.
5. Run `python -m scripts.check_stack`, then optionally `python -m scripts.verify_autoscaling`.

Promotion retains the previous directory. To roll back, stop the API, move the current artifact directory aside, restore a compatible saved directory to `production_artifacts`, and recreate the service. The pre-v2 backup also needs the matching old application code; current code intentionally rejects it.

`docker compose down` stops the stack while preserving volumes. Do not add `--volumes` unless deleting dashboard/cache/telemetry history is intended.

## Known scope limits

- Independent batches, not continuously maintained customer history.
- Synthetic dataset results, not validation on a bank's live population.
- Local single-node Ray deployment; no automatic multi-node provisioning or shared network cache.
- No claim that model scores are calibrated probabilities.
- No investigator case-management workflow, identity provider, or automatic reporting.
