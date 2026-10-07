"""FastAPI ingress for versioned, independent-batch AML scoring."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hmac
import io
import json
import logging
import os
import re
import socket
import time
from uuid import uuid4

import pandas as pd
import ray
from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest, multiprocess
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from .artifacts import ArtifactError
from .cache import BatchCache
from .observability import configure_observability, get_tracer, record_span_error
from .pipeline import read_transactions
from .schemas import Transaction
from .service import AMLScorer

ARTIFACT_DIR = os.getenv("AML_ARTIFACT_DIR", "production_artifacts")
MAX_UPLOAD_BYTES = int(os.getenv("AML_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
MAX_BATCH_ROWS = int(os.getenv("AML_MAX_BATCH_ROWS", "50000"))
logger = logging.getLogger("aml.api")
scorer = None
batch_cache = None
REQUEST_MONITOR_NAME, REQUEST_MONITOR_NAMESPACE = "aml-request-monitor", "aml"

HTTP_REQUESTS = Counter("aml_http_requests_total", "HTTP outcomes", ["method", "path", "status"])
HTTP_DURATION = Histogram("aml_http_request_duration_seconds", "HTTP latency", ["method", "path"])
HTTP_IN_FLIGHT = Gauge("aml_http_in_flight_requests", "Active API requests", multiprocess_mode="livesum")
SCORING_BATCHES = Counter("aml_scoring_batches_total", "Completed scoring requests", ["source"])
SCORING_TRANSACTIONS = Counter("aml_scoring_transactions_total", "Scored transactions", ["source"])
SCORING_ALERTS = Counter("aml_scoring_alerts_total", "Alerts", ["source"])
SCORING_FAILURES = Counter("aml_scoring_failures_total", "Rejected scoring requests", ["source", "reason"])
SCORING_STAGE_DURATION = Histogram("aml_scoring_stage_duration_seconds", "Scoring stage latency", ["source", "stage"],
    buckets=(.001, .01, .05, .1, .5, 1, 2.5, 5, 10, 30, 60))
SCORING_BATCH_SIZE = Histogram("aml_scoring_batch_size", "Rows per batch", ["source"], buckets=(1, 10, 100, 1000, 10000, 50000))
SCORING_ALERT_RATE = Histogram("aml_scoring_alert_rate", "Fraction of alerted rows", ["source"], buckets=(0, .001, .01, .05, .1, .25, .5, 1))
SCORING_MAX_RISK = Histogram("aml_scoring_max_risk", "Maximum model score", ["source"])
EXPLANATION_DURATION = Histogram("aml_explanation_duration_seconds", "Explanation latency", ["model"])
MODEL_AVAILABLE = Gauge("aml_model_available", "Model bundle readiness", multiprocess_mode="livemin")
BATCH_DRIFT = Histogram("aml_batch_max_psi", "Maximum feature PSI in batches of at least 200 rows", buckets=(.05, .1, .2, .5, 1, 2, 5, 10))
CACHE_MISS = Counter("aml_explanation_cache_misses_total", "Expired/missing batch explanations")


@asynccontextmanager
async def lifespan(application):
    global scorer, batch_cache
    configure_observability(application)
    with get_tracer("aml.api").start_as_current_span("aml.model.initialization"):
        try:
            scorer = AMLScorer(ARTIFACT_DIR)
            MODEL_AVAILABLE.set(1)
            logger.info("model_initialized model_version=%s feature_count=%s", scorer.version, len(scorer.features))
        except (ArtifactError, OSError, ValueError) as exc:
            scorer = None
            MODEL_AVAILABLE.set(0)
            logger.error("model_initialization_failed error_type=%s detail=%s", type(exc).__name__, str(exc))
    batch_cache = BatchCache(os.getenv("AML_CACHE_PATH", "data/explanations.sqlite"))
    yield
    scorer = None
    if os.getenv("PROMETHEUS_MULTIPROC_DIR"):
        multiprocess.mark_process_dead(os.getpid())


app = FastAPI(title="AML Risk Scoring API", version="2.0.0", lifespan=lifespan)
FastAPIInstrumentor.instrument_app(app, excluded_urls="health,live,metrics")


def authorize(request: Request):
    expected = os.getenv("AML_API_KEY")
    if expected:
        actual = request.headers.get("x-api-key", "")
        if not hmac.compare_digest(actual.encode(), expected.encode()):
            raise HTTPException(401, "Invalid or missing API key.")
    elif os.getenv("DEPLOYMENT_ENVIRONMENT") == "production":
        raise HTTPException(503, "Production authentication is not configured.")


def get_scorer():
    if scorer is None: raise HTTPException(503, "Model bundle is unavailable. Check readiness and the release manifest.")
    return scorer


@app.exception_handler(RequestValidationError)
async def invalid_request(request, exc):
    # Pydantic error bodies can include raw input; return only field and error type.
    return JSONResponse(status_code=422, content={"detail": [{"loc": list(e["loc"]), "type": e["type"]} for e in exc.errors()]})


@app.middleware("http")
async def observe(request: Request, call_next):
    incoming_id = request.headers.get("x-request-id", "")
    request_id = incoming_id if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", incoming_id) else uuid4().hex
    request.state.request_id = request_id
    path = request.url.path
    scoring = path in {"/predict", "/predict/csv"}
    observed = path not in {"/metrics", "/health", "/live", "/monitor/metrics"}
    started, status = time.perf_counter(), 500
    monitor = None
    if path != "/metrics": HTTP_IN_FLIGHT.inc()
    try:
        if scoring and ray.is_initialized():
            try:
                monitor = ray.get_actor(REQUEST_MONITOR_NAME, namespace=REQUEST_MONITOR_NAMESPACE)
                await asyncio.wait_for(monitor.record_received.remote(request_id), timeout=1)
            except Exception:
                monitor = None
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = request_id
        response.headers["x-content-type-options"] = "nosniff"
        response.headers["cache-control"] = "no-store"
        return response
    finally:
        if monitor is not None:
            try: await asyncio.wait_for(monitor.record_completed.remote(request_id), timeout=1)
            except Exception: pass
        # Unmatched routes use a constant label: do not emit arbitrary paths/IDs.
        route = getattr(request.scope.get("route"), "path", "unmatched")
        elapsed = time.perf_counter() - started
        if path != "/metrics":
            HTTP_REQUESTS.labels(request.method, route, str(status)).inc()
            HTTP_DURATION.labels(request.method, route).observe(elapsed)
            HTTP_IN_FLIGHT.dec()
        if observed:
            logger.info("request_complete %s", json.dumps({"event": "request_complete", "request_id": request_id,
                "route": route, "status": status, "duration_ms": round(elapsed * 1000, 2),
                "replica_host": socket.gethostname(), "pid": os.getpid()}))


def record_scoring(source, result, timings, request_id):
    transactions, alerts = len(result), int(result.alert.sum())
    SCORING_BATCHES.labels(source).inc()
    SCORING_TRANSACTIONS.labels(source).inc(transactions)
    SCORING_ALERTS.labels(source).inc(alerts)
    SCORING_BATCH_SIZE.labels(source).observe(transactions)
    SCORING_ALERT_RATE.labels(source).observe(alerts / transactions)
    SCORING_MAX_RISK.labels(source).observe(float(result.risk_score.max()))
    for stage, elapsed in timings.items(): SCORING_STAGE_DURATION.labels(source, stage).observe(elapsed)
    logger.info("score_completed %s", json.dumps({"event": "score_completed", "request_id": request_id,
        "source": source, "transactions": transactions, "alerts": alerts, "model_version": get_scorer().version,
        "timings_ms": {key: round(value * 1000, 2) for key, value in timings.items()}}))


def public_records(result):
    out = result.drop(columns=["_input_order"]).rename(columns={"ts": "timestamp"}).copy()
    out["timestamp"] = result.ts.map(lambda ts: ts.isoformat())
    out["record_id"] = result._input_order.to_numpy()
    # Converts numpy scalars/category values without retaining user-supplied extras.
    return json.loads(out.to_json(orient="records", double_precision=12))


@app.get("/live", include_in_schema=False)
def live():
    return {"status": "alive"}


@app.get("/health")
def health(response: Response):
    ready = scorer is not None and not (os.getenv("DEPLOYMENT_ENVIRONMENT") == "production" and not os.getenv("AML_API_KEY"))
    response.status_code = 200 if ready else 503
    return {"status": "ok" if ready else "not_ready", "mode": "offline_batch", "model_version": scorer.version if scorer else None}


@app.get("/model", dependencies=[Depends(authorize)])
def model_info():
    s = get_scorer()
    return {"model_version": s.version, "feature_version": s.manifest["feature_version"],
            "feature_count": len(s.features), "decision_policy": s.policy,
            "evaluation": json.loads((s.dir / "evaluation.json").read_text())}


@app.get("/schema", dependencies=[Depends(authorize)])
def schema():
    return {"required_columns": list(Transaction.model_fields), "max_batch_rows": MAX_BATCH_ROWS, "max_upload_bytes": MAX_UPLOAD_BYTES}


@app.get("/feature-store/status", dependencies=[Depends(authorize)])
def feature_store_status():
    return {"enabled": False, "mode": "offline_batch", "detail": "History is limited to earlier rows in the submitted batch."}


@app.get("/metrics", include_in_schema=False)
def metrics():
    if os.getenv("PROMETHEUS_MULTIPROC_DIR"):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/monitor/metrics", dependencies=[Depends(authorize)])
async def monitor_metrics():
    if not ray.is_initialized(): raise HTTPException(503, "Request monitor is unavailable.")
    try:
        monitor = ray.get_actor(REQUEST_MONITOR_NAME, namespace=REQUEST_MONITOR_NAMESPACE)
        return await asyncio.wait_for(monitor.snapshot.remote(), timeout=2)
    except Exception as exc:
        raise HTTPException(503, "Request monitor is unavailable.") from exc


@app.post("/monitor/reset", dependencies=[Depends(authorize)])
async def reset_monitor_metrics():
    if not ray.is_initialized(): raise HTTPException(503, "Request monitor is unavailable.")
    try:
        monitor = ray.get_actor(REQUEST_MONITOR_NAME, namespace=REQUEST_MONITOR_NAMESPACE)
        await asyncio.wait_for(monitor.reset.remote(), timeout=2)
        return {"status": "reset"}
    except Exception as exc:
        raise HTTPException(503, "Request monitor is unavailable.") from exc


@app.post("/predict", dependencies=[Depends(authorize)])
def predict(transaction: Transaction, request: Request):
    s = get_scorer()
    with get_tracer("aml.api").start_as_current_span("aml.predict.json"):
        try:
            result, X, timings = s.score_with_observability(pd.DataFrame([transaction.model_dump()]))
            with get_tracer("aml.api").start_as_current_span("aml.explain.single"):
                started = time.perf_counter()
                explained = s.explain_decision(X, 0, result.iloc[0])
                EXPLANATION_DURATION.labels(s.policy["rule"]).observe(time.perf_counter() - started)
            record_scoring("json", result, timings, request.state.request_id)
            return {"prediction": public_records(result)[0], **explained,
                    "note": "One-row submissions have no preceding history. Scores are not calibrated probabilities."}
        except ValueError as exc:
            SCORING_FAILURES.labels("json", "validation").inc()
            raise HTTPException(400, str(exc)) from exc


@app.post("/predict/csv", dependencies=[Depends(authorize)])
def predict_csv(request: Request, file: UploadFile = File(...)):
    # A sync route sends CPU work to FastAPI's thread pool instead of blocking the ASGI loop.
    s = get_scorer()
    with get_tracer("aml.api").start_as_current_span("aml.predict.csv"):
        try:
            if not file.filename or not file.filename.lower().endswith(".csv"):
                raise HTTPException(400, "Upload a CSV file.")
            content = file.file.read(MAX_UPLOAD_BYTES + 1)
            if len(content) > MAX_UPLOAD_BYTES: raise HTTPException(413, "CSV exceeds the upload-size limit.")
            raw = read_transactions(io.BytesIO(content), nrows=MAX_BATCH_ROWS + 1)
            if len(raw) > MAX_BATCH_ROWS: raise HTTPException(413, "CSV exceeds the transaction-count limit.")
            result, X, timings = s.score_with_observability(raw)
        except HTTPException:
            SCORING_FAILURES.labels("csv", "limit_or_format").inc()
            raise
        except (ValueError, pd.errors.ParserError, UnicodeError) as exc:
            SCORING_FAILURES.labels("csv", "validation").inc()
            record_span_error(exc)
            # Parser errors may contain submitted values; never echo those.
            raise HTTPException(400, "Invalid CSV. Check the schema, timestamps, identifiers, and finite nonnegative amounts.") from exc
        except Exception as exc:
            SCORING_FAILURES.labels("csv", "internal").inc()
            record_span_error(exc)
            raise HTTPException(500, "Scoring failed; use the request ID to investigate.") from exc
        batch_id = uuid4().hex
        decisions = result.set_index("_input_order")[["model_a_score", "model_b_score", "alert"]]
        cached = batch_cache.put(batch_id, X, decisions, s.version)
        drift = s.drift(X)
        if drift["status"] == "computed": BATCH_DRIFT.observe(drift["max_psi"])
        record_scoring("csv", result, timings, request.state.request_id)
        return {"batch_id": batch_id, "explanations_cached": cached, "model_version": s.version,
                "decision_policy": s.policy, "drift": drift,
                "summary": {"transactions": len(result), "alerts": int(result.alert.sum()),
                    "alert_rate": float(result.alert.mean()), "max_risk": float(result.risk_score.max())},
                "records": public_records(result)}


@app.get("/predict/csv/{batch_id}/explain/{record_id}", dependencies=[Depends(authorize)])
def explain_batch_alert(batch_id: str, record_id: int):
    s = get_scorer()
    cached = batch_cache.get(batch_id, s.version)
    if cached is None:
        CACHE_MISS.inc()
        raise HTTPException(404, "Batch expired or model release changed. Score the CSV again.")
    X, decisions = cached
    if record_id not in X.index:
        raise HTTPException(404, "No alerted transaction with this record ID exists in the batch.")
    with get_tracer("aml.api").start_as_current_span("aml.explain.alert"):
        started = time.perf_counter()
        result = s.explain_decision(X.astype("float32"), record_id, decisions.loc[record_id])
        EXPLANATION_DURATION.labels(s.policy["rule"]).observe(time.perf_counter() - started)
        return result
