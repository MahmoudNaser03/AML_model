import asyncio
import hashlib
import json

import httpx2
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from aml_app.pipeline import FEATURE_VERSION, build_features, source_hash


class ASGITestClient:
    """Small synchronous facade over HTTPX's async in-process ASGI transport."""

    def __init__(self, app):
        self.app = app

    def request(self, method, path, **kwargs):
        async def send():
            transport = httpx2.ASGITransport(app=self.app)
            async with httpx2.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                return await client.request(method, path, **kwargs)

        return asyncio.run(send())

    def get(self, path, **kwargs):
        return self.request("GET", path, **kwargs)

    def post(self, path, **kwargs):
        return self.request("POST", path, **kwargs)


@pytest.fixture
def sample():
    return {
        "timestamp": "2026-01-01T12:00:00Z",
        "from_bank": "01",
        "from_account": "0001",
        "to_bank": "02",
        "to_account": "0002",
        "amount_paid": 1000.0,
        "amount_received": 995.0,
        "payment_currency": "USD",
        "receiving_currency": "USD",
        "payment_format": "ACH",
    }


@pytest.fixture(scope="session")
def bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("test-bundle")
    rows = []
    for i in range(60):
        rows.append(
            {
                "timestamp": f"2026-01-01T00:{i:02d}:00Z",
                "from_bank": "01",
                "from_account": str(i % 3),
                "to_bank": "02",
                "to_account": str(i % 7),
                "amount_paid": float(i * 100),
                "amount_received": float(i * 100),
                "payment_currency": "USD",
                "receiving_currency": "USD",
                "payment_format": "ACH",
            }
        )
    X, _ = build_features(pd.DataFrame(rows))
    y = np.arange(60) % 3 == 0
    for filename in ["model_A_general_aml.json", "model_B_pattern_specialist.json"]:
        model = xgb.XGBClassifier(
            n_estimators=8, max_depth=2, n_jobs=1, early_stopping_rounds=2
        )
        model.fit(X, y, eval_set=[(X, y)], verbose=False)
        model.save_model(root / filename)
    pd.DataFrame({"feature": X.columns}).to_csv(
        root / "feature_columns.csv", index=False
    )
    files = {
        "decision_policy.json": {"rule": "model_b", "threshold": 0.01},
        "thresholds.json": {"OP_THRESHOLD_A": 0.99, "OP_THRESHOLD_B": 0.01},
        "reference.json": {
            "amount_paid_log": {"interior_edges": [5.0], "fractions": [0.5, 0.5]}
        },
        "evaluation.json": {"development_only": True},
        "splits.json": {},
    }
    for name, data in files.items():
        (root / name).write_text(json.dumps(data))
    manifest = {
        "schema_version": 2,
        "model_version": "test-only",
        "feature_version": FEATURE_VERSION,
        "feature_source_sha256": source_hash(),
        "feature_count": len(X.columns),
        "development_only": True,
        "files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


@pytest.fixture
def client(bundle, tmp_path, monkeypatch):
    import anyio.to_thread

    import aml_app.main as api
    from aml_app.cache import BatchCache
    from aml_app.service import AMLScorer

    async def run_sync_direct(func, *args, **kwargs):
        return func(*args)

    # Unit tests inject lifespan-managed dependencies directly. This keeps
    # endpoint tests independent of the Python 3.14 AnyIO worker-thread bug.
    monkeypatch.setattr(anyio.to_thread, "run_sync", run_sync_direct)
    monkeypatch.setattr(api, "scorer", AMLScorer(bundle, allow_development=True))
    monkeypatch.setattr(api, "batch_cache", BatchCache(tmp_path / "cache.sqlite"))
    monkeypatch.delenv("AML_API_KEY", raising=False)
    monkeypatch.delenv("DEPLOYMENT_ENVIRONMENT", raising=False)
    yield ASGITestClient(api.app)
