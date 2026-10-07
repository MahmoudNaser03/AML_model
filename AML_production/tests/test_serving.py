import shutil

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from aml_app.artifacts import ArtifactError, validate_bundle
from aml_app.cache import BatchCache
from aml_app.service import AMLScorer


def test_development_artifacts_rejected_by_default(bundle):
    with pytest.raises(ArtifactError, match="Development"):
        AMLScorer(bundle)


def test_tampered_artifact_fails_checksum(bundle, tmp_path):
    shutil.copytree(bundle, tmp_path / "changed")
    (tmp_path / "changed" / "thresholds.json").write_text("{}")
    with pytest.raises(ArtifactError, match="checksum"):
        validate_bundle(tmp_path / "changed", allow_development=True)


def test_explanation_uses_policy_model_and_early_stopping_trees(bundle, sample):
    scorer = AMLScorer(bundle, allow_development=True)
    r, X = scorer.score_with_matrix(pd.DataFrame([sample]))
    result = scorer.explain_decision(X, 0, r.iloc[0])
    assert result["explanation_model"] == "Model B"
    booster = scorer.b.get_booster()
    contrib = booster.predict(
        xgb.DMatrix(X),
        pred_contribs=True,
        iteration_range=(0, scorer.b.best_iteration + 1),
    )[0]
    predicted = 1 / (1 + np.exp(-contrib.sum()))
    assert predicted == pytest.approx(float(r.model_b_score.iloc[0]), abs=1e-6)
    assert sum(item["impact"] for item in result["explanation"]) == pytest.approx(
        float(contrib[:-1].sum()), abs=1e-6
    )


def test_cache_shared_across_instances_and_expiry(tmp_path):
    a = BatchCache(tmp_path / "cache.sqlite", ttl=100, max_batches=1)
    b = BatchCache(tmp_path / "cache.sqlite", ttl=100, max_batches=1)
    X = pd.DataFrame({"feature": [1.0]}, index=[7])
    decisions = pd.DataFrame({"alert": [True]}, index=[7])
    a.put("first", X, decisions, "v1")
    assert b.get("first", "v1")[0].loc[7, "feature"] == 1
    assert b.get("first", "v2") is None
    b.put("second", X, decisions, "v1")
    assert a.get("first", "v1") is None
    b.ttl = -1
    assert b.get("second", "v1") is None


def test_api_single_csv_and_cross_instance_explanation(client, sample, monkeypatch):
    assert client.get("/health").status_code == 200
    single = client.post("/predict", json=sample)
    assert single.status_code == 200
    assert single.json()["prediction"]["from_account"] == "0001"
    assert single.json()["explanation_model"] == "Model B"
    data = pd.DataFrame([sample]).to_csv(index=False)
    response = client.post("/predict/csv", files={"file": ("test.csv", data)})
    assert response.status_code == 200
    payload = response.json()
    assert payload["records"][0]["risk_score"] == pytest.approx(
        single.json()["prediction"]["risk_score"]
    )
    import aml_app.main as api

    monkeypatch.setattr(api, "batch_cache", BatchCache(api.batch_cache.path))
    explain = client.get(f"/predict/csv/{payload['batch_id']}/explain/0")
    assert explain.status_code == 200
    assert explain.json()["explanation_model"] == "Model B"


def test_authentication(client, sample, monkeypatch):
    monkeypatch.setenv("AML_API_KEY", "local-test-key")
    assert client.post("/predict", json=sample).status_code == 401
    assert (
        client.post(
            "/predict", json=sample, headers={"x-api-key": "local-test-key"}
        ).status_code
        == 200
    )
    assert client.get("/health").status_code == 200


def test_readiness_missing_models_is_503(client, sample, monkeypatch):
    import aml_app.main as api

    monkeypatch.setattr(api, "scorer", None)
    assert client.get("/live").status_code == 200
    assert client.get("/health").status_code == 503
    assert client.post("/predict", json=sample).status_code == 503
    assert (
        client.post("/predict/csv", files={"file": ("test.csv", "x")}).status_code
        == 503
    )


def test_errors_do_not_leak_input_and_invalid_numbers_rejected(client, sample):
    sample["amount_paid"] = "sensitive-invalid-amount"
    response = client.post("/predict", json=sample)
    assert response.status_code == 422
    assert "sensitive-invalid-amount" not in response.text
    response = client.post(
        "/predict/csv",
        files={"file": ("test.csv", pd.DataFrame([sample]).to_csv(index=False))},
    )
    assert response.status_code == 400
    assert "sensitive-invalid-amount" not in response.text


def test_csv_limits_and_metrics_cardinality(client, sample, monkeypatch):
    import aml_app.main as api

    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 4)
    assert (
        client.post(
            "/predict/csv", files={"file": ("test.csv", "too long")}
        ).status_code
        == 413
    )
    client.get("/random-sensitive-id-234")
    text = client.get("/metrics").text
    assert "aml_scoring_failures_total" in text
    assert "random-sensitive-id-234" not in text


def test_production_mode_requires_auth(client, monkeypatch):
    monkeypatch.setenv("DEPLOYMENT_ENVIRONMENT", "production")
    assert client.get("/health").status_code == 503
    assert client.get("/schema").status_code == 503
