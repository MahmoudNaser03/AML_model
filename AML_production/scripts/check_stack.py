"""Exercise the live local stack and save a concise verification report."""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Support both invocation styles:
#   python -m scripts.check_stack
#   python scripts/check_stack.py
# Direct file execution starts with ``scripts/`` on sys.path, so add the
# repository root before importing the application package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import requests

from aml_app.pipeline import read_transactions
from aml_app.service import AMLScorer


def run(base, data):
    headers = {"x-api-key": os.getenv("AML_API_KEY", "")}
    report = {}

    def get(path):
        response = requests.get(base + path, headers=headers, timeout=15)
        response.raise_for_status()
        return response

    report["health"] = get("/health").json()
    report["schema"] = get("/schema").json()
    report["policy"] = get("/model").json()["decision_policy"]
    before = get("/monitor/metrics").json()
    raw = read_transactions(data, nrows=5000)
    body = raw.to_csv(index=False).encode()
    started = time.perf_counter()
    response = requests.post(
        base + "/predict/csv",
        files={"file": ("smoke.csv", body)},
        headers=headers,
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    report["batch"] = payload["summary"]
    report["batch_seconds"] = time.perf_counter() - started
    assert len(payload["records"]) == 5000
    assert [r["record_id"] for r in payload["records"]] == list(range(5000))
    local = AMLScorer("production_artifacts").score(raw)
    for key in ["risk_score", "model_a_score", "model_b_score"]:
        np.testing.assert_allclose(
            [r[key] for r in payload["records"]], local[key], rtol=1e-5, atol=1e-7
        )
    report["local_docker_prediction_parity"] = True
    alerted = [r for r in payload["records"] if r["alert"]]
    if not alerted:
        raise AssertionError(
            "Smoke batch has no alerts; choose a larger representative batch for explanation checks."
        )
    path = f"/predict/csv/{payload['batch_id']}/explain/{alerted[0]['record_id']}"
    with ThreadPoolExecutor(max_workers=12) as pool:
        explanations = list(pool.map(lambda _: get(path).status_code, range(30)))
    report["concurrent_explanations"] = {
        "requests": len(explanations),
        "all_ok": all(s == 200 for s in explanations),
    }
    single = {k: payload["records"][0][k] for k in report["schema"]["required_columns"]}
    response = requests.post(
        base + "/predict", json=single, headers=headers, timeout=30
    )
    response.raise_for_status()
    assert response.json()["explanation_model"]
    invalid = requests.post(
        base + "/predict/csv",
        files={"file": ("bad.csv", "bad,columns\n1,2")},
        headers=headers,
        timeout=15,
    )
    assert invalid.status_code == 400
    report["invalid_csv_status"] = invalid.status_code
    after = get("/monitor/metrics").json()
    report["monitor_delta"] = {
        key: after[key] - before[key]
        for key in ["received_requests", "completed_requests"]
    }
    # Scrapes through any replica must expose one shared, monotone metrics view.
    metrics = get("/metrics").text
    assert "aml_scoring_transactions_total" in metrics
    report["metrics_available"] = True
    for port, path, name in [
        (8501, "/_stcore/health", "dashboard"),
        (8502, "/_stcore/health", "replica_monitor"),
        (9090, "/-/ready", "prometheus"),
        (3000, "/api/health", "grafana"),
    ]:
        response = requests.get(f"http://127.0.0.1:{port}{path}", timeout=15)
        report[name] = response.status_code
        response.raise_for_status()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--data", default="../aml-detection-project/data/raw/HI-Small_Trans.csv"
    )
    parser.add_argument("--output", default="artifacts/stack_verification.json")
    args = parser.parse_args()
    report = run(args.url, args.data)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
