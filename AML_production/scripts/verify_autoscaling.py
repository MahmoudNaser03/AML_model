"""Verify local Ray scale-up and shared metrics under sustained traffic."""

import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Support both invocation styles:
#   python -m scripts.verify_autoscaling
#   python scripts/verify_autoscaling.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests

from load_test import worker


def main():
    snapshots = []
    deadline = time.monotonic() + 40
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [
            pool.submit(worker, "http://127.0.0.1:8000/predict", deadline, 30)
            for _ in range(16)
        ]
        while time.monotonic() < deadline:
            response = requests.get(
                "http://127.0.0.1:8265/api/serve/applications/", timeout=10
            )
            response.raise_for_status()
            dep = response.json()["applications"]["aml"]["deployments"]["AMLAPI"]
            replicas = dep.get("replicas", [])
            snapshots.append(
                {
                    "phase": "load",
                    "time": time.time(),
                    "running": sum(r["state"] == "RUNNING" for r in replicas),
                    "replica_ids": [
                        r.get("replica_id") for r in replicas if r["state"] == "RUNNING"
                    ],
                }
            )
            time.sleep(2)
        statuses = Counter()
        for future in futures:
            counters, _ = future.result()
            statuses.update(counters)
    cooldown_end = time.monotonic() + 65
    while time.monotonic() < cooldown_end:
        dep = requests.get(
            "http://127.0.0.1:8265/api/serve/applications/", timeout=10
        ).json()["applications"]["aml"]["deployments"]["AMLAPI"]
        count = sum(r["state"] == "RUNNING" for r in dep.get("replicas", []))
        snapshots.append({"phase": "cooldown", "time": time.time(), "running": count})
        if count == 1:
            break
        time.sleep(3)
    report = {
        "statuses": dict(statuses),
        "peak_running_replicas": max(s["running"] for s in snapshots),
        "final_running_replicas": snapshots[-1]["running"],
        "snapshots": snapshots,
    }
    path = Path("artifacts/autoscaling_verification.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "snapshots"}, indent=2))
    assert report["peak_running_replicas"] >= 2, "No scale-up observed"
    assert report["final_running_replicas"] == 1, "No return to minimum observed"
    assert set(statuses) == {"200"}, "Some scoring requests failed"


if __name__ == "__main__":
    main()
