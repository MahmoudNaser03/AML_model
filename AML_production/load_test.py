"""Generate sustained scoring traffic so Ray Serve autoscaling is visible."""
from __future__ import annotations

import argparse
import os
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import requests


TRANSACTION = {
    "timestamp": "2026-01-01T12:00:00Z",
    "from_bank": "Bank A",
    "from_account": "LOAD-001",
    "to_bank": "Bank B",
    "to_account": "LOAD-002",
    "amount_received": 995.0,
    "receiving_currency": "USD",
    "amount_paid": 1000.0,
    "payment_currency": "USD",
    "payment_format": "ACH",
}


def worker(url: str, deadline: float, timeout: float) -> tuple[Counter, list[float]]:
    statuses: Counter = Counter()
    latencies: list[float] = []
    with requests.Session() as session:
        while time.monotonic() < deadline:
            started = time.perf_counter()
            try:
                response = session.post(url, json=TRANSACTION, timeout=timeout, headers={"x-api-key": os.getenv("AML_API_KEY", "")})
                statuses[str(response.status_code)] += 1
            except requests.RequestException as exc:
                statuses[type(exc).__name__] += 1
            latencies.append(time.perf_counter() - started)
    return statuses, latencies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--duration", type=float, default=45.0, help="Load duration in seconds")
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds")
    args = parser.parse_args()
    if args.concurrency < 1 or args.duration <= 0 or args.timeout <= 0:
        parser.error("concurrency, duration, and timeout must be positive")

    print(
        f"Sending traffic to {args.url} for {args.duration:g}s "
        f"with concurrency {args.concurrency}."
    )
    started = time.monotonic()
    deadline = started + args.duration
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(
            pool.map(
                lambda _: worker(args.url, deadline, args.timeout),
                range(args.concurrency),
            )
        )

    statuses: Counter = Counter()
    latencies: list[float] = []
    for worker_statuses, worker_latencies in results:
        statuses.update(worker_statuses)
        latencies.extend(worker_latencies)
    elapsed = time.monotonic() - started
    p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies, default=0.0)
    print(f"Completed {len(latencies):,} attempts in {elapsed:.1f}s ({len(latencies) / elapsed:.1f} req/s).")
    print(f"Outcomes: {dict(statuses)}")
    print(f"Median latency: {statistics.median(latencies):.3f}s; p95: {p95:.3f}s")


if __name__ == "__main__":
    main()
