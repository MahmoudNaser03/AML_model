"""Ray Serve entry point for the AML FastAPI application.

Run locally with `python serve_app.py`, then use the API at port 8000 and the
Ray dashboard at port 8265.
"""
from __future__ import annotations

import os
import tempfile

# Set before importing prometheus_client. Every worker shares this run-specific
# directory; a restart cannot resurrect counters from a previous deployment.
if __name__ == "__main__":
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = tempfile.mkdtemp(prefix="aml-prometheus-")

import ray
from ray import serve

from aml_app.main import app as fastapi_app

if __name__ == "__main__":
    from prometheus_client import multiprocess
    # The driver imports metric definitions for serialization but serves no requests.
    multiprocess.mark_process_dead(os.getpid())

REQUEST_MONITOR_NAME = "aml-request-monitor"
REQUEST_MONITOR_NAMESPACE = "aml"
RAY_METRICS_EXPORT_PORT = int(os.getenv("RAY_METRICS_EXPORT_PORT", "8080"))
RAY_METRICS_REPORT_INTERVAL_MS = int(os.getenv("RAY_METRICS_REPORT_INTERVAL_MS", "1000"))


@ray.remote(num_cpus=0)
class RequestMonitor:
    """Cluster-wide counter used by the lightweight Serve monitor."""

    def __init__(self) -> None:
        self.received_requests = 0
        self.completed_requests = 0
        self._active_request_ids: set[str] = set()

    def record_received(self, request_id: str) -> None:
        """Record one scoring request when it reaches an API replica."""
        if request_id in self._active_request_ids:
            return
        self.received_requests += 1
        self._active_request_ids.add(request_id)

    def record_completed(self, request_id: str) -> None:
        """Record completion only for a request seen in this monitor run."""
        if request_id not in self._active_request_ids:
            # A reset clears active IDs.  Ignore a late completion from the
            # preceding chart rather than corrupting the new baseline.
            return
        self.completed_requests += 1
        self._active_request_ids.remove(request_id)

    def snapshot(self) -> dict[str, int]:
        return {
            "received_requests": self.received_requests,
            "completed_requests": self.completed_requests,
            "in_flight_requests": len(self._active_request_ids),
        }

    def reset(self) -> None:
        self.received_requests = 0
        self.completed_requests = 0
        self._active_request_ids.clear()


def start_request_monitor() -> None:
    """Create (or reuse) one counter shared by every Serve replica."""
    RequestMonitor.options(
        name=REQUEST_MONITOR_NAME,
        namespace=REQUEST_MONITOR_NAMESPACE,
        # The driver does not retain the return value.  A detached actor keeps
        # this cluster-wide counter alive after this setup function returns.
        lifetime="detached",
        get_if_exists=True,
    ).remote()


@serve.deployment(
    name="AMLAPI",
    # Each offline batch is self-contained, so replicas do not share mutable
    # feature history and can safely scale independently.
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 10,
        "target_ongoing_requests": 1,
        "upscale_delay_s": 1,
        "downscale_delay_s": 5,
    },
    # A replica accepts up to ten in-flight requests before Serve applies
    # backpressure.  The autoscaler uses the lower target above to scale early.
    max_ongoing_requests=2,
    max_queued_requests=100,
    ray_actor_options={"num_cpus": 0.5},
)
@serve.ingress(fastapi_app)
class AMLServe:
    """Autoscaled Ray Serve wrapper around the existing FastAPI routes."""


def main() -> None:
    ray_address = os.getenv("RAY_ADDRESS")
    # Bind the local dashboard beyond loopback so the separate monitor
    # container can read the live Serve state.
    if ray_address:
        ray.init(address=ray_address)
    else:
        # Ray's node-level endpoint contains the Serve metrics that identify
        # every replica.  The FastAPI /metrics endpoint is intentionally kept
        # separate for model/business metrics.
        ray.init(
            object_store_memory=512 * 1024 * 1024,
            runtime_env={"env_vars": {"PROMETHEUS_MULTIPROC_DIR": os.environ["PROMETHEUS_MULTIPROC_DIR"]}},
            dashboard_host=os.getenv("RAY_DASHBOARD_HOST", "0.0.0.0"),
            _metrics_export_port=RAY_METRICS_EXPORT_PORT,
            _system_config={
                "metrics_report_interval_ms": RAY_METRICS_REPORT_INTERVAL_MS,
            },
        )
    start_request_monitor()
    serve.start(http_options={"host": "0.0.0.0", "port": 8000})
    serve.run(AMLServe.bind(), name="aml", route_prefix="/")
    # Keep this driver alive: it owns the local Ray head process in Docker.
    import time
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
