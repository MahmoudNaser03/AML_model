"""Small live Ray Serve monitor. Run with `streamlit run replica_monitor.py`."""
from __future__ import annotations

import logging
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

RAY_DASHBOARD_URL = os.getenv("RAY_DASHBOARD_URL", "http://localhost:8265").rstrip("/")
RAY_SERVE_API_URL = os.getenv("RAY_SERVE_API_URL", "http://localhost:8000").rstrip("/")
MONITOR_HISTORY_PATH = Path(os.getenv("MONITOR_HISTORY_PATH", "serve_monitor_history.csv"))
logger = logging.getLogger("aml.replica_monitor")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

st.set_page_config(page_title="AML Ray Serve monitor", page_icon="📈", layout="wide")
st.title("📈 AML Ray Serve live monitor")
st.caption("The monitor refreshes every 2 seconds and records replica state over time.")
st_autorefresh(interval=2_000, key="serve_monitor_auto_refresh")

controls, reset_control = st.columns([1, 2])
with controls:
    if st.button("Refresh now", type="primary"):
        st.rerun()
with reset_control:
    if st.button("Restart plot", help="Clear the saved timeline and begin a new chart from the next refresh."):
        try:
            MONITOR_HISTORY_PATH.unlink(missing_ok=True)
            response = requests.post(f"{RAY_SERVE_API_URL}/monitor/reset", timeout=3, headers={"x-api-key": os.getenv("AML_API_KEY", "")})
            response.raise_for_status()
            st.success("Plot history and scoring-request counter cleared. This refresh is the new baseline.")
        except OSError as exc:
            st.error(f"Could not clear plot history: {exc}")
        except requests.RequestException as exc:
            st.error(f"Plot history cleared, but the scoring-request counter could not be reset: {exc}")
        st.rerun()


def request_totals() -> dict[str, float] | None:
    """Read cluster-wide received and completed scoring-request counters."""
    try:
        response = requests.get(f"{RAY_SERVE_API_URL}/monitor/metrics", timeout=3, headers={"x-api-key": os.getenv("AML_API_KEY", "")})
        response.raise_for_status()
        payload = response.json()
        return {
            "received": float(payload["received_requests"]),
            "completed": float(payload["completed_requests"]),
            "in_flight": float(payload["in_flight_requests"]),
        }
    except (KeyError, TypeError, ValueError, requests.RequestException) as exc:
        logger.info("monitor_metrics_unavailable url=%s error=%s", RAY_SERVE_API_URL, exc)
        return None


def load_history() -> list[dict[str, object]]:
    if not MONITOR_HISTORY_PATH.exists():
        return []
    try:
        loaded = pd.read_csv(MONITOR_HISTORY_PATH, parse_dates=["checked_at"])
        return loaded.tail(240).to_dict("records")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        logger.warning("monitor_history_unavailable path=%s error=%s", MONITOR_HISTORY_PATH, exc)
        return []


def save_history(history: list[dict[str, object]]) -> None:
    try:
        MONITOR_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(MONITOR_HISTORY_PATH, index=False)
    except OSError as exc:
        logger.warning("monitor_history_save_failed path=%s error=%s", MONITOR_HISTORY_PATH, exc)

try:
    # Ray registers this REST route with a trailing slash.  Without it the
    # dashboard returns 404 instead of the deployed Serve application state.
    response = requests.get(f"{RAY_DASHBOARD_URL}/api/serve/applications/", timeout=5)
    response.raise_for_status()
    payload = response.json()
    logger.info("monitor_refresh dashboard=%s status=%s", RAY_DASHBOARD_URL, response.status_code)
except requests.RequestException as exc:
    logger.warning("monitor_unavailable dashboard=%s error=%s", RAY_DASHBOARD_URL, exc)
    st.error(f"Ray dashboard is unavailable at {RAY_DASHBOARD_URL}: {exc}")
    st.stop()

applications = payload.get("applications", {})
application = applications.get("aml")
if not application:
    st.warning("The `aml` Serve application is not deployed yet. Start `python serve_app.py` first.")
    st.stop()

st.caption(f"Last checked: {datetime.now(timezone.utc).isoformat()}")
st.metric("Application status", application.get("status", "unknown"))
rows = []
for name, deployment in application.get("deployments", {}).items():
    # The Serve REST API returns one entry per replica (``replicas``).  Older
    # API responses exposed a pre-aggregated ``replica_states`` mapping, so
    # retain that as a fallback for compatibility.
    replica_records = deployment.get("replicas", [])
    states = Counter(replica.get("state", "UNKNOWN") for replica in replica_records)
    if not states:
        states = Counter(deployment.get("replica_states", {}))
    rows.append({
        "deployment": name,
        "status": deployment.get("status", "unknown"),
        "running": states.get("RUNNING", 0),
        "starting": states.get("STARTING", 0),
        "stopping": states.get("STOPPING", 0),
        "all replica states": ", ".join(f"{state}: {count}" for state, count in states.items()) or "none",
    })

if rows:
    table = pd.DataFrame(rows)
    st.dataframe(table, hide_index=True, use_container_width=True)

    current_replicas = int(table[["running", "starting"]].sum().sum())
    now = datetime.now(timezone.utc)
    totals = request_totals()
    if totals is not None:
        metrics_left, metrics_middle, metrics_right = st.columns(3)
        metrics_left.metric("Received scoring requests", f"{int(totals['received']):,}")
        metrics_middle.metric("Completed scoring requests", f"{int(totals['completed']):,}")
        metrics_right.metric("In flight", f"{int(totals['in_flight']):,}")
    history = load_history()
    previous = history[-1] if history else None
    received_since_last_sample = None
    completed_since_last_sample = None
    if previous and totals is not None:
        for counter, result_key in (("received", "received_since_last_sample"), ("completed", "completed_since_last_sample")):
            previous_total = previous.get(f"{counter}_total")
            if previous_total is not None and pd.notna(previous_total):
                # A counter can reset when Ray is restarted.  Treat that
                # snapshot as a fresh baseline instead of drawing negatives.
                value = max(0.0, totals[counter] - float(previous_total))
                if result_key == "received_since_last_sample":
                    received_since_last_sample = value
                else:
                    completed_since_last_sample = value
    history.append({
        "checked_at": now,
        "replicas": current_replicas,
        "received_requests": received_since_last_sample,
        "completed_requests": completed_since_last_sample,
        "received_total": totals["received"] if totals else None,
        "completed_total": totals["completed"] if totals else None,
    })
    # Keep the in-browser view useful during a long monitoring session without
    # allowing Streamlit session state to grow indefinitely.
    del history[:-240]
    save_history(history)

    chart_data = pd.DataFrame(history)
    chart = go.Figure()
    chart.add_trace(go.Scatter(
        x=chart_data["checked_at"], y=chart_data["replicas"],
        name="Active replicas", mode="lines+markers", line={"color": "#2563eb", "width": 3},
        hovertemplate="%{x}<br>Active replicas: %{y}<extra></extra>",
    ))
    if "received_requests" in chart_data and chart_data["received_requests"].notna().any():
        chart.add_trace(go.Bar(
            x=chart_data["checked_at"], y=chart_data["received_requests"], name="Received requests",
            yaxis="y2", marker_color="#8b5cf6",
            hovertemplate="%{x}<br>Received scoring requests: %{y}<extra></extra>",
        ))
    if "completed_requests" in chart_data and chart_data["completed_requests"].notna().any():
        chart.add_trace(go.Bar(
            x=chart_data["checked_at"], y=chart_data["completed_requests"], name="Completed requests",
            yaxis="y2", marker_color="#f59e0b",
            hovertemplate="%{x}<br>Completed scoring requests: %{y}<extra></extra>",
        ))
    chart.update_layout(
        title="Replica count and request volume over time",
        xaxis_title="Sample time (UTC)",
        yaxis={"title": "Active replicas", "rangemode": "tozero", "dtick": 1},
        barmode="group",
        yaxis2={"title": "Scoring requests since prior sample", "overlaying": "y", "side": "right", "rangemode": "tozero"},
        legend={"orientation": "h", "y": 1.12},
        margin={"l": 10, "r": 10, "t": 65, "b": 10},
    )
    st.plotly_chart(chart, use_container_width=True)
    if totals is None:
        st.caption(f"Replica history is available, but the request counter could not be read from {RAY_SERVE_API_URL}.")
    else:
        st.caption("Purple bars show scoring requests received by an API replica; amber bars show those requests completed. Both cover /predict and /predict/csv calls since the prior sample. In flight is received minus completed.")
else:
    st.info("Ray has not reported any deployments yet.")

with st.expander("Autoscaling settings"):
    st.code("min replicas: 1\nmax replicas: 10\ntarget ongoing requests / replica: 1\nmax ongoing requests / replica: 2")
    st.write("Requests above the target cause Ray Serve to add replicas. Once the rolling 30-second demand window falls, the 5-second downscale delay applies and Ray returns to one replica.")
