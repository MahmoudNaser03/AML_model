"""Streamlit user interface. Run with: streamlit run dashboard.py"""
from __future__ import annotations

import io
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from aml_app.evaluation import (
    evaluate_alerts,
    evaluate_score_threshold,
    suggested_label_column,
)
from aml_app.history import load_history as read_saved_history
from aml_app.history import save_history

API_URL = os.getenv("AML_API_URL", "http://localhost:8000")
HISTORY_PATH = Path(os.getenv("AML_HISTORY_PATH", "data/dashboard_history.sqlite"))
PERSIST_HISTORY = os.getenv("AML_PERSIST_HISTORY", "false").lower() == "true"
REQUIRED = ["timestamp", "from_bank", "from_account", "to_bank", "to_account", "amount_received", "receiving_currency", "amount_paid", "payment_currency", "payment_format"]
st.set_page_config(page_title="AML Risk Intelligence", page_icon="🛡️", layout="wide")
st.title("🛡️ AML Risk Intelligence")
st.caption("Transaction investigation using a validated model policy and causal behavior signals.")
st.caption("Model scores rank transactions; they are not calibrated probabilities of laundering.")

# Plain-language names and explanations for the model's engineered signals.
SIGNAL_LABELS = {
    "amount_received": "Money received", "amount_paid": "Money sent", "flow_gap": "Difference between sent and received amounts",
    "flow_abs_diff": "Size of the amount difference", "flow_relative_diff": "Relative amount difference",
    "received_paid_ratio": "Received-to-sent amount ratio", "paid_received_ratio": "Sent-to-received amount ratio",
    "cross_currency": "Currency conversion", "same_bank": "Transfer within the same bank",
    "sender_prev_count": "Sender's previous transactions", "receiver_prev_count": "Receiver's previous transactions",
    "pair_prev_count": "Previous transfers between these accounts", "reverse_pair_prev_count": "Previous transfers in the reverse direction",
    "sender_delta_sec": "Time since sender's previous transfer", "receiver_delta_sec": "Time since receiver's previous transfer",
    "pair_delta_sec": "Time since the last transfer between these accounts", "reciprocal_score": "Recent two-way activity",
    "sender_is_new": "New sender", "receiver_is_new": "New receiver", "pair_is_new": "New account relationship",
    "reverse_pair_is_new": "New reverse account relationship", "corridor_is_new": "New bank-and-currency route",
    "sender_format_novelty": "New payment method for sender", "sender_currency_novelty": "New sending currency for sender",
    "receiver_currency_novelty": "New receiving currency for receiver", "sender_unique_out": "Number of different recipients used by sender",
    "receiver_unique_in": "Number of different senders used by receiver", "sender_diversity": "Sender's recipient diversity",
    "receiver_diversity": "Receiver's sender diversity", "new_sender_new_pair": "First transfer from a new sender",
    "new_pair_existing_sender": "New recipient for an established sender", "new_pair_cross_currency": "New relationship with currency conversion",
    "reverse_received_ratio": "Amount compared with reverse transfer", "reverse_paid_ratio": "Sent amount compared with reverse transfer",
    "reverse_amount_gap": "Difference from reverse-transfer amount", "sender_network_expansion": "Sender's network expansion",
    "receiver_network_expansion": "Receiver's network expansion", "cluster_distance": "Unusualness compared with normal transaction patterns",
    "struct_cycle_score": "Circular-money-flow pattern", "struct_stack_score": "Layered-transfer pattern",
    "struct_scatter_gather_score": "Scatter-and-gather pattern", "struct_gather_scatter_score": "Gather-and-scatter pattern",
    "structural_max_score": "Strongest network-pattern signal", "structural_mean_score": "Overall network-pattern signal",
}

def friendly_signal_name(feature):
    return SIGNAL_LABELS.get(feature, feature.replace("_", " ").replace("prev", "previous").replace("sec", "seconds").title())

def friendly_observation(feature, value):
    if feature in {"cross_currency", "same_bank", "sender_is_new", "receiver_is_new", "pair_is_new", "corridor_is_new", "reciprocal_score"}:
        return "Yes" if value >= .5 else "No"
    if "delta_sec" in feature:
        return f"{value / 60:.1f} minutes"
    if "amount" in feature or feature in {"flow_gap", "flow_abs_diff", "reverse_amount_gap"}:
        return f"{value:,.2f}"
    if "ratio" in feature or "share" in feature or "diversity" in feature:
        return f"{value:.2f}"
    return f"{value:,.2f}"

def friendly_reason(feature):
    f = feature.lower()
    if "new" in f or "novelty" in f: return "New behavior can warrant additional review."
    if "amount" in f or "ratio" in f or "flow" in f: return "Unusual amounts or amount relationships can be relevant."
    if "delta" in f or "velocity" in f or "activity" in f or "burst" in f: return "The timing or pace of activity can be relevant."
    if "reverse" in f or "reciprocal" in f or "cycle" in f: return "Back-and-forth movement can be relevant."
    if "network" in f or "diversity" in f or "fan" in f or "struct" in f: return "The transaction network pattern can be relevant."
    if "currency" in f or "corridor" in f: return "An unusual currency or bank route can be relevant."
    return "This differs from the behavioral patterns learned by the model."

def explainability_view(explanation):
    readable = explanation.copy()
    readable["Signal"] = readable["feature"].map(friendly_signal_name)
    if "model" in readable:
        readable["Signal"] = "Model " + readable["model"] + ": " + readable["Signal"]
    readable["What we observed"] = [friendly_observation(f, v) for f, v in zip(readable["feature"], readable["value"])]
    readable["Why it matters"] = readable["feature"].map(friendly_reason)
    readable["Effect on risk"] = readable["impact"].map(lambda x: "Raises risk" if x > 0 else "Reduces risk")
    readable["Strength"] = readable["impact"].abs().map(lambda x: "Strong" if x >= 1 else "Moderate" if x >= .25 else "Small")
    return readable

def api(path, **kwargs):
    try:
        r = requests.request(kwargs.pop("method", "GET"), f"{API_URL}{path}", timeout=120,
                             headers={"x-api-key": os.getenv("AML_API_KEY", "")}, **kwargs)
        if not r.ok:
            detail = r.json().get("detail", r.text) if r.content else r.reason
            st.error(f"API error ({r.status_code}): {detail}")
            return None
        return r.json()
    except requests.RequestException as exc:
        st.error(f"API unavailable: {exc}"); return None

def load_history():
    return read_saved_history(HISTORY_PATH) if PERSIST_HISTORY else pd.DataFrame()

def save_batch_to_history(batch, batch_id):
    history = batch.copy()
    history["upload_batch"] = batch_id
    history["uploaded_at"] = datetime.now(timezone.utc).isoformat()
    if PERSIST_HISTORY:
        save_history(HISTORY_PATH, history, batch_id)
    return pd.concat([st.session_state.get("scored_history", pd.DataFrame()), history], ignore_index=True).tail(100000)

def format_rate(value):
    return "N/A" if value is None else f"{value:.1%}"

def show_batch_evaluation(evaluation, deployed_threshold):
    st.subheader("Prediction vs actual labels")
    st.caption(
        f"Preview threshold: {evaluation['threshold']:.4f} · Deployed threshold: {deployed_threshold:.4f}. "
        "Lowering the preview threshold usually raises recall and also raises false positives. "
        "This control does not change the deployed policy."
    )
    negatives = evaluation["rows"] - evaluation["actual_positives"]
    if negatives == 0:
        st.warning(
            "This evaluation batch has no negative labels. False positives, specificity, and the real "
            "precision trade-off cannot be measured; use a representative file containing both classes."
        )
    elif evaluation["actual_positives"] == 0:
        st.warning("This evaluation batch has no positive labels, so recall cannot be measured.")
    counts = st.columns(4)
    counts[0].metric("True positives (TP)", f"{evaluation['tp']:,}")
    counts[1].metric("False positives (FP)", f"{evaluation['fp']:,}")
    counts[2].metric("False negatives (FN)", f"{evaluation['fn']:,}")
    counts[3].metric("True negatives (TN)", f"{evaluation['tn']:,}")
    rates = st.columns(5)
    rates[0].metric("Precision", format_rate(evaluation["precision"]))
    rates[1].metric("Recall", format_rate(evaluation["recall"]))
    rates[2].metric("Specificity", format_rate(evaluation["specificity"]))
    rates[3].metric("F1 score", format_rate(evaluation["f1"]))
    rates[4].metric("Accuracy", format_rate(evaluation["accuracy"]))
    matrix = pd.DataFrame(
        [[evaluation["tp"], evaluation["fn"]], [evaluation["fp"], evaluation["tn"]]],
        index=["Actual positive", "Actual negative"],
        columns=["Predicted alert", "Predicted no alert"],
    )
    st.caption("Positive means the actual label is 1. An alert is the deployed policy's binary prediction.")
    st.dataframe(matrix, width="stretch")

if "scored_history" not in st.session_state:
    st.session_state["scored_history"] = load_history()

with st.sidebar:
    st.header("Service")
    st.code(API_URL)
    health = api("/health")
    st.write("Status:", health.get("status", "unknown") if health else "unavailable")
    if health: st.caption("Model version: " + str(health.get("model_version", "unknown")))
    st.caption("History persistence: " + ("enabled, 30-day retention" if PERSIST_HISTORY else "session only"))
    st.divider()
    st.caption("CSV columns")
    st.code("\n".join(REQUIRED), language=None)

score_tab, batch_tab, insights_tab, about_tab = st.tabs(["Single transaction", "CSV batch scoring", "Data intelligence", "About the model"])

with score_tab:
    st.subheader("Score one transaction")
    st.warning("A single transaction has no preceding account history, so history-derived signals start cold. Prefer batch scoring for investigation-grade results.")
    with st.form("single"):
        c1,c2,c3=st.columns(3)
        timestamp=c1.text_input("Timestamp", "2026-01-01T12:00:00")
        from_bank=c1.text_input("From bank", "Bank A"); from_account=c1.text_input("From account", "ACC-001")
        to_bank=c2.text_input("To bank", "Bank B"); to_account=c2.text_input("To account", "ACC-002")
        amount_paid=c2.number_input("Amount paid", min_value=0.0, value=1000.0)
        amount_received=c3.number_input("Amount received", min_value=0.0, value=1000.0)
        payment_currency=c3.text_input("Payment currency", "USD"); receiving_currency=c3.text_input("Receiving currency", "USD")
        payment_format=c3.text_input("Payment format", "ACH")
        submit=st.form_submit_button("Assess risk", type="primary")
    if submit:
        payload=locals().copy(); payload={k:payload[k] for k in REQUIRED}
        response=api("/predict", method="POST", json=payload)
        if response:
            pred=response["prediction"]; a,b,c=st.columns(3)
            a.metric("Model score", f"{pred['risk_score']:.1%}"); b.metric("Risk level", pred["risk_level"]); c.metric("Alert", "YES" if pred["alert"] else "No")
            st.caption(f"Alert threshold: {pred['decision_threshold']:.4f}")
            st.subheader("Why this result")
            st.caption(f"Plain-language signals from {response['explanation_model']}, used by the deployed decision policy.")
            expl = explainability_view(pd.DataFrame(response["explanation"]))
            st.dataframe(expl[["Signal", "What we observed", "Effect on risk", "Strength", "Why it matters"]], hide_index=True, width="stretch")

with batch_tab:
    st.subheader("Score a transaction CSV")
    file=st.file_uploader("Upload CSV", type="csv")
    compare_labels = st.checkbox(
        "Compare alerts with actual labels",
        disabled=file is None,
        help="Uses a binary label column only for evaluation; the API excludes it from model features.",
    )
    label_column = None
    upload_bytes = file.getvalue() if file else None
    if file and compare_labels:
        try:
            csv_columns = pd.read_csv(io.BytesIO(upload_bytes), nrows=0).columns.tolist()
            suggested = suggested_label_column(csv_columns)
            label_column = st.selectbox(
                "Actual label column",
                csv_columns,
                index=csv_columns.index(suggested) if suggested else None,
                placeholder="Choose the binary ground-truth column",
            )
            st.caption("Accepted labels: 0/1, true/false, yes/no, positive/negative, or laundering/not laundering.")
        except (pd.errors.ParserError, UnicodeError, ValueError):
            st.error("The CSV header could not be read. Check the file format.")
    if file and st.button("Score batch", type="primary", disabled=compare_labels and label_column is None):
        labels = None
        if compare_labels:
            try:
                labels = pd.read_csv(
                    io.BytesIO(upload_bytes), usecols=[label_column], keep_default_na=False
                )[label_column]
            except (pd.errors.ParserError, UnicodeError, ValueError, KeyError):
                st.error("The selected label column could not be read from the CSV.")
                labels = None
        response=api("/predict/csv", method="POST", files={"file":(file.name, upload_bytes, "text/csv")}) if not compare_labels or labels is not None else None
        if response:
            result=pd.DataFrame(response["records"])
            st.session_state["scored_history"] = save_batch_to_history(result, response["batch_id"])
            evaluation = None
            if compare_labels:
                try:
                    result, evaluation = evaluate_alerts(result, labels)
                except ValueError as exc:
                    st.error(f"Ground-truth comparison failed: {exc}")
            st.session_state["scored"] = result
            st.session_state["batch_evaluation"] = evaluation
            st.session_state["batch_id"] = response["batch_id"]
            st.session_state["batch_explanations"] = {}
            drift = response.get("drift", {})
            if drift.get("status") == "computed":
                st.caption(f"Distribution check: {drift['features_over_0_2']} features differ from training (PSI > 0.2). Batch history length also affects this comparison.")
            st.session_state.pop("selected_batch_row", None)
            s=response["summary"]; cols=st.columns(4); cols[0].metric("Transactions", f"{s['transactions']:,}"); cols[1].metric("Alerts", f"{s['alerts']:,}"); cols[2].metric("Alert rate", f"{s['alert_rate']:.1%}"); cols[3].metric("Maximum risk", f"{s['max_risk']:.1%}")
    result = st.session_state.get("scored")
    if result is not None:
        evaluation = st.session_state.get("batch_evaluation")
        if evaluation:
            deployed_threshold = float(result["decision_threshold"].iloc[0])
            threshold_ceiling = max(0.05, deployed_threshold)
            threshold_step = 0.0001 if threshold_ceiling <= 0.1 else 0.001
            preview_threshold = st.slider(
                "Evaluation threshold",
                min_value=0.0,
                max_value=threshold_ceiling,
                value=deployed_threshold,
                step=threshold_step,
                format="%.5f",
                key=f"evaluation_threshold_{st.session_state['batch_id']}",
                help="Zoomed into the model's low-score operating range. Move left to favor recall over precision.",
            )
            source_labels = result.sort_values("record_id")["actual_label"].reset_index(drop=True)
            result, evaluation = evaluate_score_threshold(result, source_labels, preview_threshold)
            show_batch_evaluation(evaluation, deployed_threshold)
        st.caption("Alerted transactions are red. Amber rows are within 10% of the deployed policy's alert threshold but are not alerts. Full explanations are available only for alerts.")
        display = result.sort_values("risk_score", ascending=False).reset_index(drop=True)
        table_columns = display.drop(columns=["record_id", "near_alert", "risk_level", "explanation"], errors="ignore").columns
        rows_per_page = max(1, min(5000, 200000 // max(len(table_columns), 1)))
        page_count = max(1, (len(display) + rows_per_page - 1) // rows_per_page)
        page = st.selectbox(
            "Results page",
            range(1, page_count + 1),
            format_func=lambda value: f"Page {value} of {page_count}",
            key=f"batch_results_page_{st.session_state['batch_id']}",
        )
        page_start = (page - 1) * rows_per_page
        page_end = min(page_start + rows_per_page, len(display))
        page_display = display.iloc[page_start:page_end].reset_index(drop=True)
        table = page_display.loc[:, table_columns]
        st.caption(
            f"Showing risk-ranked transactions {page_start + 1:,}–{page_end:,} of {len(display):,}. "
            "The download includes the complete batch."
        )
        def risk_row_style(row):
            risk_state = page_display.loc[row.name]
            if bool(risk_state["alert"]):
                return ["background-color: #ffb3b3; color: #650000; font-weight: 600" for _ in row]
            if bool(risk_state.get("near_alert", False)):
                return ["background-color: #ffe58f; color: #5f4700" for _ in row]
            return ["" for _ in row]
        event = st.dataframe(table.style.apply(risk_row_style, axis=1), width="stretch", hide_index=True, on_select="rerun", selection_mode="single-row", key=f"batch_results_{st.session_state['batch_id']}_{page}")
        selected_rows = event.selection.rows if event else []
        if selected_rows:
            selected = page_display.iloc[selected_rows[0]]
            st.session_state["selected_batch_row"] = selected.to_dict()
        selected = st.session_state.get("selected_batch_row")
        if selected:
            risk = float(selected["risk_score"])
            st.divider()
            st.subheader("Transaction investigation")
            m1, m2, m3 = st.columns(3)
            m1.metric("Model score", f"{risk:.1%}")
            m2.metric("Risk level", str(selected["risk_level"]))
            m3.metric("Decision", "ALERT" if selected["alert"] else "No alert")
            if selected["alert"]:
                record_id = int(selected["record_id"])
                cached = st.session_state["batch_explanations"].get(record_id)
                if cached is None:
                    with st.spinner("Building the full explanation for this alert..."):
                        cached = api(f"/predict/csv/{st.session_state['batch_id']}/explain/{record_id}")
                    if cached:
                        st.session_state["batch_explanations"][record_id] = cached
                if cached:
                    st.caption(f"Plain-language explanation based on {cached['explanation_model']}. The signals below help explain why the transaction was alerted; they support, not replace, analyst judgement.")
                    explanation = pd.DataFrame(cached["explanation"])
                    readable = explainability_view(explanation)
                    risk_signals = readable[readable["impact"] > 0].nlargest(8, "impact")
                    mitigators = readable[readable["impact"] < 0].nsmallest(5, "impact")
                    st.subheader("Main reasons for this alert")
                    st.dataframe(risk_signals[["Signal", "What we observed", "Why it matters", "Strength"]], width="stretch", hide_index=True)
                    if not mitigators.empty:
                        with st.expander("Factors that reduced the risk score"):
                            st.dataframe(mitigators[["Signal", "What we observed", "Why it matters", "Strength"]], width="stretch", hide_index=True)
                    st.plotly_chart(px.bar(risk_signals.sort_values("impact"), x="impact", y="Signal", orientation="h", title="Most important risk signals"), width="stretch")
                    with st.expander("Show every model signal"):
                        st.dataframe(readable[["Signal", "What we observed", "Effect on risk", "Strength", "Why it matters"]], width="stretch", hide_index=True)
                else:
                    st.warning("The explanation could not be loaded. Score the batch again and retry.")
            else:
                st.info("This transaction did not meet the deployed policy's alert threshold. Select a red row for a full suspicious-activity explanation.")
        st.download_button("Download complete scored CSV", result.to_csv(index=False).encode(), "aml_scored_transactions.csv", "text/csv")

with insights_tab:
    st.subheader("Risk and transaction patterns")
    data=st.session_state.get("scored_history", pd.DataFrame())
    if data is None or data.empty:
        st.info("Score a CSV first; saved uploads will become an interactive investigation dataset here.")
    else:
        st.caption(f"Charts include all saved uploads ({len(data):,} transactions), not just the latest batch.")
        left,right=st.columns(2)
        with left: st.plotly_chart(px.histogram(data, x="risk_score", color="risk_level", nbins=30, title="Risk-score distribution"), width="stretch")
        with right: st.plotly_chart(px.scatter(data, x="amount_paid", y="amount_received", color="risk_level", size="risk_score", hover_data=["from_bank","to_bank"], title="Payment vs receipt amount"), width="stretch")
        c1,c2=st.columns(2)
        with c1: st.plotly_chart(px.bar(data.groupby("payment_currency", as_index=False).agg(transactions=("alert","size"), alerts=("alert","sum")).sort_values("alerts",ascending=False), x="payment_currency", y="alerts", hover_data="transactions", title="Alerts by payment currency"), width="stretch")
        with c2: st.plotly_chart(px.bar(data.groupby("payment_format", as_index=False).agg(transactions=("alert","size"), alerts=("alert","sum")).sort_values("alerts",ascending=False), x="payment_format", y="alerts", hover_data="transactions", title="Alerts by payment format"), width="stretch")
        dated = data.copy()
        if "timestamp" in dated.columns:
            dated["timestamp"] = pd.to_datetime(dated["timestamp"], errors="coerce", utc=True)
            dated["date"] = dated["timestamp"].dt.date
            timeline = dated.dropna(subset=["date"]).groupby("date", as_index=False).agg(transactions=("alert", "size"), alerts=("alert", "sum"), average_risk=("risk_score", "mean"))
        else:
            timeline = pd.DataFrame(columns=["date", "alerts", "transactions", "average_risk"])
        banks = pd.concat([data[["from_bank", "alert", "risk_score"]].rename(columns={"from_bank":"bank"}), data[["to_bank", "alert", "risk_score"]].rename(columns={"to_bank":"bank"})]).groupby("bank", as_index=False).agg(transactions=("alert","size"), alerts=("alert","sum"), average_risk=("risk_score","mean")).sort_values("alerts", ascending=False).head(15)
        c3,c4=st.columns(2)
        with c3:
            if timeline.empty: st.info("Re-score the CSV to load the transaction timeline.")
            else: st.plotly_chart(px.line(timeline, x="date", y=["alerts", "transactions"], markers=True, title="Alert volume over time"), width="stretch")
        with c4: st.plotly_chart(px.bar(banks, x="bank", y="alerts", color="average_risk", hover_data="transactions", title="Top banks by alerted activity"), width="stretch")

with about_tab:
    st.subheader("Deployed model and validation")
    information = api("/model") if health and health.get("status") == "ok" else None
    if information:
        st.write("Release:", information["model_version"])
        st.write("Decision policy:", information["decision_policy"])
        report = information["evaluation"]
        result_keys = [
            key for key in ["baseline_test", "model_a_test", "model_b_test", "test"]
            if key in report
        ]
        st.dataframe(pd.DataFrame({key: report[key] for key in result_keys}).T, width="stretch")
    st.markdown("""Training and serving use the same versioned feature implementation.
Each CSV is independent and features use only earlier rows in that file. Tune,
threshold-selection, and test periods do not overlap. Alert explanations follow
the deployed rule; their contributions are in each model's log-odds units.
These signals support analyst review.""")
