"""OpenTelemetry tracing and trace-correlated logging for the AML service.

Telemetry intentionally contains operational metadata only.  Never add raw
transactions, account identifiers, bank names, CSV filenames, or feature values
as span attributes or log fields.
"""
from __future__ import annotations

import atexit
import logging
import os
import socket

from fastapi import FastAPI
from opentelemetry import _logs, trace
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import DEPLOYMENT_ENVIRONMENT, HOST_NAME, SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode

SERVICE = "aml-model-api"
_configured = False


class TraceContextFilter(logging.Filter):
    """Expose W3C trace context on every standard-library log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        context = trace.get_current_span().get_span_context()
        record.trace_id = f"{context.trace_id:032x}" if context.is_valid else "-"
        record.span_id = f"{context.span_id:016x}" if context.is_valid else "-"
        return True


def configure_observability(app: FastAPI) -> None:
    """Configure tracing, OTLP log export, and FastAPI request instrumentation."""
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=os.getenv("AML_LOG_LEVEL", "INFO").upper(),
            format="%(asctime)s %(levelname)s %(name)s trace_id=%(trace_id)s span_id=%(span_id)s %(message)s",
        )
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s trace_id=%(trace_id)s span_id=%(span_id)s %(message)s"
    )
    for handler in root.handlers:
        handler.addFilter(TraceContextFilter())
        handler.setFormatter(formatter)

    resource = Resource.create(
        {
            SERVICE_NAME: os.getenv("OTEL_SERVICE_NAME", SERVICE),
            SERVICE_VERSION: app.version,
            DEPLOYMENT_ENVIRONMENT: os.getenv("DEPLOYMENT_ENVIRONMENT", "development"),
            HOST_NAME: socket.gethostname(),
            "service.instance.id": f"{socket.gethostname()}-{os.getpid()}",
            "aml.telemetry.data_policy": "operational_metadata_only",
        }
    )
    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))

        logger_provider = LoggerProvider(resource=resource)
        _logs.set_logger_provider(logger_provider)
        logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))
        # Export only application logs. Ray's own very high-volume logs remain
        # in the container log stream instead of overwhelming the trace backend.
        application_logger = logging.getLogger("aml")
        application_logger.addHandler(LoggingHandler(level=logging.INFO, logger_provider=logger_provider))
        atexit.register(tracer_provider.shutdown)
        atexit.register(logger_provider.shutdown)

    # Middleware is attached before startup; exporters are initialized inside
    # each replica after Ray has deserialized the application.


def get_tracer(component: str = "aml") -> trace.Tracer:
    return trace.get_tracer(component)


def record_span_error(exc: BaseException) -> None:
    """Mark the current span failed and retain the exception for investigation."""
    span = trace.get_current_span()
    if span.is_recording():
        # Exception strings and stack locals can contain transaction values.
        span.add_event("exception", {"exception.type": type(exc).__name__})
        span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
