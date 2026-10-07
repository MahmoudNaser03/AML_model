FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY aml_app ./aml_app
COPY dashboard.py ./
COPY serve_app.py ./
COPY replica_monitor.py ./

# The production model artifacts are mounted by Compose at runtime. Keeping
# them out of the image makes model updates possible without rebuilding it.
RUN useradd --create-home --uid 10001 appuser && \
    mkdir -p /app/data && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000 8080 8265 8501 8502

CMD ["uvicorn", "aml_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
