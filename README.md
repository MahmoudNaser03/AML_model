# AML Detection — DS Internship Summer 2026

This repository documents an end-to-end anti-money-laundering (AML) machine-learning project, from exploratory research and leakage-safe model development to a production-oriented scoring service with monitoring.

## Repository structure

| Path | Purpose |
|---|---|
| [`aml-detection-project/`](aml-detection-project/) | Research notebooks, causal feature engineering, trained XGBoost models, figures, and reports |
| [`AML_production/`](AML_production/) | FastAPI and Streamlit application, shared training/serving pipeline, tests, Docker stack, and observability configuration |

The research project produces and evaluates candidate models. The production project rebuilds the feature and scoring pipeline around validated artifacts, API contracts, operational checks, and monitoring.

## Quick start

### Research notebooks

Python 3.11 or newer is recommended.

```bash
cd aml-detection-project
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Run the notebooks in `notebooks/week3/` in numeric order. See the [research README](aml-detection-project/README.md) for the pipeline stages, dataset layout, architecture, and recorded results.

### Production application

```bash
cd AML_production
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
python -m pytest tests -q
```

The service requires a complete promoted model bundle in `AML_production/production_artifacts/`. Model bundles and datasets are intentionally not stored in Git. See the [production README](AML_production/README.md) for local service, dashboard, Docker, retraining, and monitoring instructions.

## Data and generated artifacts

The project uses the synthetic IBM HI-Small AML dataset. Place research inputs under:

```text
aml-detection-project/data/
├── raw/HI-Small_Trans.csv
└── external/HI-Small_Patterns.txt
```

Datasets, virtual environments, checkpoints, experiment artifacts, runtime databases, and promoted production bundles are excluded from Git because they are generated, environment-specific, or too large for normal source control. The tracked model JSON files and reports capture the selected research results.

## Reproducing the workflow

1. Install the research dependencies and add the dataset locally.
2. Run the Week 3 notebooks in order to reproduce the research checkpoints, evaluation, figures, and reports.
3. Follow the production training commands to build a release bundle from the same source data.
4. Run the production test suite before promoting or serving a bundle.
5. Start the local application directly or with Docker Compose.

## Repository hygiene

- Never commit `.env` files, credentials, customer data, local databases, or raw transaction datasets.
- Keep generated checkpoints and model bundles in ignored directories or an external artifact store.
- Update the relevant subproject README when changing commands, paths, data contracts, or model behavior.
- Run the production tests before opening a pull request.

