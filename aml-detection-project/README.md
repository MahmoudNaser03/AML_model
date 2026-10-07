# AML Detection Project — DS Internship Summer 2026

Anti-Money Laundering (AML) transaction classifier built on the **HI-Small** synthetic dataset.
Dual-model architecture: a general AML detector (Model A) paired with a patterned-laundering specialist (Model B).

---

## Project Structure

```
aml-detection-project/
├── data/
│   ├── raw/              HI-Small_Trans.csv          ← source transactions
│   ├── external/         HI-Small_Patterns.txt       ← laundering-pattern definitions
│   └── processed/        intermediate parquet/pkl artefacts
├── notebooks/
│   ├── week2_eda/        Week 2 — EDA & typology analysis
│   └── week3/            Week 3 — full ML pipeline (6 notebooks, run in order)
│       ├── 01_data_preparation.ipynb
│       ├── 02_feature_engineering.ipynb
│       ├── 03_model_training.ipynb
│       ├── 04_evaluation_and_diagnostics.ipynb
│       ├── 05_interpretation_shap.ipynb
│       ├── 06_reporting.ipynb
│       └── 07_false_positive_triage.ipynb  ← optional second-stage alert triage
├── src/                  Reusable Python modules
│   ├── config.py         Paths, seeds, hyper-parameters
│   ├── data_loading.py   CSV loading, pattern-file parsing, pattern mapping
│   ├── splitting.py      Leakage-safe chronological train/val/test split
│   ├── modeling.py       XGBoost builder, feature-matrix builder, clustering
│   ├── evaluation.py     Metrics, operating-point calibration, fusion search
│   ├── state.py          Pickle checkpoint helpers
│   └── features/
│       ├── helpers.py          safe_ratio, add_group_cumcount
│       ├── base.py             Base transaction & local anomaly features
│       ├── history.py          Causal account/pair cumcounts & deltas
│       ├── rolling.py          Vectorised causal rolling means & counts
│       ├── novelty.py          First-seen / novelty indicators
│       ├── reciprocal.py       Reverse-transaction timing & amount features
│       ├── acceleration.py     Short-vs-long-term behavioural acceleration
│       ├── network.py          Fan-in / fan-out / hub features
│       ├── corridor.py         Bank-corridor features
│       ├── pattern_meta.py     TRUE TXT membership metadata (never model features)
│       ├── veteran.py          V3 veteran-account anomaly features
│       └── structural_scores.py  Transaction-derived structural pattern scores
├── checkpoints/          Pipeline stage pickles (01_*.pkl … 05_*.pkl)
├── models/               Saved XGBoost JSON models
├── artifacts/            Experiment CSVs and reports
├── figures/              Saved plots
└── reports/
    └── week3/            Markdown experiment report
```

---

## Week 3 Pipeline

Run notebooks 01–06 **in order** from `notebooks/week3/`. Notebook 07 is an
optional post-model experiment that uses their saved checkpoints.

| # | Notebook | Input | Output checkpoint |
|---|----------|-------|-------------------|
| 01 | Data Preparation | raw CSV + patterns.txt | `01_df`, `01_pattern_map`, `01_indices` |
| 02 | Feature Engineering | `01_*` | `02_df_features` |
| 03 | Model Training | `02_*` + `01_indices` | `03_models`, `03_val_scores`, `03_matrices` |
| 04 | Evaluation & Diagnostics | `03_*` + `02_*` | `04_eval` |
| 05 | SHAP Interpretation | `03_*` | `05_shap` |
| 06 | Reporting | all above | CSVs + markdown report |
| 07 | False-positive Triage | `01_indices`, `02_df_features`, `03_val_scores`, `04_eval` | comparison CSVs + figure |

Notebook 07 is an optional post-model experiment. It compares base-score ranking,
K-Means alert segmentation, and a supervised XGBoost triage model on Model A's
positive alerts. It keeps the original test split locked for the final comparison.

### Setting up the environment

```bash
cd aml-detection-project
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

> Notebooks auto-discover the project root via sys.path injection — they work
> whether launched from the project root, `notebooks/week3/`, or anywhere else.

---

## Model Architecture

```
Transactions ──► Feature Engineering (100+ causal features)
                         │
             ┌───────────┴───────────┐
             │                       │
          Model A               Model B
       General AML          Patterned-Laundering
        Detector               Specialist
     (all laundering)     (TXT-patterned only)
             │                       │
             └───────────┬───────────┘
                    Fusion search
                  (validation-only)
```

**Key design principles:**
- **Causal features only** — no future information leaks into any feature
- **Pattern-component-aware split** — no laundering attempt crosses train/val/test
- **Zero leakage** — TRUE Pattern TXT membership is used only for grouping/diagnostics, never as a model feature

---

## Key Results (Week 3)

- Feature count: ~100 causal features across 11 families
- Operating point: 15% precision floor
- Model A: General AML detector (all laundering)
- Model B: Patterned-laundering specialist (TXT-patterned only)
- Fusion: validation-only search over weighted and non-linear rules

See `reports/week3/AML_week3_results.md` for full metrics after running the pipeline.

---

## Data

| File | Description |
|------|-------------|
| `data/raw/HI-Small_Trans.csv` | ~5M synthetic AML transactions |
| `data/external/HI-Small_Patterns.txt` | Laundering-pattern definitions (BEGIN/END blocks) |

Source: IBM Transactions for Anti Money Laundering (AML) — HI-Small subset.
