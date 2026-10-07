# Measured results — entity-v3

Full dataset: **5,078,345 transactions**, **5,177 labelled positives**. No test rows were sampled. Reproducible commands and the executed notebook are linked from the README.

| Split | Rows | Positives | Role |
|---|---:|---:|---|
| Train | 3,046,861 | 2,297 | Fit models; sample up to 250k normal training rows |
| Tune | 507,788 | 559 | Early stopping, candidate and fusion selection |
| Calibration | 507,814 | 523 | Lock operating threshold |
| Test | 1,015,882 | 1,798 | Final evaluation |

The v3 stage adds frozen training-aggregate entity features to the corrected causal-v2 features. Two depth-3 XGBoost models use all positives and patterned positives respectively. They are combined geometrically with the previous specialist using weights 0.75, 0.20, and 0.05. Candidate selection used tune; the threshold was locked on calibration.

The selected geometric policy reached 72.99% tuning recall at 15.24% precision. On calibration, the locked operating point reached 72.08% recall at 20.40% precision.

## Held-out comparison at thresholds locked on calibration

| Detector | Precision | Recall | Average precision | True positives | False positives | Alerts |
|---|---:|---:|---:|---:|---:|---:|
| Previous causal-v2 | 20.30% | 72.47% | 0.6308 | 1,303 | 5,117 | 6,420 |
| Selected entity-v3 ensemble | **21.35%** | **81.03%** | **0.7013** | **1,457** | **5,368** | **6,825** |

The selected policy catches **154 more positives** than causal-v2 while improving precision by 1.05 percentage points. Its F1 is 0.3379 and alert rate is 0.672%. Threshold: **0.0311456136405468**. This is an operating score threshold, not a calibrated probability of illicit activity.

The original notebook's stored A metrics (14.39% precision / 72.25% recall) use a different split and flawed feature semantics. They cannot support an apples-to-apples improvement claim. The comparison above uses identical corrected split semantics.

## Evidence

- `production_artifacts/evaluation.json`: locked evaluation.
- `production_artifacts/manifest.json`: hashes, versions, feature sources and dataset hashes.
- `production_artifacts/trials.json`: selected component trials.
- `production_artifacts/policy_search.json`: declared fusion candidates.

Source data in the sibling project was read but not modified. The old deployment is retained in a timestamped `production_artifacts.backup-*` directory.
