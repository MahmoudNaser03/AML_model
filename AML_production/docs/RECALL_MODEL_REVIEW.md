# Recall model review

## Outcome

The selected v3 ensemble reaches **81.03% recall at 21.35% precision** on the same corrected chronological test population used by causal-v2. The previous release reached **72.47% recall at 20.30% precision**. That is **+8.57 percentage points of recall**, **+1.05 points of precision**, and 154 additional detected positives. It is a substantial improvement but does not equal a full +10 recall points, so it is not reported as one.

| Release | TP | FP | FN | Precision | Recall | Average precision |
|---|---:|---:|---:|---:|---:|---:|
| causal-v2 | 1,303 | 5,117 | 495 | 20.30% | 72.47% | 0.6308 |
| entity-v3 | 1,457 | 5,368 | 341 | 21.35% | 81.03% | 0.7013 |

The final policy was chosen on tune. Its score threshold, 0.0311456136405468, was locked on calibration at the previous test precision target before the final candidate was scored. Calibration measured 72.08% recall and 20.40% precision. The score is an operating score, not a calibrated probability.

## Models screened on the same tuning split

All rows in tune were retained. Values below use a 15% precision floor unless noted.

| Candidate | Precision | Recall | Average precision | Decision |
|---|---:|---:|---:|---|
| causal-v2 specialist baseline | ≥15% | 56.17% | 0.4114 | Baseline |
| XGBoost unpatterned specialist | infeasible/usefully near zero | near zero | 0.029–0.044 | Rejected |
| LightGBM unpatterned specialist | infeasible | near zero | 0.027–0.031 | Rejected |
| LightGBM all positives | 17.15% | 46.87% | 0.3692 | Rejected |
| LightGBM patterned specialist | 15.05% | 55.10% | 0.3879 | Rejected |
| HistGradientBoosting unpatterned | infeasible | near zero | 0.0076 | Rejected |
| XGBoost patterned + hard negatives | 15.01% | 56.71% | 0.4293 | Rejected |
| XGBoost patterned + 1M random negatives | ≥15% | 56.35% | 0.4310 | Rejected |
| causal graph-v3 XGBoost | 15.09% | 58.14% | 0.4123 | Too small a gain |
| entity-aware general XGBoost | 15.42% | 70.30% | 0.5585 | Selected component |
| entity-aware patterned XGBoost | 15.03% | 70.84% | 0.6020 | Selected component |

Extra Trees was also attempted but exceeded the practical memory/runtime envelope and was removed from the default screen. Larger samples, deeper trees, algebraic feature transforms, and LightGBM fusions did not beat the selected entity approach.

## Why the selected model helped

The earlier specialist detected patterned positives well but detected very few unpatterned positives. The new features add fixed, training-only history for sender, receiver, either-role entity, directed pair, and bank roles. Each group contributes prior transaction count, prior positive count, a known-positive flag, and two smoothed risk rates. Training rows use leave-one-out aggregates; tune, calibration, test, and serving use only the frozen training registry.

The registry contains 128-bit hashes and aggregate counts, not transaction rows. It is about 40 MiB. Raw account identifiers, the source CSV, pattern files, and the complete `data/` directory are absent from the release and Docker build context.

## Research context and limits

The graph-feature direction was motivated by IBM's [AMLworld dataset paper](https://papers.neurips.cc/paper_files/paper/2023/file/5f38404edff6f3f642d6fa5892479c42-Paper-Datasets_and_Benchmarks.pdf), [Graph Feature Preprocessor paper](https://arxiv.org/abs/2402.08593), and [Multi-GNN reference implementation](https://github.com/IBM/Multi-GNN). Their reported numbers use different models and protocols and are not compared directly with this result.

The chronological test set was reused during this iterative project after earlier candidate summaries had been observed. The final model and weights were selected from tune and the threshold from calibration, but a genuinely fresh forward-time dataset is still required before claiming external generalization or institutional production readiness. Entity-history features also benefit known entities; unseen institutions and accounts should be monitored separately.
