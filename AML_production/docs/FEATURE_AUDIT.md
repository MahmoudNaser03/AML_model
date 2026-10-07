# Feature and evaluation audit

## What was wrong

1. In the original notebook, grouping by `(to_bank, to_account, from_bank, from_account)` merely reordered the columns of a directed-pair key. It did **not** query previously observed transactions in the reverse direction. Repeated A→B payments were therefore treated as reverse activity. The former serving implementation instead looked up B→A, creating training/serving skew.
2. The original split assigned a whole laundering pattern to the split containing its earliest transaction. Later members could therefore appear in training after validation/test had started. Counts changed materially after using strict time boundaries: test now contains 1,798 positives, versus 1,265 in the stored original output.
3. The previous serving receiver-diversity update used reverse-pair novelty. Incoming-counterparty diversity must increase when a **new forward pair** first reaches that receiver.
4. Both old boosters consumed cluster ID/distance, but the trained scaler and cluster model were missing. Supplying zeros was a material feature change.
5. CSV numeric type inference could erase leading zeros from identifiers. Negative values were clipped while JSON rejected them; blank/nonfinite values were not consistently rejected.

The original notebook and old feature engine are retained as historical references, not as production dependencies.

The original source project is `../aml-detection-project` (left unchanged). Its `src/features/reciprocal.py` uses the same reordered-key grouping, and `src/splitting.py` assigns entire pattern components by their earliest row. Its `reports/week3/AML_week3_results.md` reports 139 features and 1,265 test positives. These findings independently confirm the notebook audit. Its recorded scores are historical context, not a comparable baseline for the corrected temporal experiment.

## Current contract

`aml_app/pipeline.py` implements `causal-batch-v2`. `aml_app/features.py` re-exports the public interface. Training and serving both call `build_features`; there is no independent notebook feature implementation.

- Normalize identifiers as strings; bank/account identity is a tuple, not concatenated text.
- Reject conflicting aliases, missing fields, blank identifiers, invalid timestamps, nonfinite amounts and negative amounts.
- Drop all extra input fields before feature generation, including labels, pattern metadata and caller-supplied feature values.
- Sort UTC timestamps stably, retaining source order for ties and restoring output order after scoring.
- Use only earlier rows of the same batch. Single-row submissions start cold.
- Produce 63 float32 features. Unknown artifact features fail explicitly.
- Compute true reverse-edge history by looking up the last earlier event for the opposite directed pair.
- Count distinct incoming and outgoing counterparties correctly.
- Add real 1-hour/24-hour activity windows, prior-20/prior-100 amount means, payment-format indicators, and a sender's earlier incoming transfers before onward payments.
- Retrain without clustering. There are no optional missing inputs in the new model contract.

The implementation hash and ordered feature list are recorded in each manifest. Changing the feature source requires a new matching export, even if the edit looks harmless.

The entity-v3 release keeps this 63-feature contract intact and appends 35 features from an immutable training-cutoff registry. Registry keys are 128-bit hashes; values are aggregate transaction and positive counts. Training features are leave-one-out, while later splits and serving never update the registry. Both the registry and its implementation have independent manifest checksums. Uploaded labels are ignored by scoring.

## Experimental boundaries

The data is split chronologically into 60% fit, 10% tune, 10% threshold calibration and 20% final test. Equal timestamps stay together. Features start cold within each partition because the deployed contract is an independent uploaded batch.

Pattern TXT membership selects eligible positive examples for the specialist's **training** population only. Both detectors are evaluated on the complete tune/calibration/test populations. Patterns may span time boundaries; their later transactions are never moved backward into training. This is a temporal benchmark, not a claim of generalization to completely unseen accounts or laundering attempts.

Only normal training rows are sampled (up to 250,000, fixed seed); all eligible positive training rows remain. Evaluation prevalence is unchanged. The current deployment supports at most 50,000 uploaded rows, whereas benchmark partitions are larger: real upload-window performance still depends on the available history. Do not equate a one-row demonstration with the full benchmark.

## Research references

- [AMLworld / Realistic Synthetic Financial Transactions](https://arxiv.org/abs/2306.16424): dataset provenance and model-comparison context.
- [Graph Feature Preprocessor](https://arxiv.org/abs/2402.08593): evidence for exploring causal graph patterns with boosted trees. This project does not reproduce its proprietary feature implementation or claim its benchmark results.
- [IBM Multi-GNN](https://github.com/IBM/Multi-GNN): reference implementations for graph models with directed-multigraph adaptations; these were researched, not trained in this CPU experiment.

Their scores use different pipelines and evaluation protocols. Comparisons here are limited to models evaluated on this project's identical corrected split.
