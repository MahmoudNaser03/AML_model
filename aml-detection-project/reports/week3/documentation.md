# AML Detection Project — Technical Documentation

**Dataset:** IBM AML Synthetic — HI-Small (`HI-Small_Trans.csv`)
**Scope:** 5,078,345 transactions across 10 days (2022-09-01 to 2022-09-10, after tail trimming), laundering rate ~0.10%
**Patterns file:** `HI-Small_Patterns.txt` — 370 named laundering pattern instances, 3,209 tagged transactions

---

## Part 1: Data Splitting

### The Problem

AML data has a unique structural property: laundering doesn't happen in isolated transactions — it happens in **patterns** (chains of related transactions). A single laundering scheme (e.g., a fan-out from one account to 12 others) spans multiple rows. If a random or naive time split places some rows of a pattern in train and others in test, the model sees partial answers during training — a severe form of data leakage.

### Approaches Implemented

Three splitting strategies were developed (in `src/data/split.py`), each addressing the problem differently:

#### 1. Pure Time Split (`time_split`)

- Sorts all transactions chronologically and cuts at 70/15/15 ratio boundaries.
- **Problem:** Patterns that straddle boundaries become "broken" — the model trains on partial patterns and tests on their continuations. The `count_broken_patterns()` utility quantifies this: in a naive time split, dozens of patterns span train/val or val/test boundaries.
- **Use case:** Baseline comparison only. After splitting, `remove_broken_patterns()` can drop all affected rows, but this discards real laundering signal.

#### 2. Pattern-Based Split (`pattern_split`)

- Groups transactions by `pattern_id` and assigns entire patterns to splits (by each pattern's first timestamp, respecting temporal order).
- Legitimate (non-pattern) rows follow the time boundaries.
- **Advantage:** No pattern ever straddles a boundary — zero leakage by construction.
- **Problem discovered in practice:** Patterns grow smaller over time in this dataset. Sequential assignment means train gets mostly large patterns (early), while val/test get small patterns (late). Result: laundering rates decline across splits (train ~0.11%, val ~0.06%, test ~0.05%), making evaluation unreliable.

#### 3. Stratified Pattern Split (`stratified_pattern_split`) — CHOSEN

- Groups patterns by `pattern_type` (CYCLE, FAN-OUT, FAN-IN, etc.), orders within each type by first timestamp, then **interleaves** assignments using evenly-spaced index selection.
- Every Nth pattern goes to val/test regardless of temporal position, so all three splits share the full size distribution within each typology.
- Legitimate rows still follow chronological order (70/15/15 cut).

**Why this was chosen:**
- Zero pattern leakage (same guarantee as `pattern_split`).
- Balanced laundering rates across splits: Train 0.0882%, Val 0.0964%, Test 0.0885% — within 10% of each other.
- Each split sees all 8 pattern types (FAN-IN, FAN-OUT, CYCLE, GATHER-SCATTER, SCATTER-GATHER, BIPARTITE, STACK, RANDOM).
- The `audit_split()` function produces 4 diagnostic figures confirming balance, temporal coverage, pattern-type distribution, and feature distributions.

### Final Split Sizes

| Split | Rows | Patterns | Laundering Rate | Date Range |
|-------|------|----------|-----------------|------------|
| Train | 3,537,292 | 253 | 0.0882% | Sep 1–10 |
| Val | 758,039 | 55 | 0.0964% | Sep 1–10 |
| Test | 757,936 | 55 | 0.0885% | Sep 1–10 |

### Verification Guarantees

The `_verify_pattern_assignment()` function checks:
1. All 3,209 pattern rows from the patterns file are matched in the merged data.
2. No legitimate row (`Is Laundering == 0`) incorrectly receives a `pattern_id`.
3. Per-pattern row counts match the source file exactly.
4. No `pattern_id` appears in more than one split (leakage check in `audit_split()`).

---

## Part 2: Feature Engineering

### Design Principles

1. **Two-stage pipeline to prevent leakage:**
   - Stage 1 (pre-split): Features derivable from a single row's own fields or sequential order — safe on full data.
   - Stage 2 (post-split): Aggregation/statistical features computed on train only, then applied to val/test via lookup.

2. **Graph features built on train data only:** The directed transaction graph is constructed from training transactions; val/test rows are scored against this same graph structure. This prevents future information from leaking into the graph topology.

3. **NaN as signal, not missing data:** For features like `gs_gap_hrs` (gather-scatter gap hours) and `relay_conservation`, NaN means "this account is not a gather-scatter hub / not a relay node." XGBoost natively learns optimal missing-value routing, so these NaNs carry information.

### Feature Groups (96 total features, 70 used in modeling)

#### Group 1: Transaction-Level Features (`src/features/transaction_features.py`)

| Feature | Description | Why It Helps |
|---------|-------------|--------------|
| `amount_paid_usd` / `amount_received_usd` | Amounts normalized to USD (Sep 2022 rates) | Cross-currency comparison; laundering median ~$8.7K vs legit ~$1.4K |
| `amount_diff` | `Amount Paid - Amount Received` | Non-zero = currency conversion/fees = layering signal |
| `log_amount` / `log_amount_diff` | Log-transformed amounts | Stabilizes the 6-orders-of-magnitude skew for tree models |
| `is_cross_currency` | Payment currency != Receiving currency | Cross-currency adds obfuscation |
| `amount_round_number_flag` | Amount is exactly round (e.g., 10,000.00) | Structuring behavior uses round numbers |
| `same_bank` | Sender and receiver at same bank | Internal transfers have different risk profile |
| `same_currency` | Payment and receiving currency match | Complement to cross-currency flag |
| `dist_to_risk_zone` | Distance from the nearest high-risk amount region | Captures proximity to structuring thresholds |
| `dist_from_10000` / `just_under_10000` | Distance from $10K CTR threshold / flag for amounts $9K–$10K | Direct smurfing indicator; just-under-10K transactions have **3.26x higher fraud rate** |
| `dist_from_3000` / `just_under_3000` | Same for $3K MSB threshold | Secondary structuring signal |
| `closest_threshold_dist` | Min distance to any reporting threshold | Unified structuring feature |

**Strongest single feature:** `dist_from_10000` with univariate AUC = 0.7421.

#### Group 2: Temporal Features (`src/features/transaction_features.py`)

| Feature | Description | Why It Helps |
|---------|-------------|--------------|
| `hour_cos` | Cyclical encoding of hour: cos(2*pi*hour/24) | Laundering peaks midday; cyclical avoids the 23→0 discontinuity |
| `dow_cos` | Cyclical encoding of day-of-week | Weekend laundering rate ~2-3x weekday |
| `is_weekend` | Saturday/Sunday flag | Direct encoding of the weekend risk elevation |

**Univariate AUCs:** hour_cos = 0.5984, dow_cos = 0.5780.

#### Group 3: Account-Level Features (`src/features/account_features.py`)

Built per-account using composite IDs: `Sender_ID = Account + "_" + From Bank`, `Receiver_ID = Account.1 + "_" + To Bank`.

| Feature | Description | Why It Helps |
|---------|-------------|--------------|
| `total_sent` / `total_received` | Sum of amounts per account | High-volume accounts are suspicious |
| `avg_sent` / `avg_received` | Mean transaction size per account | Deviation from average is informative |
| `sent_count` / `received_count` | Transaction frequency per account | High frequency + small amounts = smurfing |
| `log_total_sent` / `log_avg_sent` / etc. | Log transforms of above | Normalizes extreme skew |
| `balance_turnover_ratio` | `total_sent / total_received` | Ratio ~1.0 = pass-through/mule account |
| `is_hub_sender_flag` / `is_hub_receiver_flag` | Account in extreme tail (99.96th percentile) of volume | Hub node detection |
| `dominant_payment_format` | Most-used payment format per sender | Accounts using ACH align with highest-risk channel |
| `dominant_currency` | Most-used currency per sender | Concentration in high-risk currencies |
| `sender_amount_zscore` | Z-score of this transaction vs sender's history | Deviations from own behavior are suspicious |
| `sender_amount_ratio` | This amount / sender's mean amount | Relative size signal |
| `sender_amount_pct_rank` | Percentile rank within sender's transactions | Position in personal distribution |
| `recipient_prev_tx_count` | How many times this sender→receiver pair transacted before | New pairs are higher risk |
| `new_sender_recipient_pair` | First-ever transaction between this pair | Binary novelty flag |
| `distinct_users_on_recipient` | Unique senders to this receiver | Fan-in detection via account stats |

**Fit-on-train, apply-to-val/test:** All aggregates (means, counts, thresholds) are computed from training data only. Val/test accounts not seen in train get the global training mean as fallback.

#### Group 4: Graph/Network Features (`src/features/graph_features.py`) — 18 features

The most powerful tier. A directed graph is built from training transactions (512,444 nodes, 952,438 unique edges). Each feature targets specific AML typologies:

| Typology | Features | Implementation |
|----------|----------|----------------|
| **FAN-OUT** | `sender_out_degree`, `sender_unique_recv_window`, `out_burst_count` | Out-degree count + 72h rolling window unique receivers + burst detection |
| **FAN-IN** | `receiver_in_degree`, `receiver_unique_send_window` | In-degree count + rolling window unique senders |
| **CYCLE** | `sender_in_cycle`, `sender_scc_size` | Tarjan's SCC algorithm on the CSR adjacency matrix |
| **GATHER-SCATTER** | `gs_flag`, `gs_gap_hrs` | Numba-JIT'd core: detects accounts receiving from many then quickly sending to many; gap = hours between last receive and first send |
| **SCATTER-GATHER** | `sg_flag`, `sg_width` | Reverse of gather-scatter: send to many, then a common target collects |
| **STACK/RELAY** | `relay_flag`, `relay_conservation` | Detects sequential forwarding; conservation = amount_out/amount_in ratio |
| **BIPARTITE** | `bipartite_flag`, `sender_community_size` | Pure-sender detection + weakly-connected component size |
| **RANDOM** | `relay_currency_switch`, `sender_currency_entropy` | Currency switching between in/out + Shannon entropy of currencies used |
| **STRUCTURING** | `amount_cv_window`, `near_round_amount` | Coefficient of variation in rolling window + round-number proximity |
| **TEMPORAL BURST** | `inter_txn_gap_cv` | CV of inter-transaction time gaps per sender |

**Key results:**
- `sender_in_cycle` is the most significant binary graph feature (chi-squared = 587, p ≈ 0).
- Laundering transactions in cycles have mean SCC size = 1,347 vs 851 for legitimate.
- `relay_conservation` is NOT significant (p = 0.654) — kept anyway because XGBoost might find interactions.
- Pattern detection matrix confirms alignment: CYCLE pattern has highest `sender_in_cycle`, FAN-OUT has highest `sender_out_degree`.

**Performance optimization:** Heavy computations use Numba JIT compilation (`@njit`) for the gather-scatter, scatter-gather, and relay cores — running on 3.5M training rows in under 2 minutes.

#### Group 5: Target-Encoded Features

| Feature | Description | Why It Helps |
|---------|-------------|--------------|
| `payment_format_risk` | Target-encoded Payment Format (smoothed) | ACH has 0.75% laundering rate, ~20-47x other formats |
| `bank_risk_score` | Target-encoded From Bank (James-Stein smoothing, alpha=50) | Some banks have laundering rates as high as 27% |
| Currency OHE (20 cols) | One-hot for 10 payment + 10 receiving currencies | Captures currency-specific risk |

**Smoothing formula:** `risk = (n * raw_rate + alpha * global_rate) / (n + alpha)` — prevents overfitting on rare banks.

#### Group 6: Derived Ratio Features

| Feature | Description | Why It Helps |
|---------|-------------|--------------|
| `avg_amount_per_recv` | `total_sent / sender_out_degree` | Laundering mean = $3.67M vs legit = $134K (p ≈ 0) |
| `tx_density` | `sent_count / distinct_days_active` | Laundering mean = 37.4 vs legit = 22.8 |
| `log_sender_unique_recv_window` | Log of rolling unique receivers | Normalizes skew for tree splits |
| `log_sent_count` | Log of transaction count | Same |
| `log_amount_paid_usd` | Log of USD amount | Same |

### Features Dropped Before Modeling (26 of 96)

Columns removed as leaky, redundant, or non-discriminative:
- Raw identifiers: Timestamp, From Bank, To Bank, Account, Account.1, Sender_ID, Receiver_ID
- Raw categorical strings: Payment Currency, Receiving Currency, Payment Format
- Pattern metadata: pattern_id, pattern_type
- Non-discriminative: `sender_community_size` (one giant WCC), `tx_currency_change` (always 0)
- Intermediate columns: date, ts_sec, raw amounts (superseded by USD-normalized versions)

**Final model input: 70 features.**

---

## Part 3: Modeling

### Approach: XGBoost with Optuna Hyperparameter Search

**Why XGBoost:**
- Native handling of NaN (learns optimal missing-value direction at each split) — critical for our graph features where NaN = "not this typology."
- Handles extreme class imbalance via `scale_pos_weight`.
- Fast training on 3.5M rows.
- Feature importance and SHAP integration for interpretability.

**Why Optuna (not GridSearch/RandomSearch):**
- TPE (Tree-structured Parzen Estimator) sampler is more sample-efficient than random search.
- 40 trials explore the space intelligently, focusing on promising regions.
- Pruning stops unpromising trials early.

### Hyperparameter Search Space

| Parameter | Range | Best Value | Purpose |
|-----------|-------|------------|---------|
| `n_estimators` | 150–500 (step 50) | 500 | Number of boosting rounds |
| `max_depth` | 4–10 | 10 | Tree complexity |
| `learning_rate` | 0.02–0.15 (log) | 0.121 | Step size shrinkage |
| `subsample` | 0.6–1.0 | 0.893 | Row sampling per tree |
| `colsample_bytree` | 0.5–1.0 | 0.969 | Feature sampling per tree |
| `min_child_weight` | 1–30 | 25 | Minimum leaf hessian sum (regularization) |
| `gamma` | 0.0–3.0 | 1.153 | Minimum split loss reduction |
| `reg_alpha` | 0.0–2.0 | (from Optuna) | L1 regularization |
| `reg_lambda` | 0.5–4.0 | (from Optuna) | L2 regularization |
| `spw_multiplier` | 0.8–2.5 | 1.160 | Multiplier on `scale_pos_weight` |

**Objective:** Maximize Average Precision (PR-AUC) on the validation set. PR-AUC was chosen over ROC-AUC because it is far more informative under extreme imbalance (0.1% positive rate) — ROC-AUC can appear high (>0.90) even with poor precision at low thresholds.

### Training Pipeline

1. **Optuna search (40 trials):** Each trial trains XGBoost with a candidate config, evaluates PR-AUC on val.
2. **Best model retrained** with `early_stopping_rounds=30` → stops at round 362 (out of 500).
3. **Isotonic calibration** applied on val set predictions — transforms raw scores into well-calibrated probabilities for reliable threshold selection.
4. **Threshold selection:** Operating point chosen to achieve recall >= 0.65 while maximizing precision.

### Results

#### Validation Set (used for tuning)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9449 |
| PR-AUC (Avg Precision) | 0.2326 |
| Recall (@ threshold 0.003) | 0.6599 |
| Precision (@ threshold 0.003) | 0.0111 |
| F1 (@ threshold 0.003) | 0.0219 |
| TP / FP / FN / TN | 390 / 34,679 / 201 / 726,223 |

#### Test Set (held-out, single uniform threshold)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9325 |
| PR-AUC (Avg Precision) | 0.1413 |
| Recall (@ threshold 0.003) | 0.5970 |
| Precision (@ threshold 0.003) | 0.0082 |
| F1 (@ threshold 0.003) | 0.0161 |
| TP / FP / FN / TN | 280 / 33,947 / 189 / 723,520 |

**Generalization gap:** PR-AUC drops from 0.2326 (val) to 0.1413 (test), indicating some overfitting to val patterns.

#### Test Set (segmented threshold strategy)

A two-threshold approach was developed to address the false-positive problem:
- **High-volume senders** (top 10% by `sent_count`, cutoff = 4,306 transactions): threshold = 0.01012
- **Standard senders:** threshold = 0.00213

| Metric | Uniform | Segmented | Delta |
|--------|---------|-----------|-------|
| Recall | 0.5970 | 0.6652 | +11.4% |
| Precision | 0.0082 | 0.0154 | +87.8% |
| F1 | 0.0161 | 0.0300 | +86.3% |
| FP count | 33,947 | 19,997 | -41.0% |

The segmented strategy simultaneously improves recall AND reduces false positives — a Pareto improvement.

### SHAP Analysis — Feature Importance

**Top 10 features by mean |SHAP|:**

1. `payment_format_risk` — The single strongest predictor
2. `sent_count` — Transaction frequency
3. `sender_unique_recv_window` — Rolling unique receivers (fan-out signal)
4. `bank_risk_score` — Target-encoded bank risk
5. `avg_amount_per_recv` — Average amount per distinct receiver
6. `sender_out_degree` — Total out-degree in the graph
7. `log_sent_count` — Log-transformed frequency
8. `amount_paid_usd` — Raw transaction amount
9. `sender_in_cycle` — Binary cycle membership
10. `sender_currency_entropy` — Currency diversity

### False Positive Analysis

At threshold 0.003: **1 real detection per 122 alerts** (precision = 0.82%).

**FP archetypes identified via KMeans clustering on SHAP vectors:**

| Cluster | % of FPs | Driving Features | Profile |
|---------|----------|------------------|---------|
| 0 | 27% | `payment_format_risk` + `sent_count` | High-risk format, high-volume legitimate accounts |
| 1 | 20% | `sender_unique_recv_window` + `bank_risk_score` | High-volume senders from risky banks |
| 2 | 3% | `payment_format_risk` + `sender_currency_entropy` | Currency-diverse, high-risk format |
| 3 | 51% | `sender_unique_recv_window` + `sent_count` + `bank_risk_score` | **Dominant archetype**: high-volume legitimate senders that mimic launderer behavior |

**Root cause:** `sent_count` and `sender_unique_recv_window` have nearly identical mean SHAP values in both FP and TP groups — the model cannot distinguish high-volume legitimate senders from actual launderers using these features alone.

### Suggested Next Steps (from SHAP diagnosis)

1. **Engineer disambiguation features:** account age, historical false-positive rate, counterparty diversity beyond simple degree
2. **Two-stage model:** first stage flags, second stage classifies flagged alerts as TP vs FP
3. **Post-filter rules:** suppress alerts where only volume features fire (no graph/typology signal)
4. **Segment-specific thresholds:** the implemented two-threshold approach already reduces FPs by 41%

---

## Summary of Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Stratified pattern split (not random, not pure time) | Prevents pattern leakage while maintaining balanced laundering rates |
| Two-stage feature pipeline (pre-split + post-split) | Prevents aggregation leakage from val/test into train statistics |
| Graph built on train only | Val/test use frozen graph structure — no future-information leakage |
| PR-AUC as Optuna objective (not ROC-AUC, not recall) | Most informative metric under 0.1% positive rate; avoids threshold-dependent optimization |
| Recall >= 0.65 operating point | AML systems must catch most laundering; precision is improved via post-hoc strategies |
| Isotonic calibration | Transforms raw XGBoost scores to reliable probabilities for threshold tuning |
| NaN preservation for graph features | XGBoost learns "not-a-hub" as a meaningful signal path |
| Numba JIT for graph typology cores | 3.5M-row dataset requires sub-minute performance for gather-scatter, relay detection |
| Segmented thresholds by sender volume | Addresses the dominant FP archetype without sacrificing recall |
