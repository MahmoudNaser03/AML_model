# Feature Engineering Proposal — AML Detection

**Dataset:** IBM AML — HI-Small (`HI-Small_Trans.csv`)
**Purpose:** Candidate features for a future laundering-detection model, derived from EDA findings and typology research (`HI-Small_Patterns.txt`).

Each feature includes: what it measures, why it could help detect laundering, and which typology(ies) it targets.

---

## 1. Transaction-Level Features

Features computed from a single transaction's own fields — the simplest tier, available immediately at transaction time.

| Feature | Description | Rationale | Related Typology |
|---|---|---|---|
| `amount_paid` / `amount_received` | Raw transaction amounts | Baseline signal; EDA showed laundering median (~$8.7K) is ~6x legit median (~$1.4K) | General |
| `amount_diff` | `Amount Paid - Amount Received` | Non-zero difference indicates currency conversion or fee — worth tracking as a secondary signal | Layering |
| `is_cross_currency` | `Payment Currency != Receiving Currency` | Cross-currency transfers add a layer of obfuscation and complexity to tracing | Layering |
| `payment_format` (encoded) | One-hot/target-encoded Payment Format | **Strongest single feature found in EDA** — ACH carries a 0.75% laundering rate, ~20–47x higher than other formats | All (esp. Smurfing, Layering) |
| `amount_round_number_flag` | Whether amount is suspiciously round (e.g. exactly 10,000.00) | Structuring often uses round numbers just under reporting thresholds | Smurfing |
| `amount_near_threshold_flag` | Whether amount sits just below common reporting thresholds (e.g. $9,000–$9,999 near a $10K threshold) | Direct indicator of structuring behavior | Smurfing |
| `log_amount` | `log(1 + Amount Paid)` | Normalizes the heavily skewed amount distribution for model stability | General |
| `currency_risk_score` | Encoded risk weight per currency, based on EDA laundering rate by currency | Saudi Riyal showed ~4x higher laundering rate than the next currency | General |

---

## 2. Account / Profile-Level Features

Aggregated per account (`Sender_ID` / `Receiver_ID`) — describe an account's overall behavior, not a single transaction.

| Feature | Description | Rationale | Related Typology |
|---|---|---|---|
| `total_sent` / `total_received` | Sum of amounts sent/received by the account (rolling window, e.g. 7/30 days) | Identifies accounts moving unusually large volumes | General |
| `avg_transaction_amount` | Mean transaction size for the account | A sudden deviation from an account's own average is more informative than an absolute amount threshold | General |
| `transaction_count` | Number of transactions per account (rolling window) | Very high frequency in a short period is a red flag, especially combined with small amounts | Smurfing |
| `balance_turnover_ratio` | `total_sent / total_received` per account, per time window | An account that sends out almost everything it receives (~1.0 ratio) behaves like a pass-through/mule account | Pass-through |
| `dominant_payment_format` | Most frequently used Payment Format per account | Accounts overwhelmingly using ACH align with the highest-risk channel found in EDA | Smurfing, General |
| `dominant_currency` | Most frequently used currency per account | Flags accounts concentrated in higher-risk currencies (e.g. Saudi Riyal) | General |
| `account_laundering_history_flag` | Whether the account has ever appeared in a confirmed laundering transaction (training-time only feature) | Direct historical risk signal — must be handled carefully to avoid leakage in modeling | General |
| `bank_risk_score` | Encoded risk weight per bank, based on EDA laundering rate by `From Bank`/`To Bank` | EDA found some banks with laundering rates as high as 27% (small volume, needs validation) | General |

---

## 3. Peer / Network (Graph) Features

Features that describe an account's position and role within the transaction graph — the most powerful tier for typology detection, since laundering is fundamentally a *relational* pattern, not a single-transaction anomaly.

| Feature | Description | Rationale | Related Typology |
|---|---|---|---|
| `out_degree` | # of distinct accounts a given account sends to (time window) | Directly detects Fan-Out — the dataset's FAN-OUT pattern (48 attempts) is defined by exactly this | Fan-Out |
| `in_degree` | # of distinct accounts sending to a given account (time window) | Directly detects Fan-In — matches the dataset's FAN-IN pattern (40 attempts) | Fan-In |
| `degree_ratio` | `out_degree / in_degree` | High ratio → fan-out hub; low ratio → fan-in hub; near 1 → possible pass-through or balanced actor | Fan-Out, Fan-In |
| `is_in_cycle` | Boolean flag: is the account part of a detected closed transaction loop (graph cycle detection, e.g. via `networkx`) | Directly matches the dataset's CYCLE pattern (54 attempts, the most common type found) | Circular/Cycle |
| `cycle_length` | Number of hops in the shortest detected cycle involving the account | Shorter, fast cycles are more suspicious than long incidental ones | Circular/Cycle |
| `gather_scatter_flag` | Account receives from many sources then quickly redistributes to many others | Matches GATHER-SCATTER (51 attempts) — the second most common pattern in this dataset | Fan-In → Fan-Out |
| `scatter_gather_flag` | Account sends to many destinations that later route back to a common account | Matches SCATTER-GATHER (44 attempts) | Fan-Out → Fan-In |
| `bipartite_cluster_flag` | Account belongs to a densely connected two-group transaction structure (many-to-many transfers between two clusters) | Matches BIPARTITE (49 attempts) — consistent with structuring across multiple parties | Smurfing/Structuring |
| `chain_position` | Position of the account within a detected sequential transaction chain (start/middle/end) | Matches STACK pattern (43 attempts) — sequential layering through intermediate accounts | Layering |
| `betweenness_centrality` | Graph centrality measure — how often the account sits on the shortest path between other accounts | High centrality suggests a relay/intermediary role, common in layering | Layering, Pass-through |
| `pagerank_score` | PageRank of the account within the transaction graph | Identifies accounts that are unusually "important" hubs relative to their surface-level activity | Fan-In, Gather-Scatter |
| `community_id` | Cluster/community the account belongs to (via community detection, e.g. Louvain) | Accounts in unusually small, dense, isolated communities may indicate coordinated laundering rings | All network-based typologies |
| `unique_counterparties_ratio` | `unique_counterparties / total_transactions` | A ratio close to 1 suggests an account rarely repeats counterparties — consistent with fan-out/scatter behavior | Fan-Out, Smurfing |

**Implementation note:** these require building an explicit transaction graph (e.g. `networkx`, `igraph`, or a graph database) with `Sender_ID`/`Receiver_ID` as nodes and transactions as edges. Given ~5M transactions, graph algorithms like betweenness centrality can be computationally expensive — testing on a subgraph or sampled time window is recommended before scaling.

---

## 4. Time-Level Features

Features capturing *when* transactions happen and how activity is distributed over time — both per-transaction and per-account.

| Feature | Description | Rationale | Related Typology |
|---|---|---|---|
| `hour_of_day` | Hour extracted from Timestamp | EDA found laundering rate peaks around midday (~11:00–16:00) | General |
| `day_of_week` | Weekday extracted from Timestamp | EDA found laundering rate highest on **Sunday (~0.30%)** and **Saturday (~0.20%)** — notably above weekday rates | General |
| `is_weekend` | Boolean flag for Saturday/Sunday | Direct, simple encoding of the weekend risk pattern found in EDA | General |
| `time_since_prev_transaction` | Time gap (minutes/hours) since the account's previous transaction (as sender or receiver) | Short gaps between an incoming and outgoing transaction on the same account indicate pass-through behavior | Pass-through |
| `transaction_velocity` | # of transactions per account within a rolling short window (e.g. 1 hour, 1 day) | Sudden bursts of activity — especially combined with small round amounts — indicate structuring or coordinated laundering | Smurfing, Fan-Out |
| `time_to_complete_cycle` | Total elapsed time across a detected cycle (first to last transaction in the loop) | Faster cycles suggest more deliberate/automated laundering vs. coincidental loops | Circular/Cycle |
| `burstiness_score` | Statistical measure of how clustered vs. evenly spread an account's transaction timestamps are | High burstiness (many transactions in tight clusters, then silence) is atypical of normal account behavior | Smurfing, Fan-Out |
| `inter_arrival_std` | Standard deviation of time gaps between an account's consecutive transactions | Low variance combined with high frequency suggests automated/scripted transaction generation | Smurfing |
| `days_active` | Number of distinct days the account has any transaction activity (within the dataset's 10-day window) | Very short-lived but high-activity accounts are consistent with disposable mule accounts | Pass-through |

---

## 5. Cross-Cutting / Interaction Features

Features that combine signals across the tiers above — often the most predictive in practice, since typologies rarely show up in a single dimension alone.

| Feature | Description | Rationale | Related Typology |
|---|---|---|---|
| `high_out_degree_x_ach_flag` | `out_degree` above a threshold **and** `dominant_payment_format == ACH` | Combines the two strongest individual signals found (fan-out structure + highest-risk channel) | Fan-Out |
| `weekend_x_high_velocity_flag` | High transaction velocity occurring specifically on a weekend | Combines the time-of-week finding with the burst-activity finding — weekend + burst is a compounding risk signal | Smurfing, General |
| `amount_consistency_score` | Variance of transaction amounts within a detected fan-out/fan-in group | Structuring often uses similar (not identical) amounts across many transactions — low variance within a group is suspicious | Smurfing, Fan-Out, Fan-In |
| `cross_currency_x_cycle_flag` | Account is part of a detected cycle **and** involves multiple currencies | Currency-hopping within a cycle adds obfuscation, a classic layering-within-circular pattern | Layering, Circular/Cycle |

---

## Summary — Feature Tier Priorities

If implementation time is limited, the recommended build order (highest expected value first) is:

1. **Payment Format encoding + currency risk score** (transaction-level) — cheapest to compute, strongest single signal from EDA
2. **In-degree / Out-degree** (peer-level) — simple groupby counts, directly targets the two most literal typologies (Fan-In, Fan-Out) and their combinations (Gather-Scatter, Scatter-Gather)
3. **Time-of-week / velocity features** (time-level) — cheap to compute, backed by a clear EDA finding (weekend risk elevation)
4. **Cycle detection & centrality measures** (peer-level, graph-based) — highest potential value (matches the most common pattern, CYCLE, 54 attempts) but also the most computationally expensive; recommend testing on a time-windowed subgraph first
