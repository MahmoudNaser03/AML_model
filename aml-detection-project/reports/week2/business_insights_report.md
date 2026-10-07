# Business Insights Report — AML Transaction EDA

**Dataset:** IBM AML — HI-Small (`HI-Small_Trans.csv`), cleaned
**Scope:** Week 2, Day 1 — Exploratory Data Analysis

---

## 1. Class Imbalance

| | Count | % |
|---|---|---|
| Legitimate | 5,073,159 | 99.898% |
| Laundering | 5,177 | 0.102% |

**Insight:** Laundering transactions represent roughly **1 in 980** transactions. This confirms severe class imbalance — a model that always predicts "not laundering" would be ~99.9% accurate while catching zero real cases. **Accuracy is not a usable metric** for any future detection model; precision/recall, F1, or recall-at-fixed-false-positive-rate should be used instead.

**FCC implication:** Any detection system must be evaluated on how well it finds the rare positive cases, not overall correctness — this should be stated explicitly to stakeholders before a model is built, to set the right expectations.

---

## 2. Transaction Amount

| | Legit | Laundering |
|---|---|---|
| Count | 5,073,159 | 5,177 |
| Mean | $4.48M | $36.1M |
| Median (50%) | $1,411 | $8,667 |
| Max | $1.05T | $84.9B |

**Insight:** Laundering transactions have a **higher median amount** (~$8,700 vs ~$1,400) — roughly 6x larger than a typical legitimate transaction. However, both distributions have extreme outliers pulling the mean far above the median, so median is the more reliable comparison here.

The log-scale distribution chart shows laundering transactions cluster more tightly around a specific value range, while legitimate transactions are more spread out across smaller amounts — suggesting laundering transactions in this dataset tend to sit in a specific "mid-to-large" amount band rather than being either very small or extremely large.

**Caveat:** Amount alone is a weak standalone predictor (see outlier analysis below) — it should be combined with other signals, not used in isolation.

---

## 3. Outlier Analysis (IQR method, log-transformed, per Payment Format)

| Format | % Outliers | Laundering Rate (Outliers) | Laundering Rate (Overall) |
|---|---|---|---|
| Bitcoin | 16.8% | 0.016% | 0.038% |
| Cheque | 5.1% | 0.015% | 0.017% |
| ACH | 5.3% | 0.467% | 0.746% |
| Cash | 3.4% | 0.018% | 0.022% |
| Credit Card | 2.9% | 0.010% | 0.016% |
| Wire | 2.5% | 0.000% | 0.000% |
| Reinvestment | 0.2% | 0.000% | 0.000% |

**Insight:** Being an amount-outlier does **not** increase laundering risk within any payment format — in most formats the outlier laundering rate is roughly equal to or even *lower* than the overall rate for that format. **Unusual transaction size is not, by itself, a red flag** in this dataset.

**FCC implication:** This directly supports the principle that **high-value transactions alone are insufficient** to flag laundering — contextual analysis (channel, account behavior, network position) matters more than raw amount.

---

## 4. Payment Format — the strongest signal found

| Format | Total Transactions | Laundering Count | Laundering Rate |
|---|---|---|---|
| **ACH** | 600,793 | 4,483 | **0.746%** |
| Bitcoin | 146,086 | 56 | 0.038% |
| Cash | 490,891 | 108 | 0.022% |
| Cheque | 1,864,331 | 324 | 0.017% |
| Credit Card | 1,323,324 | 206 | 0.016% |
| Reinvestment | 481,056 | 0 | 0.000% |
| Wire | 171,855 | 0 | 0.000% |

**Insight:** This is the strongest and most reportable finding in the EDA. **ACH transactions carry a laundering rate ~20–47x higher than every other payment format**, and account for **4,483 of the 5,177 total laundering cases (~87%)** — despite representing only about 11% of total transaction volume. Wire and Reinvestment show **zero** laundering cases in this dataset.

**FCC implication:** Payment channel/format is a far more discriminating feature than transaction amount. Any monitoring rule set or model should weight ACH transactions with substantially higher scrutiny, while formats like Wire and Reinvestment (in this dataset) can be deprioritized relative to ACH — though this should be validated further before being operationalized, since it's based on synthetic data (see Limitations).

---

## 5. Currency

| Currency (Receiving) | Total Transactions | Laundering Rate |
|---|---|---|
| **Saudi Riyal** | 89,971 | **0.416%** |
| Euro | 1,172,017 | 0.117% |
| US Dollar | 1,879,341 | 0.102% |
| Yen | 156,319 | 0.099% |
| Australian Dollar | 138,511 | 0.092% |
| Canadian Dollar | 141,357 | 0.091% |
| Yuan | 206,551 | 0.089% |
| Rupee | 192,065 | 0.087% |
| Ruble | 157,361 | 0.085% |
| Mexican Peso | 111,030 | 0.083% |

**Insight:** Saudi Riyal transactions show a laundering rate roughly **4x higher** than the next-highest currency (Euro) — despite being a lower-volume currency (~90K transactions vs ~1.2M–1.9M for Euro/USD). This is a large enough sample to be a meaningful signal, not noise.

**FCC implication:** Currency of settlement is a secondary but useful risk factor, worth layering on top of payment format when prioritizing review queues.

---

## 6. Banks

**Sending banks — highest laundering rates (min. volume filter applied):**
Several sending banks show laundering rates as high as **27.5%** (91 transactions, 25 flagged) — dramatically higher than the dataset-wide 0.1% average. However, these are **low-volume banks**, so this concentration could reflect either genuinely high-risk institutions or statistical noise from small sample sizes.

**Receiving banks** show a similar but less extreme pattern — top receiving banks range from ~1% to ~3.4% laundering rate.

**Insight:** A small number of banks account for a disproportionate share of laundering activity relative to their transaction volume. This is a strong candidate for **entity-level risk scoring** rather than transaction-level scoring alone.

**Caveat:** Low-volume banks with high rates need statistical caution — a bank with 91 total transactions and 25 flagged could be a real hotspot, or could be an artifact of how the synthetic data was generated. This should be flagged as requiring further investigation, not treated as a confirmed finding.

---

## 7. Time Patterns

- **Transaction volume by hour:** Roughly flat across all hours (~200K/hour), except a large spike at hour 0 (~600K) — likely a timestamp default/rounding artifact in the data generator rather than a real business pattern. Worth noting as a data-quality observation.
- **Laundering rate by hour:** Peaks around midday (~11:00–12:00 and 16:00, reaching ~0.175%), and is lowest at hour 0 (~0.025%) despite hour 0 having the highest raw transaction volume — reinforcing that the hour-0 spike is likely a data artifact, not a real behavioral pattern.
- **Laundering rate by weekday:** **Sunday is highest (~0.30%)**, followed by Saturday (~0.20%) — both weekend days show notably elevated laundering rates compared to weekdays (~0.10–0.13%), with Thursday and Friday being the lowest (~0.05–0.07%).

**Insight:** Laundering activity is **disproportionately concentrated on weekends**, when transaction monitoring staffing and oversight are typically lighter — a pattern consistent with real-world AML behavior (exploiting reduced scrutiny during off-hours).

**FCC implication:** This supports adjusting monitoring thresholds or staffing coverage for weekend transaction review, rather than treating all days as equal risk.

---

## Summary — Top Findings for FCC Use Cases

1. **Amount alone is not a reliable flag** — outlier transactions are not meaningfully riskier than typical ones.
2. **Payment Format is the single strongest signal found** — ACH transactions are dramatically over-represented in laundering cases (~87% of all laundering activity).
3. **Currency (Saudi Riyal) and specific banks show elevated risk concentration**, warranting entity-level and currency-level scrutiny alongside transaction-level rules.
4. **Weekend transactions carry higher laundering rates**, suggesting a time-based risk adjustment could improve detection coverage.
5. **Severe class imbalance (0.1% positive rate)** means any future model must be evaluated on precision/recall, not accuracy — and false positive/negative costs (covered in the Day 2 business impact analysis) must guide threshold-setting.

---

## Business Impact: False Positives vs. False Negatives

Every detection threshold trades off two error types, and this tradeoff should directly inform how any future model is tuned.

- **False Positive (FP):** a legitimate transaction incorrectly flagged as laundering — costs analyst investigation time, customer friction, and unnecessary account holds.
- **False Negative (FN):** an actual laundering transaction missed — costs regulatory exposure, reputational damage, and potential legal liability.

### Estimated cost per error

| Error type | Estimate | Basis |
|---|---|---|
| False Positive | ~$20 per flagged transaction | ~0.4 hours of analyst review time at ~$50/hour fully-loaded cost |
| False Negative | ~$10,000+ per missed case (conservative) | Investigation cost, apportioned regulatory/legal exposure, reputational risk — real systemic AML failures scale into the millions/billions, but this is a per-case floor estimate |

### Applying this to the dataset's scale

Using the actual class distribution found in this dataset (5,073,159 legitimate, 5,177 laundering), a hypothetical model with **90% recall** and a **1% false positive rate** on legitimate transactions would produce:

| Metric | Count | Estimated Cost |
|---|---|---|
| True Positives caught | ~4,659 | — |
| False Negatives missed | ~518 | ~$5,180,000 |
| False Positives raised | ~50,732 | ~$1,014,640 |

**Key insight:** even though false positives vastly outnumber false negatives in raw count, **the cost of missed laundering cases dominates the total** — roughly 5x higher than the cost of all false alerts combined. This holds even under conservative FN cost assumptions.

### Recommendation

Given this asymmetry, **the model should be tuned to prioritize recall (catching more true positives) even at the cost of a higher false positive rate**, and the business should budget for the analyst review capacity that comes with that tradeoff. A lower-recall model that looks "cleaner" on paper (fewer alerts, less analyst workload) is very likely more expensive in expectation once missed-case costs are accounted for.

This directly reinforces the class imbalance finding above: because laundering is such a rare event (0.102%), a small change in recall corresponds to a small change in raw False Negative *count*, but a large change in total cost — which is exactly why accuracy-based evaluation would be misleading for this problem.

---

- This dataset is **synthetic**, generated by IBM's AML simulator — specific rates and patterns (e.g., the exact ACH laundering rate) reflect the simulator's design assumptions and may not directly transfer to real-world transaction data. Directional patterns (e.g., "channel matters more than amount") are more trustworthy than the precise percentages.
- Bank-level findings with very low transaction counts (e.g., a bank with 91 transactions) carry high statistical uncertainty and should be treated as hypotheses for further investigation, not confirmed risk indicators.
- This analysis covers **HI-Small** only (~5M transactions, 10 days). Broader validation against HI-Medium would strengthen confidence in these findings before any operational use.
