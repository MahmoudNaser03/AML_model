# EDA Report — IBM AML Transactions (HI-Small)

**Dataset:** `HI-Small_Trans.csv`
**Notebook:** `notebooks/01_eda.ipynb`
**Scope:** Week 2 — Exploratory Data Analysis
**Date:** 2026-07-20

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| Raw rows | 5,078,345 |
| Columns | 11 |
| Memory usage | ~426 MB |
| Date range | 2022-09-01 (10-day window) |

**Columns:**

| Column | Dtype | Description |
|---|---|---|
| `Timestamp` | datetime64 (after fix) | Transaction timestamp |
| `From Bank` | str (after fix) | Sending bank ID |
| `Account` | str | Sending account number |
| `To Bank` | str (after fix) | Receiving bank ID |
| `Account.1` | str | Receiving account number |
| `Amount Received` | float64 | Amount received by destination |
| `Receiving Currency` | str | Currency at destination |
| `Amount Paid` | float64 | Amount paid by source |
| `Payment Currency` | str | Currency at source |
| `Payment Format` | str | Payment channel (ACH, Wire, etc.) |
| `Is Laundering` | int64 | Target label (0 = legit, 1 = laundering) |

---

## 2. Missing Values

No missing values across any column.

| Column | Null Count |
|---|---|
| All columns | 0 |

The dataset is complete — no imputation is needed.

---

## 3. Duplicate Rows

| Check | Count |
|---|---|
| Duplicates found | 9 |
| Duplicates after drop | 0 |

Nine exact duplicate rows were detected and removed (first occurrence kept). The cleaned dataset has **5,078,336 rows**.

---

## 4. Class Balance

| Label | Count | Percentage |
|---|---|---|
| Legitimate (0) | 5,073,159 | 99.898% |
| Laundering (1) | 5,177 | 0.102% |

The dataset is severely imbalanced — laundering cases represent roughly **1 in 980** transactions. A naive model that always predicts "not laundering" would achieve ~99.9% accuracy while catching zero real cases. **Accuracy is not a usable evaluation metric** — future models must be evaluated on precision, recall, F1, or area under the PR curve.

---

## 5. Data Type Corrections

Several columns had incorrect types after loading:

| Column | Original Type | Corrected Type | Reason |
|---|---|---|---|
| `Timestamp` | object (str) | datetime64 | Needed for temporal analysis |
| `From Bank` | int64 | str | Bank IDs are identifiers, not numeric values |
| `To Bank` | int64 | str | Same as above |

---

## 6. Sanity Checks on Amounts

| Check | Amount Paid | Amount Received |
|---|---|---|
| Negative values | 0 | 0 |
| Zero values | 0 | 0 |
| Min | 0.000001 | 0.000001 |
| Max | ~$1.05T | ~$1.05T |

No negative or zero-amount transactions exist. The maximum value of ~$1.05T is a dataset extreme and should be treated cautiously in modeling.

**Descriptive statistics for amounts:**

| Stat | Amount Paid | Amount Received |
|---|---|---|
| Mean | $4.51M | $5.99M |
| Median (50%) | $1,415 | $1,411 |
| 25th pct | $184 | $183 |
| 75th pct | $12,298 | $12,346 |
| Max | $1.05T | $1.05T |

The large gap between the median (~$1,400) and mean (~$4.5M) indicates heavily right-skewed distributions with extreme high-value outliers.

---

## 7. Account Identity Engineering

Raw account numbers (`Account`, `Account.1`) are not globally unique — the same account string can appear at different banks. A composite identifier was created:

```
Sender_ID   = Account + "_" + From Bank
Receiver_ID = Account.1 + "_" + To Bank
```

| Identifier | Unique Count |
|---|---|
| Raw `Account` | 496,995 |
| `Sender_ID` (composite) | 496,999 |

The 4-count difference confirms that a small number of account numbers collide across banks, and the composite ID is the correct granularity for entity-level analysis.

---

## 8. Outlier Analysis (IQR Method)

IQR outlier detection was applied to `Amount Paid`:

| Statistic | Value |
|---|---|
| Q1 | $184.48 |
| Q3 | $12,298.15 |
| IQR | $12,113.67 |
| Lower bound | -$17,986.03 (no practical effect — no negatives) |
| Upper bound | $30,468.66 |

All transactions with `Amount Paid > $30,469` are flagged as IQR outliers. The outlier analysis broken down by payment format (from the business report) shows that **being an amount outlier does not increase laundering rate** within any format — in several formats the outlier laundering rate is equal to or lower than the overall format rate.

---

## 9. Key Findings Summary

| Finding | Detail |
|---|---|
| Severe class imbalance | 0.1% laundering rate — accuracy is not a valid metric |
| No data quality issues | No nulls, no negatives, no zeros; 9 duplicates removed |
| Amount distributions are right-skewed | Median ~$1,400 but mean ~$4.5M due to outliers |
| Amount outliers are not riskier | IQR outliers have similar or lower laundering rates per format |
| ACH is the highest-risk payment format | ~0.75% laundering rate, 87% of all laundering cases |
| Saudi Riyal has elevated currency-level risk | ~4x higher rate than next-highest currency |
| Weekend transactions show higher laundering rates | Sunday ~0.30%, vs weekdays ~0.10% |
| Small number of banks are disproportionately risky | Some banks have laundering rates up to 27.5% (low volume — treat with caution) |

---

## 10. Figures Generated

| Figure | File | What it Shows |
|---|---|---|
| Amount histogram (log-scale) | `figures/01_amount_histogram_log.png` | Overall amount distribution on log scale |
| Amount dist: legit vs laundering | `figures/02_amount_dist_legit_vs_laundering.png` | Overlaid amount distributions by label |
| Laundering rate by payment format | `figures/03_laundering_rate_by_payment_format.png` | ACH dominance in laundering |
| Currency distribution | `figures/04_currency_distribution.png` | Transaction counts by currency |
| Laundering rate by currency | `figures/05_laundering_rate_by_currency.png` | Saudi Riyal elevated rate |
| Transactions by hour | `figures/06_transactions_by_hour.png` | Volume pattern across 24 hours |
| Laundering rate by hour | `figures/07_laundering_rate_by_hour.png` | Intraday laundering risk pattern |
| Laundering rate by weekday | `figures/08_laundering_rate_by_weekday.png` | Weekend elevated laundering rate |

---

## 11. Data Saved

Cleaned dataset saved to:

```
data/processed/HI-Small_Trans_cleaned.csv
```

Changes applied vs raw:
- 9 duplicate rows removed
- `Timestamp` cast to datetime64
- `From Bank` / `To Bank` cast to string
- `Sender_ID` and `Receiver_ID` composite columns added

---

## 12. Limitations

- The dataset is **synthetic** (IBM AML simulator) — precise rates reflect simulator design, not real-world distributions. Directional findings are more trustworthy than specific percentages.
- Covers only the **HI-Small** subset (~5M transactions, 10 days). Results should be validated against HI-Medium before any operational use.
- Bank-level findings with very low transaction volumes carry high statistical uncertainty and should be treated as hypotheses, not confirmed risk indicators.
