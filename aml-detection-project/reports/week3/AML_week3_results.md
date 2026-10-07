# AML General Detector + Pattern Specialist — Results

Generated: 2026-08-24 10:40:25

## Experiment design

- Model A: normal + ALL laundering transactions.
- Model B: normal + TRUE TXT-patterned laundering; unpatterned excluded from training.
- External Pattern TXT used only for grouping/evaluation — never as model features.
- Leakage-safe chronological split; pattern components kept whole.

## Dataset

- Total rows: **5,078,345**
- Positives: **5,177**

## Validation — 15% precision floor

- Model A | threshold=0.16650425 | precision=0.1200 | recall=0.7545
- Model B | threshold=0.01541328 | precision=0.1200 | recall=0.9748

## Complete TEST — Independent Model Evaluation

| model                        |       AP |   ROC_AUC |   precision |   recall |       F1 |   alerts |
|:-----------------------------|---------:|----------:|------------:|---------:|---------:|---------:|
| Model A — General AML        | 0.525189 |  0.979644 |   0.110094  | 0.760474 | 0.192342 |     8738 |
| Model B — Pattern Specialist | 0.534015 |  0.909436 |   0.0830335 | 0.693281 | 0.148305 |    10562 |

## TEST patterned/unpatterned breakdown

| group                  |    rows |   laundering |   Model A recall |   Model B recall |   Model A alerts |   Model B alerts |
|:-----------------------|--------:|-------------:|-----------------:|-----------------:|-----------------:|-----------------:|
| Normal                 | 1014569 |            0 |       nan        |       nan        |             7776 |             9685 |
| Unpatterned laundering |     459 |          459 |         0.444444 |         0.230937 |              204 |              106 |
| Patterned laundering   |     806 |          806 |         0.940447 |         0.956576 |              758 |              771 |

## Features
- Total feature count: **139**
- TXT/pattern-derived model features: **0**
