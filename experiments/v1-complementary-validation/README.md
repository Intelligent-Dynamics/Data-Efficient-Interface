# Complementary routing: independently verified saved validation evidence
This is an offline reanalysis of EXP-004/005 and completed EXP-006 + EXP-006R evidence. No model is fitted or run; no raw dataset, private response cache, model weight, embedding cache or official-test file is accessed. The original evidence is unchanged. The official test was mechanically opened by an earlier authorized preflight; no test predictions or quality scores have been produced. This reanalysis adds no test access and no API calls.

![Validation quality versus fallback fraction](quality_vs_fallback_fraction.png)

The x-axis is the fraction of requests using Luna, **not dollar savings**. All five seed means and sample SDs are shown for all three training budgets; the matching 20-shot seed-11 curve is included for the forthcoming single-pool retrieved-example comparison. Lines connect existing landmarks only; they do not assert measured results for intermediate thresholds. Only the 90% specialist-coverage landmark has frozen per-seed scalar thresholds.

## Independently verified comparison

The 20-shot specialist alone averages **83.7662 ± 0.6622% accuracy**; zero-shot Luna alone achieves **79.7403%** (614/770). The existing hybrid, with 693 specialist decisions and 77 Luna decisions per seed, averages **85.2727 ± 0.2693% accuracy**. The hybrid improves accuracy over its matching specialist by **1.5065 ± 0.6660 percentage points**, and macro-F1 by **1.5711 ± 0.7424 points**. All five paired gains are positive. These are reused-validation findings, not independent final-test or production measurements.

Luna is less accurate than the specialist overall but makes complementary errors: on each seed's exact 77 rejected requests, mean specialist accuracy is **45.1948 ± 4.8069%** and Luna accuracy is **60.2597 ± 3.2597%**. The comparison uses the same rejected IDs within each seed, not different difficulty groups.

| Seed | Specialist correct / 770 | Hybrid correct / 770 | Specialist correct / 77 rejected | Luna correct / 77 rejected | Both correct | Both wrong | Luna rescues | Luna harms | Net added correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 647 | 655 | 35 | 43 | 23 | 22 | 20 | 12 | +8 |
| 22 | 640 | 654 | 32 | 46 | 22 | 21 | 24 | 10 | +14 |
| 33 | 640 | 658 | 32 | 50 | 26 | 21 | 24 | 6 | +18 |
| 44 | 646 | 659 | 34 | 47 | 22 | 18 | 25 | 12 | +13 |
| 55 | 652 | 657 | 41 | 46 | 28 | 18 | 18 | 13 | +5 |

Across the five repeated model/case evaluations, Luna rescues 111 specialist errors and harms 53 previously correct specialist answers on rejected requests: net +58 correct prediction events. These 385 rejected prediction events are overlapping subsets of the same 770 validation requests, not 385 independent new requests. Both-right/both-wrong counts are preserved alongside improvements.

| Seed | Specialist accuracy (%) | Specialist macro-F1 (%) | Hybrid accuracy (%) | Hybrid macro-F1 (%) | Accuracy gain (pp) | Macro-F1 gain (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 84.0260 | 83.6608 | 85.0649 | 84.6141 | +1.0390 | +0.9533 |
| 22 | 83.1169 | 82.6533 | 84.9351 | 84.4668 | +1.8182 | +1.8135 |
| 33 | 83.1169 | 82.7323 | 85.4545 | 85.1922 | +2.3377 | +2.4599 |
| 44 | 83.8961 | 83.4131 | 85.5844 | 85.3764 | +1.6883 | +1.9632 |
| 55 | 84.6753 | 84.2908 | 85.3247 | 84.9563 | +0.6494 | +0.6654 |

More fallback is not necessarily better. The 20-shot 50%-fallback combination averages only 82.57% accuracy, below its 83.77% specialist-only mean. The 5-shot 10%-fallback combination averages 77.14%, below Luna-only 79.74%. All 90 original budget/seed/coverage combinations, including unfavorable outcomes, are reproduced in `summary.json`; no best-seed selection or deletion of negative evidence is involved.

## Labels and uncertainty

Each 20-shot specialist uses **1,540 fitting labels plus 770 additional development-validation labels**. The union across five 20-shot seeds is 5,394 fitting labels. All 10,003 source-training labels were mechanically read by the original audit/stratification pipeline; the encoder also benefits from external pretraining. This reanalysis adds zero unique labels. The retrieved-example companion's 1,540-label pool is the matching seed-11 pool, not a 20-label budget or the five-seed union.

Sample SD (`ddof=1`) describes variation across training subsets on one reused validation set and one shared Luna response set. It is not a confidence interval, uncertainty over new traffic, or five independent holdouts. The frozen thresholds were derived from this validation data; apparent advantages remain exploratory. Confidence is uncalibrated. No total-system savings, cloud price, production latency or final-test result follows from this figure.

## Reproduction

From the repository root:

```sh
.venv/bin/python experiments/v1-complementary-validation/reproduce.py
```

To regenerate the compact summary and both chart formats into a fresh directory:

```sh
MPLCONFIGDIR=/private/tmp/v1-matplotlib .venv/bin/python \
  experiments/v1-complementary-validation/reproduce.py \
  --output /private/tmp/v1-complementary-validation-reproduced
```

`reproduce.py` implements accuracy/macro-F1 independently using counters and full class denominators; it does not import the earlier evaluator, dataset loader or API runner. It checks original compact evidence hashes, exact ID/label/prediction agreement across EXP-004/005/006, confidence ordering, all 90 previously reported combinations, and the exact binary64 thresholds/693-ID acceptance sets. It retains every seed, all metric denominators and paired outcomes. `verification.json` records the guarded execution and output hashes. Synthetic unit tests exercise unresolved full-denominator metrics, harms as well as rescues, and alignment rejection. Root project verification runs the full suite once at the end.
