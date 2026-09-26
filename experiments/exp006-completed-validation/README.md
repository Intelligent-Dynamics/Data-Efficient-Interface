# Completed EXP-006 + EXP-006R validation evidence

**Verified completed validation collection and recovery; official test remains sealed.**
This is explicitly the original EXP-006 plus a separately authorized EXP-006R recovery, not a rewritten original experiment. The untouched original ended with 718 successful responses and 52 HTTP-429 outcomes. Recovery supplied 52 successful predictions for those IDs only. The merged set contains exactly 770 successful predictions and zero unresolved final requests.

The saved full merged evaluation was reproduced **exactly**, including all 15 specialist configurations, 90 coverage combinations, seed aggregates and API-component accounting. Every original artifact byte is unchanged; its inventory hash remains `bfd8ad140577004bb436a4c89a977721a3286dd358470fa05e5cf1351037a7cb`. Exporting and verifying this compact evidence made no API calls, loaded no models, and accessed no raw dataset or official-test file.

## Verified validation quality

GPT-6 Luna alone correctly classified **614/770** requests: **79.7403% accuracy** and **79.0138% macro-F1**, over all 77 intents. Its one response set is reused across every specialist budget and seed.

The table shows means ± **sample SD** across seeds 11, 22, 33, 44 and 55. Accuracy/F1/SD values are percentage points. The fallback accuracy uses only rejected requests, including any unresolved responses as errors. At 100% specialist coverage it is undefined.

| Shots/class | Actual specialist coverage | Accepted / fallback | Combined accuracy (%) | Combined macro-F1 (%) | Rejected-subset Luna accuracy (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 | 25.065% | 193 / 577 | 79.97 ± 0.25 | 79.35 ± 0.29 | 75.91 ± 0.25 |
| 5 | 50.000% | 385 / 385 | 80.49 ± 0.59 | 80.08 ± 0.59 | 73.09 ± 0.68 |
| 5 | 75.065% | 578 / 192 | 79.40 ± 1.30 | 78.89 ± 1.30 | 70.10 ± 2.10 |
| 5 | 90.000% | 693 / 77 | 77.14 ± 1.49 | 76.09 ± 1.54 | 63.38 ± 4.72 |
| 5 | 100.000% | 770 / 0 | 74.13 ± 1.30 | 72.60 ± 1.34 | undefined |
| 10 | 25.065% | 193 / 577 | 80.36 ± 0.21 | 79.70 ± 0.24 | 74.56 ± 0.26 |
| 10 | 50.000% | 385 / 385 | 82.00 ± 0.69 | 81.69 ± 0.76 | 70.75 ± 0.68 |
| 10 | 75.065% | 578 / 192 | 83.35 ± 0.85 | 83.01 ± 0.85 | 67.19 ± 1.43 |
| 10 | 90.000% | 693 / 77 | 82.05 ± 0.89 | 81.47 ± 0.96 | 61.30 ± 2.50 |
| 10 | 100.000% | 770 / 0 | 80.05 ± 0.93 | 79.19 ± 1.03 | undefined |
| 20 | 25.065% | 193 / 577 | 80.60 ± 0.24 | 79.96 ± 0.28 | 74.35 ± 0.25 |
| 20 | 50.000% | 385 / 385 | 82.57 ± 0.36 | 82.26 ± 0.36 | 68.47 ± 0.79 |
| 20 | 75.065% | 578 / 192 | 85.06 ± 0.38 | 84.86 ± 0.36 | 62.40 ± 1.70 |
| 20 | 90.000% | 693 / 77 | 85.27 ± 0.27 | 84.92 ± 0.38 | 60.26 ± 3.26 |
| 20 | 100.000% | 770 / 0 | 83.77 ± 0.66 | 83.35 ± 0.68 | undefined |

At 20 shots, the observed 90% prefix gives **85.27 ± 0.27% accuracy** and **84.92 ± 0.38% macro-F1**; Luna's accuracy on the remaining 77 cases is **60.26%** on average. The nearby 75.065% prefix gives 85.06% accuracy and 84.86% macro-F1. This small difference is descriptive, not a statistically established superiority or production threshold. The 5-shot 90% combination falls below Luna alone; the complete evidence retains that negative outcome.

All combinations and individual seed metrics are in `quality_summary.json`, including the all-fallback reference and the full-specialist reference. Hashes bind the exact accepted/rejected ID sets. The versioned EXP-005 records supply those sets and all validation truths for reproduction. No confidence threshold, prompt, specialist prediction or evaluation definition was changed for collection or this export.

These are exploratory measurements on the already reused **770-case development validation set**. Five seeds share the same validation cases and Luna predictions; their SD describes specialist training-sample variation, not five independent validation datasets or Luna replications. Each specialist used 385/770/1,540 fitting labels plus 770 additional validation labels, depending on training budget. This export adds zero labels. No official-test result, non-inferiority claim, production quality guarantee or total-system savings follows.

## Observed API accounting

| Collection | Recorded attempts | Usage-priced charges (USD) | Attempts lacking usage |
| --- | ---: | ---: | ---: |
| Original EXP-006 | 823 | 0.0775395 | 105 |
| Separate EXP-006R | 52 | 0.0055932 | 0 |
| Combined collection | 875 | 0.0831327 | 105 |

The original 105 HTTP-429 attempts remain in the record, including one followed by a successful original retry. **Exact combined API spend is unknown** because those attempts supplied no usage. Under the frozen reservation assumptions, the retained interval is **$0.0831327–$0.31921220**. The recovery's fully observed usage-priced charge is **$0.0055932**. These are usage-based charges, not invoice reconciliation; unknown usage is never treated as free.

Known usage totals are 758,272 input tokens and 14,611 output tokens across both collections; returned cached-input, cache-write and reasoning token counts are zero for the observed usage records. Frozen pricing was $0.10 input, $0.01 cached input, $0.125 cache writes and $0.50 output per million tokens, sourced from the [official pricing page](https://developers.openai.com/api/docs/pricing) on 2026-09-25. `pricing.json` preserves the original assumptions rather than replacing them with a later price quote.

`accounting.json` separately retains hypothetical routed **API-component** projections for all 90 combinations. They replay recorded original/recovery attempts and observed cache behavior only for rejected IDs; routing could change actual cache behavior. Those projections are not additional experiment spend. Specialist deployment cost, total-system savings and production latency remain unmeasured/null.

## Compact evidence and reproduction

- `predictions.json`: all 770 dataset IDs, final predicted intent/status, prediction source and original/recovery terminal statuses. No customer text or provider request ID.
- `accounting_attempts.json`: every original and recovery attempt's stage, dataset ID, status, token usage, returned model/tier and reservation/usage-priced charge. No raw requests/responses or credentials.
- `quality_summary.json`: complete standalone metrics including per-class results, all 90 combination metrics, fallback metrics and five-seed aggregates.
- `accounting.json` and `pricing.json`: independently recomputable attempt accounting, token totals and clearly labeled API-component projections.
- `verification.json`: original/recovery protocol hashes, source tree/file hashes, collection code provenance, compact artifact hashes and unchanged-source checks.
- `reproduce.py`: offline reproduction from these compact artifacts and the existing versioned EXP-005 records. It cannot issue API calls, retrain models, or read raw/test data.

Run from the repository root:

```sh
.venv/bin/python experiments/exp006-completed-validation/reproduce.py
```

Optional regeneration into a new directory:

```sh
.venv/bin/python experiments/exp006-completed-validation/reproduce.py --output /private/tmp/exp006-completed-reproduced
```

The script recomputes all standalone/combined scores, accepted/rejected-set hashes, seed means/sample SDs, per-attempt usage prices and accounting, then requires exact equality with the committed summaries. Raw response caches, original classifier/embedding artifacts and test data are unnecessary for that reproduction. The full local audit additionally validated the saved raw response/cache and reservation provenance before export; its file and tree hashes remain recorded.
