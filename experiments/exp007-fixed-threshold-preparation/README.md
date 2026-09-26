# EXP-007 — fixed-threshold routing preparation

**Preparation complete; official test evaluation NOT RUN.** No official test file was opened, parsed, hashed or evaluated. No paid API calls, model execution, fitting, calibration, new split or new labels were used. The per-request gate is deployable as a scalar decision; its test/production quality is not yet established.

## Frozen validation thresholds

Use the original EXP-004 **20-shot** MiniLM + logistic-regression classifiers for all five seeds. The encoder revision remains `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; classifier bytes, probabilities, package/encoding settings, original sample IDs and EXP-005 confidence provenance are hash-bound. No specialist was loaded or refitted during preparation.

| Seed | Exact binary64 threshold (round-trip decimal) | Accepted validation cases | Coverage | Changed IDs vs EXP-005 rank prefix |
| --- | ---: | ---: | ---: | ---: |
| 11 | 0.10354116298070118 | 693/770 | 90% | 0 |
| 22 | 0.10243964501544885 | 693/770 | 90% | 0 |
| 33 | 0.1021902311171956 | 693/770 | 90% | 0 |
| 44 | 0.10160314104812371 | 693/770 | 90% | 0 |
| 55 | 0.10612054364389162 | 693/770 | 90% | 0 |

For these actual runs, each threshold is the confidence of the **693rd request** in the existing descending EXP-005 ranking. Each rank-694 score is strictly lower. There is no split boundary tie, so `>=` reproduces every accepted ID exactly, not merely the count. Each seed sends 77/770 validation cases to fallback under this rule. [Threshold evidence](thresholds.json) includes exact decimal and hexadecimal float representations, adjacent confidence values, tied-group counts, acceptance-set hashes and added/removed IDs.

Confidence remains **UNCALIBRATED maximum class probability**. A threshold around 0.10 is not a claim of 10% correctness or a calibrated risk guarantee. The 20-shot/90% choice was informed by earlier exploratory validation outcomes; deriving its scalar cutoff uses only IDs/confidences, not labels or correctness. The same **770 additional validation labels** have already been used in development, alongside 1,540 fitting labels per specialist. This is not a new independent validation set or new annotation budget.

## Per-request rule and ties

```python
from baseline.thresholds import route

destination = route(specialist_confidence, frozen_threshold_for_seed)
# 'specialist' exactly when specialist_confidence >= frozen_threshold_for_seed
# otherwise 'gpt-6-luna'; this helper itself makes no model/API call
```

The runtime gate accepts exact binary64 floats, rejects invalid/nonfinite inputs or implicit lower-precision conversions, and uses no labels, IDs, batch ranks, percentiles, epsilon or rounding. Retain the JSON numeric value losslessly, or load the recorded decimal/hex representation. Apply the same rule independently to future requests; changing the surrounding batch cannot change the gate for an unchanged score. Existing encoder numerical settings remain pinned separately.

EXP-005's original rank ties use SHA-256 of the UTF-8 row ID, then ID. A scalar threshold cannot split an equal-score group. The frozen preparation rule compares including all boundary ties at `q` with excluding all via the next binary64 value above `q`. It chooses the minimum acceptance-set symmetric difference from the old rank prefix; equal distances include all ties. This is a deterministic, label-blind representability decision, not a quality search. At runtime every score equal to the frozen threshold is accepted. All five actual boundaries reproduce exactly, so the tie fallback was not needed.

## Exact eventual one-time test protocol

The authoritative machine-readable [protocol](protocol.json) is frozen with SHA-256:

`0ff095f523ebc8f05d295dad4545f6a730dd07c6c9bc7ac194bbbb4a759cfa00`

1. Obtain separate explicit authorization to unseal/evaluate the official test and new paid-run approval with a numeric cap. Prior validation/recovery approvals do not transfer. Verify protocol, source, classifier, encoder and package hashes. Test access is not authorized by this preparation.
2. Use all **3,080 official test rows**, original zero-based `test:NNNNN` IDs and all 77 intents. The count and expected byte checksum come only from the existing source metadata. Retain every row, including failures; no new filtering, sampling, class exclusion, prompt demonstrations or fitting.
3. Run all five existing 20-shot specialists using their saved classifiers and unchanged frozen CPU/float32 MiniLM settings. Keep the LR configuration and probability precision unchanged. Persist predictions/probabilities and apply each seed's threshold independently. **Report observed test coverage; never rank test scores or force 90%.** No recalibration, threshold retuning, refitting or selecting a best seed.
4. Collect one Luna prediction set for **all 3,080 test IDs**, shared across all five specialists. The Luna-only comparator requires all cases, even those routed to the specialist. Exact duplicate payloads may share a compatible cache response only with explicit per-ID aliases; retain all original evaluation rows. Validation predictions are not test predictions.
5. Keep the original EXP-006 zero-shot prompt/schema and settings: `gpt-6-luna`, reasoning `none`, output cap 128, Standard/default service tier, `store=false`, `stream=false`, `truncation=disabled`, implicit prompt cache/30-minute TTL, no history/tools/examples. Send one customer text, never labels, specialist predictions/confidence or IDs. Record returned model/tier, timestamps, usage details and request timing; the alias has no documented fixed snapshot and underlying weights can change.
6. Use a new ignored `artifacts/exp007-fixed-threshold-test-v1` directory, durable cache and reservation ledger. Freeze serial requests, at least five-second gaps, four attempts maximum, exponential backoff, Retry-After handling and quota-error halts from EXP-006R. Refusals/incomplete/invalid or valid-but-wrong answers are not retried to improve correctness. Missing usage remains unknown. Recompute actual request/output/retry reservations after authorized unsealing and before any paid call; insufficient cap halts for spending review without truncating, excluding cases or changing prompts/thresholds.
7. Freeze the complete prediction artifacts before scoring. For each seed, use the specialist above/equal to threshold and the real saved Luna prediction below it. Report **specialist-only, Luna-only and routed accuracy/macro-F1 over all 3,080 rows**; macro-F1 uses the fixed 77 classes and zero-division=0. Unresolved Luna outcomes are failures wherever used. Also report accepted counts/coverage, number and percentage sent to Luna, rejected-subset fallback accuracy (`null` if none), per-class metrics/acceptance and failure statuses.
8. Preserve all five seed results and mean/sample SD (`ddof=1`) of seed-dependent quantities. Luna-only results use one shared response set. Seeds do not provide independent test populations or confidence intervals. Keep actual all-case API collection spend, hypothetical routed API charges, and unmeasured specialist deployment cost separate. No test outcome may be used to revise this frozen experiment; no non-inferiority margin, production guarantee or total-system saving is claimed.

The current implementation provides the gate, validation preparation and pure offline scorer. **It deliberately exposes no live/test-loading command.** A future authorized test runner must implement this protocol before execution; this preparation is not permission to open the test or launch calls.

## Will additional Luna calls be needed, and what will they cost?

**Yes.** Plan up to 3,080 first attempts for one full test response set, not 15,400 requests for five seeds and not merely 10% of the test set. The actual routed workload's fallback rate is unknown until the frozen thresholds are applied. The full baseline study's collection volume is a separate quantity.

Current Standard short-context pricing was rechecked on **2026-09-26** and matches the original frozen rates: input/cached-input/cache-write/output **$0.10/$0.01/$0.125/$0.50 per million tokens**, excluding taxes/credits. Sources: [model](https://developers.openai.com/api/docs/models/gpt-6-luna), [pricing](https://developers.openai.com/api/docs/pricing), [cache accounting](https://developers.openai.com/api/docs/guides/prompt-caching). Reverify before paid execution.

[Cost estimate](cost_estimate.json), derived without test text:

| Scenario | Estimated USD |
| --- | ---: |
| Observed successful validation token usage ×4; one attempt, same cache behavior | **0.3325308** |
| Same token counts; all input at cache-write pricing | **0.4083580** |
| Conservative engineering allowance, one attempt per test request | **7.06013** |
| Same allowance, all four attempts for every test request | **28.24052** |
| Preparation API spend | **0** |

The 770 completed validation responses used 758,272 input and 14,611 output tokens, with zero reported cache-read/cache-write/reasoning tokens. Their known successful-response charges are $0.0831327. This is **not original total experiment spend**: 105 original failed attempts have unknown usage. [Validation completion/usage audit](validation_completion.json) records that distinction and source hashes.

The conservative estimate assumes each test request fits the largest validation payload, **4,817 compact JSON bytes**, using `2×bytes+8192 = 17,826` input tokens plus the full 128-output cap, all input at cache-write pricing and no cache-read discount. Four attempts across 3,080 rows means at most 12,320 attempts. Neither actual test lengths nor tokenizer counts have been inspected. Therefore $28.24 is a **conditional estimate, not a guaranteed ceiling or spending authorization**. After separate test-access approval, actual preflight reservations must fit the newly approved cap before any call; otherwise stop for review. No specialist deployment price or production latency is estimated here.

## Reproduction and verification

From the canonical checkout, regenerate preparation into a fresh directory:

```sh
.venv/bin/python -m baseline.fixed_routing --output artifacts/exp007-preparation-replay
.venv/bin/python experiments/exp007-fixed-threshold-preparation/independent_check.py
.venv/bin/python -m pytest -q
```

These commands do not run test evaluation or paid inference. The preparation loader reads only existing EXP-004/005 records, validation probabilities, classifier bytes for hashing, cache metadata, dataset metadata and completed validation response evidence. Local ignored source artifacts are required to reproduce their audits; no model is loaded. Existing experiments are unchanged.

Full suite: **495 passed**, including 124 new threshold/input/scoring/protocol regressions. Tests cover exact scalar comparisons, adjacent floats, invalid scores, label-independent tie handling and minimum achievable set difference, float serialization, fixed row/label/model provenance, failure denominators, scalar-vs-batch independence and forbidden raw/test/network/model/credential access. A standard-library implementation independently reproduced all five thresholds, acceptance sets and cost calculations.

An initial guarded preparation check rejected an ordering difference between original execution IDs and the existing evaluator's metadata ID order. Exact ID populations and prediction arrays already matched; the check was corrected to allow metadata permutation while still enforcing unique/full population, exact prediction order and all provenance hashes. Regression tests cover the case. No prior artifact, score, label, threshold or model was changed to resolve it. The successful preparation ran with network, credential, raw/processed dataset and model-loading access forbidden; all guard counters remained zero. See [verification](verification.json).
