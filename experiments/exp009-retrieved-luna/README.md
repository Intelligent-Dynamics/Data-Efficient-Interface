# EXP-009 — retrieved-example Luna companion

**Implemented and prepared offline; paid pilot and model evaluation NOT RUN.** No API calls or additional official-test access occurred. The official test had already been mechanically opened by an earlier authorized EXP-007 preflight; no test predictions or scores exist. The original EXP-007 protocol and zero-shot prompt remain unchanged.

Protocol SHA-256: `b78f3d32a9bd86ea130ce1f51e2feded9d52b5f5ff796507c49d6f53fccc3334`.

This is one bounded, seed-11/single-training-pool comparison. The candidate pool contains **only the exact 1,540 training IDs** used by the frozen 20-shot seed-11 MiniLM/LR specialist. Its **1,540 task labels plus 770 additional validation labels** are disclosed. Retrieving 20 examples per query does not reduce that training-label budget to 20. Validation/test examples and other seeds' training-only examples cannot enter the pool.

The pinned normalized float32 MiniLM embeddings are reused after checking archive hashes, ID, label and text alignment. Exact cosine search uses float64 dot products of these normalized vectors; choose exactly **k=20**, sorted by descending similarity and ascending training ID on exact ties. No index service, fitting, reranking, k sweep, prompt search, truncation or dropped query. Candidate IDs/hash, encoding revision/settings/packages and unchanged seed-11 threshold are frozen in `protocol.json`.

The static instructions contain all 77 official intent names. Each independent request supplies 20 training messages with their correct labels and the query message in distinct JSON data fields. Messages are explicitly untrusted data, never instructions. Query truth, specialist prediction/confidence, IDs and similarity scores are absent from the request. The existing strict schema, `gpt-6-luna`, `reasoning.effort=none`, 128 output tokens, default service tier, `store=false` and other generation settings are unchanged. An alias without an immutable provider snapshot remains a limitation.

The 20 pilot IDs were frozen using only IDs: ascending `(SHA256("EXP-009-pilot-v1:" + ID), ID)`, first 20. They are a formatting/usage/cost pilot, **not a reliable 77-class quality estimate**. Do not choose the prompt using pilot accuracy. `pilot.json` lists exact IDs, retrieval order/similarities, payload sizes, token heuristics and hashes. `validation_retrieval.json.gz` preserves the same compact evidence for all 770 queries, without raw texts. Exact bodies are in ignored `artifacts/exp009-retrieved-luna-v1/{pilot,validation}/requests.json`.

## Measured offline overhead

One serial pass across all 770 existing validation query vectors, BLAS threads=1. Every request performs exact retrieval, candidate-membership/text-isolation checks and prompt construction; timing excludes disk logging, hashing/serialization for the evidence report and query encoding. Query vectors came from the verified frozen cache. These measurements are retrieval/prompt overhead only, **not end-to-end model latency**.

| Step | Median ms | p95 ms | Total seconds |
| --- | ---: | ---: | ---: |
| Exact retrieval | 0.750563 | 0.853890 | 0.558757 |
| Prompt construction and isolation checks | 2.051438 | 2.270002 | 1.578803 |

`overhead.json` retains every duration, local runtime/hardware identity and the measurement method. These are descriptive measurements on this machine, not serving latency or economically free compute.

## Historical cost and execution duration estimates

Prices are inherited from **2026-09-25, not reverified this session**: input/cached-input/cache-write/output $0.10/$0.01/$0.125/$0.50 per million tokens. Before future paid execution, independently verify official model/settings compatibility and prices on the current UTC date. Stop if any differ; never substitute another model/settings. Neither prior EXP-007 spending discussion nor the commands below authorizes this experiment.

The inherited estimator uses nominal `ceil(compact UTF-8 body bytes / 4)` input and 32 output tokens; these are heuristics, not server token counts. Reservation uses `2*bytes +8192` input, the full 128 output tokens and four attempts. All input is charged at the historical cache-write rate; **no cache hits or read discounts are assumed**, including query-dependent examples. Static instructions share a prefix, but actual provider caching is unknown.

| Scope | Requests | Nominal one-attempt USD | Four-attempt reservation USD | Pacing floor, excluding request time |
| --- | ---: | ---: | ---: | ---: |
| Pilot | 20 | 0.004712625 | 0.22757600 | 95 seconds |
| Optional full validation | 770 | 0.183864500 | 8.83929200 | 3,845 seconds (64.1 minutes) |
| Future test, projection only | 3,080 | 0.735458000 | 39.8490400 conditional | 15,395 seconds (4.28 hours) |

**Proposed pilot-only cap: $0.26, NOT APPROVED.** Test reservation assumes all payloads fit the observed validation maximum, 8,586 bytes; the real test envelope must be recomputed after separate authorized access. Actual charges, API failures and unknown usage remain recorded separately from local inference costs. No dollar-savings claim follows.

Serial pacing uses five seconds after a response, four maximum attempts, 30/60/120-second exponential backoff, a 60-second timeout and Retry-After support. At every request using the full timeout, one-attempt request-plus-pacing envelopes are 1,295 seconds pilot, 50,045 seconds validation and 200,195 seconds test. With all four attempts/backoffs, the no-Retry-After duration envelopes are respectively 9,100, 350,350 and 1,401,400 seconds; headers can extend these, and waits over one hour pause for later resume. These are planning envelopes, not predicted service latency.

Original plus additional final comparison collects **3,080 zero-shot +3,080 retrieved-example requests**, reused across five original specialist policies and the one added seed-11 policy. Minimum combined serial pacing alone is about 8.55 hours, plus service time. A later optional full-validation collection is separate from the pilot and repeats its 20 inputs; if both are approved, count **790 validation/pilot requests**, not 770. This deliberately simple bounded implementation does not transfer reservation ledgers between different approved scopes.

## Safety and reproducibility

`retrieved_collection` reuses the established transport, response parser, bounded pacing/retry policy, atomic writes, process lock, durable reservation ledger, usage pricing and failure accounting. It has its own EXP-009 cache identity and single scope directories. Each attempt is durably reserved before dispatch; interrupted dispatches consume an attempt and unknown-usage reservation. Successful/terminal requests are not resent. Missing or tampered ledgers, responses, request identity, source code or cap halt resume. No retry is based on correctness.

Validation/pilot bodies must reproduce their frozen hashes. Future test inputs require separate companion approval before the existing authorized dataset loader can expose ID/text-only rows; labels never enter retrieval/model inference. Original EXP-007 outputs must be frozen and verified before the companion test path proceeds. Requests, specialist outputs and API predictions are persisted before scoring joins labels. All rows/failures are retained. The offline evaluator reports specialist, zero-shot Luna, zero-shot hybrid, retrieved Luna and retrieved hybrid; the two hybrids use the same unchanged scalar seed-11 gate. It reports all-class metrics, observed coverage, fallback counts/accuracy, paired differences, failure statuses and separate API accounting. Report null/deteriorating results as prominently as improvement; no tuning follows test scores.

The original EXP-007 runner does not persist query embeddings. The separate companion invocation therefore re-encodes test queries once and replays the unchanged seed-11 LR to verify agreement. **Within the companion, that one query vector is reused for LR and retrieval**. This extra offline comparison pass is documented overhead, not a proposed duplicate-encoding deployed hybrid.

A discovered existing EXP-007 implementation compatibility blocker remains: its local encoder loader rejects SentenceTransformer 5.1.1's built-in empty-valued prompt mappings. The companion uses the validated CPU-benchmark loader which accepts only inert empty prompts, with unchanged frozen state/settings. The original runner must receive a narrowly reviewed compatibility fix before its future live execution; its protocol and original source are untouched by this companion. No test inference was attempted to discover this.

## Commands

From the canonical repository, these two commands are offline and training/validation-only:

```sh
.venv/bin/python -m baseline.retrieved_run dry-run
.venv/bin/python -m baseline.retrieved_run prepare-validation
```

After **new explicit pilot-only approval**, numeric cap and current official model/pricing verification, the exact live pilot command is:

```sh
.venv/bin/python -m baseline.retrieved_run pilot-live \
  --approved-protocol-sha256 b78f3d32a9bd86ea130ce1f51e2feded9d52b5f5ff796507c49d6f53fccc3334 \
  --authorize-live \
  --spending-cap-usd "${EXP009_PILOT_CAP_USD:?Explicitly approve a pilot-only cap}" \
  --acknowledge-model-pricing-date "${EXP009_VERIFIED_UTC_DATE:?Verify official model/settings/prices today}"
.venv/bin/python -m baseline.retrieved_run evaluate-pilot
```

Set `OPENAI_API_KEY` locally only; optional `OPENAI_PROJECT_ID`. Never paste keys into commands, docs or chat. Optional full validation uses `validation-live` with its own separately approved cap, followed by `evaluate-validation`; it does not inherit the pilot budget.

The original final comparison remains `baseline.final_test live` with its original hash, test/live authorizations, sufficient approved EXP-007 cap and current pricing acknowledgement, as documented in the original runner bundle. **Do not execute it now**, and resolve the loader compatibility blocker first. Once its immutable original prediction bundle exists, prepare the added test comparison with separate test-access approval:

```sh
.venv/bin/python -m baseline.retrieved_run test-preflight \
  --approved-protocol-sha256 b78f3d32a9bd86ea130ce1f51e2feded9d52b5f5ff796507c49d6f53fccc3334 \
  --authorize-test-access
```

This future command opens and encodes test data and prepares all 3,080 requests, but issues zero API calls. It is **not authorized in the preparation session**. Review its exact reservation, then approve a new sufficient companion cap before:

```sh
.venv/bin/python -m baseline.retrieved_run test-live \
  --approved-protocol-sha256 b78f3d32a9bd86ea130ce1f51e2feded9d52b5f5ff796507c49d6f53fccc3334 \
  --authorize-test-access --authorize-live \
  --spending-cap-usd "${EXP009_TEST_CAP_USD:?Explicitly approve the companion test cap}" \
  --acknowledge-model-pricing-date "${EXP009_VERIFIED_UTC_DATE:?Verify official model/settings/prices today}"
.venv/bin/python -m baseline.retrieved_run evaluate-test \
  --approved-protocol-sha256 b78f3d32a9bd86ea130ce1f51e2feded9d52b5f5ff796507c49d6f53fccc3334 \
  --authorize-test-access
```

Use the same command/cap and intact output directories for resume; refresh current-date compatibility acknowledgement as needed. Do not delete ledgers or start another path to bypass an exhausted allowance. No paid or test-evaluation command above ran during preparation.
