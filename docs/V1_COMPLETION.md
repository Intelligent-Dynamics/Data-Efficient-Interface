# Bounded v1 completion — 2026-09-27

The local strengthening pass is complete: complementary-error evidence, measured single-model CPU performance, and an executable retrieved-example companion with synthetic safety/evaluation checks. **Paid pilot and final test are not run.** The user reports an earlier authorized mechanical test preflight; this pass performed no further test access, encoding, inference or scoring. No API call occurred.

## Verified observations

| Item | Measured result |
| --- | ---: |
| 20-shot specialist validation accuracy, five-seed mean | 83.7662% |
| Zero-shot Luna validation accuracy | 79.7403% |
| Hybrid validation accuracy, 90% specialist coverage | 85.2727% |
| Specialist / Luna accuracy on matching rejected subsets | 45.1948% / 60.2597% |
| Hybrid gain over matching specialist | +1.5065 accuracy points; +1.5711 macro-F1 points |
| Seed-11 warm request latency, median / p95 | 5.906833 / 7.824673 ms |
| Seed-11 batch-32 throughput | 370.1814 requests/s |
| Encoder / classifier loading | 2.424610 s / 0.629625 ms |
| Process RSS after loading / lifetime peak | 513.53125 / 582.921875 MiB |
| Cached-vector retrieval overhead, median / p95 | 0.750563 / 0.853890 ms |
| Prompt construction/isolation, median / p95 | 2.051438 / 2.270002 ms |

CPU: Apple M1 Pro, 32 GiB RAM, macOS 26.5.1, Python 3.13.0, one thread, unchanged frozen models/settings. One warmup and five timed validation passes per batching mode, with every duration retained. All saved labels and gates matched; maximum probability difference was 2.78e-7. RSS includes interpreter/libraries/evidence, sampled by `ps`; peak uses `resource.ru_maxrss`. Loading is not a cold-process guarantee. Retrieval/prompt timings use one 770-query pass with verified cached query vectors and exclude encoding. Do not add medians to claim a measured combined pipeline latency.

The validation comparison preserves every seed, 90 original combinations and negative results. Luna rescues 111 specialist-error events but harms 53 correct-answer events over overlapping rejected subsets. Five seeds reuse one validation set; seed SD is not a confidence interval. Each 20-shot model uses 1,540 fitting labels plus 770 additional validation labels and external encoder pretraining. No production latency, cloud price or total-system dollar saving is established.

Evidence: [validation replay/chart](../experiments/v1-complementary-validation/README.md), [CPU benchmark](../experiments/exp008-cpu-validation/README.md), [retrieval companion](../experiments/exp009-retrieved-luna/README.md).

## Frozen retrieved comparison and estimates

EXP-009 uses only the exact 1,540 seed-11 training IDs, unchanged normalized frozen MiniLM, cosine top-20 search, descending similarity and ascending training ID on exact ties. All 1,540 labels count, not only the 20 shown in a request. The prompt/schema/model/generation settings and seed-11 threshold are frozen. IDs, training isolation, similarity order and request/prompt hashes are retained; exact bodies stay ignored. One response set supplies standalone and hybrid scoring, including every failure. Paired comparisons use matching seed-11 controls. No prompt/threshold selection follows pilot or test scores.

Companion protocol SHA-256:
`b78f3d32a9bd86ea130ce1f51e2feded9d52b5f5ff796507c49d6f53fccc3334`

Original EXP-007 protocol remains byte-for-byte:
`0ff095f523ebc8f05d295dad4545f6a730dd07c6c9bc7ac194bbbb4a759cfa00`

| Retrieved-Luna scope | Initial requests | Historical nominal charge | Four-attempt reservation | Pacing floor |
| --- | ---: | ---: | ---: | ---: |
| Pilot, 20 frozen validation IDs | 20 | $0.004712625 | $0.22757600 | 95 s |
| Optional full validation | 770 | $0.183864500 | $8.83929200 | 64.1 min |
| Final test, conditional projection | 3,080 | $0.735458000 | $39.8490400 | 4.28 h |

**Proposed pilot-only cap: $0.26, NOT APPROVED.** Prices are inherited from 2026-09-25, not checked this session. Nominal input tokens are ceil(compact payload bytes/4), with 32 output tokens; reservation is 2*bytes+8192 input plus 128 output, four attempts, cache-write pricing and no cache-read discount. These are heuristics, not server counts. The test reservation assumes the largest validation request, 8,586 bytes; actual test payloads need later separately authorized sizing.

The unchanged original test arm covers 3,080 cases with at most 3,080 unique initial zero-shot calls, shared across five seeds. The additional arm has 3,080 retrieved calls, shared between its standalone/seed-11 hybrid evaluations: **at most 6,160 initial calls**, with retries bounded separately. The no-alias serial-pacing floor for both arms is **8 h 33 min 10 s**, excluding API response time and local work. The API latency is unknown. With one attempt spending its full 60-second timeout, the per-arm bound is 200,195 seconds (55.61 h); with all four attempts and 30/60/120-second backoff, it is 1,401,400 seconds (389.28 h), before larger Retry-After waits. These are conservative planning envelopes, not expected durations. Long waits pause for resume. If optional full validation follows the pilot, it is a separate approved collection that repeats those 20 inputs: count 790 calls and both budgets. It is not required before final evaluation.

## Exact future commands — none executed by this pass

Work in `/Users/Andrew/Developer/data-efficient-inference`. A local key, these commands or the old $28 discussion grants no authorization. Set each cap variable only to a separately approved positive amount; set each date only after verifying current official model/settings compatibility and unchanged prices. Stop on any incompatibility, rather than substituting a model or settings.

After explicit pilot-only approval (the proposed $0.26 is not yet approved):

```sh
.venv/bin/python -m baseline.retrieved_run pilot-live \
  --approved-protocol-sha256 b78f3d32a9bd86ea130ce1f51e2feded9d52b5f5ff796507c49d6f53fccc3334 \
  --authorize-live \
  --spending-cap-usd "${EXP009_PILOT_CAP_USD:?Explicitly approve a pilot-only cap}" \
  --acknowledge-model-pricing-date "${EXP009_VERIFIED_UTC_DATE:?Verify official model/settings/prices today}"
.venv/bin/python -m baseline.retrieved_run evaluate-pilot
```

Use local `OPENAI_API_KEY` and optional `OPENAI_PROJECT_ID`; never paste their values into chat, command arguments or Git. Pilot outcomes are formatting/usage/cost checks, not a reliable 77-class quality estimate or prompt-selection opportunity.

**The original loader blocker is resolved by the separate compatibility addendum below.** Obtain new original-test access/live approvals and a sufficient approved cap; preserve its original five-seed scope:

```sh
.venv/bin/python -m baseline.final_test live \
  --approved-protocol-sha256 0ff095f523ebc8f05d295dad4545f6a730dd07c6c9bc7ac194bbbb4a759cfa00 \
  --authorize-test-access --authorize-live \
  --spending-cap-usd "${EXP007_APPROVED_CAP_USD:?Explicitly approve the original test cap}" \
  --acknowledge-current-pricing-date "${EXP007_PRICING_VERIFIED_UTC_DATE:?Verify unchanged official prices today}"
```

After the original prediction bundle is complete and immutable, separately authorize companion test preparation (this opens/encodes test data, but makes no API call):

```sh
.venv/bin/python -m baseline.retrieved_run test-preflight \
  --approved-protocol-sha256 b78f3d32a9bd86ea130ce1f51e2feded9d52b5f5ff796507c49d6f53fccc3334 \
  --authorize-test-access
```

Review its exact full reservation, then obtain a separate sufficient companion cap and live approval:

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

Preserve each scope's output directory, cache and reservation ledger. Resume with the same protocol, code and cap; successful requests are not resent. The original runner does not save query embeddings, so the separate companion comparison re-encodes once; within the companion, the resulting vector is reused for classifier verification and retrieval. Report API usage/charges separately from all local compute.

## Verification and remaining blockers

Full suite: **775 passed in 49.96 s**, run once at the end via `experiments/v1-strengthening/verify_suite.py`. Real raw/processed data, artifacts, model caches and network were blocked; guard counters were zero. Independent metric/timing replay and exact offline request reconstruction passed. No raw response, dataset, weight, secret or private cache belongs in this checkpoint.

The real CPU setup exposed an existing `final_specialists._load_local_encoder` assertion that rejects SentenceTransformer 5.1.1's inert empty `query`/`document` prompt mapping. No prompt is applied (`default_prompt_name=None` and explicit no-prompt encoding). The new CPU helper accepts only empty prompt strings and verifies frozen state/settings; original source and protocol remain untouched. **Smallest remedy before original EXP-007 execution:** a narrowly reviewed assertion fix allowing inert empty mappings while still rejecting nonempty/default prompts, with synthetic regression coverage. No model, prompt, threshold or experimental-method change is needed. Do not bypass the guard or launch the original test first.

Other outstanding requirements are authorization and current official compatibility/pricing verification for future paid execution, plus actual companion test payload sizing after separately approved access. Retrieved-Luna quality and final-test quality remain unknown. No further model, dataset, tuning, calibration, GPU serving or frontend work is proposed.


## 2026-09-27 addendum — original loader compatibility blocker resolved

The blocker above records the bounded-v1 pass as it happened. A subsequent narrow fix changes only the original loader's prompt assertion: accept an empty mapping or string-keyed mapping with all empty-string values, only when `default_prompt_name is None`. It still rejects malformed/nonempty/default prompts and never clears the mapping or changes encoding settings.

The patched original loader loaded the pinned encoder offline and encoded all 770 existing validation texts once, without timing. All five saved classifiers passed integrity/settings/state checks. Maximum absolute probability and confidence differences were **0.0**, predicted-label and routing mismatches were **0**, and every seed accepted **693/770**. Encoder state remained equal to the frozen hash with zero trainable parameters. This does not replace the earlier CPU benchmark or its recorded numerical differences.

The full synthetic suite ran once: **807 passed in 52.55s**, including 32 new assertion cases. Suite and actual-loader guard counters were zero. No further official-test access, API call, pilot, preflight, final evaluation, model fitting or timing study occurred. Both EXP-007 and EXP-009 protocol hashes remain unchanged; all previous tracked experiment evidence is byte-identical. New source hashes and checks are kept separately in [compatibility evidence](../experiments/exp007-loader-compatibility/README.md). Historical verification records and runtime source bindings remain intact. The loader blocker is resolved; future execution permissions/pricing prerequisites above remain outstanding.


## 2026-09-28 UTC addendum — interrupted original collection

The later EXP-007 live collection stopped at the local cooldown guard after 509 successful responses/509 attempts, with 2,571 requests unattempted. Its saved source bindings require an explicit compatibility receipt for the narrow bounded-wait repair. Use the [cooldown compatibility write-up](../experiments/exp007-cooldown-compatibility/README.md) for the updated future resume command; the older command above alone cannot approve this source transition. No old artifact was rewritten and no live resume, model inference, raw test access or scoring occurred during the repair.
