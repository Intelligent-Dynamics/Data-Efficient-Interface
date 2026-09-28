# EXP-007 — completed official BANKING77 test

The frozen 20-shot MiniLM + logistic-regression specialist and zero-shot `gpt-6-luna` make complementary errors. Applying each seed's frozen scalar confidence threshold improves mean held-out test accuracy by **1.051948 percentage points** over specialist-only, with **9.012987%** of requests falling back to Luna. The gain is positive for all five seeds. No threshold, model, prompt, schema, retry limit or evaluation definition changed after seeing test results.

## Verified results

All scores use the same **3,080 official test cases**, all 77 classes, and one shared resolved Luna prediction set. Specialist and hybrid values are five-seed mean ± sample SD; SD is expressed in percentage points.

| Model | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Luna-only | 81.363636% | 80.585867% |
| Specialist-only | 85.435065% ± 0.317784 | 85.190044% ± 0.328684 |
| Hybrid | 86.487013% ± 0.435235 | 86.298575% ± 0.448328 |

Mean specialist coverage: **90.987013% ± 0.329989 percentage points**. Mean fallback count: **277.6 ± 10.163661**, or **9.012987% ± 0.329989 percentage points**. These are observed independent-threshold outcomes; coverage was not forced to 90% by ranking the test set.

| Seed | Specialist accuracy / macro-F1 (%) | Hybrid accuracy / macro-F1 (%) | Specialist accepted | Luna fallback | Fallback accuracy (%) | Accuracy gain (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 85.584416 / 85.347125 | 86.525974 / 86.389275 | 2812 | 268 (8.701299%) | 57.462687 | +0.941558 |
| 22 | 85.097403 / 84.794841 | 86.266234 / 86.066432 | 2810 | 270 (8.766234%) | 53.333333 | +1.168831 |
| 33 | 85.259740 / 85.014634 | 86.071429 / 85.843303 | 2805 | 275 (8.928571%) | 53.818182 | +0.811688 |
| 44 | 85.324675 / 85.137002 | 86.363636 / 86.174694 | 2798 | 282 (9.155844%) | 55.673759 | +1.038961 |
| 55 | 85.909091 / 85.656617 | 87.207792 / 87.019170 | 2787 | 293 (9.512987%) | 55.290102 | +1.298701 |

Luna is weaker overall on this workload, yet more accurate on the specialist's rejected subset: mean rejected-subset accuracy is **55.115613%** for Luna versus **43.466706%** for the corresponding specialist. This supports complementary routing, not a universally stronger or frontier-model claim. The paired accuracy gain has seed SD **0.190150 pp**; macro-F1 gain is **1.108531 pp**. Validation findings remain separate: 770-case validation specialist/Luna/hybrid accuracy was 83.7662%/79.7403%/85.2727% at 90% specialist coverage.

## Accounting and failures

All **3,080** final outputs are resolved. There were **3,081** recorded attempts: 3,080 HTTP-200 successes and one HTTP-503 failure, subsequently retried successfully. That failed attempt has no returned usage. Known usage prices to **$0.332622650** at the frozen 2026-09-25 rates; its retained unknown-charge reservation is **$0.0022435**, yielding **$0.332622650–$0.334866150** under the frozen accounting envelope. Exact invoice spend is unknown; `actual_api_spend_usd` remains null. The original $28 cap, $27.696146 full reservation, and $6.926280 committed reservations are safeguards, not additional charges.

This experiment collected a Luna baseline for **all 3,080 requests**, reused across five seeds. The ~9% fallback fraction describes the routed policy, not the number of requests paid for during baseline collection. Specialist deployment cost and total production savings/latency remain unmeasured.

## Compact evidence and reproduction

- `counts.json`: sparse confusion counts, accepted class counts and rejected-pool complementary errors; no row text, IDs or response bodies.
- `summary.json`: all five seeds, complete aggregates, coverage/fallback counts and sample SD.
- `per_class.json`: precision/recall/F1 and counts for Luna, each specialist and hybrid, plus per-class acceptance.
- `accounting.json`: sanitized usage aggregates, the unknown-attempt reservation, status counts and accounting verification.
- `provenance.json` and `verification.json`: original source/freeze/checkpoint hashes, executed code provenance and independent checks.

Default replay needs only versioned compact files and Python's standard library:

```sh
python3 -B experiments/exp007-official-test/reproduce.py
```

With the existing private local artifacts, `python3 -B experiments/exp007-official-test/export.py` checks the full prediction freeze, all saved probability argmaxes/scalar gates, each routed row and the saved evaluation labels, then requires compact outputs to match. It never reopens raw test data or performs model inference. `audit_accounting.py` independently rechecks saved request/response hashes, outcomes, usage and journal counts; it writes only a **new** `/private/tmp/exp007-accounting-recheck.json` and refuses to overwrite one already present. It accepts a later documentation-only Git HEAD while requiring the exact recorded runtime/environment; no live resume binding is changed. Do not use the paid runner to reproduce this checkpoint.

Raw request text/bodies, provider request IDs, API responses/credentials, weights, embeddings and large original artifacts remain ignored and uncommitted. Earlier experiment files and original live artifacts are unchanged. Frozen EXP-007 protocol: `0ff095f523ebc8f05d295dad4545f6a730dd07c6c9bc7ac194bbbb4a759cfa00`.

## Limits and next frozen comparison

Each specialist uses **1,540 fitting labels plus 770 development labels** and external encoder pretraining; 3,080 official labels are used only for scoring. Five training seeds share one test population and one Luna response set. Seed SD is not a confidence interval over new traffic. These are BANKING77 benchmark results, not production reliability or dollar-savings measurements. No tuning is authorized from these results.

Retrieved-example **EXP-009 test results are pending**. The exact next command, requiring separate future test/inference authorization, is:

```sh
.venv/bin/python -m baseline.retrieved_run test-preflight \
  --approved-protocol-sha256 b78f3d32a9bd86ea130ce1f51e2feded9d52b5f5ff796507c49d6f53fccc3334 \
  --authorize-test-access
```

Unlike EXP-007's earlier sizing-only preflight, this EXP-009 command **opens and encodes test inputs**, replays the frozen seed-11 classifier/gates, retrieves 20 demonstrations from its existing 1,540 training examples, and prepares the 3,080 requests and reservation. It makes zero API calls. It was **not run** during this checkpoint. Paid EXP-009 execution still needs its own live authorization, current pricing acknowledgement and sufficient cap.
