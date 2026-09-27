# EXP-007 guarded runner — implementation only

**Historical runner-preparation evidence: no test access or paid calls occurred during implementation.**

**2026-09-27 status update:** the user reports a subsequent authorized mechanical preflight opened the official test, but no predictions or scores have been produced. The bounded v1 CPU/retrieval pass supersedes immediate live execution. No further test access or paid call is authorized. The protocol and runner remain unchanged; the estimates below describe original preparation assumptions, not a new authorization. This checkpoint implements the already frozen protocol; it contains no test predictions, scores or measured test request sizes. All new execution tests use clearly synthetic fixtures. Existing research artifacts and protocol bytes are unchanged.

Authoritative protocol: [`../exp007-fixed-threshold-preparation/protocol.json`](../exp007-fixed-threshold-preparation/protocol.json), SHA-256:

`0ff095f523ebc8f05d295dad4545f6a730dd07c6c9bc7ac194bbbb4a759cfa00`

## Boundaries and execution sequence

`baseline.final_test` exposes `dry-run`, `preflight` and `live`; omitted mode means `dry-run`. There is no alternate model, protocol, threshold, test path, output directory, retry policy, coverage target or fitting option.

- `dry-run` reads versioned protocol/cost metadata only. It does not read/stat/hash the official test, load model weights, read credentials or make API calls.
- `preflight` requires the exact approved protocol hash and explicit test-access authorization. It requires no live/API authorization, API key, spending cap or current-price acknowledgement. It computes the complete reservation under frozen prices; an optional positive numeric cap is compared and reported, including when insufficient. It cannot execute inference, collect responses or score results.
- `live` requires all five controls before test access: the exact approved protocol hash, explicit test-access authorization, explicit live/API authorization, positive numeric USD cap and current UTC date acknowledgement that official rates still match the frozen prices. Its full-reservation cap check fails closed before any paid call. An offline preflight does not authorize a later live run.
- Frozen preparation code, source revision/checksum metadata, all encoder/classifier files and recorded package versions are verified before opening test data. Authorized preflight then hashes and parses the same CSV bytes, checks all 3,080 rows and 77 labels, and retains exactly `test:00000` through `test:03079`. It neither filters nor samples.
- The loader mechanically checks labels but discards them from inference inputs. Only dictionaries containing `id` and `text` may enter specialist inference or API collection. The scoring-only loader cannot release truth until prediction artifacts are frozen.
- Preflight constructs every exact frozen request, records payload sizes and computes the full four-attempt reservation requirement without allocating paid attempts. Exact duplicate payloads may share a response with explicit per-ID aliases; all 3,080 evaluation rows remain. Live recomputes and enforces the complete approved reservation before any call. Requests outside frozen pricing scope fail closed; an insufficient live cap stops without truncation or partial collection.
- The pinned local MiniLM encoder runs frozen on CPU with the original float32 settings. Only saved classifiers are loaded; neither encoder nor logistic regression is fitted. Hashes, packages, classifier parameters, encoder state and frozen weights are checked before/after inference. Probabilities, class order, predictions, scalar `>=` decisions and separate embedding/classifier timings are durable before paid collection.
- Every seed applies its original scalar threshold independently. No request ranking, percentile calculation or forced 90% test coverage exists. Equal-to-threshold values are accepted using the exact binary64 value.
- Live collection uses one shared all-case Luna response set. The frozen zero-shot prompt/schema/settings and four-attempt serial retry policy remain unchanged. Refusals, invalid/incomplete answers and valid wrong answers are not retried for correctness. Exponential backoff, five-second minimum gaps, Retry-After waits and quota-error halts match EXP-006R.
- Each attempt is reserved and fsynced before dispatch; raw responses and an independent reservation ledger allow safe resume. Missing usage stays unknown. Successful/terminal responses are not resent. Interrupted dispatches consume an attempt and reservation because their server outcome is unknown. A final fourth-attempt interruption is retained as a failure, not granted a fifth attempt.
- A completed collection is independently replayed from saved caches and ledger before freezing all prediction artifacts. A paused/fatal/incomplete collection is not scored. Once all requests are terminal, scoring joins truth by ID and retains failures in full-denominator metrics. An atomic evaluation checkpoint lets interrupted publication resume without new inference or another truth join.

### Authorization-scope review

The initial runner unnecessarily coupled offline preflight to API approval. The frozen protocol's `fallback.preflight` already specifies separately authorized unsealing to compute reservations before paid calls. Standalone `preflight` now uses that test-inspection scope. The approvals in `one_time_sequence[0]` continue to govern the actual live evaluation sequence: `live`, scoring access and direct collection still enforce the complete live authorization. This is an implementation permission clarification; no protocol byte, threshold, model, prompt/schema, retry policy or evaluation definition changed. A preflight report grants no API permission and is not evidence that current prices were checked.

Outputs stay under the single ignored directory `artifacts/exp007-fixed-threshold-test-v1`. Symlink redirection and manifest/code/cap/model/request drift are rejected. The root manifest binds all runtime source hashes; collection additionally records Git revision/environment. Review and commit the runner before a future launch; do not change code or Git revision midway through collection. Resume with the same command and cap, refreshing the current UTC pricing acknowledgement when necessary. Do not delete caches, reservation ledgers or experiment directories to bypass an exhausted allowance or failed check.

Scoring reports all five specialists, one Luna baseline, all five routed results, accuracy/macro-F1, observed coverage, fallback counts/percentages and rejected-subset accuracy, per-class quality/acceptance, unresolved statuses, and sample SD across seeds. Seeds share one test population and Luna response set. Actual study collection accounting remains separate from any hypothetical routed API component; specialist deployment cost and total-system savings are unmeasured. The prior 770 validation labels remain disclosed in addition to each specialist's 1,540 fitting labels.

## Commands

Run from `/Users/Andrew/Developer/data-efficient-inference`.

Safe now, without any unsealing or live authorization:

```sh
.venv/bin/python -m baseline.final_test dry-run
```

**Do not execute preflight until separate explicit test-access approval.** No spending approval or API authorization is needed to inspect the required cap.

Future authorized preflight (opens test data; no model inference or API calls):

```sh
.venv/bin/python -m baseline.final_test preflight \
  --approved-protocol-sha256 0ff095f523ebc8f05d295dad4545f6a730dd07c6c9bc7ac194bbbb4a759cfa00 \
  --authorize-test-access
```

Optionally add `--spending-cap-usd <positive-number>` to compare a proposed cap. `required_cap_usd` always reports the full requirement; `cap_sufficient` is `null` with no cap, or a boolean when supplied. This comparison is informational and never authorizes spending.

**Do not execute live until separate explicit test-access and spending approval.** Set `EXP007_APPROVED_CAP_USD` to the approved numeric cap and `EXP007_PRICING_VERIFIED_UTC_DATE` to the current UTC date only after verifying that official prices still match the frozen assumptions. The acknowledgement is an operator attestation, not an automated pricing lookup. If prices changed, stop for protocol/spending review instead of editing the frozen protocol or silently changing rates.

Future authorized collection/evaluation, or resume of that same run:

```sh
.venv/bin/python -m baseline.final_test live \
  --approved-protocol-sha256 0ff095f523ebc8f05d295dad4545f6a730dd07c6c9bc7ac194bbbb4a759cfa00 \
  --authorize-test-access --authorize-live \
  --spending-cap-usd "${EXP007_APPROVED_CAP_USD:?Set the explicitly approved numeric cap}" \
  --acknowledge-current-pricing-date "${EXP007_PRICING_VERIFIED_UTC_DATE:?Verify unchanged official prices today}"
```

Only live collection reads `OPENAI_API_KEY`; `OPENAI_PROJECT_ID` is optional. Never put either credential value in chat, command arguments or Git. The endpoint is frozen; environment overrides do not replace it. Neither a key nor these documented flags grants user permission.

## Cost before unsealing

The sealed dry run still reports the original conditional **$28.24052000** four-attempt envelope (up to 12,320 attempts), or **$7.06013000** for one attempt. The historical nominal projection is **$0.3325308**. These are unchanged frozen estimates, not newly measured test costs or spending approval.

The envelope assumes every payload is at most the validation maximum of 4,817 compact UTF-8 JSON bytes. Per actual payload, future preflight uses `2 * bytes + 8192` input tokens, the full 128 output tokens, the frozen cache-write input rate and no cache-read discount. It sums reservations over unique payloads and multiplies by four. Test lengths remain unread, so the actual required cap cannot yet be reported; $28.24052 is not a guaranteed invoice ceiling. Missing usage remains reserved/unknown and any reported usage exceeding the envelope halts the run.

Frozen rates: input/cached input/cache write/output $0.10/$0.01/$0.125/$0.50 per million tokens, historically verified 2026-09-26. Recheck the [official pricing source](https://developers.openai.com/api/docs/pricing) before authorizing execution. Implementation reused the existing transport/parser and consulted the official [rate-limit](https://developers.openai.com/api/docs/guides/rate-limits) and [structured-output](https://developers.openai.com/api/docs/guides/structured-outputs) documentation without changing request or retry settings.

## Verification

The new tests cover separate test/live authorization scopes, preflight without spending permission, strict live enforcement after preflight, authorization before file access, frozen protocol/model/settings drift, numeric/current-pricing guards, hash-checked local-only models, label-free inference, independent gates/ties, all 3,080 synthetic IDs, response aliases, cache/reservation tampering, bounded retries, failed/refused outputs, interrupted resume, successful-request reuse and atomic scoring publication. The real official test and model weights are not fixtures.

Run the full suite with `.venv/bin/python -m pytest -q`. **Reviewed full suite: 720 passed in 53.76s**, including 225 new synthetic tests since the preceding committed checkpoint and 42 additional authorization-scope regressions. An audit hook blocked real raw/processed data, existing ignored artifacts, model weights and network access throughout the full suite and sealed dry run; all access-attempt counters were zero. The accompanying `verification.json` records the final suite result, protected-access guards, source hashes and the unchanged protocol hash. No result from a synthetic fixture is a BANKING77 research result.
