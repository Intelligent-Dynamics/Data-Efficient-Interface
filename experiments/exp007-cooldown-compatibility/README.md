# EXP-007 interrupted-run cooldown compatibility

This narrow change repairs the local wait guard. It changes no experiment protocol, threshold, model, prompt/schema, retry count, rate-limit interval, cap or evaluation definition. **No live resume was executed during preparation.**

## Timing change

The collector previously slept once and raised if the wall-clock deadline still had more than 1 ms remaining. It now repeatedly computes the larger of the persisted-wall-clock remainder and an initially anchored monotonic remainder, sleeps that positive remainder, and rechecks until at most 1 ms remains. A forward clock jump cannot shorten the initially owed wait; a backward jump can conservatively extend it. Persisted epoch/UTC timestamps and `recovery.next_delay` are unchanged, so restart still reconstructs cooldowns from durable records. Global Retry-After and exponential backoff still apply across request IDs.

A delay above the unchanged `maximum_single_wait_seconds` pauses as before. Cumulative monotonic waiting is also bounded by that limit; nonadvancing monotonic time or 64 interrupted sleeps pauses safely rather than spinning or dispatching early. The existing pause status/reason is retained. These are local wait-loop bounds, not additional API attempts or a change to the four-attempt retry allowance. `pre_attempt_delay_seconds` remains the initially required delay.

## Preserved live state

Read-only inspection verified 509 successful HTTP-200 requests, 509 attempts, no unknown/unresolved attempts and 2,571 unattempted requests. Returned usage prices to $0.0548184 at frozen rates; this is not invoice reconciliation. Current cap remains $28.00. All **533** pre-existing artifact files, including all 509 response files, remain byte-identical. No customer text or response body is published here.

The original code was `aa58af8461ea3423b849d4e67a24eec93eb4dbb7`. Its outer and collection manifests bind that source; the reservation journal binds the original collection manifest digest. Merely editing the wait code would correctly fail those checks.

[receipt.json](receipt.json) separately binds the original manifest bytes, complete old/new runtime hash maps, preserved artifact hashes and original attempt counts. Only four runtime files change: the collector, the narrow compatibility validator, the authorization record's optional receipt field, and the CLI/outer-manifest integration. All other runtime files, dependencies and both frozen protocols remain unchanged. Historical evidence is not regenerated.

Resume requires explicit approval of this exact receipt SHA-256:
`687333556fc6012f36047544c2413a8b64b3beade6a1113ea3f16662559437a1`

All four source-binding sites enforce it: outer run manifest, collector preflight binding, collection resume, and completed-collection verification. Non-code fields—including cap, rows, protocol, pricing, authorization, environment and packages—must still match exactly. The original manifests are returned unchanged to ledger verification. No original success can gain another attempt. Additional future request records may append without resetting prior reservations.

A later authorized resume creates **one new** `luna/source_compatibility.json`, recording the approved receipt hash and actual patched Git HEAD/source/environment. That new record is immutable on subsequent resumes and is included in eventual prediction freezing. It has **not** been created in the existing live tree during preparation. This explicit transition does not permit arbitrary later edits or automatic migration of another experiment.

## Future resume command — NOT executed

After separately verifying that official prices still equal the frozen assumptions on the indicated UTC date, and with authorized live/test access, run from `/Users/Andrew/Developer/data-efficient-inference`:

```sh
.venv/bin/python -m baseline.final_test live \
  --approved-protocol-sha256 0ff095f523ebc8f05d295dad4545f6a730dd07c6c9bc7ac194bbbb4a759cfa00 \
  --approved-resume-compatibility-sha256 687333556fc6012f36047544c2413a8b64b3beade6a1113ea3f16662559437a1 \
  --authorize-test-access --authorize-live \
  --spending-cap-usd 28.00 \
  --acknowledge-current-pricing-date 2026-09-28
```

The date is the preparation session's current UTC date, **not a claim that prices were reverified here**. If executed later, acknowledge that later current UTC date only after checking prices. The local `OPENAI_API_KEY` stays out of commands/chat/Git. Preserve the same directory, source and cap; do not create a replacement run. Saved successful Luna requests are skipped and completed specialist checkpoints are reused without inference. This live command eventually scores only after the existing frozen-prediction checkpoint requirements are satisfied.

## Verification

The pure compatibility validator accepted both actual saved live manifests against the patched runtime, under guards prohibiting filesystem writes, dataset/model access and network; counters were zero. The full synthetic test result and before/after preservation checks are recorded separately in `verification.json`. The first diagnostic guard rejected harmless `/dev/null` access used by platform provenance lookup; it was corrected only to allow that device and then passed. No live artifact was changed by either check.

No raw test-file/label access, API calls, model inference, live resume, scoring or timing benchmark occurred during this fix. The frozen EXP-007 protocol remains `0ff095f523ebc8f05d295dad4545f6a730dd07c6c9bc7ac194bbbb4a759cfa00`; EXP-009 remains `b78f3d32a9bd86ea130ce1f51e2feded9d52b5f5ff796507c49d6f53fccc3334`.
