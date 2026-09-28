# EXP-009 exact-source resume compatibility

The interrupted test collection has **826 successful responses in 826 attempts**, leaving **2,254 unattempted requests**. Every remaining request retains the original four-attempt allowance. This checkpoint adds an explicitly approved compatibility transition; it does not execute the resume or produce test scores.

## Exact source transition

The immutable live manifest binds Git `36495acf8e90b763789792c9bbca509bc708de7b`, before the cooldown port in `38d836a5f2b0b49ca31ba7b67a222a4230cb01e1`. Against that original runtime, precisely these three paths change:

- `baseline/retrieved_collection.py`: the already-reviewed cooldown helper port, source-record extraction, explicit receipt approval and historical-manifest/immutable-provenance checks.
- `baseline/retrieved_run.py`: the approval flag and receipt-bound prepared-input path; no repeated preflight/inference on an approved resume. Completed-evidence verification receives the same approval.
- `baseline/retrieved_compatibility.py`: the EXP-009-specific validator.

The original and patched complete runtime maps (28 and 29 files) are recorded in [receipt.json](receipt.json). All other runtime files, dependencies and both frozen protocols are unchanged. The EXP-007 cooldown helper itself is unchanged; its existing hash is added to EXP-009 helper provenance. No models, settings, prompt, schema, retrieval, thresholds, retry limits, cap, request bodies or evaluation definitions change.

## Evidence and ledger preservation

The receipt binds the exact original manifest bytes, old Git/runtime identity, patched runtime hashes, all five prepared files, the 3,080-request digest, $36.00 cap, $35.32756300 full reservation, original ledger bytes/state and all 826 successful response-file hashes/counts. The validator rechecks request/response identities, outcomes, usage, retry allowance and the ledger using the existing read-only primitives. Original successes remain terminal at one attempt; deleting or retrying one fails. Remaining response files may append only under the unchanged identity/retry/journal rules.

The original manifest is returned unchanged to ledger verification; no historical evidence is rewritten. A subsequent authorized resume may append legitimate ledger counts for outstanding requests. On that resume only, a new `collection/source_compatibility.json` records the approved receipt and actual patched Git/source/environment. It is immutable thereafter; a later commit, even documentation-only, may require review before another resume. No such record was written during this preparation.

All **834 existing files** remain byte-identical. Returned usage prices to **$0.153193925** at frozen rates (not invoice-reconciled). Existing committed reservations are **$2.35222450**; the full remaining-attempt reservation is **$25.91866500**, within the unchanged $36.00 cap. These reservations are safeguards, not extra API charges.

The exact receipt SHA-256 is:

`7587d5363a6aa5fbe885a9f979a79905867865ba2b536fffb94864a65e9326fd`

Source drift remains rejected without this explicit approval. Receipt approval cannot create a new/replacement run, change requests/cap/environment, expand the allowed source transition or shorten cooldown/retry requirements. The existing raw test is not reopened by the approved receipt-based collection path. Scoring remains a distinct, separately invoked operation; this patch does not execute it.

## Future resume command — NOT executed

After explicit live approval and verification that current official model/settings/pricing match the frozen assumptions:

```sh
cd /Users/Andrew/Developer/data-efficient-inference
.venv/bin/python -m baseline.retrieved_run test-live \
  --approved-protocol-sha256 b78f3d32a9bd86ea130ce1f51e2feded9d52b5f5ff796507c49d6f53fccc3334 \
  --approved-resume-compatibility-sha256 7587d5363a6aa5fbe885a9f979a79905867865ba2b536fffb94864a65e9326fd \
  --authorize-test-access --authorize-live \
  --spending-cap-usd 36.00 \
  --acknowledge-model-pricing-date 2026-09-28
```

The acknowledgement date above is this preparation's UTC date, **not a claim of fresh pricing verification**. If executed later, verify compatibility/pricing and use that day's UTC date. Keep the same output directory and cap. The local `OPENAI_API_KEY` stays out of commands, chat and Git. This command resumes collection only; it does not score test labels.

## Verification

The focused synthetic suite checks exact/wrong receipt approval, arbitrary later source drift, non-source changes, evidence and ledger tampering, no replacement run, immutable separate provenance, preservation of successful responses, remaining retry allowance and CLI reuse without preflight or model access. The guarded real-artifact check reconstructed the exact expected manifest and verified `accepted_manifest` against the 826-success run without any artifact write, API call, test-label access, inference or preflight. **78 focused tests passed in 11.91 seconds; 917 tests passed in one full guarded suite run (76.18 seconds).** Network and protected-access counters were zero. Guards, source hashes and before/after preservation checks are recorded in [verification.json](verification.json).

EXP-009 protocol remains `b78f3d32a9bd86ea130ce1f51e2feded9d52b5f5ff796507c49d6f53fccc3334`; EXP-007 remains `0ff095f523ebc8f05d295dad4545f6a730dd07c6c9bc7ac194bbbb4a759cfa00`. Earlier preparation and interrupted-run evidence are preserved.
