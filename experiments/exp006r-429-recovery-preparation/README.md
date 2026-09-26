# EXP-006R — separate HTTP-429 recovery preparation

**PREPARATION ONLY. PAID RECOVERY NOT RUN.** No recovery predictions or recovered quality results exist. This implementation uses synthetic responses in tests; preparation API spend is **$0**. The original EXP-006 evidence remains an immutable input.

## Source and eligibility

Canonical repository: `/Users/Andrew/Developer/data-efficient-inference`; origin: `https://github.com/Intelligent-Dynamics/data-efficient-inference.git`.

The saved `artifacts/exp006-gpt6-luna-validation-v1/execution.json` contains 770 validation outcomes: **718 `ok`, 52 `http_429`, status `finished_with_unresolved`**. Its caches contain 823 attempts: 718 successful and 105 HTTP-429 attempts. Each eligible ID has already exhausted its two original attempts. EXP-006R grants no extra attempts under the original protocol: it is a separate protocol, authorization, cache, ledger and evaluation.

- Original protocol SHA-256: `d7f824ee761ca90d4dc3a848dd19e3928475c799d6713e03e336dc2a80c9ca68`.
- Recovery protocol SHA-256: `99500454a24e5b01ba3c1bba16d5076101d06a5a3d448fe400c32af716c16161`.
- Original tree SHA-256: `bfd8ad140577004bb436a4c89a977721a3286dd358470fa05e5cf1351037a7cb`.
- [Source audit](source_audit.json) records eligibility, error counts and provenance. [Inventory](source_inventory.json) records SHA-256 for **all 775 files / 16,688,254 bytes**, including hidden files; the tree digest is SHA-256 of the repository's canonical `json_bytes(inventory)`.
- [Recovery protocol](protocol.json) lists the exact 52 eligible IDs and payload cache keys, read directly from the execution record. The 718 successes cannot enter the recovery request dictionary. No correctness-based selection occurs.

**Billing prerequisite:** all 105 original HTTP-429 attempt bodies say `insufficient_quota` / `credit_balance_exhausted`. These are quota/billing errors, not evidence of transient request-rate throttling. Restore usable API credits/account quota before any future authorized recovery. Backoff alone cannot repair this. The new runner stops immediately if a quota/billing error recurs, preserves that attempt and all other unresolved IDs, and does not automatically resume past a fatal error. No account credentials or billing endpoints were inspected during preparation. The original transport did not retain `Retry-After` headers; no historical wait is invented. [Official error-code guidance](https://developers.openai.com/api/docs/guides/error-codes).

## Frozen request and data reuse

Reuse the [original prompt](../exp006-general-model-preparation/prompt.txt), [schema](../exp006-general-model-preparation/schema.json), all 77 class names, and the exact saved request bodies. Every saved body is checked against the original frozen protocol and validation-text hash before reuse. Each request includes only the customer message and fixed zero-shot classification instructions, without ground truth, specialist predictions, confidence, demonstrations or ID.

Settings remain `gpt-6-luna`, reasoning `none`, maximum output 128 tokens, `service_tier=default` (Standard), `store=false`, `stream=false`, `truncation=disabled`, implicit prompt caching with `ttl=30m`. Sampling defaults remain unchanged. The alias has no documented dated snapshot; recovery takes place later and cannot establish immutable underlying provider weights. Every returned model ID and service tier is checked and retained.

No model is fitted. No new split is generated. All 770 existing validation IDs/labels and all 15 EXP-005 specialist acceptance rankings are reused. The **770 additional validation labels** remain part of the development budget; recovery adds zero new unique labels. Five seeds share one holdout. Neither raw training files nor the official BANKING77 test are opened by recovery preparation, execution or offline merging. Prior pinned records and saved request text provide the required inputs. Local original caches are necessary to reproduce this audit; compact tracked hashes cannot reconstruct private/raw response caches.

## Execution and retry policy

- Serial requests, with at least **5 seconds after completion** before the next attempt/request.
- At most **4 new attempts per eligible ID** (including the first recovery attempt); maximum **208 attempts**, never 718 successful requests.
- Identical-payload retries only for transport errors or HTTP 408/429/500/502/503/504. Backoff starts at **30 seconds**, doubles (30/60/120 between four attempts), and also applies globally after an exhausted fourth transient attempt (240 seconds before advancing). Retry decisions never inspect correctness.
- Respect valid `Retry-After` seconds or HTTP dates, and optional `retry-after-ms`, using the greatest applicable wait. Saved timing/header evidence enforces cooldowns across IDs and restarts. A delay exceeding one hour pauses collection; it is never shortened to permit an early request. Rerun the same command later to resume after the deadline.
- Quota/billing errors are terminal for the entire collection. Refusals, incomplete/invalid outputs and valid wrong answers are retained, not retried to improve accuracy. Model/tier mismatch, token-envelope breach or malformed usage halts for review.
- Record each attempt durably before sending; interrupted/unknown attempts consume reservations. Resume checks every cache entry plus the independent attempt-count ledger. Original EXP-006 is never locked or written; its full inventory is checked before requests and after the operation. Symlink outputs and overlapping source/output paths are rejected.
- A completed recovery cache resumes without resending successes. Keep code/protocol/cap/runtime fixed for resume; a provenance change is rejected. Do not delete or edit caches/ledgers to obtain extra attempts. Fatal outcomes require review and, if another experiment is needed, a separately documented authorization rather than silently resetting this one.

The ignored live output is `artifacts/exp006r-gpt6-luna-429-recovery-v1`. It stores separate `manifest.json`, `execution.json`, `reservations.json` and per-request response/attempt JSON with timestamps, headers, request timings, returned model/tier, token details, usage-priced charges and unknown reservations. Secrets and full raw caches remain out of Git.

## No-inference dry run and cost assumptions

From the canonical checkout:

```sh
.venv/bin/python -m baseline.recovery dry-run \
  --source artifacts/exp006-gpt6-luna-validation-v1 \
  --output artifacts/exp006r-preparation-dry-run-v1
```

Choose a fresh output directory on subsequent dry runs; existing dry-run artifacts are not overwritten. This command performs **zero network/API calls**, reads no credentials or raw dataset files, and saves no predictions. [Saved dry run](dry_run.json) records every eligible request's estimate and cache key.

| Estimate for only 52 eligible requests | USD |
| --- | ---: |
| Nominal, one attempt per request | 0.008374250 |
| Conservative, one attempt per request | 0.11689400 |
| Conservative, all four attempts per request | **0.46757600** |
| Proposed future recovery cap, **not authorized** | **0.50** |
| Actual API spend in this preparation | **0** |

The nominal estimate uses `ceil(serialized UTF-8 bytes / 4)` input tokens and 32 output tokens. The reservation uses `2 * bytes + 8192` input tokens and the full 128-output cap, with **all input charged at the cache-write rate** and no cache-read discount. These are engineering token-count envelopes, not verified tokenization or guaranteed invoices. Each potential retry is budgeted before collection; actual attempts consume durable reservations, and observed breaches stop further requests. The $0.50 proposal is a **new recovery-only cap**, not the earlier EXP-006 $4.00 authorization.

Official Standard, nonregional, short-context rates rechecked **2026-09-26** match the unchanged EXP-006 pricing: input/cached-input/cache-write/output **$0.10/$0.01/$0.125/$0.50 per million tokens**, excluding taxes/credits. The original pricing object retains its original 2026-09-25 date; the new protocol separately records verification on 2026-09-26. Sources: [model](https://developers.openai.com/api/docs/models/gpt-6-luna), [pricing](https://developers.openai.com/api/docs/pricing), [rate-limit guidance](https://developers.openai.com/api/docs/guides/rate-limits).

Returned token usage, including caching/reasoning details when present, is priced separately from reservations. Missing usage remains unknown, never free; usage-priced charges are not invoice reconciliation. The offline evaluator separates original, recovery and combined collection accounting. Hypothetical routed API charges are separately labeled and include all recorded attempts for rejected IDs; this replay assumes the observed cache behavior. Specialist deployment cost, total-system savings and production latency remain unmeasured.

## Future live command — requires new explicit approval

**Do not execute during preparation.** First resolve the quota issue, recheck current prices, and obtain new explicit approval for this recovery hash and numeric cap. Pricing verification expires after seven days; if stale, update the separately versioned recovery protocol/approval rather than bypassing the check. Never change the original EXP-006 protocol. `OPENAI_API_KEY` is required locally; `OPENAI_PROJECT_ID` is optional. Never print, paste or commit credentials. Earlier live approval and the presence of a key are insufficient.

```sh
cd /Users/Andrew/Developer/data-efficient-inference
.venv/bin/python -m baseline.recovery live \
  --source artifacts/exp006-gpt6-luna-validation-v1 \
  --output artifacts/exp006r-gpt6-luna-429-recovery-v1 \
  --authorize-live \
  --max-spend-usd 0.50 \
  --approve-recovery-protocol-sha256 99500454a24e5b01ba3c1bba16d5076101d06a5a3d448fe400c32af716c16161 \
  --acknowledge-pricing-date 2026-09-26
```

The runner will reject missing authorization/cap, a mismatched protocol, changed source evidence, insufficient reservation budget, stale pricing or a missing local key before inference.

## Offline merge after a finished live recovery

```sh
.venv/bin/python -m baseline.recovery merge-evaluate \
  --source artifacts/exp006-gpt6-luna-validation-v1 \
  --recovery artifacts/exp006r-gpt6-luna-429-recovery-v1 \
  --output artifacts/exp006-plus-exp006r-evaluation-v1
```

This offline command makes no API calls. It validates both saved ledgers and provenance, copies the 718 original successful predictions unchanged and substitutes only the original 52 unresolved IDs. Output uses all **770 IDs exactly once**, in the original order. Finished collections with remaining unresolved cases are allowed, with each unresolved case counted as a failure; partial/halted collection cannot masquerade as finished. No original/recovery evidence is overwritten; the output directory must be fresh.

The unchanged EXP-006 evaluator recomputes Luna standalone accuracy/macro-F1 and all **90 combinations** (15 specialists × accepted counts 0/193/385/578/693/770), including rejected-subset fallback accuracy and mean/sample SD. Every result is labeled **EXP-006 + EXP-006R recovery**, with per-ID prediction source. No production threshold is selected. This operation has only been tested using synthetic fixtures; no real recovered evaluation has occurred.

## Verification

Full suite: **371 passed** (128 new recovery tests; no failures). Run `.venv/bin/python -m pytest -q`. Source, execution and merge regression tests cover exact eligibility, successful-request exclusion, frozen payload hashes, row/label alignment, immutable source files, all-770 merge identity, no test/raw-data access, dry-run network/credential prohibition, explicit authorization and finite numeric caps, cache/ledger resume, bounded retries, Retry-After/date parsing, global cooldown, quota stops, failures and unchanged combined-metric definitions. [Preparation verification](verification.json) records the final suite and independently recomputed estimates. No model/endpoint was exercised.
