# EXP-009 routed LLM API spend attribution

Read-only accounting of the existing seed-11 test routing decisions, at frozen 2026-09-25 EXP-009 rates. This adds no inference, scoring, tuning or experiment. Historical artifacts and both protocols remain unchanged.

| Quantity | Exact USD |
| --- | ---: |
| Routed known charges: 268 requests, 268 successful attempts | 0.051380650 |
| Routed priced retry charges / unknown-usage reservation | 0 / 0 |
| Routed spend interval | [0.051380650, 0.051380650] |
| All-Luna known charges | 0.576343155 |
| All-Luna spend interval | [0.576343155, 0.584791155] |
| Known LLM API spend reduction | 0.524962505 |
| LLM API spend reduction interval | [0.524962505, 0.533410505] |

Known usage-priced LLM spend reduction is **91.08505938619155%**; allowing the recorded unknown-attempt bounds gives **91.08505938619155%–91.21384624909383%**. This is similar to, but slightly below, the **91.2987012987013%** request reduction. The routed requests have slightly higher mean recorded charges. This is a sum of their actual token usage, not total spend multiplied by 268/3,080.

**None of the three unknown-charge attempts belongs to a routed request.** All three are on specialist-accepted requests, carrying $0.0084480 in unknown reservations. All routed requests succeeded on the first attempt; no retry charge is omitted. The all-case three successful retries cost $0.000232805, already included in the all-Luna known total (not added twice).

## Independent calculation

`specialist.json` supplies exactly 268 unique IDs with Boolean `use_specialist == false`. Every corresponding durable response/attempt is priced from its returned usage with the existing `general.priced_response` and `general.accounting` rules. A separate raw-token aggregation gives the same exact result:

- Input: 390,422 tokens, including 389,618 cache-write tokens and zero cache-read tokens; 804 ordinary input tokens remain.
- Output: 5,196 tokens; reported reasoning tokens: zero.
- USD: `(804 × 0.10 + 389618 × 0.125 + 5196 × 0.50) / 1000000 = 0.051380650`.

The script checks the original specialist/response hashes, protocol/pricing, complete ID alignment and reservation ledger. It retains all attempts. `summary.json` contains only aggregate counts/usage/prices and provenance hashes, with no request IDs, test text or responses. `verification.json` records preservation and test checks.

For shared unknown costs, let `A` be all-case known charges, `R` routed known charges, `x ∈ [0, Ur]` routed unknown cost and `y ∈ [0, Un]` nonrouted unknown cost. Absolute reduction is `A − R + y`; the routed uncertainty cancels. Percentage reduction is `100 × (A − R + y)/(A + x + y)`, with bounds at `(x=Ur, y=0)` and `(x=0, y=Un)`. Here `Ur=0` and `Un=0.0084480`.

Run from the repository root with existing private artifacts present:

```sh
.venv/bin/python -B experiments/exp009-routed-api-spend/analyze.py
```

This command only reads local evidence and prints aggregate JSON. It has no dispatch, write, inference or scoring path.

## Interpretation

This is **hypothetical usage-priced LLM API spend reduction**, under the recorded token/cache/retry behavior, not money actually saved by the all-case study. Reordering or sending only the fallback subset could change caching, retries and charges; no counterfactual traffic was executed. Actual invoice spend remains unknown/unreconciled. Specialist embedding/classification, retrieval and serving costs are excluded. **No total-system or production dollar savings are claimed.**
