"""Read-only EXP-009 usage-price attribution; never dispatch, infer, or score."""
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from baseline import general
from baseline.data import read_json, sha256
from baseline.general_protocol import PRICING, digest

PROTOCOL = 'b78f3d32a9bd86ea130ce1f51e2feded9d52b5f5ff796507c49d6f53fccc3334'


def require(value, message):
    if not value:
        raise ValueError(message)


def summarize(entries):
    """Price every attempt from returned usage, checking existing accounting."""
    known = successful = unsuccessful = retries = Decimal(0)
    usage = Counter({k: 0 for k in ('input_tokens', 'output_tokens', 'cached_tokens',
                                   'cache_write_tokens', 'reasoning_tokens')})
    statuses = Counter()
    for entry in entries.values():
        require(entry['attempts'], 'Missing attempt history')
        for number, attempt in enumerate(entry['attempts'], 1):
            require(attempt['attempt'] == number, 'Attempt sequence mismatch')
            body = attempt.get('response', {}).get('body')
            charge = general.priced_response(body)
            require(attempt.get('usage_priced_usd') == (str(charge) if charge is not None else None),
                    'Raw usage and saved price disagree')
            statuses[attempt['status']] += 1
            if charge is None:
                continue
            known += charge
            if attempt['status'] == 'ok':
                successful += charge
            else:
                unsuccessful += charge
            if number > 1:
                retries += charge  # Overlaps success/failure subtotals; never added twice.
            u = body['usage']
            for key in ('input_tokens', 'output_tokens'):
                usage[key] += u[key]
            for key in ('cached_tokens', 'cache_write_tokens'):
                usage[key] += u['input_tokens_details'][key]
            usage['reasoning_tokens'] += u.get('output_tokens_details', {}).get('reasoning_tokens', 0)
    rates = PRICING['per_million_tokens']
    # Independent group arithmetic, derived from raw response token counts.
    replay = (Decimal(usage['input_tokens']-usage['cached_tokens']-usage['cache_write_tokens'])*Decimal(rates['input']) +
              Decimal(usage['cached_tokens'])*Decimal(rates['cached_input']) +
              Decimal(usage['cache_write_tokens'])*Decimal(rates['cache_write']) +
              Decimal(usage['output_tokens'])*Decimal(rates['output'])) / 1000000
    account = general.accounting(entries)
    require(account['reservation_envelope_valid'] and replay == known == Decimal(account['usage_priced_api_charges_usd']),
            'Accounting envelope or independent price replay failed')
    return {'request_count':len(entries), 'attempt_count':account['recorded_attempts'],
            'attempt_status_counts':dict(statuses), 'known_charges_usd':str(known),
            'priced_successful_attempt_charges_usd':str(successful),
            'priced_unsuccessful_attempt_charges_usd':str(unsuccessful),
            'priced_retry_attempt_charges_usd_included_above':str(retries),
            'unknown_charge_attempts':account['unknown_charge_attempts'],
            'unknown_charge_reservation_usd':account['unknown_charge_reservation_usd'],
            'spend_interval_under_frozen_assumptions_usd':account['spend_interval_under_frozen_assumptions_usd'],
            'returned_usage_totals':dict(usage)}


def reduction(all_cost, routed):
    a, r = (Decimal(x['known_charges_usd']) for x in (all_cost, routed))
    ua, ur = (Decimal(x['unknown_charge_reservation_usd']) for x in (all_cost, routed))
    require(a > 0 and 0 <= r <= a and 0 <= ur <= ua, 'Invalid nested cost partition')
    saved, un = a-r, ua-ur
    # Shared routed unknown costs cancel in absolute savings. Percentage extrema
    # retain their correlation: (saved+y)/(a+x+y), x in [0,ur], y in [0,un].
    return {'known_dollar_reduction_usd':str(saved),
            'absolute_reduction_interval_usd':[str(saved),str(saved+un)],
            'known_spend_reduction_percent':str(100*saved/a),
            'spend_reduction_percent_interval':[str(100*saved/(a+ur)),str(100*(saved+un)/(a+un))]}


def partition(entries, specialist):
    ids = [row['id'] for row in specialist]
    require(len(ids) == len(set(ids)) and set(ids) == set(entries), 'ID alignment mismatch')
    require(all(type(row['use_specialist']) is bool for row in specialist), 'Nonboolean routing decision')
    rejected = {row['id'] for row in specialist if row['use_specialist'] is False}
    groups = {'all_luna':summarize(entries),
              'routed_luna':summarize({i:e for i,e in entries.items() if i in rejected}),
              'not_routed':summarize({i:e for i,e in entries.items() if i not in rejected})}
    for key in ('known_charges_usd', 'unknown_charge_reservation_usd'):
        require(Decimal(groups['all_luna'][key]) == Decimal(groups['routed_luna'][key])+Decimal(groups['not_routed'][key]),
                'Partition accounting mismatch')
    return {**groups, 'reduction':reduction(groups['all_luna'],groups['routed_luna']),
            'request_reduction_percent':str(100*Decimal(len(ids)-len(rejected))/Decimal(len(ids))),
            'fallback_ids_sha256':digest(sorted(rejected))}


def analyze(root=ROOT):
    run = root/'artifacts/exp009-retrieved-luna-v1/test'
    frozen = root/'experiments/exp009-retrieved-luna/protocol.json'
    original = read_json(root/'experiments/exp009-official-test/accounting.json')
    require(sha256(frozen.read_bytes()) == PROTOCOL, 'Frozen protocol changed')
    require(read_json(frozen)['pricing'] == PRICING, 'Frozen pricing changed')
    provenance = original['provenance']
    for name in ('specialist.json','collection/manifest.json','collection/reservations.json'):
        require(sha256((run/name).read_bytes()) == provenance['metadata_files_sha256'][name], 'Source evidence changed')
    files = sorted((run/'collection/responses').glob('*.json'))
    before = {str(p.relative_to(run)):sha256(p.read_bytes()) for p in files}
    require(len(files) == 3080 and digest(before) == provenance['response_file_inventory_sha256'],
            'Original response inventory changed')
    entries = {}
    for path in files:
        entry = read_json(path)
        require(entry['id'] not in entries and path.name == entry['id'].replace(':','_')+'.json', 'Duplicate/misnamed response')
        require(entry['protocol_sha256'] == PROTOCOL and entry['attempts'][-1]['status'] == 'ok', 'Incomplete/wrong protocol response')
        entries[entry['id']] = entry
    manifest = read_json(run/'collection/manifest.json')
    ledger = read_json(run/'collection/reservations.json')
    general.sync_reservation_ledger(run/'collection',manifest,entries,write=False)
    require(ledger['attempt_counts'] == {i:len(e['attempts']) for i,e in entries.items()}, 'Ledger mismatch')
    result = partition(entries,read_json(run/'specialist.json'))
    require(result['routed_luna']['request_count'] == 268, 'Expected exactly 268 fallback IDs')
    require(result['all_luna']['known_charges_usd'] == '0.576343155' and
            result['all_luna']['spend_interval_under_frozen_assumptions_usd'] == ['0.576343155','0.584791155'],
            'All-Luna reference accounting mismatch')
    require(before == {str(p.relative_to(run)):sha256(p.read_bytes()) for p in files}, 'Response evidence modified')
    return {'study_id':'EXP-009 routed LLM API spend attribution', 'pricing':PRICING,
            'protocol_sha256':PROTOCOL, **result,
            'actual_invoice_spend_usd':None, 'invoice_reconciled':False,
            'assumptions':['Replay recorded token usage, cache-read/write classification and retry histories at frozen rates.',
                           'Sending only the fallback subset could change caching or retries; this is not a new execution or guaranteed invoice.',
                           'LLM API spend reduction only; specialist/retrieval/serving costs and total-system savings are unmeasured.'],
            'provenance':{'specialist_sha256':provenance['metadata_files_sha256']['specialist.json'],
                          'response_file_inventory_sha256':digest(before),
                          'reservation_ledger_sha256':provenance['metadata_files_sha256']['collection/reservations.json']},
            'verification':{'raw_returned_usage_repriced':True, 'independent_token_total_repricing':True,
                            'ledger_consistent':True, 'response_files_byte_identical':True,
                            'no_scoring_or_inference':True, 'api_calls':0}}


if __name__ == '__main__':
    print(json.dumps(analyze(),indent=2,sort_keys=True))
