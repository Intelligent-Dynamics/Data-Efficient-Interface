"""EXP-006R: separate, capped recovery of ONLY the original 52 HTTP-429 cases."""
import argparse
from datetime import date, datetime, timezone
from decimal import Decimal
from email.utils import parsedate_to_datetime
import json
import math
import os
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, ProxyHandler

from . import general as original
from .data import ROOT, json_bytes, read_json, sha256
from .general_protocol import (ENDPOINT, PRICING, cache_key, digest, estimate, load_protocol)
from .selective import require
from .recovery_source import BUNDLE, DEFAULT_SOURCE, file_inventory, load_source

POLICY = {'max_attempts_per_request': 4, 'serial': True, 'minimum_request_gap_seconds': 5,
          'initial_backoff_seconds': 30, 'backoff_multiplier': 2, 'timeout_seconds': 60,
          'retry_http_statuses': [408, 429, 500, 502, 503, 504],
          'respect_retry_after': True, 'maximum_single_wait_seconds': 3600,
          'non_retryable_error_types': ['insufficient_quota'],
          'non_retryable_error_codes': ['insufficient_quota', 'credit_balance_exhausted', 'billing_hard_limit_reached',
                                        'organization_spend_limit_exceeded', 'project_spend_limit_exceeded',
                                        'organization_usage_limit_exceeded']}
PRICING_VERIFIED_DATE = '2026-09-26'
DEFAULT_RECOVERY = ROOT / 'artifacts/exp006r-gpt6-luna-429-recovery-v1'


def separate_paths(source, output):
    output = Path(output).expanduser().absolute()
    require(not output.is_symlink(), 'Symlink output is not allowed')
    if output.exists():
        require(output.is_dir(), 'Output must be a directory')
        require(not any(p.is_symlink() for p in output.rglob('*')), 'Symlink output contents are not allowed')
    source, output = Path(source).resolve(), output.resolve()
    require(source != output and not source.is_relative_to(output) and not output.is_relative_to(source),
            'Output must be separate from the original EXP-006 tree and its ancestors')


def assert_source_unchanged(source):
    require(file_inventory(Path(source['directory'])) == source['files_sha256'], 'Original EXP-006 evidence changed')


def make_protocol(source):
    old, _, _ = load_protocol()
    return {'study_id': 'EXP-006R', 'version': 'exp006r-http429-v1',
            'source_protocol_sha256': source['source_protocol_sha256'],
            'source_tree_sha256': source['tree_sha256'],
            'source_execution_sha256': source['files_sha256']['execution.json'],
            'eligible_ids': source['eligible_ids'], 'expected_successful_requests_not_resent': 718,
            'source_validation_count': 770, 'endpoint': ENDPOINT,
            'model': old['settings']['model'], 'settings': old['settings'],
            'prompt_sha256': old['prompt_sha256'], 'schema_sha256': old['schema_sha256'],
            'request_cache_keys': {i: cache_key(source['bodies'][i]) for i in source['eligible_ids']},
            'pricing': PRICING, 'pricing_verified_date': PRICING_VERIFIED_DATE,
            'retry_policy': POLICY,
            'eligibility': 'Exactly the 52 http_429 IDs in the original execution; no other requests',
            'authorization': 'New explicit live approval and numeric cap; EXP-006 approval does not carry over',
            'evaluation': 'Existing definitions, all 770 IDs; untouched 718 successes plus recovery-only substitutions; unresolved remain failures',
            'sources': ['https://developers.openai.com/api/docs/models/gpt-6-luna',
                        'https://developers.openai.com/api/docs/pricing',
                        'https://developers.openai.com/api/docs/guides/error-codes',
                        'https://developers.openai.com/api/docs/guides/rate-limits']}


def checked_protocol(source, bundle=BUNDLE):
    protocol = read_json(bundle / 'protocol.json')
    require(protocol == make_protocol(source), 'Recovery protocol or source changed; do not launch')
    return protocol


def validate_selection(source, protocol, bodies):
    eligible = source['eligible_ids']
    require(len(eligible) == len(set(eligible)) == 52, 'Exactly 52 original eligible IDs required')
    require(protocol['eligible_ids'] == eligible, 'Recovery eligibility changed')
    outcomes = {p['id']: p for p in source['execution']['predictions']}
    require({i for i, p in outcomes.items() if p['status'] == 'http_429'} == set(eligible), 'Only original HTTP-429 requests may be sent')
    require(len(outcomes) == 770 and sum(p['status'] == 'ok' for p in outcomes.values()) == 718, 'Original cohort changed')
    require(set(bodies) == set(eligible), 'Successful/ineligible/missing request in recovery inputs')
    for rid, body in bodies.items():
        require(body == source['bodies'][rid], 'Original request body changed')
        require(cache_key(body) == protocol['request_cache_keys'][rid], 'Frozen request hash changed')
    require(len({cache_key(b) for b in bodies.values()}) == 52, 'Duplicate recovery payloads')


def dry_plan(source, protocol):
    validate_selection(source, protocol, source['bodies'])
    requests = [{'id': i, 'original_status': 'http_429', 'cache_key': cache_key(source['bodies'][i]),
                 **estimate(source['bodies'][i])} for i in source['eligible_ids']]
    nominal = sum((Decimal(r['nominal_usd_no_cache_read_discount']) for r in requests), Decimal(0))
    one = sum((Decimal(r['attempt_reservation_usd']) for r in requests), Decimal(0))
    return {'study_id': 'EXP-006R', 'status': 'PREPARATION ONLY — PAID RECOVERY NOT RUN',
            'source_protocol_sha256': source['source_protocol_sha256'], 'source_tree_sha256': source['tree_sha256'],
            'recovery_protocol_sha256': digest(protocol), 'eligible_ids': source['eligible_ids'],
            'planned_unique_requests': 52, 'original_successes_not_resent': 718,
            'maximum_attempts': 52 * POLICY['max_attempts_per_request'], 'requests': requests,
            'estimated_one_attempt_usd': str(nominal), 'conservative_one_attempt_usd': str(one),
            'conservative_maximum_usd': str(one * POLICY['max_attempts_per_request']),
            'estimate_note': 'Same byte-based EXP-006 heuristic: nominal bytes/4 and 32 output; envelope 2*bytes+8192 input and 128 output, all-input cache-write rate. No cache discount. Conditional estimate, not a guaranteed invoice.',
            'pricing': protocol['pricing'], 'pricing_verified_date': protocol['pricing_verified_date'],
            'retry_policy': POLICY, 'inference_calls': 0, 'actual_recovery_spend_usd': '0',
            'official_test_access': False, 'required_live_environment': ['OPENAI_API_KEY'],
            'optional_live_environment': ['OPENAI_PROJECT_ID']}


def authorize(authorized, cap, approved_hash, pricing_date, protocol):
    require(authorized, 'EXP-006R requires NEW explicit --authorize-live')
    cap = original.spending_cap(cap)
    require(approved_hash == digest(protocol), 'Approval must match the recovery protocol hash, not EXP-006')
    require(pricing_date == protocol['pricing_verified_date'], 'Acknowledge recovery pricing verification date')
    require(0 <= (date.today() - date.fromisoformat(pricing_date)).days <= 7, 'Reverify stale prices before live recovery')
    return cap


def retry_after_seconds(headers, received_epoch):
    """Honor delta seconds and HTTP-date; optional millisecond header can only extend wait."""
    headers = {k.lower(): v for k, v in (headers or {}).items()}
    waits = []
    for name, scale in (('retry-after', 1), ('retry-after-ms', .001)):
        value = headers.get(name)
        if value is None:
            continue
        try:
            delay = float(value) * scale
            if math.isfinite(delay) and delay >= 0:
                waits.append(delay)
                continue
        except (TypeError, ValueError, OverflowError):
            pass
        if name == 'retry-after':
            try:
                when = parsedate_to_datetime(value)
                if when.tzinfo is None:
                    when = when.replace(tzinfo=timezone.utc)
                # When available, server Date avoids client clock skew shortening its wait.
                server_date = received_epoch
                if headers.get('date'):
                    try:
                        server = parsedate_to_datetime(headers['date'])
                        if server.tzinfo is None:
                            server = server.replace(tzinfo=timezone.utc)
                        server_date = server.timestamp()
                    except (TypeError, ValueError, OverflowError):
                        pass
                waits.append(max(0, when.timestamp() - server_date))
            except (TypeError, ValueError, KeyError, OverflowError):
                pass
    return max(waits, default=0.0)


def post_recovery(body, timeout):
    """New transport only adds response Retry-After capture; API request bytes are unchanged."""
    key = os.environ.get('OPENAI_API_KEY')
    require(bool(key), 'OPENAI_API_KEY must be set locally')
    headers = {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}
    project = os.environ.get('OPENAI_PROJECT_ID')
    if project:
        headers['OpenAI-Project'] = project
    request = Request(ENDPOINT, data=json_bytes(body), headers=headers, method='POST')
    opener = build_opener(ProxyHandler({}), original.NoRedirect())
    try:
        response = opener.open(request, timeout=timeout)
    except HTTPError as exc:
        response = exc
    with response:
        raw = response.read().decode('utf-8', errors='replace').replace(key, '[REDACTED]')
        try:
            data = json.loads(raw)
        except ValueError:
            data = {'unparsed_response': raw}
        saved_headers = {name: response.headers[name].replace(key, '[REDACTED]')
                         for name in ('Retry-After', 'retry-after-ms', 'Date') if response.headers.get(name) is not None}
        request_id = response.headers.get('x-request-id')
        return {'http_status': response.code, 'request_id': request_id.replace(key, '[REDACTED]') if request_id else None,
                'headers': saved_headers, 'body': data}


def outcome(response, labels):
    result = original.attempt_outcome(response, labels)
    # Do not spend repeated requests on exhausted billing quota even though HTTP is 429.
    body = response.get('body')
    error = body.get('error') if isinstance(body, dict) else None
    if isinstance(error, dict) and (error.get('code') in POLICY['non_retryable_error_codes'] or
                                    error.get('type') in POLICY['non_retryable_error_types']):
        result = {**result, 'retryable': False, 'halt': True}
    return result


def read_entry(output, rid, body, kind, protocol, labels):
    key = cache_key(body)
    path = output / 'responses' / (key + '.json')
    if not path.exists():
        return {'study_id': 'EXP-006R', 'kind': kind, 'id': rid, 'cache_key': key,
                'request': body, 'recovery_protocol_sha256': digest(protocol), 'attempts': []}
    require(not path.is_symlink(), 'Symlink cache rejected')
    entry = read_json(path)
    require(entry['study_id'] == 'EXP-006R' and entry['kind'] == kind and entry['id'] == rid and
            entry['request'] == body and entry['cache_key'] == key and entry['recovery_protocol_sha256'] == digest(protocol), 'Recovery cache identity changed')
    attempts = entry['attempts']
    require(isinstance(attempts, list) and len(attempts) <= POLICY['max_attempts_per_request'], 'Recovery attempts exceeded')
    bound = estimate(body)
    for index, a in enumerate(attempts, 1):
        require(a['attempt'] == index and a['reservation_usd'] == bound['attempt_reservation_usd'], 'Changed recovery reservation')
        require(isinstance(a['started_epoch'], (int, float)) and math.isfinite(a['started_epoch']), 'Invalid durable timing')
        if 'response' not in a:
            require(a['status'] in ('reserved', 'interrupted_unknown', 'transport_unknown', 'client_error') and
                    a.get('predicted_label') is None and a.get('usage_priced_usd') is None, 'Missing recovery response')
            if a['status'] == 'client_error':
                require(a.get('halt') is True and not a.get('retryable'), 'Missing terminal client error halt')
            elif a['status'] != 'reserved':
                require(a.get('retryable') is True, 'Changed transport retry policy')
        else:
            require(digest(a['response']) == a['response_sha256'], 'Recovery response changed')
            expected = outcome(a['response'], labels)
            cost = original.priced_response(a['response']['body'])
            require(a['usage_priced_usd'] == (str(cost) if cost is not None else None), 'Recovery cost changed')
            b = a['response']['body']
            if isinstance(b, dict) and original.usage_breaches(b.get('usage'), bound, cost):
                expected['halt'] = True
                require(a.get('budget_envelope_breached_or_usage_malformed') is True, 'Missing budget breach evidence')
            require(all(a.get(k) == v for k, v in expected.items()), 'Recovery outcome changed')
            delay = retry_after_seconds(a['response'].get('headers'), a['finished_epoch'])
            require(a['retry_after_seconds'] == delay, 'Retry-After timing changed')
        if 'finished_epoch' in a:
            require(math.isfinite(a['finished_epoch']) and a['finished_epoch'] >= a['started_epoch'], 'Invalid completion timing')
    return entry


def next_delay(entries, rid, clock):
    """Global header cooldown survives moving to another ID and restarting the process."""
    attempts = [a for e in entries.values() for a in e['attempts']]
    deadline = clock
    for a in attempts:
        finished = a.get('finished_epoch', a['started_epoch'] + POLICY['timeout_seconds'])
        delay = POLICY['minimum_request_gap_seconds']
        if a.get('retryable') or a['status'] == 'reserved':
            delay = max(delay, POLICY['initial_backoff_seconds'] * POLICY['backoff_multiplier'] ** (a['attempt']-1))
        delay = max(delay, a.get('retry_after_seconds', 0))
        deadline = max(deadline, finished + delay)
    return max(0, deadline - clock)


def source_code():
    code = original.code_record()
    for name in ('recovery.py', 'recovery_source.py', 'recovery_merge.py'):
        p = ROOT / 'baseline' / name
        code['files_sha256'][str(p.relative_to(ROOT))] = sha256(p.read_bytes())
    return code


def execute_recovery(source, protocol, output, cap, transport, *, kind, authorization=None,
                     sleep=time.sleep, clock=time.time):
    require(kind in ('mock', 'live') and ((kind == 'live') == (transport is post_recovery)), 'Mock/live recovery transports cannot mix')
    require(protocol == make_protocol(source), 'Recovery protocol does not match source/policy')
    cap = original.spending_cap(cap)
    if kind == 'live':
        require(isinstance(authorization, dict), 'New live recovery authorization required')
        authorize(authorization.get('authorized'), str(cap), authorization.get('protocol_sha256'),
                  authorization.get('pricing_date'), protocol)
        require(output.resolve().is_relative_to((ROOT / 'artifacts').resolve()), 'Live recovery must use ignored artifacts/')
    separate_paths(source['directory'], output)
    bodies = source['bodies']
    validate_selection(source, protocol, bodies)
    assert_source_unchanged(source)
    total = Decimal(dry_plan(source, protocol)['conservative_maximum_usd'])
    require(total <= cap, f'Recovery cap insufficient: reserve USD {total} before starting')
    manifest = {'study_id': 'EXP-006R', 'kind': kind, 'recovery_protocol_sha256': digest(protocol),
                'source_protocol_sha256': source['source_protocol_sha256'], 'source_tree_sha256': source['tree_sha256'],
                'eligible_ids': source['eligible_ids'], 'protocol': protocol, 'cap_usd': str(cap),
                'full_retry_reservation_usd': str(total), 'code': source_code(), 'live_authorization': authorization}
    try:
        with original.lock(output):
            manifest_path = output / 'manifest.json'
            if manifest_path.exists():
                require(read_json(manifest_path) == manifest, 'Recovery manifest changed; refusing reuse')
                require((output / 'reservations.json').exists(), 'Missing recovery reservation ledger')
            else:
                require(not any(p.name != '.lock' for p in output.iterdir()), 'New recovery directory must be empty')
                original.atomic_json(manifest_path, manifest)
            entries = {i: read_entry(output, i, bodies[i], kind, protocol, source['labels']) for i in source['eligible_ids']}
            original.sync_reservation_ledger(output, manifest, entries)
            for e in entries.values():
                if e['attempts'] and e['attempts'][-1]['status'] == 'reserved':
                    e['attempts'][-1].update(status='interrupted_unknown', predicted_label=None, retryable=True)
                    original.atomic_json(output / 'responses' / (e['cache_key'] + '.json'), e)
            halted = 'previous_fatal_attempt' if any(a.get('halt') for e in entries.values() for a in e['attempts']) else None
            for rid in source['eligible_ids']:
                if halted:
                    break
                e, body = entries[rid], bodies[rid]
                path = output / 'responses' / (e['cache_key'] + '.json')
                while len(e['attempts']) < POLICY['max_attempts_per_request']:
                    if e['attempts'] and not e['attempts'][-1].get('retryable'):
                        break
                    delay = next_delay(entries, rid, clock())
                    if delay > POLICY['maximum_single_wait_seconds']:
                        halted = 'cooldown_pause_resume_after_retry_after'
                        break  # Do not shorten a long Retry-After or issue another request.
                    if delay:
                        sleep(delay)
                        require(next_delay(entries, rid, clock()) <= .001, 'Cooldown has not elapsed')
                    bound = estimate(body)
                    reserved = Decimal(original.accounting(entries)['committed_reservation_usd'])
                    require(reserved + Decimal(bound['attempt_reservation_usd']) <= cap, 'Recovery spending cap reached')
                    # Recheck immutable source immediately before each network attempt.
                    assert_source_unchanged(source)
                    a = {'attempt': len(e['attempts'])+1, 'status': 'reserved', 'started_utc': original.now(),
                         'started_epoch': clock(), 'reservation_usd': bound['attempt_reservation_usd'],
                         'usage_priced_usd': None, 'pre_attempt_delay_seconds': delay}
                    e['attempts'].append(a)
                    original.atomic_json(path, e)
                    original.sync_reservation_ledger(output, manifest, entries)
                    start = time.perf_counter()
                    try:
                        response = transport(body, POLICY['timeout_seconds'])
                    except (TimeoutError, URLError, OSError, ConnectionError):
                        a.update(status='transport_unknown', predicted_label=None, retryable=True)
                    except ValueError:
                        a.update(status='client_error', predicted_label=None, retryable=False, halt=True)
                    else:
                        a.update(response=response, response_sha256=digest(response), **outcome(response, source['labels']))
                        data = response['body']
                        if isinstance(data, dict):
                            a.update(returned_model=data.get('model'), returned_service_tier=data.get('service_tier'), usage=data.get('usage'))
                            cost = original.priced_response(data)
                            a['usage_priced_usd'] = str(cost) if cost is not None else None
                            if original.usage_breaches(a['usage'], bound, cost):
                                a.update(halt=True, budget_envelope_breached_or_usage_malformed=True)
                    a.update(finished_epoch=clock(), finished_utc=original.now(), request_seconds=time.perf_counter()-start)
                    a['retry_after_seconds'] = retry_after_seconds(a.get('response', {}).get('headers'), a['finished_epoch'])
                    original.atomic_json(path, e)
                    if a.get('halt'):
                        halted = 'fatal_attempt_requires_review'
                        break
            predictions = [{'id': i, **original.final_result(entries[i])} for i in source['eligible_ids']]
            report = {'study_id': 'EXP-006R', 'kind': kind, 'recovery_protocol_sha256': digest(protocol),
                      'source_protocol_sha256': source['source_protocol_sha256'], 'source_tree_sha256': source['tree_sha256'],
                      'eligible_ids': source['eligible_ids'], 'predictions': predictions,
                      'status': 'halted' if halted else 'finished_with_unresolved' if any(p['status'] != 'ok' for p in predictions) else 'completed',
                      'halt_reason': halted, 'finished_utc': original.now(), 'accounting': original.accounting(entries),
                      'unresolved_ids': [p['id'] for p in predictions if p['status'] != 'ok']}
            if kind == 'mock':
                report['notice'] = 'SYNTHETIC MOCK; NOT RECOVERY RESEARCH RESULTS'
                report['accounting']['actual_api_spend_usd'] = '0'
            original.atomic_json(output / 'execution.json', report)
            return report, entries
    finally:
        assert_source_unchanged(source)


def load_completed_recovery(source, protocol, directory):
    separate_paths(source['directory'], directory)
    manifest = read_json(directory / 'manifest.json')
    require(manifest['study_id'] == 'EXP-006R' and manifest['kind'] == 'live' and
            manifest['protocol'] == protocol and manifest['recovery_protocol_sha256'] == digest(protocol) and
            manifest['source_tree_sha256'] == source['tree_sha256'] and
            manifest['source_protocol_sha256'] == source['source_protocol_sha256'] and
            manifest['eligible_ids'] == source['eligible_ids'], 'Recovery merge provenance mismatch')
    entries = {i: read_entry(directory, i, source['bodies'][i], 'live', protocol, source['labels']) for i in source['eligible_ids']}
    require((directory / 'reservations.json').exists(), 'Missing recovery merge reservation ledger')
    original.sync_reservation_ledger(directory, manifest, entries, write=False)
    report = read_json(directory / 'execution.json')
    require(report.get('study_id') == 'EXP-006R' and report.get('kind') == 'live' and
            report.get('recovery_protocol_sha256') == digest(protocol) and
            report.get('source_tree_sha256') == source['tree_sha256'] and
            report.get('source_protocol_sha256') == source['source_protocol_sha256'] and
            report.get('eligible_ids') == source['eligible_ids'], 'Recovery execution provenance mismatch')
    require(report['predictions'] == [{'id': i, **original.final_result(entries[i])} for i in source['eligible_ids']], 'Recovery report differs from saved responses')
    require(report['accounting'] == original.accounting(entries), 'Recovery report accounting changed')
    return report, entries


def load_specialists(source):
    # No raw dataset access: use the same versioned selective records pinned by EXP-006.
    from .selective import restore
    audit = source['manifest']['data']['prior_run_hashes']
    runs = []
    for name, hashes in audit.items():
        p = ROOT / 'experiments/exp005-selective-diagnostic/runs' / name
        require(sha256(p.read_bytes()) == hashes['exp005_sha256'], 'Original specialist evidence changed')
        runs.append(restore(read_json(p)))
    require(len(runs) == 15, 'Exactly the 15 original specialists required')
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('dry-run', 'live', 'merge-evaluate'))
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--recovery', type=Path, default=DEFAULT_RECOVERY)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--authorize-live', action='store_true')
    parser.add_argument('--max-spend-usd')
    parser.add_argument('--approve-recovery-protocol-sha256')
    parser.add_argument('--acknowledge-pricing-date')
    args = parser.parse_args()
    # Source loads are strictly read-only; they cannot resume the original experiment.
    source = load_source(args.source)
    protocol = checked_protocol(source)
    separate_paths(args.source, args.output)
    if args.mode == 'live':
        cap = authorize(args.authorize_live, args.max_spend_usd, args.approve_recovery_protocol_sha256,
                        args.acknowledge_pricing_date, protocol)
        require(bool(os.environ.get('OPENAI_API_KEY')), 'Configure OPENAI_API_KEY locally; never paste keys into chat')
        report, _ = execute_recovery(source, protocol, args.output, cap, post_recovery, kind='live', authorization={
            'authorized': args.authorize_live, 'protocol_sha256': args.approve_recovery_protocol_sha256,
            'pricing_date': args.acknowledge_pricing_date})
        print(json.dumps({'study_id': 'EXP-006R', 'status': report['status'], 'unresolved_count': len(report['unresolved_ids'])}))
    elif args.mode == 'dry-run':
        args.output.mkdir(parents=True, exist_ok=False)
        dry = dry_plan(source, protocol)
        dry.update(generated_utc=original.now(), code=source_code())
        original.atomic_json(args.output / 'dry_run.json', dry)
        print(json.dumps({k: dry[k] for k in ('status','recovery_protocol_sha256','planned_unique_requests','maximum_attempts','estimated_one_attempt_usd','conservative_maximum_usd')}, indent=2))
    else:
        separate_paths(args.recovery, args.output)
        report, entries = load_completed_recovery(source, protocol, args.recovery)
        from .recovery_merge import offline_evaluation
        result = offline_evaluation(source, report, entries, load_specialists(source))
        args.output.mkdir(parents=True, exist_ok=False)
        original.atomic_json(args.output / 'evaluation.json', result)
        print('Saved separate EXP-006 + EXP-006R recovery evaluation; original artifacts untouched.')
    assert_source_unchanged(source)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        print(f'EXP-006R stopped: {type(exc).__name__}; check source integrity, authorization, budget and local recovery evidence.', file=sys.stderr)
        raise SystemExit(1)
