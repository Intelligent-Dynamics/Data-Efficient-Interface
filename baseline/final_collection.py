"""Guarded EXP-007 shared Luna collection. No dataset reader or labels enter this module."""
from collections import Counter
from decimal import Decimal
import math
import os
from pathlib import Path
import time
from urllib.error import URLError

from . import general, recovery
from .data import ROOT, read_json, sha256
from .general_protocol import ENDPOINT, PRICING, SETTINGS, cache_key, digest, estimate, payload
from .selective import require


def _validate(protocol, prompt, schema):
    from .final_protocol import validate_protocol
    validate_protocol(protocol)
    fallback = protocol['fallback']
    require(fallback['endpoint'] == ENDPOINT and fallback['settings'] == SETTINGS and
            fallback['retry_policy'] == recovery.POLICY, 'Frozen Luna request/retry settings changed')
    require(sha256(prompt.encode()) == fallback['prompt_sha256'], 'Frozen prompt changed')
    require(digest(schema) == fallback['schema_sha256'], 'Frozen schema changed')
    require(schema['properties']['intent']['enum'] == protocol['population']['labels'], 'Frozen label order changed')


def _requests(rows, protocol, prompt, schema):
    _validate(protocol, prompt, schema)
    require(isinstance(rows, list) and len(rows) == 3080, 'Exactly 3080 inference rows required')
    require(all(isinstance(row, dict) and set(row) == {'id', 'text'} for row in rows),
            'Inference rows must contain only id/text; labels and specialist fields are forbidden')
    require([row['id'] for row in rows] == [f'test:{i:05d}' for i in range(3080)],
            'All 3080 original test IDs required exactly once in source order')
    unique, aliases = {}, []
    for row in rows:
        body = payload(row['text'], prompt, schema, protocol['fallback']['settings'])
        key = cache_key(body)
        if key not in unique:
            unique[key] = {'id': row['id'], 'body': body, 'estimate': estimate(body)}
        else:
            require(unique[key]['body'] == body, 'Request cache key collision')
        aliases.append({'id': row['id'], 'cache_key': key, 'canonical_id': unique[key]['id']})
    return unique, aliases


def collection_plan(rows, protocol, prompt, schema):
    """Exact payload envelopes for authorized in-memory rows; never opens data or network."""
    requests, aliases = _requests(rows, protocol, prompt, schema)
    first = sum((Decimal(r['estimate']['attempt_reservation_usd']) for r in requests.values()), Decimal(0))
    nominal = sum((Decimal(r['estimate']['nominal_usd_no_cache_read_discount']) for r in requests.values()), Decimal(0))
    maximum = protocol['fallback']['retry_policy']['max_attempts_per_request']
    return {'study_id': 'EXP-007', 'protocol_sha256': digest(protocol), 'planned_case_count': 3080,
            'planned_unique_requests': len(requests), 'full_retry_reservation_usd': str(first * maximum),
            'maximum_attempts': len(requests) * maximum,
            'unique_request_payloads': len(requests), 'explicit_alias_count': len(aliases) - len(requests),
            'maximum_attempts_including_retries': len(requests) * maximum,
            'requests': [{'id': r['id'], 'cache_key': key, **r['estimate']} for key, r in requests.items()],
            'aliases': aliases, 'cost_estimates_usd': {'nominal_single_attempt': str(nominal),
                'single_attempt_reservation': str(first), 'all_attempts_reservation': str(first * maximum)},
            'pricing': PRICING, 'inference_calls': 0,
            'note': 'Conditional byte envelope: 2*serialized UTF-8 bytes+8192 input, 128 output, all input at cache-write rate. No cache discount or invoice guarantee.'}


def source_code():
    code = general.code_record()
    from .final_protocol import runtime_code
    code['files_sha256'].update(runtime_code())
    return code


def _safe_output(output):
    output = Path(output).expanduser().absolute()
    require(not any(p.is_symlink() for p in [output, *output.parents]), 'Symlink output path is forbidden')
    if output.exists():
        require(output.is_dir() and not any(p.is_symlink() for p in output.rglob('*')),
                'Output must be a directory without symlinks')
    return output


def _read_entry(output, key, request, protocol, kind):
    path = output / 'responses' / (key + '.json')
    identity = {'study_id': 'EXP-007', 'kind': kind, 'id': request['id'], 'cache_key': key,
                'request': request['body'], 'protocol_sha256': digest(protocol)}
    if not path.exists():
        return {**identity, 'attempts': []}
    entry = read_json(path)
    require(all(entry.get(k) == v for k, v in identity.items()), 'Cache request/protocol identity changed')
    attempts = entry['attempts']
    policy, bound = protocol['fallback']['retry_policy'], request['estimate']
    require(isinstance(attempts, list) and len(attempts) <= policy['max_attempts_per_request'], 'Retry allowance exceeded')
    for number, attempt in enumerate(attempts, 1):
        require(attempt['attempt'] == number and attempt['reservation_usd'] == bound['attempt_reservation_usd'],
                'Durable attempt/reservation changed')
        require(type(attempt['started_epoch']) in (float, int) and math.isfinite(attempt['started_epoch']), 'Invalid attempt time')
        if number > 1:
            require(attempts[number - 2].get('retryable') and not attempts[number - 2].get('halt'),
                    'A successful/terminal request was retried')
        if 'response' not in attempt:
            require(attempt['status'] in ('reserved', 'interrupted_unknown', 'transport_unknown', 'client_error') and
                    attempt.get('predicted_label') is None and attempt.get('usage_priced_usd') is None,
                    'Missing response for a terminal request')
            if attempt['status'] == 'client_error':
                require(attempt.get('halt') is True and attempt.get('retryable') is False, 'Changed fatal client error')
            elif attempt['status'] != 'reserved':
                require(attempt.get('retryable') is True, 'Changed interrupted/transport retry policy')
        else:
            response = attempt['response']
            require(digest(response) == attempt['response_sha256'], 'Saved response hash mismatch')
            expected = recovery.outcome(response, protocol['population']['labels'])
            cost = general.priced_response(response['body'])
            require(attempt['usage_priced_usd'] == (str(cost) if cost is not None else None), 'Saved usage/accounting changed')
            data = response['body']
            if isinstance(data, dict):
                require(attempt.get('usage') == data.get('usage') and
                        attempt.get('returned_model') == data.get('model') and
                        attempt.get('returned_service_tier') == data.get('service_tier'),
                        'Saved returned-model/usage metadata changed')
            if isinstance(data, dict) and general.usage_breaches(data.get('usage'), bound, cost):
                expected['halt'] = True
                require(attempt.get('budget_envelope_breached_or_usage_malformed') is True, 'Missing budget halt')
            require(all(attempt.get(k) == v for k, v in expected.items()), 'Saved response outcome changed')
            require(attempt['retry_after_seconds'] == recovery.retry_after_seconds(response.get('headers'), attempt['finished_epoch']),
                    'Saved Retry-After changed')
        if 'finished_epoch' in attempt:
            require(type(attempt['finished_epoch']) in (float, int) and math.isfinite(attempt['finished_epoch']) and
                    attempt['finished_epoch'] >= attempt['started_epoch'], 'Invalid response completion time')
    return entry


def _manifest(rows, protocol, aliases, cap, full_reservation, authorization, kind):
    return {'study_id': 'EXP-007', 'kind': kind, 'protocol_sha256': digest(protocol),
            'requests_sha256': digest(rows), 'aliases': aliases, 'pricing': PRICING,
            'cap_usd': str(cap), 'full_retry_reservation_usd': str(full_reservation), 'code': source_code(),
            'authorization': {'approved_protocol_sha256': authorization.approved_protocol_sha256,
                'authorize_test_access': authorization.authorize_test_access, 'authorize_live': authorization.authorize_live}}


def _require_live_preflight(output, rows, protocol, plan, cap):
    """A direct live API entry requires the durable, completed upstream preflight."""
    from .final_protocol import runtime_code
    from .final_specialists import verify_saved_specialists
    parent = output.parent
    manifest = read_json(parent / 'manifest.json')
    require(manifest == {'study_id': 'EXP-007', 'protocol_sha256': digest(protocol),
                        'rows_sha256': digest(rows), 'code_files_sha256': runtime_code(),
                        'cap_usd': str(cap), 'test_sha256': protocol['population']['sealed_file_sha256_from_existing_metadata']},
            'Live collector requires matching authorized dataset/code/cap preflight')
    require(read_json(parent / 'protocol.json') == protocol and read_json(parent / 'preflight.json') == plan,
            'Missing or changed complete preflight before paid collection')
    verify_saved_specialists(rows, protocol, parent)


def collect(rows, protocol, prompt, schema, output, authorization, *, transport=None,
            sleep=time.sleep, clock=time.time):
    """Return (execution report, UNIQUE-payload entries). Synthetic injection never bypasses authorization.

    A response is shared only when the entire frozen request is identical. Aliases explicitly
    retain each original ID; accounting counts unique attempts once, never once per seed/alias.
    """
    from .final_protocol import Authorization
    require(isinstance(authorization, Authorization), 'Explicit EXP-007 authorization object required')
    cap = authorization.validate(protocol, live=True)
    requests, aliases = _requests(rows, protocol, prompt, schema)
    plan = collection_plan(rows, protocol, prompt, schema)
    full_reservation = Decimal(plan['cost_estimates_usd']['all_attempts_reservation'])
    require(full_reservation <= cap, f'Insufficient cap before collection: reserve USD {full_reservation}')
    output = _safe_output(output)
    live = transport is None
    if live:
        require(output.resolve() == (ROOT / protocol['outputs']['root'] / 'luna').resolve(),
                'Live collection must use the one frozen output directory')
        _require_live_preflight(output, rows, protocol, plan, cap)
        require(bool(os.environ.get('OPENAI_API_KEY')), 'OPENAI_API_KEY must be set locally before collection')
        transport = recovery.post_recovery
    else:
        require(transport not in (recovery.post_recovery, general.post_openai), 'Real transport cannot masquerade as synthetic')
    kind = 'live' if live else 'synthetic'
    manifest = _manifest(rows, protocol, aliases, cap, full_reservation, authorization, kind)
    policy = protocol['fallback']['retry_policy']
    with general.lock(output):
        manifest_path = output / 'manifest.json'
        if manifest_path.exists():
            require(read_json(manifest_path) == manifest, 'Resume manifest/code/rows/authorization drift')
            require((output / 'reservations.json').exists(), 'Missing durable reservation journal')
        else:
            require(not any(p.name != '.lock' for p in output.iterdir()), 'New collection directory must be empty')
            general.atomic_json(manifest_path, manifest)
        history_path = output / 'authorization_history.json'
        history = read_json(history_path) if history_path.exists() else []
        history.append({'recorded_utc': general.now(), 'pricing_acknowledged_date': authorization.pricing_date,
                        'cap_usd': str(cap), **manifest['authorization']})
        general.atomic_json(history_path, history)
        entries = {key: _read_entry(output, key, request, protocol, kind) for key, request in requests.items()}
        general.sync_reservation_ledger(output, manifest, entries)
        for key, entry in entries.items():
            if entry['attempts'] and entry['attempts'][-1]['status'] == 'reserved':
                entry['attempts'][-1].update(status='interrupted_unknown', predicted_label=None, retryable=True)
                general.atomic_json(output / 'responses' / (key + '.json'), entry)
        halted = 'previous_fatal_attempt' if any(a.get('halt') for e in entries.values() for a in e['attempts']) else None
        # All committed reservations plus every still-permitted attempt must fit BEFORE transport.
        remaining = sum((Decimal(requests[key]['estimate']['attempt_reservation_usd']) *
                         (policy['max_attempts_per_request'] - len(entry['attempts']))
                         for key, entry in entries.items()
                         if not entry['attempts'] or entry['attempts'][-1].get('retryable')), Decimal(0))
        require(Decimal(general.accounting(entries)['committed_reservation_usd']) + remaining <= cap,
                'Remaining full-run reservation exceeds cap')
        for key, entry in entries.items():
            if halted:
                break
            request, path = requests[key], output / 'responses' / (key + '.json')
            while len(entry['attempts']) < policy['max_attempts_per_request']:
                if entry['attempts'] and not entry['attempts'][-1].get('retryable'):
                    break
                delay = recovery.next_delay(entries, key, clock())
                if delay > policy['maximum_single_wait_seconds']:
                    halted = 'cooldown_pause_resume_after_retry_after'
                    break
                if delay:
                    sleep(delay)
                    require(recovery.next_delay(entries, key, clock()) <= .001, 'Cooldown has not elapsed')
                bound = request['estimate']
                require(Decimal(general.accounting(entries)['committed_reservation_usd']) +
                        Decimal(bound['attempt_reservation_usd']) <= cap, 'Spending cap reached')
                authorization.validate(protocol, live=True)
                _validate(protocol, prompt, schema)
                require(cache_key(request['body']) == key, 'Frozen request drift before transport')
                attempt = {'attempt': len(entry['attempts']) + 1, 'status': 'reserved',
                           'started_utc': general.now(), 'started_epoch': clock(),
                           'reservation_usd': bound['attempt_reservation_usd'], 'usage_priced_usd': None,
                           'pre_attempt_delay_seconds': delay}
                entry['attempts'].append(attempt)
                general.atomic_json(path, entry)
                general.sync_reservation_ledger(output, manifest, entries)
                started = time.perf_counter()
                try:
                    response = transport(request['body'], policy['timeout_seconds'])
                except (TimeoutError, URLError, OSError, ConnectionError):
                    attempt.update(status='transport_unknown', predicted_label=None, retryable=True)
                except ValueError:
                    attempt.update(status='client_error', predicted_label=None, retryable=False, halt=True)
                else:
                    attempt.update(response=response, response_sha256=digest(response),
                                   **recovery.outcome(response, protocol['population']['labels']))
                    data = response['body']
                    if isinstance(data, dict):
                        attempt.update(returned_model=data.get('model'), returned_service_tier=data.get('service_tier'), usage=data.get('usage'))
                        cost = general.priced_response(data)
                        attempt['usage_priced_usd'] = str(cost) if cost is not None else None
                        if general.usage_breaches(data.get('usage'), bound, cost):
                            attempt.update(halt=True, budget_envelope_breached_or_usage_malformed=True)
                attempt.update(finished_epoch=clock(), finished_utc=general.now(), request_seconds=time.perf_counter() - started)
                attempt['retry_after_seconds'] = recovery.retry_after_seconds(attempt.get('response', {}).get('headers'), attempt['finished_epoch'])
                general.atomic_json(path, entry)
                if attempt.get('halt'):
                    halted = 'fatal_attempt_requires_review'
                    break
        predictions = [{'id': alias['id'], 'cache_key': alias['cache_key'], 'canonical_id': alias['canonical_id'],
                        **general.final_result(entries[alias['cache_key']])} for alias in aliases]
        accounting = general.accounting(entries)
        report = {'study_id': 'EXP-007', 'kind': kind, 'protocol_sha256': digest(protocol),
                  'status': 'halted' if halted else 'finished_with_unresolved' if any(p['status'] != 'ok' for p in predictions) else 'completed',
                  'halt_reason': halted, 'finished_utc': general.now(), 'predictions': predictions,
                  'aliases': aliases, 'unique_request_payloads': len(entries), 'accounting': accounting,
                  'unresolved_ids': [p['id'] for p in predictions if p['status'] != 'ok'],
                  'unresolved_count': sum(p['status'] != 'ok' for p in predictions),
                  'status_counts': dict(Counter(p['status'] for p in predictions))}
        if not live:
            report['notice'] = 'SYNTHETIC FIXTURES ONLY; NO BANKING77 TEST RESULT OR API SPEND'
            report['accounting']['actual_api_spend_usd'] = '0'
        general.atomic_json(output / 'execution.json', report)
        return report, entries


def verify_completed(rows, protocol, prompt, schema, output, authorization):
    """Pure cache verification for the prediction freeze: no transport, data, or labels.

    Return (report, unique-payload entries). Only terminal outcomes may be scored;
    an interrupted/unknown request is terminal only after all four reservations.
    """
    from .final_protocol import Authorization
    require(isinstance(authorization, Authorization), 'Explicit EXP-007 authorization required')
    cap = authorization.validate(protocol, live=True)
    requests, aliases = _requests(rows, protocol, prompt, schema)
    plan = collection_plan(rows, protocol, prompt, schema)
    output = _safe_output(output)
    require(output.is_dir(), 'Completed collection evidence is missing')
    with general.lock(output):
        manifest = read_json(output / 'manifest.json')
        kind = manifest.get('kind')
        require(kind in ('live', 'synthetic'), 'Unknown collection evidence kind')
        require(manifest == _manifest(rows, protocol, aliases, cap,
                Decimal(plan['full_retry_reservation_usd']), authorization, kind), 'Completed collection manifest changed')
        require((output / 'reservations.json').exists(), 'Completed reservation ledger is missing')
        entries = {key: _read_entry(output, key, request, protocol, kind) for key, request in requests.items()}
        general.sync_reservation_ledger(output, manifest, entries, write=False)
        limit = protocol['fallback']['retry_policy']['max_attempts_per_request']
        require(all(entry['attempts'] and not any(a.get('halt') for a in entry['attempts']) and
                    entry['attempts'][-1]['status'] != 'reserved' and
                    (not entry['attempts'][-1].get('retryable') or len(entry['attempts']) == limit)
                    for entry in entries.values()), 'Nonterminal collection cannot be scored')
        expected = [{'id': alias['id'], 'cache_key': alias['cache_key'], 'canonical_id': alias['canonical_id'],
                     **general.final_result(entries[alias['cache_key']])} for alias in aliases]
        accounting = general.accounting(entries)
        if kind == 'synthetic':
            accounting['actual_api_spend_usd'] = '0'
        report = read_json(output / 'execution.json')
        unresolved = [r['id'] for r in expected if r['status'] != 'ok']
        require(report['study_id'] == 'EXP-007' and report['kind'] == kind and
                report['protocol_sha256'] == digest(protocol) and report['predictions'] == expected and
                report['aliases'] == aliases and report['unique_request_payloads'] == len(entries) and
                report['accounting'] == accounting and report['unresolved_ids'] == unresolved and
                report['unresolved_count'] == len(unresolved) and
                report['status_counts'] == dict(Counter(p['status'] for p in expected)) and
                report['halt_reason'] is None and
                report['status'] == ('finished_with_unresolved' if unresolved else 'completed'),
                'Execution report differs from terminal durable response evidence')
        return report, entries
