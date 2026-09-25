"""EXP-006: offline dry run by default; explicitly authorized, capped live execution only."""
import argparse
from contextlib import contextmanager
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler

from .data import ROOT, json_bytes, read_json, sha256
from .general_protocol import (ENDPOINT, MODEL, PRICING, RETRIES, digest, estimate,
                               load_inputs, load_protocol, payload, cache_key, plan)
from .selective import require


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as handle:
        handle.write(json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


@contextmanager
def lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('This response cache is already in use') from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def spending_cap(value):
    try:
        cap = Decimal(value)
    except (InvalidOperation, TypeError):
        raise ValueError('A positive numeric USD spending cap is required') from None
    require(cap.is_finite() and cap > 0, 'A positive finite USD spending cap is required')
    return cap


def authorize(authorized, cap_value, approved_digest, frozen_digest, pricing_date):
    require(authorized, 'Live execution requires explicit --authorize-live')
    cap = spending_cap(cap_value)
    require(approved_digest == frozen_digest, 'Explicit approval must match the frozen protocol SHA-256')
    require(pricing_date == PRICING['as_of'], 'Acknowledge the frozen pricing date explicitly')
    require(0 <= (date.today() - date.fromisoformat(pricing_date)).days <= 7,
            'Pricing verification is stale; review official prices and protocol before paid execution')
    return cap


def usage_cost(usage):
    """Usage-priced charges, not an invoice; missing details are unknown, never zero."""
    if not isinstance(usage, dict):
        return None
    try:
        i, o = usage['input_tokens'], usage['output_tokens']
        c, w = usage['input_tokens_details']['cached_tokens'], usage['input_tokens_details']['cache_write_tokens']
        values = (i, o, c, w)
        require(all(type(v) is int and v >= 0 for v in values) and c+w <= i, 'Invalid usage counts')
        require(i <= 272000, 'Usage outside short-context price scope')
        reasoning = usage.get('output_tokens_details', {}).get('reasoning_tokens')
        require(reasoning is None or (type(reasoning) is int and 0 <= reasoning <= o), 'Invalid reasoning usage')
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    p = PRICING['per_million_tokens']
    return (Decimal(i-c-w)*Decimal(p['input']) + Decimal(c)*Decimal(p['cached_input']) +
            Decimal(w)*Decimal(p['cache_write']) + Decimal(o)*Decimal(p['output'])) / 1000000


def priced_response(body):
    if not isinstance(body, dict) or body.get('model') != MODEL or body.get('service_tier') != 'default':
        return None
    return usage_cost(body.get('usage'))


def parse_response(body, labels):
    result = {'status': 'invalid_response', 'predicted_label': None, 'retryable': False}
    if not isinstance(body, dict):
        return result
    if body.get('model') != MODEL:
        return {**result, 'status': 'model_mismatch', 'halt': True}
    if body.get('service_tier') != 'default':
        return {**result, 'status': 'tier_mismatch', 'halt': True}
    if body.get('status') != 'completed':
        return {**result, 'status': 'incomplete' if body.get('status') == 'incomplete' else 'failed_response'}
    texts = []
    if not isinstance(body.get('output'), list):
        return result
    for item in body['output']:
        if not isinstance(item, dict):
            return result
        if not isinstance(item.get('content', []), list):
            return result
        for part in item.get('content', []):
            if not isinstance(part, dict):
                return result
            if part.get('type') == 'refusal':
                return {**result, 'status': 'refusal'}
            if part.get('type') == 'output_text':
                texts.append(part.get('text'))
    if len(texts) != 1 or not isinstance(texts[0], str):
        return result
    try:
        value = json.loads(texts[0])
    except (ValueError, TypeError):
        return result
    if isinstance(value, dict) and set(value) == {'intent'} and value['intent'] in labels:
        return {**result, 'status': 'ok', 'predicted_label': value['intent']}
    return result


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def post_openai(body, timeout):
    """The ONLY network entry point. No SDK retries, redirects, tools or proxy overrides."""
    key = os.environ.get('OPENAI_API_KEY')
    require(bool(key), 'Set OPENAI_API_KEY securely in the local environment; never paste it into chat')
    headers = {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}
    if os.environ.get('OPENAI_PROJECT_ID'):
        headers['OpenAI-Project'] = os.environ['OPENAI_PROJECT_ID']
    request = Request(ENDPOINT, data=json_bytes(body), headers=headers, method='POST')
    opener = build_opener(ProxyHandler({}), NoRedirect())
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
        return {'http_status': response.code, 'request_id': response.headers.get('x-request-id'), 'body': data}


def attempt_outcome(response, labels):
    code = response['http_status']
    if code != 200:
        return {'status': f'http_{code}', 'predicted_label': None,
                'retryable': code in RETRIES['retry_http_statuses'],
                'halt': code not in RETRIES['retry_http_statuses']}
    return parse_response(response['body'], labels)


def final_result(entry):
    if not entry['attempts']:
        return {'status': 'not_attempted', 'predicted_label': None}
    last = entry['attempts'][-1]
    return {'status': last.get('status', 'interrupted_unknown'), 'predicted_label': last.get('predicted_label')}


def accounting(entries):
    known, unknown, reserved = Decimal(0), Decimal(0), Decimal(0)
    missing = 0
    envelope_valid = True
    for e in entries.values():
        for a in e['attempts']:
            if a.get('status') in ('model_mismatch', 'tier_mismatch') or a.get('budget_envelope_breached_or_usage_malformed'):
                envelope_valid = False
            bound = Decimal(a['reservation_usd'])
            reserved += bound
            charge = a.get('usage_priced_usd')
            if charge is None:
                missing += 1
                unknown += bound
            else:
                known += Decimal(charge)
    return {'recorded_attempts': sum(len(e['attempts']) for e in entries.values()),
            'usage_priced_api_charges_usd': str(known), 'unknown_charge_attempts': missing,
            'unknown_charge_reservation_usd': str(unknown), 'committed_reservation_usd': str(reserved),
            'actual_api_spend_usd': str(known) if missing == 0 and envelope_valid else None,
            'reservation_envelope_valid': envelope_valid,
            'spend_interval_under_frozen_assumptions_usd': [str(known), str(known+unknown)] if envelope_valid else None,
            'invoice_reconciled': False, 'note': 'Actual experiment usage priced at frozen rates; missing usage is reserved, not assumed free.'}


def usage_breaches(usage, bound, cost):
    if usage is None:
        return False  # Retain the entire reservation as unknown spend.
    if not isinstance(usage, dict):
        return True
    for name, limit in (('input_tokens', bound['input_token_reservation']),
                        ('output_tokens', bound['output_token_reservation'])):
        value = usage.get(name)
        if type(value) is not int or not 0 <= value <= limit:
            return True
    return cost is not None and cost > Decimal(bound['attempt_reservation_usd'])


def read_entry(output, rid, body, kind, labels):
    key = cache_key(body)
    path = output / 'responses' / (key + '.json')
    if not path.exists():
        return {'kind': kind, 'id': rid, 'cache_key': key, 'request': body, 'attempts': []}
    e = read_json(path)
    require(e['kind'] == kind and e['id'] == rid and e['cache_key'] == key and e['request'] == body,
            'Response cache alignment mismatch')
    require(isinstance(e['attempts'], list) and len(e['attempts']) <= RETRIES['max_attempts_per_request'], 'Retry count exceeded')
    bound = estimate(body)
    for number, a in enumerate(e['attempts'], 1):
        require(a['attempt'] == number and a['reservation_usd'] == bound['attempt_reservation_usd'], 'Changed budget reservation')
        if 'response' not in a:
            require(a['status'] in ('reserved', 'interrupted_unknown', 'transport_unknown') and
                    a.get('predicted_label') is None and a.get('usage_priced_usd') is None,
                    'Missing saved response for terminal result')
            require(a['status'] == 'reserved' or a.get('retryable') is True, 'Changed transport retry state')
            continue
        require(a['response_sha256'] == digest(a['response']), 'Cached response changed')
        expected = attempt_outcome(a['response'], labels)
        require(all(a.get(k) == v for k, v in expected.items()), 'Cached prediction/status changed')
        b = a['response']['body']
        cost = priced_response(b)
        require(a['usage_priced_usd'] == (str(cost) if cost is not None else None), 'Cached usage cost changed')
        if isinstance(b, dict) and usage_breaches(b.get('usage'), bound, cost):
            require(a.get('halt') is True, 'Missing persisted budget halt')
    return e


def sync_reservation_ledger(output, manifest, entries, *, write=True):
    """Independent durable attempt counts detect deleted/truncated cache files on resume."""
    path = output / 'reservations.json'
    current = {rid: len(e['attempts']) for rid, e in entries.items() if e['attempts']}
    if path.exists():
        saved = read_json(path)
        require(saved['manifest_sha256'] == digest(manifest), 'Reservation ledger provenance changed')
        for rid, count in saved['attempt_counts'].items():
            require(current.get(rid, 0) >= count, 'Missing/truncated cache for a durable paid reservation')
        for rid, count in current.items():
            previous = saved['attempt_counts'].get(rid, 0)
            require(count == previous or (count == previous+1 and entries[rid]['attempts'][-1]['status'] == 'reserved'),
                    'Unjournaled response attempts')
    else:
        require(all(len(e['attempts']) <= 1 and (not e['attempts'] or e['attempts'][-1]['status'] == 'reserved')
                    for e in entries.values()), 'Missing reservation ledger for completed attempts')
    if write:
        atomic_json(path, {'manifest_sha256': digest(manifest), 'attempt_counts': current})


def code_record():
    import subprocess
    paths = sorted((ROOT / 'baseline').glob('general*.py')) + [ROOT / 'pyproject.toml', ROOT / 'uv.lock']
    return {'git_head': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
            'files_sha256': {str(p.relative_to(ROOT)): sha256(p.read_bytes()) for p in paths},
            'python': platform.python_version(), 'platform': platform.platform(),
            'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'scikit-learn')},
            'http_client': 'Python standard library urllib; no SDK or hidden retries'}


def execute(rows, protocol, prompt, schema, output, cap, transport, *, kind, provenance, sleep=time.sleep, live_authorization=None):
    """Resumable execution engine. Tests inject a synthetic transport and kind='mock'."""
    require(kind in ('mock', 'live'), 'Unknown execution kind')
    require((kind == 'live') == (transport is post_openai), 'Mock and live transports cannot share a cache')
    cap = spending_cap(cap)
    if kind == 'live':
        require(isinstance(live_authorization, dict), 'Live engine requires explicit authorization evidence')
        checked = authorize(live_authorization.get('authorized'), str(cap),
                            live_authorization.get('approved_protocol_sha256'), digest(protocol),
                            live_authorization.get('acknowledged_pricing_date'))
        require(checked == cap, 'Authorization spending cap mismatch')
    bodies = {row['id']: payload(row['text'], prompt, schema, protocol['settings']) for row in rows}
    require(len(bodies) == len(rows) and rows, 'Unique nonempty request IDs required')
    require(len({cache_key(b) for b in bodies.values()}) == len(rows), 'Duplicate request cache keys')
    total = sum((Decimal(estimate(b)['attempt_reservation_usd']) * RETRIES['max_attempts_per_request']
                 for b in bodies.values()), Decimal(0))
    require(total <= cap, f'Cap insufficient before launch: full retry reservation requires USD {total}')
    manifest = {'kind': kind, 'protocol_sha256': digest(protocol), 'data': provenance,
                'requests': [{'id': i, 'cache_key': cache_key(b)} for i, b in bodies.items()],
                'cap_usd': str(cap), 'full_retry_reservation_usd': str(total),
                'protocol': protocol, 'code': code_record(), 'live_authorization': live_authorization}
    with lock(output):
        path = output / 'manifest.json'
        if path.exists():
            require(read_json(path) == manifest, 'Cache provenance/configuration/cap changed; refusing reuse')
            require((output / 'reservations.json').exists(), 'Missing durable reservation ledger; restore it before resuming')
        else:
            atomic_json(path, manifest)
        entries = {rid: read_entry(output, rid, body, kind, protocol['labels']) for rid, body in bodies.items()}
        sync_reservation_ledger(output, manifest, entries)
        for e in entries.values():
            if e['attempts'] and e['attempts'][-1]['status'] == 'reserved':
                e['attempts'][-1].update(status='interrupted_unknown', retryable=True, predicted_label=None)
                atomic_json(output / 'responses' / (e['cache_key'] + '.json'), e)
        halted = None
        # A recorded model/tier/usage mismatch remains halted on resume; never skip it and continue.
        if any(any(a.get('halt') for a in e['attempts']) for e in entries.values()):
            halted = 'previous_fatal_attempt_requires_review'
        for rid, body in bodies.items():
            if halted:
                break
            e = entries[rid]
            p = output / 'responses' / (e['cache_key'] + '.json')
            while len(e['attempts']) < RETRIES['max_attempts_per_request']:
                if e['attempts']:
                    last = e['attempts'][-1]
                    if last['status'] == 'reserved':
                        last.update(status='interrupted_unknown', retryable=True, predicted_label=None)
                        atomic_json(p, e)
                    if not last.get('retryable', False):
                        break
                    sleep(RETRIES['backoff_seconds'])
                bound = estimate(body)
                reserved = Decimal(accounting(entries)['committed_reservation_usd'])
                require(reserved + Decimal(bound['attempt_reservation_usd']) <= cap, 'Spending cap reached before request')
                a = {'attempt': len(e['attempts'])+1, 'started_utc': now(), 'status': 'reserved',
                     'reservation_usd': bound['attempt_reservation_usd'], 'usage_priced_usd': None}
                e['attempts'].append(a)
                atomic_json(p, e)  # Reserve durably BEFORE any network activity, including crash/timeout cases.
                sync_reservation_ledger(output, manifest, entries)
                started = time.perf_counter()
                try:
                    response = transport(body, RETRIES['timeout_seconds'])
                except (TimeoutError, URLError, ConnectionError, OSError):
                    a.update(status='transport_unknown', predicted_label=None, retryable=True)
                else:
                    a.update(response=response, response_sha256=digest(response), **attempt_outcome(response, protocol['labels']))
                    result = response['body']
                    if isinstance(result, dict):
                        a['returned_model'] = result.get('model')
                        a['returned_service_tier'] = result.get('service_tier')
                        a['usage'] = result.get('usage')
                        cost = priced_response(result)
                        a['usage_priced_usd'] = str(cost) if cost is not None else None
                        if usage_breaches(a['usage'], bound, cost):
                            a['halt'] = True
                            a['budget_envelope_breached_or_usage_malformed'] = True
                a.update(finished_utc=now(), request_seconds=time.perf_counter()-started)
                atomic_json(p, e)
                if a.get('halt'):
                    halted = 'fatal_attempt_requires_review'
                    break
                if not a.get('retryable', False):
                    break
        predictions = [{'id': r['id'], **final_result(entries[r['id']])} for r in rows]
        report = {'study_id': 'EXP-006', 'kind': kind,
                  'status': 'halted' if halted else 'finished_with_unresolved' if any(p['status'] != 'ok' for p in predictions) else 'completed',
                  'halt_reason': halted, 'finished_utc': now(), 'predictions': predictions,
                  'accounting': accounting(entries), 'protocol_sha256': digest(protocol)}
        if kind == 'mock':
            report['notice'] = 'SYNTHETIC MOCK RESPONSES ONLY; NOT A BANKING77 MODEL RESULT OR ACTUAL SPEND'
            report['accounting']['actual_api_spend_usd'] = '0'
        atomic_json(output / 'execution.json', report)
        return report, entries


def routed_api_costs(evaluation, entries):
    """Counterfactual API component only; no specialist or total-system savings."""
    for combination in evaluation['combinations']:
        rejected = {i: entries[i] for i in combination['rejected_ids']}
        ac = accounting(rejected)
        unattempted = sum(not e['attempts'] for e in rejected.values())
        combination['hypothetical_routed_api_charges'] = {
            'replayed_usage_usd': ac['actual_api_spend_usd'] if not unattempted else None,
            'requests_without_attempts': unattempted,
            'replayed_usage_interval_usd': ac['spend_interval_under_frozen_assumptions_usd'] if not unattempted else None,
            'assumption': 'Reuse observed per-request cache read/write mix and retries; routing order could change server caching. Local replay incurs no API charge.',
            'specialist_deployment_cost_usd': None, 'total_system_cost_usd': None,
            'total_system_savings_usd': None, 'production_latency_seconds': None}
        cold_upper = sum((Decimal(a['reservation_usd']) for e in rejected.values() for a in e['attempts']), Decimal(0))
        combination['hypothetical_routed_api_charges']['no_discount_attempt_envelope_usd'] = str(cold_upper)
    return evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('dry-run', 'live', 'evaluate'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--authorize-live', action='store_true')
    parser.add_argument('--max-spend-usd')
    parser.add_argument('--approve-protocol-sha256')
    parser.add_argument('--acknowledge-pricing-date')
    args = parser.parse_args()
    protocol, prompt, schema = load_protocol()
    cap = None
    if args.mode == 'live':
        cap = authorize(args.authorize_live, args.max_spend_usd, args.approve_protocol_sha256,
                        digest(protocol), args.acknowledge_pricing_date)
        require(args.output.resolve().is_relative_to((ROOT / 'artifacts').resolve()), 'Paid response artifacts must stay under ignored artifacts/')
        require(bool(os.environ.get('OPENAI_API_KEY')), 'OPENAI_API_KEY must be configured locally; never paste keys into chat')
    rows, labels, runs, provenance = load_inputs()
    require(labels == protocol['labels'], 'Frozen protocol labels changed')
    dry = plan(rows, protocol, prompt, schema, provenance)
    if args.mode == 'dry-run':
        args.output.mkdir(parents=True, exist_ok=False)
        dry['generated_utc'] = now()
        dry['runtime'] = {'python': platform.python_version(), 'platform': platform.platform(),
                          'http_client': 'Python standard library urllib; no SDK/automatic retries',
                          'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'scikit-learn', 'pytest')}}
        import subprocess
        dry['source'] = {'git_head': subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
            'files_sha256': {str(p.relative_to(ROOT)): sha256(p.read_bytes()) for p in
                sorted((ROOT / 'baseline').glob('general*.py')) + sorted((ROOT / 'tests').glob('test_general*.py'))}}
        atomic_json(args.output / 'dry_run.json', dry)
        print(json.dumps({k: dry[k] for k in ('status','model','planned_unique_requests','maximum_attempts_including_retries','cost_estimates_usd','protocol_sha256')}, indent=2))
        return
    if args.mode == 'evaluate':
        require((args.output / 'manifest.json').exists(), 'Missing existing live run')
        cap = spending_cap(read_json(args.output / 'manifest.json')['cap_usd'])
        # Reading/evaluating must not call execute: it could retry unresolved requests.
        manifest = read_json(args.output / 'manifest.json')
        require(manifest['kind'] == 'live' and manifest['protocol_sha256'] == digest(protocol) and manifest['data'] == provenance, 'Evaluation provenance mismatch')
        entries = {row['id']: read_entry(args.output, row['id'], payload(row['text'], prompt, schema), 'live', labels)
                   for row in rows}
        require((args.output / 'reservations.json').exists(), 'Missing evaluation reservation ledger')
        sync_reservation_ledger(args.output, manifest, entries, write=False)
        report = {'predictions': [{'id': r['id'], **final_result(entries[r['id']])} for r in rows], 'accounting': accounting(entries)}
    else:
        report, entries = execute(rows, protocol, prompt, schema, args.output, cap, post_openai,
                                  kind='live', provenance=provenance, live_authorization={
                                      'authorized': args.authorize_live,
                                      'approved_protocol_sha256': args.approve_protocol_sha256,
                                      'acknowledged_pricing_date': args.acknowledge_pricing_date})
    from .general_metrics import evaluate
    evaluation = routed_api_costs(evaluate(report['predictions'], runs), entries)
    evaluation.update(study_id='EXP-006', protocol_sha256=digest(protocol), actual_experiment_accounting=report['accounting'])
    atomic_json(args.output / 'evaluation.json', evaluation)
    print('Evaluation saved; inspect execution status, unresolved cases and accounting before interpreting scores.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as exc:
        # Never print HTTP headers, request bodies, environment values or exception payloads.
        print(f'EXP-006 stopped: {type(exc).__name__}. Review frozen configuration, authorization, budget and local artifacts.', file=sys.stderr)
        raise SystemExit(1)
