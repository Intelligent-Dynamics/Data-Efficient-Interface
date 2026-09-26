"""Synthetic EXP-006R execution: no API, credentials, original cache, or dataset access."""
from copy import deepcopy
from datetime import date, datetime, timezone
from decimal import Decimal
from email.utils import format_datetime
import json
import os
from pathlib import Path
from urllib.error import URLError

import pytest

from baseline import general, general_protocol, recovery
from baseline.data import sha256
from baseline.general_protocol import SETTINGS, PRICING, cache_key, digest, payload


def forbidden(*args, **kwargs):
    pytest.fail('Recovery runner test reached network, credentials, or raw/test data')


@pytest.fixture(autouse=True)
def no_external_access(monkeypatch):
    monkeypatch.setattr(recovery, 'post_recovery', forbidden)
    monkeypatch.setattr(recovery, 'build_opener', forbidden)
    monkeypatch.setattr(general, 'post_openai', forbidden)
    monkeypatch.setattr(general, 'execute', forbidden)
    monkeypatch.setattr(general_protocol, 'load_inputs', forbidden)
    monkeypatch.setattr('urllib.request.urlopen', forbidden)
    getitem = type(os.environ).__getitem__
    def guarded_environment(mapping, key):
        if key in ('OPENAI_API_KEY', 'OPENAI_PROJECT_ID'):
            forbidden()
        return getitem(mapping, key)
    monkeypatch.setattr(type(os.environ), '__getitem__', guarded_environment)
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        if '/data/raw/' in str(path) or '/data/processed/' in str(path):
            forbidden()
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)


class Clock:
    def __init__(self, start=1_700_000_000.0):
        self.start = self.time = start
        self.sleeps = []

    def __call__(self):
        return self.time

    def sleep(self, delay):
        assert delay >= 0
        self.sleeps.append(delay)
        self.time += delay


def success(label='a', headers=None):
    return {'http_status': 200, 'request_id': 'SYNTHETIC', 'headers': headers or {}, 'body': {
        'model': 'gpt-6-luna', 'service_tier': 'default', 'status': 'completed',
        'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': json.dumps({'intent': label})}]}],
        'usage': {'input_tokens': 100, 'input_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 10},
                  'output_tokens': 10, 'output_tokens_details': {'reasoning_tokens': 0}}}}


def retryable(headers=None, *, code=429):
    return {'http_status': code, 'request_id': 'SYNTHETIC-ERROR', 'headers': headers or {},
            'body': {'error': {'type': 'rate_limit_error', 'code': 'rate_limit_exceeded'}}}


@pytest.fixture
def case(tmp_path, monkeypatch):
    directory = tmp_path / 'original'
    directory.mkdir()
    (directory / 'sentinel.bin').write_bytes(b'IMMUTABLE SYNTHETIC SOURCE')
    predictions = [{'id': f'synthetic:{i:04d}', 'status': 'http_429' if i < 52 else 'ok',
                    'predicted_label': None if i < 52 else 'a'} for i in range(770)]
    eligible = [p['id'] for p in predictions if p['status'] == 'http_429']
    schema = {'type': 'object', 'properties': {'intent': {'type': 'string', 'enum': ['a', 'b']}},
              'required': ['intent'], 'additionalProperties': False}
    bodies = {rid: payload(f'Synthetic message {rid}', 'Synthetic classification.', schema) for rid in eligible}
    source = {'directory': str(directory), 'eligible_ids': eligible, 'bodies': bodies, 'labels': ['a', 'b'],
              'execution': {'kind': 'live', 'status': 'finished_with_unresolved', 'predictions': predictions},
              'source_protocol_sha256': 'a' * 64, 'tree_sha256': 'b' * 64,
              'files_sha256': {'sentinel.bin': sha256(b'IMMUTABLE SYNTHETIC SOURCE')}}
    protocol = {'eligible_ids': eligible.copy(), 'request_cache_keys': {rid: cache_key(body) for rid, body in bodies.items()},
                'pricing': deepcopy(PRICING), 'pricing_verified_date': recovery.PRICING_VERIFIED_DATE,
                'settings': deepcopy(SETTINGS), 'retry_policy': deepcopy(recovery.POLICY)}
    frozen = deepcopy(protocol)
    monkeypatch.setattr(recovery, 'make_protocol', lambda _: deepcopy(frozen))
    monkeypatch.setattr(recovery, 'source_code', lambda: {'synthetic_code_fixture': True})
    clock = Clock()
    return {'source': source, 'protocol': protocol, 'output': tmp_path / 'recovery', 'cap': '1',
            'kind': 'mock', 'clock': clock, 'sleep': clock.sleep}


def run(case, transport):
    return recovery.execute_recovery(**case, transport=transport)


def cache_file(case, rid=None):
    rid = rid or case['source']['eligible_ids'][0]
    key = cache_key(case['source']['bodies'][rid])
    path = case['output'] / 'responses' / (key + '.json')
    return path, json.loads(path.read_text())


def test_only_exact_52_eligible_requests_sent_serially_and_successes_resume_without_calls(case):
    before = deepcopy(case['source'])
    seen, times = [], []
    keys = {cache_key(body): rid for rid, body in case['source']['bodies'].items()}
    def transport(body, timeout):
        assert timeout == 60
        seen.append(keys[cache_key(body)])
        times.append(case['clock']())
        return success()
    report, entries = run(case, transport)
    assert report['status'] == 'completed'
    assert seen == case['source']['eligible_ids'] and len(seen) == len(set(seen)) == 52
    original_ok = {p['id'] for p in case['source']['execution']['predictions'] if p['status'] == 'ok'}
    assert len(original_ok) == 718 and original_ok.isdisjoint(seen)
    assert all(b - a >= 5 for a, b in zip(times, times[1:]))
    assert case['clock'].sleeps == [5] * 51
    assert report['accounting']['recorded_attempts'] == 52
    assert report['accounting']['actual_api_spend_usd'] == '0'
    assert 'SYNTHETIC' in report['notice']
    repeated, loaded = run(case, lambda *_: pytest.fail('Successful recovery was resent'))
    assert repeated['predictions'] == report['predictions'] and loaded == entries
    assert case['source'] == before
    assert (Path(case['source']['directory']) / 'sentinel.bin').read_bytes() == b'IMMUTABLE SYNTHETIC SOURCE'


@pytest.mark.parametrize('change', ['missing', 'extra_success', 'changed_body', 'changed_status', 'duplicate'])
def test_selection_rejects_any_changed_52_request_cohort_before_transport(case, change):
    bodies = case['source']['bodies']
    first = case['source']['eligible_ids'][0]
    if change == 'missing':
        bodies.pop(first)
    elif change == 'extra_success':
        bodies['synthetic:0052'] = deepcopy(bodies[first])
    elif change == 'changed_body':
        bodies[first]['max_output_tokens'] = 64
    elif change == 'changed_status':
        case['source']['execution']['predictions'][0]['status'] = 'ok'
    else:
        case['source']['eligible_ids'][-1] = first
    with pytest.raises(ValueError):
        run(case, lambda *_: pytest.fail('Invalid cohort reached transport'))
    assert not case['output'].exists()


@pytest.mark.parametrize('code', [408, 429, 500, 502, 503, 504])
def test_four_attempt_limit_backoff_and_exhausted_request_global_cooldown(case, code):
    first = cache_key(case['source']['bodies'][case['source']['eligible_ids'][0]])
    times = []
    def transport(body, _):
        times.append(case['clock']() - case['clock'].start)
        return retryable(code=code) if cache_key(body) == first else success()
    report, entries = run(case, transport)
    assert times[:5] == [0, 30, 90, 210, 450]
    assert len(times) == 4 + 51
    assert len(entries[case['source']['eligible_ids'][0]]['attempts']) == 4
    assert report['status'] == 'finished_with_unresolved'
    assert report['unresolved_ids'] == [case['source']['eligible_ids'][0]]
    assert report['accounting']['unknown_charge_attempts'] == 4
    assert all(len(e['attempts']) <= 4 for e in entries.values())
    run(case, lambda *_: pytest.fail('Exhausted retry allowance was reset'))


@pytest.mark.parametrize('error', [TimeoutError('synthetic'), URLError('synthetic'), ConnectionError('synthetic')])
def test_transport_unknown_attempts_are_bounded_and_remain_reserved(case, error):
    first = cache_key(case['source']['bodies'][case['source']['eligible_ids'][0]])
    counts = []
    def transport(body, _):
        if cache_key(body) == first:
            counts.append(1)
            raise error
        return success()
    report, entries = run(case, transport)
    assert len(counts) == 4
    assert report['accounting']['unknown_charge_attempts'] == 4
    assert all(a['usage_priced_usd'] is None for a in entries[case['source']['eligible_ids'][0]]['attempts'])


@pytest.mark.parametrize('headers,expected', [
    ({'Retry-After': '90'}, 90), ({'retry-after': '30.5'}, 30.5),
    ({'Retry-After': '10', 'Retry-After-Ms': '50000'}, 50),
    ({'Retry-After': '-1'}, 0), ({'Retry-After': 'NaN'}, 0),
    ({'Retry-After': 'Infinity'}, 0), ({'Retry-After': 'garbage'}, 0),
])
def test_retry_after_numeric_and_invalid_headers(headers, expected):
    assert recovery.retry_after_seconds(headers, 1_700_000_000) == expected


def test_retry_after_http_date_honors_server_clock_and_invalid_date_falls_back():
    received = 1_700_000_000
    when = format_datetime(datetime.fromtimestamp(received + 100, timezone.utc), usegmt=True)
    server = format_datetime(datetime.fromtimestamp(received - 20, timezone.utc), usegmt=True)
    assert recovery.retry_after_seconds({'Retry-After': when}, received) == 100
    assert recovery.retry_after_seconds({'Retry-After': when, 'Date': server}, received) == 120
    assert recovery.retry_after_seconds({'Retry-After': when, 'Date': 'invalid'}, received) == 100
    past = format_datetime(datetime.fromtimestamp(received - 100, timezone.utc), usegmt=True)
    assert recovery.retry_after_seconds({'Retry-After': past}, received) == 0


def test_retry_after_survives_exhausted_id_before_next_id(case):
    first = cache_key(case['source']['bodies'][case['source']['eligible_ids'][0]])
    times, number = [], 0
    def transport(body, _):
        nonlocal number
        times.append(case['clock']() - case['clock'].start)
        if cache_key(body) == first:
            number += 1
            return retryable({'Retry-After': '1000'} if number == 4 else {})
        return success()
    run(case, transport)
    assert times[:5] == [0, 30, 90, 210, 1210]


def test_persisted_retry_after_respected_after_process_interruption(case):
    calls = []
    def fail_once(*_):
        calls.append(case['clock']())
        return retryable({'Retry-After': '100'})
    def interrupt_wait(_):
        raise KeyboardInterrupt('Synthetic stop during cooldown')
    case['sleep'] = interrupt_wait
    with pytest.raises(KeyboardInterrupt):
        run(case, fail_once)
    assert len(calls) == 1
    case['clock'].time += 20
    case['sleep'] = case['clock'].sleep
    def finish(*_):
        calls.append(case['clock']())
        return success()
    report, _ = run(case, finish)
    assert calls[1] - calls[0] == 100
    assert case['clock'].sleeps[0] == 80
    assert len(calls) == 53 and report['status'] == 'completed'
    assert report['accounting']['recorded_attempts'] == 53


def test_long_retry_after_pauses_and_cannot_be_shortened_by_resume(case):
    report, _ = run(case, lambda *_: retryable({'Retry-After': '7200'}))
    assert report['halt_reason'] == 'cooldown_pause_resume_after_retry_after'
    assert report['accounting']['recorded_attempts'] == 1
    assert case['clock'].sleeps == []
    case['clock'].time += 100
    again, _ = run(case, lambda *_: pytest.fail('Long server cooldown was bypassed'))
    assert again['accounting']['recorded_attempts'] == 1
    case['clock'].time += 7100
    resumed, _ = run(case, lambda *_: success())
    assert resumed['status'] == 'completed' and resumed['accounting']['recorded_attempts'] == 53


@pytest.mark.parametrize('code,error_type', [
    ('credit_balance_exhausted', 'insufficient_quota'), ('credit_balance_exhausted', None),
    ('unknown', 'insufficient_quota'), ('organization_spend_limit_exceeded', None),
    ('project_spend_limit_exceeded', None), ('organization_usage_limit_exceeded', None),
])
def test_quota_or_billing_failure_halts_once_and_stays_halted(case, code, error_type):
    record = retryable()
    record['body']['error'] = {'code': code, 'type': error_type}
    calls = []
    def transport(*_):
        calls.append(1)
        return deepcopy(record)
    report, entries = run(case, transport)
    repeated, _ = run(case, lambda *_: pytest.fail('Billing failure was retried on resume'))
    assert len(calls) == 1
    assert report['status'] == repeated['status'] == 'halted'
    assert report['accounting']['recorded_attempts'] == 1
    assert sum(bool(e['attempts']) for e in entries.values()) == 1


def test_interrupted_reserved_attempt_consumes_budget_and_waits_conservatively(case):
    def interrupt(*_):
        _, entry = cache_file(case)
        assert entry['attempts'][0]['status'] == 'reserved'
        ledger = json.loads((case['output'] / 'reservations.json').read_text())
        assert ledger['attempt_counts'] == {case['source']['eligible_ids'][0]: 1}
        raise KeyboardInterrupt('Synthetic unknown dispatched attempt')
    with pytest.raises(KeyboardInterrupt):
        run(case, interrupt)
    start = case['clock']()
    calls = []
    def transport(*_):
        calls.append(case['clock']())
        return success()
    report, entries = run(case, transport)
    assert calls[0] - start >= 60 + 30
    first = entries[case['source']['eligible_ids'][0]]['attempts']
    assert [a['status'] for a in first] == ['interrupted_unknown', 'ok']
    assert report['accounting']['recorded_attempts'] == 53
    assert report['accounting']['unknown_charge_attempts'] == 1


@pytest.mark.parametrize('corruption', ['delete_cache', 'truncate_cache', 'delete_ledger', 'changed_response'])
def test_cached_budget_and_responses_cannot_be_reset_or_corrupted(case, corruption):
    run(case, lambda *_: success())
    path, entry = cache_file(case)
    if corruption == 'delete_cache':
        path.unlink()
    elif corruption == 'truncate_cache':
        entry['attempts'] = []
        path.write_text(json.dumps(entry))
    elif corruption == 'delete_ledger':
        (case['output'] / 'reservations.json').unlink()
    else:
        entry['attempts'][0]['response']['body']['usage']['output_tokens'] += 1
        path.write_text(json.dumps(entry))
    with pytest.raises(ValueError):
        run(case, lambda *_: pytest.fail('Corrupted evidence reached transport'))


@pytest.mark.parametrize('cap', [None, '0', '-1', 'NaN', 'Infinity', 'not numeric', '0.00001'])
def test_numeric_cap_and_full_attempt_envelope_checked_before_transport(case, cap):
    case['cap'] = cap
    with pytest.raises(ValueError):
        run(case, lambda *_: pytest.fail('Invalid/insufficient cap reached transport'))
    assert not case['output'].exists()


def test_new_authorization_hash_and_pricing_acknowledgement_are_required(case, monkeypatch):
    class FrozenDate(date):
        @classmethod
        def today(cls):
            return cls.fromisoformat(recovery.PRICING_VERIFIED_DATE)
    monkeypatch.setattr(recovery, 'date', FrozenDate)
    args = [True, '1', digest(case['protocol']), recovery.PRICING_VERIFIED_DATE, case['protocol']]
    assert recovery.authorize(*args) == Decimal('1')
    for index, value in [(0, False), (1, None), (2, case['source']['source_protocol_sha256']), (3, '2000-01-01')]:
        changed = args.copy()
        changed[index] = value
        with pytest.raises(ValueError):
            recovery.authorize(*changed)
    class StaleDate(FrozenDate):
        @classmethod
        def today(cls):
            return date.fromordinal(date.fromisoformat(recovery.PRICING_VERIFIED_DATE).toordinal() + 8)
    monkeypatch.setattr(recovery, 'date', StaleDate)
    with pytest.raises(ValueError, match='stale'):
        recovery.authorize(*args)


def test_live_engine_guard_cannot_reuse_mock_or_old_authorization(case):
    case['kind'] = 'live'
    with pytest.raises(ValueError, match='authorization'):
        run(case, recovery.post_recovery)
    with pytest.raises(ValueError, match='transports'):
        run(case, lambda *_: success())
    assert not case['output'].exists()


@pytest.mark.parametrize('relation', ['same', 'child', 'ancestor', 'symlink_source', 'symlink_other'])
def test_source_output_overlap_and_symlink_outputs_are_rejected(case, tmp_path, relation):
    source = Path(case['source']['directory'])
    if relation == 'same':
        output = source
    elif relation == 'child':
        output = source / 'new'
    elif relation == 'ancestor':
        output = source.parent
    else:
        output = tmp_path / 'link'
        target = source if relation == 'symlink_source' else tmp_path / 'separate'
        target.mkdir(exist_ok=True)
        output.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError):
        recovery.separate_paths(source, output)


def test_dry_run_cli_uses_only_saved_synthetic_source_without_keys_or_network(case, monkeypatch):
    source_before = deepcopy(case['source'])
    monkeypatch.setattr(recovery, 'load_source', lambda _: case['source'])
    monkeypatch.setattr(recovery, 'checked_protocol', lambda _: case['protocol'])
    monkeypatch.setattr('sys.argv', ['baseline.recovery', 'dry-run', '--source', case['source']['directory'],
                                   '--output', str(case['output'])])
    recovery.main()
    dry = json.loads((case['output'] / 'dry_run.json').read_text())
    assert dry['inference_calls'] == 0 and dry['actual_recovery_spend_usd'] == '0'
    assert dry['planned_unique_requests'] == 52 and dry['maximum_attempts'] == 208
    assert dry['original_successes_not_resent'] == 718 and dry['official_test_access'] is False
    assert dry['eligible_ids'] == case['source']['eligible_ids']
    assert case['source'] == source_before
    with pytest.raises(FileExistsError):
        recovery.main()


def test_merge_cli_writes_fresh_separate_output_and_never_resumes_original(case, monkeypatch, tmp_path):
    from baseline import recovery_merge
    recovery_directory = tmp_path / 'saved-synthetic-recovery'
    recovery_directory.mkdir()
    marker = recovery_directory / 'marker'
    marker.write_bytes(b'IMMUTABLE SYNTHETIC RECOVERY')
    report, entries, runs = {'fixture': 'report'}, {'fixture': 'entries'}, ['fixture-runs']
    merged = {'study_id': 'SYNTHETIC CLI INTEGRATION ONLY', 'n_examples': 770}
    monkeypatch.setattr(recovery, 'load_source', lambda _: case['source'])
    monkeypatch.setattr(recovery, 'checked_protocol', lambda _: case['protocol'])
    monkeypatch.setattr(recovery, 'load_completed_recovery', lambda source, protocol, directory: (report, entries))
    monkeypatch.setattr(recovery, 'load_specialists', lambda source: runs)
    called = []
    def offline(source, saved_report, saved_entries, saved_runs):
        assert source is case['source'] and saved_report is report and saved_entries is entries and saved_runs is runs
        called.append(1)
        return merged
    monkeypatch.setattr(recovery_merge, 'offline_evaluation', offline)
    monkeypatch.setattr('sys.argv', ['baseline.recovery', 'merge-evaluate', '--source', case['source']['directory'],
        '--recovery', str(recovery_directory), '--output', str(case['output'])])
    recovery.main()
    assert json.loads((case['output'] / 'evaluation.json').read_text()) == merged
    assert len(called) == 1 and marker.read_bytes() == b'IMMUTABLE SYNTHETIC RECOVERY'
    assert (Path(case['source']['directory']) / 'sentinel.bin').read_bytes() == b'IMMUTABLE SYNTHETIC SOURCE'
    with pytest.raises(FileExistsError):
        recovery.main()


@pytest.mark.parametrize('breach', ['oversized_input', 'oversized_output', 'malformed_usage'])
def test_usage_envelope_breaches_remain_halted_and_are_revalidated_from_cache(case, breach):
    rid = case['source']['eligible_ids'][0]
    body = case['source']['bodies'][rid]
    bound = general_protocol.estimate(body)
    response = success()
    usage = response['body']['usage']
    if breach == 'oversized_input':
        usage['input_tokens'] = bound['input_token_reservation'] + 1
    elif breach == 'oversized_output':
        usage['output_tokens'] = bound['output_token_reservation'] + 1
    else:
        response['body']['usage'] = []
    calls = []
    def transport(*_):
        calls.append(1)
        return response
    report, entries = run(case, transport)
    assert calls == [1] and report['status'] == 'halted'
    assert report['halt_reason'] == 'fatal_attempt_requires_review'
    assert report['accounting']['recorded_attempts'] == 1
    assert report['accounting']['reservation_envelope_valid'] is False
    assert report['accounting']['spend_interval_under_frozen_assumptions_usd'] is None
    attempt = entries[rid]['attempts'][0]
    assert attempt['halt'] is True and attempt['budget_envelope_breached_or_usage_malformed'] is True
    repeated, loaded = run(case, lambda *_: pytest.fail('Usage breach resumed transport'))
    assert repeated['status'] == 'halted' and repeated['halt_reason'] == 'previous_fatal_attempt'
    assert repeated['accounting']['recorded_attempts'] == 1 and loaded == entries
    assert general.accounting(loaded)['actual_api_spend_usd'] is None
    path, saved = cache_file(case, rid)
    reader_args = (case['output'], rid, body, 'mock', case['protocol'], case['source']['labels'])
    assert recovery.read_entry(*reader_args) == saved
    # The raw response/hash remain intact: derived halt/evidence fields must still be rechecked.
    for field, message in [('halt', 'Recovery outcome changed'),
                           ('budget_envelope_breached_or_usage_malformed', 'Missing budget breach evidence')]:
        altered = deepcopy(saved)
        altered['attempts'][0].pop(field)
        path.write_text(json.dumps(altered))
        with pytest.raises(ValueError, match=message):
            recovery.read_entry(*reader_args)
    path.write_text(json.dumps(saved))


@pytest.fixture
def offline_reader_case(case):
    """Synthetic disk evidence only: change kind fields to exercise the offline live-record reader.

    Every response comes from the guarded mock engine. No live execution, authorization,
    network transport, or real artifacts are used by this fixture.
    """
    report, entries = run(case, lambda *_: success())
    output = case['output']
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['kind'] = 'live'
    manifest_path.write_text(json.dumps(manifest))
    for entry in entries.values():
        entry['kind'] = 'live'
        (output / 'responses' / (entry['cache_key'] + '.json')).write_text(json.dumps(entry))
    # Deliberately regenerate the temporary fixture ledger after changing its manifest.
    ledger = {'manifest_sha256': digest(manifest),
              'attempt_counts': {rid: len(entry['attempts']) for rid, entry in entries.items()}}
    (output / 'reservations.json').write_text(json.dumps(ledger))
    report['kind'] = 'live'
    report['accounting'] = general.accounting(entries)
    (output / 'execution.json').write_text(json.dumps(report))
    return case, report, entries


def test_completed_reader_validates_disk_evidence_without_writing_or_transport(offline_reader_case):
    case, report, entries = offline_reader_case
    paths = [p for root in (Path(case['source']['directory']), case['output'])
             for p in root.rglob('*') if p.is_file()]
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
    loaded_report, loaded_entries = recovery.load_completed_recovery(
        case['source'], case['protocol'], case['output'])
    assert loaded_report == report and loaded_entries == entries
    assert loaded_report['recovery_protocol_sha256'] == digest(case['protocol'])
    assert len(loaded_entries) == 52 and loaded_report['accounting']['recorded_attempts'] == 52
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths} == before


@pytest.mark.parametrize('corruption', [
    'manifest_protocol_hash', 'report_protocol_hash', 'report_predictions', 'report_accounting',
    'cached_response', 'ledger_attempt_count', 'missing_ledger',
])
def test_completed_reader_rejects_inconsistent_disk_evidence(offline_reader_case, corruption):
    case, _, _ = offline_reader_case
    output = case['output']
    if corruption in ('manifest_protocol_hash', 'report_protocol_hash'):
        path = output / ('manifest.json' if corruption.startswith('manifest') else 'execution.json')
        value = json.loads(path.read_text())
        value['recovery_protocol_sha256'] = case['source']['source_protocol_sha256']
    elif corruption.startswith('report'):
        path = output / 'execution.json'
        value = json.loads(path.read_text())
        if corruption == 'report_predictions':
            value['predictions'][0]['predicted_label'] = 'b'
        else:
            value['accounting']['recorded_attempts'] += 1
    elif corruption == 'cached_response':
        path, value = cache_file(case)
        value['attempts'][0]['response']['body']['usage']['input_tokens'] += 1
    else:
        path = output / 'reservations.json'
        value = json.loads(path.read_text())
        if corruption == 'missing_ledger':
            path.unlink()
        else:
            value['attempt_counts'][case['source']['eligible_ids'][0]] += 1
    if corruption != 'missing_ledger':
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        recovery.load_completed_recovery(case['source'], case['protocol'], output)
