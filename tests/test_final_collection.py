"""Synthetic-only final-test collection; no dataset, credentials, model or API access."""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
from urllib.error import URLError

import pytest

from baseline import final_collection as runner, final_protocol, general, recovery
from baseline.data import ROOT, read_json
from baseline.general_protocol import cache_key, digest


def forbidden(*args, **kwargs):
    pytest.fail('Synthetic EXP-007 test reached data, credentials, or network')


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    monkeypatch.setattr(recovery, 'post_recovery', forbidden)
    monkeypatch.setattr(recovery, 'build_opener', forbidden)
    monkeypatch.setattr(general, 'post_openai', forbidden)
    monkeypatch.setattr('urllib.request.urlopen', forbidden)
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        if '/data/raw/' in str(path) or '/data/processed/' in str(path):
            forbidden()
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)
    getitem = type(os.environ).__getitem__
    def guarded_environment(mapping, key):
        if key in ('OPENAI_API_KEY', 'OPENAI_PROJECT_ID'):
            forbidden()
        return getitem(mapping, key)
    monkeypatch.setattr(type(os.environ), '__getitem__', guarded_environment)


class Clock:
    def __init__(self):
        self.time = 1_700_000_000.0
        self.sleeps = []
        self.elapsed = 0.
    def monotonic(self):
        return self.elapsed
    def __call__(self):
        return self.time
    def sleep(self, value):
        assert value >= 0
        self.sleeps.append(value)
        self.time += value
        self.elapsed += value


def success(label='card_arrival', **changes):
    body = {'model': 'gpt-6-luna', 'service_tier': 'default', 'status': 'completed',
            'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': json.dumps({'intent': label})}]}],
            'usage': {'input_tokens': 100, 'input_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 10},
                      'output_tokens': 10, 'output_tokens_details': {'reasoning_tokens': 0}}}
    body.update(changes)
    return {'http_status': 200, 'request_id': 'SYNTHETIC', 'headers': {}, 'body': body}


def retryable(code=429, headers=None):
    return {'http_status': code, 'request_id': 'SYNTHETIC-429', 'headers': headers or {},
            'body': {'error': {'type': 'rate_limit_error', 'code': 'rate_limit_exceeded'}}}


@pytest.fixture
def case(tmp_path, monkeypatch):
    protocol, prompt, schema = final_protocol.load_frozen()
    authorization = final_protocol.Authorization(final_protocol.FROZEN_PROTOCOL_SHA256, True, True,
                                                '1', datetime.now(timezone.utc).date().isoformat())
    monkeypatch.setattr(runner, 'source_code', lambda: {'synthetic_implementation_fixture': True})
    clock = Clock()
    return {'rows': [{'id': f'test:{i:05d}', 'text': 'SYNTHETIC customer ' + str(i % 2)} for i in range(3080)],
            'protocol': protocol, 'prompt': prompt, 'schema': schema, 'output': tmp_path / 'luna',
            'authorization': authorization, 'clock': clock, 'sleep': clock.sleep, 'monotonic': clock.monotonic}


def run(case, transport):
    return runner.collect(**case, transport=transport)


def entry_path(case, number=0):
    plan = runner.collection_plan(case['rows'], case['protocol'], case['prompt'], case['schema'])
    return case['output'] / 'responses' / (plan['requests'][number]['cache_key'] + '.json')


def test_plan_zero_calls_explicit_aliases_and_exact_payload_reservations(case):
    plan = runner.collection_plan(case['rows'], case['protocol'], case['prompt'], case['schema'])
    assert plan['inference_calls'] == 0 and plan['planned_case_count'] == 3080
    assert plan['planned_unique_requests'] == 2 and plan['explicit_alias_count'] == 3078
    assert len(plan['aliases']) == 3080 and plan['maximum_attempts'] == 8
    assert Decimal(plan['full_retry_reservation_usd']) == sum(Decimal(r['attempt_reservation_usd']) for r in plan['requests']) * 4
    assert not case['output'].exists()


def test_shared_collection_preserves_3080_ids_and_never_resends_successful_payload(case):
    times, sent = [], []
    def transport(body, timeout):
        assert timeout == 60
        assert 'test:' not in json.dumps(body)
        times.append(case['clock']())
        sent.append(cache_key(body))
        return success()
    report, entries = run(case, transport)
    assert len(sent) == len(set(sent)) == len(entries) == 2
    assert times[1] - times[0] == 5
    assert report['status'] == 'completed' and report['unresolved_count'] == 0
    assert [p['id'] for p in report['predictions']] == [f'test:{i:05d}' for i in range(3080)]
    assert len(set(p['id'] for p in report['predictions'])) == 3080
    assert report['accounting']['recorded_attempts'] == 2
    assert report['accounting']['actual_api_spend_usd'] == '0'
    assert 'SYNTHETIC' in report['notice']
    resumed, again = run(case, lambda *_: forbidden())
    assert resumed['predictions'] == report['predictions'] and again == entries


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'reorder', 'training_id', 'label', 'confidence', 'empty'])
def test_input_contract_rejects_labels_and_population_drift_before_calls(case, mutation):
    rows = case['rows']
    if mutation == 'missing':
        rows.pop()
    elif mutation == 'duplicate':
        rows[-1] = rows[0]
    elif mutation == 'reorder':
        rows.reverse()
    elif mutation == 'training_id':
        rows[0]['id'] = 'train:00000'
    elif mutation in ('label', 'confidence'):
        rows[0][mutation] = 'not allowed'
    else:
        rows[0]['text'] = ' '
    with pytest.raises(ValueError):
        run(case, lambda *_: forbidden())
    assert not case['output'].exists()


@pytest.mark.parametrize('changes', [
    {'authorize_test_access': False}, {'authorize_live': False}, {'spending_cap_usd': ''},
    {'spending_cap_usd': '0'}, {'spending_cap_usd': '-1'}, {'spending_cap_usd': 'NaN'},
    {'spending_cap_usd': 'Infinity'}, {'spending_cap_usd': True},
    {'approved_protocol_sha256': '0' * 64}, {'pricing_date': '2000-01-01'},
])
def test_complete_authorization_required_even_for_injected_transport(case, changes):
    case['authorization'] = replace(case['authorization'], **changes)
    with pytest.raises(ValueError):
        run(case, lambda *_: forbidden())
    assert not case['output'].exists()


def test_cap_covers_all_four_attempts_before_first_call(case):
    plan = runner.collection_plan(case['rows'], case['protocol'], case['prompt'], case['schema'])
    case['authorization'] = replace(case['authorization'], spending_cap_usd=plan['cost_estimates_usd']['single_attempt_reservation'])
    with pytest.raises(ValueError, match='Insufficient cap'):
        run(case, lambda *_: forbidden())
    assert not case['output'].exists()


@pytest.mark.parametrize('code', [408, 429, 500, 502, 503, 504])
def test_frozen_four_attempt_policy_serial_backoff_and_full_denominator(case, code):
    times = []
    def transport(body, _):
        times.append(case['clock']())
        return retryable(code) if len(times) <= 4 else success()
    report, entries = run(case, transport)
    assert [t - times[0] for t in times] == [0, 30, 90, 210, 450]
    assert sorted(len(e['attempts']) for e in entries.values()) == [1, 4]
    assert report['status'] == 'finished_with_unresolved' and report['unresolved_count'] == 1540
    assert report['status_counts'] == {f'http_{code}': 1540, 'ok': 1540}
    assert report['accounting']['unknown_charge_attempts'] == 4
    run(case, lambda *_: forbidden())


@pytest.mark.parametrize('error_code', recovery.POLICY['non_retryable_error_codes'])
def test_quota_halt_preserves_not_attempted_rows_on_resume(case, error_code):
    def quota(*_):
        return {'http_status': 429, 'body': {'error': {'code': error_code}}, 'headers': {}}
    report, entries = run(case, quota)
    assert report['status'] == 'halted' and report['accounting']['recorded_attempts'] == 1
    assert report['status_counts'] == {'http_429': 1540, 'not_attempted': 1540}
    resumed, _ = run(case, lambda *_: forbidden())
    assert resumed['predictions'] == report['predictions']


@pytest.mark.parametrize('response,expected,halt', [
    (success(status='incomplete'), 'incomplete', False),
    (success(output=[{'content': [{'type': 'refusal', 'refusal': 'SYNTHETIC refusal'}]}]), 'refusal', False),
    (success(label='NOT_AN_INTENT'), 'invalid_response', False),
    (success(model='different-model'), 'model_mismatch', True),
    (success(service_tier='priority'), 'tier_mismatch', True),
])
def test_terminal_outputs_not_retried_or_discarded(case, response, expected, halt):
    calls = []
    def transport(*_):
        calls.append(1)
        return deepcopy(response)
    report, _ = run(case, transport)
    assert len(calls) == (1 if halt else 2)
    assert report['status_counts'][expected] == (1540 if halt else 3080)
    assert report['unresolved_count'] == 3080 and len(report['predictions']) == 3080
    run(case, lambda *_: forbidden())


def test_reservations_fsynced_before_transport_and_interruption_does_not_reset_allowance(case):
    calls = []
    def interrupt_second(body, _):
        calls.append(cache_key(body))
        ledger = read_json(case['output'] / 'reservations.json')
        saved = read_json(case['output'] / 'responses' / (cache_key(body) + '.json'))
        assert saved['attempts'][-1]['status'] == 'reserved'
        assert ledger['attempt_counts'][cache_key(body)] == len(saved['attempts'])
        if len(calls) == 2:
            raise KeyboardInterrupt('SYNTHETIC interruption after durable reservation')
        return success()
    with pytest.raises(KeyboardInterrupt):
        run(case, interrupt_second)
    resumed_calls = []
    def resume(body, _):
        resumed_calls.append(cache_key(body))
        return success()
    report, entries = run(case, resume)
    assert resumed_calls == [calls[1]]  # First success cannot be sent again.
    assert sorted(len(e['attempts']) for e in entries.values()) == [1, 2]
    assert report['accounting']['unknown_charge_attempts'] == 1
    assert report['accounting']['recorded_attempts'] == 3
    assert case['clock'].sleeps == [5, 90]


@pytest.mark.parametrize('tamper', ['delete_cache', 'truncate_cache', 'delete_ledger', 'delete_manifest', 'response', 'reservation', 'status'])
def test_durable_evidence_rejects_deletion_truncation_and_tampering(case, tamper):
    run(case, lambda *_: success())
    path = entry_path(case)
    entry = read_json(path)
    if tamper == 'delete_cache':
        path.unlink()
    elif tamper == 'delete_ledger':
        (case['output'] / 'reservations.json').unlink()
    elif tamper == 'delete_manifest':
        (case['output'] / 'manifest.json').unlink()
    else:
        if tamper == 'truncate_cache':
            entry['attempts'] = []
        elif tamper == 'response':
            entry['attempts'][0]['response']['body']['model'] = 'changed'
        elif tamper == 'reservation':
            entry['attempts'][0]['reservation_usd'] = '0'
        else:
            entry['attempts'][0]['status'] = 'invalid_response'
        general.atomic_json(path, entry)
    with pytest.raises(ValueError):
        run(case, lambda *_: forbidden())


def test_retry_after_long_pause_can_resume_without_shortening_header(case):
    report, _ = run(case, lambda *_: retryable(headers={'Retry-After': '4000'}))
    assert report['halt_reason'] == 'cooldown_pause_resume_after_retry_after'
    assert report['accounting']['recorded_attempts'] == 1
    case['clock'].time += 4000
    report, _ = run(case, lambda *_: success())
    assert report['status'] == 'completed' and report['accounting']['recorded_attempts'] == 3


def test_retry_after_and_unknown_usage_survive_interrupted_wait(case):
    case['sleep'] = lambda _: (_ for _ in ()).throw(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        run(case, lambda *_: retryable(headers={'Retry-After': '100'}))
    case['clock'].time += 20
    case['sleep'] = case['clock'].sleep
    report, _ = run(case, lambda *_: success(usage=None))
    assert case['clock'].sleeps == [80, 5]
    assert report['accounting']['unknown_charge_attempts'] == 3
    assert Decimal(report['accounting']['unknown_charge_reservation_usd']) > 0


@pytest.mark.parametrize('mutation', ['protocol', 'threshold', 'classifier', 'prompt', 'schema', 'retry', 'settings'])
def test_frozen_configuration_drift_rejected_before_network(case, mutation):
    if mutation == 'prompt':
        case['prompt'] += 'changed'
    elif mutation == 'schema':
        case['schema']['additionalProperties'] = True
    elif mutation == 'threshold':
        case['protocol']['thresholds']['11']['value'] += .0001
    elif mutation == 'classifier':
        case['protocol']['thresholds']['11']['classifier_sha256'] = '0' * 64
    elif mutation == 'retry':
        case['protocol']['fallback']['retry_policy']['max_attempts_per_request'] = 5
    elif mutation == 'settings':
        case['protocol']['fallback']['settings']['max_output_tokens'] = 256
    else:
        case['protocol']['study_id'] = 'changed'
    with pytest.raises(ValueError):
        run(case, lambda *_: forbidden())
    assert not case['output'].exists()


def test_resuming_with_changed_input_or_code_is_rejected(case, monkeypatch):
    run(case, lambda *_: success())
    original_text = case['rows'][0]['text']
    case['rows'][0]['text'] = 'DIFFERENT SYNTHETIC INPUT'
    with pytest.raises(ValueError, match='Resume manifest'):
        run(case, lambda *_: forbidden())
    case['rows'][0]['text'] = original_text
    monkeypatch.setattr(runner, 'source_code', lambda: {'changed_code': True})
    with pytest.raises(ValueError, match='Resume manifest'):
        run(case, lambda *_: forbidden())


def test_real_transport_cannot_use_synthetic_output_or_injection_bypass(case):
    with pytest.raises(ValueError, match='one frozen output directory'):
        runner.collect(**case)
    with pytest.raises(ValueError, match='masquerade'):
        run(case, recovery.post_recovery)


def test_full_3080_distinct_payload_plan_keeps_every_original_id(case):
    for row in case['rows']:
        row['text'] += ' ' + row['id']
    plan = runner.collection_plan(case['rows'], case['protocol'], case['prompt'], case['schema'])
    assert plan['planned_unique_requests'] == 3080 and plan['maximum_attempts'] == 12320
    assert plan['explicit_alias_count'] == 0 and len(plan['requests']) == 3080
    assert [row['id'] for row in plan['requests']] == [row['id'] for row in case['rows']]


def test_usage_envelope_breach_halts_before_next_request(case):
    response = success()
    response['body']['usage']['output_tokens'] = 129
    report, _ = run(case, lambda *_: response)
    assert report['status'] == 'halted' and report['accounting']['recorded_attempts'] == 1
    assert report['accounting']['reservation_envelope_valid'] is False
    run(case, lambda *_: forbidden())


def test_transport_unknown_uses_only_four_reserved_attempts(case):
    def unknown(*_):
        raise URLError('SYNTHETIC timeout')
    report, _ = run(case, unknown)
    assert report['status_counts'] == {'transport_unknown': 3080}
    assert report['accounting']['recorded_attempts'] == 8
    assert report['accounting']['unknown_charge_attempts'] == 8
    run(case, lambda *_: forbidden())


def test_fresh_pricing_acknowledgement_can_resume_next_day_without_resending(case, monkeypatch):
    from datetime import timedelta
    run(case, lambda *_: success())
    tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
    class NextDay:
        @staticmethod
        def now(_):
            return tomorrow
    monkeypatch.setattr(final_protocol, 'datetime', NextDay)
    case['authorization'] = replace(case['authorization'], pricing_date=tomorrow.date().isoformat())
    report, _ = run(case, lambda *_: forbidden())
    assert report['status'] == 'completed'
    history = read_json(case['output'] / 'authorization_history.json')
    assert len(history) == 2 and history[0]['pricing_acknowledged_date'] != history[1]['pricing_acknowledged_date']


def test_symlink_output_rejected_before_writes_or_transport(case, tmp_path):
    directory = tmp_path / 'separate'
    directory.mkdir()
    case['output'].symlink_to(directory, target_is_directory=True)
    with pytest.raises(ValueError, match='Symlink'):
        run(case, lambda *_: forbidden())
    assert list(directory.iterdir()) == []


def test_serial_cooldown_cannot_be_shortened_by_sleep_implementation(case):
    case['sleep'] = lambda _: None
    calls = []
    def response(*_):
        calls.append(1)
        return success()
    report, _ = run(case, response)
    assert report['halt_reason'] == 'cooldown_pause_resume_after_retry_after'
    assert report['accounting']['recorded_attempts'] == 1
    assert len(calls) == 1


def test_mutated_usage_provenance_is_rejected_on_resume(case):
    run(case, lambda *_: success())
    path = entry_path(case)
    entry = read_json(path)
    entry['attempts'][0]['usage']['input_tokens'] += 1
    general.atomic_json(path, entry)
    with pytest.raises(ValueError):
        run(case, lambda *_: forbidden())


def test_completed_cache_replay_is_read_only_and_reproduces_all_rows(case):
    report, entries = run(case, lambda *_: success())
    before = {str(p): p.read_bytes() for p in case['output'].rglob('*.json')}
    checked, checked_entries = runner.verify_completed(case['rows'], case['protocol'], case['prompt'], case['schema'],
                                                       case['output'], case['authorization'])
    assert checked == report and checked_entries == entries
    assert before == {str(p): p.read_bytes() for p in case['output'].rglob('*.json')}


def test_completed_cache_replay_rejects_report_tampering(case):
    run(case, lambda *_: success())
    path = case['output'] / 'execution.json'
    report = read_json(path)
    report['predictions'][0]['predicted_label'] = 'card_linking'
    general.atomic_json(path, report)
    with pytest.raises(ValueError, match='Execution report differs'):
        runner.verify_completed(case['rows'], case['protocol'], case['prompt'], case['schema'],
                                case['output'], case['authorization'])


def test_fatal_collection_cannot_be_marked_complete_for_scoring(case):
    run(case, lambda *_: success(model='unapproved'))
    with pytest.raises(ValueError, match='Nonterminal'):
        runner.verify_completed(case['rows'], case['protocol'], case['prompt'], case['schema'],
                                case['output'], case['authorization'])


def test_exhausted_unknown_collection_is_complete_with_failures_not_dropped(case):
    def unknown(*_):
        raise TimeoutError('SYNTHETIC')
    run(case, unknown)
    report, _ = runner.verify_completed(case['rows'], case['protocol'], case['prompt'], case['schema'],
                                       case['output'], case['authorization'])
    assert report['unresolved_count'] == 3080 and len(report['predictions']) == 3080


def test_direct_live_call_requires_durable_parent_preflight_receipt(case, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    case['output'] = tmp_path / case['protocol']['outputs']['root'] / 'luna'
    with pytest.raises(FileNotFoundError):
        runner.collect(**case)
    assert not case['output'].exists()


@pytest.mark.parametrize('tamper', ['cap', 'rows', 'code', 'test_checksum', 'preflight', 'protocol'])
def test_live_receipt_drift_is_rejected_before_saved_models_or_transport(case, monkeypatch, tamper):
    from baseline import final_specialists
    monkeypatch.setattr(final_protocol, 'runtime_code', lambda: {'SYNTHETIC': 'code'})
    monkeypatch.setattr(final_specialists, 'verify_saved_specialists', forbidden)
    plan = runner.collection_plan(case['rows'], case['protocol'], case['prompt'], case['schema'])
    parent = case['output'].parent
    manifest = {'study_id': 'EXP-007', 'protocol_sha256': digest(case['protocol']),
                'rows_sha256': digest(case['rows']), 'code_files_sha256': {'SYNTHETIC': 'code'},
                'cap_usd': '1', 'test_sha256': case['protocol']['population']['sealed_file_sha256_from_existing_metadata']}
    protocol = deepcopy(case['protocol'])
    if tamper == 'cap':
        manifest['cap_usd'] = '0.1'
    elif tamper == 'rows':
        manifest['rows_sha256'] = '0' * 64
    elif tamper == 'code':
        manifest['code_files_sha256'] = {}
    elif tamper == 'test_checksum':
        manifest['test_sha256'] = '0' * 64
    elif tamper == 'preflight':
        plan['planned_unique_requests'] = 1
    else:
        protocol['study_id'] = 'wrong'
    general.atomic_json(parent / 'manifest.json', manifest)
    general.atomic_json(parent / 'protocol.json', protocol)
    general.atomic_json(parent / 'preflight.json', plan)
    correct_plan = runner.collection_plan(case['rows'], case['protocol'], case['prompt'], case['schema'])
    with pytest.raises(ValueError):
        runner._require_live_preflight(case['output'], case['rows'], case['protocol'], correct_plan, Decimal('1'))


def test_missing_key_does_not_create_a_paid_reservation(case, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    case['output'] = tmp_path / case['protocol']['outputs']['root'] / 'luna'
    monkeypatch.setattr(runner, '_require_live_preflight', lambda *a: None)
    # A synthetic empty environment; the real credential is never inspected.
    monkeypatch.setattr(runner.os, 'environ', {})
    with pytest.raises(ValueError, match='OPENAI_API_KEY must be set locally'):
        runner.collect(**case)
    assert not case['output'].exists()


def cooldown_entries(clock, *, retry_after=0, retryable=False, attempt=1):
    """Timing-only synthetic durable state, without any request or response text."""
    return {'saved': {'attempts': [{'attempt': attempt, 'status': 'http_429' if retryable else 'ok',
                                  'started_epoch': clock(), 'finished_epoch': clock(),
                                  'retryable': retryable, 'retry_after_seconds': retry_after}]}}


def wait(clock, entries, *, sleep=None, maximum=3600):
    return runner._wait_for_cooldown(entries, 'next-request', maximum,
                                    sleep=sleep or clock.sleep, clock=clock,
                                    monotonic=clock.monotonic)


def test_cooldown_rechecks_early_sleep_and_sleeps_only_remaining_time():
    clock = Clock()
    clock.time = 100.
    entries = cooldown_entries(clock)
    requested = []
    def early(delay):
        requested.append(delay)
        clock.sleep(delay / 2 if len(requested) <= 2 else delay)
    assert wait(clock, entries, sleep=early) == 5
    assert requested == [5, 2.5, 1.25]
    assert clock.elapsed == 5 and recovery.next_delay(entries, 'next-request', clock()) == 0


@pytest.mark.parametrize('residual,sleeps', [(.0005, 0), (.0009765625, 0), (.0015, 1), (.02, 1)])
def test_small_residual_cooldowns_respect_one_millisecond_tolerance(residual, sleeps):
    clock = Clock()
    clock.time = 0.
    entries = cooldown_entries(clock)
    clock.time = 5 - residual
    result = wait(clock, entries)
    assert result == pytest.approx(residual)
    assert len(clock.sleeps) == sleeps
    assert all(delay > .001 for delay in clock.sleeps)
    assert recovery.next_delay(entries, 'next-request', clock()) <= .001 + 1e-12


@pytest.mark.parametrize('wall_jump,expected', [(100., [5., 3.]), (-3., [5., 6.])])
def test_wall_clock_jumps_cannot_shorten_initial_monotonic_cooldown(wall_jump, expected):
    clock = Clock()
    clock.time = 100.
    entries = cooldown_entries(clock)
    requested = []
    def adjusted(delay):
        requested.append(delay)
        if len(requested) == 1:
            clock.sleep(2.)
            clock.time += wall_jump
        else:
            clock.sleep(delay)
    assert wait(clock, entries, sleep=adjusted) == 5.
    assert requested == expected
    assert clock.elapsed >= 5.
    assert recovery.next_delay(entries, 'next-request', clock()) == 0


@pytest.mark.parametrize('retry_after,attempt,expected', [(100, 1, 100), (10, 2, 60), (0, 4, 240)])
def test_cooldown_keeps_retry_after_and_exponential_backoff_across_ids(retry_after, attempt, expected):
    clock = Clock()
    entries = cooldown_entries(clock, retry_after=retry_after, retryable=True, attempt=attempt)
    assert wait(clock, entries) == expected
    assert clock.sleeps == [expected]
    assert recovery.next_delay(entries, 'another-request', clock()) == 0


def test_restart_reconstructs_remaining_retry_after_from_durable_timestamps(tmp_path):
    original_clock = Clock()
    entries = cooldown_entries(original_clock, retry_after=100, retryable=True)
    path = tmp_path / 'synthetic-timing.json'
    general.atomic_json(path, entries)
    before = path.read_bytes()
    restarted = Clock()
    restarted.time += 20
    assert restarted.elapsed == 0
    assert wait(restarted, read_json(path)) == 80
    assert restarted.sleeps == [80] and path.read_bytes() == before


@pytest.mark.parametrize('delay,expected_sleeps', [(3600, [3600]), (3601, [])])
def test_maximum_single_wait_boundary_still_pauses_without_shortening(delay, expected_sleeps):
    clock = Clock()
    result = wait(clock, cooldown_entries(clock, retry_after=delay))
    assert result == (delay if delay <= 3600 else None)
    assert clock.sleeps == expected_sleeps


def test_backward_wall_jump_cannot_extend_cumulative_wait_past_bound():
    clock = Clock()
    entries = cooldown_entries(clock)
    def backward(delay):
        clock.sleep(2)
        clock.time -= 20
    assert wait(clock, entries, sleep=backward, maximum=10) is None
    assert clock.elapsed == 2 and clock.sleeps == [2]
    assert recovery.next_delay(entries, 'next-request', clock()) == 23


def test_nonadvancing_sleeper_pauses_without_spinning():
    clock = Clock()
    calls = []
    assert wait(clock, cooldown_entries(clock), sleep=calls.append) is None
    assert calls == [5] and clock.elapsed == 0


def test_continually_interrupted_sleeper_is_bounded_without_zero_sleeps():
    clock = Clock()
    requested = []
    def interrupted(delay):
        requested.append(delay)
        clock.sleep(.00001)
    assert wait(clock, cooldown_entries(clock), sleep=interrupted) is None
    assert len(requested) == 64 and all(delay > .001 for delay in requested)
    assert clock.elapsed < .001


def test_resume_after_early_wait_preserves_success_and_only_dispatches_unattempted(case):
    sent = []
    def successful(body, _):
        sent.append(cache_key(body))
        return success()
    case['sleep'] = lambda _: (_ for _ in ()).throw(KeyboardInterrupt('SYNTHETIC wait interruption'))
    with pytest.raises(KeyboardInterrupt):
        run(case, successful)
    assert len(sent) == 1
    successful_path = case['output'] / 'responses' / (sent[0] + '.json')
    successful_bytes = successful_path.read_bytes()
    restarted = Clock()
    restarted.time = case['clock']() + 1
    sleeps = []
    def early(delay):
        sleeps.append(delay)
        restarted.sleep(delay / 2 if len(sleeps) == 1 else delay)
    case.update(clock=restarted, monotonic=restarted.monotonic, sleep=early)
    report, entries = run(case, successful)
    assert len(sent) == len(set(sent)) == 2
    assert sleeps == [4, 2]
    assert report['accounting']['recorded_attempts'] == 2
    assert report['status'] == 'completed' and all(len(e['attempts']) == 1 for e in entries.values())
    assert successful_path.read_bytes() == successful_bytes


def test_exact_one_millisecond_remaining_needs_no_sleep(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(recovery, 'next_delay', lambda *_: .001)
    assert wait(clock, {}) == .001
    assert clock.sleeps == []
