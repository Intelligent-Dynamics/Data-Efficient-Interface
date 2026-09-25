"""Offline execution tests with synthetic responses; real network access is forbidden."""
from copy import deepcopy
from datetime import date
from decimal import Decimal
import json
from urllib.error import URLError

import pytest

from baseline import general
from baseline.general_protocol import MODEL, PRICING, RETRIES, SETTINGS, cache_key, payload


def forbidden_network(*args, **kwargs):
    raise AssertionError('A synthetic runner test attempted live network access')


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(general, 'post_openai', forbidden_network)
    monkeypatch.setattr(general, 'build_opener', forbidden_network)
    monkeypatch.setattr('urllib.request.urlopen', forbidden_network)


def response(intent='a', *, code=200):
    return {'http_status': code, 'request_id': 'synthetic-request', 'body': {
        'model': MODEL, 'service_tier': 'default', 'status': 'completed',
        'output': [{'type': 'message', 'content': [
            {'type': 'output_text', 'text': json.dumps({'intent': intent})}]}],
        'usage': {'input_tokens': 100,
                  'input_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 10},
                  'output_tokens': 10, 'output_tokens_details': {'reasoning_tokens': 0}},
    }}


@pytest.fixture
def case(tmp_path):
    return {'rows': [{'id': 'v:0', 'text': 'Synthetic message one'},
                     {'id': 'v:1', 'text': 'Synthetic message two'}],
            'protocol': {'settings': deepcopy(SETTINGS), 'labels': ['a', 'b']},
            'prompt': 'Choose exactly one synthetic intent.',
            'schema': {'type': 'object', 'properties': {'intent': {'type': 'string', 'enum': ['a', 'b']}},
                       'required': ['intent'], 'additionalProperties': False},
            'output': tmp_path / 'responses', 'cap': '1', 'kind': 'mock',
            'provenance': {}, 'sleep': lambda _: None}


def run(case, transport):
    return general.execute(**case, transport=transport)


def cached(case, rid='v:0'):
    row = next(r for r in case['rows'] if r['id'] == rid)
    body = payload(row['text'], case['prompt'], case['schema'], case['protocol']['settings'])
    path = case['output'] / 'responses' / (cache_key(body) + '.json')
    return path, json.loads(path.read_text())


def write_cache(path, entry):
    path.write_text(json.dumps(entry))


def test_successful_calls_keep_alignment_and_resume_without_transport(case):
    calls = []
    def transport(body, timeout):
        calls.append(body)
        assert timeout == RETRIES['timeout_seconds']
        assert body['model'] == MODEL and body['reasoning'] == {'effort': 'none'}
        assert body['store'] is False and body['service_tier'] == 'default'
        message = json.loads(body['input'][0]['content'][0]['text'])
        assert set(message) == {'customer_message'}
        return response('a' if message['customer_message'].endswith('one') else 'b')
    first, entries = run(case, transport)
    assert len(calls) == 2
    assert first['status'] == 'completed'
    assert first['predictions'] == [{'id': 'v:0', 'status': 'ok', 'predicted_label': 'a'},
                                    {'id': 'v:1', 'status': 'ok', 'predicted_label': 'b'}]
    assert first['accounting']['recorded_attempts'] == 2
    assert first['accounting']['actual_api_spend_usd'] == '0'
    assert 'SYNTHETIC' in first['notice']
    resumed, again = run(case, lambda *_: pytest.fail('Completed cache was called again'))
    assert resumed['predictions'] == first['predictions']
    assert again == entries
    assert resumed['accounting'] == first['accounting']


@pytest.mark.parametrize('change', ['text', 'ids', 'prompt', 'schema', 'settings', 'provenance', 'cap'])
def test_changed_cache_configuration_is_rejected_before_transport(case, change):
    run(case, lambda *_: response())
    if change == 'text':
        case['rows'][0]['text'] += ' changed'
    elif change == 'ids':
        case['rows'][0]['id'] = 'v:different'
    elif change == 'prompt':
        case['prompt'] += ' changed'
    elif change == 'schema':
        case['schema']['description'] = 'changed'
    elif change == 'settings':
        case['protocol']['settings']['max_output_tokens'] = 64
    elif change == 'provenance':
        case['provenance']['version'] = 'changed'
    else:
        case['cap'] = '2'
    with pytest.raises(ValueError, match='Cache provenance'):
        run(case, lambda *_: pytest.fail('Changed cache reached transport'))


@pytest.mark.parametrize('change', ['response', 'id', 'prediction', 'reservation', 'missing_response'])
def test_corrupt_cached_evidence_is_rejected(case, change):
    run(case, lambda *_: response())
    path, entry = cached(case)
    attempt = entry['attempts'][0]
    if change == 'response':
        attempt['response']['body']['output'][0]['content'][0]['text'] = '{"intent":"b"}'
    elif change == 'id':
        entry['id'] = 'v:different'
    elif change == 'prediction':
        attempt['predicted_label'] = 'b'
    elif change == 'reservation':
        attempt['reservation_usd'] = '0'
    else:
        del attempt['response']
        del attempt['response_sha256']
    write_cache(path, entry)
    with pytest.raises(ValueError):
        run(case, lambda *_: pytest.fail('Corrupt cache reached transport'))


def test_distinct_ids_with_identical_payloads_cannot_overwrite_cache(case):
    case['rows'][1]['text'] = case['rows'][0]['text']
    with pytest.raises(ValueError):
        run(case, lambda *_: pytest.fail('Duplicate payload reached transport'))


@pytest.mark.parametrize('outcome', ['refusal', 'incomplete', 'non_json', 'unknown_label', 'extra_field'])
def test_terminal_outputs_are_preserved_and_never_retried(case, outcome):
    case['rows'] = case['rows'][:1]
    record = response()
    body = record['body']
    expected = 'invalid_response'
    if outcome == 'refusal':
        body['output'][0]['content'] = [{'type': 'refusal', 'refusal': 'Synthetic refusal'}]
        expected = 'refusal'
    elif outcome == 'incomplete':
        body['status'] = 'incomplete'
        body['incomplete_details'] = {'reason': 'max_output_tokens'}
        expected = 'incomplete'
    elif outcome == 'non_json':
        body['output'][0]['content'][0]['text'] = 'not JSON'
    elif outcome == 'unknown_label':
        body['output'][0]['content'][0]['text'] = '{"intent":"missing"}'
    else:
        body['output'][0]['content'][0]['text'] = '{"intent":"a","reason":"extra"}'
    calls = []
    def transport(*_):
        calls.append(1)
        return deepcopy(record)
    report, _ = run(case, transport)
    again, _ = run(case, transport)
    assert len(calls) == 1
    assert report['status'] == 'finished_with_unresolved'
    assert report['predictions'] == [{'id': 'v:0', 'status': expected, 'predicted_label': None}]
    assert again['predictions'] == report['predictions']


@pytest.mark.parametrize('status', RETRIES['retry_http_statuses'])
def test_retryable_http_statuses_are_bounded_and_reserved(case, status):
    case['rows'] = case['rows'][:1]
    calls = []
    def transport(*_):
        calls.append(1)
        return {'http_status': status, 'request_id': 'synthetic-error', 'body': {'error': 'synthetic'}}
    report, entries = run(case, transport)
    run(case, lambda *_: pytest.fail('Exhausted retries called again'))
    assert len(calls) == RETRIES['max_attempts_per_request']
    assert report['predictions'][0]['status'] == f'http_{status}'
    attempts = entries['v:0']['attempts']
    assert [a['attempt'] for a in attempts] == [1, 2]
    assert report['accounting']['unknown_charge_attempts'] == 2
    assert Decimal(report['accounting']['committed_reservation_usd']) == sum(Decimal(a['reservation_usd']) for a in attempts)


@pytest.mark.parametrize('error', [TimeoutError(), URLError('synthetic'), ConnectionError('synthetic')])
def test_transport_failures_use_bounded_attempts_without_assuming_free_requests(case, error):
    case['rows'] = case['rows'][:1]
    calls = []
    def transport(*_):
        calls.append(1)
        raise error
    report, entries = run(case, transport)
    assert len(calls) == 2
    assert report['predictions'][0]['status'] == 'transport_unknown'
    assert report['accounting']['unknown_charge_attempts'] == 2
    assert all(a['usage_priced_usd'] is None for a in entries['v:0']['attempts'])


def test_retry_can_succeed_without_discarding_failed_attempt(case):
    case['rows'] = case['rows'][:1]
    sequence = iter([response(code=429), response('b')])
    waits = []
    case['sleep'] = waits.append
    report, entries = run(case, lambda *_: next(sequence))
    assert report['status'] == 'completed'
    assert report['predictions'][0]['predicted_label'] == 'b'
    assert [a['status'] for a in entries['v:0']['attempts']] == ['http_429', 'ok']
    assert waits == [RETRIES['backoff_seconds']]
    assert report['accounting']['recorded_attempts'] == 2


def test_nonretryable_http_error_halts_remaining_requests_and_resume(case):
    calls = []
    def transport(*_):
        calls.append(1)
        return response(code=401)
    report, _ = run(case, transport)
    resumed, _ = run(case, lambda *_: pytest.fail('Fatal HTTP failure resumed transport'))
    assert len(calls) == 1
    assert report['status'] == resumed['status'] == 'halted'
    assert report['predictions'][1]['status'] == 'not_attempted'


def test_interrupted_reserved_attempt_is_durable_and_consumes_retry_budget(case):
    case['rows'] = case['rows'][:1]
    def interrupted(*_):
        _, record = cached(case)
        assert record['attempts'][0]['status'] == 'reserved'
        assert Decimal(record['attempts'][0]['reservation_usd']) > 0
        raise KeyboardInterrupt('Synthetic interruption after reservation')
    with pytest.raises(KeyboardInterrupt):
        run(case, interrupted)
    report, entries = run(case, lambda *_: response())
    attempts = entries['v:0']['attempts']
    assert len(attempts) == 2
    assert [a['status'] for a in attempts] == ['interrupted_unknown', 'ok']
    assert report['accounting']['unknown_charge_attempts'] == 1
    assert Decimal(report['accounting']['committed_reservation_usd']) == sum(Decimal(a['reservation_usd']) for a in attempts)
    run(case, lambda *_: pytest.fail('Interrupted attempt was forgotten on resume'))


def test_tiny_cap_denies_execution_before_transport_or_cache_creation(case):
    case['cap'] = '0.000000001'
    with pytest.raises(ValueError, match='Cap insufficient'):
        run(case, lambda *_: pytest.fail('Budget denial reached transport'))
    assert not case['output'].exists()


@pytest.mark.parametrize('value', [None, '0', '-1', 'NaN', 'Infinity', 'not money'])
def test_invalid_spending_caps_are_rejected(value):
    with pytest.raises(ValueError):
        general.spending_cap(value)


def test_live_authorization_requires_all_explicit_fields_and_fresh_pricing(monkeypatch):
    class FrozenDate(date):
        @classmethod
        def today(cls):
            return cls.fromisoformat(PRICING['as_of'])
    monkeypatch.setattr(general, 'date', FrozenDate)
    args = [True, '1', 'frozen-sha', 'frozen-sha', PRICING['as_of']]
    assert general.authorize(*args) == Decimal('1')
    for index, value in [(0, False), (1, None), (2, 'other-sha'), (4, '2000-01-01')]:
        changed = list(args)
        changed[index] = value
        with pytest.raises(ValueError):
            general.authorize(*changed)
    class StaleDate(FrozenDate):
        @classmethod
        def today(cls):
            return date.fromordinal(date.fromisoformat(PRICING['as_of']).toordinal() + 8)
    monkeypatch.setattr(general, 'date', StaleDate)
    with pytest.raises(ValueError, match='stale'):
        general.authorize(*args)


def test_mock_cache_cannot_use_network_entrypoint_and_live_cannot_use_fake(case):
    with pytest.raises(ValueError, match='Mock and live'):
        run(case, general.post_openai)
    case['kind'] = 'live'
    with pytest.raises(ValueError, match='Mock and live'):
        run(case, lambda *_: response())


@pytest.mark.parametrize('field', ['output', 'content'])
def test_malformed_response_containers_are_terminal_invalid_not_crashes(case, field):
    case['rows'] = case['rows'][:1]
    record = response()
    if field == 'output':
        record['body']['output'] = None
    else:
        record['body']['output'][0]['content'] = None
    report, _ = run(case, lambda *_: deepcopy(record))
    assert report['predictions'][0]['status'] == 'invalid_response'
    assert report['accounting']['recorded_attempts'] == 1
    run(case, lambda *_: pytest.fail('Malformed terminal response retried'))


@pytest.mark.parametrize('usage', [[], {'input_tokens': 'bad', 'output_tokens': 10},
                                  {'input_tokens': 100, 'output_tokens': None}])
def test_malformed_usage_is_unknown_accounting_without_uncontrolled_retry(case, usage):
    case['rows'] = case['rows'][:1]
    record = response()
    record['body']['usage'] = usage
    report, entries = run(case, lambda *_: deepcopy(record))
    assert report['accounting']['recorded_attempts'] == 1
    assert entries['v:0']['attempts'][0]['usage_priced_usd'] is None
    run(case, lambda *_: pytest.fail('Malformed usage caused another paid attempt'))


def test_usage_cost_separates_cache_writes_and_does_not_double_count_reasoning():
    usage = {'input_tokens': 100, 'input_tokens_details': {'cached_tokens': 20, 'cache_write_tokens': 30},
             'output_tokens': 10, 'output_tokens_details': {'reasoning_tokens': 7}}
    # 50 ordinary*.10 +20 reads*.01 +30 writes*.125 +10 output*.50, per million.
    assert general.usage_cost(usage) == Decimal('0.00001395')
    usage['output_tokens_details']['reasoning_tokens'] = 0
    assert general.usage_cost(usage) == Decimal('0.00001395')
    assert general.usage_cost(None) is None
    del usage['input_tokens_details']['cache_write_tokens']
    assert general.usage_cost(usage) is None


@pytest.mark.parametrize('change', ['negative', 'bool', 'overlap', 'long_context', 'bad_reasoning', 'malformed_details'])
def test_invalid_usage_details_are_unknown(change):
    usage = response()['body']['usage']
    if change == 'negative':
        usage['input_tokens'] = -1
    elif change == 'bool':
        usage['input_tokens'] = True
    elif change == 'overlap':
        usage['input_tokens_details']['cached_tokens'] = 100
    elif change == 'long_context':
        usage['input_tokens'] = 272001
    elif change == 'bad_reasoning':
        usage['output_tokens_details']['reasoning_tokens'] = 11
    else:
        usage['output_tokens_details'] = []
    assert general.usage_cost(usage) is None


@pytest.mark.parametrize('change', ['model', 'tier', 'budget'])
def test_fatal_drift_halts_remaining_requests_persistently(case, change):
    record = response()
    if change == 'model':
        record['body']['model'] = 'unexpected-model'
    elif change == 'tier':
        record['body']['service_tier'] = 'priority'
    else:
        record['body']['usage']['output_tokens'] = SETTINGS['max_output_tokens'] + 1
    report, entries = run(case, lambda *_: deepcopy(record))
    resumed, _ = run(case, lambda *_: pytest.fail('Fatal drift was skipped on resume'))
    assert report['status'] == resumed['status'] == 'halted'
    assert report['accounting']['recorded_attempts'] == 1
    assert report['predictions'][1]['status'] == 'not_attempted'
    assert entries['v:0']['attempts'][0]['halt'] is True
    if change in ('model', 'tier'):
        assert entries['v:0']['attempts'][0]['usage_priced_usd'] is None


@pytest.mark.parametrize('change', ['delete', 'truncate'])
def test_durable_ledger_rejects_deleted_or_truncated_response_cache(case, change):
    run(case, lambda *_: response())
    path, entry = cached(case)
    if change == 'delete':
        path.unlink()
    else:
        entry['attempts'] = []
        write_cache(path, entry)
    with pytest.raises(ValueError, match='Missing/truncated cache'):
        run(case, lambda *_: pytest.fail('Lost paid reservation reached transport'))


def test_interruption_at_last_reserved_attempt_cannot_create_a_third_call(case):
    case['rows'] = case['rows'][:1]
    calls = []
    def transport(*_):
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError('Synthetic first attempt')
        ledger = json.loads((case['output'] / 'reservations.json').read_text())
        assert ledger['attempt_counts'] == {'v:0': 2}
        raise KeyboardInterrupt('Synthetic final reserved attempt')
    with pytest.raises(KeyboardInterrupt):
        run(case, transport)
    report, entries = run(case, lambda *_: pytest.fail('Final reserved attempt was not counted'))
    assert len(calls) == 2
    assert len(entries['v:0']['attempts']) == 2
    assert entries['v:0']['attempts'][-1]['status'] == 'interrupted_unknown'
    assert report['predictions'][0]['status'] == 'interrupted_unknown'
    assert report['accounting']['recorded_attempts'] == 2
    assert report['accounting']['unknown_charge_attempts'] == 2
    assert report['status'] == 'finished_with_unresolved'


def test_budget_breach_halt_cannot_be_removed_from_saved_attempt(case):
    record = response()
    record['body']['usage']['output_tokens'] = SETTINGS['max_output_tokens'] + 1
    run(case, lambda *_: deepcopy(record))
    path, entry = cached(case)
    entry['attempts'][0].pop('halt')
    write_cache(path, entry)
    with pytest.raises(ValueError, match='persisted budget halt'):
        run(case, lambda *_: pytest.fail('A hidden budget breach resumed transport'))


def test_live_cli_without_authorization_stops_before_reading_inputs(case, monkeypatch):
    monkeypatch.setattr(general, 'load_protocol', lambda: (case['protocol'], case['prompt'], case['schema']))
    monkeypatch.setattr(general, 'load_inputs', lambda: pytest.fail('Unauthorized live CLI read dataset'))
    monkeypatch.setattr('sys.argv', ['baseline.general', 'live', '--output', str(case['output'])])
    with pytest.raises(ValueError, match='authorize-live'):
        general.main()
    assert not case['output'].exists()


@pytest.mark.parametrize('authorization', [None, {}, {'authorized': False},
    {'authorized': True, 'approved_protocol_sha256': 'wrong', 'acknowledged_pricing_date': PRICING['as_of']}])
def test_live_engine_itself_requires_explicit_valid_authorization(case, authorization):
    case['kind'] = 'live'
    with pytest.raises(ValueError):
        general.execute(**case, transport=general.post_openai, live_authorization=authorization)
    assert not case['output'].exists()


def test_missing_reservation_ledger_cannot_resume_a_completed_cache(case):
    run(case, lambda *_: response())
    (case['output'] / 'reservations.json').unlink()
    with pytest.raises(ValueError, match='Missing durable reservation ledger'):
        run(case, lambda *_: pytest.fail('Missing ledger reached transport'))


def test_read_only_ledger_validation_preserves_files_and_rejects_lost_attempts(case):
    _, entries = run(case, lambda *_: response())
    manifest = json.loads((case['output'] / 'manifest.json').read_text())
    ledger_path = case['output'] / 'reservations.json'
    original = ledger_path.read_bytes()
    timestamp = ledger_path.stat().st_mtime_ns
    general.sync_reservation_ledger(case['output'], manifest, entries, write=False)
    assert ledger_path.read_bytes() == original
    assert ledger_path.stat().st_mtime_ns == timestamp
    entries['v:0']['attempts'] = []
    with pytest.raises(ValueError, match='Missing/truncated cache'):
        general.sync_reservation_ledger(case['output'], manifest, entries, write=False)
    assert ledger_path.read_bytes() == original
    assert ledger_path.stat().st_mtime_ns == timestamp
