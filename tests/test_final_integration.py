"""Actual EXP-007 collector/orchestrator integration using synthetic cases only.

No models, official data, API credentials or network are accessed. Three mocked
request payloads have 3080 explicit aliases, exercising whole-population handling.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket

import pytest

from baseline import final_collection as collection, final_test as runner
from baseline import recovery
from baseline.data import ROOT, read_json
from baseline.final_protocol import Authorization, FROZEN_PROTOCOL_SHA256, load_frozen
from baseline.general import atomic_json


def forbidden(*args, **kwargs):
    pytest.fail('Synthetic integration reached official data, model, credentials or network')


@pytest.fixture(autouse=True)
def no_external_access(monkeypatch):
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr('urllib.request.OpenerDirector.open', forbidden)
    monkeypatch.setattr(recovery, 'post_recovery', forbidden)
    monkeypatch.setattr(runner, 'verify_specialist_files', forbidden)
    monkeypatch.setattr(runner, 'unseal_inputs', forbidden)
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        if any(path.is_relative_to(ROOT / name) for name in ('data/raw', 'data/processed', '.cache')):
            forbidden()
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)
    original_get = type(os.environ).__getitem__
    def guarded_environment(mapping, key):
        if key in ('OPENAI_API_KEY', 'OPENAI_PROJECT_ID', 'OPENAI_ORG_ID'):
            forbidden()
        return original_get(mapping, key)
    monkeypatch.setattr(type(os.environ), '__getitem__', guarded_environment)


class Clock:
    def __init__(self):
        self.value = 1700000000.
        self.sleeps = []

    def __call__(self):
        return self.value

    def sleep(self, delay):
        self.sleeps.append(delay)
        self.value += delay


def success(label):
    return {'http_status': 200, 'request_id': 'SYNTHETIC ONLY', 'headers': {}, 'body': {
        'model': 'gpt-6-luna', 'service_tier': 'default', 'status': 'completed',
        'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': json.dumps({'intent': label})}]}],
        'usage': {'input_tokens': 100, 'output_tokens': 10,
                  'input_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 0},
                  'output_tokens_details': {'reasoning_tokens': 0}}}}


def refusal():
    response = success('unused')
    response['body']['output'] = [{'type': 'message', 'content': [{'type': 'refusal', 'refusal': 'SYNTHETIC refusal'}]}]
    return response


def rate_limit():
    return {'http_status': 429, 'request_id': 'SYNTHETIC ONLY', 'headers': {},
            'body': {'error': {'type': 'rate_limit_error', 'code': 'rate_limit_exceeded'}}}


def request_number(body):
    message = json.loads(body['input'][0]['content'][0]['text'])['customer_message']
    return int(message.rsplit(' ', 1)[1])


@pytest.fixture
def integration(tmp_path, monkeypatch):
    protocol, prompt, schema = load_frozen()
    labels = protocol['population']['labels']
    rows = [{'id': f'test:{i:05d}', 'text': f'SYNTHETIC shared request {i % 3}'} for i in range(3080)]
    truth = {row['id']: labels[i % 77] for i, row in enumerate(rows)}
    auth = Authorization(FROZEN_PROTOCOL_SHA256, True, True, '100',
                         datetime.now(timezone.utc).date().isoformat())
    plan = collection.collection_plan(rows, protocol, prompt, schema)
    monkeypatch.setattr(runner, 'prepare_authorized', lambda *args: (protocol, prompt, schema, rows, plan))
    monkeypatch.setattr(runner, 'runtime_code', lambda *args: {'SYNTHETIC': 'unchanged'})
    monkeypatch.setattr(collection, 'source_code', lambda: {'SYNTHETIC': 'unchanged'})
    output = tmp_path / protocol['outputs']['root']
    calls = {'specialists': 0, 'truth': 0}
    def fake_specialists(inputs, frozen, directory, **kwargs):
        assert inputs == rows and all(set(row) == {'id', 'text'} for row in inputs)
        calls['specialists'] += 1
        saved = {}
        for seed in frozen['specialists']['seeds']:
            values = [{'id': row['id'], 'predicted_label': labels[i % 77],
                       'confidence': .2 if i % 2 else .05, 'use_specialist': bool(i % 2)}
                      for i, row in enumerate(inputs)]
            saved[str(seed)] = values
            for name, value in [('predictions', values),
                                ('probabilities', {'notice': 'SYNTHETIC STUB: real probability validation is tested separately'}),
                                ('metadata', {'notice': 'SYNTHETIC STUB: no encoder/classifier was loaded'})]:
                runner._persist_exact(directory / f'specialists/{seed}/{name}.json', value)
        return saved
    def fake_truth(authorization, frozen, root, frozen_predictions_verified):
        assert frozen_predictions_verified is True
        runner.verify_frozen_predictions(output, frozen)
        calls['truth'] += 1
        return truth
    monkeypatch.setattr(runner, 'run_specialists', fake_specialists)
    monkeypatch.setattr(runner, 'scoring_truth', fake_truth)
    return {'root': tmp_path, 'protocol': protocol, 'authorization': auth, 'output': output,
            'calls': calls, 'rows': rows, 'labels': labels, 'clock': Clock()}


def execute(case, transport):
    return runner.execute(case['authorization'], case['root'], transport=transport,
                          sleep=case['clock'].sleep, clock=case['clock'])


def test_actual_collector_aliases_failures_and_freeze_keep_all_3080_once(integration):
    case, sent = integration, []
    def transport(body, timeout):
        assert timeout == 60
        assert case['calls']['truth'] == 0
        assert all((case['output'] / f'specialists/{seed}/predictions.json').exists()
                   for seed in case['protocol']['specialists']['seeds'])
        number = request_number(body)
        sent.append(number)
        return refusal() if number == 1 else success(case['labels'][number])
    result = execute(case, transport)
    assert sent == [0, 1, 2]
    expected_failures = sum(i % 3 == 1 for i in range(3080))
    assert result['n_examples'] == 3080 and result['unresolved_count'] == expected_failures
    assert result['luna_only']['n_examples'] == 3080
    assert result['luna_status_counts'] == {'ok': 3080 - expected_failures, 'refusal': expected_failures}
    assert result['api_accounting']['recorded_attempts'] == 3
    assert result['api_accounting']['actual_api_spend_usd'] == '0'
    execution = read_json(case['output'] / 'luna/execution.json')
    assert execution['unique_request_payloads'] == 3 and len(execution['aliases']) == 3080
    assert len({row['id'] for row in execution['predictions']}) == 3080
    assert len({row['cache_key'] for row in execution['predictions']}) == 3
    for seed in result['seeds'].values():
        assert seed['observed_coverage'] == .5 and seed['luna_request_percentage'] == 50
        assert len({p['id'] for p in seed['predictions']}) == 3080
        assert seed['fallback_unresolved_count'] == sum(i % 2 == 0 and i % 3 == 1 for i in range(3080))
    repeated = execute(case, lambda *_: pytest.fail('Completed shared Luna request was resent'))
    assert repeated == result
    assert case['calls'] == {'specialists': 1, 'truth': 1}


def test_actual_collector_resume_keeps_success_and_durable_unknown_attempt(integration):
    case, first = integration, []
    def interrupted(body, timeout):
        number = request_number(body)
        first.append(number)
        if number == 1:
            raise KeyboardInterrupt('SYNTHETIC interruption after paid reservation')
        return success(case['labels'][number])
    with pytest.raises(KeyboardInterrupt):
        execute(case, interrupted)
    assert first == [0, 1]
    assert case['calls']['truth'] == 0
    assert not (case['output'] / 'predictions_frozen.json').exists()
    resumed = []
    def finish(body, timeout):
        number = request_number(body)
        resumed.append(number)
        return success(case['labels'][number])
    result = execute(case, finish)
    assert resumed == [1, 2], 'Successful payload0 must never be resent'
    assert result['unresolved_count'] == 0
    assert result['api_accounting']['recorded_attempts'] == 4
    assert result['api_accounting']['unknown_charge_attempts'] == 1
    assert result['n_examples'] == 3080 and case['calls']['truth'] == 1
    assert case['clock'].sleeps[-2:] == [90, 5]
    execute(case, lambda *_: pytest.fail('Frozen resume reached transport'))
    assert case['calls']['truth'] == 1


def test_fourth_interrupted_attempt_is_bounded_failure_not_infinite_block(integration):
    case, first = integration, []
    def interrupt_fourth(body, timeout):
        assert request_number(body) == 0
        first.append(0)
        if len(first) == 4:
            raise KeyboardInterrupt('SYNTHETIC fourth reserved attempt interrupted')
        return rate_limit()
    with pytest.raises(KeyboardInterrupt):
        execute(case, interrupt_fourth)
    assert len(first) == 4 and case['calls']['truth'] == 0
    resumed = []
    def finish(body, timeout):
        number = request_number(body)
        assert number != 0, 'Four-attempt allowance cannot reset on resume'
        resumed.append(number)
        return success(case['labels'][number])
    result = execute(case, finish)
    assert resumed == [1, 2]
    exhausted_aliases = sum(i % 3 == 0 for i in range(3080))
    assert result['unresolved_count'] == exhausted_aliases
    assert result['luna_status_counts']['interrupted_unknown'] == exhausted_aliases
    assert result['api_accounting']['recorded_attempts'] == 6
    assert result['api_accounting']['unknown_charge_attempts'] == 4
    assert case['calls']['truth'] == 1
    assert result['collection_status'] == 'finished_with_unresolved'
    assert execute(case, lambda *_: pytest.fail('Exhausted/frozen requests were resent')) == result


@pytest.mark.parametrize('filename', ['evaluation.json', 'verification.json'])
def test_atomic_scoring_checkpoint_recovers_partial_publication_without_truth_or_calls(integration, monkeypatch, filename):
    case = integration
    atomic = runner.atomic_json
    def interrupted_write(path, value):
        if path == case['output'] / filename:
            raise KeyboardInterrupt('SYNTHETIC interruption after scoring checkpoint')
        return atomic(path, value)
    monkeypatch.setattr(runner, 'atomic_json', interrupted_write)
    sent = []
    def transport(body, timeout):
        sent.append(request_number(body))
        return success(case['labels'][request_number(body)])
    with pytest.raises(KeyboardInterrupt):
        execute(case, transport)
    assert sent == [0, 1, 2]
    assert (case['output'] / 'evaluation_checkpoint.json').exists()
    assert case['calls'] == {'specialists': 1, 'truth': 1}
    monkeypatch.setattr(runner, 'atomic_json', atomic)
    monkeypatch.setattr(runner, 'scoring_truth', forbidden)
    result = execute(case, lambda *_: pytest.fail('Checkpoint recovery reached API'))
    assert result['n_examples'] == 3080
    assert read_json(case['output'] / 'evaluation.json') == result
    assert (case['output'] / 'verification.json').exists()
    assert case['calls'] == {'specialists': 1, 'truth': 1}
