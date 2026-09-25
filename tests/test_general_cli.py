"""Synthetic CLI/accounting tests; guarded against API keys and network activity."""
from copy import deepcopy
from datetime import date
from decimal import Decimal
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest

from baseline import general
from baseline.general_protocol import SETTINGS, PRICING, RETRIES, ENDPOINT, digest, payload, cache_key
from baseline.general_metrics import SEEDS, SHOTS
from baseline.selective import compact, selective_curve


@pytest.fixture(autouse=True)
def network_and_secret_guard(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Synthetic CLI tests must never execute a transport or access the network')
    monkeypatch.setattr(general, 'post_openai', forbidden)
    monkeypatch.setattr(general, 'build_opener', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    original_getitem = type(os.environ).__getitem__
    def guarded_getitem(mapping, key):
        if key in ('OPENAI_API_KEY', 'OPENAI_PROJECT_ID'):
            pytest.fail('Dry-run/evaluate/unauthorized live must not read API credentials')
        return original_getitem(mapping, key)
    monkeypatch.setattr(type(os.environ), '__getitem__', guarded_getitem)


@pytest.fixture
def synthetic_cli(monkeypatch, tmp_path):
    labels = ['synthetic_a', 'synthetic_b']
    prompt = 'Synthetic frozen instruction; return one declared intent.'
    schema = {'type': 'object', 'properties': {'intent': {'type': 'string', 'enum': labels}},
              'required': ['intent'], 'additionalProperties': False}
    protocol = {'labels': labels, 'settings': deepcopy(SETTINGS), 'pricing': deepcopy(PRICING),
                'retries': deepcopy(RETRIES), 'endpoint': ENDPOINT,
                'prompt_sha256': general.sha256(prompt.encode()), 'schema_sha256': digest(schema)}
    rows = [{'id': f'synthetic:{i}', 'text': f'synthetic customer message {i}', 'label': labels[i % 2]}
            for i in range(4)]
    ranked = [{'id': row['id'], 'true_label': row['label'], 'predicted_label': labels[0],
               'confidence': .9 - i / 10} for i, row in enumerate(rows)]
    runs = [{'run_id': f'synthetic-n{n}-s{seed}', 'shots': n, 'seed': seed,
             **compact(selective_curve(ranked, labels, .5))} for n in SHOTS for seed in SEEDS]
    provenance = {'synthetic_fixture': True, 'validation_ids': [row['id'] for row in rows]}
    calls = []
    def load_inputs():
        calls.append('load_inputs')
        return deepcopy(rows), list(labels), deepcopy(runs), deepcopy(provenance)
    monkeypatch.setattr(general, 'load_protocol', lambda: (deepcopy(protocol), prompt, deepcopy(schema)))
    monkeypatch.setattr(general, 'load_inputs', load_inputs)
    root = tmp_path / 'synthetic-worktree'
    root.mkdir()
    monkeypatch.setattr(general, 'ROOT', root)
    monkeypatch.setattr(subprocess, 'check_output', lambda *args, **kwargs: 'synthetic-git-head\n')
    monkeypatch.setattr(general.platform, 'platform', lambda: 'synthetic-platform')
    monkeypatch.setattr(general.platform, 'python_version', lambda: 'synthetic-python')
    monkeypatch.setattr(general.importlib.metadata, 'version', lambda name: 'synthetic-version')
    def forbidden_execute(*args, **kwargs):
        pytest.fail('Dry-run/evaluate/unauthorized live must not call execute')
    monkeypatch.setattr(general, 'execute', forbidden_execute)
    class FrozenDate(date):
        @classmethod
        def today(cls):
            return date.fromisoformat(PRICING['as_of'])
    monkeypatch.setattr(general, 'date', FrozenDate)
    return {'rows': rows, 'labels': labels, 'runs': runs, 'data': provenance, 'protocol': protocol,
            'prompt': prompt, 'schema': schema, 'calls': calls, 'root': root}


def invoke(monkeypatch, mode, output, *flags):
    monkeypatch.setattr(sys, 'argv', ['baseline.general', mode, '--output', str(output), *flags])
    general.main()


def test_dry_run_never_reads_credentials_or_executes_and_emits_frozen_request_plan(synthetic_cli, monkeypatch, capsys):
    data = synthetic_cli
    output = data['root'] / 'dry-plan'
    invoke(monkeypatch, 'dry-run', output)
    saved = json.loads((output / 'dry_run.json').read_text())
    printed = json.loads(capsys.readouterr().out)
    assert saved['status'] == 'PAID EXPERIMENT NOT RUN'
    assert saved['inference_calls'] == 0 and saved['actual_api_spend_usd'] == '0'
    assert saved['prompt'] == data['prompt'] and saved['schema'] == data['schema']
    assert saved['model'] == data['protocol']['settings']['model']
    assert saved['planned_unique_requests'] == len(data['rows'])
    assert saved['maximum_attempts_including_retries'] == len(data['rows']) * RETRIES['max_attempts_per_request']
    assert saved['settings'] == data['protocol']['settings']
    assert saved['protocol_sha256'] == digest(data['protocol'])
    assert printed['model'] == saved['model'] and printed['planned_unique_requests'] == 4
    assert data['calls'] == ['load_inputs']
    assert not (output / 'responses').exists()
    assert not (output / 'evaluation.json').exists()


@pytest.mark.parametrize('missing', ['authorization', 'cap', 'protocol_hash', 'pricing_acknowledgement'])
def test_live_requires_every_explicit_guard_before_inputs_credentials_or_transport(synthetic_cli, monkeypatch, missing):
    data = synthetic_cli
    groups = {
        'authorization': ['--authorize-live'],
        'cap': ['--max-spend-usd', '1'],
        'protocol_hash': ['--approve-protocol-sha256', digest(data['protocol'])],
        'pricing_acknowledgement': ['--acknowledge-pricing-date', PRICING['as_of']],
    }
    flags = [item for name, values in groups.items() if name != missing for item in values]
    output = data['root'] / 'artifacts' / 'not-authorized'
    with pytest.raises(ValueError):
        invoke(monkeypatch, 'live', output, *flags)
    assert data['calls'] == []
    assert not output.exists()


@pytest.fixture
def existing_synthetic_run(synthetic_cli, monkeypatch):
    data = synthetic_cli
    output = data['root'] / 'artifacts' / 'existing-synthetic-run'
    output.mkdir(parents=True)
    entries = {}
    for row in data['rows']:
        body = payload(row['text'], data['prompt'], data['schema'])
        entries[row['id']] = {'kind': 'live', 'id': row['id'], 'cache_key': cache_key(body), 'request': body,
                              'attempts': [{'status': 'ok', 'predicted_label': row['label'],
                                            'reservation_usd': '0.01', 'usage_priced_usd': '0.001'}]}
    manifest = {'kind': 'live', 'protocol_sha256': digest(data['protocol']), 'data': data['data'],
                'requests': [{'id': row_id, 'cache_key': entry['cache_key']} for row_id, entry in entries.items()],
                'cap_usd': '1', 'full_retry_reservation_usd': '0.08', 'protocol': data['protocol'],
                'code': {'synthetic_fixture': True}}
    (output / 'manifest.json').write_text(json.dumps(manifest))
    (output / 'reservations.json').write_text(json.dumps({'manifest_sha256': digest(manifest),
         'attempt_counts': {row_id: 1 for row_id in entries}}))
    reads = []
    def read_entry(actual_output, row_id, body, kind, labels):
        assert actual_output == output and kind == 'live' and labels == data['labels']
        assert body == entries[row_id]['request']
        reads.append(row_id)
        return deepcopy(entries[row_id])
    monkeypatch.setattr(general, 'read_entry', read_entry)
    return data, output, entries, reads


def test_evaluate_reuses_existing_responses_without_execute_transport_or_credentials(existing_synthetic_run, monkeypatch):
    data, output, entries, reads = existing_synthetic_run
    ledger_before = (output / 'reservations.json').read_bytes()
    invoke(monkeypatch, 'evaluate', output)
    assert (output / 'reservations.json').read_bytes() == ledger_before
    saved = json.loads((output / 'evaluation.json').read_text())
    assert set(reads) == set(entries)
    assert data['calls'] == ['load_inputs']
    assert saved['general']['accuracy'] == 1
    assert saved['protocol_sha256'] == digest(data['protocol'])
    assert saved['actual_experiment_accounting']['actual_api_spend_usd'] == '0.004'
    assert len(saved['combinations']) == 90
    zero = saved['combinations'][0]['hypothetical_routed_api_charges']
    full = saved['combinations'][-1]['hypothetical_routed_api_charges']
    assert Decimal(zero['replayed_usage_usd']) == Decimal('0.004')
    assert Decimal(full['replayed_usage_usd']) == 0
    assert full['total_system_cost_usd'] is None and full['total_system_savings_usd'] is None


@pytest.mark.parametrize('field', ['kind', 'protocol_sha256', 'data'])
def test_evaluate_rejects_changed_existing_provenance_before_reading_responses(existing_synthetic_run, monkeypatch, field):
    _, output, _, reads = existing_synthetic_run
    manifest = json.loads((output / 'manifest.json').read_text())
    manifest[field] = 'changed'
    (output / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='provenance'):
        invoke(monkeypatch, 'evaluate', output)
    assert reads == []
    assert not (output / 'evaluation.json').exists()


def test_routed_api_costs_separate_replay_from_actual_full_experiment_accounting():
    entries = {'a': {'attempts': [{'reservation_usd': '.02', 'usage_priced_usd': '.001'},
                                 {'reservation_usd': '.02', 'usage_priced_usd': '.002'}]},
               'b': {'attempts': [{'reservation_usd': '.02', 'usage_priced_usd': '.004'}]}}
    original_entries = deepcopy(entries)
    actual = general.accounting(entries)
    report = general.routed_api_costs({'combinations': [{'rejected_ids': ['a']}, {'rejected_ids': []}]}, entries)
    rejected = report['combinations'][0]['hypothetical_routed_api_charges']
    assert Decimal(actual['actual_api_spend_usd']) == Decimal('.007')
    assert Decimal(rejected['replayed_usage_usd']) == Decimal('.003')
    assert Decimal(rejected['no_discount_attempt_envelope_usd']) == Decimal('.04')
    assert entries == original_entries
    for combination in report['combinations']:
        cost = combination['hypothetical_routed_api_charges']
        assert cost['specialist_deployment_cost_usd'] is None
        assert cost['total_system_cost_usd'] is None
        assert cost['total_system_savings_usd'] is None
        assert cost['production_latency_seconds'] is None
        assert 'Local replay incurs no API charge' in cost['assumption']
    assert Decimal(report['combinations'][1]['hypothetical_routed_api_charges']['replayed_usage_usd']) == 0


@pytest.mark.parametrize('state', ['unknown_usage', 'unattempted'])
def test_unresolved_rejected_charge_is_unknown_never_assumed_free(state):
    entries = {'a': {'attempts': [] if state == 'unattempted' else [
        {'reservation_usd': '.02', 'usage_priced_usd': None}]}}
    result = general.routed_api_costs({'combinations': [{'rejected_ids': ['a']}]}, entries)
    cost = result['combinations'][0]['hypothetical_routed_api_charges']
    assert cost['replayed_usage_usd'] is None
    if state == 'unattempted':
        assert cost['requests_without_attempts'] == 1
        assert cost['replayed_usage_interval_usd'] is None
    else:
        assert [Decimal(value) for value in cost['replayed_usage_interval_usd']] == [Decimal(0), Decimal('.02')]


@pytest.mark.parametrize('change', ['missing_ledger', 'truncated_response', 'changed_ledger_provenance'])
def test_evaluate_rejects_missing_paid_reservations_instead_of_reporting_zero_spend(existing_synthetic_run, monkeypatch, change):
    _, output, entries, _ = existing_synthetic_run
    ledger_path = output / 'reservations.json'
    if change == 'missing_ledger':
        ledger_path.unlink()
    elif change == 'truncated_response':
        entries[next(iter(entries))]['attempts'] = []
    else:
        ledger = json.loads(ledger_path.read_text())
        ledger['manifest_sha256'] = 'changed'
        ledger_path.write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match='reservation|provenance'):
        invoke(monkeypatch, 'evaluate', output)
    assert not (output / 'evaluation.json').exists()
