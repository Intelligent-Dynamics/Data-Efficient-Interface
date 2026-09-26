"""Synthetic immutable-source checks; no original artifacts, dataset or API access."""
from copy import deepcopy
from pathlib import Path
import json
import os

import pytest

from baseline import general, general_protocol, recovery_source as source
from baseline.data import json_bytes, sha256
from baseline.general_protocol import cache_key, digest, estimate, payload


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def predictions():
    return [{'id': f'synthetic:{i:04d}', 'status': 'http_429' if i % 15 == 0 else 'ok',
             'predicted_label': None if i % 15 == 0 else 'synthetic_intent'} for i in range(770)]


@pytest.fixture(autouse=True)
def no_external_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Recovery source tests must not execute inference, read the dataset, or use credentials')
    monkeypatch.setattr(general, 'post_openai', forbidden)
    monkeypatch.setattr(general, 'execute', forbidden)
    monkeypatch.setattr(general_protocol, 'load_inputs', forbidden)
    getitem = type(os.environ).__getitem__
    def guarded(mapping, key):
        if key in ('OPENAI_API_KEY', 'OPENAI_PROJECT_ID'):
            forbidden()
        return getitem(mapping, key)
    monkeypatch.setattr(type(os.environ), '__getitem__', guarded)


def test_inventory_includes_every_file_and_dotfile_without_writing(tmp_path):
    (tmp_path / 'responses').mkdir()
    (tmp_path / '.lock').write_bytes(b'')
    (tmp_path / 'responses' / '.hidden').write_bytes(b'hidden synthetic data')
    (tmp_path / 'manifest.json').write_bytes(b'synthetic manifest')
    before = {str(path): path.stat().st_mtime_ns for path in tmp_path.rglob('*') if path.is_file()}
    actual = source.file_inventory(tmp_path)
    assert actual == {'.lock': sha256(b''), 'responses/.hidden': sha256(b'hidden synthetic data'),
                      'manifest.json': sha256(b'synthetic manifest')}
    assert before == {str(path): path.stat().st_mtime_ns for path in tmp_path.rglob('*') if path.is_file()}
    assert source.file_inventory(tmp_path) == actual


@pytest.mark.parametrize('kind', ['file', 'directory', 'dangling', 'root'])
def test_inventory_rejects_symlinks(tmp_path, kind):
    tree = tmp_path / 'tree'
    tree.mkdir()
    target = tmp_path / 'target'
    if kind == 'directory':
        target.mkdir()
    elif kind != 'dangling':
        target.write_bytes(b'synthetic')
    if kind == 'root':
        link = tmp_path / 'linked-root'
        link.symlink_to(tree, target_is_directory=True)
        checked = link
    else:
        (tree / 'link').symlink_to(target, target_is_directory=kind == 'directory')
        checked = tree
    with pytest.raises(ValueError, match='symlink'):
        source.file_inventory(checked)


def test_eligible_selection_uses_recorded_status_only_and_preserves_execution_order():
    execution = {'kind': 'live', 'status': 'finished_with_unresolved', 'predictions': predictions()}
    eligible = source.select_eligible(execution)
    assert len(eligible) == 52
    assert eligible == [f'synthetic:{i:04d}' for i in range(0, 770, 15)]
    execution['predictions'].reverse()
    assert source.select_eligible(execution) == list(reversed(eligible))


@pytest.mark.parametrize('change', ['kind', 'status', 'count', 'duplicate', 'counts', 'other_failure', 'label_429', 'null_ok'])
def test_selection_rejects_any_changed_failure_population(change):
    execution = {'kind': 'live', 'status': 'finished_with_unresolved', 'predictions': predictions()}
    rows = execution['predictions']
    if change == 'kind':
        execution['kind'] = 'mock'
    elif change == 'status':
        execution['status'] = 'completed'
    elif change == 'count':
        rows.pop()
    elif change == 'duplicate':
        rows[1]['id'] = rows[0]['id']
    elif change == 'counts':
        rows[0].update(status='ok', predicted_label='synthetic_intent')
    elif change == 'other_failure':
        rows[0]['status'] = 'http_500'
    elif change == 'label_429':
        rows[0]['predicted_label'] = 'synthetic_intent'
    else:
        rows[1]['predicted_label'] = None
    with pytest.raises(ValueError):
        source.select_eligible(execution)


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    """Full 770-record synthetic fixture, using the real saved-response validators."""
    directory = tmp_path / 'synthetic-original'
    directory.mkdir()
    labels = [f'synthetic_intent_{i:02d}' for i in range(77)]
    prompt = 'Synthetic classification instructions.'
    schema = {'type': 'object', 'properties': {'intent': {'type': 'string', 'enum': labels}},
              'required': ['intent'], 'additionalProperties': False}
    protocol = {'labels': labels, 'settings': deepcopy(general_protocol.SETTINGS),
                'pricing': deepcopy(general_protocol.PRICING), 'retries': deepcopy(general_protocol.RETRIES),
                'endpoint': general_protocol.ENDPOINT, 'prompt_sha256': sha256(prompt.encode()),
                'schema_sha256': digest(schema)}
    frozen_digest = digest(protocol)
    monkeypatch.setattr(source, 'SOURCE_PROTOCOL_SHA256', frozen_digest)
    monkeypatch.setattr(source, 'load_protocol', lambda: (deepcopy(protocol), prompt, deepcopy(schema)))
    ids = [f'train:{i:05d}' for i in range(770)]
    reference = {'samples': {'validation_ids': ids, 'labels': labels},
                 'predictions': [{'id': row_id, 'true_label': labels[i // 10]} for i, row_id in enumerate(ids)],
                 'metadata': {'protocol_id': 'synthetic-val10-v2', 'split_manifest_sha256': 'synthetic_split_hash',
                              'source': {'revision': 'synthetic_dataset_revision',
                                         'files': {'train.csv': {'sha256': 'synthetic_train_hash'}}}}}
    reference_path = tmp_path / 'versioned/runs/exp004-minilm-v2-n5-s11.json'
    audit_path = tmp_path / 'versioned/analysis_audit.json'
    write(reference_path, reference)
    ref_hash = sha256(reference_path.read_bytes())
    write(audit_path, {'compact_records_sha256': {'runs/' + reference_path.name: ref_hash}})
    monkeypatch.setattr(source, 'REFERENCE', reference_path)
    monkeypatch.setattr(source, 'REFERENCE_AUDIT', audit_path)
    data = {'protocol_id': 'synthetic-val10-v2', 'split_manifest_sha256': 'synthetic_split_hash',
            'dataset_revision': 'synthetic_dataset_revision', 'source_train_sha256': 'synthetic_train_hash',
            'validation_ids_sha256': digest(ids), 'validation_text_sha256_by_id': {},
            'prior_run_hashes': {reference_path.name: {'exp004_sha256': ref_hash}},
            'official_test_access': False, 'new_labels': 0, 'additional_validation_labels': 770}
    entries, requests, final = {}, [], []
    for i, row_id in enumerate(ids):
        text = f'synthetic input {i}'
        body = payload(text, prompt, schema, protocol['settings'])
        key = cache_key(body)
        is_429 = i % 15 == 0
        if is_429:
            response = {'http_status': 429, 'body': {'error': {'type': 'synthetic_rate_limit'}}}
        else:
            response = {'http_status': 200, 'body': {
                'model': general_protocol.MODEL, 'service_tier': 'default', 'status': 'completed',
                'output': [{'content': [{'type': 'output_text', 'text': json.dumps({'intent': labels[i % 77]})}]}],
                'usage': {'input_tokens': 20, 'output_tokens': 4,
                          'input_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 0}}}}
        outcome = general.attempt_outcome(response, labels)
        cost = general.priced_response(response['body'])
        attempts = [{'attempt': attempt, 'reservation_usd': estimate(body)['attempt_reservation_usd'],
                     'response': deepcopy(response), 'response_sha256': digest(response), **outcome,
                     'usage_priced_usd': str(cost) if cost is not None else None}
                    for attempt in range(1, 3 if is_429 else 2)]
        entry = {'kind': 'live', 'id': row_id, 'cache_key': key, 'request': body, 'attempts': attempts}
        entries[row_id] = entry
        requests.append({'id': row_id, 'cache_key': key})
        final.append({'id': row_id, **general.final_result(entry)})
        data['validation_text_sha256_by_id'][row_id] = sha256(text.encode())
        write(directory / 'responses' / (key + '.json'), entry)
    manifest = {'kind': 'live', 'protocol_sha256': frozen_digest, 'protocol': protocol, 'data': data,
                'requests': requests}
    execution = {'kind': 'live', 'status': 'finished_with_unresolved', 'protocol_sha256': frozen_digest,
                 'predictions': final, 'accounting': general.accounting(entries)}
    ledger = {'manifest_sha256': digest(manifest), 'attempt_counts': {row_id: len(entry['attempts']) for row_id, entry in entries.items()}}
    write(directory / 'manifest.json', manifest)
    write(directory / 'execution.json', execution)
    write(directory / 'reservations.json', ledger)
    (directory / '.lock').touch()
    write(directory / 'evaluation.json', {'synthetic': 'retained unrelated source artifact'})
    return {'directory': directory, 'manifest': manifest, 'execution': execution, 'entries': entries,
            'ids': ids, 'labels': labels, 'reference': reference_path, 'audit': audit_path,
            'protocol_hash': frozen_digest}


def entry_path(fixture, index):
    key = fixture['manifest']['requests'][index]['cache_key']
    return fixture['directory'] / 'responses' / (key + '.json')


def test_load_source_audits_exact_770_and_selects_only_52_without_modifying_tree(source_tree):
    directory = source_tree['directory']
    before = source.file_inventory(directory)
    mtimes = {name: (directory / name).stat().st_mtime_ns for name in before}
    result = source.load_source(directory)
    assert result['directory'] == str(directory.absolute())
    assert result['files_sha256'] == before == source.file_inventory(directory)
    assert mtimes == {name: (directory / name).stat().st_mtime_ns for name in before}
    assert len(result['files_sha256']) == 775
    assert result['tree_sha256'] == digest(before)
    assert result['source_protocol_sha256'] == source_tree['protocol_hash']
    assert result['validation_ids'] == source_tree['ids']
    assert result['labels'] == source_tree['labels']
    assert len(result['entries']) == 770
    assert result['eligible_ids'] == source_tree['ids'][::15]
    assert list(result['bodies']) == source_tree['ids'][::15]
    assert set(result['bodies']).isdisjoint(row['id'] for row in result['execution']['predictions'] if row['status'] == 'ok')
    for row_id, body in result['bodies'].items():
        assert body == result['entries'][row_id]['request']
        assert len(result['entries'][row_id]['attempts']) == 2


@pytest.mark.parametrize('change', ['missing_response', 'changed_response_hash', 'changed_execution_prediction',
                                    'changed_text_hash', 'changed_request_body', 'changed_request_order',
                                    'changed_execution_order', 'changed_accounting', 'missing_ledger',
                                    'truncated_attempts', 'extra_response', 'changed_protocol', 'changed_reference'])
def test_source_tampering_fails_closed_without_additional_writes(source_tree, change):
    directory = source_tree['directory']
    if change == 'missing_response':
        entry_path(source_tree, 0).unlink()
    elif change == 'changed_response_hash':
        entry = deepcopy(source_tree['entries'][source_tree['ids'][0]])
        entry['attempts'][0]['response_sha256'] = 'changed'
        write(entry_path(source_tree, 0), entry)
    elif change == 'changed_execution_prediction':
        execution = deepcopy(source_tree['execution'])
        execution['predictions'][1]['predicted_label'] = source_tree['labels'][2]
        write(directory / 'execution.json', execution)
    elif change == 'changed_text_hash':
        manifest = deepcopy(source_tree['manifest'])
        manifest['data']['validation_text_sha256_by_id'][source_tree['ids'][0]] = 'changed'
        write(directory / 'manifest.json', manifest)
    elif change == 'changed_request_body':
        entry = deepcopy(source_tree['entries'][source_tree['ids'][0]])
        entry['request']['instructions'] += ' changed'
        write(entry_path(source_tree, 0), entry)
    elif change == 'changed_request_order':
        manifest = deepcopy(source_tree['manifest'])
        manifest['requests'].reverse()
        write(directory / 'manifest.json', manifest)
    elif change == 'changed_execution_order':
        execution = deepcopy(source_tree['execution'])
        execution['predictions'].reverse()
        write(directory / 'execution.json', execution)
    elif change == 'changed_accounting':
        execution = deepcopy(source_tree['execution'])
        execution['accounting']['actual_api_spend_usd'] = '0'
        write(directory / 'execution.json', execution)
    elif change == 'missing_ledger':
        (directory / 'reservations.json').unlink()
    elif change == 'truncated_attempts':
        entry = deepcopy(source_tree['entries'][source_tree['ids'][0]])
        entry['attempts'].pop()
        write(entry_path(source_tree, 0), entry)
    elif change == 'extra_response':
        write(directory / 'responses/extra.json', {'synthetic': True})
    elif change == 'changed_protocol':
        execution = deepcopy(source_tree['execution'])
        execution['protocol_sha256'] = 'changed'
        write(directory / 'execution.json', execution)
    else:
        reference = json.loads(source_tree['reference'].read_text())
        reference['samples']['validation_ids'].reverse()
        write(source_tree['reference'], reference)
    before = source.file_inventory(directory)
    with pytest.raises(ValueError):
        source.load_source(directory)
    assert source.file_inventory(directory) == before


def test_recovery_requires_two_429_attempts_not_just_final_429(source_tree):
    directory = source_tree['directory']
    entry = deepcopy(source_tree['entries'][source_tree['ids'][0]])
    response = {'http_status': 500, 'body': {'error': {'type': 'synthetic_transient'}}}
    entry['attempts'][0].update(response=response, response_sha256=digest(response),
                              **general.attempt_outcome(response, source_tree['labels']))
    write(entry_path(source_tree, 0), entry)
    with pytest.raises(ValueError, match='exactly two original HTTP 429'):
        source.load_source(directory)


def test_audit_detects_concurrent_source_tree_change(source_tree, monkeypatch):
    inventory = source.file_inventory
    calls = []
    def changed_on_second_scan(path):
        calls.append(path)
        found = inventory(path)
        if len(calls) == 2:
            found['synthetic-concurrent-file'] = 'synthetic-hash'
        return found
    monkeypatch.setattr(source, 'file_inventory', changed_on_second_scan)
    with pytest.raises(ValueError, match='changed during read-only audit'):
        source.load_source(source_tree['directory'])
    assert len(calls) == 2
