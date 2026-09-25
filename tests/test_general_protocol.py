"""Offline synthetic protocol tests; no real dataset, model, or API is accessed."""
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path

import pytest

from baseline import general_protocol as protocol
from baseline.data import json_bytes, sha256, text_key


@pytest.fixture
def bundle_data():
    labels = ['synthetic_intent_a', 'synthetic_intent_b']
    prompt = 'Synthetic instruction: return exactly one declared intent.'
    schema = {'type': 'object', 'properties': {'intent': {'type': 'string', 'enum': labels}},
              'required': ['intent'], 'additionalProperties': False}
    frozen = {'labels': labels, 'settings': deepcopy(protocol.SETTINGS), 'pricing': deepcopy(protocol.PRICING),
              'retries': deepcopy(protocol.RETRIES), 'endpoint': protocol.ENDPOINT,
              'prompt_sha256': sha256(prompt.encode()), 'schema_sha256': protocol.digest(schema)}
    return frozen, prompt, schema


def test_payload_contains_exact_message_without_truth_confidence_or_specialist_fields(bundle_data):
    frozen, prompt, schema = bundle_data
    row = {'id': 'PRIVATE_ID_MARKER', 'text': '  Café \"quote\"\nsecond line  ',
           'label': 'TRUTH_MARKER', 'true_label': 'TRUTH_MARKER', 'confidence': .876543,
           'specialist_prediction': 'SPECIALIST_MARKER', 'predicted_label': 'SPECIALIST_MARKER'}
    body = protocol.payload(row['text'], prompt, schema)
    assert body['instructions'] == prompt
    assert body['input'] == [{'role': 'user', 'content': [{'type': 'input_text',
                             'text': json.dumps({'customer_message': row['text']}, ensure_ascii=False)}]}]
    assert json.loads(body['input'][0]['content'][0]['text']) == {'customer_message': row['text']}
    assert body['text']['format'] == {'type': 'json_schema', 'name': 'banking77_intent', 'strict': True, 'schema': schema}
    assert set(body) == set(frozen['settings']) | {'instructions', 'input', 'text'}
    serialized = json.dumps(body)
    for marker in ('PRIVATE_ID_MARKER', 'TRUTH_MARKER', 'SPECIALIST_MARKER', '0.876543'):
        assert marker not in serialized


@pytest.mark.parametrize('text', ['', '  \n\t', None, {'customer_message': 'a'}])
def test_payload_rejects_non_text_and_empty_messages(bundle_data, text):
    _, prompt, schema = bundle_data
    with pytest.raises(ValueError):
        protocol.payload(text, prompt, schema)


@pytest.mark.parametrize('changed', ['text', 'model', 'prompt', 'schema', 'settings'])
def test_cache_key_changes_with_every_request_defining_component(bundle_data, changed):
    _, prompt, schema = bundle_data
    original = protocol.payload('synthetic request', prompt, schema)
    body = deepcopy(original)
    if changed == 'text':
        body['input'][0]['content'][0]['text'] = json.dumps({'customer_message': 'different request'})
    elif changed == 'model':
        body['model'] = 'synthetic-different-model'
    elif changed == 'prompt':
        body['instructions'] += ' Changed instruction.'
    elif changed == 'schema':
        body['text']['format']['schema']['properties']['intent']['enum'].append('synthetic_intent_c')
    else:
        body['max_output_tokens'] += 1
    assert protocol.cache_key(body) != protocol.cache_key(original)
    assert protocol.cache_key(dict(reversed(list(original.items())))) == protocol.cache_key(original)


@pytest.fixture
def validation():
    rows = [{'id': 'v:one', 'label': 'a', 'text': 'first synthetic message'},
            {'id': 'v:two', 'label': 'b', 'text': 'second synthetic message'},
            {'id': 'unused', 'label': 'unused', 'text': 'not in validation'}]
    ids = ['v:two', 'v:one']
    truth = {'v:one': 'a', 'v:two': 'b'}
    sealed = [{'id': 'sealed:synthetic', 'key': text_key('unopened synthetic fixture')}]
    return rows, ids, truth, sealed


def test_validation_resolves_exact_existing_order_and_ignores_unselected_rows(validation):
    rows, ids, truth, sealed = validation
    result = protocol.resolve_validation(reversed(rows), ids, truth, sealed)
    assert [row['id'] for row in result] == ids
    assert [row['label'] for row in result] == ['b', 'a']
    assert result == [rows[1], rows[0]]


@pytest.mark.parametrize('change', ['duplicate_ids', 'truth_ids', 'missing_row', 'duplicate_source',
                                    'wrong_label', 'duplicate_text', 'sealed_id', 'sealed_hash'])
def test_validation_rejects_alignment_truth_and_isolation_failures(validation, change):
    rows, ids, truth, sealed = deepcopy(validation)
    if change == 'duplicate_ids':
        ids.append(ids[0])
    elif change == 'truth_ids':
        truth['unexpected'] = 'a'
    elif change == 'missing_row':
        rows.pop(0)
    elif change == 'duplicate_source':
        rows.append(deepcopy(rows[0]))
    elif change == 'wrong_label':
        rows[0]['label'] = 'b'
    elif change == 'duplicate_text':
        rows[1]['text'] = '  FIRST   SYNTHETIC MESSAGE  '
    elif change == 'sealed_id':
        sealed[0]['id'] = rows[0]['id']
    else:
        sealed[0]['key'] = text_key(rows[0]['text'])
    with pytest.raises(ValueError):
        protocol.resolve_validation(rows, ids, truth, sealed)


def test_plan_counts_retry_reservations_and_utf8_token_envelopes(bundle_data, monkeypatch):
    frozen, prompt, schema = bundle_data
    rows = [{'id': 'v:1', 'text': 'synthetic café', 'label': 'SHOULD_NOT_PASS', 'confidence': .999},
            {'id': 'v:2', 'text': 'second synthetic request', 'predicted_label': 'SHOULD_NOT_PASS'}]
    original_payload = protocol.payload
    seen = []
    def inspect_boundary(text, actual_prompt, actual_schema, settings):
        assert isinstance(text, str)
        assert actual_prompt == prompt and actual_schema == schema and settings == frozen['settings']
        seen.append(text)
        return original_payload(text, actual_prompt, actual_schema, settings)
    monkeypatch.setattr(protocol, 'payload', inspect_boundary)
    result = protocol.plan(rows, frozen, prompt, schema, {'synthetic': True, 'additional_validation_labels': 2})
    assert seen == [row['text'] for row in rows]
    assert result['planned_unique_requests'] == 2
    assert result['maximum_attempts_including_retries'] == 2 * frozen['retries']['max_attempts_per_request']
    assert result['inference_calls'] == 0
    assert result['actual_api_spend_usd'] == '0'
    assert result['status'] == 'PAID EXPERIMENT NOT RUN'
    assert [request['id'] for request in result['requests']] == ['v:1', 'v:2']
    rates = frozen['pricing']['per_million_tokens']
    reservations, nominal_costs = [], []
    for request, row in zip(result['requests'], rows):
        body = original_payload(row['text'], prompt, schema, frozen['settings'])
        size = len(json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode())
        assert request['serialized_request_bytes'] == size
        assert request['nominal_input_tokens'] == (size + 3) // 4
        assert request['input_token_reservation'] == 2 * size + 8192
        assert request['output_token_reservation'] == frozen['settings']['max_output_tokens']
        expected = (Decimal(2 * size + 8192) * Decimal(rates['cache_write']) +
                    Decimal(frozen['settings']['max_output_tokens']) * Decimal(rates['output'])) / 1000000
        assert Decimal(request['attempt_reservation_usd']) == expected
        reservations.append(expected)
        nominal_costs.append(Decimal(request['nominal_usd_no_cache_read_discount']))
    costs = result['cost_estimates_usd']
    assert Decimal(costs['nominal_single_attempt']) == sum(nominal_costs)
    assert Decimal(costs['single_attempt_reservation']) == sum(reservations)
    assert Decimal(costs['all_attempts_reservation']) == sum(reservations) * frozen['retries']['max_attempts_per_request']
    assert 'Heuristics' in result['token_count_method']
    assert result['data'] == {'synthetic': True, 'additional_validation_labels': 2}
    assert 'SHOULD_NOT_PASS' not in json.dumps(result)


@pytest.mark.parametrize('change', ['duplicate_id', 'duplicate_input', 'schema_labels'])
def test_plan_rejects_duplicate_inputs_or_changed_labels(bundle_data, change):
    frozen, prompt, schema = bundle_data
    rows = [{'id': 'v:1', 'text': 'synthetic one'}, {'id': 'v:2', 'text': 'synthetic two'}]
    if change == 'duplicate_id':
        rows[1]['id'] = rows[0]['id']
    elif change == 'duplicate_input':
        rows[1]['text'] = rows[0]['text']
    else:
        schema['properties']['intent']['enum'] = ['different_intent']
    with pytest.raises(ValueError):
        protocol.plan(rows, frozen, prompt, schema, {})


def test_estimate_rejects_requests_outside_frozen_pricing_envelope(bundle_data):
    _, prompt, schema = bundle_data
    with pytest.raises(ValueError, match='pricing scope'):
        protocol.estimate(protocol.payload('synthetic ' * 16000, prompt, schema))


@pytest.fixture
def synthetic_bundle(tmp_path, bundle_data):
    frozen, prompt, schema = bundle_data
    (tmp_path / 'protocol.json').write_text(json.dumps(frozen))
    (tmp_path / 'prompt.txt').write_text(prompt)
    (tmp_path / 'schema.json').write_text(json.dumps(schema))
    return tmp_path, bundle_data


def test_loads_frozen_synthetic_prompt_schema_and_settings(synthetic_bundle):
    path, expected = synthetic_bundle
    assert protocol.load_protocol(path) == expected


@pytest.mark.parametrize('change', ['prompt', 'schema', 'settings', 'pricing', 'retries', 'endpoint'])
def test_frozen_bundle_rejects_tampering(synthetic_bundle, change):
    path, _ = synthetic_bundle
    if change == 'prompt':
        with (path / 'prompt.txt').open('a') as stream:
            stream.write(' corrupted')
    elif change == 'schema':
        schema = json.loads((path / 'schema.json').read_text())
        schema['properties']['intent']['enum'].append('corrupted')
        (path / 'schema.json').write_text(json.dumps(schema))
    else:
        frozen = json.loads((path / 'protocol.json').read_text())
        if change == 'settings':
            frozen['settings']['max_output_tokens'] += 1
        elif change == 'pricing':
            frozen['pricing']['per_million_tokens']['input'] = '999'
        elif change == 'retries':
            frozen['retries']['max_attempts_per_request'] += 1
        else:
            frozen['endpoint'] = 'https://invalid.synthetic.example/responses'
        (path / 'protocol.json').write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match='Frozen'):
        protocol.load_protocol(path)


def test_load_inputs_uses_only_mocked_training_and_prior_evidence(monkeypatch, tmp_path):
    """Every read and CSV iterator is replaced; no real dataset or experiment is opened."""
    labels = [f'synthetic_{i:02d}' for i in range(77)]
    ids = [f'train:{i:05d}' for i in range(770)]
    predictions = [{'id': row_id, 'true_label': labels[i // 10], 'predicted_label': labels[i // 10]}
                   for i, row_id in enumerate(ids)]
    raw = tmp_path / 'data/raw/banking77'
    payloads = {raw / 'train.csv': b'synthetic mocked training bytes', raw / 'categories.json': json_bytes(labels)}
    source = {'revision': 'synthetic-source', 'files': {
        name: {'sha256': sha256(payloads[raw / name])} for name in ('train.csv', 'categories.json')}}
    manifest = {'protocol_id': protocol.PROTOCOL_ID, 'source': source, 'labels': labels, 'validation_ids': ids,
                'sealed_test': [{'id': 'sealed:synthetic', 'key': text_key('synthetic stored audit key')}]}
    manifest_path = tmp_path / protocol.DEFAULT_MANIFEST
    objects = {raw / 'categories.json': labels, manifest_path: manifest}
    payloads[manifest_path] = json_bytes(manifest)
    study4 = tmp_path / 'experiments/exp004-minilm-learning-curve'
    study5 = tmp_path / 'experiments/exp005-selective-diagnostic'
    audit4, audit5 = {'compact_records_sha256': {}}, {'analysis_artifact_sha256': {}}
    for n in protocol.SHOTS:
        for seed in protocol.SEEDS:
            name = f'exp004-minilm-v2-n{n}-s{seed}.json'
            r4 = {'samples': {'validation_ids': ids, 'labels': labels,
                              'train_ids': [f'synthetic-fit:{i}' for i in range(n * 77)]},
                  'metadata': {'split_manifest_sha256': sha256(payloads[manifest_path])}, 'predictions': predictions}
            h4 = sha256(json_bytes(r4))
            r5 = {'shots': n, 'seed': seed, 'provenance': {'exp004_record_sha256': h4}, 'ranked_rows': predictions}
            for study, record, audit, key in ((study4, r4, audit4, 'compact_records_sha256'),
                                               (study5, r5, audit5, 'analysis_artifact_sha256')):
                objects[study / 'runs' / name] = record
                payloads[study / 'runs' / name] = json_bytes(record)
                audit[key]['runs/' + name] = sha256(json_bytes(record))
    objects[study4 / 'analysis_audit.json'] = audit4
    objects[study5 / 'metadata.json'] = audit5
    reads = []
    def fake_json(path):
        reads.append(Path(path))
        assert Path(path) in objects, f'Unexpected JSON access: {path}'
        return deepcopy(objects[Path(path)])
    def fake_bytes(path):
        reads.append(path)
        assert path in payloads, f'Unexpected byte access: {path}'
        return payloads[path]
    def fake_csv(path):
        assert path == raw / 'train.csv'
        for i, row in enumerate(predictions):
            yield i, {'text': f'synthetic validation message {i}', 'category': row['true_label']}
    monkeypatch.setattr(protocol, 'ROOT', tmp_path)
    monkeypatch.setattr(protocol, 'source_spec', lambda: deepcopy(source))
    monkeypatch.setattr(protocol, 'read_json', fake_json)
    monkeypatch.setattr(protocol, 'csv_rows', fake_csv)
    monkeypatch.setattr(protocol, 'restore', lambda record: record)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'read_bytes', fake_bytes)
        rows, result_labels, runs, provenance = protocol.load_inputs()
    assert [row['id'] for row in rows] == ids
    assert result_labels == labels and len(runs) == 15
    assert provenance['official_test_access'] is False
    assert provenance['additional_validation_labels'] == 770 and provenance['new_labels'] == 0
    assert all(path.name != 'test.csv' for path in reads)
    assert all(str(path).startswith(str(tmp_path)) for path in reads)
