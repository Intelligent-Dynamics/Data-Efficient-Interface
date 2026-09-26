"""Synthetic EXP-007 scoring/preparation; sealed test, models and APIs stay untouched."""
from copy import deepcopy
from decimal import Decimal
import json
import os
from pathlib import Path
import socket
import urllib.request

import pytest

from baseline import fixed_routing, general, general_protocol, recovery
from baseline.data import SEEDS, sha256
from baseline.threshold_inputs import MODEL_ID, MODEL_REVISION


def forbidden(*args, **kwargs):
    pytest.fail('Fixed-routing test reached an API, credentials, model, or sealed/raw dataset')


@pytest.fixture(autouse=True)
def no_external_execution(monkeypatch):
    monkeypatch.setattr(fixed_routing, 'load_inputs', forbidden)
    monkeypatch.setattr(fixed_routing, 'validation_usage', forbidden)
    monkeypatch.setattr(general_protocol, 'load_inputs', forbidden)
    monkeypatch.setattr(general, 'post_openai', forbidden)
    monkeypatch.setattr(recovery, 'post_recovery', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(urllib.request, 'urlopen', forbidden)
    monkeypatch.setattr(urllib.request.OpenerDirector, 'open', forbidden)
    from sklearn.linear_model import LogisticRegression
    import joblib
    monkeypatch.setattr(LogisticRegression, 'fit', forbidden)
    monkeypatch.setattr(joblib, 'load', forbidden)
    original_getitem = type(os.environ).__getitem__
    def guarded_environment(mapping, key):
        if key in ('OPENAI_API_KEY', 'OPENAI_PROJECT_ID'):
            forbidden()
        return original_getitem(mapping, key)
    monkeypatch.setattr(type(os.environ), '__getitem__', guarded_environment)
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        if '/data/raw/' in str(path) or '/data/processed/' in str(path) or path.name == 'test.csv':
            forbidden()
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)


@pytest.fixture
def score_rows():
    specialist = [
        {'id': 'synthetic:0', 'true_label': 'a', 'predicted_label': 'a', 'confidence': .9},
        {'id': 'synthetic:1', 'true_label': 'b', 'predicted_label': 'a', 'confidence': .8},
        {'id': 'synthetic:2', 'true_label': 'a', 'predicted_label': 'b', 'confidence': .7},
        {'id': 'synthetic:3', 'true_label': 'b', 'predicted_label': 'b', 'confidence': .6},
    ]
    luna = [
        {'id': 'synthetic:0', 'predicted_label': 'b', 'status': 'ok'},
        {'id': 'synthetic:1', 'predicted_label': 'b', 'status': 'ok'},
        {'id': 'synthetic:2', 'predicted_label': None, 'status': 'http_429'},
        {'id': 'synthetic:3', 'predicted_label': 'a', 'status': 'ok'},
    ]
    return specialist, luna, ['a', 'b']


def test_scalar_gate_id_alignment_and_full_denominator_metrics(score_rows):
    before = deepcopy(score_rows)
    result = fixed_routing.evaluate_fixed(*score_rows, .8)
    assert result == fixed_routing.evaluate_fixed(score_rows[0], list(reversed(score_rows[1])), score_rows[2], .8)
    assert score_rows == before
    assert result['n_examples'] == 4 and result['accepted_count'] == 2
    assert result['observed_coverage'] == .5  # The scalar gate never forces the validation's 90% prefix.
    assert result['luna_request_count'] == 2 and result['luna_request_percentage'] == 50
    assert result['accepted_ids'] == ['synthetic:0', 'synthetic:1']  # Equality is accepted.
    assert result['fallback_ids'] == ['synthetic:2', 'synthetic:3']
    assert result['fallback_accuracy_on_rejected'] == 0
    assert [r['predicted_label'] for r in result['predictions']] == ['a', 'a', None, 'a']
    # Hand calculation: specialist class F1=(1/2,1/2), Luna=(0,1/2), routed=(2/5,0).
    for name, accuracy, f1, unresolved in [('specialist_only', .5, .5, 0),
                                          ('luna_only', .25, .25, 1), ('routed', .25, .2, 1)]:
        metrics = result[name]
        assert metrics['n_examples'] == 4 and metrics['accuracy'] == accuracy
        assert metrics['macro_f1'] == pytest.approx(f1)
        assert metrics['unresolved_count'] == unresolved
        assert metrics['error_count'] == 4 - metrics['correct_count']
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('threshold,accepted', [(0.0, 4), (1.0, 0)])
def test_all_specialist_or_all_luna_endpoint_and_undefined_empty_fallback(score_rows, threshold, accepted):
    result = fixed_routing.evaluate_fixed(*score_rows, threshold)
    assert result['accepted_count'] == accepted
    assert result['routed'] == result['specialist_only' if accepted else 'luna_only']
    assert result['fallback_accuracy_on_rejected'] == (None if accepted else .25)


@pytest.mark.parametrize('corruption', [
    'duplicate_specialist', 'duplicate_luna', 'missing_luna', 'extra_luna', 'missing_id', 'empty_id',
    'missing_status', 'nonstring_status', 'ok_without_label', 'failure_with_label', 'unknown_luna_label',
    'unknown_specialist_label', 'unknown_truth', 'conflicting_truth', 'duplicate_classes', 'missing_class',
])
def test_score_rejects_id_status_truth_and_class_drift(score_rows, corruption):
    specialist, luna, labels = score_rows
    if corruption == 'duplicate_specialist': specialist.append(deepcopy(specialist[0]))
    elif corruption == 'duplicate_luna': luna.append(deepcopy(luna[0]))
    elif corruption == 'missing_luna': luna.pop()
    elif corruption == 'extra_luna': luna.append({'id': 'extra', 'status': 'ok', 'predicted_label': 'a'})
    elif corruption == 'missing_id': specialist[0].pop('id')
    elif corruption == 'empty_id': luna[0]['id'] = ''
    elif corruption == 'missing_status': luna[0].pop('status')
    elif corruption == 'nonstring_status': luna[0]['status'] = 429
    elif corruption == 'ok_without_label': luna[0]['predicted_label'] = None
    elif corruption == 'failure_with_label': luna[0]['status'] = 'refusal'
    elif corruption == 'unknown_luna_label': luna[0]['predicted_label'] = 'unknown'
    elif corruption == 'unknown_specialist_label': specialist[0]['predicted_label'] = 'unknown'
    elif corruption == 'unknown_truth': specialist[0]['true_label'] = 'unknown'
    elif corruption == 'conflicting_truth': luna[0]['true_label'] = 'b'
    elif corruption == 'duplicate_classes': labels.append('a')
    else: labels.remove('b')
    with pytest.raises(ValueError):
        fixed_routing.evaluate_fixed(specialist, luna, labels, .8)


@pytest.fixture
def synthetic_usage():
    # Deliberately synthetic round token totals, not measured project results.
    return {'validation_count': 770, 'successful_response_usage': {
                'input_tokens': 770000, 'output_tokens': 15400,
                'cached_tokens': 0, 'cache_write_tokens': 0, 'reasoning_tokens': 0},
            'successful_usage_priced_usd': '0.0847', 'largest_validation_payload_bytes': 5000,
            'source_files_sha256': {}, 'new_api_calls': 0, 'official_test_access': False}


def test_cost_projection_uses_one_shared_response_set_and_explicit_conditional_reservations(synthetic_usage):
    before = deepcopy(synthetic_usage)
    costs = fixed_routing.estimate_test_cost(synthetic_usage, 3080)
    assert synthetic_usage == before
    assert Decimal(costs['nominal_usage_projection_usd']) == Decimal('.3388')
    assert Decimal(costs['same_tokens_all_input_cache_write_usd']) == Decimal('.4158')
    assert Decimal(costs['conservative_one_attempt_usd']) == Decimal('7.20104')
    assert Decimal(costs['conservative_all_attempts_usd']) == Decimal('28.80416')
    assert costs['planned_test_rows'] == costs['planned_first_attempts'] == 3080
    assert costs['response_sets_across_all_five_specialists'] == 1
    assert costs['max_attempts_per_request'] == 4 and costs['maximum_attempts'] == 12320
    assert costs['input_token_envelope_per_attempt'] == 18192
    assert costs['output_token_envelope_per_attempt'] == 128
    assert costs['approved_spending_cap_usd'] is None and costs['live_authorization'] is False
    assert costs['api_calls_during_preparation'] == 0 and costs['actual_preparation_spend_usd'] == '0'
    assumptions = ' '.join(costs['assumptions'])
    assert 'NOT been read' in assumptions and 'not a guaranteed invoice' in assumptions
    assert 'no cache-read discount' in assumptions and 'failures/retries' in assumptions
    json.dumps(costs, allow_nan=False)


@pytest.mark.parametrize('changed', ['validation_count', 'test_count'])
def test_cost_estimate_rejects_changed_population(synthetic_usage, changed):
    if changed == 'validation_count': synthetic_usage['validation_count'] = 769
    with pytest.raises(ValueError, match='population'):
        fixed_routing.estimate_test_cost(synthetic_usage, 3079 if changed == 'test_count' else 3080)


@pytest.fixture
def preparation_case(tmp_path, monkeypatch, synthetic_usage):
    root = tmp_path / 'synthetic-repository'
    root.mkdir()
    labels = [f'class-{i:02}' for i in range(77)]
    inputs = {'labels': labels, 'runs': [], 'source_files_sha256': {},
              'dataset': {'dataset': 'BANKING77', 'revision': 'SYNTHETIC-METADATA',
                          'expected_counts': {'train': 10003, 'test': 3080, 'classes': 77},
                          'files': {'test.csv': {'sha256': 'a' * 64}}},
              'embedding': {'model_id': MODEL_ID, 'revision': MODEL_REVISION,
                            'encoding': {'device': 'cpu', 'precision': 'float32', 'normalize_embeddings': True}}}
    for seed in SEEDS:
        # All confidences are distinct: rank 693 is exactly the scalar boundary.
        rows = [{'id': f'synthetic:{i:04}', 'confidence': 1.0 - i / 1000,
                 'true_label': labels[i % 77], 'predicted_label': labels[(i + 1) % 77]} for i in range(770)]
        inputs['runs'].append({'seed': seed, 'run_id': f'SYNTHETIC-n20-s{seed}',
            'classifier_path': f'synthetic/classifier-{seed}.bin', 'classifier_sha256': 'b' * 64,
            'probabilities_sha256': 'c' * 64,
            'selective': {'ranked_rows': rows, 'landmarks': [{'target_coverage': .9, 'accepted_count': 693}]}})
    for target, name in [(inputs, 'synthetic/specialist-evidence.json'),
                         (synthetic_usage, 'synthetic/usage-evidence.json')]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'SYNTHETIC EVIDENCE; NO MODEL OR DATASET CONTENT')
        target['source_files_sha256'] = {name: sha256(path.read_bytes())}
    code_paths = ['baseline/thresholds.py', 'baseline/threshold_inputs.py', 'baseline/fixed_routing.py',
                  'baseline/general_metrics.py', 'pyproject.toml', 'uv.lock']
    for name in code_paths:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'SYNTHETIC PROVENANCE FILE')
    old = {'labels': labels.copy(), 'endpoint': general_protocol.ENDPOINT, 'settings': deepcopy(general_protocol.SETTINGS),
           'prompt_sha256': 'd' * 64, 'schema_sha256': 'e' * 64}
    monkeypatch.setattr(fixed_routing, 'ROOT', root)
    monkeypatch.setattr(fixed_routing, 'load_inputs', lambda: deepcopy(inputs))
    monkeypatch.setattr(fixed_routing, 'validation_usage', lambda: deepcopy(synthetic_usage))
    monkeypatch.setattr(fixed_routing, 'load_protocol', lambda: (deepcopy(old), 'SYNTHETIC PROMPT', {}))
    def git_head(command, **kwargs):
        assert command == ['git', '-C', str(root), 'rev-parse', 'HEAD']
        assert kwargs == {'text': True}
        return 'f' * 40 + '\n'
    monkeypatch.setattr(fixed_routing.subprocess, 'check_output', git_head)
    return {'root': root, 'output': tmp_path / 'prepared', 'inputs': inputs, 'usage': synthetic_usage, 'old': old}


def test_frozen_protocol_preserves_models_settings_five_thresholds_and_new_approval(preparation_case):
    case = preparation_case
    thresholds = fixed_routing.summarize_thresholds(case['inputs'])
    costs = fixed_routing.estimate_test_cost(case['usage'], 3080)
    protocol = fixed_routing.make_test_protocol(case['inputs'], thresholds, costs)
    assert protocol['specialists']['seeds'] == list(SEEDS) and protocol['specialists']['shots'] == 20
    assert protocol['specialists']['embedding'] == case['inputs']['embedding']
    assert all(protocol['specialists'][key] is False for key in ('refit', 'recalibrate', 'tune'))
    assert set(protocol['thresholds']) == {str(seed) for seed in SEEDS}
    for point in thresholds:
        frozen = protocol['thresholds'][str(point['seed'])]
        assert float.fromhex(frozen['binary64_hex']) == float(frozen['decimal']) == frozen['value'] == point['threshold']
        assert point['accepted_count'] == 693 and point['actual_coverage'] == .9
    assert protocol['fallback']['settings'] == case['old']['settings']
    assert protocol['fallback']['source_protocol_sha256'] == general_protocol.digest(case['old'])
    assert protocol['fallback']['planned_all_case_requests'] == 3080 and protocol['fallback']['response_sets'] == 1
    assert protocol['fallback']['retry_policy']['max_attempts_per_request'] == 4
    assert 'New explicit live authorization' in protocol['fallback']['auth']
    assert 'previous approvals do not transfer' in protocol['fallback']['auth']
    assert protocol['population']['expected_count_from_existing_metadata'] == 3080
    assert protocol['population']['sealed_file_sha256_from_existing_metadata'] == 'a' * 64
    assert protocol['population']['include_all_rows'] is True and protocol['population']['preparation_read_test_file'] is False
    assert protocol['threshold_derivation']['validation_labels_already_used'] == 770
    assert protocol['threshold_derivation']['new_labels'] == protocol['threshold_derivation']['calibration_fits'] == 0
    assert 'Never rank test scores' in ' '.join(protocol['one_time_sequence'])
    assert protocol['accounting']['cost_estimate_sha256'] == general_protocol.digest(costs)
    assert protocol['accounting']['specialist_deployment_cost'] is protocol['accounting']['total_system_savings'] is None


def test_prepare_cli_creates_only_fresh_offline_artifacts_and_preserves_inputs(preparation_case, monkeypatch, capsys):
    case = preparation_case
    before = {p: p.read_bytes() for p in case['root'].rglob('*') if p.is_file()}
    monkeypatch.setattr('sys.argv', ['baseline.fixed_routing', '--output', str(case['output'])])
    fixed_routing.main()
    printed = json.loads(capsys.readouterr().out)
    expected = {'thresholds.json', 'cost_estimate.json', 'validation_completion.json',
                'protocol.json', 'manifest.json', 'protocol_sha256.txt'}
    assert {p.name for p in case['output'].iterdir()} == expected
    manifest = json.loads((case['output'] / 'manifest.json').read_text())
    assert printed == manifest and manifest['official_test_access'] is False
    assert all(manifest[key] == 0 for key in ('model_fits', 'model_executions', 'api_calls'))
    assert manifest['observed_validation_coverage_by_seed'] == {str(seed): .9 for seed in SEEDS}
    protocol_bytes = (case['output'] / 'protocol.json').read_bytes()
    assert sha256(protocol_bytes) == manifest['protocol_sha256']
    assert (case['output'] / 'protocol_sha256.txt').read_text() == manifest['protocol_sha256'] + '\n'
    assert {p: p.read_bytes() for p in before} == before
    with pytest.raises(ValueError, match='fresh output'):
        fixed_routing.prepare(case['output'])


def test_prepare_rejects_changed_evidence_before_output_write(preparation_case):
    case = preparation_case
    evidence = next(iter(case['usage']['source_files_sha256']))
    (case['root'] / evidence).write_bytes(b'CHANGED SYNTHETIC EVIDENCE')
    with pytest.raises(ValueError, match='Input changed'):
        fixed_routing.prepare(case['output'])
    assert not case['output'].exists()


def test_preparation_cli_exposes_no_live_mode(preparation_case, monkeypatch):
    case = preparation_case
    monkeypatch.setattr('sys.argv', ['baseline.fixed_routing', '--output', str(case['output']), '--authorize-live'])
    with pytest.raises(SystemExit) as result:
        fixed_routing.main()
    assert result.value.code == 2 and not case['output'].exists()


@pytest.mark.parametrize('changed', ['classes', 'seed_order'])
def test_protocol_rejects_specialist_luna_class_or_seed_drift(preparation_case, changed):
    case = preparation_case
    thresholds = fixed_routing.summarize_thresholds(case['inputs'])
    costs = fixed_routing.estimate_test_cost(case['usage'], 3080)
    if changed == 'classes':
        case['inputs']['labels'][0] = 'different-class'
    else:
        thresholds.reverse()
    with pytest.raises(ValueError):
        fixed_routing.make_test_protocol(case['inputs'], thresholds, costs)


@pytest.fixture
def completed_merge():
    ids = [f'SYNTHETIC-VALIDATION:{i:04d}' for i in range(770)]
    predictions = [{'id': rid, 'status': 'ok', 'predicted_label': 'a'} for rid in ids]
    source = {'validation_ids': ids, 'source_protocol_sha256': 'a' * 64, 'tree_sha256': 'b' * 64}
    recovery_protocol = {'study_id': 'SYNTHETIC-RECOVERY', 'fixed': True}
    merged = {'study_id': 'EXP-006 + EXP-006R recovery', 'status': 'completed',
              'validation_ids': list(reversed(ids)), 'predictions': deepcopy(predictions),
              'source_protocol_sha256': source['source_protocol_sha256'],
              'recovery_protocol_sha256': general_protocol.digest(recovery_protocol),
              'source_tree_sha256': source['tree_sha256']}
    return merged, predictions, source, recovery_protocol


def test_completed_merge_accepts_reordered_metadata_ids_without_mutating_evidence(completed_merge):
    before = deepcopy(completed_merge)
    merged, predictions, source, protocol = completed_merge
    assert merged['validation_ids'] != source['validation_ids']
    fixed_routing.validate_completed_merge(merged, predictions, source, protocol)
    assert completed_merge == before


@pytest.mark.parametrize('corruption', [
    'duplicate_id', 'missing_id', 'foreign_id', 'prediction_order', 'prediction_label',
    'source_protocol_sha256', 'recovery_protocol_sha256', 'source_tree_sha256',
])
def test_completed_merge_rejects_population_prediction_or_provenance_changes(completed_merge, corruption):
    merged, predictions, source, protocol = completed_merge
    if corruption == 'duplicate_id':
        merged['validation_ids'][-1] = merged['validation_ids'][0]
    elif corruption == 'missing_id':
        merged['validation_ids'].pop()
    elif corruption == 'foreign_id':
        merged['validation_ids'][-1] = 'SYNTHETIC-FOREIGN-ID'
    elif corruption == 'prediction_order':
        merged['predictions'].reverse()
    elif corruption == 'prediction_label':
        merged['predictions'][0]['predicted_label'] = 'b'
    else:
        merged[corruption] = 'c' * 64
    with pytest.raises(ValueError, match='Merged validation'):
        fixed_routing.validate_completed_merge(merged, predictions, source, protocol)
