"""Synthetic end-to-end EXP-007 orchestration; no real test rows/models/API calls."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import socket

import pytest

from baseline import final_test as runner
from baseline.data import read_json
from baseline.final_protocol import Authorization, FROZEN_PROTOCOL_SHA256, load_frozen
from baseline.general import atomic_json
from baseline.general_protocol import digest


@pytest.fixture(autouse=True)
def forbid_external(monkeypatch):
    import urllib.request
    from baseline import recovery
    def forbidden(*args, **kwargs):
        pytest.fail('Synthetic final-runner tests attempted external access')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(urllib.request.OpenerDirector, 'open', forbidden)
    monkeypatch.setattr(recovery, 'post_recovery', forbidden)
    original = Path.open
    def guarded(path, *args, **kwargs):
        from baseline.data import ROOT
        if path.is_relative_to(ROOT / 'data/raw') or path.is_relative_to(ROOT / 'data/processed'):
            forbidden()
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)


@pytest.fixture
def synthetic():
    protocol, prompt, schema = load_frozen()
    labels = protocol['population']['labels']
    ids = [f'test:{i:05d}' for i in range(3080)]
    truth = {rid: labels[i % 77] for i, rid in enumerate(ids)}
    rows = [{'id': rid, 'text': f'SYNTHETIC ONLY example {i}'} for i, rid in enumerate(ids)]
    specialists = {str(seed): [{'id': rid, 'predicted_label': labels[i % 77],
                               'confidence': .2 if i % 2 else .05, 'use_specialist': bool(i % 2)}
                              for i, rid in enumerate(ids)] for seed in protocol['specialists']['seeds']}
    luna = [{'id': rid, 'status': 'ok', 'predicted_label': labels[i % 77]} for i, rid in enumerate(ids)]
    auth = Authorization(FROZEN_PROTOCOL_SHA256, True, True, '100', datetime.now(timezone.utc).date().isoformat())
    return protocol, prompt, schema, rows, specialists, luna, truth, auth


def test_dry_run_remains_sealed_and_uses_only_frozen_estimate(monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail('Dry-run reached test/model/collection')
    for name in ('unseal_inputs', 'verify_specialist_files', 'collect', 'run_specialists', 'scoring_truth'):
        monkeypatch.setattr(runner, name, forbidden)
    result = runner.dry_run()
    assert result['official_test_opened'] is False
    assert result['api_calls'] == result['model_executions'] == 0
    assert result['conditional_conservative_maximum_usd'] == '28.24052000'
    assert result['planned_test_rows'] == 3080


@pytest.mark.parametrize('mode', ['preflight', 'live'])
def test_cli_cannot_unseal_without_all_authorizations(mode, monkeypatch):
    monkeypatch.setattr(runner, 'verify_preparation', lambda *a: pytest.fail('Unauthorized preflight'))
    with pytest.raises(SystemExit) as exc:
        runner.main([mode])
    assert exc.value.code == 2


def test_model_verification_precedes_unseal_and_low_cap_prevents_inference(synthetic, monkeypatch):
    protocol, prompt, schema, rows, *_rest, auth = synthetic
    events = []
    monkeypatch.setattr(runner, 'verify_preparation', lambda *a: events.append('code'))
    monkeypatch.setattr(runner, 'verify_specialist_files', lambda *a, **kw: events.append('models'))
    def unseal(*a):
        assert events == ['code', 'models']
        events.append('unseal')
        return rows
    monkeypatch.setattr(runner, 'unseal_inputs', unseal)
    cheap = Authorization(FROZEN_PROTOCOL_SHA256, True, True, '.001', auth.pricing_date)
    with pytest.raises(ValueError, match='reservation exceeds'):
        runner.prepare_authorized(cheap)
    assert events == ['code', 'models', 'unseal']


def test_full_population_scores_and_observed_coverage_not_forced(synthetic):
    protocol, _, _, _, specialists, luna, truth, _ = synthetic
    luna[0].update(status='refusal', predicted_label=None)
    result = runner.evaluate_predictions(specialists, luna, truth, protocol)
    assert result['n_examples'] == 3080 and result['unresolved_count'] == 1
    assert result['luna_only']['accuracy'] == 3079 / 3080
    for row in result['seeds'].values():
        assert row['observed_coverage'] == .5
        assert row['accepted_count'] == row['luna_request_count'] == 1540
        assert row['luna_request_percentage'] == 50
        assert row['fallback_accuracy_on_rejected'] == 1539 / 1540
        assert row['routed']['unresolved_count'] == 1
        assert len(row['per_class_acceptance']) == 77
        assert len({r['id'] for r in row['predictions']}) == 3080
    assert result['seed_summary']['routed']['accuracy']['sample_sd'] == 0


def test_truth_cannot_change_routing(synthetic):
    protocol, _, _, _, specialists, luna, truth, _ = synthetic
    before = deepcopy(specialists)
    first = runner.evaluate_predictions(specialists, luna, truth, protocol)
    rotated = {rid: protocol['population']['labels'][(i+1) % 77] for i, rid in enumerate(truth)}
    second = runner.evaluate_predictions(specialists, luna, rotated, protocol)
    assert specialists == before
    for seed in specialists:
        assert first['seeds'][seed]['accepted_ids'] == second['seeds'][seed]['accepted_ids']
        assert first['seeds'][seed]['fallback_ids'] == second['seeds'][seed]['fallback_ids']
    assert first['luna_only']['accuracy'] != second['luna_only']['accuracy']


@pytest.mark.parametrize('fault', ['luna_duplicate', 'specialist_missing', 'truth_missing', 'gate', 'label_field', 'threshold'])
def test_scoring_rejects_population_or_gate_drift(synthetic, fault):
    protocol, _, _, _, specialists, luna, truth, _ = synthetic
    if fault == 'luna_duplicate': luna[-1] = luna[0]
    if fault == 'specialist_missing': specialists['11'].pop()
    if fault == 'truth_missing': truth.pop(next(iter(truth)))
    if fault == 'gate': specialists['11'][0]['use_specialist'] = True
    if fault == 'label_field': specialists['11'][0]['true_label'] = 'hidden'
    if fault == 'threshold': protocol['thresholds']['11']['value'] += .001
    with pytest.raises(ValueError):
        runner.evaluate_predictions(specialists, luna, truth, protocol)


def install_fake_execution(monkeypatch, tmp_path, synthetic, status='completed'):
    protocol, prompt, schema, rows, specialists, luna, truth, auth = synthetic
    calls = {'specialists': 0, 'collection': 0, 'truth': 0}
    plan = {'full_retry_reservation_usd': '28', 'synthetic': True}
    monkeypatch.setattr(runner, 'prepare_authorized', lambda *a: (protocol, prompt, schema, rows, plan))
    monkeypatch.setattr(runner, 'runtime_code', lambda *a: {'SYNTHETIC': 'fixed'})
    def fake_specialists(rows_arg, protocol_arg, output, **kw):
        assert all(set(r) == {'id', 'text'} for r in rows_arg)
        calls['specialists'] += 1
        for seed, predictions in specialists.items():
            for name, value in [('predictions', predictions), ('probabilities', {'synthetic': True}), ('metadata', {'synthetic': True})]:
                atomic_json(output / f'specialists/{seed}/{name}.json', value)
        return specialists
    def fake_collect(rows_arg, protocol_arg, prompt_arg, schema_arg, output, authorization, **kw):
        assert all(set(r) == {'id', 'text'} for r in rows_arg)
        calls['collection'] += 1
        for seed in specialists:
            assert (output.parent / f'specialists/{seed}/predictions.json').exists()
        report = {'status': status, 'halt_reason': None if status == 'completed' else 'synthetic pause',
                  'predictions': luna, 'accounting': {'actual_api_spend_usd': '0', 'synthetic': True}}
        atomic_json(output / 'execution.json', report)
        return report, {}
    def fake_truth(authorization, protocol_arg, root, frozen_predictions_verified):
        calls['truth'] += 1
        assert frozen_predictions_verified is True
        output = tmp_path / protocol['outputs']['root']
        assert (output / 'predictions_frozen.json').exists()
        runner.verify_frozen_predictions(output, protocol)
        return truth
    monkeypatch.setattr(runner, 'run_specialists', fake_specialists)
    monkeypatch.setattr(runner, 'collect', fake_collect)
    monkeypatch.setattr(runner, 'verify_completed', lambda *a: None)
    monkeypatch.setattr(runner, 'scoring_truth', fake_truth)
    return calls


def test_end_to_end_predictions_frozen_before_truth_and_idempotent_resume(synthetic, monkeypatch, tmp_path):
    calls = install_fake_execution(monkeypatch, tmp_path, synthetic)
    first = runner.execute(synthetic[-1], tmp_path)
    second = runner.execute(synthetic[-1], tmp_path)
    assert first == second
    assert calls == {'specialists': 1, 'collection': 1, 'truth': 1}
    assert first['api_accounting']['actual_api_spend_usd'] == '0'
    assert first['prediction_freeze_sha256']


def test_halted_collection_does_not_release_truth(synthetic, monkeypatch, tmp_path):
    calls = install_fake_execution(monkeypatch, tmp_path, synthetic, 'halted')
    result = runner.execute(synthetic[-1], tmp_path)
    assert result['labels_joined'] is False and calls['truth'] == 0
    assert not (tmp_path / synthetic[0]['outputs']['root'] / 'evaluation.json').exists()


def test_resume_after_freeze_before_score_does_not_infer_again(synthetic, monkeypatch, tmp_path):
    calls = install_fake_execution(monkeypatch, tmp_path, synthetic)
    original = runner.scoring_truth
    monkeypatch.setattr(runner, 'scoring_truth', lambda *a, **kw: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        runner.execute(synthetic[-1], tmp_path)
    monkeypatch.setattr(runner, 'scoring_truth', original)
    result = runner.execute(synthetic[-1], tmp_path)
    assert result['n_examples'] == 3080
    assert calls == {'specialists': 1, 'collection': 1, 'truth': 1}


def test_saved_predictions_cannot_be_tampered_on_resume(synthetic, monkeypatch, tmp_path):
    calls = install_fake_execution(monkeypatch, tmp_path, synthetic)
    runner.execute(synthetic[-1], tmp_path)
    path = tmp_path / synthetic[0]['outputs']['root'] / 'specialists/11/predictions.json'
    rows = read_json(path)
    rows[0]['use_specialist'] = True
    atomic_json(path, rows)
    with pytest.raises(ValueError, match='Immutable'):
        runner.execute(synthetic[-1], tmp_path)
    assert calls == {'specialists': 1, 'collection': 1, 'truth': 1}


@pytest.mark.parametrize('cutoff', ['evaluation.json', 'verification.json'])
def test_atomic_scoring_checkpoint_survives_materialization_crash(synthetic, monkeypatch, tmp_path, cutoff):
    calls = install_fake_execution(monkeypatch, tmp_path, synthetic)
    original = runner._persist_exact
    def interrupted(path, value):
        if path.name == cutoff:
            raise KeyboardInterrupt()
        return original(path, value)
    monkeypatch.setattr(runner, '_persist_exact', interrupted)
    with pytest.raises(KeyboardInterrupt):
        runner.execute(synthetic[-1], tmp_path)
    monkeypatch.setattr(runner, '_persist_exact', original)
    result = runner.execute(synthetic[-1], tmp_path)
    assert result['n_examples'] == 3080
    assert calls == {'specialists': 1, 'collection': 1, 'truth': 1}
