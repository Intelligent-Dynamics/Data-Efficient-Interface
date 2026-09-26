"""Synthetic-only EXP-006R merge fixtures; no live/cache/model or test data access."""
from copy import deepcopy
from decimal import Decimal
import builtins
import json
from pathlib import Path

import pytest

from baseline.general import accounting
from baseline.general_metrics import SEEDS, SHOTS
from baseline.recovery_merge import merge_predictions, offline_evaluation
from baseline.selective import compact, selective_curve


def prediction_sets():
    """770 fabricated IDs/classes/statuses, never BANKING77 predictions."""
    original = [
        {'id': f'train:{index:05}', 'status': 'ok' if index < 718 else 'http_429',
         'predicted_label': f'synthetic_{index // 10:02}' if index < 718 else None,
         'fixture_note': 'SYNTHETIC MOCK ONLY'}
        for index in range(770)
    ]
    eligible = [row['id'] for row in original if row['status'] == 'http_429']
    recovery = [{'id': row['id'], 'status': 'ok', 'predicted_label': f'synthetic_{index // 10:02}'}
                for index, row in enumerate(original) if index >= 718]
    return original, recovery, eligible


def test_merge_keeps_original_success_values_order_and_all_770_ids_exactly_once():
    original, recovery, eligible = prediction_sets()
    snapshots = deepcopy((original, recovery, eligible))
    merged = merge_predictions(original, list(reversed(recovery)), list(reversed(eligible)))
    assert merged[:718] == original[:718]
    assert merged[718:] == recovery
    assert [row['id'] for row in merged] == [row['id'] for row in original]
    assert len(merged) == len({row['id'] for row in merged}) == 770
    assert (original, recovery, eligible) == snapshots
    merged[0]['status'] = 'mutated_output'
    assert original == snapshots[0]


def test_all_recovery_failures_are_preserved_and_original_successes_stay_identical():
    original, recovery, eligible = prediction_sets()
    for row in recovery:
        row.update(status='http_429', predicted_label=None)
    merged = merge_predictions(original, recovery, eligible)
    assert merged[:718] == original[:718]
    assert sum(row['status'] == 'http_429' and row['predicted_label'] is None for row in merged) == 52


@pytest.mark.parametrize('change', [
    'success_replaced', 'unrelated_id', 'missing_recovery', 'duplicate_recovery',
    'missing_eligible', 'duplicate_eligible', 'successful_eligible',
    'missing_original', 'duplicate_original', 'wrong_original_failure',
    'too_many_original_successes', 'ok_without_label', 'failed_with_label',
])
def test_merge_rejects_any_changed_eligibility_or_bad_record(change):
    original, recovery, eligible = prediction_sets()
    if change == 'success_replaced':
        recovery[0]['id'] = original[0]['id']
    elif change == 'unrelated_id':
        recovery[0]['id'] = 'train:99999'
    elif change == 'missing_recovery':
        recovery.pop()
    elif change == 'duplicate_recovery':
        recovery[-1] = deepcopy(recovery[0])
    elif change == 'missing_eligible':
        eligible.pop()
    elif change == 'duplicate_eligible':
        eligible[-1] = eligible[0]
    elif change == 'successful_eligible':
        eligible[0] = original[0]['id']
    elif change == 'missing_original':
        original.pop()
    elif change == 'duplicate_original':
        original[-1] = deepcopy(original[-2])
    elif change == 'wrong_original_failure':
        original[-1]['status'] = 'http_500'
    elif change == 'too_many_original_successes':
        original[-1].update(status='ok', predicted_label='synthetic_76')
    elif change == 'ok_without_label':
        recovery[0]['predicted_label'] = None
    else:
        recovery[0]['status'] = 'refused'
    with pytest.raises(ValueError):
        merge_predictions(original, recovery, eligible)


@pytest.fixture(scope='module')
def evaluation_fixture():
    original, recovery, eligible = prediction_sets()
    # Keep one valid wrong response and one unresolved refusal in the merged set.
    recovery[0]['predicted_label'] = 'synthetic_00'
    recovery[1].update(status='refused', predicted_label=None)
    def entries(rows, original_stage=False):
        result = {}
        for index, row in enumerate(rows):
            def attempt(status, label):
                return {'status': status, 'predicted_label': label,
                        'reservation_usd': '0.01',
                        'usage_priced_usd': '0.001' if status == 'ok' else None}
            attempts = [attempt(row['status'], row['predicted_label'])]
            if original_stage and (row['status'] == 'http_429' or index == 0):
                attempts.insert(0, attempt('http_429', None))
            result[row['id']] = {'id': row['id'], 'kind': 'live', 'attempts': attempts,
                                 'fixture_notice': 'SYNTHETIC MOCK ONLY, NOT LIVE EVIDENCE'}
        return result
    original_entries = entries(original, True)
    recovery_entries = entries(recovery)
    protocol_hash, tree_hash, recovery_hash = 'a' * 64, 'b' * 64, 'c' * 64
    source = {
        'manifest': {'protocol_sha256': protocol_hash},
        'execution': {'study_id': 'EXP-006', 'kind': 'live', 'status': 'finished_with_unresolved',
                      'protocol_sha256': protocol_hash, 'predictions': original,
                      'accounting': accounting(original_entries)},
        'entries': original_entries, 'eligible_ids': eligible,
        'validation_ids': [row['id'] for row in original], 'tree_sha256': tree_hash,
    }
    report = {
        'study_id': 'EXP-006R', 'kind': 'live', 'status': 'finished_with_unresolved',
        'source_protocol_sha256': protocol_hash, 'source_tree_sha256': tree_hash,
        'recovery_protocol_sha256': recovery_hash, 'eligible_ids': list(eligible),
        'predictions': recovery, 'accounting': accounting(recovery_entries),
    }
    labels = [f'synthetic_{index:02}' for index in range(77)]
    rows = [{'id': row['id'], 'true_label': labels[index // 10],
             'predicted_label': labels[index // 10] if index % 2 else labels[(index // 10 + 1) % 77],
             'confidence': 1 - index / 1000} for index, row in enumerate(original)]
    diagnostic = compact(selective_curve(rows, labels, .5))
    runs = [{'run_id': f'SYNTHETIC-n{shots}-s{seed}', 'shots': shots, 'seed': seed, **deepcopy(diagnostic)}
            for shots in SHOTS for seed in SEEDS]
    return source, report, recovery_entries, runs


def test_offline_evaluation_uses_unchanged_metrics_and_separate_attempt_accounting(evaluation_fixture):
    original_inputs = deepcopy(evaluation_fixture)
    result = offline_evaluation(*evaluation_fixture)
    source, report, recovery_entries, runs = evaluation_fixture
    assert evaluation_fixture == original_inputs
    assert result['study_id'] == 'EXP-006 + EXP-006R recovery'
    assert result['status'] == 'finished_with_unresolved'
    assert result['general']['n_examples'] == 770
    assert result['general']['accuracy'] == 768 / 770
    assert result['general']['unresolved_count'] == 1
    assert result['unresolved_ids'] == [report['predictions'][1]['id']]
    assert result['preserved_original_success_count'] == 718
    assert len(result['combinations']) == 90
    for combination in result['combinations']:
        assert combination['combined']['n_examples'] == 770
        costs = combination['hypothetical_routed_api_charges']
        assert costs['specialist_deployment_cost_usd'] is None
        assert costs['total_system_savings_usd'] is None
        if combination['accepted_count'] == 0:
            assert combination['combined'] == result['general']
            assert combination['fallback']['accuracy'] == 768 / 770
        elif combination['accepted_count'] == 770:
            assert combination['combined']['accuracy'] == .5
            assert combination['fallback']['accuracy'] is None
    ledger = result['actual_experiment_accounting']
    assert ledger['original_exp006']['recorded_attempts'] == 823
    assert ledger['recovery_exp006r']['recorded_attempts'] == 52
    assert ledger['combined_collection']['recorded_attempts'] == 875
    assert ledger['combined_collection']['unknown_charge_attempts'] == 106
    assert ledger['combined_collection']['actual_api_spend_usd'] is None
    for field in ('usage_priced_api_charges_usd', 'committed_reservation_usd', 'unknown_charge_reservation_usd'):
        assert Decimal(ledger['combined_collection'][field]) == (
            Decimal(ledger['original_exp006'][field]) + Decimal(ledger['recovery_exp006r'][field]))
    assert list(result['prediction_source_by_id'].values()).count('EXP-006') == 718
    assert list(result['prediction_source_by_id'].values()).count('EXP-006R') == 52
    json.dumps(result, allow_nan=False)


def test_offline_evaluation_cannot_access_any_files_test_split_or_network(evaluation_fixture, monkeypatch):
    import baseline.general as general
    import baseline.general_protocol as protocol
    import urllib.request
    def blocked(*args, **kwargs):
        pytest.fail('Offline merge attempted filesystem/data/network access')
    monkeypatch.setattr(builtins, 'open', blocked)
    monkeypatch.setattr(Path, 'open', blocked)
    monkeypatch.setattr(urllib.request.OpenerDirector, 'open', blocked)
    monkeypatch.setattr(general, 'post_openai', blocked)
    monkeypatch.setattr(general, 'execute', blocked)
    monkeypatch.setattr(protocol, 'load_inputs', blocked)
    result = offline_evaluation(*evaluation_fixture)
    assert result['official_test_access'] is False
    assert result['api_calls_by_offline_evaluation'] == 0
    assert result['source_artifacts_modified'] is False


@pytest.mark.parametrize('change', [
    'mock_report', 'wrong_study', 'unfinished', 'false_completed', 'source_protocol',
    'source_tree', 'invalid_recovery_hash', 'eligibility_order', 'validation_order',
    'missing_entry', 'success_entry', 'cache_status', 'cache_kind', 'empty_attempts',
    'original_accounting', 'recovery_accounting', 'original_protocol',
])
def test_offline_evaluation_rejects_inconsistent_or_partial_evidence(evaluation_fixture, change):
    source, report, entries, runs = deepcopy(evaluation_fixture)
    if change == 'mock_report':
        report['kind'] = 'mock'
    elif change == 'wrong_study':
        report['study_id'] = 'EXP-006'
    elif change == 'unfinished':
        report['status'] = 'running'
    elif change == 'false_completed':
        report['status'] = 'completed'
    elif change == 'source_protocol':
        report['source_protocol_sha256'] = 'd' * 64
    elif change == 'source_tree':
        report['source_tree_sha256'] = 'd' * 64
    elif change == 'invalid_recovery_hash':
        report['recovery_protocol_sha256'] = 'invalid'
    elif change == 'eligibility_order':
        report['eligible_ids'].reverse()
    elif change == 'validation_order':
        source['validation_ids'].reverse()
    elif change == 'missing_entry':
        entries.pop(source['eligible_ids'][0])
    elif change == 'success_entry':
        entries[source['validation_ids'][0]] = deepcopy(source['entries'][source['validation_ids'][0]])
    elif change == 'cache_status':
        entries[source['eligible_ids'][0]]['attempts'][-1]['status'] = 'http_429'
    elif change == 'cache_kind':
        entries[source['eligible_ids'][0]]['kind'] = 'mock'
    elif change == 'empty_attempts':
        entries[source['eligible_ids'][0]]['attempts'].clear()
    elif change == 'original_accounting':
        source['execution']['accounting']['recorded_attempts'] -= 1
    elif change == 'recovery_accounting':
        report['accounting']['recorded_attempts'] -= 1
    else:
        source['execution']['protocol_sha256'] = 'd' * 64
    with pytest.raises(ValueError):
        offline_evaluation(source, report, entries, runs)
