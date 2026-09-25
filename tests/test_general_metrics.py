"""Synthetic scoring checks: failed responses are errors, never discarded cases."""

from copy import deepcopy
import json
import statistics

import pytest

from baseline.general_metrics import SEEDS, SHOTS, evaluate, general_metrics
from baseline.selective import compact, selective_curve


def test_unresolved_predictions_remain_in_denominator_and_macro_excludes_failure_class():
    score = general_metrics(['a', 'a', 'b', 'b', 'c'], ['a', None, 'a', 'b', None], ['a', 'b', 'c'])
    assert score['n_examples'] == 5
    assert score['correct_count'] == 2
    assert score['error_count'] == 3
    assert score['unresolved_count'] == 2
    assert score['accuracy'] == .4
    assert score['macro_f1'] == pytest.approx((.5 + 2 / 3 + 0) / 3)
    assert set(score['per_class']) == {'a', 'b', 'c'}
    assert score['per_class']['a'] == {'precision': .5, 'recall': .5, 'f1': .5, 'support': 2,
                                     'true_positive': 1, 'false_positive': 1, 'false_negative': 1}
    assert score['per_class']['c']['false_negative'] == 1
    json.dumps(score, allow_nan=False)


def test_all_unresolved_has_zero_scores_and_preserves_every_declared_class():
    score = general_metrics(['a', 'b'], [None, None], ['a', 'b', 'absent'])
    assert score['accuracy'] == score['macro_f1'] == 0
    assert score['unresolved_count'] == score['error_count'] == 2
    assert score['per_class']['absent']['support'] == 0
    assert score['per_class']['absent']['f1'] == 0
    # Even an absent declared class remains in the macro denominator.
    score = general_metrics(['a', 'b'], ['a', 'b'], ['a', 'b', 'absent'])
    assert score['accuracy'] == 1
    assert score['macro_f1'] == pytest.approx(2 / 3)


@pytest.mark.parametrize('truth,predictions,labels', [
    ([], [], ['a']),
    (['a'], [], ['a']),
    (['a'], ['a'], []),
    (['a'], ['a'], ['a', 'a']),
    (['unknown'], ['a'], ['a']),
    (['a'], ['unknown'], ['a']),
    (['a'], [1], ['a']),
    (['a'], ['a'], ['']),
])
def test_invalid_metric_inputs_are_rejected(truth, predictions, labels):
    with pytest.raises(ValueError):
        general_metrics(truth, predictions, labels)


@pytest.fixture
def inputs():
    rows = [
        {'id': 'v:0', 'true_label': 'a', 'predicted_label': 'a', 'confidence': .9},
        {'id': 'v:1', 'true_label': 'b', 'predicted_label': 'a', 'confidence': .8},
        {'id': 'v:2', 'true_label': 'a', 'predicted_label': 'b', 'confidence': .7},
        {'id': 'v:3', 'true_label': 'b', 'predicted_label': 'b', 'confidence': .6},
    ]
    runs = [
        {'run_id': f'n{shots}-s{seed}', 'shots': shots, 'seed': seed,
         **compact(selective_curve(rows, ['a', 'b'], .5))}
        for shots in SHOTS for seed in SEEDS
    ]
    general = [
        {'id': 'v:0', 'predicted_label': 'b', 'status': 'ok'},
        {'id': 'v:1', 'predicted_label': 'b', 'status': 'ok'},
        {'id': 'v:2', 'predicted_label': None, 'status': 'timeout'},
        {'id': 'v:3', 'predicted_label': 'a', 'status': 'ok'},
    ]
    return general, runs


def test_real_fallback_predictions_are_used_without_assuming_they_are_correct(inputs):
    result = evaluate(*inputs)
    assert result['general']['accuracy'] == .25
    assert result['general']['unresolved_count'] == 1
    assert result['general_response_count'] == result['unique_validation_examples'] == 4
    assert result['general_response_sets'] == 1
    assert len(result['combinations']) == 15 * 6
    points = result['combinations'][:6]
    assert [point['accepted_count'] for point in points] == [0, 1, 2, 3, 4, 4]
    assert [point['combined']['accuracy'] for point in points] == [.25, .5, .25, .25, .5, .5]
    assert [point['fallback']['error_count'] for point in points] == [3, 2, 2, 1, 0, 0]
    assert [point['fallback']['unresolved_count'] for point in points] == [1, 1, 1, 0, 0, 0]
    assert points[1]['fallback']['accuracy'] == pytest.approx(1 / 3)
    assert points[2]['fallback']['accuracy'] == 0
    assert points[0]['combined'] == result['general']
    assert points[-1]['combined']['accuracy'] == inputs[1][0]['full_accuracy']
    assert points[-1]['fallback'] == {'accuracy': None, 'n_examples': 0, 'correct_count': 0,
                                     'error_count': 0, 'unresolved_count': 0}
    for point in result['combinations']:
        assert set(point['accepted_ids']).isdisjoint(point['rejected_ids'])
        assert set(point['accepted_ids']) | set(point['rejected_ids']) == set(result['validation_ids'])
        assert point['combined']['n_examples'] == 4
        assert point['coverage'] == point['accepted_count'] / 4
    for regime in result['regimes']:
        assert regime['landmarks'][0]['accuracy']['mean'] == .25
        assert regime['landmarks'][0]['accuracy']['sample_std'] == 0
        assert regime['landmarks'][-1]['fallback_accuracy']['mean'] is None
        assert regime['landmarks'][-1]['fallback_accuracy']['sample_std'] is None
    json.dumps(result, allow_nan=False)


def test_response_status_overrides_an_apparently_valid_label(inputs):
    general, runs = inputs
    general[0]['predicted_label'] = 'a'  # Would be right if a refused response were credited.
    general[0]['status'] = 'refused'
    result = evaluate(general, runs)
    assert result['general']['accuracy'] == .25
    assert result['general']['unresolved_count'] == 2
    assert result['status_counts'] == {'ok': 2, 'refused': 1, 'timeout': 1}


def test_alignment_is_by_unique_id_and_preserves_inputs(inputs):
    original = deepcopy(inputs)
    result = evaluate(*inputs)
    reordered = evaluate(list(reversed(inputs[0])), list(reversed(inputs[1])))
    assert reordered == result
    assert inputs == original


@pytest.mark.parametrize('change', ['missing', 'extra', 'duplicate', 'empty_id', 'ok_unknown', 'ok_none', 'no_status'])
def test_bad_general_response_records_are_rejected(inputs, change):
    general, runs = inputs
    if change == 'missing':
        general.pop()
    elif change == 'extra':
        general.append({'id': 'v:extra', 'predicted_label': 'a', 'status': 'ok'})
    elif change == 'duplicate':
        general.append(deepcopy(general[0]))
    elif change == 'empty_id':
        general[0]['id'] = ''
    elif change == 'ok_unknown':
        general[0]['predicted_label'] = 'unknown'
    elif change == 'ok_none':
        general[0]['predicted_label'] = None
    else:
        del general[0]['status']
    with pytest.raises(ValueError):
        evaluate(general, runs)


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'duplicate_run_id', 'changed_ids', 'changed_truth',
                                    'changed_label_order', 'rank_order', 'landmark_count', 'full_accuracy'])
def test_changed_specialist_records_are_rejected(inputs, change):
    general, runs = inputs
    if change == 'missing':
        runs.pop()
    elif change == 'duplicate':
        runs[-1] = deepcopy(runs[0])
    elif change == 'duplicate_run_id':
        runs[-1]['run_id'] = runs[0]['run_id']
    elif change == 'changed_ids':
        runs[-1]['ranked_rows'][0]['id'] = 'different'
    elif change == 'changed_truth':
        runs[-1]['ranked_rows'][0]['true_label'] = 'b'
        rows = runs[-1]['ranked_rows']
        runs[-1].update(compact(selective_curve(rows, ['a', 'b'], .25)))
    elif change == 'changed_label_order':
        runs[-1]['labels'] = ['b', 'a']
    elif change == 'rank_order':
        runs[-1]['ranked_rows'].reverse()
    elif change == 'landmark_count':
        runs[-1]['landmarks'][0]['accepted_count'] = 2
    else:
        runs[-1]['full_accuracy'] = .75
    with pytest.raises(ValueError):
        evaluate(general, runs)


def test_summary_retains_every_seed_and_uses_sample_sd(inputs):
    general, runs = inputs
    for run in runs:
        seed_index = SEEDS.index(run['seed'])
        rows = deepcopy(run['ranked_rows'])
        for index, row in enumerate(rows):
            row['predicted_label'] = row['true_label'] if index < seed_index else ('b' if row['true_label'] == 'a' else 'a')
        run.update(compact(selective_curve(rows, ['a', 'b'], seed_index / 4)))
    result = evaluate(general, runs)
    for regime in result['regimes']:
        full = regime['landmarks'][-1]
        values = [0, .25, .5, .75, 1]
        assert full['accuracy']['mean'] == .5
        assert full['accuracy']['sample_std'] == pytest.approx(statistics.stdev(values))
        assert full['accuracy']['by_seed'] == {str(seed): value for seed, value in zip(SEEDS, values)}
        f1s = [point['combined']['macro_f1'] for point in result['combinations']
               if point['shots'] == regime['shots'] and point['target_coverage'] == 1]
        assert full['macro_f1']['mean'] == pytest.approx(statistics.mean(f1s))
        assert full['macro_f1']['sample_std'] == pytest.approx(statistics.stdev(f1s))


def test_same_general_predictions_follow_different_ranked_rejected_sets(inputs):
    general, runs = inputs
    rows = deepcopy(runs[-1]['ranked_rows'])
    # Exchange confidence ranks without changing any validation ID, truth or prediction.
    rows[0]['confidence'], rows[1]['confidence'] = rows[1]['confidence'], rows[0]['confidence']
    runs[-1].update(compact(selective_curve(rows, ['a', 'b'], .5)))
    result = evaluate(general, runs)
    first = next(point for point in result['combinations'] if point['run_id'] == 'n5-s11' and point['target_coverage'] == .25)
    last = next(point for point in result['combinations'] if point['run_id'] == 'n20-s55' and point['target_coverage'] == .25)
    assert first['accepted_ids'] == ['v:0']
    assert last['accepted_ids'] == ['v:1']
    assert first['combined']['accuracy'] == .5
    assert last['combined']['accuracy'] == 0
    assert first['fallback']['accuracy'] == pytest.approx(1 / 3)
    assert last['fallback']['accuracy'] == 0
