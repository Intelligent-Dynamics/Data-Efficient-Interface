"""Synthetic exact-score and whole-tie threshold tests; no dataset/model access."""
from decimal import Decimal
import inspect
import itertools
import json
import math

import numpy as np
import pytest

from baseline.selective import confidence_order
from baseline.thresholds import MAX_THRESHOLD, derive_threshold, route, use_specialist


def test_runtime_gate_is_exact_greater_than_or_equal_with_adjacent_binary64_values():
    cutoff = .5
    below = math.nextafter(cutoff, -math.inf)
    above = math.nextafter(cutoff, math.inf)
    assert use_specialist(below, cutoff) is False
    assert use_specialist(cutoff, cutoff) is True
    assert use_specialist(above, cutoff) is True
    assert route(below, cutoff) == 'gpt-6-luna'
    assert route(cutoff, cutoff) == route(above, cutoff) == 'specialist'
    assert use_specialist(np.float64(cutoff), np.float64(cutoff)) is True


def test_runtime_gate_endpoints_and_accept_none_sentinel():
    assert use_specialist(0.0, 0.0) is True
    assert use_specialist(1.0, 0.0) is True
    assert use_specialist(0.0, math.nextafter(0.0, math.inf)) is False
    assert use_specialist(math.nextafter(0.0, math.inf), math.nextafter(0.0, math.inf)) is True
    assert use_specialist(1.0, MAX_THRESHOLD) is False
    assert math.nextafter(1.0, math.inf) == MAX_THRESHOLD


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf'), -.1,
                               math.nextafter(1.0, math.inf), True, 1, '0.5', Decimal('.5'), np.float32(.5)])
def test_runtime_gate_rejects_invalid_or_implicitly_converted_confidences(bad):
    with pytest.raises(ValueError):
        use_specialist(bad, .5)
    with pytest.raises(ValueError):
        route(bad, .5)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf'), -.1,
                               math.nextafter(MAX_THRESHOLD, math.inf), False, 0, '.5', Decimal('.5'), np.float32(.5)])
def test_runtime_gate_rejects_invalid_thresholds(bad):
    with pytest.raises(ValueError):
        use_specialist(.5, bad)


def test_no_ties_reproduce_exact_frozen_prefix_with_unrounded_kth_score():
    ids = ['v:low', 'v:high', 'v:middle']
    scores = [.10000000000000002, .9, .4000000000000001]
    result = derive_threshold(ids, scores, 2)
    assert result['threshold'] == scores[2]
    assert result['threshold_hex'] == scores[2].hex()
    assert result['original_accepted_ids'] == ['v:high', 'v:middle']
    assert result['fixed_accepted_ids'] == ['v:high', 'v:middle']
    assert result['target_accepted_count'] == result['accepted_count'] == 2
    assert result['actual_coverage'] == 2 / 3
    assert result['minimum_symmetric_difference'] == 0
    assert result['added_ids'] == result['removed_ids'] == []
    assert result['boundary_ties_count'] == 1


@pytest.mark.parametrize('target,decision,count,difference', [
    (2, 'exclude_boundary_ties', 1, 1),
    (3, 'include_boundary_ties', 5, 2),  # Equal error: include the four boundary ties.
    (4, 'include_boundary_ties', 5, 1),
])
def test_boundary_ties_choose_nearest_whole_group_and_include_on_equal_error(target, decision, count, difference):
    ids = ['v:top', 'v:a', 'v:b', 'v:c', 'v:d', 'v:last']
    scores = [.9, .7, .7, .7, .7, .1]
    result = derive_threshold(ids, scores, target)
    assert result['boundary_confidence'] == .7
    assert result['boundary_ties_count'] == 4
    assert result['boundary_include_count'] == 5
    assert result['boundary_exclude_count'] == 1
    assert result['boundary_decision'] == decision
    assert result['accepted_count'] == count
    assert result['minimum_symmetric_difference'] == difference
    expected_threshold = .7 if decision == 'include_boundary_ties' else math.nextafter(.7, math.inf)
    assert result['threshold'] == expected_threshold
    desired = set(result['original_accepted_ids'])
    fixed = set(result['fixed_accepted_ids'])
    assert result['added_ids'] == sorted(fixed - desired)
    assert result['removed_ids'] == sorted(desired - fixed)
    assert len(desired ^ fixed) == difference


@pytest.mark.parametrize('scores', [[0.0], [1.0], [.25, .25, .25], [0.0, .7, 1.0], [-0.0, 0.0]])
def test_derive_endpoint_thresholds_match_none_and_all_of_the_observed_cases(scores):
    ids = [f'v:{index}' for index in range(len(scores))]
    empty = derive_threshold(ids, scores, 0)
    full = derive_threshold(ids, scores, len(ids))
    assert empty['threshold'] == math.nextafter(max(scores), math.inf)
    assert empty['accepted_count'] == 0
    assert empty['actual_coverage'] == 0
    assert empty['fixed_accepted_ids'] == []
    assert empty['minimum_symmetric_difference'] == 0
    assert full['threshold'] == min(scores)
    assert full['accepted_count'] == len(ids)
    assert full['fixed_accepted_ids'] == sorted(ids)
    assert full['actual_coverage'] == 1
    assert full['minimum_symmetric_difference'] == 0


def test_nearest_realizable_set_matches_brute_force_all_unique_cutpoints():
    ids = ['v:c', 'v:a', 'v:e', 'v:d', 'v:b']
    for scores in itertools.product((0.0, .5, 1.0), repeat=len(ids)):
        candidates = [*set(scores), math.nextafter(max(scores), math.inf)]
        order = confidence_order(ids, scores)
        for k in range(len(ids) + 1):
            desired = {ids[index] for index in order[:k]}
            differences = [len(desired ^ {row_id for row_id, score in zip(ids, scores) if score >= cutoff})
                           for cutoff in candidates]
            result = derive_threshold(ids, scores, k)
            assert result['minimum_symmetric_difference'] == min(differences)
            assert result['minimum_symmetric_difference'] == abs(result['accepted_count'] - k)
            assert result['fixed_accepted_ids'] == sorted(row_id for row_id, score in zip(ids, scores)
                                                        if use_specialist(score, result['threshold']))


def test_input_permutation_and_single_request_processing_cannot_change_decisions():
    ids = ['v:c', 'v:a', 'v:d', 'v:b']
    scores = [.9, .5, .5, .2]
    expected = derive_threshold(ids, scores, 2)
    for permutation in itertools.permutations(range(4)):
        permuted_ids = [ids[i] for i in permutation]
        permuted_scores = [scores[i] for i in permutation]
        assert derive_threshold(permuted_ids, permuted_scores, 2) == expected
        independent = {row_id: route(score, expected['threshold']) for row_id, score in zip(permuted_ids, permuted_scores)}
        assert {row_id for row_id, target in independent.items() if target == 'specialist'} == set(expected['fixed_accepted_ids'])
    # Other requests and batch size are absent from the runtime API entirely.
    assert list(inspect.signature(use_specialist).parameters) == ['confidence', 'threshold']
    assert list(inspect.signature(route).parameters) == ['confidence', 'threshold']
    with pytest.raises(TypeError):
        use_specialist(.8, .5, label='synthetic')
    with pytest.raises(TypeError):
        route(.8, .5, request_id='v:unrelated')


def test_derivation_never_receives_labels_and_is_unchanged_when_labels_are_permuted():
    rows = [{'id': 'v:a', 'confidence': .8, 'true_label': 'a'},
            {'id': 'v:b', 'confidence': .8, 'true_label': 'b'},
            {'id': 'v:c', 'confidence': .4, 'true_label': 'c'}]
    expected = derive_threshold([row['id'] for row in rows], [row['confidence'] for row in rows], 1)
    for labels in itertools.permutations(['a', 'b', 'c']):
        for row, label in zip(rows, labels):
            row['true_label'] = label
        assert derive_threshold([row['id'] for row in rows], [row['confidence'] for row in rows], 1) == expected
    assert list(inspect.signature(derive_threshold).parameters) == ['ids', 'confidence', 'accepted_count']


def test_signed_zero_representation_is_permutation_invariant():
    ids, scores = ['a', 'b', 'c'], [-0.0, 0.0, .5]
    for count in range(4):
        first = derive_threshold(ids, scores, count)
        second = derive_threshold(list(reversed(ids)), list(reversed(scores)), count)
        assert first == second
        assert first['threshold_hex'] == second['threshold_hex']
        assert first['boundary_confidence'].hex() == second['boundary_confidence'].hex()


@pytest.mark.parametrize('count', [0, 1, 2, 3, 4])
def test_json_and_decimal_hex_roundtrip_preserve_threshold_bits_and_routing(count):
    ids = ['v:0', 'v:1', 'v:2', 'v:3']
    scores = [1.0, .12345678901234568, .12345678901234568, 0.0]
    original = derive_threshold(ids, scores, count)
    restored = json.loads(json.dumps(original, allow_nan=False))
    assert restored == original
    assert restored['threshold'].hex() == original['threshold_hex']
    assert float(restored['threshold_decimal']).hex() == original['threshold_hex']
    assert float.fromhex(restored['threshold_hex']).hex() == original['threshold_hex']
    assert [route(score, original['threshold']) for score in scores] == [route(score, restored['threshold']) for score in scores]


@pytest.mark.parametrize('ids,scores,count', [
    ([], [], 0), (['a'], [], 0), (['a'], [.1, .2], 0), (['a', 'a'], [.1, .2], 1),
    ([''], [.1], 0), ([None], [.1], 0), (['a'], [.1], -1), (['a'], [.1], 2),
    (['a'], [.1], 1.0), (['a'], [.1], True), (['a'], [.1], None),
    (['a'], [float('nan')], 0), (['a'], [float('inf')], 0), (['a'], [-.1], 0),
    (['a'], [1.1], 0), (['a'], [np.float32(.5)], 0), (['a'], [1], 0),
])
def test_invalid_derivation_inputs_fail_closed(ids, scores, count):
    with pytest.raises(ValueError):
        derive_threshold(ids, scores, count)
