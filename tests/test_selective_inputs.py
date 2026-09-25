"""Saved probability alignment and compact evidence must fail closed on corruption."""
from copy import deepcopy

import pytest

from baseline.experiment import classification_metrics
from baseline.selective import MODEL, compact, probability_rows, restore, selective_curve


@pytest.fixture
def inputs():
    labels = ['b', 'a', 'c']  # Deliberately not sorted like sklearn.classes_.
    predictions = [
        {'id': 'v:1', 'true_label': 'a', 'predicted_label': 'b'},
        {'id': 'v:2', 'true_label': 'a', 'predicted_label': 'a'},
        {'id': 'v:3', 'true_label': 'c', 'predicted_label': 'c'},
    ]
    record = {'samples': {'labels': labels, 'validation_ids': [p['id'] for p in predictions]},
              'predictions': predictions,
              'metrics': {MODEL: classification_metrics(['a', 'a', 'c'], ['b', 'a', 'c'], labels)}}
    probs = {'labels': labels, 'validation_ids': ['v:1', 'v:2', 'v:3'],
             'probabilities': [[.5, .4, .1], [.4, .4, .2], [.1, .2, .7]]}
    return record, probs


def test_extracts_max_confidence_with_exact_rows_and_class_order(inputs):
    record, probs = inputs
    rows = probability_rows(record, probs)
    assert [r['confidence'] for r in rows] == [.5, .4, .7]
    assert [r['id'] for r in rows] == probs['validation_ids']
    assert [r['true_label'] for r in rows] == ['a', 'a', 'c']
    # The second predicted class is valid despite tying a preceding probability column.
    assert rows[1]['predicted_label'] == 'a'


@pytest.mark.parametrize('change', ['class_order', 'row_order', 'prediction_order', 'nan', 'infinite',
                                    'negative', 'over_one', 'sum', 'shape', 'argmax', 'metric', 'unknown'])
def test_rejects_misalignment_or_invalid_probabilities(inputs, change):
    record, probs = deepcopy(inputs)
    if change == 'class_order':
        probs['labels'].reverse()
    elif change == 'row_order':
        probs['validation_ids'].reverse()
    elif change == 'prediction_order':
        record['predictions'].reverse()
    elif change in ('nan', 'infinite', 'negative', 'over_one'):
        probs['probabilities'][0][0] = {'nan': float('nan'), 'infinite': float('inf'), 'negative': -.5, 'over_one': 1.5}[change]
    elif change == 'sum':
        probs['probabilities'][0][0] = .2
    elif change == 'shape':
        probs['probabilities'][0].append(0)
    elif change == 'argmax':
        record['predictions'][0]['predicted_label'] = 'a'
    elif change == 'metric':
        record['metrics'][MODEL]['accuracy'] = 0
    else:
        record['predictions'][0]['true_label'] = 'unknown'
    with pytest.raises(ValueError):
        probability_rows(record, probs)


def test_compact_roundtrip_reconstructs_every_prefix_and_rejects_changed_counts(inputs):
    record, probs = inputs
    run = {'run_id': 'synthetic', 'shots': 5, 'seed': 11,
           **selective_curve(probability_rows(record, probs), record['samples']['labels'], 2 / 3)}
    saved = compact(run)
    assert 'curve' not in saved
    assert restore(saved) == run
    saved['landmarks'][0]['error_count'] += 1
    with pytest.raises(ValueError, match='landmarks'):
        restore(saved)
