"""Regression checks on compact saved evidence; no private data/model/API access."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / 'experiments/exp009-official-test'
spec = importlib.util.spec_from_file_location('exp009_official_evidence_replay', DIRECTORY / 'reproduce.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def inputs():
    return tuple(json.loads((DIRECTORY / name).read_text()) for name in
                 ('summary.json', 'per_class.json', 'accounting.json'))


def test_completed_compact_evidence_reproduces():
    result = replay.reproduce(DIRECTORY)
    assert result['verified'] and result['n_examples'] == 3080
    assert result['metrics']['retrieved_luna']['correct_count'] == 2839
    assert result['metrics']['retrieved_hybrid']['correct_count'] == 2713
    assert result['maximum_absolute_arithmetic_difference'] == 1.1102230246251565e-16
    assert result['spend_interval_under_frozen_assumptions_usd'] == ['0.576343155', '0.584791155']


def test_unknown_prediction_keeps_full_metric_denominator():
    result, _ = replay.metrics({'a': {'true_positive': 1, 'false_positive': 0,
                                    'false_negative': 1, 'support': 2,
                                    'precision': 1., 'recall': .5, 'f1': 2/3}})
    assert result['accuracy'] == .5 and result['unresolved_count'] == 1
    assert result['n_examples'] == 2 and result['macro_f1'] == 2/3


@pytest.mark.parametrize('key,value', [('true_positive', -1), ('support', 41),
                                     ('false_positive', True), ('precision', float('nan'))])
def test_invalid_or_inconsistent_class_counts_rejected(key, value):
    summary, classes, accounting = inputs()
    classes['metrics']['specialist']['card_arrival'][key] = value
    with pytest.raises(ValueError):
        replay.replay(summary, classes, accounting)


@pytest.mark.parametrize('mutation', ['missing_class', 'missing_arm', 'coverage', 'paired_delta',
                                      'fallback_correct', 'budget', 'protocol', 'invoice', 'usage'])
def test_evidence_drift_rejected(mutation):
    summary, classes, accounting = inputs()
    if mutation == 'missing_class':
        del classes['metrics']['retrieved_luna']['card_arrival']
    elif mutation == 'missing_arm':
        del summary['metrics']['specialist']
    elif mutation == 'coverage':
        summary['accepted_count'] -= 1
    elif mutation == 'paired_delta':
        summary['paired_differences']['retrieved_hybrid minus specialist']['accuracy'] = 0
    elif mutation == 'fallback_correct':
        summary['fallback_correct_counts']['retrieved'] += 1
    elif mutation == 'budget':
        summary['additional_validation_labels'] = 0
    elif mutation == 'protocol':
        summary['protocol_sha256'] = '0'*64
    elif mutation == 'invoice':
        accounting['accounting']['actual_api_spend_usd'] = '0.576343155'
    elif mutation == 'usage':
        accounting['usage_totals']['output_tokens'] += 1
    with pytest.raises(ValueError):
        replay.replay(summary, classes, accounting)


def test_compact_file_tampering_rejected(tmp_path):
    for path in DIRECTORY.iterdir():
        if path.is_file():
            shutil.copyfile(path, tmp_path / path.name)
    with (tmp_path / 'summary.json').open('a') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='hash mismatch'):
        replay.reproduce(tmp_path)


def test_export_contains_no_raw_rows_or_private_response_fields():
    prohibited = {'predictions', 'predicted_label', 'true_label', 'request', 'response', 'body',
                  'request_id', 'provider_request_id', 'api_key', 'OPENAI_API_KEY', 'customer_message',
                  'canonical_id', 'cache_key'}
    def inspect(value):
        if isinstance(value, dict):
            assert not prohibited.intersection(value)
            for child in value.values():
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)
        elif isinstance(value, str):
            assert not value.startswith('test:')
    for path in DIRECTORY.glob('*.json'):
        inspect(json.loads(path.read_text()))


def test_exact_original_paired_floats_and_fallback_ratios_preserved():
    summary, _, _ = inputs()
    expected = {
        'retrieved_luna minus zero_shot_luna': (.10811688311688317, .11548946592207376),
        'retrieved_luna minus specialist': (.06590909090909092, .06787688103908784),
        'retrieved_hybrid minus specialist': (.025000000000000022, .026354967228937043),
        'retrieved_hybrid minus retrieved_luna': (-.040909090909090895, -.0415219138101508),
        'retrieved_hybrid minus zero_shot_hybrid': (.01558441558441559, .015933466598594737),
    }
    assert summary['paired_differences'] == {
        pair: {'accuracy': values[0], 'macro_f1': values[1]} for pair, values in expected.items()}
    assert summary['fallback_accuracy'] == {'retrieved': .753731343283582, 'zero_shot': .5746268656716418}
