"""Synthetic independent arithmetic checks; no dataset or model accesses."""
import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / 'experiments/v1-complementary-validation/reproduce.py'
spec = importlib.util.spec_from_file_location('v1_validation_evidence', MODULE_PATH)
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


def test_independent_metrics_keep_unresolved_and_missing_class_denominators():
    result = evidence.metrics(['a', 'a', 'b', 'b'], ['a', None, 'a', None], ['a', 'b', 'c'])
    assert result['n_examples'] == 4
    assert result['correct_count'] == 1
    assert result['error_count'] == 3
    assert result['accuracy'] == 0.25
    # a: 2TP/(truth count + predicted count)=2/4; b and c: zero.
    assert result['macro_f1'] == pytest.approx(1 / 6)


def test_complementarity_counts_harms_as_well_as_rescues():
    cells = evidence.paired(['a'] * 5, ['a', 'a', 'b', 'b', 'b'], ['a', 'b', 'a', 'a', 'b'])
    assert cells == {'both_correct': 1, 'specialist_only_correct': 1,
                     'luna_only_correct': 2, 'both_wrong': 1, 'net_correct_added_by_luna': 1}
    assert sum(cells[k] for k in ('both_correct', 'specialist_only_correct', 'luna_only_correct', 'both_wrong')) == 5


def test_independent_metrics_reject_alignment_loss():
    with pytest.raises(ValueError, match='row mismatch'):
        evidence.metrics(['a', 'b'], ['a'], ['a', 'b'])
