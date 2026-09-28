"""Offline aggregate replay checks; synthetic cases and compact versioned evidence only."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import statistics

import pytest

PATH = Path(__file__).resolve().parents[1]/'experiments/exp007-official-test/reproduce.py'
spec = importlib.util.spec_from_file_location('exp007_official_replay', PATH)
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


def test_independent_confusions_keep_failed_outputs_in_full_denominator():
    cells = evidence.confusion(['a','a','b','b'], ['a',None,'a',None], ['a','b','c'])
    result = evidence.metrics(cells, ['a','b','c'])
    assert result['n_examples'] == 4 and result['unresolved_count'] == 2
    assert result['accuracy'] == .25 and result['macro_f1'] == pytest.approx(1/6)
    assert result['per_class']['a']['false_positive'] == 1
    assert result['per_class']['b']['false_negative'] == 2


@pytest.mark.parametrize('cells', [[], [[0,0,0]], [[0,0,-1]], [[0,0,1.5]],
                                  [[0,0,1],[0,0,2]], [[1,0,1]], [[0,-2,1]]])
def test_confusion_replay_rejects_invalid_or_duplicated_counts(cells):
    with pytest.raises(ValueError):
        evidence.metrics(cells,['a'])


def test_confusion_creation_rejects_alignment_loss():
    with pytest.raises(ValueError):
        evidence.confusion(['a','b'],['a'],['a','b'])


def synthetic_counts():
    labels = [f'class_{i}' for i in range(77)]
    full = [[i,i,40] for i in range(77)]
    seeds = {}
    for i,seed in enumerate(evidence.SEEDS,1):
        rejected = [[j,j,40] for j in range(i)]
        seeds[seed] = {'threshold': .1, 'specialist':full, 'routed':full,
            'rejected_luna':rejected, 'rejected_specialist':rejected,
            'per_class_accepted':[0]*i+[40]*(77-i),
            'rejected_complementarity':{'both_correct':40*i,'specialist_only_correct':0,'luna_only_correct':0,'both_wrong':0}}
    return {'labels':labels,'n_examples':3080,'luna':full,'seeds':seeds}


def test_aggregate_uses_all_five_seeds_and_sample_sd():
    actual, classes = evidence.summary(synthetic_counts())
    assert actual['aggregate']['fallback_count']['mean'] == 120
    assert actual['aggregate']['fallback_count']['sample_sd'] == statistics.stdev([40,80,120,160,200])
    assert actual['aggregate']['specialist_only']['accuracy']['mean'] == 1
    assert len(classes['seeds']) == 5 and len(classes['luna_only']) == 77


@pytest.mark.parametrize('fault',['missing_seed','coverage','negative_acceptance','paired_counts'])
def test_aggregate_rejects_population_coverage_or_pairing_drift(fault):
    data = deepcopy(synthetic_counts())
    if fault=='missing_seed': del data['seeds']['11']
    elif fault=='coverage': data['seeds']['11']['per_class_accepted'][0]=1
    elif fault=='negative_acceptance': data['seeds']['11']['per_class_accepted'][0]=-1
    else: data['seeds']['11']['rejected_complementarity']['both_wrong']=1
    with pytest.raises(ValueError): evidence.summary(data)


def test_compact_official_results_reproduce_without_private_artifacts():
    result = evidence.reproduce()
    assert result['n_examples']==3080 and result['luna_only']['correct_count']==2506
    assert result['aggregate']['accuracy_gain_percentage_points']['mean']==pytest.approx(1.051948051948055)
    assert [s['fallback_count'] for s in result['seeds'].values()]==[268,270,275,282,293]


def test_exported_json_has_no_private_response_or_request_fields():
    import json
    forbidden = {'request','response','body','request_id','customer_message','true_label','predicted_label','api_key','OPENAI_API_KEY'}
    def check(value):
        if isinstance(value,dict):
            assert not forbidden.intersection(value)
            for child in value.values(): check(child)
        elif isinstance(value,list):
            for child in value: check(child)
    for path in evidence.BUNDLE.glob('*.json'): check(json.loads(path.read_text()))
