"""Read completed EXP-007 evidence; export only aggregated counts and provenance.

No dataset, model or API invocation. Historical artifacts are never written.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

from reproduce import BUNDLE, SEEDS, confusion, digest, require, summary

ROOT = BUNDLE.parents[1]
RUN = ROOT/'artifacts/exp007-fixed-threshold-test-v1'
PROTOCOL = '0ff095f523ebc8f05d295dad4545f6a730dd07c6c9bc7ac194bbbb4a759cfa00'


def read(path): return json.loads(path.read_bytes())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(name, value):
    path = BUNDLE/name
    options = {'separators': (',', ':')} if name in ('counts.json', 'per_class.json') else {'indent': 2}
    raw = (json.dumps(value, sort_keys=True, allow_nan=False, **options)+'\n').encode()
    if path.exists(): require(path.read_bytes() == raw, 'Refusing changed compact evidence: '+name)
    else: path.write_bytes(raw)


def verify_and_export():
    before = {str(p.relative_to(RUN)): sha(p) for p in RUN.rglob('*') if p.is_file()}
    protocol = read(RUN/'protocol.json')
    require(sha(RUN/'protocol.json') == PROTOCOL == sha(ROOT/'experiments/exp007-fixed-threshold-preparation/protocol.json'), 'Frozen protocol drift')
    frozen, evaluation, checked = [read(RUN/p) for p in ('predictions_frozen.json','evaluation.json','verification.json')]
    checkpoint = read(RUN/'evaluation_checkpoint.json')
    require(checkpoint == {'evaluation': evaluation, 'verification': checked}, 'Atomic evaluation checkpoint mismatch')
    require(checked['evaluation_sha256'] == digest(evaluation) and
            checked['prediction_freeze_sha256'] == evaluation['prediction_freeze_sha256'] == digest(frozen), 'Evaluation/freeze digest mismatch')
    require(frozen['protocol_sha256'] == evaluation['protocol_sha256'] == checked['protocol_sha256'] == PROTOCOL, 'Protocol identity mismatch')
    require(all(checked[k] is True for k in ('all_3080_ids_retained','all_five_seeds_reported','predictions_frozen_before_label_join')),
            'Saved verification flags inconsistent')
    inventory = {'manifest.json', 'protocol.json', 'preflight.json'}
    for group in ('specialists','luna'):
        inventory.update(str(p.relative_to(RUN)) for p in (RUN/group).rglob('*.json') if p.name != 'authorizations.json')
    require(inventory == set(frozen['files_sha256']), 'Prediction freeze membership changed')
    require(all(before[p] == expected for p,expected in frozen['files_sha256'].items()), 'Frozen artifact bytes changed')
    labels = protocol['population']['labels']
    ids = [f'test:{i:05d}' for i in range(3080)]
    require(evaluation['n_examples'] == frozen['n_examples'] == 3080 and set(evaluation['seeds']) == set(SEEDS), 'Population mismatch')
    scored_rows = evaluation['seeds']['11']['predictions']
    require([r['id'] for r in scored_rows] == ids, 'Saved truth ID alignment mismatch')
    truth = [r['true_label'] for r in scored_rows]
    require(Counter(truth) == Counter({label:40 for label in labels}), 'Scoring-label support mismatch')
    execution = read(RUN/'luna/execution.json')
    luna_rows = execution['predictions']
    require(execution['status'] == 'completed' and execution['unresolved_count'] == 0 and
            [r['id'] for r in luna_rows] == ids and all(r['status']=='ok' for r in luna_rows), 'Shared Luna response set incomplete')
    luna = [r['predicted_label'] for r in luna_rows]
    counts = {'study_id':'EXP-007','evaluation_split':'official test','n_examples':3080,'labels':labels,
              'sparse_confusion_format':['true_class_index','predicted_class_index_or_minus_one_for_failure','count'],
              'luna':confusion(truth,luna,labels),'seeds':{}}
    max_difference = 0.
    def compare(actual, saved):
        nonlocal max_difference
        if isinstance(actual, dict):
            require(isinstance(saved,dict) and set(actual) <= set(saved), 'Saved metric keys mismatch')
            for k,v in actual.items(): compare(v,saved[k])
        elif isinstance(actual, float):
            difference = abs(actual-saved); max_difference = max(max_difference,difference)
            require(difference <= 1e-12, 'Saved numeric metric differs from independent replay')
        else: require(actual == saved, 'Saved metric/count differs from independent replay')
    for seed in SEEDS:
        saved = evaluation['seeds'][seed]
        rows = read(RUN/f'specialists/{seed}/predictions.json')
        prob = read(RUN/f'specialists/{seed}/probabilities.json')
        meta = read(RUN/f'specialists/{seed}/metadata.json')
        threshold = protocol['thresholds'][seed]['value']
        require([r['id'] for r in rows] == prob['ids'] == ids and prob['labels'] == labels, 'Specialist alignment mismatch')
        require(len(prob['values']) == 3080 and all(len(v)==77 and all(math.isfinite(x) and 0<=x<=1 for x in v) and abs(sum(v)-1)<1e-10 for v in prob['values']), 'Saved probability format changed')
        for row, values in zip(rows, prob['values']):
            best = max(range(77),key=values.__getitem__)
            require(row['confidence'] == values[best] and row['predicted_label'] == labels[best] and
                    type(row['use_specialist']) is bool and row['use_specialist'] == (values[best]>=threshold), 'Prediction or independent scalar gate mismatch')
        require(meta['binding']['protocol_sha256'] == PROTOCOL and meta['binding']['threshold_hex'] == threshold.hex() and
                meta['binding']['classifier_sha256'] == protocol['thresholds'][seed]['classifier_sha256'] and
                meta['frozen_encoder']['state_sha256_before'] == meta['frozen_encoder']['state_sha256_after'] == protocol['specialists']['embedding']['encoder_state_sha256'] and
                meta['frozen_encoder']['trainable_parameters'] == 0 and
                meta['classifier_state_sha256_before'] == meta['classifier_state_sha256_after'], 'Saved frozen-model provenance mismatch')
        require(all(sha(RUN/f'specialists/{seed}'/name)==expected for name,expected in meta['files_sha256'].items()), 'Specialist checkpoint hash mismatch')
        accepted = [r['use_specialist'] for r in rows]; sp=[r['predicted_label'] for r in rows]
        routed = [p if accept else l for p,l,accept in zip(sp,luna,accepted)]
        expected_rows = [{'id':rid,'true_label':t,'predicted_label':p,'source':'specialist' if a else 'gpt-6-luna'} for rid,t,p,a in zip(ids,truth,routed,accepted)]
        require(saved['predictions'] == expected_rows and saved['accepted_ids']==[rid for rid,a in zip(ids,accepted) if a] and
                saved['fallback_ids']==[rid for rid,a in zip(ids,accepted) if not a], 'Saved routing/label-join mismatch')
        rejected = [i for i,a in enumerate(accepted) if not a]
        pairs=Counter({'both_correct':0,'specialist_only_correct':0,'luna_only_correct':0,'both_wrong':0})
        for i in rejected:
            a,b=sp[i]==truth[i],luna[i]==truth[i]
            pairs['both_correct' if a and b else 'specialist_only_correct' if a else 'luna_only_correct' if b else 'both_wrong']+=1
        counts['seeds'][seed]={'threshold':threshold,'specialist':confusion(truth,sp,labels),
            'routed':confusion(truth,routed,labels),'rejected_luna':confusion([truth[i] for i in rejected],[luna[i] for i in rejected],labels),
            'rejected_specialist':confusion([truth[i] for i in rejected],[sp[i] for i in rejected],labels),
            'per_class_accepted':[sum(a and t==label for a,t in zip(accepted,truth)) for label in labels],
            'rejected_complementarity':dict(pairs)}
    result, classes = summary(counts)
    compare(result['luna_only'],evaluation['luna_only'])
    compare(classes['luna_only'],evaluation['luna_only']['per_class'])
    for seed,s in result['seeds'].items():
        saved=evaluation['seeds'][seed]
        for model in ('specialist_only','routed'):
            compare(s[model],saved[model]); compare(classes['seeds'][seed][model],saved[model]['per_class'])
        compare(classes['seeds'][seed]['acceptance'],saved['per_class_acceptance'])
        for a,b in [('accepted_count','accepted_count'),('observed_coverage','observed_coverage'),('fallback_count','luna_request_count'),
                    ('fallback_percentage','luna_request_percentage'),('fallback_accuracy_on_rejected','fallback_accuracy_on_rejected')]:compare(s[a],saved[b])
    for name in ('specialist_only','routed'):compare(result['aggregate'][name],evaluation['seed_summary'][name])
    for a,b in [('accepted_count','accepted_count'),('observed_coverage','observed_coverage'),('fallback_count','luna_request_count'),('fallback_percentage','luna_request_percentage'),('fallback_accuracy_on_rejected','fallback_accuracy_on_rejected')]:compare(result['aggregate'][a],evaluation['seed_summary'][b])
    write('counts.json',counts);write('summary.json',result);write('per_class.json',classes)
    require(before=={str(p.relative_to(RUN)):sha(p) for p in RUN.rglob('*') if p.is_file()}, 'Source artifacts changed during export')
    evidence={'source_directory':str(RUN.relative_to(ROOT)),'source_artifact_count':len(before),
        'source_artifact_inventory_sha256':digest(before),'source_prediction_freeze_sha256':digest(frozen),
        'source_files_sha256':{name:before[name] for name in ['evaluation.json','evaluation_checkpoint.json','verification.json','predictions_frozen.json','protocol.json','manifest.json','luna/execution.json','luna/manifest.json','luna/reservations.json','luna/source_compatibility.json']},
        'protocol_sha256':PROTOCOL,'dataset_revision':protocol['population']['revision'],
        'evaluation_utc':evaluation['evaluated_utc'],'source_code':read(RUN/'luna/source_compatibility.json'),
        'all_ids_and_probability_argmaxes_and_scalar_gates_verified':True,
        'all_scores_counts_per_class_and_sample_sd_independently_reproduced':True,
        'maximum_absolute_numeric_metric_difference':max_difference,'source_artifacts_byte_identical':True,
        'raw_test_file_opened':False,'label_source':'Existing saved evaluation rows; no raw dataset access',
        'api_calls':0,'model_executions':0,'exp009_executed':False}
    write('provenance.json',evidence)
    return result


if __name__ == '__main__':
    import os
    def guard(event,args):
        if event in ('socket.connect','socket.connect_ex','socket.getaddrinfo'):
            raise RuntimeError('Offline export forbids network')
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0])).absolute(); mode,flags=args[1:]
            if path.is_relative_to(ROOT/'data') or path.is_relative_to(ROOT/'.cache'):
                raise RuntimeError('Offline export forbids dataset/model access')
            writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if writing and not path.is_relative_to(BUNDLE): raise RuntimeError('Only compact bundle writes permitted')
    sys.addaudithook(guard)
    result=verify_and_export()
    print(json.dumps({'verified':True,'luna_only':result['luna_only'],'aggregate':result['aggregate']},indent=2))
