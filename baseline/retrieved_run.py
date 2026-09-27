"""Bounded EXP-009 CLI: offline preparation, separately approved pilot, final companion."""
import argparse
from dataclasses import asdict
from decimal import Decimal
import gzip
import json
from pathlib import Path
import sys
import time
import os

import numpy as np

from .data import ROOT, read_json, json_bytes, sha256
from .general import atomic_json, lock, now
from .general_protocol import digest, cache_key, PRICING, SETTINGS, ENDPOINT
from .final_protocol import load_frozen, Authorization, unseal_inputs, scoring_truth, FROZEN_PROTOCOL_SHA256
from .recovery import POLICY
from .selective import require
from .retrieved import (BUNDLE, PREPARED, REFERENCE, load_validation_assets, instructions, pilot_ids,
                        prepare_requests, cost_plan, evaluate_companion)
from .retrieved_collection import Approval, collect, verify_completed

PROTOCOL_SHA256 = 'b78f3d32a9bd86ea130ce1f51e2feded9d52b5f5ff796507c49d6f53fccc3334'


def load_protocol():
    raw=(BUNDLE/'protocol.json').read_bytes()
    require(sha256(raw)==PROTOCOL_SHA256,'EXP-009 frozen protocol bytes changed')
    protocol=json.loads(raw)
    prompt=(BUNDLE/'prompt.txt').read_text()
    require(sha256(prompt.encode())==protocol['prompt_sha256'],'EXP-009 prompt changed')
    old,_,schema=load_frozen()
    require(protocol['source_exp007_sha256']==digest(old) and protocol['schema']==schema and
            protocol['threshold']==old['thresholds']['11'] and protocol['embedding']==old['specialists']['embedding'],
            'Original model/schema/threshold identity drift')
    return protocol,prompt,schema


def _write_once(path,value):
    if path.exists():
        require(read_json(path)==value,'Prepared evidence changed; do not overwrite')
    else:
        atomic_json(path,value)


def _load_prepared(scope, protocol):
    directory=ROOT/protocol['outputs'][scope]
    prepared=read_json(directory/'prepared.json')
    require(prepared['protocol_sha256']==digest(protocol) and prepared['scope']==scope,'Prepared scope/protocol drift')
    requests=read_json(directory/'requests.json')
    require(digest(requests)==prepared['requests_sha256'],'Prepared exact request bytes changed')
    require(set(requests)==set(prepared['ids']) and len(prepared['ids'])==len(requests), 'Prepared row membership changed')
    requests={i:requests[i] for i in prepared['ids']}
    if scope=='test':
        require(digest(read_json(directory/'retrieval_evidence.json'))==prepared['retrieval_evidence_sha256'],
                'Saved retrieval evidence changed')
        require(digest(read_json(directory/'specialist.json'))==prepared['specialist_predictions_sha256'],
                'Saved specialist evidence changed')
    return directory,requests,prepared


def prepare_validation():
    protocol,prompt,schema=load_protocol()
    old,_,record,pool,queries,values=load_validation_assets()
    require(pool['sha256']==protocol['candidate_pool_sha256'] and pool['ids']==protocol['candidate_ids'] and
            [q['id'] for q in queries]==protocol['validation_ids'],'Pool/validation changed')
    requests,evidence,durations=prepare_requests(pool,queries,values,prompt,schema)
    require(digest(evidence)==protocol['validation_request_evidence_sha256'],'Retrieval/request drift')
    for scope,ids in [('pilot',protocol['pilot_ids']),('validation',protocol['validation_ids'])]:
        selected={i:requests[i] for i in ids}
        directory=ROOT/protocol['outputs'][scope]
        _write_once(directory/'requests.json',selected)
        _write_once(directory/'prepared.json',{'study_id':'EXP-009','protocol_sha256':digest(protocol),'scope':scope,
                                              'requests_sha256':digest(selected),'ids':ids})
    return {'status':'PREPARED; NO API CALLS','protocol_sha256':digest(protocol),
            'pilot_requests':20,'validation_requests':770,'test_access':False,'api_calls':0,
            'costs':cost_plan(evidence,protocol['pilot_ids'])}


def _test_bridge(approval, protocol, *, live):
    approval.validate(protocol,live=live,test=True)
    return Authorization(FROZEN_PROTOCOL_SHA256,True,approval.authorize_live,approval.cap_usd,approval.compatibility_date)


def prepare_test(approval):
    """Future authorized path only; never used during preparation or synthetic checks."""
    protocol,prompt,schema=load_protocol()
    bridge=_test_bridge(approval,protocol,live=False)
    old,_,_=load_frozen()
    queries=unseal_inputs(bridge,old,live=False)
    from .final_test import verify_frozen_predictions
    from .final_specialists import (verify_saved_specialists,verify_specialist_files,
                                    _load_classifier,_predictions,_validate_classifier,_classifier_state)
    from .embeddings import assert_frozen,state_digest,ENCODING
    from threadpoolctl import threadpool_limits
    import torch
    original=ROOT/old['outputs']['root']
    frozen=verify_frozen_predictions(original,old)
    specialists=verify_saved_specialists(queries,old,original)
    _,_,_,pool,_,_=load_validation_assets()
    require(pool['sha256']==protocol['candidate_pool_sha256'],'Candidate pool changed')
    directory=ROOT/protocol['outputs']['test']
    if (directory/'prepared.json').exists():
        saved,requests,receipt=_load_prepared('test',protocol)
        require(receipt['query_inputs_sha256']==digest(queries) and receipt['original_prediction_freeze_sha256']==digest(frozen),
                'Test resume input/original prediction drift')
        return {'status':'PREPARED','requests':len(requests),'reservation_usd':receipt['reservation_usd']}
    checked=verify_specialist_files(old)
    os.environ['TOKENIZERS_PARALLELISM']='false'
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1: torch.set_num_interop_threads(1)
    torch.manual_seed(11);torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1):
        from .cpu_benchmark import load_benchmark_encoder
        encoder=load_benchmark_encoder(checked['snapshot'])
        assert_frozen(encoder);before=state_digest(encoder)
        require(before==protocol['embedding']['encoder_state_sha256'],'Encoder state changed')
        start=time.perf_counter()
        with torch.inference_mode():
            values=encoder.encode([q['text'] for q in queries],batch_size=32,precision='float32',device='cpu',
                                  normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False,prompt=None,prompt_name=None)
        encoding_seconds=time.perf_counter()-start
        assert_frozen(encoder);require(state_digest(encoder)==before,'Encoder changed during inference')
        classifier=_load_classifier(ROOT/protocol['threshold']['classifier_path'])
        order=_validate_classifier(classifier,protocol['labels'],protocol['embedding']['logistic_regression'])
        classifier_before=_classifier_state(classifier)
        predictions=_predictions([q['id'] for q in queries],protocol['labels'],classifier.predict_proba(values)[:,order],protocol['threshold']['value'])
        require(_classifier_state(classifier)==classifier_before and predictions==specialists['11'],
                'Replayed seed-11 predictions/gates differ from frozen original')
        requests,evidence,durations=prepare_requests(pool,queries,values,prompt,schema)
    reserve=sum((Decimal(r['attempt_reservation_usd'])*4 for r in evidence),Decimal(0))
    _write_once(directory/'requests.json',requests)
    _write_once(directory/'specialist.json',predictions)
    _write_once(directory/'retrieval_evidence.json',evidence)
    if not (directory/'overhead.json').exists():
        _write_once(directory/'overhead.json',{'query_encoding_seconds':encoding_seconds,'per_query':durations,
                                        'note':'Query vector reused for retrieval and seed-11 classifier; original EXP007 does not persist vectors, so separate comparison invocation re-encodes once.'})
    _write_once(directory/'prepared.json',{'study_id':'EXP-009','scope':'test','protocol_sha256':digest(protocol),
                'requests_sha256':digest(requests),'ids':list(requests),'query_inputs_sha256':digest(queries),
                'original_prediction_freeze_sha256':digest(frozen),'reservation_usd':str(reserve),
                'specialist_predictions_sha256':digest(predictions),'retrieval_evidence_sha256':digest(evidence)})
    return {'status':'PREPARED; NO API CALLS','requests':len(requests),'reservation_usd':str(reserve)}


def evaluate(scope,approval):
    protocol,_,_=load_protocol()
    if scope=='test':
        _test_bridge(approval,protocol,live=False)
    directory,requests,prepared=_load_prepared(scope,protocol)
    report=verify_completed(requests,protocol,directory/'collection')
    frozen={'requests_sha256':digest(requests),'execution_sha256':digest(report),'protocol_sha256':digest(protocol)}
    _write_once(directory/'predictions_frozen.json',frozen)
    if (directory/'evaluation.json').exists():
        result=read_json(directory/'evaluation.json')
        require(result['prediction_freeze_sha256']==digest(frozen),'Completed score binding changed')
        return result
    if scope=='test':
        from .final_test import verify_frozen_predictions
        old,_,_=load_frozen()
        original=ROOT/old['outputs']['root']
        source_frozen=verify_frozen_predictions(original,old)
        require(digest(source_frozen)==prepared['original_prediction_freeze_sha256'],'Original evidence drift')
        specialist=read_json(directory/'specialist.json')
        require(digest(specialist)==prepared['specialist_predictions_sha256'],'Companion specialist evidence drift')
        zero=read_json(original/'luna/execution.json')['predictions']
        # Reuse existing protected loader at its test-only authorization scope. All
        # predictions are immutable before labels are made available to scoring.
        from .final_protocol import _authorized_csv
        bridge=_test_bridge(approval,protocol,live=False)
        records=_authorized_csv(bridge,old,ROOT,live=False)
        truth={f'test:{i:05d}':r['category'] for i,r in enumerate(records)}
    else:
        old,_,record,_,_,_=load_validation_assets()
        ids=list(requests)
        truth={r['id']:r['true_label'] for r in record['predictions'] if r['id'] in ids}
        # Use EXP005's full precision confidences rather than rounded record display.
        from .selective import restore
        run=restore(read_json(ROOT/'experiments/exp005-selective-diagnostic/runs/exp004-minilm-v2-n20-s11.json'))
        specialist=[{'id':r['id'],'predicted_label':r['predicted_label'],'confidence':r['confidence'],
                     'use_specialist':r['confidence']>=protocol['threshold']['value']}
                    for r in run['ranked_rows'] if r['id'] in ids]
        saved=read_json(ROOT/'experiments/exp006-completed-validation/predictions.json')
        zero=[r for r in saved['predictions'] if r['id'] in ids] if isinstance(saved,dict) else [r for r in saved if r['id'] in ids]
    result=evaluate_companion(truth,specialist,zero,report['predictions'],protocol['labels'],protocol['threshold']['value'])
    result.update(protocol_sha256=digest(protocol),prediction_freeze_sha256=digest(frozen),scope=scope,
                  api_accounting=report['accounting'],status='PAID RESULTS' if report['kind']=='live' else 'SYNTHETIC ONLY',
                  limitation='Pilot is formatting/usage/cost only; not a reliable 77-class quality estimate. Never select prompt using pilot accuracy.' if scope=='pilot' else 'Single seed/single training pool; report deterioration/null effects equally.')
    _write_once(directory/'evaluation.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['dry-run','prepare-validation','pilot-live','validation-live','test-preflight','test-live','evaluate-pilot','evaluate-validation','evaluate-test'])
    parser.add_argument('--approved-protocol-sha256',default='')
    parser.add_argument('--authorize-live',action='store_true')
    parser.add_argument('--authorize-test-access',action='store_true')
    parser.add_argument('--spending-cap-usd',default='')
    parser.add_argument('--acknowledge-model-pricing-date',default='')
    args=parser.parse_args(argv)
    approval=Approval(args.approved_protocol_sha256,args.authorize_live,args.spending_cap_usd,args.acknowledge_model_pricing_date,args.authorize_test_access)
    protocol,_,_=load_protocol()
    if args.mode=='dry-run':
        result={'status':'PREPARED; NOT EVALUATED; NO API CALLS','protocol_sha256':digest(protocol),
                'costs':read_json(BUNDLE/'costs.json'),'test_access':False,'api_calls':0}
    elif args.mode=='prepare-validation': result=prepare_validation()
    elif args.mode=='test-preflight': result=prepare_test(approval)
    elif args.mode.endswith('-live'):
        scope=args.mode.removesuffix('-live')
        approval.validate(protocol,live=True,test=scope=='test')
        if scope=='test': prepare_test(approval)
        else: prepare_validation()
        directory,requests,_=_load_prepared(scope,protocol)
        result=collect(requests,protocol,directory/'collection',approval,scope)
        # Collection and scoring remain separate commands, never tune after scores.
        result={k:result[k] for k in ('status','halt_reason','accounting')}
    else: result=evaluate(args.mode.removeprefix('evaluate-'),approval)
    print(json.dumps({k:v for k,v in result.items() if k not in ('predictions','metrics')},indent=2,allow_nan=False))


if __name__=='__main__':
    try: main()
    except (ValueError,OSError,KeyError):
        print('EXP-009 stopped: check frozen evidence, explicit approval, scope, model/pricing compatibility and cap; no substitutions.',file=sys.stderr)
        raise SystemExit(2)
