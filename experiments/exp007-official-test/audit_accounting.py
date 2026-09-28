"""Read-only accounting audit from private saved responses; print aggregates only.

Writes a NEW temporary report, never modifies the live run or compact checkpoint.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json, os, socket
from collections import Counter, defaultdict
from copy import deepcopy
from decimal import Decimal
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'artifacts/exp007-fixed-threshold-test-v1'
sys.path.insert(0,str(ROOT))
guards={'network_attempts':0,'raw_data_or_model_access_attempts':0,'run_write_attempts':0}
def guard(event,args):
    if event in ('socket.connect','socket.getaddrinfo','socket.bind'):
        guards['network_attempts']+=1; raise RuntimeError('Network forbidden')
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0])).absolute()
        protected=[ROOT/'data/raw',ROOT/'data/processed',ROOT/'.cache',ROOT/'artifacts/exp004-minilm-learning-curve']
        if any(p==r or p.is_relative_to(r) for r in protected):
            guards['raw_data_or_model_access_attempts']+=1; raise RuntimeError('Raw dataset/model cache read forbidden')
        mode=args[1] or ''
        if p.is_relative_to(RUN) and (any(c in mode for c in 'wax+') or (args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))):
            guards['run_write_attempts']+=1; raise RuntimeError('Run write forbidden')
sys.addaudithook(guard)
from baseline import final_collection as fc, general, recovery
from baseline.final_protocol import load_frozen, Authorization, FROZEN_PROTOCOL_SHA256, runtime_code, verify_preparation
from baseline.final_compatibility import validate_receipt, accepted_manifest, record_resume_source
from baseline.general_protocol import digest, estimate, cache_key, payload, PRICING
from baseline.data import sha256, read_json
assert RUN.is_dir()
def inv():
    return {str(p.relative_to(RUN)):sha256(p.read_bytes()) for p in sorted(RUN.rglob('*')) if p.is_file()}
before=inv()
protocol,prompt,schema=load_frozen(ROOT)
verify_preparation(ROOT,protocol)
manifest=read_json(RUN/'luna/manifest.json')
outer=read_json(RUN/'manifest.json')
plan=read_json(RUN/'preflight.json')
execution=read_json(RUN/'luna/execution.json')
ledger=read_json(RUN/'luna/reservations.json')
compat=read_json(RUN/'luna/source_compatibility.json')
auth=Authorization(FROZEN_PROTOCOL_SHA256,True,True,manifest['cap_usd'],'2026-09-28',compat['receipt_sha256'])
receipt=validate_receipt(ROOT,auth)
# A documentation/evidence checkpoint changes Git HEAD, not the executed runtime.
# Audit the recorded execution identity; do not pretend this is a live resume.
current_code=fc.source_code()
assert {k:v for k,v in current_code.items() if k!='git_head'}=={k:v for k,v in compat['code'].items() if k!='git_head'}
record_resume_source(RUN/'luna',auth,compat['code'],root=ROOT,write=False)
entries={}
rows=[]
for p in sorted((RUN/'luna/responses').glob('*.json')):
    e=read_json(p)
    key=p.stem
    assert key==e['cache_key']==cache_key(e['request'])
    text=json.loads(e['request']['input'][0]['content'][0]['text'])['customer_message']
    assert e['request']==payload(text,prompt,schema,protocol['fallback']['settings'])
    assert e['id'] not in {r['id'] for r in rows}
    rows.append({'id':e['id'],'text':text})
    request={'id':e['id'],'body':e['request'],'estimate':estimate(e['request'])}
    entries[key]=fc._read_entry(RUN/'luna',key,request,protocol,'live')
assert len(entries)==3080
rows.sort(key=lambda r:r['id'])
assert plan==fc.collection_plan(rows,protocol,prompt,schema)
expected_manifest=fc._manifest(rows,protocol,plan['aliases'],Decimal(manifest['cap_usd']),Decimal(plan['full_retry_reservation_usd']),auth,'live')
accepted_manifest(RUN/'luna/manifest.json',expected_manifest,auth,root=ROOT)
expected_outer={'study_id':'EXP-007','protocol_sha256':digest(protocol),'rows_sha256':digest(rows),'code_files_sha256':runtime_code(),'cap_usd':manifest['cap_usd'],'test_sha256':protocol['population']['sealed_file_sha256_from_existing_metadata']}
accepted_manifest(RUN/'manifest.json',expected_outer,auth,root=ROOT)
general.sync_reservation_ledger(RUN/'luna',manifest,entries,write=False)
assert ledger['attempt_counts']=={key:len(e['attempts']) for key,e in entries.items()}
expected=[{'id':a['id'],'cache_key':a['cache_key'],'canonical_id':a['canonical_id'],**general.final_result(entries[a['cache_key']])} for a in plan['aliases']]
assert execution['predictions']==expected and execution['aliases']==plan['aliases']
assert execution['status']=='completed' and execution['halt_reason'] is None and execution['unresolved_count']==0 and execution['unresolved_ids']==[]
assert execution['status_counts']==dict(Counter(p['status'] for p in expected))=={'ok':3080}
assert execution['unique_request_payloads']==3080
assert execution['protocol_sha256']==digest(protocol)
accounting=general.accounting(entries)
assert execution['accounting']==accounting
all_attempts=[a for e in entries.values() for a in e['attempts']]
assert all(not a.get('halt') for a in all_attempts)
assert all(e['attempts'][-1]['status']=='ok' for e in entries.values())
bins={}
usage_unknown=Counter()
for a in all_attempts:
    if a['usage_priced_usd'] is None:
        usage_unknown[a['status']]+=1
        continue
    u=a['usage']; key=(a['status'],a['returned_model'],a['returned_service_tier'])
    if key not in bins:
        bins[key]={'status':key[0],'model':key[1],'service_tier':key[2],'attempts':0,'input_tokens':0,'cached_input_tokens':0,'cache_write_input_tokens':0,'output_tokens':0,'reasoning_output_tokens':0,'reasoning_detail_missing_attempts':0,'known_usage_priced_usd':Decimal(0)}
    b=bins[key];b['attempts']+=1;b['input_tokens']+=u['input_tokens'];b['cached_input_tokens']+=u['input_tokens_details']['cached_tokens'];b['cache_write_input_tokens']+=u['input_tokens_details']['cache_write_tokens'];b['output_tokens']+=u['output_tokens']
    r=u.get('output_tokens_details',{}).get('reasoning_tokens')
    if r is None:b['reasoning_detail_missing_attempts']+=1
    else:b['reasoning_output_tokens']+=r
    b['known_usage_priced_usd']+=Decimal(a['usage_priced_usd'])
    assert u['total_tokens']==u['input_tokens']+u['output_tokens']
for b in bins.values():
    rates=PRICING['per_million_tokens']
    cost=(Decimal(b['input_tokens']-b['cached_input_tokens']-b['cache_write_input_tokens'])*Decimal(rates['input'])+Decimal(b['cached_input_tokens'])*Decimal(rates['cached_input'])+Decimal(b['cache_write_input_tokens'])*Decimal(rates['cache_write'])+Decimal(b['output_tokens'])*Decimal(rates['output']))/1000000
    assert cost==b['known_usage_priced_usd'];b['known_usage_priced_usd']=str(cost)
remaining=sum((Decimal(estimate(e['request'])['attempt_reservation_usd'])*(protocol['fallback']['retry_policy']['max_attempts_per_request']-len(e['attempts'])) for e in entries.values() if not e['attempts'] or e['attempts'][-1].get('retryable')),Decimal(0))
assert remaining==0
freeze=read_json(RUN/'predictions_frozen.json')
assert freeze['n_examples']==3080 and freeze['protocol_sha256']==digest(protocol)
for name,h in freeze['files_sha256'].items():assert before[name]==h
verification=read_json(RUN/'verification.json')
checkpoint=read_json(RUN/'evaluation_checkpoint.json')
assert verification==checkpoint['verification']
assert verification['evaluation_sha256']==before['evaluation.json']
assert verification['prediction_freeze_sha256']==before['predictions_frozen.json']
assert checkpoint['evaluation']==read_json(RUN/'evaluation.json')
assert inv()==before
source_meta=[name for name in before if not name.startswith('luna/responses/') and not name.startswith('specialists/')]
result={'study_id':'EXP-007','status':'completed','audited_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':digest(protocol),'case_count':3080,'unique_requests':len(entries),'explicit_alias_count':plan['explicit_alias_count'],'final_status_counts':dict(Counter(p['status'] for p in expected)),'attempt_status_counts':dict(Counter(a['status'] for a in all_attempts)),'http_status_counts':dict(Counter(str(a['response']['http_status']) for a in all_attempts if 'response'in a)),'attempt_count_distribution':dict(Counter(str(len(e['attempts'])) for e in entries.values())),'unresolved_cases':0,'usage_missing_attempt_status_counts':dict(usage_unknown),'unknown_usage_bins':[{'status':status,'reservation_usd':reservation,'count':count} for (status,reservation),count in sorted(Counter((a['status'],a['reservation_usd']) for a in all_attempts if a['usage_priced_usd'] is None).items())],'usage_totals':{'input_tokens':sum(b['input_tokens'] for b in bins.values()),'output_tokens':sum(b['output_tokens'] for b in bins.values()),'cached_tokens':sum(b['cached_input_tokens'] for b in bins.values()),'cache_write_tokens':sum(b['cache_write_input_tokens'] for b in bins.values()),'reasoning_tokens':sum(b['reasoning_output_tokens'] for b in bins.values())},'pricing':PRICING,'usage_bins':list(bins.values()),'accounting':accounting,'reservation':{'approved_cap_usd':manifest['cap_usd'],'preflight_full_retry_reservation_usd':plan['full_retry_reservation_usd'],'remaining_permitted_attempt_reservation_usd':str(remaining)},'resume_compatibility':{'receipt_sha256':compat['receipt_sha256'],'original_successful_responses_preserved':len(receipt['original_attempt_counts']),'original_preserved_files_verified':len(receipt['preserved_files_sha256']),'original_git_head':receipt['original_git_head'],'resumed_code_git_head':compat['code']['git_head'],'current_runtime_matches_receipt':True,'historical_manifests_unchanged':True},'verification':{'all_response_hashes_outcomes_usage_pricing_replayed':True,'preflight_recomputed_from_saved_requests':True,'manifest_bindings_verified':True,'ledger_exactly_matches_all_response_attempts':True,'execution_matches_terminal_response_evidence':True,'frozen_prediction_file_hashes_verified':True,'evaluation_checkpoint_verified':True,'artifact_tree_unchanged':True,'no_raw_dataset_or_model_access':True,'no_api_calls':True,'guards':guards},'provenance':{'run_relative_path':str(RUN.relative_to(ROOT)),'file_count':len(before),'tree_sha256':digest(before),'response_file_count':len(entries),'response_file_inventory_sha256':digest({k:v for k,v in before.items() if k.startswith('luna/responses/')}),'metadata_files_sha256':{name:before[name] for name in source_meta}},'limitations':['Charges are priced from returned usage at the frozen rates, not reconciled against a provider invoice.','One HTTP 503 attempt has no returned token usage; its retained reservation bounds unknown charges under the frozen envelope. Its subsequent retry resolved successfully.','Ledger reservations are safeguards, not additional charges.','No API, inference, raw test-file reading, scoring, protocol changes or artifact writes were performed by this audit.']}
output=Path('/private/tmp/exp007-accounting-recheck.json')
with output.open('x') as handle:
    handle.write(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('status','case_count','attempt_status_counts','http_status_counts','attempt_count_distribution','usage_missing_attempt_status_counts','usage_bins','accounting','reservation','resume_compatibility','verification')},indent=2))
