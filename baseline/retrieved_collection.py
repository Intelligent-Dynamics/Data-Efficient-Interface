"""Small EXP-009 adapter over the project's durable transport/accounting primitives."""
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import time
import math
import os
from urllib.error import URLError

from . import general, recovery
from .final_collection import _wait_for_cooldown
from .retrieved_compatibility import accepted_manifest, record_resume_source
from .data import ROOT, read_json, sha256
from .general_protocol import SETTINGS, PRICING, ENDPOINT, cache_key, digest, estimate
from .selective import require


@dataclass(frozen=True)
class Approval:
    protocol_sha256: str = ''
    authorize_live: bool = False
    cap_usd: str = ''
    compatibility_date: str = ''
    authorize_test_access: bool = False
    resume_compatibility_sha256: str = ''

    def validate(self, protocol, *, live, test=False):
        require(self.protocol_sha256 == digest(protocol), 'Separate EXP-009 protocol approval required')
        if test:
            require(self.authorize_test_access is True, 'Separate explicit test access required')
        if not live:
            return None
        require(self.authorize_live is True, 'Explicit live/API approval required')
        require(not isinstance(self.cap_usd,bool), 'Positive numeric cap required')
        cap = general.spending_cap(self.cap_usd)
        require(self.compatibility_date == datetime.now(timezone.utc).date().isoformat(),
                'Verify current official model/settings compatibility and unchanged prices today')
        return cap


def validate_requests(requests, protocol):
    require(protocol['settings'] == SETTINGS and protocol['pricing'] == PRICING and
            protocol['retry_policy'] == recovery.POLICY and protocol['endpoint'] == ENDPOINT, 'Frozen settings drift')
    require(requests and len(requests) == len(set(requests)), 'Nonempty unique request IDs required')
    for body in requests.values():
        require(all(body.get(k) == v for k,v in SETTINGS.items()), 'Request generation settings changed')
        require(sha256(body['instructions'].encode()) == protocol['prompt_sha256'], 'Static prompt changed')
        require(digest(body['text']['format']['schema']) == protocol['schema_sha256'], 'Schema changed')
        require(body['text']['format'] == {'type':'json_schema','name':'banking77_intent','strict':True,
                                          'schema':protocol['schema']}, 'Output format changed')
    keys = [cache_key(b) for b in requests.values()]
    # Each ID has independent evidence; no silent merging or dropped query.
    return keys


def _entry(output, rid, body, protocol, kind):
    key = cache_key(body)
    identity = {'study_id':'EXP-009','kind':kind,'id':rid,'cache_key':key,'request':body,
                'protocol_sha256':digest(protocol)}
    path = output / 'responses' / (rid.replace(':','_')+'.json')
    entry = read_json(path) if path.exists() else {**identity,'attempts':[]}
    require(all(entry.get(k)==v for k,v in identity.items()), 'Response cache identity drift')
    attempts = entry['attempts']
    require(len(attempts) <= recovery.POLICY['max_attempts_per_request'], 'Retry budget exceeded')
    bound = estimate(body)
    for n,a in enumerate(attempts,1):
        require(a['attempt']==n and a['reservation_usd']==bound['attempt_reservation_usd'], 'Attempt reservation drift')
        if n>1:
            require(attempts[n-2].get('retryable') and not attempts[n-2].get('halt'), 'Successful/terminal request resent')
        if 'response' in a:
            require(digest(a['response'])==a['response_sha256'], 'Response changed')
            expected = recovery.outcome(a['response'],protocol['labels'])
            cost = general.priced_response(a['response']['body'])
            require(a['usage_priced_usd']==(str(cost) if cost is not None else None), 'Usage price drift')
            data = a['response']['body']
            if isinstance(data,dict):
                require(a.get('usage')==data.get('usage') and a.get('returned_model')==data.get('model') and
                        a.get('returned_service_tier')==data.get('service_tier'), 'Returned usage/model metadata drift')
            if isinstance(data,dict) and general.usage_breaches(data.get('usage'),bound,cost):
                expected['halt']=True
            require(all(a.get(k)==v for k,v in expected.items()), 'Response outcome drift')
            require(a['retry_after_seconds']==recovery.retry_after_seconds(a['response'].get('headers'),a['finished_epoch']),
                    'Retry-After drift')
        else:
            require(a['status'] in ('reserved','interrupted_unknown','transport_unknown','client_error') and
                    a.get('predicted_label') is None and a.get('usage_priced_usd') is None, 'Missing response')
            if a['status']=='client_error':
                require(a.get('halt') is True and a.get('retryable') is False, 'Fatal client error changed')
            elif a['status']!='reserved':
                require(a.get('retryable') is True and not a.get('halt'), 'Transport retry policy changed')
        require(type(a['started_epoch']) in (int,float) and math.isfinite(a['started_epoch']), 'Invalid attempt time')
        if 'finished_epoch' in a:
            require(math.isfinite(a['finished_epoch']) and a['finished_epoch']>=a['started_epoch'], 'Invalid completion time')
    return path,entry


def source_record():
    return {'code_sha256':{p.name:sha256(p.read_bytes()) for p in sorted((ROOT/'baseline').glob('retrieved*.py'))},
            'reused_code':general.code_record(),
            'reused_helper_sha256':{name:sha256((ROOT/'baseline'/name).read_bytes()) for name in
                ('recovery.py','final_collection.py','final_protocol.py','final_test.py','final_specialists.py','cpu_benchmark.py',
                 'embeddings.py','data.py','selective.py','thresholds.py')}}


def collect(requests, protocol, output, approval, scope, *, transport=None,
            sleep=time.sleep, clock=time.time, monotonic=time.monotonic):
    require(scope in ('pilot','validation','test'), 'Unknown collection scope')
    require(isinstance(approval,Approval), 'Explicit approval object required')
    cap = approval.validate(protocol,live=True,test=scope=='test')
    validate_requests(requests,protocol)
    expected = protocol['pilot_ids'] if scope=='pilot' else protocol['validation_ids'] if scope=='validation' else [f'test:{i:05d}' for i in range(3080)]
    require(list(requests)==expected, 'Exactly frozen scope IDs required; no dropping/reordering')
    if scope!='test' and 'validation_request_sha256' in protocol:
        require(all(cache_key(b)==protocol['validation_request_sha256'][i] for i,b in requests.items()),
                'Frozen demonstration/query requests changed')
    reserve = sum((Decimal(estimate(b)['attempt_reservation_usd'])*4 for b in requests.values()),Decimal(0))
    require(reserve<=cap, f'Insufficient cap; full-run reservation USD {reserve}')
    output = Path(output)
    require(not any(p.is_symlink() for p in [output,*output.parents]), 'Symlink output forbidden')
    if output.exists():
        require(not any(p.is_symlink() for p in output.rglob('*')), 'Symlink cache forbidden')
    live = transport is None
    if live:
        require(bool(os.environ.get('OPENAI_API_KEY')), 'Set OPENAI_API_KEY locally before live execution')
        receipt=read_json(output.parent/'prepared.json')
        require(receipt['protocol_sha256']==digest(protocol) and receipt['scope']==scope and
                receipt['requests_sha256']==digest(requests), 'Complete exact request preparation required before API calls')
        require(output.resolve()==(ROOT/protocol['outputs'][scope]/'collection').resolve(), 'Use single frozen live directory')
        transport = recovery.post_recovery
    else:
        require(transport not in (recovery.post_recovery,general.post_openai), 'Real transport cannot masquerade as synthetic')
    kind = 'live' if live else 'synthetic'
    manifest = {'study_id':'EXP-009','kind':kind,'scope':scope,'protocol_sha256':digest(protocol),
                'requests_sha256':digest(requests),'cap_usd':str(cap),'full_retry_reservation_usd':str(reserve),
                **source_record()}
    if approval.resume_compatibility_sha256:
        accepted_manifest(output/'manifest.json',manifest,approval,root=ROOT)
    with general.lock(output):
        mp=output/'manifest.json'
        if mp.exists():
            manifest=accepted_manifest(mp,manifest,approval,root=ROOT)
            require((output/'reservations.json').exists(),'Reservation ledger missing')
        else:
            require(not any(p.name!='.lock' for p in output.iterdir()),'New collection directory must be empty')
            general.atomic_json(mp,manifest)
        entries={i:_entry(output,i,b,protocol,kind)[1] for i,b in requests.items()}
        general.sync_reservation_ledger(output,manifest,entries,write=False)
        record_resume_source(output,approval,source_record(),root=ROOT)
        general.sync_reservation_ledger(output,manifest,entries)
        halted='previous_fatal_attempt' if any(a.get('halt') for e in entries.values() for a in e['attempts']) else None
        for rid,body in requests.items():
            if halted:
                break
            path,entry=_entry(output,rid,body,protocol,kind)
            entries[rid]=entry
            if entry['attempts'] and entry['attempts'][-1]['status']=='reserved':
                entry['attempts'][-1].update(status='interrupted_unknown',predicted_label=None,retryable=True)
                general.atomic_json(path,entry)
            while len(entry['attempts'])<4 and (not entry['attempts'] or entry['attempts'][-1].get('retryable')):
                delay=_wait_for_cooldown(entries,rid,recovery.POLICY['maximum_single_wait_seconds'],
                                         sleep=sleep,clock=clock,monotonic=monotonic)
                if delay is None:
                    halted='cooldown_pause'; break
                approval.validate(protocol,live=True,test=scope=='test')
                bound=estimate(body)
                require(Decimal(general.accounting(entries)['committed_reservation_usd'])+Decimal(bound['attempt_reservation_usd'])<=cap,
                        'Spending cap reached')
                a={'attempt':len(entry['attempts'])+1,'status':'reserved','started_epoch':clock(),
                   'started_utc':general.now(),'reservation_usd':bound['attempt_reservation_usd'],'usage_priced_usd':None}
                entry['attempts'].append(a)
                general.atomic_json(path,entry)
                general.sync_reservation_ledger(output,manifest,entries)
                start=time.perf_counter()
                try:
                    response=transport(body,recovery.POLICY['timeout_seconds'])
                except (TimeoutError,URLError,OSError,ConnectionError):
                    a.update(status='transport_unknown',predicted_label=None,retryable=True)
                except ValueError:
                    a.update(status='client_error',predicted_label=None,retryable=False,halt=True)
                else:
                    a.update(response=response,response_sha256=digest(response),**recovery.outcome(response,protocol['labels']))
                    data=response['body']
                    if isinstance(data,dict):
                        a.update(usage=data.get('usage'),returned_model=data.get('model'),returned_service_tier=data.get('service_tier'))
                        cost=general.priced_response(data)
                        a['usage_priced_usd']=str(cost) if cost is not None else None
                        if general.usage_breaches(a['usage'],bound,cost):
                            a.update(halt=True,budget_envelope_breached_or_usage_malformed=True)
                a.update(finished_epoch=clock(),finished_utc=general.now(),request_seconds=time.perf_counter()-start)
                a['retry_after_seconds']=recovery.retry_after_seconds(a.get('response',{}).get('headers'),a['finished_epoch'])
                general.atomic_json(path,entry)
                if a.get('halt'):
                    halted='fatal_attempt';break
        result={'study_id':'EXP-009','kind':kind,'protocol_sha256':digest(protocol),'scope':scope,
                'status':'halted' if halted else 'finished','halt_reason':halted,
                'predictions':[{'id':i,**general.final_result(entries[i])} for i in requests],
                'accounting':general.accounting(entries),'finished_utc':general.now()}
        if not live:
            result['notice']='SYNTHETIC MOCK; NO BANKING77 QUALITY RESULT OR API SPEND'
            result['accounting']['actual_api_spend_usd']='0'
        general.atomic_json(output/'execution.json',result)
        return result


def verify_completed(requests, protocol, output, approval=None):
    """Offline replay never resumes or allocates a paid request."""
    output=Path(output)
    validate_requests(requests,protocol)
    require(not any(p.is_symlink() for p in [output,*output.parents]) and
            not any(p.is_symlink() for p in output.rglob('*')), 'Symlink evidence forbidden')
    manifest=read_json(output/'manifest.json')
    scope=manifest['scope']
    if getattr(approval,'resume_compatibility_sha256','') or (output/'source_compatibility.json').exists():
        expected={**manifest,**source_record()}
        accepted_manifest(output/'manifest.json',expected,approval,root=ROOT)
        record_resume_source(output,approval,source_record(),root=ROOT,write=False)
    require(scope in ('pilot','validation','test') and manifest['kind'] in ('live','synthetic'), 'Invalid saved scope/kind')
    expected=protocol['pilot_ids'] if scope=='pilot' else protocol['validation_ids'] if scope=='validation' else [f'test:{i:05d}' for i in range(3080)]
    require(list(requests)==expected, 'Completed scope IDs changed')
    require(manifest['protocol_sha256']==digest(protocol) and manifest['requests_sha256']==digest(requests),'Replay identity drift')
    entries={i:_entry(output,i,b,protocol,manifest['kind'])[1] for i,b in requests.items()}
    general.sync_reservation_ledger(output,manifest,entries,write=False)
    require(all(e['attempts'] and e['attempts'][-1]['status']!='reserved' and not any(a.get('halt') for a in e['attempts']) and
                (not e['attempts'][-1].get('retryable') or len(e['attempts'])==4) for e in entries.values()),'Nonterminal collection')
    report=read_json(output/'execution.json')
    require(report['study_id']=='EXP-009' and report['kind']==manifest['kind'] and report['scope']==scope and
            report['protocol_sha256']==digest(protocol) and report['status']=='finished' and report['predictions']==[{'id':i,**general.final_result(entries[i])} for i in requests],
            'Predictions differ from original cached responses')
    expected=general.accounting(entries)
    if manifest['kind']=='synthetic': expected['actual_api_spend_usd']='0'
    require(report['accounting']==expected,'Accounting drift')
    return report
