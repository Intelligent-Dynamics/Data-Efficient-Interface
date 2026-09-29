"""Synthetic cost partition tests; no private run, labels, model or API access."""
import importlib.util
import json
from decimal import Decimal
from pathlib import Path
import pytest

BUNDLE = Path(__file__).resolve().parents[1]/'experiments/exp009-routed-api-spend'
spec = importlib.util.spec_from_file_location('exp009_routed_cost',BUNDLE/'analyze.py')
cost = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cost)


def attempt(number=1, known=True, status='ok'):
    body = {'model':'gpt-6-luna','service_tier':'default',
            'usage':{'input_tokens':1000,'output_tokens':100,
                     'input_tokens_details':{'cached_tokens':200,'cache_write_tokens':300},
                     'output_tokens_details':{'reasoning_tokens':0}}}
    # 500*0.10 + 200*0.01 + 300*0.125 + 100*0.50, per million.
    return {'attempt':number,'status':status,'reservation_usd':'.003',
            'usage_priced_usd':'.0001395' if known else None,
            **({'response':{'body':body}} if known else {})}


def test_raw_usage_reprices_success_and_retry_without_double_counting():
    a=attempt(status='http_429');a['usage_priced_usd']='0.0001395'
    b=attempt(2);b['usage_priced_usd']='0.0001395'
    result=cost.summarize({'a':{'attempts':[a,b]}})
    assert Decimal(result['known_charges_usd'])==Decimal('.0002790')
    assert Decimal(result['priced_unsuccessful_attempt_charges_usd'])==Decimal('.0001395')
    assert Decimal(result['priced_successful_attempt_charges_usd'])==Decimal('.0001395')
    assert Decimal(result['priced_retry_attempt_charges_usd_included_above'])==Decimal('.0001395')
    assert result['returned_usage_totals']['cache_write_tokens']==600


def test_unknown_usage_is_reserved_not_free():
    result=cost.summarize({'a':{'attempts':[attempt(known=False,status='transport_unknown')]}})
    assert result['unknown_charge_attempts']==1
    assert list(map(Decimal,result['spend_interval_under_frozen_assumptions_usd']))==[Decimal(0),Decimal('.003')]


def test_partition_uses_saved_boolean_ids_instead_of_population_ratio():
    a=attempt();a['usage_priced_usd']='0.0001395'
    result=cost.partition({'short':{'attempts':[a]},'unknown':{'attempts':[attempt(known=False,status='transport_unknown')]}},
                          [{'id':'unknown','use_specialist':True},{'id':'short','use_specialist':False}])
    assert Decimal(result['routed_luna']['known_charges_usd'])==Decimal('.0001395')
    assert result['routed_luna']['unknown_charge_attempts']==0
    assert result['not_routed']['unknown_charge_attempts']==1


@pytest.mark.parametrize('rows',[[{'id':'a','use_specialist':False},{'id':'a','use_specialist':True}],
                                 [{'id':'b','use_specialist':False}],
                                 [{'id':'a','use_specialist':0}],[]])
def test_partition_rejects_duplicate_missing_or_nonboolean_ids(rows):
    with pytest.raises(ValueError):cost.partition({'a':{'attempts':[]}},rows)


def test_raw_usage_disagreement_cannot_be_hidden_by_saved_price():
    a=attempt();a['usage_priced_usd']='0'
    with pytest.raises(ValueError,match='Raw usage'):cost.summarize({'a':{'attempts':[a]}})


def test_unknown_shared_between_totals_cancels_in_absolute_reduction():
    total={'known_charges_usd':'10','unknown_charge_reservation_usd':'5'}
    routed={'known_charges_usd':'2','unknown_charge_reservation_usd':'2'}
    r=cost.reduction(total,routed)
    assert r['absolute_reduction_interval_usd']==['8','11']
    assert list(map(Decimal,r['spend_reduction_percent_interval']))==[Decimal(800)/12,Decimal(1100)/13]


@pytest.mark.parametrize('r,ur',[('11','0'),('1','6'),('-1','0')])
def test_invalid_nested_accounting_rejected(r,ur):
    with pytest.raises(ValueError):cost.reduction({'known_charges_usd':'10','unknown_charge_reservation_usd':'5'},
                                                {'known_charges_usd':r,'unknown_charge_reservation_usd':ur})


def test_compact_report_reproduces_reduction_and_retains_invoice_limit():
    r=json.loads((BUNDLE/'summary.json').read_text())
    assert cost.reduction(r['all_luna'],r['routed_luna'])==r['reduction']
    assert r['routed_luna']['request_count']==268 and r['routed_luna']['attempt_count']==268
    assert r['routed_luna']['unknown_charge_attempts']==0
    assert r['all_luna']['unknown_charge_attempts']==3
    assert Decimal(r['routed_luna']['known_charges_usd'])==Decimal('.051380650')
    assert r['actual_invoice_spend_usd'] is None and not r['invoice_reconciled']
    forbidden={'id','request','response','body','predicted_label','true_label','request_id','api_key'}
    def inspect(v):
        if isinstance(v,dict):
            assert not set(v)&forbidden
            for x in v.values():inspect(x)
        elif isinstance(v,list):
            for x in v:inspect(x)
    inspect(r)
