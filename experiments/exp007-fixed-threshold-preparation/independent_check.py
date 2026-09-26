"""Standard-library replay of EXP-007 validation thresholds; never opens test data."""
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256((json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()).hexdigest()


def verify():
    protocol = json.loads((BUNDLE/'protocol.json').read_text())
    assert digest(protocol) == (BUNDLE/'protocol_sha256.txt').read_text().strip()
    manifest = json.loads((BUNDLE/'manifest.json').read_text())
    assert manifest['protocol_sha256'] == digest(protocol)
    for name, expected in manifest['code_files_sha256'].items():
        assert name.startswith('baseline/') or name in ('pyproject.toml', 'uv.lock')
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected
    for name, expected in protocol['input_files_sha256'].items():
        assert name.startswith(('experiments/exp004-', 'experiments/exp005-', 'artifacts/exp004-')) or name == 'data/banking77-source.json'
        assert '..' not in Path(name).parts
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected
    results = []
    summaries = json.loads((BUNDLE/'thresholds.json').read_text())
    assert [s['seed'] for s in summaries] == [11,22,33,44,55]
    for s in summaries:
        p = ROOT/'experiments/exp005-selective-diagnostic/runs'/f"exp004-minilm-v2-n20-s{s['seed']}.json"
        rows = json.loads(p.read_text())['ranked_rows']
        ordered = sorted(rows, key=lambda r: (-r['confidence'], hashlib.sha256(r['id'].encode()).hexdigest(), r['id']))
        assert rows == ordered and len(rows) == 770
        wanted = {r['id'] for r in rows[:693]}
        q = rows[692]['confidence']
        includes = {r['id'] for r in rows if r['confidence'] >= q}
        excludes = {r['id'] for r in rows if r['confidence'] > q}
        expected = q if len(includes ^ wanted) <= len(excludes ^ wanted) else math.nextafter(q, math.inf)
        t = s['threshold']
        assert t == expected == float(s['threshold_decimal']) == float.fromhex(s['threshold_hex'])
        fixed = {r['id'] for r in rows if r['confidence'] >= t}
        assert len(fixed) == s['accepted_count'] and len(fixed)/770 == s['actual_coverage']
        assert len(fixed ^ wanted) == s['minimum_symmetric_difference']
        assert sorted(fixed-wanted) == s['added_ids'] and sorted(wanted-fixed) == s['removed_ids']
        assert digest(sorted(wanted)) == s['original_accepted_ids_sha256']
        assert digest(sorted(fixed)) == s['fixed_accepted_ids_sha256']
        assert protocol['thresholds'][str(s['seed'])]['value'] == t
        results.append({'seed':s['seed'],'threshold':t,'accepted':len(fixed),'difference':len(fixed ^ wanted)})
    usage = json.loads((BUNDLE/'validation_completion.json').read_text())
    costs = json.loads((BUNDLE/'cost_estimate.json').read_text())
    assert digest(costs) == protocol['accounting']['cost_estimate_sha256']
    counts = usage['successful_response_usage']
    rates = {k:Decimal(v) for k,v in costs['pricing']['per_million_tokens'].items()}
    i,o,c,w = [counts[k] for k in ('input_tokens','output_tokens','cached_tokens','cache_write_tokens')]
    nominal = ((i-c-w)*rates['input']+c*rates['cached_input']+w*rates['cache_write']+o*rates['output']) / 1000000 * 4
    bound = ((2*usage['largest_validation_payload_bytes']+8192)*rates['cache_write']+128*rates['output']) / 1000000 * 3080 * 4
    assert nominal == Decimal(costs['nominal_usage_projection_usd'])
    assert bound == Decimal(costs['conservative_all_attempts_usd'])
    return {'status':'passed','thresholds':results,'protocol_sha256':digest(protocol),
            'costs_recomputed':True,'official_test_access':False,'api_calls':0}


if __name__ == '__main__':
    print(json.dumps(verify(),indent=2))
