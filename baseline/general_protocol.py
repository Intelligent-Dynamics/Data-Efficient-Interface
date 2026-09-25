"""EXP-006 frozen protocol and offline validation-only input preparation."""
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path

from .data import ROOT, DEFAULT_MANIFEST, PROTOCOL_ID, SEEDS, SHOTS, csv_rows, json_bytes, read_json, sha256, source_spec, text_key
from .selective import restore, require

BUNDLE = ROOT / 'experiments/exp006-general-model-preparation'
MODEL = 'gpt-6-luna'
ENDPOINT = 'https://api.openai.com/v1/responses'
PRICING = {'currency': 'USD', 'as_of': '2026-09-25', 'per_million_tokens': {
    'input': '0.10', 'cached_input': '0.01', 'cache_write': '0.125', 'output': '0.50'},
    'source': 'https://developers.openai.com/api/docs/pricing',
    'cache_source': 'https://developers.openai.com/api/docs/guides/prompt-caching',
    'scope': 'Standard, nonregional, <=272000 input tokens, no tools; excluding taxes/credits'}
SETTINGS = {'model': MODEL, 'reasoning': {'effort': 'none'}, 'max_output_tokens': 128,
    'service_tier': 'default', 'store': False, 'stream': False, 'truncation': 'disabled',
    'prompt_cache_options': {'mode': 'implicit', 'ttl': '30m'}}
RETRIES = {'max_attempts_per_request': 2, 'retry_http_statuses': [408, 429, 500, 502, 503, 504],
           'backoff_seconds': 2, 'timeout_seconds': 60,
           'terminal': 'refusal/incomplete/invalid output/model or tier mismatch; never retry correctness'}


def digest(value):
    return sha256(json_bytes(value))


def load_protocol(bundle=BUNDLE):
    protocol = read_json(bundle / 'protocol.json')
    prompt = (bundle / 'prompt.txt').read_text()
    schema = read_json(bundle / 'schema.json')
    require(sha256(prompt.encode()) == protocol['prompt_sha256'], 'Frozen prompt changed')
    require(digest(schema) == protocol['schema_sha256'], 'Frozen schema changed')
    require(protocol['settings'] == SETTINGS and protocol['pricing'] == PRICING and
            protocol['retries'] == RETRIES and protocol['endpoint'] == ENDPOINT, 'Frozen configuration changed')
    return protocol, prompt, schema


def payload(text, prompt, schema, settings=SETTINGS):
    """Only message text crosses the request boundary; no row truth/specialist fields."""
    require(isinstance(text, str) and text.strip(), 'Nonempty request text required')
    return {**settings, 'instructions': prompt,
            'input': [{'role': 'user', 'content': [{'type': 'input_text',
                       'text': json.dumps({'customer_message': text}, ensure_ascii=False)}]}],
            'text': {'format': {'type': 'json_schema', 'name': 'banking77_intent',
                               'strict': True, 'schema': schema}}}


def cache_key(body):
    return digest({'endpoint': ENDPOINT, 'body': body})


def estimate(body):
    # Deliberately loose engineering envelope, NOT a verified model-token count.
    size = len(json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode())
    nominal = (size + 3) // 4
    upper = 2 * size + 8192
    require(upper <= 272000, 'Input estimate exceeds frozen short-context pricing scope')
    rates = PRICING['per_million_tokens']
    def amount(i, o, rate):
        return str((Decimal(i) * Decimal(rate) + Decimal(o) * Decimal(rates['output'])) / 1000000)
    return {'serialized_request_bytes': size, 'nominal_input_tokens': nominal,
            'nominal_output_tokens': 32, 'input_token_reservation': upper,
            'output_token_reservation': body['max_output_tokens'],
            'nominal_usd_no_cache_read_discount': amount(nominal, 32, rates['cache_write']),
            'attempt_reservation_usd': amount(upper, body['max_output_tokens'], rates['cache_write'])}


def resolve_validation(source_rows, ids, truth, sealed_test):
    require(len(ids) == len(set(ids)) and set(ids) == set(truth), 'Validation ID alignment mismatch')
    by_id = {}
    for row in source_rows:
        if row['id'] in truth:
            require(row['id'] not in by_id, 'Duplicate source ID')
            require(row['label'] == truth[row['id']], 'Validation label mismatch')
            by_id[row['id']] = row
    require(set(by_id) == set(ids), 'Missing validation text')
    rows = [by_id[i] for i in ids]
    keys = [text_key(r['text']) for r in rows]
    require(len(set(keys)) == len(keys), 'Duplicate validation text')
    require(not set(keys) & {r['key'] for r in sealed_test}, 'Stored test hash overlap')
    require(not set(ids) & {r['id'] for r in sealed_test}, 'Stored test ID overlap')
    return rows


def load_inputs():
    """Read pinned TRAIN CSV + existing IDs only; never open test.csv or make splits."""
    study4 = ROOT / 'experiments/exp004-minilm-learning-curve'
    study5 = ROOT / 'experiments/exp005-selective-diagnostic'
    audit4, audit5 = read_json(study4 / 'analysis_audit.json'), read_json(study5 / 'metadata.json')
    manifest_path = ROOT / DEFAULT_MANIFEST
    manifest = read_json(manifest_path)
    require(manifest['protocol_id'] == PROTOCOL_ID, 'Split protocol mismatch')
    source = source_spec()
    require(source == manifest['source'], 'Source version mismatch')
    raw = ROOT / 'data/raw/banking77'
    for name in ('train.csv', 'categories.json'):
        require(sha256((raw / name).read_bytes()) == source['files'][name]['sha256'], 'Pinned training/category file changed')
    labels = read_json(raw / 'categories.json')
    require(labels == manifest['labels'] and len(labels) == 77, 'Official intent mapping changed')
    ids = manifest['validation_ids']
    truth = None
    runs, hashes = [], {}
    for n in SHOTS:
        for seed in SEEDS:
            name = f'exp004-minilm-v2-n{n}-s{seed}.json'
            p4, p5 = study4 / 'runs' / name, study5 / 'runs' / name
            h4, h5 = sha256(p4.read_bytes()), sha256(p5.read_bytes())
            require(h4 == audit4['compact_records_sha256']['runs/' + name], 'EXP-004 evidence changed')
            require(h5 == audit5['analysis_artifact_sha256']['runs/' + name], 'EXP-005 evidence changed')
            r4, r5 = read_json(p4), restore(read_json(p5))
            require(r5['shots'] == n and r5['seed'] == seed and r5['provenance']['exp004_record_sha256'] == h4, 'Specialist identity mismatch')
            samples = r4['samples']
            require(samples['validation_ids'] == ids and samples['labels'] == labels, 'Original validation IDs changed')
            require(r4['metadata']['split_manifest_sha256'] == sha256(manifest_path.read_bytes()), 'Original split hash changed')
            require(len(samples['train_ids']) == n*77 and set(samples['train_ids']).isdisjoint(ids), 'Specialist sample leakage')
            truth4 = {r['id']: r['true_label'] for r in r4['predictions']}
            rows5 = {r['id']: (r['true_label'], r['predicted_label']) for r in r5['ranked_rows']}
            require(rows5 == {r['id']: (r['true_label'], r['predicted_label']) for r in r4['predictions']}, 'Specialist labels/predictions changed')
            if truth is None:
                truth = truth4
            require(truth == truth4 and set(truth) == set(ids), 'Shared validation truth mismatch')
            runs.append(r5)
            hashes[name] = {'exp004_sha256': h4, 'exp005_sha256': h5}
    require(len(ids) == 770 and Counter(truth.values()) == {label: 10 for label in labels}, 'Validation budget changed')
    # Parse row positions mechanically, retaining text and category only for chosen validation IDs.
    source_rows = ({'id': f'train:{i:05d}', 'text': r['text'], 'label': r['category']}
                   for i, r in csv_rows(raw / 'train.csv') if f'train:{i:05d}' in truth)
    rows = resolve_validation(source_rows, ids, truth, manifest['sealed_test'])
    provenance = {'dataset_revision': source['revision'], 'protocol_id': PROTOCOL_ID,
        'split_manifest_sha256': sha256(manifest_path.read_bytes()), 'source_train_sha256': source['files']['train.csv']['sha256'],
        'validation_ids_sha256': digest(ids), 'validation_text_sha256_by_id': {r['id']: sha256(r['text'].encode()) for r in rows},
        'prior_run_hashes': hashes, 'official_test_access': False, 'new_labels': 0,
        'additional_validation_labels': 770, 'training_source_access': 'byte checksum and CSV positions; retain only 770 validation texts/labels'}
    return rows, labels, runs, provenance


def plan(rows, protocol, prompt, schema, provenance):
    require(schema['properties']['intent']['enum'] == protocol['labels'], 'Schema label mismatch')
    require(len({r['id'] for r in rows}) == len(rows), 'Duplicate request IDs')
    requests = []
    for row in rows:
        body = payload(row['text'], prompt, schema, protocol['settings'])
        requests.append({'id': row['id'], 'cache_key': cache_key(body), **estimate(body)})
    require(len({r['cache_key'] for r in requests}) == len(requests), 'Duplicate request inputs')
    retries = protocol['retries']['max_attempts_per_request']
    nominal = sum((Decimal(r['nominal_usd_no_cache_read_discount']) for r in requests), Decimal(0))
    first = sum((Decimal(r['attempt_reservation_usd']) for r in requests), Decimal(0))
    return {'study_id': 'EXP-006', 'status': 'PAID EXPERIMENT NOT RUN', 'inference_calls': 0,
        'actual_api_spend_usd': '0', 'model': MODEL, 'protocol_sha256': digest(protocol),
        'prompt': prompt, 'prompt_sha256': sha256(prompt.encode()), 'schema': schema,
        'settings': protocol['settings'], 'pricing': protocol['pricing'], 'planned_unique_requests': len(rows),
        'maximum_attempts_including_retries': len(rows)*retries, 'requests': requests,
        'cost_estimates_usd': {'nominal_single_attempt': str(nominal), 'single_attempt_reservation': str(first),
            'all_attempts_reservation': str(first*retries)},
        'token_count_method': 'Nominal ceil(serialized UTF-8 bytes/4), 32 output; reservation 2*bytes+8192 input and 128 output. Heuristics, not server token counts.',
        'budget_limits': 'Conditional on frozen prices and token envelope; halt on observed breaches. No cache discounts assumed; all input charged at cache-write rate. No invoice/tax guarantee.',
        'environment_variables': {'required_for_live': ['OPENAI_API_KEY'], 'optional': ['OPENAI_PROJECT_ID']},
        'data': provenance}
