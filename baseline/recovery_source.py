"""Read-only audit of EXP-006 source artifacts for a separately authorized recovery.

This module never requests inference, locks/writes the source directory, reads
raw dataset files, or resubmits successful responses. The original tree remains
an immutable input; callers must store recovery evidence elsewhere.
"""
from collections import Counter
import json
from pathlib import Path
import re
import stat

from .data import ROOT, read_json, sha256
from .general import accounting, final_result, read_entry, sync_reservation_ledger
from .general_protocol import cache_key, digest, load_protocol, payload
from .selective import require


DEFAULT_SOURCE = ROOT / 'artifacts/exp006-gpt6-luna-validation-v1'
BUNDLE = ROOT / 'experiments/exp006r-429-recovery-preparation'
SOURCE_PROTOCOL_SHA256 = 'd7f824ee761ca90d4dc3a848dd19e3928475c799d6713e03e336dc2a80c9ca68'
REFERENCE = ROOT / 'experiments/exp004-minilm-learning-curve/runs/exp004-minilm-v2-n5-s11.json'
REFERENCE_AUDIT = ROOT / 'experiments/exp004-minilm-learning-curve/analysis_audit.json'


def file_inventory(source):
    """Hash every regular file, including hidden files; reject symlink traversal."""
    source = Path(source).expanduser().absolute()
    require(not source.is_symlink() and source.is_dir(), 'Source must be a real directory, not a symlink')
    result = {}
    for path in sorted(source.rglob('*')):
        require(not path.is_symlink(), 'Source symlinks are prohibited')
        mode = path.stat().st_mode
        if stat.S_ISDIR(mode):
            continue
        require(stat.S_ISREG(mode), 'Source contains a nonregular file')
        result[path.relative_to(source).as_posix()] = sha256(path.read_bytes())
    return result


def select_eligible(execution):
    """Use recorded terminal HTTP status only, never correctness or confidence."""
    require(execution.get('kind') == 'live' and execution.get('status') == 'finished_with_unresolved',
            'Expected the original live run finished with unresolved responses')
    predictions = execution.get('predictions')
    require(isinstance(predictions, list) and len(predictions) == 770, 'Exactly 770 original predictions required')
    require(all(isinstance(row, dict) and isinstance(row.get('id'), str) and row['id'] for row in predictions),
            'Prediction IDs must be nonempty strings')
    ids = [row['id'] for row in predictions]
    require(len(set(ids)) == 770, 'Original prediction IDs must be unique')
    statuses = Counter(row.get('status') for row in predictions)
    require(statuses == {'ok': 718, 'http_429': 52}, 'Expected exactly 718 ok and 52 HTTP 429 responses')
    for row in predictions:
        if row['status'] == 'http_429':
            require(row.get('predicted_label') is None, 'HTTP 429 cannot contain a prediction')
        else:
            require(isinstance(row.get('predicted_label'), str) and row['predicted_label'],
                    'Successful original responses need a class label')
    return [row['id'] for row in predictions if row['status'] == 'http_429']


def _reference(manifest, labels):
    audit = read_json(REFERENCE_AUDIT)
    relative = 'runs/' + REFERENCE.name
    require(sha256(REFERENCE.read_bytes()) == audit['compact_records_sha256'][relative],
            'Versioned EXP-004 reference hash changed')
    reference = read_json(REFERENCE)
    samples, metadata = reference['samples'], reference['metadata']
    ids = samples['validation_ids']
    require(len(ids) == 770 and len(set(ids)) == 770 and samples['labels'] == labels,
            'Original validation IDs or class order changed')
    predictions = reference['predictions']
    require([row['id'] for row in predictions] == ids and
            Counter(row['true_label'] for row in predictions) == {label: 10 for label in labels},
            'Versioned validation label budget changed')
    data = manifest['data']
    require(data['validation_ids_sha256'] == digest(ids), 'Source validation ID provenance changed')
    require(data['protocol_id'] == metadata['protocol_id'] and
            data['dataset_revision'] == metadata['source']['revision'] and
            data['source_train_sha256'] == metadata['source']['files']['train.csv']['sha256'] and
            data['split_manifest_sha256'] == metadata['split_manifest_sha256'],
            'Source dataset or split provenance changed')
    require(data['prior_run_hashes'][REFERENCE.name]['exp004_sha256'] == sha256(REFERENCE.read_bytes()),
            'Source EXP-004 reference provenance changed')
    require(data['official_test_access'] is False and data['new_labels'] == 0 and data['additional_validation_labels'] == 770,
            'Source label budget or test-access provenance changed')
    require(set(data['validation_text_sha256_by_id']) == set(ids), 'Source text-hash IDs changed')
    return ids


def _request_text(body):
    try:
        value = json.loads(body['input'][0]['content'][0]['text'])
    except (KeyError, IndexError, TypeError, ValueError):
        raise ValueError('Original saved request message is malformed') from None
    require(isinstance(value, dict) and set(value) == {'customer_message'} and
            isinstance(value['customer_message'], str) and value['customer_message'].strip(),
            'Original saved request must contain exactly one customer message')
    return value['customer_message']


def load_source(source=DEFAULT_SOURCE):
    """Audit the existing 770 records and return only 52 status-eligible request bodies."""
    source = Path(source).expanduser().absolute()
    before = file_inventory(source)
    for required in ('manifest.json', 'execution.json', 'reservations.json'):
        require(required in before, f'Missing original source artifact: {required}')
    manifest, execution = read_json(source / 'manifest.json'), read_json(source / 'execution.json')
    protocol, prompt, schema = load_protocol()
    require(digest(protocol) == SOURCE_PROTOCOL_SHA256, 'Original frozen protocol hash changed')
    require(manifest['kind'] == 'live' and manifest['protocol_sha256'] == SOURCE_PROTOCOL_SHA256 and
            execution['protocol_sha256'] == SOURCE_PROTOCOL_SHA256 and manifest['protocol'] == protocol,
            'Original execution protocol provenance changed')
    labels = protocol['labels']
    require(len(labels) == 77 and len(set(labels)) == 77 and all(isinstance(label, str) and label for label in labels),
            'Exactly 77 frozen class labels required')
    require(schema['properties']['intent']['enum'] == labels, 'Original schema class order changed')
    eligible_ids = select_eligible(execution)
    validation_ids = _reference(manifest, labels)
    requests = manifest['requests']
    require(len(requests) == 770 and [request['id'] for request in requests] == validation_ids,
            'Source request IDs or original validation order changed')
    require([row['id'] for row in execution['predictions']] == validation_ids,
            'Source prediction order differs from original validation order')
    keys = [request['cache_key'] for request in requests]
    require(all(isinstance(key, str) and re.fullmatch(r'[0-9a-f]{64}', key) for key in keys) and len(set(keys)) == 770,
            'Original request cache keys are invalid or duplicated')
    require({f'responses/{key}.json' for key in keys} == {path for path in before if path.startswith('responses/')},
            'Missing or unexpected original response cache files')
    entries, bodies = {}, {}
    eligible = set(eligible_ids)
    for request, prediction in zip(requests, execution['predictions']):
        row_id, key = request['id'], request['cache_key']
        raw_entry = read_json(source / 'responses' / (key + '.json'))
        body = raw_entry['request']
        text = _request_text(body)
        require(body == payload(text, prompt, schema, protocol['settings']), 'Original request body differs from frozen protocol')
        require(cache_key(body) == key, 'Original request cache key differs from body')
        require(sha256(text.encode()) == manifest['data']['validation_text_sha256_by_id'][row_id],
                'Original request text hash changed')
        entry = read_entry(source, row_id, body, 'live', labels)
        require(final_result(entry) == {key: prediction[key] for key in ('status', 'predicted_label')},
                'Original execution prediction differs from saved response')
        if row_id in eligible:
            require(len(entry['attempts']) == 2 and all(attempt['status'] == 'http_429' and
                    attempt['response']['http_status'] == 429 and attempt.get('predicted_label') is None
                    for attempt in entry['attempts']), 'Recovery requires exactly two original HTTP 429 attempts')
            bodies[row_id] = body
        else:
            require(prediction['predicted_label'] in labels, 'Successful original label is outside the frozen classes')
        entries[row_id] = entry
    sync_reservation_ledger(source, manifest, entries, write=False)
    require(execution['accounting'] == accounting(entries), 'Original execution accounting differs from attempts')
    after = file_inventory(source)
    require(before == after, 'Original artifact tree changed during read-only audit')
    return {'directory': str(source), 'manifest': manifest, 'execution': execution, 'entries': entries,
            'eligible_ids': eligible_ids, 'bodies': bodies, 'labels': labels, 'files_sha256': before,
            'tree_sha256': digest(before), 'source_protocol_sha256': SOURCE_PROTOCOL_SHA256,
            'validation_ids': validation_ids}
