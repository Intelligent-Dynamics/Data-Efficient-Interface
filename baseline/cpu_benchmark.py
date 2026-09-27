"""EXP-008: frozen seed-11 CPU timing, training-source validation rows only.

No fitting, network, test loader, cached embeddings, or prediction reuse in timers.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from .data import ROOT, csv_rows, json_bytes, read_json, sha256, write_json
from .embeddings import ENCODING, assert_frozen, freeze_encoder, state_digest
from .final_protocol import FROZEN_PROTOCOL_SHA256, load_frozen
from .final_specialists import (_classifier_state, _load_classifier,
                                _predictions, _validate_classifier, verify_specialist_files)

BUNDLE = Path('experiments/exp008-cpu-validation')
REFERENCE = Path('experiments/exp004-minilm-learning-curve/runs/exp004-minilm-v2-n20-s11.json')
PROBABILITIES = Path('artifacts/exp004-minilm-learning-curve/runs/exp004-minilm-v2-n20-s11/probabilities.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def offline_guard(root=ROOT):
    """Install before reading inputs; deny official-test and network operations."""
    root = Path(root).resolve()
    counts = {'official_test_access_attempts': 0, 'network_access_attempts': 0}

    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path.name == 'test.csv' or root / 'artifacts/exp007-fixed-threshold-test-v1' in path.parents:
                counts['official_test_access_attempts'] += 1
                raise PermissionError('EXP-008 prohibits official-test access')
        if event in ('socket.connect', 'socket.connect_ex', 'socket.getaddrinfo', 'socket.sendto',
                     'http.client.connect', 'urllib.Request'):
            counts['network_access_attempts'] += 1
            raise PermissionError('EXP-008 prohibits network access')
    sys.addaudithook(audit)
    return counts


def validation_inputs(protocol, root=ROOT):
    """Resolve exact existing validation IDs from train.csv; never use split loaders."""
    root = Path(root)
    record = read_json(root / REFERENCE)
    require(sha256((root / REFERENCE).read_bytes()) == protocol['input_files_sha256'][str(REFERENCE)],
            'Validation reference changed')
    ids = record['samples']['validation_ids']
    labels = record['samples']['labels']
    require(len(ids) == len(set(ids)) == 770 and all(x.startswith('train:') for x in ids),
            'Expected exact 770 training-source validation IDs')
    require(set(ids).isdisjoint(record['samples']['train_ids']), 'Validation/training overlap')
    path = root / 'data/raw/banking77/train.csv'
    expected = record['metadata']['source']['files']['train.csv']['sha256']
    require(sha256(path.read_bytes()) == expected, 'Pinned training CSV changed')
    wanted = set(ids)
    by_id = {f'train:{index:05d}': row for index, row in csv_rows(path)
             if f'train:{index:05d}' in wanted}
    require(set(by_id) == wanted, 'Missing validation rows')
    truth = {p['id']: p['true_label'] for p in record['predictions']}
    require(all(by_id[row_id]['category'] == truth[row_id] for row_id in ids), 'Validation labels shifted')
    require(Counter(truth.values()) == Counter({label: 10 for label in labels}), 'Validation class counts changed')
    rows = [{'id': row_id, 'text': by_id[row_id]['text']} for row_id in ids]
    require(sha256((root / PROBABILITIES).read_bytes()) == protocol['input_files_sha256'][str(PROBABILITIES)],
            'Saved validation probabilities changed')
    probabilities = read_json(root / PROBABILITIES)
    require(probabilities['validation_ids'] == ids and probabilities['labels'] == labels,
            'Saved validation probability alignment changed')
    return rows, labels, np.asarray(probabilities['probabilities'], dtype=np.float64), expected


def load_benchmark_encoder(snapshot):
    """Same frozen encoder settings; empty named prompts are inert in ST 5.1.1."""
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(
        str(snapshot), device='cpu', backend='torch', local_files_only=True,
        trust_remote_code=False, prompts={}, default_prompt_name=None,
        model_kwargs={'use_safetensors': True, 'attn_implementation': 'eager'})
    encoder.float()
    require(encoder.max_seq_length == ENCODING['max_seq_length'] and
            encoder.get_sentence_embedding_dimension() == ENCODING['dimensions'],
            'Pinned encoder architecture/settings mismatch')
    require(encoder.default_prompt_name is None and all(value == '' for value in encoder.prompts.values()),
            'Encoder must not apply prompt text')
    freeze_encoder(encoder)
    return encoder


def infer_batch(encoder, classifier, texts, labels, column_order, threshold, batch_size):
    """Entire timed path: raw text -> tokenizer/encoder/normalization -> LR -> gate."""
    with torch.inference_mode():
        values = encoder.encode(texts, batch_size=batch_size, precision='float32', device='cpu',
                                normalize_embeddings=True, convert_to_numpy=True,
                                show_progress_bar=False, prompt=None, prompt_name=None)
        probabilities = classifier.predict_proba(values)[:, column_order]
        # Classifier argmax tie behavior is alphabetical, as in the frozen runner.
        order = np.asarray(sorted(range(len(labels)), key=lambda index: labels[index]))
        best = order[probabilities[:, order].argmax(axis=1)]
        predictions = [labels[int(index)] for index in best]
        confidences = probabilities[np.arange(len(texts)), best]
        gates = confidences >= threshold
    return probabilities, predictions, confidences, gates


def timed_pass(encoder, classifier, rows, labels, order, threshold, batch_size, clock=time.perf_counter_ns):
    durations, values, predictions, confidences, gates = [], [], [], [], []
    for start in range(0, len(rows), batch_size):
        texts = [row['text'] for row in rows[start:start + batch_size]]
        begin = clock()
        probability, predicted, confidence, gate = infer_batch(
            encoder, classifier, texts, labels, order, threshold, batch_size)
        durations.append(clock() - begin)
        # Result retention and disk logging are outside each local-inference timer.
        values.extend(probability)
        predictions.extend(predicted)
        confidences.extend(float(value) for value in confidence)
        gates.extend(bool(value) for value in gate)
    return durations, np.asarray(values), predictions, confidences, gates


def comparison(ids, actual, predictions, confidence, gates, reference, labels, threshold):
    expected = _predictions(ids, labels, reference, threshold)
    return {
        'maximum_absolute_probability_difference': float(np.max(np.abs(actual - reference))),
        'maximum_absolute_confidence_difference': max(abs(a - b['confidence']) for a, b in zip(confidence, expected)),
        'prediction_mismatch_ids': [row_id for row_id, a, b in zip(ids, predictions, expected) if a != b['predicted_label']],
        'gate_mismatch_ids': [row_id for row_id, a, b in zip(ids, gates, expected) if a != b['use_specialist']],
        'accepted_count': sum(gates), 'reference_accepted_count': sum(row['use_specialist'] for row in expected),
        'predictions_confidences_gates_sha256': sha256(json_bytes([predictions, confidence, gates])),
    }


def summarize(passes, count, batch_size):
    require(len(passes) == 5, 'Exactly five measured passes required')
    expected_batches = (count + batch_size - 1) // batch_size
    require(all(len(p['durations_ns']) == expected_batches and all(v > 0 for v in p['durations_ns']) for p in passes),
            'Missing/nonpositive batch durations')
    times = np.asarray([t for p in passes for t in p['durations_ns']], dtype=np.float64)
    totals = [sum(p['durations_ns']) / 1e9 for p in passes]
    result = {'batch_size': batch_size, 'passes': 5, 'requests_per_pass': count,
              'total_request_events': count * 5, 'batches_per_pass': expected_batches,
              'pass_inference_seconds': totals, 'pass_throughput_requests_per_second': [count / s for s in totals],
              'pooled_throughput_requests_per_second': count * 5 / sum(totals),
              'median_batch_duration_ms': float(np.median(times) / 1e6),
              'p95_batch_duration_ms': float(np.percentile(times, 95) / 1e6)}
    if batch_size == 1:
        result.update(median_request_latency_ms=result['median_batch_duration_ms'],
                      p95_request_latency_ms=result['p95_batch_duration_ms'])
    return result


def memory_snapshot():
    # macOS ru_maxrss is bytes; Linux reports KiB. RSS is sampled by OS ps.
    rss = int(subprocess.check_output(['ps', '-o', 'rss=', '-p', str(os.getpid())], text=True).strip()) * 1024
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    maximum *= 1 if sys.platform == 'darwin' else 1024
    return {'rss_bytes': rss, 'lifetime_peak_rss_bytes': maximum}


def run(root=ROOT):
    root = Path(root).resolve()
    guard_counts = offline_guard(root)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                      TOKENIZERS_PARALLELISM='false')
    output = root / BUNDLE
    protocol_bytes = (output / 'protocol.json').read_bytes()
    timing_protocol = json.loads(protocol_bytes)
    require(not (output / 'results.json').exists(), 'Preserve prior timing evidence; results already exist')
    require(timing_protocol['batch_sizes'] == [1, 32] and timing_protocol['timed_passes'] == 5 and
            timing_protocol['untimed_warmup_passes_per_batch_size'] == 1, 'Timing protocol changed')
    frozen, _, _ = load_frozen(root)
    require(timing_protocol['exp007_protocol_sha256'] == FROZEN_PROTOCOL_SHA256, 'Source protocol changed')
    rows, labels, reference, training_sha = validation_inputs(frozen, root)
    verified = verify_specialist_files(frozen, root)
    threshold = frozen['thresholds']['11']['value']
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(11)
    hardware = {'platform': platform.platform(), 'python': platform.python_version(),
                'machine': platform.machine(), 'logical_cpus': os.cpu_count()}
    if sys.platform == 'darwin':
        hardware.update({key: subprocess.check_output(['sysctl', '-n', key], text=True).strip()
                         for key in ('machdep.cpu.brand_string', 'hw.memsize', 'hw.physicalcpu', 'hw.logicalcpu')})
    memory = {'before_model_loading': memory_snapshot()}
    started = datetime.now(timezone.utc).isoformat()
    with threadpool_limits(limits=1):
        start = time.perf_counter()
        encoder = load_benchmark_encoder(verified['snapshot'])
        encoder_load_seconds = time.perf_counter() - start
        start = time.perf_counter()
        classifier = _load_classifier(root / frozen['thresholds']['11']['classifier_path'])
        classifier_load_seconds = time.perf_counter() - start
        order = _validate_classifier(classifier, labels, frozen['specialists']['embedding']['logistic_regression'])
        before = state_digest(encoder)
        require(before == frozen['specialists']['embedding']['encoder_state_sha256'], 'Encoder weights changed')
        classifier_before = _classifier_state(classifier)
        memory['after_model_loading'] = memory_snapshot()
        runtime = {'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
                   'native_threadpools': threadpool_info(), 'encoding': ENCODING,
                   'packages': verified['packages']}
        results = {}
        ids = [row['id'] for row in rows]
        for batch_size in timing_protocol['batch_sizes']:
            # Exactly one untimed full validation pass for this batching mode.
            for start in range(0, len(rows), batch_size):
                infer_batch(encoder, classifier, [r['text'] for r in rows[start:start + batch_size]],
                            labels, order, threshold, batch_size)
            memory[f'batch_{batch_size}_after_warmup'] = memory_snapshot()
            passes = []
            for pass_index in range(5):
                durations, values, predicted, confidence, gates = timed_pass(
                    encoder, classifier, rows, labels, order, threshold, batch_size)
                agreement = comparison(ids, values, predicted, confidence, gates, reference, labels, threshold)
                passes.append({'pass': pass_index + 1, 'durations_ns': durations, 'comparison': agreement})
                if pass_index == 0:
                    write_json(output / f'predictions_batch_{batch_size}.json',
                               [{'id': i, 'predicted_label': p, 'confidence': c, 'use_specialist': g}
                                for i, p, c, g in zip(ids, predicted, confidence, gates)])
                memory[f'batch_{batch_size}_after_pass_{pass_index + 1}'] = memory_snapshot()
            results[str(batch_size)] = {'passes': passes, 'summary': summarize(passes, len(rows), batch_size)}
        assert_frozen(encoder)
        after = state_digest(encoder)
        classifier_after = _classifier_state(classifier)
        require(before == after and classifier_before == classifier_after, 'Frozen model state changed during timing')
    source_paths = ['baseline/cpu_benchmark.py', 'baseline/final_specialists.py', 'baseline/embeddings.py',
                    'baseline/thresholds.py', 'uv.lock']
    evidence = {'study_id': 'EXP-008', 'status': 'completed', 'started_utc': started,
                'finished_utc': datetime.now(timezone.utc).isoformat(),
                'protocol_sha256': sha256(protocol_bytes), 'exp007_protocol_sha256': FROZEN_PROTOCOL_SHA256,
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                'source_files_sha256': {p: sha256((root / p).read_bytes()) for p in source_paths},
                'validation_ids': ids, 'validation_texts_sha256': sha256(json_bytes(rows)),
                'training_csv_sha256': training_sha, 'labels': labels, 'threshold': threshold,
                'model_provenance': verified, 'encoder_state_sha256_before': before,
                'encoder_state_sha256_after': after, 'classifier_state_sha256_before': classifier_before,
                'classifier_state_sha256_after': classifier_after, 'hardware': hardware, 'runtime': runtime,
                'loading': {'encoder_seconds': encoder_load_seconds, 'classifier_seconds': classifier_load_seconds,
                            'note': 'Warm filesystem possible; excludes file hash validation. Encoder timer includes lazy library import.'},
                'memory': {'method': 'ps RSS in KiB converted to bytes at checkpoints; resource.ru_maxrss lifetime process peak (Darwin bytes). Includes Python/libraries and evidence, not isolated model allocation; no claim of exact transient inference peak.',
                           'snapshots': memory}, 'batching': results, 'audit_guard': guard_counts,
                'official_test_access': False, 'api_calls': 0, 'model_fits': 0,
                'additional_labels': 0, 'existing_training_labels': 1540, 'existing_validation_labels': 770,
                'limitations': 'Single machine, fixed reused validation workload, sequential modes, no load/concurrency study. Batch throughput is not interactive latency. Local compute is not free; no deployment cost estimate.'}
    require(all(value == 0 for value in guard_counts.values()), 'Prohibited access was attempted')
    write_json(output / 'results.json', evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    result = run()
    print(json.dumps({'study_id': result['study_id'], 'batching': {k: v['summary'] for k, v in result['batching'].items()},
                      'loading': result['loading'], 'audit_guard': result['audit_guard']}, indent=2))


if __name__ == '__main__':
    main()
