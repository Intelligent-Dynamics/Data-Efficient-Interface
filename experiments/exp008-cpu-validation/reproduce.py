"""Recompute EXP-008 summaries from compact timing arrays, without models/data/network."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from baseline.data import json_bytes, read_json, sha256


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def verify(directory):
    result = read_json(directory / 'results.json')
    assert result['protocol_sha256'] == sha256((directory / 'protocol.json').read_bytes())
    assert len(result['validation_ids']) == len(set(result['validation_ids'])) == 770
    assert result['audit_guard'] == {'official_test_access_attempts': 0, 'network_access_attempts': 0}
    assert result['model_fits'] == result['api_calls'] == 0
    assert result['encoder_state_sha256_before'] == result['encoder_state_sha256_after']
    assert result['classifier_state_sha256_before'] == result['classifier_state_sha256_after']
    for key, item in result['batching'].items():
        batch = int(key)
        expected_count = (770 + batch - 1) // batch
        assert len(item['passes']) == 5
        assert all(len(p['durations_ns']) == expected_count for p in item['passes'])
        values = [t for p in item['passes'] for t in p['durations_ns']]
        assert all(type(t) is int and t > 0 for t in values)
        totals = [sum(p['durations_ns']) / 1e9 for p in item['passes']]
        summary = item['summary']
        assert summary['pass_inference_seconds'] == totals
        assert summary['pass_throughput_requests_per_second'] == [770 / x for x in totals]
        assert summary['pooled_throughput_requests_per_second'] == 3850 / sum(totals)
        assert summary['median_batch_duration_ms'] == statistics.median(values) / 1e6
        assert summary['p95_batch_duration_ms'] == percentile(values, .95) / 1e6
        predicted = read_json(directory / f'predictions_batch_{batch}.json')
        assert [row['id'] for row in predicted] == result['validation_ids']
        assert len(predicted) == 770
        payload = [[r['predicted_label'] for r in predicted], [r['confidence'] for r in predicted],
                   [r['use_specialist'] for r in predicted]]
        digest = sha256(json_bytes(payload))
        assert item['passes'][0]['comparison']['predictions_confidences_gates_sha256'] == digest
        assert all(r['use_specialist'] == (r['confidence'] >= result['threshold']) for r in predicted)
    return {key: value['summary'] for key, value in result['batching'].items()}


if __name__ == '__main__':
    print(json.dumps({'status': 'passed', 'summary': verify(Path(__file__).resolve().parent)}, indent=2))
