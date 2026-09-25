"""Label-blind ranking and selective risk must remain auditable at every count."""

import copy
import hashlib
import math
import statistics

import pytest

from baseline.data import SEEDS, SHOTS
from baseline.selective import aggregate, confidence_order, selective_curve


TARGETS = (.25, .5, .75, .9, 1.0)


@pytest.fixture
def rows():
    return [
        {"id": "v:3", "true_label": "c", "predicted_label": "a", "confidence": .6},
        {"id": "v:0", "true_label": "a", "predicted_label": "a", "confidence": .9},
        {"id": "v:2", "true_label": "b", "predicted_label": "b", "confidence": .7},
        {"id": "v:1", "true_label": "b", "predicted_label": "a", "confidence": .8},
    ]


def test_confidence_order_descends_and_accepts_probability_endpoints():
    assert confidence_order(["low", "high", "middle"], [0.0, 1.0, .5]) == [1, 2, 0]


def test_ties_use_sha256_of_id_and_do_not_depend_on_input_order():
    ids = ["validation:12", "validation:3", "validation:40", "validation:1"]
    expected = sorted(ids, key=lambda value: (hashlib.sha256(value.encode("utf-8")).hexdigest(), value))
    assert [ids[i] for i in confidence_order(ids, [.5] * len(ids))] == expected
    reversed_ids = list(reversed(ids))
    assert [reversed_ids[i] for i in confidence_order(reversed_ids, [.5] * len(ids))] == expected


@pytest.mark.parametrize("ids,confidence", [
    ([], []),
    (["a"], []),
    (["a"], [.2, .3]),
    (["a", "a"], [.8, .2]),
    (["a"], [float("nan")]),
    (["a"], [float("inf")]),
    (["a"], [float("-inf")]),
    (["a"], [-.0001]),
    (["a"], [1.0001]),
])
def test_rejects_invalid_rank_inputs(ids, confidence):
    with pytest.raises(ValueError):
        confidence_order(ids, confidence)


def test_curve_counts_and_risk_are_hand_calculated(rows):
    result = selective_curve(rows, ["a", "b", "c"], .5)
    assert [row["id"] for row in result["ranked_rows"]] == ["v:0", "v:1", "v:2", "v:3"]
    assert len(result["curve"]) == len(rows) + 1
    for point, k, errors, correct in zip(result["curve"], range(5), [0, 0, 1, 1, 2], [0, 1, 1, 2, 2]):
        assert point["accepted_count"] == k
        assert point["error_count"] == errors
        assert point["correct_count"] == correct
        assert point["coverage"] == k / 4
        if k:
            assert point["accepted_accuracy"] == pytest.approx(correct / k)
            assert point["selective_risk"] == pytest.approx(errors / k)
            assert point["accepted_accuracy"] + point["selective_risk"] == pytest.approx(1)
        else:
            assert point["accepted_accuracy"] is None
            assert point["selective_risk"] is None
    assert result["curve"][-1]["accepted_accuracy"] == .5


def test_landmark_rounding_reports_actual_coverage_and_per_class_counts(rows):
    rows.append({"id": "v:4", "true_label": "c", "predicted_label": "c", "confidence": .5})
    result = selective_curve(rows, ["a", "b", "c"], .6)
    for point, target in zip(result["landmarks"], TARGETS):
        k = math.ceil(target * len(rows))
        assert point["target_coverage"] == target
        assert point["accepted_count"] == k
        assert point["coverage"] == k / len(rows)
        expected_counts = {label: sum(row["true_label"] == label for row in result["ranked_rows"][:k])
                           for label in ("a", "b", "c")}
        assert point["per_class_acceptance"] == expected_counts
        assert sum(point["per_class_acceptance"].values()) == k
        assert point["zero_acceptance_classes"] == [label for label, count in expected_counts.items() if count == 0]
    assert result["landmarks"][0]["per_class_acceptance"] == {"a": 1, "b": 1, "c": 0}
    assert result["landmarks"][0]["zero_acceptance_classes"] == ["c"]


def test_tied_acceptance_does_not_consult_correctness(rows):
    for row in rows:
        row["confidence"] = .7
    original = selective_curve(rows, ["a", "b", "c"], .5)
    assert [point["boundary_splits_exact_tie"] for point in original["landmarks"]] == [True, True, True, False, False]
    altered = copy.deepcopy(rows)
    for row in altered:
        row["predicted_label"] = row["true_label"]
    corrected = selective_curve(list(reversed(altered)), ["a", "b", "c"], 1.0)
    assert [row["id"] for row in original["ranked_rows"]] == [row["id"] for row in corrected["ranked_rows"]]
    assert [point["per_class_acceptance"] for point in original["landmarks"]] == [
        point["per_class_acceptance"] for point in corrected["landmarks"]]


def test_curve_does_not_modify_input_rows(rows):
    original = copy.deepcopy(rows)
    selective_curve(rows, ["a", "b", "c"], .5)
    assert rows == original


def test_full_coverage_must_match_original_experiment_accuracy(rows):
    with pytest.raises(ValueError):
        selective_curve(rows, ["a", "b", "c"], .6)


def test_all_wrong_is_risk_one_except_undefined_empty_prefix():
    result = selective_curve([
        {"id": "v:0", "true_label": "a", "predicted_label": "b", "confidence": 1.0},
    ], ["a", "b"], 0.0)
    assert result["curve"][0]["accepted_accuracy"] is None
    assert result["curve"][0]["selective_risk"] is None
    assert result["curve"][1]["accepted_accuracy"] == 0.0
    assert result["curve"][1]["selective_risk"] == 1.0
    assert result["landmarks"][-1]["per_class_acceptance"] == {"a": 1, "b": 0}


@pytest.fixture
def runs():
    labels = ["a", "b", "c"]
    result = []
    for regime_index, shots in enumerate(SHOTS):
        for seed_index, seed in enumerate(SEEDS):
            correct_count = 3 + regime_index + seed_index
            rows = [{"id": f"v:{i}", "true_label": labels[i % 3],
                     "predicted_label": labels[i % 3] if i < correct_count else labels[(i + 1) % 3],
                     "confidence": 1 - i / 12} for i in range(12)]
            result.append({"run_id": f"synthetic-n{shots}-s{seed}", "shots": shots, "seed": seed,
                           "labels": labels, **selective_curve(rows, labels, correct_count / 12)})
    return result


def test_aggregate_uses_five_seeds_and_sample_standard_deviation(runs):
    result = aggregate(runs)
    assert [regime["shots"] for regime in result["regimes"]] == list(SHOTS)
    for regime in result["regimes"]:
        empty = regime["curve"][0]
        assert empty["accepted_count"] == 0
        assert empty["coverage"] == 0
        for metric in ("accepted_accuracy", "selective_risk"):
            assert empty[metric]["mean"] is None
            assert empty[metric]["sample_std"] is None
            assert empty[metric]["by_seed"] == {str(seed): None for seed in SEEDS}
        source = [run for run in runs if run["shots"] == regime["shots"]]
        for index, point in enumerate(regime["landmarks"]):
            assert point["target_coverage"] == TARGETS[index]
            for metric in ("coverage", "accepted_count", "error_count", "correct_count", "accepted_accuracy",
                           "selective_risk", "zero_acceptance_class_count"):
                values = [len(run["landmarks"][index]["zero_acceptance_classes"])
                          if metric == "zero_acceptance_class_count" else run["landmarks"][index][metric]
                          for run in source]
                assert point[metric]["mean"] == pytest.approx(statistics.mean(values))
                assert point[metric]["sample_std"] == pytest.approx(statistics.stdev(values))
                assert point[metric]["by_seed"] == {str(run["seed"]): value for run, value in zip(source, values)}
    assert aggregate(list(reversed(runs))) == result


def test_aggregate_rejects_missing_or_duplicate_configurations(runs):
    with pytest.raises(ValueError):
        aggregate(runs[:-1])
    runs[-1] = copy.deepcopy(runs[0])
    with pytest.raises(ValueError):
        aggregate(runs)


def test_aggregate_rejects_changed_validation_ids(runs):
    changed = runs[-1]
    rows = copy.deepcopy(changed["ranked_rows"])
    rows[0]["id"] = "different-validation-id"
    changed.update(selective_curve(rows, changed["labels"], changed["curve"][-1]["accepted_accuracy"]))
    with pytest.raises(ValueError):
        aggregate(runs)


def test_aggregate_rejects_changed_validation_labels(runs):
    changed = runs[-1]
    rows = copy.deepcopy(changed["ranked_rows"])
    rows[0]["true_label"] = "b"
    full_accuracy = sum(row["true_label"] == row["predicted_label"] for row in rows) / len(rows)
    changed.update(selective_curve(rows, changed["labels"], full_accuracy))
    with pytest.raises(ValueError):
        aggregate(runs)
