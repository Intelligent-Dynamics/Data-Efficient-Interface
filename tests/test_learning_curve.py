"""Aggregation must include every seed and reject incomparable or corrupt records."""

import copy
import statistics

import pytest

from baseline.data import PROTOCOL_ID, SEEDS, SHOTS
from baseline.experiment import classification_metrics
from baseline.learning_curve import MODEL, aggregate


@pytest.fixture
def records():
    labels = [f"intent_{i:02d}" for i in range(77)]
    truth = [label for label in labels for _ in range(10)]
    validation_ids = [f"val:{i}" for i in range(770)]
    results = []
    for regime_index, shots in enumerate(SHOTS):
        for seed_index, seed in enumerate(SEEDS):
            correct_per_class = 3 + regime_index + seed_index
            predictions = []
            for i, label in enumerate(labels):
                for j in range(10):
                    predictions.append({"id": validation_ids[i * 10 + j], "true_label": label,
                                        "predicted_label": label if j < correct_per_class else labels[(i + 1) % 77],
                                        "dummy_predicted_label": labels[0]})
            results.append({
                "metadata": {"status": "completed", "evaluation_split": "validation", "protocol_id": PROTOCOL_ID,
                             "shots": shots, "seed": seed, "run_id": f"n{shots}-s{seed}", "warnings": [],
                             "configuration": {"fixture": True}, "source": {"revision": "synthetic"},
                             "split_manifest_sha256": "fixed", "resources": {"fit_seconds": .01},
                             "label_budgets": {"training": 77 * shots, "validation": 770, "test_evaluation": 0}},
                "samples": {"shots": shots, "seed": seed, "labels": labels, "validation_ids": list(validation_ids),
                            "train_ids": [f"train:{seed}:{label}:{i}" for label in labels for i in range(shots)]},
                "predictions": predictions,
                "metrics": {"evaluation_split": "validation",
                            MODEL: classification_metrics(truth, [p["predicted_label"] for p in predictions], labels),
                            "dummy_most_frequent": classification_metrics(truth, [labels[0]] * 770, labels)},
            })
    return results


def test_aggregate_uses_all_seeds_and_sample_standard_deviation(records):
    summary, errors = aggregate(records)
    for regime, mean in zip(summary["regimes"], (.5, .6, .7)):
        for metric in ("accuracy", "macro_f1"):
            result = regime[metric]
            assert result["mean"] == pytest.approx(mean)
            assert result["sample_std"] == pytest.approx(statistics.stdev([.3, .4, .5, .6, .7]))
            assert result["min"] == pytest.approx(mean - .2)
            assert result["max"] == pytest.approx(mean + .2)
            assert set(result["by_seed"]) == {str(s) for s in SEEDS}
    assert summary["primary_run_count"] == 15
    assert summary["unique_training_examples_across_study"] == 5 * 77 * 20
    assert summary["paired_gains"][0]["macro_f1"]["mean_gain_pp"] == pytest.approx(10)
    assert errors["by_regime"]["5"]["common_confusions"][0]["prediction_events"] == 25
    assert errors["by_regime"]["5"]["common_confusions"][0]["distinct_validation_examples"] == 7
    assert aggregate(list(reversed(records))) == (summary, errors)


def test_rejects_missing_or_duplicate_seeds(records):
    with pytest.raises(ValueError, match="exactly 15"):
        aggregate(records[:-1])
    records[-1] = records[0]
    with pytest.raises(ValueError, match="Duplicate or missing"):
        aggregate(records)


def test_rejects_changed_validation(records):
    record = records[-1]
    record["samples"]["validation_ids"][0] = "different-validation-id"
    record["predictions"][0]["id"] = "different-validation-id"
    with pytest.raises(ValueError, match="Mixed validation_ids"):
        aggregate(records)


def test_rejects_fabricated_metrics(records):
    records[-1]["metrics"][MODEL]["accuracy"] += .001
    with pytest.raises(ValueError, match="Saved metrics differ"):
        aggregate(records)


def test_rejects_training_id_leakage(records):
    records[-1]["samples"]["train_ids"][0] = records[-1]["samples"]["validation_ids"][0]
    with pytest.raises(ValueError, match="ID leakage"):
        aggregate(records)


def test_rejects_mixed_model_configuration(records):
    records[-1]["metadata"]["configuration"] = {"fixture": False}
    with pytest.raises(ValueError, match="Mixed configuration"):
        aggregate(records)
