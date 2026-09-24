import json
import warnings

import pytest
from sklearn.exceptions import ConvergenceWarning

from baseline import experiment
from baseline.data import PROTOCOL_ID, read_json, text_key, write_json


def test_metric_json_schema_and_zero_support_class():
    metrics = experiment.classification_metrics(["a", "b", "b"], ["a", "a", "a"], ["a", "b", "c"])
    result = json.loads(json.dumps(metrics, allow_nan=False))
    assert result["accuracy"] == pytest.approx(1 / 3)
    assert result["macro_f1"] == pytest.approx(1 / 6)
    assert result["n_examples"] == 3
    assert result["confusion_matrix"] == [[1, 0, 0], [2, 0, 0], [0, 0, 0]]
    assert result["confusion_matrix_labels"] == ["a", "b", "c"]
    assert result["per_class"]["c"] == {"precision": 0, "recall": 0, "f1": 0, "support": 0}
    for values in result["per_class"].values():
        assert set(values) == {"precision", "recall", "f1", "support"}
        assert isinstance(values["support"], int)
        assert all(0 <= values[k] <= 1 for k in ("precision", "recall", "f1"))


def test_metrics_reject_unknown_labels():
    with pytest.raises(ValueError, match="Unknown label"):
        experiment.classification_metrics(["a"], ["unknown"], ["a", "b"])


def test_validation_cannot_change_vocabulary_or_idf():
    model = experiment.make_pipeline()
    texts = ["card arrival", "card shipping", "cancel transfer", "transfer pending",
             "cash withdrawal", "cash charge"]
    model.fit(texts, ["a", "a", "b", "b", "c", "c"])
    vectorizer = model.named_steps["tfidf"]
    vocabulary, idf = dict(vectorizer.vocabulary_), vectorizer.idf_.copy()
    model.predict(["validationexclusive card", "validationexclusive transfer"])
    assert "validationexclusive" not in vectorizer.vocabulary_
    assert vectorizer.vocabulary_ == vocabulary
    assert (vectorizer.idf_ == idf).all()


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch):
    # Synthetic unit fixture only; never loads BANKING77 or evaluates official test.
    labels = ["a", "b", "c"]
    def row(row_id, label, text):
        return {"id": row_id, "label": label, "text": text, "key": text_key(text)}
    pool = [row(f"train:{label}:{i}", label, f"intent{label} example {i}")
            for label in labels for i in range(6)]
    validation = [row(f"val:{label}", label, f"intent{label} validationexclusive") for label in labels]
    manifest = {"labels": labels, "protocol_id": PROTOCOL_ID,
                "source": {"dataset": "synthetic"}, "sealed_test": [],
                "audit": {"original_train": 21}}
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)
    monkeypatch.setattr(experiment, "load_development", lambda *args: (manifest, pool, validation))
    monkeypatch.setattr(experiment, "code_provenance", lambda *args: {"fixture": "synthetic"})
    return tmp_path, manifest_path


def test_run_serializes_metrics_metadata_and_predictions(synthetic_run):
    raw, manifest = synthetic_run
    output = raw / "run"
    experiment.run(raw, manifest, output, 5, 11)
    metadata, metrics = read_json(output / "metadata.json"), read_json(output / "metrics.json")
    predictions = read_json(output / "predictions.json")
    probabilities = read_json(output / "probabilities.json")
    assert metadata["status"] == "completed"
    assert metadata["protocol_id"] == PROTOCOL_ID
    assert metadata["label_budgets"]["training"] == 15
    assert metadata["label_budgets"]["test_evaluation"] == 0
    assert metrics["schema_version"] == 1
    assert metrics["evaluation_split"] == "validation"
    assert probabilities["labels"] == ["a", "b", "c"]
    assert len(probabilities["probabilities"]) == 3
    assert all(sum(p) == pytest.approx(1) for p in probabilities["probabilities"])
    for model, field in [("tfidf_logistic_regression", "predicted_label"),
                         ("dummy_most_frequent", "dummy_predicted_label")]:
        recomputed = experiment.classification_metrics(
            [p["true_label"] for p in predictions], [p[field] for p in predictions], ["a", "b", "c"])
        assert metrics[model] == recomputed
    import joblib
    model = joblib.load(output / "model.joblib")
    assert "validationexclusive" not in model.named_steps["tfidf"].vocabulary_
    with pytest.raises(FileExistsError):
        experiment.run(raw, manifest, output, 5, 11)


def test_convergence_warning_retains_failed_run(synthetic_run, monkeypatch):
    class Nonconvergent:
        def fit(self, *args):
            warnings.warn("synthetic failure", ConvergenceWarning)

    raw, manifest = synthetic_run
    output = raw / "failed"
    monkeypatch.setattr(experiment, "make_pipeline", Nonconvergent)
    with pytest.raises(RuntimeError, match="did not converge"):
        experiment.run(raw, manifest, output, 5, 11)
    metadata = read_json(output / "metadata.json")
    assert metadata["status"] == "failed"
    assert any("ConvergenceWarning" in w for w in metadata["warnings"])
    assert not (output / "metrics.json").exists()
