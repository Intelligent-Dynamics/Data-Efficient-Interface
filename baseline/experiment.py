"""One validation-only experiment, with explicit provenance and JSON artifacts."""

import importlib.metadata
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_info, threadpool_limits

from .data import (ROOT, few_shot, json_bytes, load_development, sha256,
                   verify_isolation, write_json)

TFIDF = {
    "lowercase": True, "analyzer": "word", "ngram_range": (1, 2),
    "token_pattern": r"(?u)\b\w\w+\b", "min_df": 1, "max_df": 1.0,
    "stop_words": None, "smooth_idf": True, "use_idf": True,
    "norm": "l2", "sublinear_tf": True,
}
LOGISTIC = {
    "penalty": "l2", "C": 1.0, "solver": "lbfgs", "max_iter": 2000,
    "tol": 1e-4, "fit_intercept": True, "class_weight": None,
}


def make_pipeline():
    # scikit-learn 1.7.2: lbfgs uses multinomial loss for >2 classes.
    return Pipeline([("tfidf", TfidfVectorizer(**TFIDF)),
                     ("classifier", LogisticRegression(**LOGISTIC))])


def classification_metrics(truth, predicted, labels):
    if not truth or len(truth) != len(predicted) or not labels or len(set(labels)) != len(labels):
        raise ValueError("Metrics require nonempty aligned predictions and unique labels")
    if not set(truth + predicted) <= set(labels):
        raise ValueError("Unknown label in predictions or truth")
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=labels, zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1.mean()), "n_examples": len(truth),
        "per_class": {
            label: {"precision": float(precision[i]), "recall": float(recall[i]),
                    "f1": float(f1[i]), "support": int(support[i])}
            for i, label in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
        "confusion_matrix_labels": labels,
        "confusion_matrix_axes": {"rows": "true", "columns": "predicted"},
    }


def code_provenance(output):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()

    if Path(git("rev-parse", "--show-toplevel")).resolve() != ROOT.resolve():
        raise RuntimeError("Project checkout is missing; refusing provenance from a parent repository")

    # Snapshot new/uncommitted source too; a base Git SHA alone cannot identify this run.
    paths = [*sorted((ROOT / "baseline").glob("*.py")),
             *sorted((ROOT / "tests").glob("*.py")),
             ROOT / "pyproject.toml", ROOT / "uv.lock", ROOT / ".python-version",
             ROOT / "data/banking77-source.json", ROOT / "docs/CURRENT_PLAN.md"]
    hashes = {}
    for source in paths:
        relative = source.relative_to(ROOT)
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[str(relative)] = sha256(source.read_bytes())
    patch = subprocess.check_output(["git", "-C", str(ROOT), "diff", "--binary", "HEAD"])
    (output / "working-tree.patch").write_bytes(patch)
    return {"git_head": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
            "working_tree_patch_sha256": sha256(patch), "source_files_sha256": hashes,
            "source_snapshot_sha256": sha256(json_bytes(hashes))}


def prediction_timings(model, texts):
    result = {"warmup_passes": 1, "timed_passes": 5, "units": "seconds"}
    for name, batch_size in (("single_request", 1), ("full_validation", len(texts))):
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        for batch in batches:
            model.predict(batch)
        durations = []
        for _ in range(5):
            start = time.perf_counter()
            for batch in batches:
                model.predict(batch)
            durations.append(time.perf_counter() - start)
        median = statistics.median(durations)
        result[name] = {"batch_size": batch_size, "requests_per_pass": len(texts),
                        "pass_seconds": durations, "median_pass_seconds": median,
                        "median_seconds_per_request": median / len(texts),
                        "requests_per_second": len(texts) / median}
    return result


def run(raw, manifest_path, output, shots, seed):
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite a prior run.
    metadata = {
        "schema_version": 1, "run_id": output.name, "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "single validation experiment; not a final project result",
        "evaluation_split": "validation", "shots": shots, "seed": seed,
        "command": shlex.join([sys.executable, "-m", "baseline", *sys.argv[1:]]),
        "configuration": {"tfidf": TFIDF, "logistic_regression": LOGISTIC,
                          "multiclass_loss": "multinomial", "compute_threads": 1},
        "warnings": [],
    }
    write_json(output / "metadata.json", metadata)
    try:
        metadata["code"] = code_provenance(output)
        metadata["environment"] = {
            "python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("scikit-learn", "numpy", "scipy", "joblib", "threadpoolctl", "pytest")},
        }
        manifest, pool, validation = load_development(raw, manifest_path)
        train = few_shot(pool, manifest["labels"], shots, seed)
        verify_isolation(train, validation, manifest["sealed_test"])
        labels = manifest["labels"]
        metadata["source"] = manifest["source"]
        metadata["data_audit"] = manifest["audit"]
        metadata["label_budgets"] = {
            "training": len(train), "validation": len(validation),
            "development_unique": len(train) + len(validation),
            "calibration": 0, "prompt_examples": 0, "test_evaluation": 0,
            "training_source_labels_read_for_stratification_and_duplicate_audit":
                manifest["audit"]["original_train"],
            "note": "Simulated few-shot access to public labels, not an annotation-cost measurement",
        }
        shutil.copyfile(manifest_path, output / "split_manifest.json")
        metadata["split_manifest_sha256"] = sha256(manifest_path.read_bytes())
        samples = {"labels": labels, "shots": shots, "seed": seed,
                   "train_ids": [r["id"] for r in train],
                   "validation_ids": [r["id"] for r in validation]}
        write_json(output / "samples.json", samples)
        metadata["samples_sha256"] = sha256(json_bytes(samples))
        x_train, y_train = [r["text"] for r in train], [r["label"] for r in train]
        x_val, truth = [r["text"] for r in validation], [r["label"] for r in validation]
        model = make_pipeline()
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            start = time.perf_counter()
            model.fit(x_train, y_train)
            fit_seconds = time.perf_counter() - start
            metadata["warnings"] = [f"{w.category.__name__}: {w.message}" for w in caught]
            if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                raise RuntimeError("Logistic regression did not converge; run is incomplete")
            predicted = model.predict(x_val).tolist()
            probability_order = [model.classes_.tolist().index(label) for label in labels]
            probabilities = model.predict_proba(x_val)[:, probability_order].tolist()
            dummy = DummyClassifier(strategy="most_frequent").fit([[0]] * len(train), y_train)
            dummy_predicted = dummy.predict([[0]] * len(validation)).tolist()
            timing = prediction_timings(model, x_val)
            metadata["environment"]["threadpools"] = threadpool_info()
            metadata["warnings"] = [f"{w.category.__name__}: {w.message}" for w in caught]
        metrics = {"schema_version": 1, "evaluation_split": "validation",
                   "tfidf_logistic_regression": classification_metrics(truth, predicted, labels),
                   "dummy_most_frequent": classification_metrics(truth, dummy_predicted, labels)}
        predictions = [
            {"id": row["id"], "true_label": truth[i], "predicted_label": predicted[i],
             "dummy_predicted_label": dummy_predicted[i]}
            for i, row in enumerate(validation)
        ]
        write_json(output / "predictions.json", predictions)
        write_json(output / "probabilities.json", {"labels": labels,
                   "validation_ids": samples["validation_ids"], "probabilities": probabilities})
        write_json(output / "metrics.json", metrics)
        joblib.dump(model, output / "model.joblib")
        metadata["resources"] = {"fit_seconds": fit_seconds, "prediction_timing": timing,
                                  "model_bytes": (output / "model.joblib").stat().st_size}
        metadata["dummy_tie_break"] = {"rule": "first class in sklearn sorted class order",
                                      "predicted_label": dummy_predicted[0]}
        metadata["vocabulary_size"] = len(model.named_steps["tfidf"].vocabulary_)
        metadata["optimizer_iterations"] = model.named_steps["classifier"].n_iter_.tolist()
        metadata["artifacts_sha256"] = {p.name: sha256(p.read_bytes()) for p in sorted(output.iterdir())
                                         if p.is_file() and p.name != "metadata.json"}
        metadata["status"] = "completed"
        return metrics
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "metadata.json", metadata)
