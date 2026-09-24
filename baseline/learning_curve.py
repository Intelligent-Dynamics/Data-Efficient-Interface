"""Verify the complete validation study and reproduce its aggregate figures.

No test evaluation and no model fitting: only saved-model checks, train-only
TF-IDF reconstruction, prediction replay on validation, and metric aggregation.
"""

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits

from .data import (DEFAULT_MANIFEST, PROTOCOL_ID, SEEDS, SHOTS, few_shot,
                   json_bytes, load_development, read_json, sha256,
                   verify_isolation, write_json)
from .experiment import LOGISTIC, TFIDF, classification_metrics

MODEL = "tfidf_logistic_regression"
MODEL_SOURCES = ["baseline/data.py", "baseline/experiment.py", "baseline/__main__.py"]
REQUIRED_ARTIFACTS = {"metadata.json", "metrics.json", "samples.json", "predictions.json",
                      "probabilities.json", "split_manifest.json", "model.joblib",
                      "working-tree.patch"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_record(record):
    metadata, samples, predictions, metrics = (record[k] for k in
                                             ("metadata", "samples", "predictions", "metrics"))
    require(metadata["status"] == "completed", "Incomplete run")
    require(metadata["evaluation_split"] == metrics["evaluation_split"] == "validation",
            "Only validation records are allowed")
    require(metadata["protocol_id"] == PROTOCOL_ID, "Mixed protocol")
    require(not metadata["warnings"], "Review warnings before including a run")
    require((metadata["shots"], metadata["seed"]) == (samples["shots"], samples["seed"]),
            "Sample/configuration mismatch")
    require(len(samples["labels"]) == 77, "Expected all 77 classes")
    require(len(samples["train_ids"]) == 77 * metadata["shots"], "Training budget mismatch")
    require(len(samples["validation_ids"]) == 770, "Validation budget mismatch")
    require(len(set(samples["train_ids"])) == len(samples["train_ids"]), "Repeated training IDs")
    require(len(set(samples["validation_ids"])) == 770, "Repeated validation IDs")
    require(not set(samples["train_ids"]) & set(samples["validation_ids"]), "ID leakage")
    require(metadata["label_budgets"]["training"] == len(samples["train_ids"])
            and metadata["label_budgets"]["validation"] == 770
            and metadata["label_budgets"]["test_evaluation"] == 0, "Label budget mismatch")
    require([p["id"] for p in predictions] == samples["validation_ids"], "Prediction IDs differ")
    truth = [p["true_label"] for p in predictions]
    require(Counter(truth) == {label: 10 for label in samples["labels"]}, "Class support mismatch")
    for model, field in ((MODEL, "predicted_label"), ("dummy_most_frequent", "dummy_predicted_label")):
        computed = classification_metrics(truth, [p[field] for p in predictions], samples["labels"])
        require(computed == metrics[model], f"Saved metrics differ from predictions: {model}")


def load_and_verify_run(path, manifest, pool, validation):
    require(REQUIRED_ARTIFACTS <= {p.name for p in path.iterdir()}, f"Missing artifacts: {path}")
    record = {name: read_json(path / f"{name}.json")
              for name in ("metadata", "samples", "predictions", "metrics")}
    validate_record(record)
    metadata, samples = record["metadata"], record["samples"]
    require(metadata["run_id"] == path.name, "Run ID does not match directory")
    require(REQUIRED_ARTIFACTS - {"metadata.json"} <= set(metadata["artifacts_sha256"]),
            "Required artifact hashes missing")
    for name, expected in metadata["artifacts_sha256"].items():
        require(sha256((path / name).read_bytes()) == expected, f"Artifact hash mismatch: {path}/{name}")
    for name, expected in metadata["code"]["source_files_sha256"].items():
        require(sha256((path / "source" / name).read_bytes()) == expected, f"Source hash mismatch: {name}")
    require(metadata["samples_sha256"] == sha256(json_bytes(samples)), "Sample hash mismatch")
    require(metadata["source"] == manifest["source"], "Wrong dataset revision/checksums")
    require(metadata["split_manifest_sha256"] == sha256(json_bytes(manifest)), "Wrong split hash")
    require(read_json(path / "split_manifest.json") == manifest, "Split content mismatch")
    require(metadata["configuration"] == json.loads(json_bytes({
        "tfidf": TFIDF, "logistic_regression": LOGISTIC,
        "multiclass_loss": "multinomial", "compute_threads": 1})), "Model configuration changed")
    train = few_shot(pool, manifest["labels"], metadata["shots"], metadata["seed"])
    require(samples["train_ids"] == [r["id"] for r in train], "Not the intended few-shot subset")
    require(samples["validation_ids"] == [r["id"] for r in validation], "Validation set changed")
    require([p["true_label"] for p in record["predictions"]] == [r["label"] for r in validation],
            "Validation labels changed")
    verify_isolation(train, validation, manifest["sealed_test"])
    require(Counter(r["label"] for r in train) ==
            {label: metadata["shots"] for label in manifest["labels"]}, "Class budget mismatch")
    model = joblib.load(path / "model.joblib")  # Only this project's locally generated models.
    require(model.named_steps["classifier"].classes_.tolist() == sorted(manifest["labels"]),
            "Wrong classifier labels")
    for key, value in LOGISTIC.items():
        require(model.named_steps["classifier"].get_params()[key] == value, "Classifier config mismatch")
    for key, value in TFIDF.items():
        require(model.named_steps["tfidf"].get_params()[key] == value, "Vectorizer config mismatch")
    expected_vectorizer = TfidfVectorizer(**TFIDF).fit([r["text"] for r in train])
    require(model.named_steps["tfidf"].vocabulary_ == expected_vectorizer.vocabulary_,
            "Vocabulary was not fitted on the intended training subset")
    require(np.array_equal(model.named_steps["tfidf"].idf_, expected_vectorizer.idf_),
            "IDF was not fitted on the intended training subset")
    texts = [r["text"] for r in validation]
    require(model.predict(texts).tolist() == [p["predicted_label"] for p in record["predictions"]],
            "Serialized model predictions differ")
    probabilities = read_json(path / "probabilities.json")
    require(probabilities["labels"] == manifest["labels"] and
            probabilities["validation_ids"] == samples["validation_ids"], "Probability ordering mismatch")
    values = np.asarray(probabilities["probabilities"])
    require(values.shape == (770, 77) and np.isfinite(values).all()
            and (values >= 0).all() and (values <= 1).all()
            and np.allclose(values.sum(axis=1), 1, atol=1e-12, rtol=0), "Invalid probabilities")
    order = [model.classes_.tolist().index(label) for label in manifest["labels"]]
    require(np.allclose(values, model.predict_proba(texts)[:, order], atol=1e-12, rtol=0),
            "Serialized model probabilities differ")
    return record, model


def verify_study(runs_dir, reproductions_dir):
    manifest, pool, validation = load_development(Path("data/raw/banking77"), DEFAULT_MANIFEST)
    records, reproduced = [], []
    expected_names = {f"exp002-v2-n{n}-s{s}" for n in SHOTS for s in SEEDS}
    require({p.name for p in runs_dir.iterdir() if p.is_dir()} == expected_names,
            "Primary directory must contain exactly the 15 planned runs")
    require({p.name for p in reproductions_dir.iterdir() if p.is_dir()} ==
            {name + "-reproduction" for name in expected_names}, "Incomplete reproduction matrix")
    checks = []
    with threadpool_limits(limits=1):
        for shots in SHOTS:
            for seed in SEEDS:
                name = f"exp002-v2-n{shots}-s{seed}"
                original, model = load_and_verify_run(runs_dir / name, manifest, pool, validation)
                repeat, repeat_model = load_and_verify_run(
                    reproductions_dir / (name + "-reproduction"), manifest, pool, validation)
                require(original["samples"] == repeat["samples"], "Reproduction samples differ")
                require(original["predictions"] == repeat["predictions"], "Reproduction predictions differ")
                require(original["metrics"] == repeat["metrics"], "Reproduction metrics differ")
                for source in MODEL_SOURCES:
                    require(original["metadata"]["code"]["source_files_sha256"][source] ==
                            repeat["metadata"]["code"]["source_files_sha256"][source], "Model source changed")
                classifier, repeated = model.named_steps["classifier"], repeat_model.named_steps["classifier"]
                require(np.allclose(classifier.coef_, repeated.coef_, rtol=0, atol=1e-10)
                        and np.allclose(classifier.intercept_, repeated.intercept_, rtol=0, atol=1e-10),
                        "Reproduced classifier parameters differ")
                records.append(original)
                reproduced.append(repeat)
                checks.append({"run_id": name, "reproduction_id": name + "-reproduction",
                               "identical_samples_predictions_metrics": True,
                               "classifier_parameter_max_abs_difference": float(max(
                                   np.max(np.abs(classifier.coef_ - repeated.coef_)),
                                   np.max(np.abs(classifier.intercept_ - repeated.intercept_))))})
    report = {"protocol_id": PROTOCOL_ID, "primary_run_count": len(records),
              "independent_reproduction_count": len(reproduced), "checks": checks,
              "intended_training_subsets_verified": True, "train_only_vocabulary_and_idf_verified": True,
              "shared_validation_verified": True, "all_artifact_hashes_verified": True,
              "official_test_evaluation": False, "test_access": "byte integrity only during this study",
              "official_test_sha256": manifest["source"]["files"]["test.csv"]["sha256"],
              "split_manifest_sha256": sha256(json_bytes(manifest))}
    return records, reproduced, report


def aggregate(records):
    require(len(records) == len(SHOTS) * len(SEEDS), "Expected exactly 15 primary runs")
    for record in records:
        validate_record(record)
    by_pair = {(r["metadata"]["shots"], r["metadata"]["seed"]): r for r in records}
    require(set(by_pair) == {(n, s) for n in SHOTS for s in SEEDS}, "Duplicate or missing seed/regime")
    reference = by_pair[(5, 11)]
    for record in records:
        for field in ("configuration", "source", "split_manifest_sha256"):
            require(record["metadata"][field] == reference["metadata"][field], f"Mixed {field}")
        for field in ("labels", "validation_ids"):
            require(record["samples"][field] == reference["samples"][field], f"Mixed {field}")
    labels = reference["samples"]["labels"]
    regimes = []
    for shots in SHOTS:
        selected = [by_pair[(shots, seed)] for seed in SEEDS]
        regime = {"shots": shots, "seeds": list(SEEDS), "training_examples_per_run": 77 * shots,
                  "validation_examples": 770}
        for metric in ("accuracy", "macro_f1"):
            values = [r["metrics"][MODEL][metric] for r in selected]
            regime[metric] = {"mean": statistics.mean(values), "sample_std": statistics.stdev(values),
                              "min": min(values), "max": max(values),
                              "by_seed": {str(seed): v for seed, v in zip(SEEDS, values)}}
        regime["mean_fit_seconds"] = statistics.mean(r["metadata"]["resources"]["fit_seconds"] for r in selected)
        regime["unique_training_examples_across_seeds"] = len(set().union(
            *(set(r["samples"]["train_ids"]) for r in selected)))
        regimes.append(regime)
    gains = []
    for low, high in zip(SHOTS, SHOTS[1:]):
        entry = {"from_shots": low, "to_shots": high}
        for metric in ("accuracy", "macro_f1"):
            delta = {str(seed): 100 * (by_pair[(high, seed)]["metrics"][MODEL][metric] -
                                      by_pair[(low, seed)]["metrics"][MODEL][metric]) for seed in SEEDS}
            entry[metric] = {"mean_gain_pp": statistics.mean(delta.values()), "by_seed_gain_pp": delta,
                             "min_gain_pp": min(delta.values()), "max_gain_pp": max(delta.values())}
        gains.append(entry)
    unique_train = set().union(*(set(r["samples"]["train_ids"]) for r in records))
    summary = {"study_id": "EXP-002", "protocol_id": PROTOCOL_ID, "evaluation_split": "validation",
               "metric_scale": "fraction 0..1", "standard_deviation": "sample SD across 5 training seeds (ddof=1), not a confidence interval",
               "primary_run_count": len(records), "validation_examples": 770, "regimes": regimes,
               "paired_gains": gains, "unique_training_examples_across_study": len(unique_train),
               "unique_training_plus_validation_examples": len(unique_train) + 770,
               "validation_ids_sha256": sha256(json_bytes(reference["samples"]["validation_ids"])),
               "split_manifest_sha256": reference["metadata"]["split_manifest_sha256"],
               "run_ids": [by_pair[(n, s)]["metadata"]["run_id"] for n in SHOTS for s in SEEDS]}
    errors = {"note": "Each regime repeats the SAME 10 validation examples per class across five models; counts are prediction events, not independent examples.",
              "by_regime": {}}
    def class_means(selected):
        return [{"label": label, **{f"mean_{metric}": statistics.mean(
            r["metrics"][MODEL]["per_class"][label][metric] for r in selected)
            for metric in ("precision", "recall", "f1")}}
            for label in labels]
    errors["all_regimes_class_means"] = sorted(class_means(records), key=lambda x: (x["mean_recall"], x["mean_f1"], x["label"]))
    for shots in SHOTS:
        selected = [by_pair[(shots, seed)] for seed in SEEDS]
        confusion = np.sum([r["metrics"][MODEL]["confusion_matrix"] for r in selected], axis=0)
        pairs = []
        for i, true_label in enumerate(labels):
            for j, predicted_label in enumerate(labels):
                if i != j and confusion[i, j]:
                    ids = {p["id"] for r in selected for p in r["predictions"]
                           if p["true_label"] == true_label and p["predicted_label"] == predicted_label}
                    pairs.append({"true_label": true_label, "predicted_label": predicted_label,
                                  "prediction_events": int(confusion[i, j]), "out_of_events_for_true_class": 50,
                                  "distinct_validation_examples": len(ids)})
        errors["by_regime"][str(shots)] = {
            "classes_by_mean_recall": sorted(class_means(selected), key=lambda x: (x["mean_recall"], x["mean_f1"], x["label"])),
            "common_confusions": sorted(pairs, key=lambda p: (-p["prediction_events"], p["true_label"], p["predicted_label"])),
            "summed_confusion_matrix": confusion.tolist(), "labels": labels}
    return summary, errors


def plot_summary(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "exp002-learning-curve"
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for ax, metric, title, color in zip(axes, ("macro_f1", "accuracy"),
                                      ("Macro F1 (primary)", "Accuracy"), ("#175b9c", "#277b51")):
        regimes = summary["regimes"]
        ax.errorbar([r["shots"] for r in regimes], [r[metric]["mean"] for r in regimes],
                    yerr=[r[metric]["sample_std"] for r in regimes], fmt="o-", capsize=5,
                    color=color, linewidth=2, label="Mean ± 1 sample SD")
        for r in regimes:
            values = list(r[metric]["by_seed"].values())
            ax.scatter([r["shots"] + (i - 2) * .16 for i in range(5)], values,
                       s=20, alpha=.45, color=color, zorder=3)
        ax.set(title=title, xlabel="Labeled training examples per class", ylabel=f"Validation {metric.replace('_', ' ')}",
               xticks=[5, 10, 20], ylim=(0, 1), xlim=(3, 22))
        ax.grid(axis="y", alpha=.2)
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("BANKING77 · TF-IDF + logistic regression\n770 fixed validation examples · five training seeds", fontsize=13)
    fig.savefig(output / "learning_curve.png", dpi=180)
    svg = output / "learning_curve.svg"
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def save_outputs(records, output):
    summary, errors = aggregate(records)
    write_json(output / "summary.json", summary)
    write_json(output / "error_analysis.json", errors)
    with (output / "per_run.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["run_id", "shots", "seed", "train_examples", "validation_examples", "accuracy", "macro_f1", "fit_seconds"])
        for r in sorted(records, key=lambda r: (r["metadata"]["shots"], r["metadata"]["seed"])):
            m = r["metadata"]
            writer.writerow([m["run_id"], m["shots"], m["seed"], m["label_budgets"]["training"],
                             m["label_budgets"]["validation"], r["metrics"][MODEL]["accuracy"],
                             r["metrics"][MODEL]["macro_f1"], m["resources"]["fit_seconds"]])
    plot_summary(summary, output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--runs-dir", type=Path)
    group.add_argument("--records-dir", type=Path, help="Reaggregate checked, versioned run JSON without raw data")
    parser.add_argument("--reproductions-dir", type=Path)
    args = parser.parse_args()
    if args.runs_dir:
        require(args.reproductions_dir is not None, "Independent reproduction directory required")
        records, repeats, report = verify_study(args.runs_dir, args.reproductions_dir)
        args.output.mkdir(parents=True, exist_ok=False)
        for name, collection in (("runs", records), ("reproductions", repeats)):
            directory = args.output / name
            directory.mkdir()
            for record in collection:
                (directory / f"{record['metadata']['run_id']}.json").write_text(
                    json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        write_json(args.output / "verification.json", report)
    else:
        records = [read_json(p) for p in sorted(args.records_dir.glob("*.json"))]
        args.output.mkdir(parents=True, exist_ok=False)
    summary = save_outputs(records, args.output)
    print(json.dumps(summary["regimes"], indent=2))


if __name__ == "__main__":
    main()
