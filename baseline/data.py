"""Pinned downloads, mechanical duplicate auditing, and deterministic splits."""

import csv
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/banking77-source.json"
SPLIT_SEED = 20260924
SHOTS = (5, 10, 20, 50)
SEEDS = (11, 22, 33, 44, 55)
VALIDATION_PER_CLASS = 20


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def write_json(path, value):
    Path(path).write_bytes(json_bytes(value))


def read_json(path):
    return json.loads(Path(path).read_text())


def text_key(text):
    """Audit only: never substitute normalized text for model input."""
    normalized = " ".join(unicodedata.normalize("NFKC", text).lower().split())
    return sha256(normalized.encode())


def source_spec():
    source = read_json(SOURCE)
    for spec in source["files"].values():
        spec["url"] = (
            "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
            f"{source['revision']}/{spec['path']}"
        )
    return source


def verify_files(raw, source):
    for name, spec in source["files"].items():
        if sha256((raw / name).read_bytes()) != spec["sha256"]:
            raise ValueError(f"Checksum mismatch: {name}; refusing to use altered data")


def download(raw, source):
    raw.mkdir(parents=True, exist_ok=True)
    for name, spec in source["files"].items():
        target = raw / name
        if not target.exists():
            with urlopen(spec["url"], timeout=60) as response:
                payload = response.read()
            if sha256(payload) != spec["sha256"]:
                raise ValueError(f"Downloaded checksum mismatch: {name}")
            target.write_bytes(payload)
    verify_files(raw, source)


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["text", "category"]:
            raise ValueError(f"Unexpected CSV schema: {path.name}")
        for index, row in enumerate(reader):
            if set(row) != {"text", "category"} or not row["text"].strip():
                raise ValueError(f"Malformed row {index} in {path.name}")
            yield index, row


def load_train(raw, labels):
    rows = []
    for index, row in csv_rows(raw / "train.csv"):
        if row["category"] not in labels:
            raise ValueError(f"Unknown training label at row {index}")
        rows.append({"id": f"train:{index:05d}", "text": row["text"],
                     "label": row["category"], "key": text_key(row["text"])})
    return rows


def scan_sealed_test(raw):
    # The label field is intentionally never accessed or retained.
    return [{"id": f"test:{i:05d}", "key": text_key(row["text"])}
            for i, row in csv_rows(raw / "test.csv")]


def clean_train(rows, sealed_test):
    test_keys = {r["key"] for r in sealed_test}
    groups = defaultdict(list)
    for row in rows:
        groups[row["key"]].append(row)
    retained, exclusions = [], []
    for key, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda r: r["id"])
        if key in test_keys:
            reason, removed = "matches_official_test", ordered
        elif len({r["label"] for r in ordered}) > 1:
            reason, removed = "conflicting_training_labels", ordered
        else:
            retained.append(ordered[0])
            reason, removed = "duplicate_training_text", ordered[1:]
        exclusions.extend({"id": r["id"], "key": key, "reason": reason} for r in removed)
    return sorted(retained, key=lambda r: r["id"]), exclusions


def ranked(rows, seed, namespace):
    """Stable seeded permutation: sort SHA-256([namespace, seed, row ID]).

    Independent of input ordering and Python/NumPy random-number implementations.
    """
    return sorted(rows, key=lambda r: (sha256(
        json.dumps([namespace, seed, r["id"]], separators=(",", ":")).encode()
    ), r["id"]))


def validation_split(rows, labels, per_class=VALIDATION_PER_CLASS, min_shots=5):
    pool, validation = [], []
    for label in labels:
        group = ranked([r for r in rows if r["label"] == label], SPLIT_SEED, "validation")
        if len(group) < per_class + min_shots:
            raise ValueError(f"Insufficient unique training examples for {label}: {len(group)}")
        validation.extend(group[:per_class])
        pool.extend(group[per_class:])
    return pool, validation


def few_shot(pool, labels, shots, seed):
    if shots not in SHOTS or seed not in SEEDS:
        raise ValueError("Use only the predeclared shot counts and seeds")
    selected = []
    for label in labels:
        group = ranked([r for r in pool if r["label"] == label], seed, "training")
        if len(group) < shots:
            raise ValueError(f"Insufficient examples for {label}")
        selected.extend(group[:shots])
    return selected


def verify_isolation(train, validation, sealed_test):
    splits = (train, validation, sealed_test)
    for rows in splits:
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Repeated row ID within split")
    for rows in (train, validation):
        if len({r["key"] for r in rows}) != len(rows):
            raise ValueError("Repeated normalized text within development split")
    for left, right in ((0, 1), (0, 2), (1, 2)):
        for field in ("id", "key"):
            if {r[field] for r in splits[left]} & {r[field] for r in splits[right]}:
                raise ValueError(f"Split leakage detected by {field}")


def prepare(raw, manifest_path):
    source = source_spec()
    download(raw, source)
    labels = read_json(raw / "categories.json")
    if len(labels) != source["expected_counts"]["classes"] or len(set(labels)) != len(labels):
        raise ValueError("Invalid label mapping")
    train = load_train(raw, labels)
    sealed_test = scan_sealed_test(raw)
    if len(train) != source["expected_counts"]["train"] or len(sealed_test) != source["expected_counts"]["test"]:
        raise ValueError("Dataset row counts differ from pinned source")
    cleaned, exclusions = clean_train(train, sealed_test)
    pool, validation = validation_split(cleaned, labels)
    verify_isolation(pool, validation, sealed_test)
    available = Counter(r["label"] for r in pool)
    infeasible = {str(n): {label: available[label] for label in labels if available[label] < n}
                  for n in SHOTS if any(available[label] < n for label in labels)}
    manifest = {
        "schema_version": 1, "source": source, "labels": labels,
        "split_seed": SPLIT_SEED, "validation_per_class": VALIDATION_PER_CLASS,
        "sampling_algorithm": "SHA256 compact JSON [namespace, seed, row_id]; ascending hex then row_id",
        "normalization": "NFKC, lowercase, collapse whitespace; SHA256 UTF-8; audit only",
        "pool_ids": [r["id"] for r in pool],
        "validation_ids": [r["id"] for r in validation],
        "sealed_test": sealed_test, "exclusions": exclusions,
        "audit": {
            "original_train": len(train), "cleaned_train": len(cleaned),
            "pool": len(pool), "validation": len(validation), "official_test": len(sealed_test),
            "exclusions_by_reason": dict(Counter(r["reason"] for r in exclusions)),
            "available_training_per_class": dict(available),
            "feasible_shots": [n for n in SHOTS if str(n) not in infeasible],
            "infeasible_shots_available_counts": infeasible,
            "test_access": "mechanical schema/count/text-hash audit only; labels unused; no evaluation",
            "near_duplicates_checked": False,
        },
    }
    payload = json_bytes(manifest)
    if manifest_path.exists() and manifest_path.read_bytes() != payload:
        raise ValueError("Existing split manifest differs; refusing to overwrite")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(payload)
    return manifest


def load_development(raw, manifest_path):
    """Offline training path. No parsing of official test texts or labels."""
    manifest = read_json(manifest_path)
    source = source_spec()
    if manifest["source"] != source:
        raise ValueError("Manifest source differs from pinned source")
    verify_files(raw, source)  # Includes byte integrity of the sealed test file.
    labels = read_json(raw / "categories.json")
    if manifest["labels"] != labels:
        raise ValueError("Label mapping changed")
    rows = load_train(raw, labels)
    cleaned, exclusions = clean_train(rows, manifest["sealed_test"])
    pool, validation = validation_split(cleaned, labels)
    if (manifest["exclusions"] != exclusions
            or manifest["pool_ids"] != [r["id"] for r in pool]
            or manifest["validation_ids"] != [r["id"] for r in validation]):
        raise ValueError("Split manifest does not match the deterministic protocol")
    verify_isolation(pool, validation, manifest["sealed_test"])
    return manifest, pool, validation
