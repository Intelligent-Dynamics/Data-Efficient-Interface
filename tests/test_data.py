from collections import Counter

import pytest

from baseline.data import (SEEDS, SHOTS, clean_train, download, few_shot,
                           scan_sealed_test, sha256, text_key, validation_split,
                           verify_isolation)


def row(row_id, text, label="a"):
    return {"id": row_id, "text": text, "label": label, "key": text_key(text)}


@pytest.fixture
def synthetic_data():
    labels = [f"intent_{i:02d}" for i in range(77)]
    rows = [row(f"train:{i * 80 + j:05d}", f"{label} unique request {j}", label)
            for i, label in enumerate(labels) for j in range(80)]
    return labels, rows


@pytest.mark.parametrize("seed", SEEDS)
def test_exact_budgets_determinism_and_nested_samples(synthetic_data, seed):
    labels, rows = synthetic_data
    pool, validation = validation_split(rows, labels)
    assert Counter(r["label"] for r in validation) == {label: 20 for label in labels}
    assert validation_split(list(reversed(rows)), labels) == (pool, validation)
    previous = set()
    for shots in SHOTS:
        sample = few_shot(pool, labels, shots, seed)
        assert Counter(r["label"] for r in sample) == {label: shots for label in labels}
        assert few_shot(list(reversed(pool)), labels, shots, seed) == sample
        ids = {r["id"] for r in sample}
        assert previous <= ids
        previous = ids
        verify_isolation(sample, validation, [row("test:00000", "held out test text")])
    assert few_shot(pool, labels, 5, 11) != few_shot(pool, labels, 5, 22)


def test_short_class_fails_instead_of_oversampling():
    rows = [row(f"train:{i:05d}", f"request {i}") for i in range(24)]
    with pytest.raises(ValueError, match="Insufficient"):
        validation_split(rows, ["a"])


def test_smallest_class_allows_five_and_ten_but_rejects_twenty_and_fifty():
    rows = [row(f"train:{i:05d}", f"request {i}") for i in range(35)]
    pool, validation = validation_split(rows, ["a"])
    assert len(validation) == 20
    assert len(few_shot(pool, ["a"], 5, 11)) == 5
    assert len(few_shot(pool, ["a"], 10, 11)) == 10
    for shots in (20, 50):
        with pytest.raises(ValueError, match="Insufficient"):
            few_shot(pool, ["a"], shots, 11)


def test_duplicates_conflicts_and_test_overlap():
    rows = [row("train:00000", "  ＣＡＲＤ\tarrival  "),
            row("train:00001", "card arrival"),
            row("train:00002", "conflicting text", "a"),
            row("train:00003", "CONFLICTING text", "b"),
            row("train:00004", "matches test"),
            row("train:00005", "unique training text")]
    sealed = [row("test:00000", "  MATCHES   test")]
    clean, exclusions = clean_train(list(reversed(rows)), sealed)
    assert [r["id"] for r in clean] == ["train:00000", "train:00005"]
    assert Counter(r["reason"] for r in exclusions) == {
        "duplicate_training_text": 1, "conflicting_training_labels": 2,
        "matches_official_test": 1,
    }
    verify_isolation(clean, [], sealed)


@pytest.mark.parametrize("field", ["id", "key"])
@pytest.mark.parametrize("pair", [(0, 1), (0, 2), (1, 2)])
def test_isolation_rejects_ids_and_normalized_text_across_every_pair(field, pair):
    splits = [[row("train:00000", "one")], [row("train:00001", "two")],
              [row("test:00000", "three")]]
    splits[pair[1]][0][field] = splits[pair[0]][0][field]
    with pytest.raises(ValueError, match="leakage"):
        verify_isolation(*splits)


def test_repeated_ids_rejected_within_split():
    with pytest.raises(ValueError, match="Repeated row ID"):
        verify_isolation([row("train:0", "one"), row("train:0", "two")], [], [])


def test_test_audit_does_not_expose_labels_or_text(tmp_path):
    path = tmp_path / "test.csv"
    original = b'text,category\nsealed text,not-even-a-known-label\n'
    path.write_bytes(original)
    assert scan_sealed_test(tmp_path) == [{"id": "test:00000", "key": text_key("sealed text")}]
    assert path.read_bytes() == original


def test_download_is_offline_when_cached_and_rejects_corruption(tmp_path, monkeypatch):
    def fail_network(*args, **kwargs):
        pytest.fail("Cached file must not require network")

    monkeypatch.setattr("baseline.data.urlopen", fail_network)
    path = tmp_path / "train.csv"
    path.write_bytes(b"original")
    source = {"files": {"train.csv": {"url": "https://unused.example", "sha256": sha256(b"original")}}}
    download(tmp_path, source)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download(tmp_path, source)
