from pathlib import Path

import pytest

from signaldesk.data import clean_training_rows, normalized_text, read_rows, split_training_rows


def test_exact_duplicates_do_not_cross_partitions():
    rows = [{"text": f"class {label} example {index}", "category": label} for label in ("a", "b", "c") for index in range(40)]
    split = split_training_rows(rows, 42)
    repeated = split_training_rows(rows, 42)
    assert split == repeated
    all_texts = [row["text"] for partition in split.values() for row in partition]
    assert len(set(all_texts)) == len(rows)
    assert all(set(row["category"] for row in partition) == {"a", "b", "c"} for partition in split.values())


def test_cleaning_preserves_test_and_excludes_overlap_and_conflicts():
    test = [{"text": " ATM   LOST card ", "category": "card"}]
    original_test = [dict(row) for row in test]
    train = [
        {"text": "atm lost CARD", "category": "card"},
        {"text": "hello", "category": "one"},
        {"text": " HELLO ", "category": "one"},
        {"text": "conflict", "category": "one"},
        {"text": "CONFLICT", "category": "two"},
    ]
    cleaned, audit = clean_training_rows(train, test)
    assert cleaned == [{"text": "hello", "category": "one"}]
    assert test == original_test
    assert audit["train_rows_removed_for_test_text_overlap"] == 1
    assert audit["train_duplicate_rows_removed"] == 1
    assert audit["train_conflicting_label_rows_removed"] == 2
    assert normalized_text("ＦＯＯ \tBar") == "foo bar"


def test_read_rows_rejects_schema_and_empty_text(tmp_path: Path):
    source = tmp_path / "sample.csv"
    source.write_text("message,label\nhello,a\n")
    with pytest.raises(ValueError, match="columns"):
        read_rows(source, verify=False)
    source.write_text("text,category\n ,a\n")
    with pytest.raises(ValueError, match="non-empty"):
        read_rows(source, verify=False)
