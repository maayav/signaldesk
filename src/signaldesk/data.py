"""Download and audit the immutable, attributed BANKING77 benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import unicodedata
from urllib.request import urlopen

import numpy as np
from sklearn.model_selection import train_test_split

SOURCE_COMMIT = "57ec275d8078af65b7731c2a98be812d844a6d6b"
SOURCE_ROOT = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{SOURCE_COMMIT}"
SOURCE_URL = "https://github.com/PolyAI-LDN/task-specific-datasets/tree/" + SOURCE_COMMIT
FILES = {
    "train.csv": ("banking_data/train.csv", "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b"),
    "test.csv": ("banking_data/test.csv", "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d"),
    "LICENSE": ("LICENSE", "7e7170e3cebf88a9f60c7b8421418323c09304da1af4d5e90f4da1dc1c8a2661"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_data(data_dir: Path) -> dict:
    """Verify cached bytes; download only missing files and verify before replacing."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename, (relative, expected) in FILES.items():
        destination = data_dir / filename
        if destination.exists():
            if sha256(destination) != expected:
                raise ValueError(f"Checksum mismatch: {destination}. Remove it explicitly to download again.")
            continue
        with urlopen(f"{SOURCE_ROOT}/{relative}", timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"Upstream checksum mismatch for {filename}; no unverified data saved.")
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(destination)
    manifest = {
        "dataset": "BANKING77",
        "source_url": SOURCE_URL,
        "source_commit": SOURCE_COMMIT,
        "license": "CC BY 4.0",
        "attribution": "Casanueva et al. (2020), Efficient Intent Detection with Dual Sentence Encoders; PolyAI.",
        "paper": "https://aclanthology.org/2020.nlp4convai-1.5/",
        "files": {name: {"sha256": digest, "url": f"{SOURCE_ROOT}/{relative}"} for name, (relative, digest) in FILES.items()},
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def read_rows(path: Path, verify: bool = True) -> list[dict[str, str]]:
    if verify and path.name in FILES and sha256(path) != FILES[path.name][1]:
        raise ValueError(f"Checksum mismatch for {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["text", "category"]:
            raise ValueError("Dataset requires exactly the columns text,category")
        rows = list(reader)
    if not rows or any(not row["text"].strip() or not row["category"].strip() for row in rows):
        raise ValueError("Dataset must contain non-empty text and category values")
    return rows


def normalized_text(text: str) -> str:
    """Conservative exact-match normalization, not a claim of semantic deduplication."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def clean_training_rows(train: list[dict], test: list[dict]) -> tuple[list[dict], dict]:
    """Keep the official test intact and exclude its exact normalized texts from fit data."""
    test_keys = {normalized_text(row["text"]) for row in test}
    grouped: dict[str, list[dict]] = {}
    for row in train:
        grouped.setdefault(normalized_text(row["text"]), []).append(row)
    cleaned, overlap, duplicates, conflicts = [], 0, 0, 0
    for key, rows in grouped.items():
        if key in test_keys:
            overlap += len(rows)
        elif len({row["category"] for row in rows}) > 1:
            conflicts += len(rows)
        else:
            cleaned.append(rows[0])
            duplicates += len(rows) - 1
    return cleaned, {
        "normalization": "Unicode NFKC + casefold + whitespace collapse",
        "train_rows_removed_for_test_text_overlap": overlap,
        "train_duplicate_rows_removed": duplicates,
        "train_conflicting_label_rows_removed": conflicts,
        "test_duplicate_rows_preserved": len(test) - len(test_keys),
        "remaining_train_rows": len(cleaned),
        "test_unchanged": True,
        "limitation": "Only exact normalized duplicates are removed; paraphrases can remain.",
    }


def split_training_rows(rows: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    """Disjoint, stratified 55/15/15/15 fit/calibration/selection/policy partitions."""
    indices = np.arange(len(rows))
    labels = np.asarray([row["category"] for row in rows])
    fit, rest = train_test_split(indices, test_size=0.45, stratify=labels, random_state=seed)
    calibration, remainder = train_test_split(rest, test_size=2 / 3, stratify=labels[rest], random_state=seed + 1)
    selection, policy = train_test_split(remainder, test_size=0.5, stratify=labels[remainder], random_state=seed + 2)
    partitions = {"fit": fit, "calibration": calibration, "selection": selection, "policy_validation": policy}
    result = {name: [rows[int(i)] for i in subset] for name, subset in partitions.items()}
    key_sets = [{normalized_text(row["text"]) for row in partition} for partition in result.values()]
    if sum(len(keys) for keys in key_sets) != len(set.union(*key_sets)):
        raise ValueError("Duplicate normalized text crosses a training partition")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    manifest = fetch_data(args.data_dir)
    print(f"Verified BANKING77 at {args.data_dir} ({manifest['source_commit']})")


if __name__ == "__main__":
    main()
