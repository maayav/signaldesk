# Contributing to SignalDesk

Small, well-scoped contributions are welcome: reproducibility fixes, clear error handling, useful tests, documentation corrections, and model experiments with a stated evaluation protocol.

## Set up

Use Python 3.12 and run these commands from your clone:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
PIP_USER=0 python -m pip install -r requirements-lock.txt
PIP_USER=0 python -m pip install --no-deps -e .
```

Run the included model with `bash scripts/serve.sh`. Read the [architecture](docs/architecture.md) and [model card](docs/model-card.md) before changing the training or serving path.

## Before opening a pull request

```bash
python -m ruff check src app tests scripts
python -m pytest -q
```

Describe the concrete problem, the resulting behavior, and how you verified the change. Include a small reproduction or screenshot when it helps a reviewer inspect a bug or UI change. Keep unrelated cleanup separate.

The repository’s tests exercise real shipped artifacts as well as unit-level behavior. If you change artifact-dependent behavior, state whether you tested the included model or a newly trained one. Do not claim a test or experiment was run if it was not.

## Model and data changes

- Keep fit, calibration, selection, and policy-validation roles separate. Select model changes with development data; reserve test labels for a declared final evaluation.
- Preserve pinned dataset provenance, source hashes, normalized duplicate checks, and attribution. Do not edit source CSVs to improve reported scores.
- Report coverage alongside routed-query accuracy. Record calibration metrics and common errors rather than reporting only a favorable headline metric.
- Publish an artifact change with its matching metrics, partition record, and manifest. Explain changed seeds, dependency versions, or training settings.
- Keep proposed experiments reproducible with exact commands and configurations. The [learning roadmap](docs/learning-roadmap.md) suggests useful comparisons.

## Repository hygiene

Do not commit credentials, `.env` files, private customer messages, local SQLite databases, virtual environments, caches, or build outputs. Use synthetic examples when reproducing a user-input issue. Original public BANKING77 examples remain subject to the dataset attribution and license.

The included joblib file is a trusted project artifact, not an upload format. Do not introduce code that deserializes arbitrary user-supplied model files. See [SECURITY.md](SECURITY.md) for reporting potential security issues.

## Issues and pull requests

Use the bug form for reproducible failures and the feature form for proposed behavior. For model improvements, include the hypothesis, evaluation partitions, and metrics you would compare. Check existing issues first and keep discussion focused on evidence and the intended behavior.

Contributions are provided under the repository’s MIT code license unless a file has a separate stated license. Preserve all third-party notices and dataset licensing.
