# Verification record

## Prepublication verification — 24 November 2026

The shipped model is `banking77-c9813bba904c`. A fresh seed-42 rebuild produced exactly the same predictions and probabilities on all 3,080 official test queries as the original model. The version changed after the artifact-integrity fix; the benchmark metrics and routing threshold are unchanged.

### Automated checks

- `python -m pytest -q`: **63 passed in 6.90 seconds**. This includes the original 30 tests, 23 dashboard regressions against the real API, 8 artifact-integrity regressions, and 2 launcher regressions.
- `python -m ruff check src app tests scripts`: passed.
- `python -m pip check`: passed.
- Every code cell in `notebooks/01_data_and_evaluation.ipynb` executed successfully against the final artifact.
- The core HTTP checker completed 33 checks against the actual API.
- Artifact loading verifies every manifest-listed file and rejects malformed manifests before deserialization. Dashboard metadata caching is keyed by model version.

### Final Docker stack

The final Docker image built successfully. The API became healthy and the dashboard health endpoint responded. All six groups in `scripts/verify_stack.py` passed against the final model version:

1. Readiness, model metadata, and metrics.
2. Input rejection and atomic batch validation.
3. Single-ticket route/review decisions and prediction evidence.
4. Sample-CSV batch agreement.
5. Feedback, event counts, and monitoring without raw ticket text.
6. Dashboard HTTP availability.

Restarting the Docker API retained **20 predictions and 2 feedback records**, confirming persistence through the configured event volume.

### Browser and launcher checks

The real Docker dashboard was exercised in a browser across all four pages:

- Workspace: a single prediction rendered and corrected-intent feedback saved successfully.
- Batch lab: an actual file chooser uploaded the eight-row sample CSV; results contained seven route recommendations and one review recommendation. The downloaded CSV was checked for eight rows, the final model version, and the review count. The review-only toggle showed the expected single row.
- Evaluation: the displayed rounded results matched the artifact: 88.8% overall accuracy, 96.6% top-3 accuracy, 83.9% routing coverage, and 95.2% routed-query accuracy.
- Monitor: charts and recorded prediction events rendered.

The Bash launcher started the complete local stack on custom ports 18002/18502 and shut it down cleanly. Regression tests verified fail-fast behavior for occupied ports and identical API/UI ports.

These are completed local prepublication checks. Remote CI execution is tracked separately by the [Verify SignalDesk workflow](https://github.com/maayav/signaldesk/actions/workflows/ci.yml); its live status is shown in the README badge. No hosted application URL or screenshot is claimed by this record.

## Historical verification — 16 November 2026

Verified locally on 16 November 2026 with Python 3.12.3, scikit-learn 1.7.2, and the exact dependencies in `requirements-lock.txt`.

### Automated checks

- `python -m pytest -q`: **30 passed**. Includes the actual shipped model, class/probability alignment, exact reconciliation of all official test metrics, single/batch/direct inference agreement, input validation, duplicate leakage checks, threshold tie handling, corrupted artifact rejection, and text-free SQLite persistence.
- `python -m ruff check src app tests`: passed.
- Every code cell in `notebooks/01_data_and_evaluation.ipynb` executed successfully. Plot windows were suppressed during this execution check.
- Streamlit AppTest exercised all four pages against the real API, including single prediction, reviewer feedback, a two-row uploaded CSV, and the review-only filter.

The test run emits one upstream deprecation warning from Starlette's TestClient referencing an AnyIO alias; no test failed.

### Service and container checks

- Live local FastAPI readiness, prediction, feedback, and monitoring endpoints responded successfully.
- Browser inspection confirmed the Workspace prediction, Evaluation results, and Monitor charts rendered with the trained model.
- `docker compose config --quiet` passed.
- The Docker image built successfully and both Compose services started as a non-root user. The API became healthy, and the Streamlit health endpoint responded.
- A prediction initiated inside the dashboard container reached the API container over the Compose service network.
- The local process and Docker API returned exactly equal intent, confidence, alternatives, neighbors, contributions, routing decision, and model version for the same message.
- Temporary QA containers were stopped after verification. The local demo can be run using the README commands.

### Measured model

Model `banking77-6cac29686c26` uses 18,178 word/character features and a calibrated one-versus-rest logistic classifier. The official test set contains 3,080 queries across 77 intents.

| Measurement | Result |
| --- | ---: |
| Whole-test accuracy | 88.7987% |
| Macro F1 | 0.8876857 |
| Top-3 accuracy | 96.5584% |
| Routed queries | 2,585 (83.9286%) |
| Routed-query accuracy | 95.2031% |
| Queries sent for review | 495 |
| Top-label ECE, 10 bins | 0.1700144 |

These checks describe the local system as of 16 November 2026. GitHub Actions had not run on a remote repository at that time. For current remote execution status, see the [Verify SignalDesk workflow](https://github.com/maayav/signaldesk/actions/workflows/ci.yml). The application has not been published to a hosting service. See the model card for data and confidence limitations.
