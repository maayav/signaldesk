# Resume and portfolio wording

Use these after you can run the project, explain the evaluation protocol, and discuss changes you made. Describe your own contribution and any assistance accurately. Do not present a public benchmark as company customer data or claim business impact that was not measured.

## Project entry

**SignalDesk — Support Ticket Triage and ML Serving**

Python · scikit-learn · FastAPI · Streamlit · SQLite · Docker · GitHub Actions

- Built a reproducible text-classification pipeline for 77 support intents using word/character TF-IDF, linear classifier comparison, independent probability calibration, and a validation-tuned human-review policy.
- Served versioned predictions through FastAPI and an interactive Streamlit app with batch inference, labeled-example retrieval, feature contributions, and prediction feedback.
- Added dataset hash verification, exact-duplicate leakage checks, held-out evaluation, automated tests, and confidence-distribution monitoring without storing submitted ticket text.

The shipped model, `banking77-c9813bba904c`, supports this measured-result bullet:

- Achieved **0.8877 macro F1 and 88.80% accuracy** on BANKING77’s 3,080 official test queries; a validation-tuned review policy routed **83.93%** of queries with **95.20% accuracy among routed queries**.

These values come from `artifacts/metrics.json`: 2,585 routed and 495 reviewed queries. Keep coverage and routed accuracy together. If you modify the data or model, rerun evaluation and update the figures before publishing.

## Compact portfolio description

SignalDesk is a locally deployable ML system for fine-grained support-ticket triage. It combines reproducible data processing, calibrated intent classification, a confidence-based review policy, an inference API, and an operational dashboard. Its report separates overall classification quality from routing coverage and accepted-ticket accuracy.

## Claims the project supports

- A public benchmark with 77 intents and a reproducible experiment.
- A working local API and dashboard, packaged for Docker.
- Separate model-selection and policy-validation partitions.
- Measured benchmark results saved with artifact provenance.
- Local monitoring and a feedback data path.

## Claims that need additional evidence

Production customer adoption, reduced support costs, live processing volume, uptime guarantees, performance improvements from unrun ablations, transformer superiority, and guaranteed out-of-domain detection. The validation policy’s 95% target is not a blanket future-data accuracy claim.

## Before adding it to an application

1. Run setup, training, the dashboard, and the tests yourself.
2. Read `data.py`, `train.py`, `policy.py`, `modeling.py`, and the serving/monitoring modules.
3. Run an experiment, such as removing character features, and record the result.
4. Confirm the measured-result bullet matches your exact artifact.
5. Link the [project repository](https://github.com/maayav/signaldesk) and add a hosted demo only if you have deployed and verified one. Keep secrets and local event databases out of the repository.
