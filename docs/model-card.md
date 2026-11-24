# Model card

## Purpose

SignalDesk predicts one of 77 English banking-support intents and uses a confidence threshold to decide whether to route a message or request human review. It is a reproducible portfolio benchmark and local demonstration. It does not perform banking actions or generate customer advice.

## Data and provenance

[BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data) contains 10,003 official training queries and 3,080 official test queries. The release is pinned to upstream commit `57ec275d8078af65b7731c2a98be812d844a6d6b`; the downloader validates SHA-256 digests before accepting the original bytes. The files have `text` and `category` columns.

The source is licensed CC BY 4.0. Attribution and the paper citation are in [ATTRIBUTION.md](ATTRIBUTION.md). The dataset’s public benchmark setting is different from a continuously changing customer-support queue.

## Partition and leakage protocol

Normalized exact text uses Unicode NFKC normalization, case folding, and collapsed whitespace. Before splitting, the pipeline removes official-training rows whose normalized text occurs in the official test file, removes conflicting-label duplicate groups, and collapses remaining same-label duplicates. The official test file is preserved. Test text is used for overlap detection; test labels do not influence fitting, candidate selection, calibration, or policy selection.

The cleaned official training set is stratified by intent with seed 42:

| Partition | Approximate share | Used for |
| --- | --- | --- |
| Fit | 55% | TF-IDF vocabulary/IDF, classifier fitting, retrieval reference. |
| Calibration | 15% | Sigmoid probability calibration for each candidate. |
| Selection | 15% | Candidate choice by macro F1. |
| Policy | 15% | Routing threshold and monitoring reference distribution. |
| Official test | Separate release | Final evaluation after candidate and threshold are fixed. |

The shipped run removed 7 official-training rows overlapping test text and 4 duplicate training rows; no conflicting-label groups were present. It used 5,495 fit rows and 1,499 rows in each of the calibration, selection, and policy partitions. All 3,080 test rows were retained, including one normalized duplicate. The exact audit is recorded in `artifacts/metrics.json`, and row identities are in `artifacts/splits.json`. There is no refit on calibration, selection, or policy rows after choosing the winner. This keeps calibration and the routing policy aligned with the exact classifier being served. Normalized exact-text checks do not detect paraphrase leakage or other near-duplicates.

## Features and candidate models

Both candidates use the same fit-only feature union:

- Word TF-IDF: word unigrams and bigrams, `min_df=2`, at most 25,000 features.
- Character TF-IDF: `char_wb` 3–5 grams, `min_df=2`, at most 40,000 features.
- Feature-union weights: words `1.0`, characters `0.7`.

The candidates are `LinearSVC(C=1.5)` and one-versus-rest `LogisticRegression(C=4, solver="liblinear", max_iter=600)`. Each is calibrated with sigmoid calibration on the independent calibration partition through scikit-learn’s `FrozenEstimator`. Selection uses macro F1 on the selection partition. Inspect the metrics report for the selected model and both candidate scores.

Word features capture recurring phrases, while character features help with spelling variants and word fragments. The linear models make CPU training and local inspection practical. This project has not established that they outperform modern sentence encoders or transformers.

## Routing policy

A message receives a route recommendation if its maximum calibrated class probability is at least the selected threshold and its transformed feature vector is nonempty. Otherwise it receives a review recommendation. Inputs with no known features always go to review.

On the separate policy partition, the algorithm considers unique confidence values and chooses the lowest threshold that accepts at least 100 examples with empirical accepted-example accuracy of at least 95%. All examples tied at a threshold are accepted together. If no threshold satisfies the constraints, a threshold above 1 sends all examples to review.

The 95% setting is a tuning target on that partition. It is not a statistical lower bound or a guarantee about the test set or future traffic. Report test selective accuracy together with coverage and accepted counts. The threshold is selected once before test evaluation.

## Measured results

Model version `banking77-c9813bba904c` selected one-versus-rest logistic regression. Selection macro F1 was 0.872966 versus 0.870340 for the linear SVM.

| Final official-test metric | Result |
| --- | ---: |
| Queries | 3,080 |
| Accuracy | 0.887987 |
| Macro F1 | 0.887686 |
| Top-3 accuracy | 0.965584 |
| Log loss | 0.543579 |
| Expected calibration error | 0.170014 |
| Policy coverage | 0.839286 |
| Accuracy among routed queries | 0.952031 |
| Routed / reviewed | 2,585 / 495 |

The routing threshold is 0.5126584306. On policy-validation data it accepted 1,249 of 1,499 queries with 0.950360 selective accuracy. Those observations determined the threshold; the table above reports the subsequent held-out test result.

The reliability bins show substantial underconfidence: for example, queries with confidence in [0.7, 0.8) had mean confidence 0.7537 and accuracy 0.9622. The system includes calibration, but this run does not compare calibrated and uncalibrated probabilities, so it does not establish that calibration improved the scores. The ECE and reliability table should stay visible alongside headline accuracy.

The resulting feature union has 18,178 features. The saved model occupies 9,507,530 bytes. Training took approximately 5.0 seconds in the recorded local run; this is a host-specific observation, not a portable performance benchmark.

## Evaluation and artifacts

`artifacts/metrics.json` is the authoritative machine-readable report. It includes overall accuracy, macro and weighted F1, top-3 accuracy, multiclass log loss, expected calibration error, a reliability table, per-intent metrics, common confusion pairs, candidate scores, and selective-routing results. The Streamlit Evaluation page displays this report.

Macro F1 treats intents equally; overall accuracy weights individual examples. Top-3 accuracy measures whether the true intent occurs among three suggestions. Calibration error summarizes confidence bins and depends on the binning choice. Confusion pairs help identify labels needing clearer boundaries or better examples.

The saved `model.joblib` contains the feature pipeline, selected calibrated classifier, raw classifier, routing policy, and fit-only retrieval examples. `manifest.json` records provenance and versions. The loader verifies every manifest-listed file and rejects malformed manifests before deserializing the model. Only load joblib files from a trusted source, because deserialization can execute code; matching checksums do not establish trust if an attacker can replace both the files and the manifest.

## Evidence shown with predictions

Similar examples come from the fit partition, ranked by cosine similarity in word TF-IDF space. They are examples of the model’s lexical neighborhood and do not validate the ticket’s true label. A low-similarity or unrelated query may still receive neighbors.

Feature contributions are nonzero TF-IDF values multiplied by the selected class’s raw linear coefficients. They describe contributions to a class margin. Calibration changes that margin into probabilities, so contribution values must not be interpreted as percentages or causal explanations.

## Monitoring and feedback

The service records prediction metadata in SQLite without storing submitted ticket text. The monitoring endpoint compares a recent prediction-confidence histogram with the fixed policy-partition reference using Jensen–Shannon divergence. It also reports volume, mean confidence, routing/review rates, and intent mix.

Confidence drift is an operational signal. It does not measure label drift, model accuracy, fairness, or a confirmed service failure. Small windows are marked insufficient; reference traffic and deployment traffic may differ even when prediction quality is stable.

User feedback records a corrected intent for a prediction. It does not update model weights. Feedback can be selective and should not be treated as an unbiased test set.

## Limitations and appropriate use

- Fixed English, banking-specific taxonomy; no trained catch-all intent.
- Confidence-based review is not a complete out-of-domain detection system.
- No claims of production deployment, real financial impact, or customer savings.
- No measurement of fairness across demographic groups, because this benchmark does not supply the required attributes.
- No evaluation of evolving intents, multilingual inputs, adversarial text, or long support threads.
- Exact-text deduplication does not remove all semantic overlap.

Before applying this design to a real queue, obtain representative labeled traffic, evaluate out-of-domain messages, set a review budget, and validate the routing policy against operational costs. Add authentication, transport security, retention controls, and deployment monitoring appropriate to the host environment.
