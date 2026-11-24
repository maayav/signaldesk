# Interview guide

Use this guide to understand and demonstrate the implementation. Read the corresponding modules, rerun training, and make an experiment you can explain before putting the project on your resume.

## A 60-second explanation

“SignalDesk classifies short support messages into 77 banking intents. I built an end-to-end workflow around a public benchmark: verified data, leakage checks, TF-IDF feature engineering, two linear classifiers, independent probability calibration, and a confidence-based review policy. A FastAPI service exposes the selected model, and a Streamlit app lets someone inspect predictions, run CSV batches, and view evaluation and confidence drift. The official test set is held out from all model and threshold selection.”

Add the actual macro F1, routing coverage, and accepted-ticket accuracy from the shipped report. State that these are benchmark results. Describe your own contributions and assistance accurately.

## Why did you choose this problem?

Fine-grained intent classification is understandable to an interviewer and has a clear operational decision: route a ticket or request review. It also exposes engineering questions beyond fitting a classifier, including probability calibration, evaluation leakage, serving, observability, and artifact provenance.

## Why TF-IDF and linear classifiers?

They are strong, inexpensive baselines for short-text classification and fit a local CPU workflow. Word n-grams capture phrases; character n-grams provide partial resilience to misspellings and morphological variations. Sparse representations avoid sending text to a remote API. This implementation does not establish that a linear model beats a transformer; that requires an experiment under the same splits.

Find `build_vectorizer` and candidate construction in the training/modeling modules and explain each configured parameter.

## Why separate fit, calibration, selection, and policy sets?

Each step consumes labels. Fitting establishes weights; calibration maps scores to probabilities; selection chooses a candidate; policy validation chooses a decision threshold. Reusing the test set for any of those choices would make reported test performance optimistic. The separation trades some fitting data for a cleaner estimate of the final pipeline. No final refit is performed because it would change the classifier underlying the calibrated probabilities and threshold.

## What does calibration do?

An SVM margin is not a probability. Sigmoid calibration learns a mapping on independent examples. The service can then use a common probability-like scale for the review policy. Calibration is still imperfect, can shift under new traffic, and should be assessed with log loss and reliability diagnostics rather than assumed correct.

## What exactly does the review threshold optimize?

It seeks the lowest confidence threshold with at least 100 accepted policy examples and at least 95% empirical accepted-example accuracy. Lowering a threshold generally routes more tickets and increases the review-policy coverage. The code handles confidence ties as a group. If no threshold meets the constraints, it reviews all messages.

The 95% number is a validation target. It is not a future-data guarantee. Explain the measured test selective accuracy and coverage together, and say what a real support team would need to choose its own operating point.

## How did you prevent leakage?

The vectorizer and retrieval index are fit-only. Normalized exact-text groups are audited before internal splitting; training examples overlapping official test text are removed. Selection and threshold choice have separate label partitions. The official test labels are reserved for the final report. These checks address exact duplicates, not semantic paraphrases or unknown dataset construction biases.

Open the recorded split sizes and overlap audit in the artifact files while explaining this.

## What does a similar example prove?

It shows which labeled fitting examples have high word-TF-IDF cosine similarity to the input. It helps inspect local behavior, but it does not verify the predicted intent. It is lexical retrieval, not a sentence-embedding search system or a generated answer backed by sources.

## What do feature contributions explain?

They multiply active features by a class’s raw linear coefficients. Positive contributions raise the selected class’s score. They do not decompose the calibrated probability, prove causality, or guarantee that the classifier is using the right concepts. Character fragments can be harder to interpret than complete word features.

## Does confidence drift mean accuracy fell?

No. The monitor compares a recent confidence histogram to a fixed policy-validation reference with Jensen–Shannon divergence. It can flag a changed workload, a new vocabulary, or an issue worth inspecting. Accuracy requires representative labels. A stable confidence distribution can coexist with incorrect predictions, and a shifted distribution can coexist with stable accuracy.

## How are feedback and retraining handled?

Feedback attaches a corrected intent to a prediction ID. Serving does not learn online. A retraining extension would validate feedback quality, account for selective response bias, create a reviewed dataset version, train and evaluate a candidate, then release an immutable artifact. A human correction should not silently modify the live model.

## What would change for production scale?

Start with representative operational data and a real review-cost objective. Add authentication, request limits, HTTPS, artifact release controls, structured monitoring, and a shared database when running multiple API instances. Benchmark load and tail latency before choosing replica counts. The current project demonstrates local deployment; it has not demonstrated enterprise scale.

## Questions to prepare with evidence

1. Which two intents are most often confused, and what examples explain that?
2. Did character features help? Run a word-only ablation before claiming they did.
3. What happens when the routing threshold rises? Show coverage and accepted accuracy together.
4. How does an unrelated message behave? Show examples without claiming they measure out-of-domain accuracy.
5. Can another person reproduce your model? Point to the pinned data, seed, versions, report, and tests.
6. What did you change after studying the initial implementation? Keep a short experiment log with commands and results.
