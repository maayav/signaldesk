# Learning and extension roadmap

Work through these in order. Each stage should produce evidence you can explain, rather than another unsupported resume claim.

## 1. Reproduce the system

Create a fresh Python environment, download the pinned data, rerun training, and inspect the generated metrics and manifest. Trace one prediction from request validation through text features, class scores, calibrated probabilities, routing policy, and the event store.

Deliverable: a short notebook or experiment note containing the command, model version, split sizes, and measured test results.

## 2. Understand the experiment

Read the data, training, and policy modules. Explain why the vectorizer is fitted only on fit rows, why calibration labels are separate, why selection and threshold labels have different roles, and why the official test labels are held out.

Plot or inspect the most confused intent pairs and read their source examples. Choose a plausible explanation supported by those examples; do not assume every error is a model defect rather than a boundary or labeling issue.

Deliverable: three error analyses with concrete examples and a proposed improvement.

## 3. Run an ablation

Compare the shipped feature union with word-only TF-IDF using identical partitions and training settings. Record training time, model size, macro F1, calibration metrics, and review coverage. Select changes using validation data. Keep the official test for a predeclared final comparison rather than repeatedly tuning against it.

Deliverable: an experiment table, exact configuration, and a conclusion supported by the results.

## 4. Evaluate domain mismatch

Create a separate, clearly documented set of messages outside the banking taxonomy, with enough variety to assess false confident routing. Keep it out of model and threshold tuning. Compare confidence-only review with an explicit out-of-domain detector or retrieval-similarity signal, including the effect on valid in-domain messages.

Deliverable: a labeled out-of-domain evaluation protocol with false-route and in-domain coverage results. Do not equate a few hand-written examples with a representative benchmark.

## 5. Compare a modern text encoder

Add a sentence-transformer representation or a small fine-tuned encoder. Use the same partition roles, declare the pretrained source/version, and report preprocessing, training cost, artifact size, serving latency, and classification quality. A larger model is useful only if the measured tradeoff supports it.

Deliverable: a reproducible model comparison and a decision about which model to serve.

## 6. Build a reviewed release loop

Design a data version for collected feedback. Distinguish prediction mistakes, ambiguous messages, and new intents. Account for the fact that users tend to correct unusual cases. Train a candidate offline, compare it with the existing version, and release the model, metrics, and manifest together.

Deliverable: a reviewed candidate-release workflow with rollback instructions. Add external hosting, authentication, and load tests only when you can verify the resulting deployment.
