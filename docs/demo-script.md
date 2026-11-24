# Five-minute demo

## Before the demo

Start the API and dashboard using the README. Confirm `GET /readyz` succeeds. Keep the interactive API documentation and `artifacts/metrics.json` available. Explain that the data is a public benchmark and that the app routes support intents; it does not access accounts or answer banking questions.

## 0:00–0:45 — Show the workflow

Open the Workspace and submit:

> The ATM kept my card and I need it back.

Explain the predicted intent, calibrated confidence, and route/review decision. Do not promise a specific confidence value; use what the running model returns.

## 0:45–1:30 — Inspect the evidence

Show the top intent candidates, similar labeled fit examples, and feature contributions. Explain that retrieval uses word TF-IDF cosine similarity and that contributions describe a raw linear class score. These are inspection tools rather than proof of correctness.

Try a less precise message:

> Something is wrong with my card.

Discuss ambiguity using the actual response. A low-confidence ticket is sent for review, but confidence alone cannot identify all incorrect predictions.

## 1:30–2:15 — Show a batch

Use Batch lab with a small CSV containing a `text` column:

```csv
text
My bank transfer is still pending
How can I change my PIN?
I think someone stole my card
Please help me book a flight
```

Run the batch and download the result. The last message illustrates a domain mismatch; one example is not an out-of-domain benchmark. Explain the batch-size limit and input validation.

## 2:15–3:15 — Show measured evaluation

Open Evaluation. Point out the selected model, both candidate scores, overall macro F1, and the confidence policy’s test coverage and selective accuracy. Show a common confusion pair and a calibration diagnostic.

Explain the four internal training partitions and the separate official test set. The threshold target is empirical on policy-validation data, while the test report measures what happened on held-out examples.

## 3:15–4:15 — Show the service and monitoring

Open API documentation and call the prediction endpoint. Show that the response includes the model version and prediction ID.

Open Monitor and show volume, review rate, intent mix, and the recent confidence distribution. Small traffic windows may be marked insufficient. Explain that metadata is logged without the raw submitted message and that drift is a distribution signal rather than measured accuracy loss.

## 4:15–5:00 — Close with engineering evidence

Show the artifact manifest, tests, and container configuration. Name one actual limitation and one specific next experiment, such as a sentence-transformer baseline under the same partitions or a labeled out-of-domain evaluation set.

Only show a passing test run or CI badge when it actually passed. Only claim a hosted deployment when you have deployed and verified one.
