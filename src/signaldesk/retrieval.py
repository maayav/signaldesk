"""Sparse TF-IDF cosine retrieval over the fit partition only."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import normalize


def build_reference(vectorizer, texts: list[str]):
    """Use word features so retrieved examples have an easily explained score."""
    return normalize(vectorizer.transformer_list[0][1].transform(texts), copy=False).tocsr()


def similar_tickets(bundle: dict, text: str, k: int = 3) -> list[dict]:
    word_vectorizer = bundle["vectorizer"].transformer_list[0][1]
    query = normalize(word_vectorizer.transform([text]), copy=False)
    scores = (bundle["retrieval_matrix"] @ query.T).toarray().ravel()
    indices = np.argsort(-scores, kind="stable")[:k]
    return [
        {"text": bundle["train_texts"][int(index)], "intent": bundle["train_labels"][int(index)], "similarity": float(scores[index])}
        for index in indices if scores[index] > 0
    ]
