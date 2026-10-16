"""Majority-class and zero-shot SBERT baselines on the full test set.

    python src/baselines.py
"""
import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from config import RAW_DIR, RESULTS_DIR, ZEROSHOT_MODEL, get_device


def evaluate(y_true, y_pred, n_labels):
    # labels= fixes the macro average over all 77 intents, including never-predicted ones.
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(n_labels)),
                                   average="macro", zero_division=0)),
        "n_test": len(y_true),
    }


def label_to_phrase(name):
    """card_arrival -> 'card arrival'."""
    return name.replace("_", " ").lower().strip()


def majority(train, test, n_labels):
    counts = Counter(train["label"])
    # Highest count; ties broken by the lowest label id so the choice is deterministic.
    top = min(counts, key=lambda lab: (-counts[lab], lab))
    name = train.loc[train["label"] == top, "label_name"].iloc[0]
    y_pred = np.full(len(test), top)
    return {"method": "majority_class", "train_split": "full train",
            "predicted_label": int(top), "predicted_intent": name,
            "train_count": counts[top], **evaluate(test["label"], y_pred, n_labels)}


def zeroshot(test, label_names):
    import sentence_transformers
    from sentence_transformers import SentenceTransformer

    device = get_device()
    model = SentenceTransformer(ZEROSHOT_MODEL, device=device)
    phrases = [label_to_phrase(n) for n in label_names]
    label_emb = model.encode(phrases, normalize_embeddings=True, convert_to_numpy=True)
    text_emb = model.encode(test["text"].tolist(), batch_size=128,
                            normalize_embeddings=True, convert_to_numpy=True)
    y_pred = (text_emb @ label_emb.T).argmax(axis=1)
    return {"method": "zeroshot_sbert_nearest_label", "model": ZEROSHOT_MODEL,
            "sentence_transformers_version": sentence_transformers.__version__,
            "similarity": "cosine", "device": device, "label_phrase_examples": phrases[:5],
            **evaluate(test["label"], y_pred, len(label_names))}


def main():
    train = pd.read_csv(RAW_DIR / "train.csv")
    test = pd.read_csv(RAW_DIR / "test.csv")
    with open(RAW_DIR / "label_names.json") as f:
        label_names = json.load(f)

    for fname, result in [("majority.json", majority(train, test, len(label_names))),
                          ("zeroshot.json", zeroshot(test, label_names))]:
        with open(RESULTS_DIR / fname, "w") as f:
            json.dump(result, f, indent=2)
        print(fname, json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
