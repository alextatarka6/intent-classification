"""Steps 2-3: download Banking77, inspect it, and build the k-shot / new-intent splits.

Run from the repo root:  python src/prepare_data.py
"""
import csv
import hashlib
import json
import urllib.request
import warnings

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from config import (DATA_DIR, DATASET_NAME, DATASET_REVISION, EXPECTED_SIZES, INSPECT_SEED,
                    KS, N_NEW_INTENTS, NEW_INTENT_SEED, NEW_INTENTS_PATH, RAW_DIR,
                    RESULTS_DIR, SEEDS, SOURCE_URLS, SPLITS_DIR)


def load_raw():
    """Load train/test exactly as the HF banking77 script does; save copies to data/raw/.

    The source CSVs are saved untouched (data/raw/source/) along with their sha256,
    and data/raw/{train,test}.csv add the HF integer label id.
    """
    infos = hf_hub_download(DATASET_NAME, "dataset_infos.json", repo_type="dataset",
                            revision=DATASET_REVISION)
    with open(infos) as f:
        label_names = json.load(f)["default"]["features"]["label"]["names"]
    label_id = {n: i for i, n in enumerate(label_names)}

    src_dir = RAW_DIR / "source"
    src_dir.mkdir(parents=True, exist_ok=True)
    frames, checksums = {}, {}
    for split, url in SOURCE_URLS.items():
        path = src_dir / f"{split}.csv"
        urllib.request.urlretrieve(url, path)
        checksums[split] = {"url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        # Same csv.reader settings as the HF loading script.
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL,
                                skipinitialspace=True)
            next(reader)
            rows = [(text, label_id[label], label) for text, label in reader]
        df = pd.DataFrame(rows, columns=["text", "label", "label_name"])
        assert len(df) == EXPECTED_SIZES[split], (split, len(df))
        df.to_csv(RAW_DIR / f"{split}.csv", index=False)
        frames[split] = df

    with open(RAW_DIR / "label_names.json", "w") as f:
        json.dump(label_names, f, indent=2)
    with open(src_dir / "checksums.json", "w") as f:
        json.dump(checksums, f, indent=2)
    return frames["train"], frames["test"], label_names


def compute_stats(train, test, label_names):
    counts = train["label_name"].value_counts().reindex(label_names, fill_value=0)
    lo, hi = int(counts.min()), int(counts.max())
    train_texts = set(train["text"])
    stats = {
        "dataset": DATASET_NAME,
        "hf_revision": DATASET_REVISION,
        "source_urls": SOURCE_URLS,
        "n_train": len(train),
        "n_test": len(test),
        "n_intents": len(label_names),
        "n_intents_in_train": int(train["label"].nunique()),
        "n_intents_in_test": int(test["label"].nunique()),
        "train_per_intent": {
            "min": lo,
            "min_intents": sorted(counts[counts == lo].index.tolist()),
            "max": hi,
            "max_intents": sorted(counts[counts == hi].index.tolist()),
            "mean": float(counts.mean()),
            "median": float(counts.median()),
        },
        "test_per_intent": {
            "min": int(test["label"].value_counts().min()),
            "max": int(test["label"].value_counts().max()),
        },
        # Data-quality checks (reported only; the test set is never modified).
        "n_duplicate_texts_within_train": int(train["text"].duplicated().sum()),
        "n_test_texts_also_in_train": int(test["text"].isin(train_texts).sum()),
    }
    return stats, counts


def show_examples(train):
    lines = ["=== 5 random training rows ==="]
    sample = train.sample(5, random_state=INSPECT_SEED)
    for _, r in sample.iterrows():
        lines.append(f"[{r['label_name']}] {r['text']}")
    for intent in ["card_arrival", "card_delivery_estimate"]:
        lines.append(f"\n=== 3 examples: {intent} ===")
        sub = train[train["label_name"] == intent].sample(3, random_state=INSPECT_SEED)
        for t in sub["text"]:
            lines.append(f"- {t}")
    text = "\n".join(lines)
    print(text)
    (RESULTS_DIR / "examples.txt").write_text(text + "\n")


def get_new_intents(label_names):
    """Pick the held-out intents once; afterwards always read the frozen file."""
    if NEW_INTENTS_PATH.exists():
        with open(NEW_INTENTS_PATH) as f:
            frozen = json.load(f)
        assert frozen["seed"] == NEW_INTENT_SEED and len(frozen["new_intents"]) == N_NEW_INTENTS
        print(f"Using frozen new intents from {NEW_INTENTS_PATH}")
        return frozen["new_intents"]
    rng = np.random.default_rng(NEW_INTENT_SEED)
    chosen = sorted(rng.choice(label_names, size=N_NEW_INTENTS, replace=False).tolist())
    with open(NEW_INTENTS_PATH, "w") as f:
        json.dump({"seed": NEW_INTENT_SEED, "new_intents": chosen}, f, indent=2)
    print(f"Wrote new intents to {NEW_INTENTS_PATH}")
    return chosen


def intent_orderings(train, label_names, seed):
    """One random permutation of each intent's training rows for this seed.

    Taking the first k rows gives the k-shot sample, so splits are nested
    (the k=5 sample is a subset of the k=10 sample for the same seed).
    """
    rng = np.random.default_rng(seed)
    return {name: rng.permutation(train.index[train["label_name"] == name].to_numpy())
            for name in label_names}


def build_splits(train, test, label_names, new_intents):
    std_dir, new_dir = SPLITS_DIR / "standard", SPLITS_DIR / "new_intent"
    std_dir.mkdir(parents=True, exist_ok=True)
    new_dir.mkdir(parents=True, exist_ok=True)
    old_intents = [n for n in label_names if n not in new_intents]
    old_rows = train[train["label_name"].isin(old_intents)]
    manifest = {"seeds": SEEDS, "ks": KS, "new_intents": new_intents, "files": [], "capped": []}

    for seed in SEEDS:
        order = intent_orderings(train, label_names, seed)
        for k in KS:
            picks = {}
            for name in label_names:
                n_avail = len(order[name])
                if n_avail < k:
                    warnings.warn(f"seed={seed} k={k}: intent '{name}' has only {n_avail} "
                                  f"training examples; capped at {n_avail}")
                    manifest["capped"].append({"seed": seed, "k": k, "intent": name,
                                               "available": n_avail})
                picks[name] = order[name][:k]

            std_idx = np.concatenate([picks[n] for n in label_names])
            std = train.loc[std_idx]
            assert not std.index.duplicated().any()
            path = std_dir / f"seed{seed}_k{k}.csv"
            std.to_csv(path, index=False)
            manifest["files"].append({"path": str(path.relative_to(DATA_DIR)), "type": "standard",
                                      "seed": seed, "k": k, "n_rows": len(std)})

            new_idx = np.concatenate([picks[n] for n in new_intents])
            mixed = pd.concat([old_rows, train.loc[new_idx]]).sort_index()
            assert not mixed.index.duplicated().any()
            path = new_dir / f"seed{seed}_k{k}.csv"
            mixed.to_csv(path, index=False)
            manifest["files"].append({"path": str(path.relative_to(DATA_DIR)), "type": "new_intent",
                                      "seed": seed, "k": k, "n_rows": len(mixed),
                                      "n_old_rows": len(old_rows), "n_new_rows": len(new_idx)})

    # Test subsets for the new-intent experiment (rows copied unchanged).
    test_dir = SPLITS_DIR / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    subsets = {"test_all": test,
               "test_old": test[test["label_name"].isin(old_intents)],
               "test_new": test[test["label_name"].isin(new_intents)]}
    assert len(subsets["test_old"]) + len(subsets["test_new"]) == len(test)
    for name, df in subsets.items():
        path = test_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        manifest["files"].append({"path": str(path.relative_to(DATA_DIR)), "type": "test",
                                  "n_rows": len(df)})

    with open(SPLITS_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train, test, label_names = load_raw()

    stats, _ = compute_stats(train, test, label_names)
    with open(RESULTS_DIR / "data_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print()
    show_examples(train)
    print()

    new_intents = get_new_intents(label_names)
    print("New (held-out) intents:", new_intents)
    manifest = build_splits(train, test, label_names, new_intents)
    print(f"Wrote {len(manifest['files'])} split files; "
          f"{len(manifest['capped'])} (seed, k, intent) cases capped")


if __name__ == "__main__":
    main()
