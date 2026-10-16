"""Per-example test predictions for one fine-tuned model, plus example errors to look at.

Uses seed 4, k = 10 since its accuracy is closest to the k = 10 mean. Loads
checkpoints/seed4_k10/ if it exists, otherwise retrains it with finetune.train_model.

Outputs:
    results/predictions_k10_seed4.csv   one row per test example, in test_all.csv order
    results/example_candidates.md       sampled examples, top confusion pairs, lowest-recall intents

    python src/predict_examples.py [--force]
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from baselines import evaluate
from config import FINETUNE_MODEL, FT_HPARAMS, RAW_DIR, RESULTS_DIR, ROOT, SPLITS_DIR, get_device
from finetune import FT_DIR, make_loader, train_model

SEED, K = 4, 10
RUN = f"seed{SEED}_k{K}"
CKPT_DIR = ROOT / "checkpoints" / RUN
PRED_PATH = RESULTS_DIR / f"predictions_k{K}_seed{SEED}.csv"
MD_PATH = RESULTS_DIR / "example_candidates.md"

EXAMPLE_SEED = 0
N_EXAMPLES = 8
MIN_CONFUSABLE = 4  # how many of the incorrect examples come from confusable intent pairs
MAX_WORDS = 15
HIGH_CONF_QUANTILE = 0.75


@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()
    probs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        probs.append(torch.softmax(model(**batch).logits.float(), -1).cpu())
    return torch.cat(probs).numpy()


def load_or_train(n_labels, device):
    if CKPT_DIR.exists():
        print(f"loading checkpoint {CKPT_DIR.relative_to(ROOT)}")
        tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(CKPT_DIR).to(device)
        return model, tokenizer
    train = pd.read_csv(SPLITS_DIR / "standard" / f"{RUN}.csv")
    print(f"no checkpoint; training {RUN}: {len(train)} train rows on {device}", flush=True)
    model, tokenizer, stats = train_model(train, n_labels, SEED, device)
    print(f"  steps={stats['total_steps']} epochs={stats['epochs']} "
          f"time={stats['train_seconds']}s final_loss={stats['final_train_loss']:.4f}")
    CKPT_DIR.mkdir(parents=True)
    model.save_pretrained(CKPT_DIR)
    tokenizer.save_pretrained(CKPT_DIR)
    (CKPT_DIR / "train_stats.json").write_text(json.dumps(
        {"model": FINETUNE_MODEL, "hparams": FT_HPARAMS, "device": device, **stats}, indent=2))
    return model, tokenizer


def confusable(a, b):
    """Rule for 'plausibly confusable': the intent names share their first word (card_*, top_up_*...)."""
    return a.split("_")[0] == b.split("_")[0]


def sample_distinct_intents(df, n, rng, exclude_intents=()):
    """Shuffle with rng, then keep the first row per true intent until n rows."""
    out, seen = [], set(exclude_intents)
    for i in rng.permutation(len(df)):
        row = df.iloc[i]
        if row["true_intent"] not in seen:
            seen.add(row["true_intent"])
            out.append(row)
        if len(out) == n:
            break
    return out


def fmt(row, with_pred):
    line = f"- `{row['text']}` (test row {row.name})  \n  true: **{row['true_intent']}**"
    if with_pred:
        line += f", predicted: **{row['predicted_intent']}**"
    return line + f", confidence: {row['confidence']:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = ap.parse_args()
    if not args.force and PRED_PATH.exists() and MD_PATH.exists():
        print(f"skip {RUN} predictions (exist; use --force to redo)")
        return

    device = get_device()
    test = pd.read_csv(SPLITS_DIR / "test" / "test_all.csv")
    names = json.loads((RAW_DIR / "label_names.json").read_text())
    n_labels = len(names)

    model, tokenizer = load_or_train(n_labels, device)
    loader = make_loader(test, tokenizer, FT_HPARAMS["eval_batch_size"], shuffle=False)
    probs = predict_proba(model, loader, device)
    y_pred = probs.argmax(1)
    y_true = test["label"].to_numpy()

    preds = pd.DataFrame({
        "text": test["text"],
        "true_intent": [names[i] for i in y_true],
        "predicted_intent": [names[i] for i in y_pred],
        "correct": y_pred == y_true,
        "confidence": probs.max(1),
    })
    assert (preds["true_intent"] == test["label_name"]).all()
    preds.to_csv(PRED_PATH, index=False)
    print(f"wrote {PRED_PATH.relative_to(ROOT)} ({len(preds)} rows)")

    # Sanity check against the original sweep result.
    rerun = evaluate(y_true, y_pred, n_labels)
    orig = json.loads((FT_DIR / f"{RUN}.json").read_text())
    print(f"\nsanity check ({RUN}, test_all, n={rerun['n_test']})")
    print(f"{'metric':<10}{'original':>10}{'rerun':>10}{'gap (pts)':>11}")
    gaps = {}
    for m in ("accuracy", "macro_f1"):
        gaps[m] = 100 * (rerun[m] - orig[m])
        print(f"{m:<10}{orig[m]:>10.4f}{rerun[m]:>10.4f}{gaps[m]:>+11.2f}")
    if any(abs(g) > 1 for g in gaps.values()):
        print("WARNING: rerun differs from the original by more than 1 point")

    rng = np.random.default_rng(EXAMPLE_SEED)
    short = (preds["text"].str.split().str.len() <= MAX_WORDS) \
        & ~preds["text"].str.contains("\n|`", regex=True)
    correct = preds[preds["correct"]]
    conf_cut = correct["confidence"].quantile(HIGH_CONF_QUANTILE)
    good = correct[short[correct.index] & (correct["confidence"] >= conf_cut)]
    good_rows = sample_distinct_intents(good, N_EXAMPLES, rng)

    wrong = preds[~preds["correct"] & short]
    is_conf = np.array([confusable(t, p) for t, p in zip(wrong["true_intent"],
                                                         wrong["predicted_intent"])])
    bad_conf = sample_distinct_intents(wrong[is_conf], MIN_CONFUSABLE, rng)
    bad_other = sample_distinct_intents(wrong[~is_conf], N_EXAMPLES - MIN_CONFUSABLE, rng,
                                        exclude_intents={r["true_intent"] for r in bad_conf})

    errors = preds[~preds["correct"]]
    pairs = errors.groupby(["true_intent", "predicted_intent"]).size().reset_index(name="n_pairs")
    pairs = pairs.sort_values(["n_pairs", "true_intent", "predicted_intent"],
                              ascending=[False, True, True]).head(10)

    recall = preds.groupby("true_intent").agg(recall=("correct", "mean"),
                                              n=("correct", "size"),
                                              n_correct=("correct", "sum")).reset_index()
    recall = recall.sort_values(["recall", "true_intent"]).head(5)

    n_err = len(errors)
    lines = [
        f"# Example candidates: {FINETUNE_MODEL}, k = {K}, seed {SEED}",
        "",
        f"Source: `{PRED_PATH.relative_to(ROOT)}` (test_all, n = {len(preds)}). "
        f"Rerun accuracy {rerun['accuracy']:.4f}, macro-F1 {rerun['macro_f1']:.4f} "
        f"(original sweep: {orig['accuracy']:.4f} / {orig['macro_f1']:.4f}).",
        "",
        f"Examples are sampled with RNG seed {EXAMPLE_SEED}, one per true intent, from messages "
        f"of at most {MAX_WORDS} words. \"test row\" is the 0-based row in test_all.csv.",
        "",
        "## a. Correct predictions",
        "",
        f"Filter: correct, and confidence >= {conf_cut:.3f} (the {int(HIGH_CONF_QUANTILE * 100)}th "
        f"percentile of confidence over the {len(correct)} correct predictions).",
        "",
        *[fmt(r, with_pred=False) for r in good_rows],
        "",
        "## b. Incorrect predictions",
        "",
        f"\"Confusable\" means the true and predicted intent names share their first word "
        f"(e.g. card_*, transfer_*). {len(bad_conf)} sampled from confusable errors, "
        f"{len(bad_other)} from the remaining errors.",
        "",
        "Confusable:",
        "",
        *[fmt(r, with_pred=True) for r in bad_conf],
        "",
        "Other:",
        "",
        *[fmt(r, with_pred=True) for r in bad_other],
        "",
        f"## c. Top 10 confusion pairs (of {n_err} errors)",
        "",
        "| # | true intent | predicted intent | count |",
        "|---|---|---|---|",
        *[f"| {i} | {r.true_intent} | {r.predicted_intent} | {r.n_pairs} |"
          for i, r in enumerate(pairs.itertuples(), 1)],
        "",
        "## d. 5 intents with the lowest recall",
        "",
        "| intent | recall | correct / support |",
        "|---|---|---|",
        *[f"| {r.true_intent} | {r.recall:.3f} | {r.n_correct} / {r.n} |"
          for r in recall.itertuples()],
        "",
    ]
    MD_PATH.write_text("\n".join(lines))
    print(f"wrote {MD_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
