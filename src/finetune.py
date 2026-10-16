"""k-shot fine-tuning learning curve: BERT on each standard split and on the full train set.

Each run is saved to results/finetune/{run}.json and skipped next time if that file exists,
so an interrupted sweep can be resumed. Mean/std per k go to results/finetune_curve.json.

    python src/finetune.py                      # full sweep
    python src/finetune.py --seeds 0 --ks 1     # subset
    python src/finetune.py --force              # rerun even if results exist
"""
import argparse
import json
import math
import platform
import random
import time

import numpy as np
import pandas as pd
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from baselines import evaluate
from config import (FINETUNE_MODEL, FT_HPARAMS, KS, RAW_DIR, RESULTS_DIR, ROOT, SEEDS, SPLITS_DIR,
                    get_device)

FT_DIR = RESULTS_DIR / "finetune"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(df, tokenizer, batch_size, shuffle, seed=None):
    enc = tokenizer(df["text"].tolist(), truncation=True, max_length=FT_HPARAMS["max_length"])
    items = [{"input_ids": ids, "attention_mask": am, "labels": int(y)}
             for ids, am, y in zip(enc["input_ids"], enc["attention_mask"], df["label"])]

    def collate(batch):
        out = tokenizer.pad([{k: b[k] for k in ("input_ids", "attention_mask")} for b in batch],
                            return_tensors="pt")
        out["labels"] = torch.tensor([b["labels"] for b in batch])
        return out

    gen = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(items, batch_size=batch_size, shuffle=shuffle, generator=gen,
                      collate_fn=collate)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        preds.append(model(**batch).logits.argmax(-1).cpu())
    return torch.cat(preds).numpy()


def train_model(train_df, n_labels, seed, device):
    """Fine-tune FINETUNE_MODEL on train_df; returns (model, tokenizer, training stats)."""
    hp = FT_HPARAMS
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(FINETUNE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        FINETUNE_MODEL, num_labels=n_labels).to(device)

    train_loader = make_loader(train_df, tokenizer, hp["batch_size"], shuffle=True, seed=seed)
    steps_per_epoch = len(train_loader)
    epochs = max(hp["epochs"], math.ceil(hp["min_steps"] / steps_per_epoch))
    total_steps = epochs * steps_per_epoch

    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)],
         "weight_decay": hp["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=hp["lr"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(hp["warmup_ratio"] * total_steps), total_steps)

    start = time.time()
    epoch_losses = []
    for _ in range(epochs):
        model.train()
        losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        epoch_losses.append(float(np.mean(losses)))
    train_seconds = time.time() - start
    stats = {"epochs": epochs, "total_steps": total_steps, "train_seconds": round(train_seconds, 1),
             "final_train_loss": epoch_losses[-1], "epoch_train_losses": epoch_losses}
    return model, tokenizer, stats


def train_and_eval(train_df, test_df, n_labels, seed, device):
    model, tokenizer, stats = train_model(train_df, n_labels, seed, device)
    test_loader = make_loader(test_df, tokenizer, FT_HPARAMS["eval_batch_size"], shuffle=False)
    y_pred = predict(model, test_loader, device)
    return {**stats, **evaluate(test_df["label"].to_numpy(), y_pred, n_labels)}


def aggregate():
    runs = [json.loads(p.read_text()) for p in sorted(FT_DIR.glob("*.json"))]
    points = []
    for k in KS + ["full"]:
        sel = [r for r in runs if r["k"] == k]
        if not sel:
            continue
        acc = [r["accuracy"] for r in sel]
        f1 = [r["macro_f1"] for r in sel]
        points.append({"k": k, "n_seeds": len(sel), "seeds": [r["seed"] for r in sel],
                       "n_train_mean": float(np.mean([r["n_train"] for r in sel])),
                       "accuracy_mean": float(np.mean(acc)), "accuracy_std": float(np.std(acc)),
                       "macro_f1_mean": float(np.mean(f1)), "macro_f1_std": float(np.std(f1))})
    refs = {}
    for name in ("majority", "zeroshot"):
        r = json.loads((RESULTS_DIR / f"{name}.json").read_text())
        refs[name] = {"accuracy": r["accuracy"], "macro_f1": r["macro_f1"]}
    curve = {"model": FINETUNE_MODEL, "hparams": FT_HPARAMS, "test_split": "test_all",
             "std": "population std over seeds", "curve": points, "reference_baselines": refs}
    with open(RESULTS_DIR / "finetune_curve.json", "w") as f:
        json.dump(curve, f, indent=2)
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--ks", nargs="+", default=[str(k) for k in KS] + ["full"],
                    help="k values and/or 'full'")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    FT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    test = pd.read_csv(SPLITS_DIR / "test" / "test_all.csv")
    with open(RAW_DIR / "label_names.json") as f:
        n_labels = len(json.load(f))

    for k in [k if k == "full" else int(k) for k in args.ks]:
        for seed in args.seeds:
            run = f"seed{seed}_{'full' if k == 'full' else f'k{k}'}"
            out = FT_DIR / f"{run}.json"
            if out.exists() and not args.force:
                print(f"skip {run} (exists)")
                continue
            if k == "full":
                train_path = RAW_DIR / "train.csv"
            else:
                train_path = SPLITS_DIR / "standard" / f"seed{seed}_k{k}.csv"
            train = pd.read_csv(train_path)
            print(f"run {run}: {len(train)} train rows on {device}", flush=True)
            result = train_and_eval(train, test, n_labels, seed, device)
            result = {"method": "finetune", "model": FINETUNE_MODEL, "seed": seed, "k": k,
                      "train_file": str(train_path.relative_to(ROOT)),
                      "n_train": len(train), "test_file": "data/splits/test/test_all.csv",
                      "device": device, "hparams": FT_HPARAMS,
                      "torch_version": torch.__version__,
                      "transformers_version": transformers.__version__,
                      "platform": platform.platform(), **result}
            out.write_text(json.dumps(result, indent=2))
            print(f"  acc={result['accuracy']:.4f} macro_f1={result['macro_f1']:.4f} "
                  f"steps={result['total_steps']} time={result['train_seconds']}s", flush=True)

    print(json.dumps(aggregate()["curve"], indent=2))


if __name__ == "__main__":
    main()
