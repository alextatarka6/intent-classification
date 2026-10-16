# Few-shot intent classification on Banking77

## Question

Chatbots are constantly being updated with new intents, but it's not always the case that there is an abundance of training data. 
Collecting and labeling data for training could end up costing more than adding the new feature itself which raises a research question:
How much labelled data per intent does fine-tuning need before it beats a zero-shot baseline, and how close does it get to full-data training?

## Setup

- **Data:** Banking77, 77 intents, 10,003 train / 3,080 test. Details in [data/README.md](data/README.md).
- **k-shot splits:** k ∈ {1, 5, 10, 20, 50} examples per intent, 5 seeds each, nested across k.
- **Baselines:**
  - majority class
  - zero-shot: embed the utterance and each intent name with `all-MiniLM-L6-v2`, pick the nearest
- **Fine-tuning:** `bert-base-uncased`, fixed hyperparameters (no dev set at small k), at least
  500 steps per run. See `FT_HPARAMS` in [src/config.py](src/config.py).
- **Metrics:** accuracy and macro-F1 on the full test set, mean ± std over 5 seeds.


## Results

| Method | Train rows | Accuracy | Macro-F1 |
|---|---|---|---|
| Majority class | 10,003 | 1.3 | 0.0 |
| Zero-shot MiniLM | 0 | 62.2 | 61.2 |
| BERT, k = 1 | 77 | 22.0 ± 1.5 | 21.2 ± 1.4 |
| BERT, k = 5 | 385 | 62.6 ± 1.6 | 61.3 ± 1.7 |
| BERT, k = 10 | 770 | 73.9 ± 1.7 | 72.9 ± 2.2 |
| BERT, k = 20 | 1,540 | 85.6 ± 0.6 | 85.6 ± 0.6 |
| BERT, k = 50 | 3,826 | 91.1 ± 0.4 | 91.1 ± 0.4 |
| BERT, full train | 10,003 | 93.6 ± 0.3 | 93.6 ± 0.3 |


Takeaways:
- fine-tuning on all 77 intents catches up to zero-shot at about k = 5
- k = 50 is within 2.5 points of accuracy of the full data with about a third of the data

## Error analysis (k = 10, seed 4)

The main finding from the errors the model made was that intents that are very similar are the ones that are confused most often for example: why_verify_identity -> verify_my_identity (24 errors).

## Reproducing

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-lock.txt
./run_all.sh
```

Takes roughly 2 hours to run the full baseline results on the different subsamples.

| Script | Output |
|---|---|
| `src/prepare_data.py` | `data/`, `results/data_stats.json` |
| `src/baselines.py` | `results/majority.json`, `results/zeroshot.json` |
| `src/finetune.py` | `results/finetune/`, `results/finetune_curve.json` |
| `src/collect_results.py` | `results/baseline_results.json` |
| `src/predict_examples.py` | `results/predictions_k10_seed4.csv`, `results/example_candidates.md` |

## Next steps

The next step is to train a BERT-base model on the full data of 70 intents and then do k-shot adaptation on the 7 holdouts. The first 7 holdouts have already been chosen at random and are in `data\new_intents.json` but another 7 will also be tested for robustness.
