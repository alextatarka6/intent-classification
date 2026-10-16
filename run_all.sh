#!/usr/bin/env bash
# Rebuild data, splits, baselines and fine-tuning results. Finished runs are skipped.
set -euo pipefail
cd "$(dirname "$0")"
.venv/bin/python src/prepare_data.py
.venv/bin/python src/baselines.py
.venv/bin/python src/finetune.py
.venv/bin/python src/collect_results.py
.venv/bin/python src/predict_examples.py
