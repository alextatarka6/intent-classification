#!/usr/bin/env bash
# Rebuild data, splits and baselines from scratch (seeds are fixed in src/config.py).
set -euo pipefail
cd "$(dirname "$0")"
.venv/bin/python src/prepare_data.py
.venv/bin/python src/baselines.py
.venv/bin/python src/finetune.py
