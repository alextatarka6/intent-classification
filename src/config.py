"""Single source of truth for seeds, k values and paths."""
from pathlib import Path

# Global seed list: every seeded step in the project iterates over this.
SEEDS = [0, 1, 2, 3, 4]

# Shots per intent for the k-shot splits.
KS = [1, 5, 10, 20, 50]

DATASET_NAME = "PolyAI/banking77"
# The Hub repo contains only a loading script (no data files), which datasets>=4
# refuses to run. We replicate it: label names/order come from the Hub repo's
# dataset_infos.json at a pinned commit, and the text comes from the same GitHub
# CSVs the script downloads, pinned to a commit instead of `master`.
DATASET_REVISION = "90d4e2ee5521c04fc1488f065b8b083658768c57"
SOURCE_COMMIT = "9d081458ff52e53cf7e848f414e6e9344e4e6696"
SOURCE_URLS = {
    split: "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
           f"{SOURCE_COMMIT}/banking_data/{split}.csv"
    for split in ["train", "test"]
}
EXPECTED_SIZES = {"train": 10003, "test": 3080}

# New-intent hold-out: chosen once with this seed, then frozen in data/new_intents.json.
NEW_INTENT_SEED = 2025
N_NEW_INTENTS = 7

# Seed for picking the example rows printed during data inspection.
INSPECT_SEED = SEEDS[0]

ZEROSHOT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = ROOT / "results"
NEW_INTENTS_PATH = DATA_DIR / "new_intents.json"


def get_device():
    """Apple MPS when available, otherwise CPU."""
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"
