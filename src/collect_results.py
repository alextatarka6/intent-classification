"""Merge the baselines, the fine-tuning curve and all fine-tuning runs into one JSON file.

    python src/collect_results.py
"""
import json

from config import RESULTS_DIR


def load(path):
    return json.loads(path.read_text())


def main():
    curve = load(RESULTS_DIR / "finetune_curve.json")
    runs = [load(p) for p in sorted((RESULTS_DIR / "finetune").glob("*.json"))]
    # Sort runs by k (with "full" last), then seed, to match the curve order.
    runs.sort(key=lambda r: (r["k"] == "full", r["k"] if r["k"] != "full" else 0, r["seed"]))
    combined = {
        "dataset": "Banking77 (PolyAI/banking77)",
        "test_split": "data/splits/test/test_all.csv",
        "metrics": "accuracy and macro-F1 over all 77 intents",
        "majority": load(RESULTS_DIR / "majority.json"),
        "zeroshot": load(RESULTS_DIR / "zeroshot.json"),
        "finetune": {
            "model": curve["model"],
            "hparams": curve["hparams"],
            "std": curve["std"],
            "curve": curve["curve"],
            "runs": runs,
        },
    }
    out = RESULTS_DIR / "baseline_results.json"
    out.write_text(json.dumps(combined, indent=2))
    print(f"wrote {out.relative_to(RESULTS_DIR.parent)}: 2 reference baselines, "
          f"{len(curve['curve'])} curve points, {len(runs)} fine-tuning runs")


if __name__ == "__main__":
    main()
