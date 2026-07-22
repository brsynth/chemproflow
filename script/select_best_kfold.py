import argparse
import json
import os

import pandas as pd
from chemproflow.utils.misc import read_json


def select_best_fold(stats, analysis_dir):
    """Pick the fold whose target-recovery F1 (chemproflow.tcid.train's
    evaluate_target_tcid output) is highest."""
    best_fold_idx = None
    best_metrics = None
    best_f1 = float("-inf")

    for fold_idx, stats_fold in stats.items():
        target_metrics = stats_fold.get("target_tcid_metrics")
        if target_metrics is None:
            raise ValueError(
                f"Fold {fold_idx!r} in {analysis_dir!r} has no 'target_tcid_metrics'; "
                "this tool expects a train.py run launched with --parameter-tcid-str."
            )
        if target_metrics["f1"] > best_f1:
            best_f1 = target_metrics["f1"]
            best_fold_idx = fold_idx
            best_metrics = target_metrics

    return best_fold_idx, best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-analysis-dir-str", nargs="*", help="Output directory produced by chemproflow.tcid.train (contains kfold.json)")
    parser.add_argument("--output-results-csv", required=True, help="Where to write the best-fold summary. ")
    args = parser.parse_args()

    analysis_dir = args.input_analysis_dir_str
    output_csv = args.output_results_csv

    print("Read kfold.json")
    stats = read_json(path=os.path.join(analysis_dir, "kfold.json"))

    print("Select best fold by F1")
    best_fold_idx, best_metrics = select_best_fold(stats, analysis_dir)

    result = {
        "tcid": best_metrics["tcid"],
        "splitter": best_metrics["splitter"],
        "seed": best_metrics["seed"],
        "fold": best_fold_idx,
        "support": best_metrics["support"],
        "n_samples": best_metrics["n_samples"],
        "precision": best_metrics["precision"],
        "recall": best_metrics["recall"],
        "f1": best_metrics["f1"],
    }

    print(result)

