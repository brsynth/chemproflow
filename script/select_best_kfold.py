import argparse
import itertools
import json
import os
import pickle

import pandas as pd
import torch
from chemproflow.utils.misc import read_json
from torch_geometric.loader import DataLoader
from tqdm import tqdm


def build_dataset(curdir: str, kfold: int):
    # curdir: chemical_dir, "chemproflow", "tcid_vs_smiles_scaffold"
    file_encoder_transport_pkl = os.path.join(curdir, "encoder.pkl")
    with open(file_encoder_transport_pkl, "rb") as f:
        encoder = pickle.load(f)

    datas = []
    for section in ["train", "valid", "test"]:
        if section == "test":
            file_dataset_pt = os.path.join(curdir, f"{section}.pt")
        else:
            file_dataset_pt = os.path.join(curdir, f"kfold-{kfold}", f"{section}.pt")
        dataset = torch.load(file_dataset_pt, weights_only=False)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        with torch.no_grad():
            for batch in loader:
                labels = encoder.inverse_transform(batch.y)
                for label, smiles in zip(labels, batch.smiles):
                    data = dict(label=label, smiles=smiles, section=section)
                    datas.append(data)
    return pd.DataFrame(datas)

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
    parser.add_argument("--input-analysis-str", required=True, help="Output directory produced by chemproflow.tcid.train (contains kfold.json)")
    parser.add_argument("--output-results-csv", required=True, help="Where to write the best-fold summary. ")
    args = parser.parse_args()

    analysis_dir = args.input_analysis_str
    output_csv = args.output_results_csv

    datas = []
    for tcid, split, seed in tqdm(itertools.product(["2.A.66.1.8", "2.A.66.1.16", "2.A.1.19.29", "2.A.66.1.14", "2.A.6.2.7"], ["random", "scaffold"], ["42", "43", "44"]), desc="Loop over dirs"):
        curdir = os.path.join(analysis_dir, f"tcid_vs_smiles_{tcid}_{split}_{seed}")
        file_kfold_json =os.path.join(curdir, "kfold.json")
        stats = read_json(path=file_kfold_json)
        best_fold_idx, best_metrics = select_best_fold(stats, curdir)
        data = {
            "tcid": best_metrics["tcid"],
            "splitter": best_metrics["splitter"],
            "seed": best_metrics["seed"],
            "fold": best_fold_idx,
            "support": best_metrics["support"],
            "precision": best_metrics["precision"],
            "recall": best_metrics["recall"],
            "f1": best_metrics["f1"],
        }
        df_dataset = build_dataset(curdir=curdir, kfold=best_fold_idx)
        df_dataset = df_dataset[df_dataset["label"].apply(lambda x: best_metrics["tcid"] in x)]
        count = df_dataset["section"].value_counts()
        for section in count.index:
            data[f"smiles_{section}"] = df_dataset[df_dataset["section"] == section]["smiles"].to_list()
        data.update(df_dataset["section"].value_counts().to_dict())

        datas.append(data)

    df = pd.DataFrame(datas)
    df.to_csv(output_csv, index=False)
    print("Done")
