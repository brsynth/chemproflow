import argparse
from collections import Counter
import os

from chemproflow.model.dataset import mol_to_graph
from chemproflow.utils.misc import set_seed, write_json, write_pickle
from chemproflow.tcid.model import ModelTcid
from chemproflow.model.wrap import WrapModel
from chemproflow.utils.molecule import fmt_smiles
from chemproflow.utils.splitters import ScaffoldSplitter
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from natsort import natsorted
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split
import torch
from torch_geometric.loader import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger


def count_labels(datas):
    counter = Counter()
    for data in datas:
        counter.update(data.tcid)
    return dict(counter)


def collect_probs_and_targets(model, loader):
    device = next(model.parameters()).device
    model.eval()
    probs, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(batch.y.view_as(logits).cpu().numpy())
    probs = np.concatenate(probs, axis=0)
    targets = np.concatenate(targets, axis=0)
    return probs, targets


def tune_per_label_thresholds(model, loader, default_threshold=0.5):
    probs, targets = collect_probs_and_targets(model, loader)
    num_classes = probs.shape[1]
    thresholds = np.full(num_classes, default_threshold, dtype=float)
    for class_idx in range(num_classes):
        y_true = targets[:, class_idx]
        if np.all(y_true == 0) or np.all(y_true == 1):
            continue
        y_scores = probs[:, class_idx]
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
        denom = np.clip(precision + recall, a_min=1e-8, a_max=None)
        f1_scores = 2 * precision * recall / denom
        best_idx = int(np.nanargmax(f1_scores))
        if best_idx >= len(pr_thresholds):
            best_th = 1.0
        else:
            best_th = float(pr_thresholds[best_idx])
        thresholds[class_idx] = best_th
    return thresholds


def evaluate_with_thresholds(model, loader, thresholds, average="weighted"):
    probs, targets = collect_probs_and_targets(model, loader)
    thresholds = np.asarray(thresholds, dtype=np.float32).reshape(1, -1)
    preds = (probs >= thresholds).astype(int)
    metrics = {
        "f1": float(f1_score(targets, preds, average=average, zero_division=0)),
        "precision": float(precision_score(targets, preds, average=average, zero_division=0)),
        "recall": float(recall_score(targets, preds, average=average, zero_division=0)),
        "per_label_precision": precision_score(targets, preds, average=None, zero_division=0).tolist(),
        "per_label_recall": recall_score(targets, preds, average=None, zero_division=0).tolist(),
        "per_label_f1": f1_score(targets, preds, average=None, zero_division=0).tolist(),
    }
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dataset-csv", required=True, help="Dataset"
    )
    parser.add_argument(
        "--input-pyoverdine-xlsx", help="Dataset pyoverdines: name, smiles, valid columns"
    )
    parser.add_argument(
        "--parameter-kfold-int", default= 5, type=int, help="Count K-Fold"
    )
    parser.add_argument(
        "--parameter-seed-int", default=42, type=int, help="Seed"
    )
    parser.add_argument(
        "--parameter-batch-size-int", default=128, type=int, help="Batch size"
    )
    parser.add_argument(
        "--parameter-splitter-str", default="random", choices=["random", "scaffold"], help="Splitter to use"
    )
    parser.add_argument(
        "--output-dir-str", required=True, help="Output directory"
    )
    args = parser.parse_args()

    # Init
    outdir = args.output_dir_str
    kfold = args.parameter_kfold_int
    seed = args.parameter_seed_int
    batch_size = args.parameter_batch_size_int
    file_dataset_csv = args.input_dataset_csv
    splitter_params = args.parameter_splitter_str
    file_pyoverdine_xlsx = args.input_pyoverdine_xlsx

    file_encoder_pkl = os.path.join(outdir, "encoder.pkl")

    os.makedirs(outdir, exist_ok=True)
    set_seed(seed=seed, workers=True)

    print("Read dataset file")
    df = pd.read_csv(file_dataset_csv)
    if file_pyoverdine_xlsx:
        df_pyo = pd.read_excel(file_pyoverdine_xlsx)
        df_pyo = df_pyo[df_pyo["valid"]]
        df_pyo["smiles"] = df_pyo["smiles"].apply(fmt_smiles)  
        # Split
        df_pyo_fir = df_pyo[df_pyo["class"].isin(["PvdI", "PvdII"])].copy()
        df_pyo_las = df_pyo[df_pyo["class"].isin(["PvdIII"])].copy()
        df_pyo_fir.to_csv(os.path.join(outdir, "pyoverdine_in_train.csv"), index=False)
        df_pyo_las.to_csv(os.path.join(outdir, "pyoverdine_not_in_train.csv"), index=False)
        # Concat
        df_pyo = df_pyo_fir.copy()
        df_pyo = df_pyo[["smiles", "tcid"]]
        df_pyo = df_pyo.sample(frac=1, random_state=seed)
        df = pd.concat([df, df_pyo], ignore_index=False)
    df = df.groupby("smiles", as_index=False).agg({"tcid": set})

    print("Encode labels")
    encoder = MultiLabelBinarizer()
    tcids = [natsorted(list(x)) for x in df["tcid"]]
    labels = encoder.fit_transform(tcids)
    write_pickle(data=encoder, path=file_encoder_pkl)

    print("Build data points")
    df["graph"] = None
    for ix, row in df.iterrows():
        # Use real binary label vectors here
        graph = mol_to_graph(smiles=row["smiles"])
        graph.tcid = tcids[ix]
        # Keep class labels 2D so PyG batches into [batch, num_classes]
        graph.y = torch.tensor(labels[ix], dtype=torch.float).unsqueeze(0)
        df.at[ix, "graph"] = graph

    print("Build indices for cross validation")
    # Shuffle once so iterative_train_test_split (which has no random_state) does not
    # see a systematically ordered dataset (e.g. grouped by SMILES after groupby).
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(df))
    df = df.iloc[perm].reset_index(drop=True)
    labels = labels[perm]

    if splitter_params == "random":
        print("Build indices for cross validation")
        all_indices = np.arange(len(df))

        # First create a single hold-out test split so cross-validation only touches
        # the training portion of the data.
        train_indices_all, _, test_indices, _ = iterative_train_test_split(
            all_indices.reshape(-1, 1),
            labels,
            test_size=0.2,
        )
        train_indices_all = train_indices_all.flatten()
        test_indices = test_indices.flatten()

        # Test
        test_datas = [df.loc[ix, "graph"] for ix in test_indices]
        torch.save(test_datas, os.path.join(outdir, f"test.pt"))
        test_loader = DataLoader(test_datas, batch_size=batch_size, shuffle=False)
        
        if kfold == 1:
            print("K-Fold set to 1, using a single stratified hold-out split for validation")
            if len(train_indices_all) < 2:
                raise ValueError("Need at least 2 training samples when --parameter-kfold-int == 1")

            val_fraction = 0.2
            min_fraction = 1.0 / max(len(train_indices_all), 2)
            val_fraction = max(val_fraction, min_fraction)
            val_fraction = min(val_fraction, 0.5)

            train_features = train_indices_all.reshape(-1, 1)
            holdout_train, _, holdout_valid, _ = iterative_train_test_split(
                train_features,
                labels[train_indices_all],
                test_size=val_fraction,
            )

            index_lookup = {idx: pos for pos, idx in enumerate(train_indices_all)}
            train_pos = np.array([index_lookup[int(idx)] for idx in holdout_train.flatten()], dtype=int)
            val_pos = np.array([index_lookup[int(idx)] for idx in holdout_valid.flatten()], dtype=int)
            split_iterator = [(0, train_indices_all[train_pos].tolist(), train_indices_all[val_pos].tolist())]
        else:
            mskf = MultilabelStratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)
            split_iterator = (
                (fold_idx, train_indices_all[train_pos].tolist(), train_indices_all[val_pos].tolist())
                for fold_idx, (train_pos, val_pos) in enumerate(
                    mskf.split(train_indices_all.reshape(-1, 1), labels[train_indices_all])
                )
            )
    elif splitter_params == "scaffold":
        scaffold_splitter = ScaffoldSplitter()
        if kfold == 1:
            train_indices, valid_indices, test_indices = scaffold_splitter.split_stratified(
                df=df,
                labels=labels,
                frac_train=0.6,
                frac_valid=0.2,
                frac_test=0.2,
                random_state=seed,
            )
            split_iterator = ((0, train_indices, valid_indices),)
        else:
            train_indices_all, _, test_indices = scaffold_splitter.split(
                df=df,
                frac_train=0.8,
                frac_valid=0.0,
                frac_test=0.2,
            )
            train_df = df.loc[train_indices_all]
            split_iterator = scaffold_splitter.k_fold_split_stratified(
                df=train_df, labels=labels, k=kfold, random_state=seed
            )
        test_datas = [df.loc[ix, "graph"] for ix in test_indices]
        torch.save(test_datas, os.path.join(outdir, f"test.pt"))
        test_loader = DataLoader(test_datas, batch_size=batch_size, shuffle=False)
    else:
        raise ValueError("Splitter parameter unknown")
    
    fold_metrics = []
    per_fold_thresholds = {}
    stats = {}
    for fold_idx, train_indices, valid_indices in split_iterator:
        print(f"=== Fold {fold_idx} / {kfold} ===")
        outdir_kfold = os.path.join(outdir, f"kfold-{fold_idx}")
        os.makedirs(outdir_kfold, exist_ok=True)

        train_datas = [df.loc[ix, "graph"] for ix in train_indices]
        valid_datas = [df.loc[ix, "graph"] for ix in valid_indices]

        torch.save(train_datas, os.path.join(outdir_kfold, f"train.pt"))
        torch.save(valid_datas, os.path.join(outdir_kfold, f"valid.pt"))
        
        stats_fold = {}
        stats_fold["train"] = count_labels(datas=train_datas)
        stats_fold["train"]["total"] = len(train_datas)
        stats_fold["valid"] = count_labels(datas=valid_datas)
        stats_fold["valid"]["total"] = len(valid_datas)
        stats_fold["test"] = count_labels(datas=test_datas)
        stats_fold["test"]["total"] = len(test_datas)

        print("Make loader")
        train_loader = DataLoader(train_datas, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_datas, batch_size=batch_size, shuffle=False)

        print("Build model")
        model = ModelTcid(
            node_feat_dim=df.iloc[0]["graph"].x.shape[1],
            edge_feat_dim=df.iloc[0]["graph"].edge_attr.shape[1],
            hidden_dim=300,
            num_classes=len(encoder.classes_),
        )
        wrap_model = WrapModel(model=model)

        print("Fit")
        tb_logger = TensorBoardLogger(save_dir=outdir_kfold, name="tensorboard_logs")
        wrap_model.train(
            outdir=outdir_kfold,
            train_loader=train_loader,
            valid_loader=valid_loader,
            logger=tb_logger,
        )

        print("Tune per-label thresholds")
        fold_threshold_values = tune_per_label_thresholds(wrap_model.model, valid_loader)
        thresholds_dict = {
            label: float(fold_threshold_values[idx])
            for idx, label in enumerate(encoder.classes_)
        }
        file_thresholds_json = os.path.join(outdir_kfold, "thresholds.json")
        write_json(data=thresholds_dict, path=file_thresholds_json)
        stats_fold["thresholds"] = thresholds_dict
        per_fold_thresholds[str(fold_idx)] = {
            "thresholds": thresholds_dict,
            "thresholds_file": file_thresholds_json,
        }

        print("Test")
        test_metrics = evaluate_with_thresholds(wrap_model.model, test_loader, fold_threshold_values)
        stats_fold["test_threshold_metrics"] = test_metrics
        fold_metrics.append(test_metrics["f1"])
        print(
            f"Thresholded Test F1={test_metrics['f1']:.3f} "
            f"Precision={test_metrics['precision']:.3f} Recall={test_metrics['recall']:.3f}"
        )

        stats[str(fold_idx)] = stats_fold

    print("Write stats")
    file_stats_kfold_json = os.path.join(outdir, "kfold.json")
    write_json(data=stats, path=file_stats_kfold_json)
    file_thresholds_summary = os.path.join(outdir, "per_label_thresholds.json")
    write_json(data=per_fold_thresholds, path=file_thresholds_summary)

    print("Fold results:", fold_metrics)
    mean_roc = np.nanmean(fold_metrics)

    print("Mean ROC:", mean_roc)
