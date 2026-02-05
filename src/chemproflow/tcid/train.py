import argparse
from collections import Counter
import os

from chemproflow.model.dataset import mol_to_graph
from chemproflow.utils.misc import set_seed, write_json, write_pickle
from chemproflow.tcid.model import ModelTcid
from chemproflow.model.wrap import WrapModel
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
        "--parameter-kfold-int", default= 5, type=int, help="Count K-Fold"
    )
    parser.add_argument(
        "--parameter-seed-int", default=42, type=int, help="Seed"
    )
    parser.add_argument(
        "--parameter-batch-size-int", default=128, type=int, help="Batch size"
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
    
    file_stats_overlap_csv = os.path.join(outdir, "overlap.csv")
    file_stats_overlap_json = os.path.join(outdir, "overlap.json")
    file_encoder_pkl = os.path.join(outdir, "encoder.pkl")

    os.makedirs(outdir, exist_ok=True)
    set_seed(seed=seed, workers=True)

    print("Read dataset file")
    df = pd.read_csv(file_dataset_csv)
    df = df.groupby("smiles", as_index=False).agg({"tcid": set})

    print("Encode labels")
    encoder = MultiLabelBinarizer()
    tcids = [natsorted(list(x)) for x in df["tcid"]]
    labels = encoder.fit_transform(tcids)
    write_pickle(data=encoder, path=file_encoder_pkl)

    print("Build data points")
    datas = []
    for ix, row in df.iterrows():
        # Use real binary label vectors here
        graph = mol_to_graph(smiles=row["smiles"])
        graph.tcid = tcids[ix]
        # Keep class labels 2D so PyG batches into [batch, num_classes]
        graph.y = torch.tensor(labels[ix], dtype=torch.float).unsqueeze(0)
        datas.append(graph)

    # Shuffle once for the initial split; skmultilearn's splitter has no random_state
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(datas))
    datas = [datas[i] for i in perm]
    labels = labels[perm]

    print("Build indices for cross validation")
    all_indices = np.arange(len(datas)).reshape(-1, 1)

    # First create a single hold-out test split so cross-validation only touches
    # the training portion of the data.
    X_train_idx, _, X_test_idx, _ = iterative_train_test_split(
        all_indices,
        labels,
        test_size=0.2,
    )

    # Test
    train_indices_all = X_train_idx.flatten()
    test_indices = X_test_idx.flatten().tolist()
    test_datas = [datas[ix] for ix in test_indices]
    torch.save(test_datas, os.path.join(outdir, f"test.pt"))
    test_loader = DataLoader(test_datas, batch_size=batch_size, shuffle=False)
    
    mskf = MultilabelStratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)

    fold_metrics = []
    per_fold_thresholds = {}
    stats = {}
    for fold_idx, (train_pos, val_pos) in enumerate(
        mskf.split(train_indices_all.reshape(-1, 1), labels[train_indices_all])
    ):
        print(f"=== Fold {fold_idx} / {kfold} ===")
        outdir_kfold = os.path.join(outdir, f"kfold-{fold_idx}")
        os.makedirs(outdir_kfold, exist_ok=True)

        train_indices = train_indices_all[train_pos].tolist()
        valid_indices = train_indices_all[val_pos].tolist()

        train_datas = [datas[ix] for ix in train_indices]
        valid_datas = [datas[ix] for ix in valid_indices]

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

        train_label_matrix = labels[train_indices]

        print("Build model")
        model = ModelTcid(
            node_feat_dim=datas[0].x.shape[1],
            edge_feat_dim=datas[0].edge_attr.shape[1],
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
