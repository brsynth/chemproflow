import argparse
import os
from collections import Counter
from numbers import Number

from chemproflow.model.dataset import mol_to_graph
from chemproflow.utils.misc import set_seed, write_json, write_pickle
from chemproflow.model.wrap import WrapModel
from chemproflow.pu.model import ModelTransport
from chemproflow.utils.splitters import ScaffoldSplitter
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm


def count_labels(datas):
    counter = Counter()
    for data in datas:
        if isinstance(data.y, torch.Tensor):
            counter.update([int(data.y.item())])  # wrap scalar
        else:
            counter.update([int(data.y)])
    return dict(counter)


def expected_calibration_error(probabilities, targets, n_bins=15):
    probabilities = np.asarray(probabilities)
    targets = np.asarray(targets)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(probabilities, bins[1:-1], right=True)
    total = len(probabilities)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        if not np.any(mask):
            continue
        bin_conf = probabilities[mask].mean()
        bin_acc = targets[mask].mean()
        ece += (mask.sum() / total) * abs(bin_acc - bin_conf)
    return float(ece)


def dirichlet_feature_map(probabilities, eps=1e-6):
    probs = np.clip(probabilities, eps, 1 - eps)
    return np.column_stack((np.log(probs), np.log(1 - probs), probs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dataset-csv", required=True, help="Dataset")
    parser.add_argument(
        "--parameter-kfold-int", default=5, type=int, help="Count K-Fold"
    )
    parser.add_argument("--parameter-seed-int", default=42, type=int, help="Seed")
    parser.add_argument(
        "--parameter-batch-size-int", default=128, type=int, help="Batch size"
    )
    parser.add_argument(
        "--parameter-splitter-str",
        default="random",
        choices=["random", "scaffold"],
        help="Splitter to use",
    )
    parser.add_argument("--output-dir-str", required=True, help="Output directory")
    args = parser.parse_args()

    # Init
    outdir = args.output_dir_str
    kfold = args.parameter_kfold_int
    seed = args.parameter_seed_int
    batch_size = args.parameter_batch_size_int
    splitter_params = args.parameter_splitter_str
    file_dataset_csv = args.input_dataset_csv

    os.makedirs(outdir, exist_ok=True)
    set_seed(seed, workers=True)

    df = pd.read_csv(file_dataset_csv)  # columns: ["smiles", "label"]

    print("Encode labels")
    encoder = OrdinalEncoder(
        categories=[["unlabeled", "positive"]]
    )  ## Encode labels with scikit-learn to ensure positive=1 and unlabeled=0
    labels = encoder.fit_transform(df[["label"]]).astype(int).ravel()
    file_encoder_pkl = os.path.join(outdir, "encoder.pkl")
    write_pickle(data=encoder, path=file_encoder_pkl)

    print("Build data points")
    df["graph"] = None
    for ix, row in tqdm(df.iterrows(), total=len(df)):
        graph = mol_to_graph(
            smiles=row["smiles"]
        )  # should set x, edge_index, edge_attr
        # Use single-target binary label with shape [1]
        graph.y = torch.tensor(labels[ix], dtype=torch.float)
        df.at[ix, "graph"] = graph

    print("Build indices for cross validation")
    if kfold < 2:
        raise ValueError("--parameter-kfold-int must be >= 2")

    if splitter_params == "random":
        all_indices = np.arange(len(df))

        # First create a single hold-out test split to avoid leakage across folds
        train_indices_all, test_indices = train_test_split(
            all_indices,
            test_size=0.2,
            random_state=seed,
            stratify=labels,
        )

        # Test dataset
        test_indices = test_indices.tolist()
        test_datas = [df.loc[ix, "graph"] for ix in test_indices]
        torch.save(test_datas, os.path.join(outdir, f"test.pt"))
        test_loader = DataLoader(test_datas, batch_size=batch_size, shuffle=False)

        skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)
        split_iterator = skf.split(train_indices_all, labels[train_indices_all])
        split_iterator = (
            (
                fold_idx,
                train_indices_all[train_pos].tolist(),
                train_indices_all[val_pos].tolist(),
            )
            for fold_idx, (train_pos, val_pos) in enumerate(split_iterator)
        )
    elif splitter_params == "splitter":
        scaffold_splitter = ScaffoldSplitter()
        train_indices_all, _, test_indices = scaffold_splitter.split(
            df=df,
            frac_train=0.8,
            frac_valid=0.0,
            frac_test=0.2,
        )
        test_datas = [df.loc[ix, "graph"] for ix in test_indices]
        torch.save(test_datas, os.path.join(outdir, f"test.pt"))
        test_loader = DataLoader(test_datas, batch_size=batch_size, shuffle=False)
        train_df = df.loc[train_indices_all]
        split_iterator = scaffold_splitter.k_fold_split(df=train_df, k=kfold)
    else:
        raise ValueError("Splitter parameter unknown")

    fold_metrics = []
    stats = {}
    for fold_idx, train_pos, val_pos in split_iterator:
        print(f"=== Fold {fold_idx} / {kfold} ===")

        outdir_kfold = os.path.join(outdir, f"kfold-{fold_idx}")
        os.makedirs(outdir_kfold, exist_ok=True)

        fold_train_indices = np.array(train_pos)
        val_indices = list(val_pos)

        fold_train_labels = labels[fold_train_indices]
        positive_train_indices = fold_train_indices[fold_train_labels == 1]
        if len(positive_train_indices) < 2:
            raise ValueError(
                f"Fold {fold_idx} has fewer than two positive training samples; "
                "cannot create an Elkan-Noto calibration split."
            )

        _, calibration_indices = train_test_split(
            positive_train_indices,
            test_size=0.2,
            random_state=seed + fold_idx,
        )
        train_indices = np.setdiff1d(fold_train_indices, calibration_indices).tolist()
        calibration_indices = calibration_indices.tolist()

        train_datas = [df.loc[ix, "graph"] for ix in train_indices]
        calibration_datas = [df.loc[ix, "graph"] for ix in calibration_indices]
        valid_datas = [df.loc[ix, "graph"] for ix in val_indices]

        torch.save(train_datas, os.path.join(outdir_kfold, f"train.pt"))
        torch.save(calibration_datas, os.path.join(outdir_kfold, f"calibration.pt"))
        torch.save(valid_datas, os.path.join(outdir_kfold, f"valid.pt"))

        stats_fold = {}
        stats_fold["train"] = count_labels(datas=train_datas)
        stats_fold["train"]["total"] = len(train_datas)
        stats_fold["calibration"] = count_labels(datas=calibration_datas)
        stats_fold["calibration"]["total"] = len(calibration_datas)
        stats_fold["valid"] = count_labels(datas=valid_datas)
        stats_fold["valid"]["total"] = len(valid_datas)
        stats_fold["test"] = count_labels(datas=test_datas)
        stats_fold["test"]["total"] = len(test_datas)

        stats[str(fold_idx)] = stats_fold

        print("Make loader")
        train_loader = DataLoader(train_datas, batch_size=batch_size, shuffle=True)
        calibration_loader = DataLoader(
            calibration_datas, batch_size=batch_size, shuffle=False
        )
        val_loader = DataLoader(valid_datas, batch_size=batch_size, shuffle=False)

        print("Build model")
        # Estimate positive prior using only the training portion of the current fold
        fold_pos_prior = (
            float(labels[train_indices].mean()) if len(train_indices) else 0.0
        )

        model = ModelTransport(
            node_feat_dim=df.iloc[0]["graph"].x.shape[1],
            edge_feat_dim=df.iloc[0]["graph"].edge_attr.shape[1],
            hidden_dim=300,
            num_classes=1,
            lr=1e-3,
            pos_prior=fold_pos_prior,
            pu_type="elkan-noto",
        )
        model.set_elkan_calibration_loader(calibration_loader)
        wrap_model = WrapModel(model=model)

        print("Fit")
        tb_logger = TensorBoardLogger(save_dir=outdir_kfold, name="tensorboard_logs")
        wrap_model.train(
            outdir=outdir_kfold,
            train_loader=train_loader,
            valid_loader=val_loader,
            logger=tb_logger,
        )

        model.estimate_elkan_c(calibration_loader)
        print("Test")
        wrap_model.test(test_loader=test_loader)

        model.eval()
        device = next(model.parameters()).device

        val_probs, val_y = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch).squeeze(-1)
                val_probs.append(torch.sigmoid(logits).cpu().numpy())
                val_y.append(batch.y.cpu().numpy())
        val_probs = np.concatenate(val_probs)
        val_y = np.concatenate(val_y).astype(int)

        c_hat = float(getattr(model, "elkan_c", torch.tensor(1.0)).item())
        if c_hat <= 0:
            c_hat = 1.0
        print(f"Estimated Elkan-Noto c (fold {fold_idx}): {c_hat:.4f}")

        val_corrected = np.clip(val_probs / c_hat, 0.0, 1.0)

        ths = np.linspace(0.01, 0.99, 99)

        test_probs, test_y = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                logits = model(batch).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                test_probs.append(probs)
                test_y.append(batch.y.cpu().numpy())

        test_probs = np.concatenate(test_probs)
        test_y = np.concatenate(test_y).astype(int)

        test_corrected = np.clip(test_probs / c_hat, 0.0, 1.0)
        torch.save(test_corrected, os.path.join(outdir_kfold, f"test_corrected.pt"))
        torch.save(test_y, os.path.join(outdir_kfold, f"test_y.pt"))
        ece_uncalibrated = expected_calibration_error(test_corrected, test_y)
        print(
            f"ECE (uncalibrated Elkan-Noto probs, 15 bins): {ece_uncalibrated * 100:.2f}%"
        )

        print("== Dirichlet-calibrated ==")
        dirichlet_clf = LogisticRegression(max_iter=1000)
        val_dirichlet_features = dirichlet_feature_map(val_corrected)
        dirichlet_clf.fit(val_dirichlet_features, val_y)
        val_dirichlet_probs = dirichlet_clf.predict_proba(val_dirichlet_features)[:, 1]
        f1s_dir = [
            f1_score(val_y, (val_dirichlet_probs >= t).astype(int), zero_division=0)
            for t in ths
        ]
        best_idx_dir = int(np.argmax(f1s_dir))
        best_th_dir = float(ths[best_idx_dir])
        best_f1_dir = float(f1s_dir[best_idx_dir])
        print(
            f"Best threshold (Dirichlet, fold {fold_idx}): {best_th_dir:.3f} with F1={best_f1_dir:.3f}"
        )

        test_dirichlet_features = dirichlet_feature_map(test_corrected)
        test_dirichlet_probs = dirichlet_clf.predict_proba(test_dirichlet_features)[
            :, 1
        ]
        test_pred_dir = (test_dirichlet_probs >= best_th_dir).astype(int)

        print(classification_report(test_y, test_pred_dir, digits=3))
        classif_report = classification_report(
            test_y, test_pred_dir, digits=3, output_dict=True
        )
        roc_auc_dir = float(roc_auc_score(test_y, test_dirichlet_probs))
        pr_auc_dir = float(average_precision_score(test_y, test_dirichlet_probs))
        test_f1_dir = float(f1_score(test_y, test_pred_dir, zero_division=0))
        brier_dir = float(brier_score_loss(test_y, test_dirichlet_probs))
        torch.save(
            test_dirichlet_probs, os.path.join(outdir_kfold, f"test_dirichlet_probs.pt")
        )
        ece_dir = expected_calibration_error(test_dirichlet_probs, test_y)

        dirichlet_bundle = {
            "model": dirichlet_clf,
            "threshold": best_th_dir,
            "feature_names": ["log_p", "log_1_minus_p", "p"],
        }
        file_dirichlet_pkl = os.path.join(outdir_kfold, "dirichlet_calibrator.pkl")
        write_pickle(data=dirichlet_bundle, path=file_dirichlet_pkl)

        print(f"Saved Dirichlet calibrator to {file_dirichlet_pkl}")

        print(f"ROC AUC (Dirichlet): {roc_auc_dir:.3f}")
        print(f"PR AUC (Dirichlet): {pr_auc_dir:.3f}")
        print(f"Test F1 (Dirichlet threshold): {test_f1_dir:.3f}")
        print(f"Brier score (Dirichlet): {brier_dir:.3f}")
        print(f"ECE (Dirichlet, 15 bins): {ece_dir * 100:.2f}%")

        fold_metrics.append(
            {
                "ece_uncal": ece_uncalibrated,
                "roc_auc_dir": roc_auc_dir,
                "pr_auc_dir": pr_auc_dir,
                "f1_dir": test_f1_dir,
                "brier_dir": brier_dir,
                "ece_dir": ece_dir,
                "classification_report": classif_report,
            }
        )

    print("Write stats")
    file_stats_kfold_json = os.path.join(outdir, "kfold.json")
    write_json(data=stats, path=file_stats_kfold_json)
    file_fold_metrics_json = os.path.join(outdir, "fold_metrics.json")
    write_json(data=dict(fold_metrics=fold_metrics), path=file_fold_metrics_json)

    print("=== Cross-validation summary ===")
    for idx, metrics in enumerate(fold_metrics, start=1):
        print(
            f"Fold {idx}: ECE(unscaled)={metrics['ece_uncal'] * 100:.2f}% | Dir ROC AUC={metrics['roc_auc_dir']:.3f}, "
            f"PR AUC={metrics['pr_auc_dir']:.3f}, F1={metrics['f1_dir']:.3f}, "
            f"Brier={metrics['brier_dir']:.3f}, ECE={metrics['ece_dir'] * 100:.2f}%"
        )

    mean_metrics = {}
    for key in fold_metrics[0]:
        values = [metrics[key] for metrics in fold_metrics]
        if all(isinstance(value, Number) for value in values):
            mean_metrics[key] = float(np.mean(values))

    print(f"Mean ROC AUC (Dirichlet): {mean_metrics['roc_auc_dir']:.3f}")
    print(f"Mean PR AUC (Dirichlet): {mean_metrics['pr_auc_dir']:.3f}")
    print(f"Mean Test F1 (Dirichlet threshold): {mean_metrics['f1_dir']:.3f}")
    print(f"Mean Brier score (Dirichlet): {mean_metrics['brier_dir']:.3f}")
    print(f"Mean ECE (uncalibrated): {mean_metrics['ece_uncal'] * 100:.2f}%")
    print(f"Mean ECE (Dirichlet): {mean_metrics['ece_dir'] * 100:.2f}%")
