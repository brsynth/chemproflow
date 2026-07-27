import argparse
import json
import os
import pickle
import random
from typing import Dict, Optional

import deepchem as dc
import numpy as np
import pandas as pd
import torch

from deepchem.models import AttentiveFPModel
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
)

## Utilities
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def fmt_value_three(value: float) -> str:
    return "{:.3f}".format(value)

def fmt_value_six(value: float) -> str:
    return "{:.6f}".format(value)

def fmt_value_sci(value: float) -> str:
    return "{:.6e}".format(value)

## Elkan-noto
def estimate_elkan_noto_c_from_deepchem(model, calib_dataset):
    """
    Estimate c = P(s=1 | y=1) using held-out labeled positives.
    """
    proba = predict_positive_class_proba(model, calib_dataset)
    c = np.mean(proba)
    return float(np.clip(c, 0.0, 1.0))

def estimate_elkan_noto_c_from_sklearn(model, X_pos_holdout):
    """
    Estimate c = P(s=1 | y=1) using held-out labeled positives.
    """
    proba = model.predict_proba(X_pos_holdout)[:, 1]
    c = np.mean(proba)
    return float(np.clip(c, 0.0, 1.0))

def apply_elkan_noto(proba_s, c):
    """Correct PU scores P(s=1|x) into approx P(y=1|x) via Elkan-Noto: divide by c, clip to [0, 1]."""
    return np.clip(np.asarray(proba_s, dtype=float) / c, 0.0, 1.0)

def normalize_label(label):
    if torch.is_tensor(label):
        label = label.detach().cpu().numpy()
    label = np.asarray(label).squeeze().item()
    if label == -1:
        label = 0
    return int(label)

def softmax(logits, axis=-1):
    logits = np.asarray(logits, dtype=float)
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)

def predict_positive_class_proba(model, dataset):
    """Return DeepChem binary-class probabilities for class 1."""
    proba = np.asarray(model.predict(dataset))

    if proba.ndim == 3:
        # DeepChem classification shape: (samples, tasks, classes).
        if proba.shape[-1] < 2:
            raise ValueError(f"Expected at least 2 classes, got prediction shape {proba.shape}.")
        if np.any(proba < 0.0) or np.any(proba > 1.0):
            proba = softmax(proba, axis=-1)
        return proba[:, 0, 1]

    if proba.ndim == 2:
        if proba.shape[1] == 2:
            if np.any(proba < 0.0) or np.any(proba > 1.0):
                proba = softmax(proba, axis=1)
            return proba[:, 1]
        if proba.shape[1] == 1:
            proba = proba[:, 0]
            if np.any(proba < 0.0) or np.any(proba > 1.0):
                proba = 1.0 / (1.0 + np.exp(-proba))
            return proba

    raise ValueError(f"Unsupported prediction shape from DeepChem model: {proba.shape}.")

def is_valid_graph_feature(feat):
    if feat is None:
        return False
    if isinstance(feat, np.ndarray) and feat.size == 0:
        return False
    return hasattr(feat, "node_features")

def parse_pt_for_sklearn(path: str, cache: Optional[Dict] = None):
    datas = torch.load(path, weights_only=False)
    print("Count data:", len(datas))
    labels, features, smiles_list = [], [], []
    for data in tqdm(datas, total=len(datas)):
        # Check cache
        feats = np.zeros((1,), dtype=np.float32)
        if cache is not None and data.smiles in cache:
            feats = cache[data.smiles]
        else:
            mol = Chem.MolFromSmiles(data.smiles)
            if mol is None:
                continue
            fp = generator.GetFingerprint(mol)
            feats = np.zeros((2048,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, feats)
            if cache is not None:
                cache[data.smiles] = feats
        # Check results
        if len(feats) == 1:
            continue
        label = data.y
        label = normalize_label(label)
        labels.append(label)
        features.append(feats)
        smiles_list.append(data.smiles)
    X = np.stack(features).astype(np.float32)
    y = np.asarray(labels, dtype=int).squeeze()
    smiles = np.asarray(smiles_list, dtype=object)
    return X, y, smiles

def parse_pt_for_deepchem(path: str, cache: Optional[Dict] = None):
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    datas = torch.load(path, weights_only=False)
    print("Count data:", len(datas))
    features, labels, smiles_list = [], [], []
    skipped = 0
    for data in tqdm(datas, total=len(datas)):
        if cache is not None and data.smiles in cache:
            feat = cache[data.smiles]
        else:
            feat = featurizer.featurize([data.smiles])[0]
            if not is_valid_graph_feature(feat):
                skipped += 1
                continue
            if cache is not None:
                cache[data.smiles] = feat
        features.append(feat)
        labels.append(normalize_label(data.y))
        smiles_list.append(data.smiles)

    if skipped:
        print(f"Skipping {skipped} molecules that DeepChem could not featurize.")
    if not features:
        raise ValueError(f"No valid molecules could be featurized from {path}.")

    X = np.empty(len(features), dtype=object)
    X[:] = features
    labels = np.asarray(labels, dtype=np.int64).reshape(-1, 1)
    # ids carries the smiles through select_dataset() so predictions stay
    # traceable back to molecules for cross-environment comparison.
    ids = np.asarray(smiles_list, dtype=object)

    return dc.data.NumpyDataset(X=X, y=labels, ids=ids)

def select_dataset(dataset, indices):
    indices = np.asarray(indices, dtype=int)
    return dc.data.NumpyDataset(
        X=dataset.X[indices],
        y=dataset.y[indices],
        w=dataset.w[indices] if dataset.w is not None else None,
        ids=dataset.ids[indices] if dataset.ids is not None else None,
    )

def with_balanced_binary_weights(dataset):
    y = dataset.y.reshape(-1).astype(int)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return dataset

    class_weights = {
        cls: len(y) / (len(classes) * count)
        for cls, count in zip(classes, counts)
    }
    # IMPORTANT: weights must be 1-D, shape (N,), NOT (N, 1).
    #
    # For a single-task classifier, DeepChem's per-sample loss
    # (CrossEntropyLoss(reduction='none')) has shape (batch,). DeepChem's
    # _StandardLoss only reshapes the weight tensor when
    # `len(w.shape) < len(losses.shape)`. With a (batch, 1) weight tensor that
    # condition is False (2 < 1), so `losses * w` broadcasts (batch,) against
    # (batch, 1) into a (batch, batch) matrix. After `.mean()` this equals
    # mean(losses) * mean(weights) -- i.e. every sample is scaled by the SAME
    # constant and the class balancing is silently destroyed. Providing (N,)
    # weights makes the multiply element-wise and the balancing effective.
    weights = np.asarray([class_weights[label] for label in y], dtype=np.float32).reshape(-1)
    return dc.data.NumpyDataset(
        X=dataset.X,
        y=dataset.y,
        w=weights,
        ids=dataset.ids,
    )

def score_binary_predictions(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)
    return {
        "acc": accuracy_score(y_true=y_true, y_pred=y_pred),
        "f1": f1_score(y_true=y_true, y_pred=y_pred, zero_division=0),
        "prec": precision_score(y_true=y_true, y_pred=y_pred, zero_division=0),
        "rec": recall_score(y_true=y_true, y_pred=y_pred, zero_division=0),
    }

def tune_threshold(y_true, y_score):
    """Grid-search a threshold maximizing F1 -- same mechanism as
    chemproflow.pu.train (best_th_uncal/best_th_dir there): a fixed,
    evenly spaced grid over (0.01, 0.99), argmax on F1 alone."""
    y_score = np.asarray(y_score, dtype=float)
    thresholds = np.linspace(0.01, 0.99, 99)
    f1_scores = [
        f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
        for t in thresholds
    ]
    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx])
    best_metrics = score_binary_predictions(y_true, y_score, threshold=best_threshold)
    return best_threshold, best_metrics

def safe_average_precision(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))

def safe_roc_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))

def score_summary(y_score):
    y_score = np.asarray(y_score, dtype=float)
    return {
        "score_min": float(np.min(y_score)),
        "score_p25": float(np.quantile(y_score, 0.25)),
        "score_median": float(np.median(y_score)),
        "score_p75": float(np.quantile(y_score, 0.75)),
        "score_max": float(np.max(y_score)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-chemproflow-str", required=True, help="ChemProFlow transport_vs_unlabeled directory")
    parser.add_argument("--output-dir-str", required=True, help="Output directory")
    parser.add_argument("--parameter-seed-int", default=42, type=int, help="Seed")
    parser.add_argument("--parameter-deepchem-epochs-int", default=25, type=int, help="Max DeepChem epochs")
    parser.add_argument("--parameter-batch-size-int", default=128, type=int, help="Batch size")
    #parser.add_argument(
    #    "--no-deepchem-balanced-weights",
    #    action="store_false",
    #    dest="deepchem_balanced_weights",
    #    help="Disable class-balanced DeepChem training weights",
    #)
    #parser.set_defaults(deepchem_balanced_weights=True)
    args = parser.parse_args()

    input_chemproflow_str = args.input_chemproflow_str
    output_dir_str = args.output_dir_str
    seed = args.parameter_seed_int
    deepchem_epochs = args.parameter_deepchem_epochs_int
    batch_size = args.parameter_batch_size_int
    #deepchem_balanced_weights = args.deepchem_balanced_weights

    set_seed(seed)

    print("DeepChem")
    cache = {}
    test_dataset = parse_pt_for_deepchem(path=os.path.join(input_chemproflow_str, "test.pt"), cache=cache)
    y_test = test_dataset.y.reshape(-1).astype(int)
    datas = []
    attentivefp_records = []
    for kfold_ix in range(5):
        kfold_dir = os.path.join(input_chemproflow_str, f"kfold-{kfold_ix}")
        train_dataset = parse_pt_for_deepchem(path=os.path.join(kfold_dir, "train.pt"), cache=cache)
        #if deepchem_balanced_weights:
        #    train_dataset = with_balanced_binary_weights(train_dataset)
        valid_dataset = parse_pt_for_deepchem(path=os.path.join(kfold_dir, "valid.pt"), cache=cache)
        calib_dataset = parse_pt_for_deepchem(path=os.path.join(kfold_dir, "calibration.pt"), cache=cache)

        # Elkan-Noto needs labeled positives held out from training to estimate c.
        y_calib = calib_dataset.y.reshape(-1).astype(int)
        positive_idx = np.where(y_calib == 1)[0]
        if len(positive_idx) == 0:
            raise ValueError("Cannot estimate Elkan-Noto c: calibration dataset has no positive labels.")
        calib_dataset_pos = select_dataset(calib_dataset, positive_idx)

        fold_dir = os.path.join(output_dir_str, "attentivefp", f"kfold-{kfold_ix}")
        best_model_dir = os.path.join(fold_dir, "best_deepchem_model")
        os.makedirs(best_model_dir, exist_ok=True)
        batch_size = 64
        model = AttentiveFPModel(
            mode="classification",
            n_tasks=1,
            batch_size=batch_size,
            learning_rate=0.001,
            model_dir=fold_dir,
        )

        # Best-epoch selection via DeepChem's ValidationCallback (replaces a manual
        # per-epoch fit loop). It evaluates the validation set every `interval` training
        # steps -- one epoch's worth of batches here -- and writes the best-scoring weights
        # to best_model_dir. The monitored metric is PR-AUC: a threshold-free ranking metric
        # well suited to imbalanced PU data (it is the validation signal the old loop ranked
        # on). save_on_minimum=False because higher PR-AUC is better. deterministic=False
        # shuffles batches each epoch; set_seed() anchors reproducibility, and fixed-order
        # batches on imbalanced PU data would bias SGD toward the majority class.
        steps_per_epoch = max(1, (len(train_dataset) + batch_size - 1) // batch_size)
        valid_callback = dc.models.ValidationCallback(
            valid_dataset,
            interval=steps_per_epoch,
            metrics=[dc.metrics.Metric(dc.metrics.f1_score, mode="classification")],
            save_dir=best_model_dir,
            save_on_minimum=False,
        )
        model.fit(
            train_dataset,
            nb_epoch=deepchem_epochs,
            deterministic=False,
            callbacks=[valid_callback],
        )
        # Load the best-PR-AUC weights the callback saved (fall back to the final-epoch
        # weights if the metric never improved and nothing was written).
        if model.get_checkpoints(model_dir=best_model_dir):
            model.restore(model_dir=best_model_dir)

        # --- Elkan-Noto management ---
        # 1. Estimate c = P(s=1|y=1) as the mean predicted positive score over held-out positives.
        # 2. Clamp by min_elkan_c so a tiny c cannot blow the 1/c correction up.
        # 3. Correct every raw score P(s=1|x) into a calibrated P(y=1|x) = clip(P(s=1|x)/c, 0, 1),
        #    and use those corrected scores for thresholding and reporting.
        # AP stays computed on the raw scores: it is a ranking metric and the /c scaling is
        # monotonic, while clipping at 1.0 would needlessly flatten the top of the ranking.
        c_en = estimate_elkan_noto_c_from_deepchem(model, calib_dataset_pos)

        y_valid = valid_dataset.y.reshape(-1).astype(int)
        valid_s = predict_positive_class_proba(model, valid_dataset)
        valid_score = apply_elkan_noto(valid_s, c_en)
        threshold, valid_metrics = tune_threshold(y_valid, valid_score)
        valid_ap = safe_average_precision(y_valid, valid_s)
        valid_roc_auc = safe_roc_auc(y_valid, valid_s)

        test_s = predict_positive_class_proba(model, test_dataset)
        test_score = apply_elkan_noto(test_s, c_en)
        test_metrics = score_binary_predictions(y_test, test_score, threshold=threshold)
        test_ap = safe_average_precision(y_test, test_s)
        test_roc_auc = safe_roc_auc(y_test, test_s)
        test_score_summary = score_summary(test_score)
        data = {
            "model": "attentivefp",
            "kfold": kfold_ix,
            "elkan_noto_c": fmt_value_three(c_en),
            "score_mode": "elkan_noto_corrected",
            "threshold": fmt_value_three(threshold),
            "valid_ap_raw": fmt_value_three(valid_ap),
            "valid_roc_auc_raw": fmt_value_three(valid_roc_auc),
            "valid_acc": fmt_value_three(valid_metrics["acc"]),
            "valid_f1": fmt_value_three(valid_metrics["f1"]),
            "valid_prec": fmt_value_three(valid_metrics["prec"]),
            "valid_rec": fmt_value_three(valid_metrics["rec"]),
            "test_ap_raw": fmt_value_three(test_ap),
            "test_roc_auc_raw": fmt_value_three(test_roc_auc),
            "test_score_min": fmt_value_six(test_score_summary["score_min"]),
            "test_score_p25": fmt_value_six(test_score_summary["score_p25"]),
            "test_score_median": fmt_value_six(test_score_summary["score_median"]),
            "test_score_p75": fmt_value_six(test_score_summary["score_p75"]),
            "test_score_max": fmt_value_six(test_score_summary["score_max"]),
            "test_score_min_sci": fmt_value_sci(test_score_summary["score_min"]),
            "test_score_p25_sci": fmt_value_sci(test_score_summary["score_p25"]),
            "test_score_median_sci": fmt_value_sci(test_score_summary["score_median"]),
            "test_score_p75_sci": fmt_value_sci(test_score_summary["score_p75"]),
            "test_score_max_sci": fmt_value_sci(test_score_summary["score_max"]),
            "acc": fmt_value_three(test_metrics["acc"]),
            "f1": fmt_value_three(test_metrics["f1"]),
            "prec": fmt_value_three(test_metrics["prec"]),
            "rec": fmt_value_three(test_metrics["rec"]),
            "selected": False,
        }
        datas.append(data)
        attentivefp_records.append(
            {
                "kfold": kfold_ix,
                "valid_ap": valid_ap,
                "data": data,
                "test_score": test_score,
                "threshold_raw": threshold,
            }
        )
        model.save_checkpoint()

    # Select the final AttentiveFP model on validation PR-AUC (threshold-free,
    # suited to imbalanced PU data -- consistent with the ValidationCallback
    # metric choice above). The test set is never involved in this choice.
    best_attentivefp_idx = int(np.argmax([r["valid_ap"] for r in attentivefp_records]))
    attentivefp_records[best_attentivefp_idx]["data"]["selected"] = True
    best_attentivefp = attentivefp_records[best_attentivefp_idx]["data"]
    print(
        f"AttentiveFP: selected fold {best_attentivefp_idx} as final model "
        f"(valid PR-AUC={attentivefp_records[best_attentivefp_idx]['valid_ap']:.3f})"
    )

    # Smiles-aligned, framework-agnostic export (plain csv) of the selected
    # model's test predictions -- so a DeepChem-free environment can compare
    # against the other model families without needing this stack installed.
    attentivefp_dir = os.path.join(output_dir_str, "attentivefp")
    os.makedirs(attentivefp_dir, exist_ok=True)
    attentivefp_predictions_df = pd.DataFrame(
        {
            "smiles": test_dataset.ids,
            "y_true": y_test,
            "score": attentivefp_records[best_attentivefp_idx]["test_score"],
            "threshold": attentivefp_records[best_attentivefp_idx]["threshold_raw"],
        }
    )
    assert not attentivefp_predictions_df["smiles"].duplicated().any(), (
        "Duplicate SMILES in test set; downstream cross-environment merge by "
        "smiles would silently corrupt alignment."
    )
    attentivefp_predictions_df.to_csv(
        os.path.join(attentivefp_dir, "test_predictions.csv"), index=False
    )

    print("RF")
    cache = {}
    X_test, y_test, smiles_test = parse_pt_for_sklearn(path=os.path.join(input_chemproflow_str, "test.pt"), cache=cache)
    rf_records = []
    for kfold_ix in range(5):
        kfold_dir = os.path.join(input_chemproflow_str, f"kfold-{kfold_ix}")
        X_train, y_train, _ = parse_pt_for_sklearn(path=os.path.join(kfold_dir, "train.pt"), cache=cache)
        X_valid, y_valid, _ = parse_pt_for_sklearn(path=os.path.join(kfold_dir, "valid.pt"), cache=cache)
        X_calib, y_calib, _ = parse_pt_for_sklearn(path=os.path.join(kfold_dir, "calibration.pt"), cache=cache)

        model = RandomForestClassifier(
            n_estimators=500,
            random_state=seed,
            #class_weight="balanced",
        )

        model.fit(X_train, y_train)
        X_pos_calib = X_calib[y_calib == 1]
        if len(X_pos_calib) == 0:
            raise ValueError("Cannot estimate Elkan-Noto c: calibration dataset has no positive labels.")
        c_en = estimate_elkan_noto_c_from_sklearn(model, X_pos_calib)

        valid_s = model.predict_proba(X_valid)[:, 1]
        valid_score = apply_elkan_noto(valid_s, c_en)
        best_threshold, valid_metrics = tune_threshold(y_valid, valid_score)
        valid_ap = safe_average_precision(y_valid, valid_s)
        valid_roc_auc = safe_roc_auc(y_valid, valid_s)

        test_s = model.predict_proba(X_test)[:, 1]
        test_score = apply_elkan_noto(test_s, c_en)
        test_metrics = score_binary_predictions(y_test, test_score, threshold=best_threshold)
        test_ap = safe_average_precision(y_test, test_s)
        test_roc_auc = safe_roc_auc(y_test, test_s)
        test_score_summary = score_summary(test_score)
        data = {
            "model": "rf",
            "kfold": kfold_ix,
            "elkan_noto_c": fmt_value_three(c_en),
            "score_mode": "elkan_noto_corrected",
            "threshold": fmt_value_three(best_threshold),
            "valid_ap_raw": fmt_value_three(valid_ap),
            "valid_roc_auc_raw": fmt_value_three(valid_roc_auc),
            "valid_acc": fmt_value_three(valid_metrics["acc"]),
            "valid_f1": fmt_value_three(valid_metrics["f1"]),
            "valid_prec": fmt_value_three(valid_metrics["prec"]),
            "valid_rec": fmt_value_three(valid_metrics["rec"]),
            "test_ap_raw": fmt_value_three(test_ap),
            "test_roc_auc_raw": fmt_value_three(test_roc_auc),
            "test_score_min": fmt_value_six(test_score_summary["score_min"]),
            "test_score_p25": fmt_value_six(test_score_summary["score_p25"]),
            "test_score_median": fmt_value_six(test_score_summary["score_median"]),
            "test_score_p75": fmt_value_six(test_score_summary["score_p75"]),
            "test_score_max": fmt_value_six(test_score_summary["score_max"]),
            "test_score_min_sci": fmt_value_sci(test_score_summary["score_min"]),
            "test_score_p25_sci": fmt_value_sci(test_score_summary["score_p25"]),
            "test_score_median_sci": fmt_value_sci(test_score_summary["score_median"]),
            "test_score_p75_sci": fmt_value_sci(test_score_summary["score_p75"]),
            "test_score_max_sci": fmt_value_sci(test_score_summary["score_max"]),
            "acc": fmt_value_three(test_metrics["acc"]),
            "f1": fmt_value_three(test_metrics["f1"]),
            "prec": fmt_value_three(test_metrics["prec"]),
            "rec": fmt_value_three(test_metrics["rec"]),
            "selected": False,
        }
        datas.append(data)
        rf_records.append(
            {
                "kfold": kfold_ix,
                "valid_ap": valid_ap,
                "data": data,
                "test_score": test_score,
                "threshold_raw": best_threshold,
            }
        )

        rf_dir = os.path.join(output_dir_str, "rf", f"kfold-{kfold_ix}")
        os.makedirs(rf_dir, exist_ok=True)
        with open(os.path.join(rf_dir, "model.pkl"), "wb") as fd:
            pickle.dump(model, fd, protocol=pickle.HIGHEST_PROTOCOL)

    # Select the final RF model on validation PR-AUC only -- test set not involved.
    best_rf_idx = int(np.argmax([r["valid_ap"] for r in rf_records]))
    rf_records[best_rf_idx]["data"]["selected"] = True
    best_rf = rf_records[best_rf_idx]["data"]
    print(
        f"RF: selected fold {best_rf_idx} as final model "
        f"(valid PR-AUC={rf_records[best_rf_idx]['valid_ap']:.3f})"
    )

    rf_predictions_df = pd.DataFrame(
        {
            "smiles": smiles_test,
            "y_true": y_test,
            "score": rf_records[best_rf_idx]["test_score"],
            "threshold": rf_records[best_rf_idx]["threshold_raw"],
        }
    )
    assert not rf_predictions_df["smiles"].duplicated().any(), (
        "Duplicate SMILES in test set; downstream cross-environment merge by "
        "smiles would silently corrupt alignment."
    )
    rf_predictions_df.to_csv(os.path.join(output_dir_str, "rf", "test_predictions.csv"), index=False)

    df = pd.DataFrame(datas)
    df.to_csv(os.path.join(output_dir_str, "results.csv"), index=False)

    # Per-model final pick, chosen on validation only. results.csv keeps every
    # fold's test row for inspection (via the "selected" column), but this file
    # is the one honest, non-cherry-picked test score per model family.
    final_summary = {
        "attentivefp": {
            "selected_fold": best_attentivefp_idx,
            "selection_criterion": "valid_ap",
            **{k: v for k, v in best_attentivefp.items() if k != "selected"},
        },
        "rf": {
            "selected_fold": best_rf_idx,
            "selection_criterion": "valid_ap",
            **{k: v for k, v in best_rf.items() if k != "selected"},
        },
    }
    with open(os.path.join(output_dir_str, "final_summary.json"), "w") as fd:
        json.dump(final_summary, fd, indent=2)

    print("End")
