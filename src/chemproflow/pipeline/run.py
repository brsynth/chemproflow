import argparse
import ast
import json
import logging
import os
import pickle
from typing import List

from chemproflow.utils.misc import read_json
from chemproflow.model.dataset import build_loader
from chemproflow.pu.model import ModelTransport
from chemproflow.tcid.model import ModelTcid
from chemproflow.utils.molecule import fmt_smiles
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df_col = pd.DataFrame(columns=["dataset_transport", "dataset_tcid", "pred_transport", "pred_tcid", "id_micro_organisms", "name_micro_organisms"])
    df = pd.concat([df, df_col], axis=1)
    df["pred_tcid"] = [[] for _ in range(len(df))]
    df["accession_micro_organisms"] = [[] for _ in range(len(df))]
    df["smiles"] = df["smiles"].apply(fmt_smiles)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-smiles-csv", type=str, required=True, help="Input file, with \"smiles\" column"
    )
    parser.add_argument(
        "--input-dataset-transport-csv", type=str, required=True, help="Path to transport dataset CSV"
    )
    parser.add_argument(
        "--input-dataset-tcid-csv", type=str, required=True, help="Path to TCID dataset CSV"
    )
    parser.add_argument("--input-model-transport-pkl", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input-encoder-transport-pkl", type=str, required=True, help="Path to encoder pickle")
    parser.add_argument(
        "--input-dirichlet-calibrator-pkl",
        type=str,
        default=None,
        help="Path to Dirichlet calibration pickle (defaults to dirichlet_calibrator.pkl next to the model)",
    )
    parser.add_argument("--input-model-tcid-pkl", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input-encoder-tcid-pkl", type=str, required=True, help="Path to encoder pickle")
    parser.add_argument("--input-threshold-tcid-json", type=str, required=True, help="Path to threshold json")
    parser.add_argument("--input-catalog-micro-organisms-csv", type=str, required=True, help="Path to catalog of micro-organisms")
    parser.add_argument("--input-tcid-equivalent-json", type=str, required=True, help="Path input TCID file")
    parser.add_argument("--output-resuts-csv", type=str, required=True, help="Path to output CSV file")
    args = parser.parse_args()

    # Init
    file_smiles_csv = args.input_smiles_csv
    file_dataset_transport_csv = args.input_dataset_transport_csv
    file_dataset_tcid_csv = args.input_dataset_tcid_csv
    file_model_transport_pkl = args.input_model_transport_pkl
    file_encoder_transport_pkl = args.input_encoder_transport_pkl
    file_dirichlet_calibrator_pkl = (
        args.input_dirichlet_calibrator_pkl
        if args.input_dirichlet_calibrator_pkl
        else os.path.join(os.path.dirname(file_model_transport_pkl), "dirichlet_calibrator.pkl")
    )
    file_model_tcid_pkl = args.input_model_tcid_pkl
    file_encoder_tcid_pkl = args.input_encoder_tcid_pkl
    file_threshold_tcid_json = args.input_threshold_tcid_json
    file_catalog_micro_organisms_csv = args.input_catalog_micro_organisms_csv
    file_tcid_equivalent_json = args.input_tcid_equivalent_json
    file_output_results_csv = args.output_resuts_csv

    batch_size = 16
    
    logging.info("Parse input")
    df_input = pd.read_csv(file_smiles_csv)
    df = prepare_df(df=df_input)

    # Check if input is in dataset - transport: columns = ['smiles', 'activity']
    logging.info(f"Loading transport dataset from {file_dataset_transport_csv}")
    df_transport = pd.read_csv(args.input_dataset_transport_csv)
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking transport dataset"):
        mask = df_transport['smiles'] == row['smiles']
        if mask.any():
            df.at[idx, 'dataset_transport'] = True
        else:
            df.at[idx, 'dataset_transport'] = False
    
    # Check if input is in dataset - TCID: columns = ['tcid', 'smiles']
    logging.info(f"Loading TCID dataset from {file_dataset_tcid_csv}")
    df_tcid = pd.read_csv(args.input_dataset_tcid_csv)
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking TCID dataset"):
        mask = df_tcid['smiles'] == row['smiles']
        if mask.any():
            df.at[idx, 'dataset_tcid'] = True
        else:
            df.at[idx, 'dataset_tcid'] = False

    # Run prediction - transport
    logging.info(f"Loading transport model from {file_model_transport_pkl}")
    model = ModelTransport.load_from_checkpoint(args.input_model_transport_pkl)
    model.eval()
    with open(file_encoder_transport_pkl, "rb") as f:
        encoder = pickle.load(f)
    device = next(model.parameters()).device
    loader = build_loader(df['smiles'].tolist(), batch_size=batch_size, to_fmt=False)

    logits_list = []
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), desc="Predicting transport"):
            batch = batch.to(device)
            logits = model(batch).squeeze(-1)  # [B]
            logits_list.append(logits.cpu())
    logits = torch.cat(logits_list, dim=0)  # [N]
    #logits = logits / float(model.temperature)
    probs = torch.sigmoid(logits)
    probs = model.elkan_correct_probs(probs).numpy()
    
    if not os.path.exists(file_dirichlet_calibrator_pkl):
        raise FileNotFoundError(f"Dirichlet calibrator not found at {file_dirichlet_calibrator_pkl}")

    with open(file_dirichlet_calibrator_pkl, "rb") as f:
        dirichlet_bundle = pickle.load(f)

    dirichlet_clf = dirichlet_bundle["model"]
    dirichlet_threshold = dirichlet_bundle["threshold"]

    def dirichlet_feature_map(probabilities, eps=1e-6):
        probs_clipped = np.clip(probabilities, eps, 1 - eps)
        return np.column_stack((np.log(probs_clipped), np.log(1 - probs_clipped), probs_clipped))

    dirichlet_features = dirichlet_feature_map(probs)
    dirichlet_probs = dirichlet_clf.predict_proba(dirichlet_features)[:, 1]

    preds = (dirichlet_probs >= dirichlet_threshold).astype(int)
    preds = encoder.inverse_transform(preds.reshape(-1, 1)).reshape(-1)
    df['pred_transport'] = preds

    # Run prediction - TCID
    logging.info(f"Loading TCID model from {file_model_tcid_pkl}")
    smiles = df[df["pred_transport"] == "positive"]['smiles'].tolist()
    print("Smiles:", smiles)
    if len(smiles) > 0:
        logging.info(f"Loading TCID model from {file_model_tcid_pkl}")
        model = ModelTcid.load_from_checkpoint(file_model_tcid_pkl)
        model.eval()
        logging.info(f"Loading TCID encoder from {file_encoder_tcid_pkl}")
        with open(file_encoder_tcid_pkl, "rb") as f:
            encoder = pickle.load(f)
        logging.info(f"Loading TCID threshold from {file_threshold_tcid_json}")
        data_thresholds = read_json(path=file_threshold_tcid_json)
        thresholds = [data_thresholds[label] for label in encoder.classes_]
        thresholds = np.asarray(thresholds, dtype=np.float32).reshape(1, -1)

        device = next(model.parameters()).device
        loader = build_loader(smiles, batch_size=batch_size, to_fmt=False)
        tcids = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                # Extract labels from dataset batch
                logits = model(batch)  # shape: [batch, labels]
                probs = torch.sigmoid(logits)
                preds = (probs.cpu().numpy() >= thresholds).astype(int)
                y_vals = encoder.inverse_transform(preds)  # shape: [batch, labels]
                tcids.extend(y_vals)

        # Assign predicted TCIDs to dataframe
        cur_idx = 0
        for idx, row in df.iterrows():
            if row['pred_transport'] == 'positive':
                df.at[idx, 'pred_tcid'] = tcids.pop(0)

    # Augment catalog
    logging.info(f"Loading catalog of micro-organisms from {file_catalog_micro_organisms_csv}")
    df_catalog = pd.read_csv(file_catalog_micro_organisms_csv)  # columns = ['accession', 'tcids']
    df_catalog["tcids"] = df_catalog["tcids"].apply(ast.literal_eval)

    with open(file_tcid_equivalent_json) as fd:
        tcid_equivalent = json.load(fd)
    tcid_groups = tcid_equivalent["tcid_groups"]  # List[List[str]]

    # Build a mapping from tcid to its group (set of equivalents)
    tcid_to_group = {}
    for group in tcid_groups:
        group_set = set(group)
        for tcid in group:
            tcid_to_group[tcid] = group_set

    # Expand each row's tcids with equivalents using the mapping
    def expand_tcids(tcids):
        expanded = set(tcids)
        for tcid in tcids:
            expanded.update(tcid_to_group.get(tcid, []))
        return expanded

    df_catalog["tcids"] = df_catalog["tcids"].apply(expand_tcids)

    # Find micro-organisms
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Finding micro-organisms"):
        if row['pred_tcid']:
            ids = []
            names = []
            for tcid in row['pred_tcid']:
                mask = df_catalog['tcids'].apply(lambda x: tcid in x)
                ids.append(list(set(df_catalog[mask]['accession'])))
            df.at[idx, 'accession_micro_organisms'] = ids
    
    # Save results
    logging.info(f"Saving results to {file_output_results_csv}")
    df.to_csv(file_output_results_csv, index=False)
    
    logging.info("Done.")
