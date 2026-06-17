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

class ModelTransportInfer:
    def __init__(
            self,
            file_dataset_transport_csv: str,
            file_model_transport_pkl: str,
            file_encoder_transport_pkl: str,
            file_dirichlet_calibrator_pkl: str,
        ):
        self.df_dataset = pd.read_csv(file_dataset_transport_csv)
        self.model = ModelTransport.load_from_checkpoint(file_model_transport_pkl)
        self.model.eval()
        with open(file_encoder_transport_pkl, "rb") as f:
            self.encoder = pickle.load(f)
        with open(file_dirichlet_calibrator_pkl, "rb") as f:
            self.dirichlet_bundle = pickle.load(f)
        self.device = next(self.model.parameters()).device

    def in_dataset(self, smiles: str | List[str]) -> List[bool]:
        if isinstance(smiles, str):
            smiles = [smiles]
        preds = []
        for smi in smiles:
            mask = smi == self.df_dataset['smiles']
            preds.append(mask.any())
        return preds

    @classmethod
    def dirichlet_feature_map(cls, probabilities, eps=1e-6):
        probs_clipped = np.clip(probabilities, eps, 1 - eps)
        return np.column_stack((np.log(probs_clipped), np.log(1 - probs_clipped), probs_clipped))
        
    def predict(self, smiles: str | List[str], batch_size: int = 16) -> List[bool]:
        if isinstance(smiles, str):
            smiles = [smiles]

        n = len(smiles)
        loader = build_loader(smiles, batch_size=batch_size, to_fmt=False)
        logits_list = []
        orig_indices = []
        with torch.no_grad():
            for batch in tqdm(loader, total=len(loader)):
                batch = batch.to(self.device)
                logits = self.model(batch).squeeze(-1)  # [B]
                logits_list.append(logits.cpu())
                orig_indices.append(batch.orig_idx.cpu())

        preds = np.full(n, None, dtype=object)
        if not logits_list:
            return list(preds)

        logits = torch.cat(logits_list, dim=0)  # [N_valid]
        orig_indices = torch.cat(orig_indices, dim=0).numpy()  # [N_valid]
        probs = torch.sigmoid(logits)
        probs = self.model.elkan_correct_probs(probs).numpy()

        dirichlet_clf = self.dirichlet_bundle["model"]
        dirichlet_threshold = self.dirichlet_bundle["threshold"]
        dirichlet_features = self.dirichlet_feature_map(probs)
        dirichlet_probs = dirichlet_clf.predict_proba(dirichlet_features)[:, 1]
        preds_valid = (dirichlet_probs >= dirichlet_threshold).astype(int)
        preds_valid = self.encoder.inverse_transform(preds_valid.reshape(-1, 1)).reshape(-1)
        preds[orig_indices] = preds_valid
        return list(preds)

class ModelTcidInfer:
    def __init__(
            self,
            file_dataset_tcid_csv: str,
            file_model_tcid_pkl: str,
            file_encoder_tcid_pkl: str,
            file_threshold_tcid_json: str,
        ):
        self.df_dataset = pd.read_csv(file_dataset_tcid_csv)
        self.model = ModelTcid.load_from_checkpoint(file_model_tcid_pkl)
        self.model.eval()
        with open(file_encoder_tcid_pkl, "rb") as f:
            self.encoder = pickle.load(f)
        data_thresholds = read_json(path=file_threshold_tcid_json)
        thresholds = [data_thresholds[label] for label in self.encoder.classes_]
        self.thresholds = np.asarray(thresholds, dtype=np.float32).reshape(1, -1)
        self.device = next(self.model.parameters()).device

    def in_dataset(self, smiles: str | List[str]) -> List[bool]:
        if isinstance(smiles, str):
            smiles = [smiles]
        preds = []
        for smi in smiles:
            mask = smi == self.df_dataset['smiles']
            preds.append(mask.any())
        return preds

    def predict(self, smiles: str | List[str], batch_size: int = 16) -> List[str]:
        if isinstance(smiles, str):
            smiles = [smiles]
        n = len(smiles)
        loader = build_loader(smiles, batch_size=batch_size, to_fmt=False)
        tcids = [None] * n
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                logits = self.model(batch)  # shape: [batch, labels]
                probs = torch.sigmoid(logits)
                preds = (probs.cpu().numpy() >= self.thresholds).astype(int)
                y_vals = self.encoder.inverse_transform(preds)  # shape: [batch, labels]
                for i, orig_idx in enumerate(batch.orig_idx.cpu().tolist()):
                    tcids[orig_idx] = y_vals[i]
        return tcids

class CatalogTcid:

    def __init__(
        self,
        file_catalog_micro_organisms_csv: str,
        file_tcid_equivalent_json: str,
    ):
        self.df_catalog = pd.read_csv(file_catalog_micro_organisms_csv)  # columns = ['accession', 'tcids']
        self.df_catalog["tcids"] = self.df_catalog["tcids"].apply(ast.literal_eval)

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

        self.df_catalog["tcids"] = self.df_catalog["tcids"].apply(expand_tcids)

    def map_tcids_organisms(self, tcids: List[str|int], value: str = "accession") -> List[List[str | int]]:
        ids = []
        for tcid in tcids:
            mask = self.df_catalog['tcids'].apply(lambda x: tcid in x)
            ids.append(list(set(self.df_catalog[mask][value])))
        return ids