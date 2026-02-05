import argparse
import os
from typing import Set

import optuna
import numpy as np
import pandas as pd
from chemproflow.homology.rbh import run_rbh
from sklearn import preprocessing
from natsort import natsorted


def encode(stcid_sample, stcid_ref, missing_label='__MISSING__'):
    stcid_all = stcid_sample.union(stcid_ref)
    tcid_all = natsorted(list(stcid_all))

    encoded_sample = [tcid if tcid in stcid_sample else missing_label for tcid in tcid_all]
    encoded_ref    = [tcid if tcid in stcid_ref else missing_label for tcid in tcid_all]

    # Fit encoder on the combined list (shared vocabulary)
    shared_vocab = encoded_sample + encoded_ref
    enc = preprocessing.OrdinalEncoder()
    enc.fit(np.array(shared_vocab).reshape(-1, 1))  # Single feature

    # Encode both using the same encoder
    arr_sample = enc.transform(np.array(encoded_sample).reshape(-1, 1)).flatten()
    arr_ref    = enc.transform(np.array(encoded_ref).reshape(-1, 1)).flatten()

    # Get code for missing label
    missing_value = enc.transform([[missing_label]])[0][0]
    return np.vstack([arr_sample, arr_ref]).T, missing_value

def compute_score(encoded_array, missing_value):
    sample = encoded_array[:, 0]
    ref = encoded_array[:, 1]

    true_positive = (sample != missing_value) & (ref != missing_value) & (sample == ref)
    false_positive = (sample != missing_value) & (ref == missing_value)
    false_negative = (sample == missing_value) & (ref != missing_value)

    tp = true_positive.sum()
    fp = false_positive.sum()
    fn = false_negative.sum()

    f1 = 2*tp / (2*tp + fp + fn)
    p = tp + fn
    sens = tp / p # sensitivity/recall
    prec = tp / (tp + fp)

    def fmt_score(float_np):
        v = float(float_np)
        return round(v, 3)

    return {
        'f1': fmt_score(f1),
        'sensitivity': fmt_score(sens),
        'precision': fmt_score(prec),
    }

def objective(trial, path_genome: str, path_prot: str, tcid_ref: Set, threads: int):
    # Blastp
    evalue = trial.suggest_categorical("evalue", ["1e-12", "1e-11", "1e-10", "1e-9", "1e-8", "1e-7", "1e-6", "1e-5", "1e-4", "1e-3"])
    seg = trial.suggest_categorical("seg", [False, True])
    mask = trial.suggest_categorical("mask", [False, True])
    sw = trial.suggest_categorical("sw", [False, True])
    stats = trial.suggest_categorical("stats", [0, 1, 2, 3])
    cov_hsp = trial.suggest_categorical("cov_hsp", [20, 30, 40, 50, 60, 70, 80, 90])
    # Filtering
    min_length = trial.suggest_categorical("min_length", [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    # Selection
    col_top = trial.suggest_categorical("col_top", ["bitscore", "evalue"])

    cmd = ["-num_threads", str(threads)]
    cmd += ["-evalue", str(evalue)]
    if seg:
        cmd += ["-seg", "yes"]
        if mask:
            cmd += ["-soft_masking", "true"]
    cmd += ["-comp_based_stats", str(stats)]
    cmd += ["-qcov_hsp_perc", str(cov_hsp)]
    if sw:
        cmd += ["-use_sw_tback"]
    df_rbh = run_rbh(path_genome=path_genome, path_ref=path_prot, args_blast=cmd, min_length=min_length, col_top=col_top)

    df_rbh["tcid"] = df_rbh["qseqid"].apply(lambda x: x.split("|")[-1])
    ar, missing_value = encode(set(df_rbh["tcid"].tolist()), tcid_ref)

    data = compute_score(ar, missing_value)
    return data["f1"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-bifidobacterium-xlsx", required=True, help="Bifidobacterium dataset excel file"
    )
    parser.add_argument(
        "--input-genome-fasta", required=True, help="Genome file, faa"
    )
    parser.add_argument(
        "--input-tcdb-fasta", required=True, help="TCDB proteins fasta file"
    )
    parser.add_argument(
        "--parameter-sheet-name-str", default="Bli", help="Sheet name"
    )
    parser.add_argument(
        "--parameter-thread-int", default=1, type=int, help="Threads"
    )

    args = parser.parse_args()

    # Init
    file_bifidobacterium_xlsx = args.input_bifidobacterium_xlsx
    file_genome_fasta = args.input_genome_fasta
    file_tcdb_fasta = args.input_tcdb_fasta
    sheet_name = args.parameter_sheet_name_str
    threads = args.parameter_thread_int

    df = pd.read_excel(file_bifidobacterium_xlsx, sheet_name=sheet_name)
    tcid_ref = set(df["Hit TCID"].tolist())

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(lambda trial: objective(trial, file_genome_fasta, file_tcdb_fasta, tcid_ref, threads), n_trials=50)

    print("Best hyperparameters:", study.best_params)

