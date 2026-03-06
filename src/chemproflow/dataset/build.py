import argparse
import ast
import json
import os
import sys
from typing import Set

from biorgroup.pubchem.compound import Compound
from chemproflow.utils.molecule import fmt_smiles
from natsort import natsorted, natsort_keygen
import pandas as pd
from tqdm import tqdm
from rdkit import Chem, RDLogger
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
sys.path.append(os.path.join(os.environ['CONDA_PREFIX'],'share','RDKit','Contrib'))
from NP_Score import npscorer

RDLogger.DisableLog("rdApp.*")


def tcid_dataset(df: pd.DataFrame, path_out: str, path_group: str, count: int):
    print("Group smiles in a set")
    df = df.groupby("tcid", as_index=False).agg({"smiles": set})
    df["count"] = df["smiles"].apply(len)

    print("Sort tcid")
    natsort_key = natsort_keygen()
    df = df.sort_values(by="tcid", key=lambda x: x.map(natsort_key)).reset_index(
        drop=True
    )

    print("Deduplicate based on smiles")
    tcid_groups = (
        df.groupby(df["smiles"].map(frozenset))["tcid"]
        .apply(list)
        .tolist()
    )
    stats = dict(tcid_groups=tcid_groups)
    with open(path_group, "w") as fd:
        json.dump(stats, fd)

    df["smiles"] = df["smiles"].map(frozenset)
    df.drop_duplicates("smiles", inplace=True)

    print("Filter, min item by class")
    df = df[df["count"] > count]

    print("Format")
    datas = []
    for _, row in df.iterrows():
        for smiles in row["smiles"]:
            datas.append(dict(tcid=row["tcid"], smiles=smiles))
    df = pd.DataFrame(datas)

    print("Save")
    df.to_csv(path_out, index=False)


def filter_smiles(smiles: Set[str]):
    n_smiles = set()
    for smi in smiles:
        for s in smi.split("."):
            if "*" in s:
                continue
            m = Chem.MolFromSmiles(s)
            if m:
                n_smiles.add(s)
    return n_smiles

np_model = npscorer.readNPModel()
def compute_np_likeness(mol):
    score = npscorer.scoreMolWConfidence(mol, np_model)
    return (score.nplikeness, score.confidence)


def find_pubchem(path: str, visited: Set) -> pd.DataFrame:
    engine = create_engine(f"sqlite:///{path}")
    Session = sessionmaker(bind=engine)
    session = Session()

    print("Parse pubchem")
    queries = (
        session.query(Compound.cid, Compound.inchi)
        .yield_per(200_000)
        .order_by(Compound.cid)
    )

    total = len(visited)
    datas = set()
    for query in tqdm(queries):
        if query[1] in visited:
            continue
        mol = Chem.MolFromInchi(query[1])
        if mol is None:
            continue
        for smiles in Chem.MolToSmiles(mol).split("."):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            smiles = fmt_smiles(smiles=smiles)
            if smiles in visited:
                continue
            if npscorer.scoreMol(mol, np_model) < 0:
                continue
            datas.add(smiles)
            visited.add(smiles)
            if len(datas) + 1 > total:
                df = pd.DataFrame(datas, columns=["smiles"])
                return df
    return pd.DataFrame()


def pu_dataset(df: pd.DataFrame, path_output: str, path_pubchem: str):
    df = pd.DataFrame(set(df["smiles"]), columns=["smiles"])
    df["label"] = "positive"

    df_un = find_pubchem(path=path_pubchem, visited=set(df["smiles"]))
    df_un["label"] = "unlabeled"
    df = pd.concat([df, df_un])

    df.to_csv(path_output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-substrates-csv", required=True, help="Substrates expanded from ChEBI"
    )
    parser.add_argument(
        "--input-pubchem-sql", required=True, help="PubChem database"
    )
    parser.add_argument(
        "--output-tcid-csv", required=True, help="Dataset file, TCID"
    )
    parser.add_argument(
        "--output-tcid-json", required=True, help="Dataset file, TCID Group"
    )
    parser.add_argument(
        "--output-pu-csv", required=True, help="Dataset file, PU"
    )
    parser.add_argument(
        "--output-expand-csv", required=True, help="Dataset file, expand"
    )
    args = parser.parse_args()

    ## Init
    file_substrates_csv = args.input_substrates_csv
    file_pubchem_db = args.input_pubchem_sql
    file_output_tcid_csv = args.output_tcid_csv
    file_output_tcid_json = args.output_tcid_json
    file_output_pu_csv = args.output_pu_csv
    file_output_expand_csv = args.output_expand_csv

    print("Parse Substrates file")
    df_substrates = pd.read_csv(file_substrates_csv)

    print("Format")
    def expand_to_set(x):
        if pd.isna(x):
            return set()
        return set(ast.literal_eval(x))
    df_substrates["smiles_expanded"] = df_substrates["smiles_expanded"].apply(expand_to_set)
    df_substrates["smiles_expanded_rgroup"] = df_substrates["smiles_expanded_rgroup"].apply(expand_to_set)

    print("Expanded R-group - start")
    datas = set()
    for _, row in tqdm(df_substrates.iterrows(), total=df_substrates.shape[0]):
        for smiles in filter_smiles(smiles=row["smiles_expanded"] | row["smiles_expanded_rgroup"]):
            datas.add((row["tcid"], smiles))

    print("Get first stereo isomer")
    df = pd.DataFrame(datas, columns=["tcid", "smiles"])
    del datas

    print("Format SMILES")
    df_smi = pd.DataFrame(df["smiles"].unique(), columns=["smiles"])
    df_smi["stereo"] = df_smi["smiles"].apply(fmt_smiles)
    df = df.merge(df_smi, on="smiles", how="left")
    df = df[~pd.isna(df["stereo"])]
    df.drop(columns="smiles", inplace=True)
    df.rename(columns={"stereo": "smiles"}, inplace=True)
    df.to_csv(file_output_expand_csv, index=False)

    print("Build pu dataset")
    pu_dataset(df=df, path_output=file_output_pu_csv, path_pubchem=file_pubchem_db)

    print("TCID dataset")
    tcid_dataset(df=df, path_out=file_output_tcid_csv, path_group=file_output_tcid_json, count=49)

    print("End")
