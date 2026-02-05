import argparse
import ast
import json
import os
import sys
import tempfile
import time
from typing import Set

from chemproflow.utils.molecule import fmt_smiles
from chemproflow.taxonomy.db import TaxonomyDb
from chemproflow.utils.cmd import url_download_to_memory
from natsort import natsorted, natsort_keygen
import pandas as pd
from tqdm import tqdm
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-tcid-uniprot-tsv", required=True, help="TCID to Uniprot, tcid_to_uniprot.tsv"
    )
    parser.add_argument(
        "--input-taxonomy-db", required=True, help="Taxonomy database"
    )
    parser.add_argument(
        "--parameter-tmpdir-str", help="Directory to store uniprot.json data"
    )
    parser.add_argument(
        "--output-tcid-uniprot-csv", required=True, help="Output TCID to Uniprot to Taxonomy"
    )
    args = parser.parse_args()

    file_tcdb_uniprot_tsv = args.input_tcid_uniprot_tsv
    file_taxonomy_db = args.input_taxonomy_db
    file_tcdb_uniprot_csv = args.output_tcid_uniprot_csv

    if args.parameter_tmpdir_str is None:
        dir_uniprot = tempfile.mkdtemp()
    else:
        dir_uniprot = args.parameter_tmpdir_str
    os.makedirs(dir_uniprot, exist_ok=True)

    print("Initialize Taxonomy database")
    tax_db = TaxonomyDb(path=file_taxonomy_db)

    print("Parse TCDB Uniprot file")
    df = pd.read_csv(file_tcdb_uniprot_tsv, sep="\t", names=["uniprot", "tcid"])

    # Parse Uniprot
    print("Map Genome Id and Uniprot")
    uniprots = df["uniprot"].dropna().unique().tolist()

    # Download
    print("Download")
    for uniprot_id in tqdm(uniprots, total=len(uniprots)):
        file_uniprot_json = os.path.join(dir_uniprot, f"{uniprot_id}.json")
        if os.path.isfile(file_uniprot_json):
            continue
        data, code = url_download_to_memory(url=f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json")
        result = {}
        if data:
            result = json.load(data)
        with open(file_uniprot_json, "w") as fd:
            json.dump(result, fd)
        time.sleep(0.1)

    # Parse
    print("Parse")
    datas = []
    for uniprot_id in tqdm(uniprots, total=len(uniprots)):
        file_uniprot_json = os.path.join(dir_uniprot, f"{uniprot_id}.json")
        if not os.path.isfile(file_uniprot_json):
            continue
        result = json.load(open(file_uniprot_json))
        if len(result) > 0 and "primaryAccession" in result:
            data = dict(uniprot=result["primaryAccession"], taxid=result.get("organism", {}).get("taxonId", -1))
            datas.append(data)
    df_uni = pd.DataFrame(datas)

    print("Build taxonomy")
    df_uni = tax_db.build_lineage(df=df_uni)

    print("Merge data")
    df = df.merge(df_uni, on="uniprot", how="left")

    print("Save")
    df.to_csv(file_tcdb_uniprot_csv, index=False)

    print("End")