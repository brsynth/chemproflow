import argparse
import ast
import re
from collections import defaultdict

from chemproflow.dataset.ontology import Ontology
import pandas as pd
from tqdm import tqdm


def parse_rgroup(path: str):
    df_rgroup = pd.read_csv(path)
    cols = ["chebi", "rgroup_extended_smiles"]
    df_rgroup = df_rgroup[cols]
    for col in cols:
        df_rgroup[col] = df_rgroup[col].fillna("[]").apply(ast.literal_eval)
    data_rgroup = defaultdict(set)
    for _, row in df_rgroup.iterrows():
        if len(row["rgroup_extended_smiles"]) < 1:
            continue
        for chebi in row["chebi"]:
            data_rgroup[chebi].update(row["rgroup_extended_smiles"])
            is_deprecated, chebi_id_updated = ontology.check_deprecated(chebi_id=chebi)
            if is_deprecated and chebi_id_updated:
                data_rgroup[chebi_id_updated].update(row["rgroup_extended_smiles"])
    return data_rgroup


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-chebi-owl", required=True, help="ChEBI ontology",
    )
    parser.add_argument(
        "--input-chemproflow-rhea-tcdb-tsv", required=True, help="Substrates file created from the rhea.py file"
    )
    parser.add_argument(
        "--input-tcdb-substrates-tsv", required=True, help="Substrates file from TCDB"
    )
    parser.add_argument(
        "--input-biorgroup-csv", required=True, help="BioRGroup dataset"
    )
    parser.add_argument(
        "--output-substrates-csv", required=True, help="Substrates file"
    )
    args = parser.parse_args()

    ## Init
    file_input_owl = args.input_chebi_owl
    file_chemproflow_rhea_tcdb_tsv = args.input_chemproflow_rhea_tcdb_tsv
    file_substrates_tsv = args.input_tcdb_substrates_tsv
    file_rgroup_csv = args.input_biorgroup_csv
    file_output_csv = args.output_substrates_csv

    ## Process
    print("Read ontology")
    ontology = Ontology(path=file_input_owl)

    print("Read R-group")
    data_rgroup = parse_rgroup(path=file_rgroup_csv)

    print("Read csv file")
    df_chemproflow = pd.read_csv(file_chemproflow_rhea_tcdb_tsv, sep="\t", names=["tcid", "chebi"])
    df_tcdb = pd.read_csv(file_substrates_tsv, sep="\t", names=["tcid", "chebi"])
    df = pd.concat([df_chemproflow, df_tcdb])

    pat = re.compile(r"CHEBI:\d+")
    # format CHEBI:\d+;..|CHEBI:\d+;...
    df["chebis"] = (
        df["chebi"].fillna("")
        .apply(lambda s: list(dict.fromkeys(pat.findall(s))))  # dedup, keep first-seen order
    )
    all_chebis = set().union(*df["chebis"])

    print("Analyse")
    def expand_childs(data, data_rgroup):
        is_deprecated, chebi_id_updated = ontology.check_deprecated(chebi_id=data["chebi"])
        chebi_id = data["chebi"]
        if is_deprecated and chebi_id_updated:
            data["chebis_updated"] = chebi_id_updated
            chebi_id = chebi_id_updated
        else:
            data["chebis_updated"] = chebi_id # Update for traceability
        smiles = ontology.get_smiles(chebi_id=chebi_id)
        data["smiles"] = []
        if smiles:
            data["smiles"] = [smiles]
        lsmiles, lchebis = ontology.expand_all_smiles(chebi_id=chebi_id)
        data["smiles_expanded"] = lsmiles
        data["chebis_expanded"] = lchebis
        lsmiles, lchebis = ontology.expand_all_smiles(chebi_id=chebi_id, data_rgroup=data_rgroup)
        data["smiles_expanded_rgroup"] = lsmiles
        data["chebis_expanded_rgroup"] = lchebis
        return data

    datas = [dict(chebi=chebi) for chebi in all_chebis]
    map_chebi_smiles = {}
    for data in tqdm(datas, total=len(datas)):
            result = expand_childs(data=data, data_rgroup=data_rgroup)
            map_chebi_smiles[result["chebi"]] = result

    print("Parse results")
    OUT_COLS = [
        "chebis_updated",
        "smiles",
        "smiles_expanded",
        "chebis_expanded",
        "smiles_expanded_rgroup",
        "chebis_expanded_rgroup",
    ]

    # optional: normalize your mapping so missing keys don't KeyError
    def _norm_entry(e):
        if not e:
            return {
                "chebis_updated": [],
                "smiles": [],
                "smiles_expanded": [],
                "chebis_expanded": [],
                "smiles_expanded_rgroup": [],
                "chebis_expanded_rgroup": [],
            }
        return {
            "chebis_updated": e.get("chebis_updated", []),
            "smiles": e.get("smiles", []),
            "smiles_expanded": e.get("smiles_expanded", []),
            "chebis_expanded": e.get("chebis_expanded", []),
            "smiles_expanded_rgroup": e.get("smiles_expanded_rgroup", []),
            "chebis_expanded_rgroup": e.get("chebis_expanded_rgroup", []),
        }

    # fast aggregator over a list of CHEBI IDs
    def _aggregate_row(chebi_list):
        acc = {
            "chebis_updated": set(),
            "smiles": set(),
            "smiles_expanded": set(),
            "chebis_expanded": set(),
            "smiles_expanded_rgroup": set(),
            "chebis_expanded_rgroup": set(),
        }
        for c in chebi_list:
            e = _norm_entry(map_chebi_smiles.get(c))
            acc["chebis_updated"].add(e["chebis_updated"])
            acc["smiles"].update(e["smiles"])
            acc["smiles_expanded"].update(e["smiles_expanded"])
            acc["chebis_expanded"].update(e["chebis_expanded"])
            acc["smiles_expanded_rgroup"].update(e["smiles_expanded_rgroup"])
            acc["chebis_expanded_rgroup"].update(e["chebis_expanded_rgroup"])
        # convert sets -> sorted lists once
        return {k: sorted(v) for k, v in acc.items()}

    # vectorized build of all six columns
    df[OUT_COLS] = df["chebis"].apply(_aggregate_row).apply(pd.Series)

    print("Write")
    df.to_csv(file_output_csv, index=False)

    print("End")
