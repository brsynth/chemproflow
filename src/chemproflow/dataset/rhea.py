import argparse
import re
from typing import List

from chemproflow.dataset.ontology import Ontology
import numpy as np
import pandas as pd
from pybiopax import api
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


fpgen = AllChem.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
regex = re.compile(r"(CHEBI:\d+)")


def parse_side_reaction(model, items) -> List[str]:
    chebi_ids = []
    for mol in items:
        if mol.cellular_location:
            entity_reference = model.objects.get(mol.entity_reference.uid)
            for comment in entity_reference.comment:
                match = re.search(regex, comment)
                if match:
                    chebi_id = match.group(1)
                    display_name = mol.display_name
                    label = f"{chebi_id};{display_name}"
                    chebi_ids.append(label)
                    break
    return chebi_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-chebi-owl", required=True, help="ChEBI ontology",
    )
    parser.add_argument(
        "--input-rhea-biopax-owl", required=True, help="Rhea Biopax, rhea-biopax.owl"
    )
    parser.add_argument(
        "--input-rhea-sprot-tsv", required=True, help="Rhea vs uniprot, rhea2uniprot_sprot.tsv"
    )
    parser.add_argument(
        "--input-rhea-trembl-tsv", required=True, help="Rhea vs uniprot, rhea2uniprot_trembl.tsv"
    )
    parser.add_argument(
        "--input-tcdb-uniprot-tsv", required=True, help="TCDB vs uniprot, tcid_to_uniprot.tsv"
    )
    parser.add_argument(
        "--output-dataset-tsv", required=True, help="Output file tsv, formatted in a same way than get_substrates.tsv file"
    )

    args = parser.parse_args()

    # Init
    file_input_owl = args.input_chebi_owl
    file_rhea_biopax_owl = args.input_rhea_biopax_owl
    file_rhea_sprot_tsv = args.input_rhea_sprot_tsv
    file_rhea_trembl_tsv = args.input_rhea_trembl_tsv
    file_tcdb_uniprot_tsv = args.input_tcdb_uniprot_tsv
    file_dataset_tsv = args.output_dataset_tsv

    ## Process
    print("Read ontology")
    ontology = Ontology(path=file_input_owl)

    print("Parse BioPax")
    model = api.model_from_owl_file(file_rhea_biopax_owl)

    print("Get Transport reactions")
    datas = []
    for obj in model.objects.values():
        if "transport" not in type(obj).__name__.lower():
            continue
        rhea_id = obj.uid.split("/")[-1]

        chebi_ids = set()
        left_sides = parse_side_reaction(model=model, items=obj.left)
        right_sides = parse_side_reaction(model=model, items=obj.right)

        # Parsing
        mols = []
        for side in left_sides + right_sides:
            chebi_id = side.split(";")[0]
            comp = "in"
            if "(out)" in side:
                comp = "out"
            data = dict(chebi_id=chebi_id, smiles=ontology.get_smiles(chebi_id), comp=comp)
            mols.append(data)
        ins = [x for x in mols if x["comp"] == "in"]
        outs = [x for x in mols if x["comp"] == "out"]

        if len(left_sides) == len(right_sides) == 1:
            chebi_ids.update(left_sides)
            chebi_ids.update(right_sides)
        elif sorted([x["chebi_id"] for x in ins]) == sorted([x["chebi_id"] for x in outs]):
            chebi_ids.update(left_sides)
        else:
            # Init matrix
            matrix = np.zeros((len(ins), len(outs)), dtype=float)
            for idx, din in enumerate(ins):
                for jdx, dout in enumerate(outs):
                    if din["smiles"] and dout["smiles"]:
                        molin = Chem.MolFromSmiles(din["smiles"])
                        molout = Chem.MolFromSmiles(dout["smiles"])
                        if molin and molout:
                            fpin = fpgen.GetCountFingerprint(molin)
                            fpout = fpgen.GetCountFingerprint(molout)
                            value = DataStructs.TanimotoSimilarity(fpin, fpout)
                            matrix[idx, jdx] = value
            df_overlap = pd.DataFrame(matrix, index=[x["chebi_id"] for x in ins], columns=[x["chebi_id"] for x in outs])
            if df_overlap.empty:
                continue
            print("-" * 100)
            print("Rhea id:", rhea_id)
            print("left sides:", left_sides)
            print("right sides:", right_sides)
            print("ins:", ins)
            print("outs:", outs)
            print("overlap")
            print(df_overlap)
            df_overlap[df_overlap < 0.1] = 0.
            if df_overlap.max().max() == 0:
                continue
            # drop columns if there is only zero
            df_overlap = df_overlap.drop(columns=df_overlap.columns[df_overlap.sum() == 0])
            ser = df_overlap.idxmax()
            if not ser.empty:
                short_chebi_ids = set()
                short_chebi_ids.update(ser.index)
                short_chebi_ids.update(ser.values)
                for side in left_sides + right_sides:
                    if side.split(";")[0] in short_chebi_ids:
                        chebi_ids.add(side)
            print("chebi ids")
            print(chebi_ids)
            print("-" * 100)

        if chebi_ids:
            data = dict(rhea_id=rhea_id, chebi_id="|".join(sorted(chebi_ids)))
            datas.append(data)

    df = pd.DataFrame(datas)
    df["rhea_id"] = df["rhea_id"].astype(int)

    print("Associate Rhea <-> Uniprot")
    df_sprot = pd.read_csv(file_rhea_sprot_tsv, sep="\t")
    df_trembl = pd.read_csv(file_rhea_trembl_tsv, sep="\t")
    df_uni = pd.concat([df_sprot, df_trembl])
    df_uni.rename(columns={"RHEA_ID": "rhea_id", "ID": "uniprot_id"}, inplace=True)
    df_uni.drop(columns=["DIRECTION", "MASTER_ID"], inplace=True)
    df_uni["rhea_id"] = df_uni["rhea_id"].astype(int)
    df = df.merge(df_uni, how="left", on="rhea_id") # columns: ["rhea_id", "chebi_id", "uniprot_id"]
    df = df[~pd.isna(df["uniprot_id"])]

    print("Associate Uniprot <-> TCID")
    df_tcdb = pd.read_csv(file_tcdb_uniprot_tsv, sep="\t", names=["uniprot_id", "tcdb_id"])
    df = df.merge(df_tcdb, how="left", on="uniprot_id") # columns: ["rhea_id", "chebi_id", "uniprot_id", "tcdb_id"]

    print("Save")
    df = df[["tcdb_id", "chebi_id"]]
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.to_csv(file_dataset_tsv, sep="\t", index=False, header=False)

    print("End")
