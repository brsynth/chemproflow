import argparse
from collections import defaultdict
import glob
import os
import re
from typing import Dict, List, Optional, Tuple

from chemproflow.taxonomy.db import TaxonomyDb
import pandas as pd
from rdkit import Chem, RDLogger
from tqdm import tqdm
RDLogger.DisableLog("rdApp.*")  # Disable RDKit warnings


class Metabolite(object):
    def __init__(
        self,
        id: Optional[str] = None,
        name: Optional[str] = None,
        inchi: Optional[str] = None,
        inchi_non_std: Optional[str] = None,
        smiles: Optional[str] = None,
        cid: Optional[str] = None,
        metanetx: Optional[str] = None,
        chebi: Optional[str] = None,
        metabolite_ids: Optional[List[str]] = None,
        *args,
        **kwargs,
    ) -> None:
        super(Metabolite, self).__init__(*args, **kwargs)
        self.id = id
        self.name = name
        self.inchi = inchi
        self.inchi_non_std = inchi_non_std
        self.smiles = smiles
        self.cid = cid
        self.metanetx = metanetx
        self.chebi = chebi
        if metabolite_ids is None:
            metabolite_ids = []
        self.metabolite_ids = metabolite_ids

    def get_inchi(self) -> Optional[str]:
        if self.inchi is not None:
            return self.inchi
        if self.inchi_non_std is not None:
            return self.inchi_non_std
        return None

    def is_init(self) -> bool:
        has_common = self.id is not None and self.name is not None
        if self.get_inchi() is not None:
            return True
        if len(self.metabolite_ids) > 0:
            return True
        return False

    @classmethod
    def get_key_value(cls, line: str) -> Tuple[str, str]:
        tline = line.split(" - ")
        if len(tline) < 2:
            return "", ""
        key = tline[0]
        value = tline[1]
        value = value.replace("\n", "")
        return key, value

    @classmethod
    def get_key(cls, line: str) -> str:
        value = line.split(" - ")[0]
        return value

    def to_dict(self) -> Dict[str, str]:
        return dict(
            id=self.id,
            name=self.name,
            inchi=self.inchi,
            smiles=self.smiles,
            cid=self.cid,
            metanetx=self.metanetx,
            chebi=self.chebi,
        )

    def __repr__(self) -> str:
        return str(self.to_dict())

    @classmethod
    def from_file(cls, path: str) -> List["Metabolite"]:
        metabolites = []
        # unitialize = 0
        cid_regex = re.compile(r'PUBCHEM\s"(\d+)"')
        metanetx_regex = re.compile(r'METANETX\s"(\w+)"\s')
        chebi_regex = re.compile(r'CHEBI\s"(\w+)"\s')
        with open(path, encoding="iso-8859-1") as fod:
            metabolite = Metabolite()
            for line in fod:
                if line.startswith("#"):
                    continue
                if line.startswith("//"):
                    # if metabolite.is_init():
                    metabolites.append(metabolite)
                    # else:
                    # print("Warning unitialize:", str(metabolite))
                    #    unitialize += 1
                    metabolite = Metabolite()
                    continue
                if line.startswith("/"):
                    continue
                key, value = Metabolite.get_key_value(line=line)
                if key == "UNIQUE-ID":
                    metabolite.id = value
                elif key == "COMMON-NAME":
                    metabolite.name = value
                elif key == "INCHI":
                    metabolite.inchi = value
                elif key == "NON-STANDARD-INCHI":
                    metabolite.inchi_non_std = value
                elif key == "SMILES":
                    metabolite.smiles = value
                elif key == "COMPONENTS":
                    metabolite.metabolite_ids.append(value)
                elif key == "DBLINKS":
                    m = re.search(cid_regex, value)
                    if m:
                        metabolite.cid = m.group(1)
                    m = re.search(metanetx_regex, value)
                    if m:
                        metabolite.metanetx = m.group(1)
                    m = re.search(chebi_regex, value)
                    if m:
                        metabolite.chebi = m.group(1)
        # print("Unitialize: ", unitialize)
        return metabolites

    @classmethod
    def from_file_link(cls, path: str) -> List["Metabolite"]:
        metabolites = []
        with open(path, encoding="iso-8859-1") as fod:
            for line in fod:
                if line.startswith("#"):
                    continue
                metabolite = Metabolite()
                tab = line.split("\t")
                metabolite.id = tab[0]
                metabolite.inchi = tab[1]
                metabolite.smiles = tab[2]

                metabolites.append(metabolite)
        return metabolites


class Reaction(object):
    def __init__(
        self,
        id: Optional[str] = None,
        name: Optional[str] = None,
        substrates: Optional[List[str]] = None,
        products: Optional[List[str]] = None,
        types: Optional[List[str]] = None,
        compounds_in_two_compartments: Optional[List[str]] = None,
        compounds_only_one_side: Optional[List[str]] = None,
        has_in: bool = False,
        has_out: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super(Reaction, self).__init__(*args, **kwargs)
        self.id = id
        self.name = name
        if substrates is None:
            substrates = []
        self.substrates = substrates
        if products is None:
            products = []
        self.products = products
        if types is None:
            types = []
        if compounds_in_two_compartments is None:
            compounds_in_two_compartments = []
        self.compounds_in_two_compartments = compounds_in_two_compartments
        if compounds_only_one_side is None:
            compounds_only_one_side = []
        self.compounds_only_one_side = compounds_only_one_side
        self.types = types
        self.has_in = has_in
        self.has_out = has_out

    def is_init(self) -> bool:
        if len(self.substrates) > 0 and len(self.products) > 0:
            return True
        return False

    @classmethod
    def get_value(cls, line: str) -> str:
        value = line.split(" - ")[1]
        value = value.replace("\n", "")
        return value

    def to_dict(self) -> Dict[str, str]:
        return dict(
            id=self.id,
            name=self.name,
            substrates=self.substrates,
            products=self.products,
            compounds_in_two_compartments = self.compounds_in_two_compartments,
            compounds_only_one_side=self.compounds_only_one_side,
            types=self.types,
            has_in=self.has_in,
            has_out=self.has_out,
        )

    def __repr__(self) -> str:
        return str(self.to_dict())
    
    @classmethod
    def from_file(cls, path: str) -> List["Reaction"]:
        reactions: List[Reaction] = []
        uninitialized = 0
        # Local helpers attached to each Reaction while parsing
        def make_compound_index():
            # compound -> {sides:Set['LEFT'|'RIGHT'], compartments:Set[str], sides_by_compartment: Dict[str, Set[str]]}
            return defaultdict(lambda: {
                "sides": set(),
                "compartments": set(),
                "sides_by_compartment": defaultdict(set),
            })
        with open(path, encoding="iso-8859-1") as fod:
            reaction = Reaction()
            # Per-reaction parsing scratch
            compound_index = make_compound_index()
            last_lr: Dict[str, str] | None = None  # {"compound": str, "side": "LEFT"|"RIGHT"}

            for raw in fod:
                line = raw.rstrip("\n")
                # Ignore comments
                if line.startswith("#"):
                    continue
                # Record separator
                if line.startswith("//"):
                    if reaction.is_init():
                        # finalize helpers for this reaction
                        Reaction._finalize_compound_helpers(
                            reaction, compound_index
                        )
                        reactions.append(reaction)
                    else:
                        uninitialized += 1
                    # reset for next record
                    reaction = Reaction()
                    compound_index = make_compound_index()
                    last_lr = None
                    continue
                # Core fields (your original behavior)
                if line.startswith("UNIQUE-ID"):
                    reaction.id = Reaction.get_value(line=line)
                    last_lr = None
                    continue
                elif line.startswith("COMMON-NAME"):
                    reaction.name = Reaction.get_value(line=line)
                    last_lr = None
                    continue
                elif line.startswith("TYPES"):
                    reaction.types.append(Reaction.get_value(line=line))
                    last_lr = None
                    continue
                # LEFT / RIGHT compounds
                if line.startswith("LEFT"):
                    cpd = Reaction.get_value(line=line)
                    reaction.substrates.append(cpd)
                    compound_index[cpd]["sides"].add("LEFT")
                    last_lr = {"compound": cpd, "side": "LEFT"}
                    continue
                if line.startswith("RIGHT"):
                    cpd = Reaction.get_value(line=line)
                    reaction.products.append(cpd)
                    compound_index[cpd]["sides"].add("RIGHT")
                    last_lr = {"compound": cpd, "side": "RIGHT"}
                    continue
                # COMPARTMENT lines must be attached to the immediately preceding LEFT/RIGHT
                if line.startswith("^COMPARTMENT"):
                    comp = Reaction.get_value(line=line)
                    if last_lr is not None:
                        cpd = last_lr["compound"]
                        side = last_lr["side"]
                        compound_index[cpd]["compartments"].add(comp)
                        compound_index[cpd]["sides_by_compartment"][comp].add(side)
                        # maintain your has_in / has_out flags
                        if comp == "CCO-IN":
                            reaction.has_in = True
                        elif comp == "CCO-OUT":
                            reaction.has_out = True
                    # Clear after consuming so we don't over-attach stray compartments
                    last_lr = None
                    continue
                # Fallbacks to preserve your original IN/OUT detection if present elsewhere
                if "CCO-IN" in line:
                    reaction.has_in = True
                if "CCO-OUT" in line:
                    reaction.has_out = True
                # Any unrelated line: don’t keep a pending attachment
                last_lr = None
            # EOF: finalize the last record if initialized
            if reaction.is_init():
                Reaction._finalize_compound_helpers(reaction=reaction, compound_index=compound_index)
                reactions.append(reaction)
            else:
                if (reaction.types or reaction.name or reaction.substrates or reaction.products):
                    uninitialized += 1
        # print("Uninitialized: ", uninitialized)
        return reactions

    @staticmethod
    def _finalize_compound_helpers(reaction: "Reaction", compound_index):
        """Compute convenience selections:
           - compounds_in_two_compartments
           - compounds_only_on_right
        """
        in_two = set()
        one_side = set()
        for cpd, info in compound_index.items():
            if len(info["compartments"]) >= 2:
                in_two.add(cpd)
            if info["sides"] == {"RIGHT"} or info["sides"] == {"LEFT"}:
                one_side.add(cpd)
        # attach results (create attributes if they don't exist)
        reaction.compounds_in_two_compartments = list(in_two)
        reaction.compounds_only_one_side = list(one_side)

class Ontology:
    def __init__(
        self,
        id: Optional[str] = None,
        types: Optional[List[str]] = None,
        common_names: Optional[List[str]] = None,
        comment: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        super(Ontology, self).__init__(*args, **kwargs)
        self.id = id
        if types is None:
            types = []
        self.types = types
        if common_names is None:
            common_names = []
        self.common_names = common_names
        self.comment = comment

    def is_init(self) -> bool:
        return self.id is not None
    
    def to_dict(self) -> Dict[str, str]:
        return dict(
            id=self.id,
            types=self.types,
            comment=self.comment,
        )

    def __repr__(self) -> str:
        return str(self.to_dict())

    @classmethod
    def from_file(cls, path: str) -> List["Ontology"]:
        def parse_key_value(line: str):
            # Expected format: KEY - VALUE
            # Strip trailing newline; handle weird encodings upstream via open(..., encoding=...)
            line = line.rstrip("\n")
            # Some files could have multiple ' - ' instances in comments; split once.
            parts = line.split(" - ", 1)
            if len(parts) != 2:
                return None, None
            return parts[0].strip(), parts[1].strip()
        ontologies: List[Ontology] = []
        uninitialized = 0
        current = Ontology()
        last_field: Optional[str] = None  # track the last seen field for '/' continuations
        with open(path, encoding="iso-8859-1") as fod:
            for raw in fod:
                line = raw.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                # Record separator
                if line.startswith("//"):
                    if current.is_init():
                        ontologies.append(current)
                    else:
                        uninitialized += 1
                    current = Ontology()
                    last_field = None
                    continue
                # Continuation line: append to whichever field we last touched
                if line.startswith("/"):
                    continuation = line[1:].lstrip()  # drop leading '/' and left spaces
                    if last_field == "COMMENT":
                        current.comment = (current.comment or "")
                        if current.comment and not current.comment.endswith(" "):
                            current.comment += " "
                        current.comment += continuation
                    elif last_field == "COMMON-NAME":
                        # Extend the last common name (rare, but safe)
                        if current.common_names:
                            if current.common_names[-1] and not current.common_names[-1].endswith(" "):
                                current.common_names[-1] += " "
                            current.common_names[-1] += continuation
                        else:
                            current.common_names.append(continuation)
                    elif last_field == "TYPES":
                        # Types are usually single tokens; still support continuation
                        if current.types:
                            if current.types[-1] and not current.types[-1].endswith(" "):
                                current.types[-1] += " "
                            current.types[-1] += continuation
                        else:
                            current.types.append(continuation)
                    # If last_field is None or unknown, we just ignore the stray continuation.
                    continue
                # New field line
                key, value = parse_key_value(line)
                if key is None:
                    # Line didn't match expected pattern; skip or log
                    last_field = None
                    continue
                if key == "UNIQUE-ID":
                    current.id = value
                elif key == "TYPES":
                    current.types.append(value)
                elif key == "COMMON-NAME":
                    current.common_names.append(value)
                elif key == "COMMENT":
                    current.comment = value
                else:
                    # Unknown key; you could store these in a dict on `current` if desired
                    pass
                last_field = key
        # Finalize last record if file doesn't end with //
        if current.is_init():
            ontologies.append(current)
        else:
            if (current.comment or current.types or current.common_names):
                # If there are partials without an ID you can decide whether to keep or discard.
                uninitialized += 1
        return ontologies
    

def select_taxon(df: pd.DataFrame) -> pd.DataFrame:
    # Attempt to have columns: ["kingdom", "domain", "has_proteome"]
    df = df[((df["kingdom"] == 4751) | (df["domain"] == 2)) & (df["has_proteome"])].copy()
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-tier-dir", required=True, help="Input directory from Biocyc, tier1-tier2"
    )
    parser.add_argument(
        "--input-taxonomy-db", required=True, help="Taxonomy database"
    )
    parser.add_argument(
        "--output-compound-csv", required=True, help="Output compound"
    )
    parser.add_argument(
        "--output-taxonomy-csv", required=True, help="Output taxonomy"
    )
    args = parser.parse_args()

    # Init
    dir_biocyc = args.input_tier_dir
    file_taxonomy_db = args.input_taxonomy_db
    file_compound_csv = args.output_compound_csv
    file_taxonomy_csv = args.output_taxonomy_csv
    
    print("Init taxonomy")
    tax_db = TaxonomyDb(path=file_taxonomy_db)
    
    print("Parse biocyc species.dat")
    datas = []
    for biocyc_path in glob.glob(os.path.join(dir_biocyc, "*")):
        biocyc = os.path.basename(biocyc_path)
        with open(os.path.join(biocyc_path, "species.dat"), encoding="iso-8859-1") as handle:
            lines = handle.read().splitlines()
        taxid = -1
        for line in lines:
            if line.startswith("NCBI-TAXONOMY-ID"):
                try:
                    taxid = int(re.match(r"NCBI-TAXONOMY-ID - (\d+)", line).groups()[0])
                    break
                except Exception:
                    pass
        data = dict(biocyc=biocyc, taxid=taxid)
        datas.append(data)
    df = pd.DataFrame(datas)
    
    print("Build lineage")
    print(df)
    df = tax_db.build_lineage(df=df)

    print("Determine if a proteome is provided")
    for idx, row in df.iterrows():
        path_prot = os.path.join(dir_biocyc, row["biocyc"], "protseq.fasta")
        if not os.path.isfile(path_prot):
            path_prot = os.path.join(dir_biocyc, row["biocyc"], "protseq.fsa")
        df.at[idx, "has_proteome"] = os.path.isfile(path_prot)
    df = df[df["has_proteome"]]

    df.to_csv(file_taxonomy_csv, index=False)
    df = df[(df["domain"] == 2) | (df["kingdom"] == 4751)]

    print("Process classes.dat, compounds.dat, reactions.dat")
    def select_transport(row):
        if row["types"] and "Transport-Reactions" in row["types"]:
            return True
        if row["comment"] and "IUBMB" in row["comment"]:
            return True
        return False

    df_mols = pd.DataFrame()
    for _, row in tqdm(df.iterrows(), total=len(df)):
        biocyc = row["biocyc"]
        # Ontology
        ontologies = Ontology.from_file(
            path=os.path.join(dir_biocyc, biocyc, "classes.dat")
        )
        df_onto = pd.DataFrame([x.to_dict() for x in ontologies])
        # Filter transport
        df_onto_transport = df_onto[df_onto.apply(select_transport, axis=1)]
        transport_classes = {"Transport-Reactions"} | set(df_onto_transport["id"].values)
        # Metabolites
        metabolites = Metabolite.from_file(path=os.path.join(dir_biocyc, biocyc, "compounds.dat"))
        df_met = pd.DataFrame([x.to_dict() for x in metabolites])
        # Reactions
        reactions = Reaction.from_file(path=os.path.join(dir_biocyc, biocyc, "reactions.dat"))
        df_reac = pd.DataFrame([x.to_dict() for x in reactions])
        def to_filter(row):
            has_in = row["has_in"]
            has_out = row["has_out"]
            types = row["types"]
            by_type = bool(set(types) & transport_classes)
            by_compartment = has_in and has_out
            return (by_type or by_compartment)
        df_reac = df_reac[df_reac.apply(to_filter, axis=1)]
        molecules = set(df_reac["compounds_in_two_compartments"].explode()) | set(df_reac["compounds_only_one_side"].explode())
        
        df_mol = pd.DataFrame(molecules, columns=["molecule_id"])
        df_mol = df_mol.merge(df_met, left_on="molecule_id", right_on="id", how="left")
        
        df_mol.drop(columns="id", inplace=True)
        df_mol = df_mol[~df_mol["smiles"].isna()]
        df_mol["is_valid"] = df_mol["smiles"].apply(lambda x: True if Chem.MolFromSmiles(x) else False)
        df_mol.reset_index(inplace=True, drop=True)
        df_mol["biocyc"] = biocyc
        
        df_mols = pd.concat([df_mols, df_mol])
    
    df_mols = df_mols.merge(df, on="biocyc", how="left")
    df_mols.to_csv(file_compound_csv, index=False)
    print("End")