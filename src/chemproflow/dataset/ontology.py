import argparse
import os
import time
from collections import defaultdict
from functools import cache
from typing import Dict, List, Optional, Tuple

from owlready2 import get_ontology
from rdflib import URIRef
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


class Ontology:

    URL_CHEBI_PREFIX = "http://purl.obolibrary.org/obo/"

    def __init__(self, path: str):
        path = os.path.abspath(path)
        self.onto = get_ontology(f"file://{path}").load()
        graph = self.onto.world.as_rdflib_graph()
        self.graph = [(subj, pred, obj) for subj, pred, obj in graph]
        self.graph_chebi = [
            (subj, pred, obj)
            for subj, pred, obj in self.graph
            if "CHEBI" in subj.split("/")[-1] or "CHEBI" in obj.split("/")[-1]
        ]
        self.graph_subclassof = [
            (subj, pred, obj)
            for subj, pred, obj in self.graph_chebi
            if "#subClassOf" in pred.split("/")[-1]
        ]

        # Build
        self.map_chebi_smiles = defaultdict(set)
        self.map_chebi_rels = defaultdict(list)
        for subj, pred, obj in self.graph_chebi:
            if (
                "CHEBI" in Ontology.get_entity_label(uri=subj)
                and Ontology.get_entity_label(uri=pred) == "smiles"
                and Ontology.get_entity_label(uri=obj).count("*") < 1
            ):
                chebi_id = Ontology.get_entity_label(uri=subj).replace("_", ":")
                smiles = Ontology.get_entity_label(uri=obj)
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    smiles = Chem.MolToSmiles(mol)
                    self.map_chebi_smiles[chebi_id].add(smiles)
            if "CHEBI" in Ontology.get_entity_label(uri=subj):
                chebi_id = Ontology.get_entity_label(uri=subj).replace("_", ":")
                self.map_chebi_rels[chebi_id].append((subj, pred, obj))
            if "CHEBI" in Ontology.get_entity_label(uri=obj):
                chebi_id = Ontology.get_entity_label(uri=obj).replace("_", ":")
                self.map_chebi_rels[chebi_id].append((subj, pred, obj))

    @classmethod
    def build_chebi_uri(cls, chebi_id: str):
        chebi_iri = Ontology.URL_CHEBI_PREFIX + chebi_id.replace(":", "_")
        return URIRef(chebi_iri)

    @classmethod
    def get_entity_label(cls, uri):
        if isinstance(uri, URIRef):
            return uri.split("/")[-1]
        return str(uri)

    @cache
    def get_smiles(self, chebi_id: str) -> Optional[str]:
        target_class_uri = Ontology.build_chebi_uri(chebi_id=chebi_id)
        rels = self.get_rels(target_class_uri=target_class_uri, income=False)
        for rel in rels:
            if (
                Ontology.get_entity_label(uri=rel[1]) == "smiles"
                and Ontology.get_entity_label(uri=rel[0]).count("*") < 1
            ):
                mol = Chem.MolFromSmiles(Ontology.get_entity_label(uri=rel[0]))
                if mol:
                    return Chem.MolToSmiles(mol)
        return None

    def get_synonyms(self, chebi_id: str):
        target_class_uri = Ontology.build_chebi_uri(chebi_id=chebi_id)
        rels = self.get_rels(target_class_uri=target_class_uri, income=False)
        chebi_ids = []
        for rel in rels:
            if rel and "#hasAlternativeId" in Ontology.get_entity_label(uri=rel[1]):
                chebi_ids.append(Ontology.get_entity_label(uri=rel[0]))
        return chebi_ids

    def get_rels(self, target_class_uri, income: bool = True) -> List:
        rels = []
        for s, p, o in self.graph_chebi:
            if income and o == target_class_uri:
                rels.append((s, p))
            elif income is False and s == target_class_uri:
                rels.append((o, p))
        return rels

    def get_childs(self, chebi_id: str) -> List:
        chebi_ids = []
        # Check deprecated
        is_deprecated = self.check_deprecated(chebi_id=chebi_id)
        if is_deprecated[0] and is_deprecated[1]:
            return self.get_childs(chebi_id=is_deprecated[1])
        # Check ontology has smiles
        target_class_uri = Ontology.build_chebi_uri(chebi_id=chebi_id)
        if self.map_chebi_smiles[chebi_id]:
            chebi_ids.append(chebi_id)
            return chebi_ids
        for subj, pred, obj in self.graph_subclassof:
            if obj == target_class_uri:
                chebi_ids.extend(
                    self.get_childs(
                        chebi_id=Ontology.get_entity_label(uri=subj).replace("_", ":")
                    )
                )
        return chebi_ids

    @cache
    def check_deprecated(self, chebi_id: str) -> Tuple[bool, str]:
        target_class_uri = Ontology.build_chebi_uri(chebi_id=chebi_id)
        rels = self.map_chebi_rels[chebi_id]
        rels = [rel for rel in rels if rel[0] == target_class_uri]
        res = (False, None)
        for rel in rels:
            if "#deprecated" in Ontology.get_entity_label(uri=rel[1]):
                res = (True, None)
                break
        if res[0]:
            for rel in rels:
                if "IAO_0100001" in Ontology.get_entity_label(uri=rel[1]):
                    res = (
                        True,
                        Ontology.get_entity_label(uri=rel[2]).replace("_", ":"),
                    )
                    break
        return res

    def has_incoming(self, chebi_id: str) -> bool:
        target_class_uri = Ontology.build_chebi_uri(chebi_id=chebi_id)
        rels = [(subj, pred, obj) for subj, pred, obj in self.graph_subclassof if obj == target_class_uri]
        return len(rels) > 0

    def expand_all_smiles(self, chebi_id: str, data_rgroup: Optional[Dict] = None) -> Tuple[List[str], List[str]]:
        # df_rgroup: key(chebi), value(set(smiles))
        if data_rgroup is None:
            data_rgroup = {}
        chebi_smiles = set()
        chebi_rgroup = set()
        for chebi_child in [chebi_id] + self.get_childs(chebi_id=chebi_id):
            if chebi_child in data_rgroup and not self.has_incoming(chebi_id=chebi_child):
                chebi_smiles.update(data_rgroup[chebi_child])
                chebi_rgroup.add(chebi_child)
                continue
            chebi_smiles.update(self.map_chebi_smiles.get(chebi_child, set()))
        return list(chebi_smiles), list(chebi_rgroup)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", default=os.environ.get("CHEMICALDIR"), help="Data dir"
    )
    args = parser.parse_args()

    ## Init
    file_input_owl = os.path.join(args.data_dir, "chebi", "chebi.owl")

    ## Process
    print("Read ontology")
    ontology = Ontology(path=file_input_owl)

    ## Synonyms
    print("Read synonyms")
    synonyms = ontology.get_synonyms(chebi_id="CHEBI:18303")
    assert synonyms == ["CHEBI:14801", "CHEBI:26041", "CHEBI:8137"]
    print("Synonyms:", synonyms)

    ## Smiles
    childs = ontology.get_childs(chebi_id="CHEBI:90041")
    print("Smiles:", childs)
    assert childs == ["CHEBI:90041"]

    childs = ontology.get_childs(chebi_id="CHEBI:194106")
    print("Smiles:", childs)
    assert childs == ["CHEBI:194098"]

    start = time.time()
    childs = ontology.get_childs(chebi_id="CHEBI:183098")
    print("CHEBI:183098", "Smiles:", childs)
    smiles = []
    for child in childs:
        smiles.append(ontology.get_smiles(child))
    print("Smiles:", smiles)
    smiles_expanded = ontology.expand_all_smiles(chebi_id="CHEBI:183098")
    print("Smiles expanded:", smiles_expanded)
    assert sorted(smiles) == sorted(ontology.expand_all_smiles(chebi_id="CHEBI:183098"))

    print("Compute time:", round(time.time() - start, 2))
    # TODO: CHEBI:60311

    ## Check incoming
    print("CHEBI:183098")
    res = ontology.has_incoming(chebi_id="CHEBI:183098")
    assert res == True
    print("CHEBI:194098")
    res = ontology.has_incoming(chebi_id="CHEBI:194098")
    assert res is False

    print("end")
