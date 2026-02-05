import os
from typing import Any, Dict, List, Tuple

from rdkit import Chem
from rdkit.Chem import Descriptors, Draw


class Sample:
    ATOM_COLORS = {
        0: (102/255.0, 102/255.0, 102/255.0), # dummy
        7: (0.33, 0.6, 0.78), # azote
        8: (0.8, 0.38, 0.33), # oxygen
        15: (0.92, 0.6, 0.3), # phosphore
    }
    
    def __init__(
            self,
            name: str,
            smiles: str,
            is_transported: bool,
            #dataset_transport: bool,
            pred_transport: bool,
            pred_tcids: List,
            prot_tcids: List[Tuple],
            similar_substrates: List,
            name_micro_organisms: List[List],
        ):
        # Identity card
        self.name = name
        self.smiles = smiles
        self.svg = Sample.draw_mol(smiles=self.smiles)
        self.weight = Sample.compute_weight(smiles=self.smiles)
        self.is_transported = is_transported
        # Is involved in a cellular transport
        #self.dataset_transport = bool(dataset_transport)
        self.pred_transport = False
        self.pred_tcids = pred_tcids
        if pred_transport == "positive":
            self.pred_transport = True
        self.prot_tcids = prot_tcids
        self.similar_substrates = similar_substrates
        self.similar_substrates_svg = [Sample.draw_mol(smiles) for smiles in self.similar_substrates]
        self.name_micro_organisms = name_micro_organisms

    @classmethod
    def draw_mol(cls, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        # Draw
        d2d = Draw.MolDraw2DSVG(1000, 1000)    
        opts = d2d.drawOptions()
        opts.updateAtomPalette(Sample.ATOM_COLORS)
        #opts.bondLineWidth = 5
        d2d.DrawMolecule(mol)
        d2d.FinishDrawing()
        return d2d.GetDrawingText()
    
    @classmethod
    def compute_weight(cls, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        weight = -1.
        if mol:
            weight = Descriptors.ExactMolWt(mol)
        return "{:.2f}".format(weight)
    
    def to_dict(self) -> Dict:
        return dict(
            name=self.name,
            smiles=self.smiles,
            weigth=self.weight,
            is_transported=self.is_transported,
            #dataset_transport=self.dataset_transport,
            pred_transport=self.pred_transport,
            pred_tcids=self.pred_tcids,
            prot_tcids=self.prot_tcids,
            similar_substrates=self.similar_substrates,
            name_micro_organisms=self.name_micro_organisms,
        )

    @classmethod
    def from_dict(cls, data: Dict) -> "Sample":
        return Sample(
            name=data.get("name", ""),
            smiles=data.get("smiles", ""),
            is_transported=data.get("is_transported"),
            #dataset_transport=data.get("dataset_transport"),
            pred_transport=data.get("pred_transport"),
            pred_tcids=data.get("pred_tcid"),
            prot_tcids=data.get("prot_tcids"),
            similar_substrates=data.get("similar_substrates"),
            name_micro_organisms=data.get("name_micro_organisms"),
        )