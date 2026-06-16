import os
from typing import Any, Dict, List, Tuple

import jinja2
import pandas as pd
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
            smiles_canonical: str,
            is_transported: bool,
            #dataset_transport: bool,
            pred_transport: bool,
            pred_tcids: List,
            prot_tcids: List[Tuple],
            similar_substrates: List,
            name_micro_organisms: List[List],
        ):
        # Identity card
        self.name = Sample._as_text(name)
        self.smiles = Sample._as_text(smiles)
        self.smiles_canonical = Sample._as_text(smiles_canonical)
        self.svg = Sample.draw_mol(smiles=self.smiles_canonical)
        self.weight = Sample.compute_weight(smiles=self.smiles_canonical)
        self.is_transported = None if Sample._is_missing(is_transported) else is_transported
        # Is involved in a cellular transport
        #self.dataset_transport = bool(dataset_transport)
        self.pred_transport = Sample._as_bool(pred_transport)
        self.pred_tcids = Sample._as_list(pred_tcids)
        self.prot_tcids = Sample._as_list(prot_tcids)
        self.similar_substrates = Sample._as_list(similar_substrates)
        self.similar_substrates_svg = [Sample.draw_mol(smiles) for smiles in self.similar_substrates]
        self.name_micro_organisms = Sample._as_list(name_micro_organisms)

    @classmethod
    def _is_missing(cls, value: Any) -> bool:
        if value is None:
            return True
        try:
            return bool(pd.isna(value))
        except (TypeError, ValueError):
            return False

    @classmethod
    def _as_list(cls, value: Any) -> List:
        if cls._is_missing(value):
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, set):
            return list(value)
        return [value]

    @classmethod
    def _as_text(cls, value: Any) -> str:
        if cls._is_missing(value):
            return ""
        return str(value)

    @classmethod
    def _as_bool(cls, value: Any) -> bool | None:
        if cls._is_missing(value):
            return None
        if isinstance(value, bool):
            return value
        value_str = str(value).strip().lower()
        if value_str in {"positive", "true", "1", "yes", "y"}:
            return True
        if value_str in {"negative", "false", "0", "no", "n"}:
            return False
        return None

    @classmethod
    def draw_mol(cls, smiles: str):
        smiles = cls._as_text(smiles)
        if not smiles:
            return ""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
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
        smiles = cls._as_text(smiles)
        if not smiles:
            return ""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        return "{:.2f}".format(Descriptors.ExactMolWt(mol))
    
    def to_dict(self) -> Dict:
        return dict(
            name=self.name,
            smiles=self.smiles,
            smiles_canonical=self.smiles_canonical,
            weight=self.weight,
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
        smiles_canonical = data.get("smiles_canonical") or data.get("smiles") or ""
        micro_organisms = data.get("name_micro_organisms")
        if cls._is_missing(micro_organisms):
            micro_organisms = data.get("accession_micro_organisms")
        return Sample(
            name=data.get("name", ""),
            smiles=data.get("smiles", ""),
            smiles_canonical=smiles_canonical,
            is_transported=data.get("is_transported"),
            #dataset_transport=data.get("dataset_transport"),
            pred_transport=data.get("pred_transport"),
            pred_tcids=cls._as_list(data.get("pred_tcid")),
            prot_tcids=cls._as_list(data.get("prot_tcids")),
            similar_substrates=cls._as_list(data.get("similar_substrates")),
            name_micro_organisms=cls._as_list(micro_organisms),
        )
    

class Report:
    """Utility class to render a Jinja2 HTML report."""

    def __init__(self, template_path: str) -> None:
        if not os.path.isfile(template_path):
            raise ValueError(f"Template file does not exist: {template_path}")

        # Create a Jinja2 environment
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(os.path.dirname(template_path) or "."),
            autoescape=jinja2.select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Add global functions if needed
        self.env.globals.update(zip=zip, len=len, str=str)

        # Load the template by filename
        self.template = self.env.get_template(os.path.basename(template_path))

    def to_html(self, output_path: str, context: Dict[Any, Any]) -> None:
        """Render the template with context and write the output HTML file.

        Parameters
        ----------
        output_path : str
            Path of the output HTML file.
        context : dict
            Data to inject into the template.
        """
        rendered = self.template.render(context=context)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fout:
            fout.write(rendered)
