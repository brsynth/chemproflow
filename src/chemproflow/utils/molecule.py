from typing import List, Optional

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

RDLogger.DisableLog("rdApp.*")

StereoOptions = StereoEnumerationOptions(onlyUnassigned=True, unique=True)


def fmt_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    # Generate stereoisomers (as a generator)
    stereoisomers = EnumerateStereoisomers(mol, options=StereoOptions)
    # Get only the first stereoisomer
    first_isomer = next(stereoisomers, None)  # None is a default if the generator is empty
    return Chem.MolToSmiles(first_isomer)


def to_fp(smiles: str, radius: int = 2, fp_size: int = 2048) -> Optional[List[int]]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return generator.GetFingerprint(mol)