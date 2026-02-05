from typing import List

from chemproflow.utils.molecule import fmt_smiles
from rdkit import Chem, RDLogger
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

RDLogger.DisableLog("rdApp.*")


# Helper functions for one-hot encoding
def one_hot_encoding(x, allowable_set):
    return [int(x == s) for s in allowable_set]

# Atom feature vector
def get_atom_features(atom):
    chiral_types = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ]

    hybridization_types = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ]

    return torch.tensor(
        one_hot_encoding(atom.GetAtomicNum(), list(range(1, 119))) +     # atomic number
        one_hot_encoding(atom.GetDegree(), list(range(6))) +             # number of bonds
        one_hot_encoding(atom.GetFormalCharge(), [-1, 0, 1]) +           # formal charge
        one_hot_encoding(atom.GetChiralTag(), chiral_types) +            # chirality
        one_hot_encoding(atom.GetTotalNumHs(), list(range(5))) +         # num hydrogens
        one_hot_encoding(atom.GetHybridization(), hybridization_types) + # hybridization
        [atom.GetIsAromatic()] +                                         # aromaticity
        [atom.GetMass() / 100],                                          # scaled mass
        dtype=torch.float
    )

# Bond feature vector
def get_bond_features(bond):
    if bond is None:
        return torch.zeros(10, dtype=torch.float)

    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ]

    stereo_types = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE
    ]

    return torch.tensor(
        one_hot_encoding(bond.GetBondType(), bond_types) +
        [bond.GetIsConjugated()] +
        [bond.IsInRing()] +
        one_hot_encoding(bond.GetStereo(), stereo_types),
        dtype=torch.float
    )

def mol_to_graph(smiles):
    smiles = fmt_smiles(smiles)
    if len(smiles) < 1:
        raise ValueError("Invalid SMILES:", smiles)
    mol = Chem.MolFromSmiles(smiles)

    Chem.Kekulize(mol, clearAromaticFlags=True)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()

    # Node features
    x = torch.stack([get_atom_features(atom) for atom in atoms])

    # Edge features
    edge_index = []
    edge_attr = []

    for bond in bonds:
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_feat = get_bond_features(bond)

        # Add both directions for directed graph
        edge_index += [[i, j], [j, i]]
        edge_attr += [bond_feat, bond_feat]

    if edge_index:  # Safe if non-empty
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attr)
    else:  # No bonds: return empty edge_index and edge_attr
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, get_bond_features(None).numel()), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)

def build_loader(smiles: List[str], batch_size: int, shuffle: bool = False) -> DataLoader:
    graphs = []
    for s in smiles:
        g = mol_to_graph(smiles=s)
        graphs.append(g)
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=shuffle)
    return loader