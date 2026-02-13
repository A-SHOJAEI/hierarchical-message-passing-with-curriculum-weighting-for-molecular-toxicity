"""Data preprocessing utilities for molecular graphs."""

import logging
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


# Atom feature dimensions
ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),
    'degree': [0, 1, 2, 3, 4, 5, 6],
    'formal_charge': [-3, -2, -1, 0, 1, 2, 3],
    'chiral_tag': [0, 1, 2, 3],
    'num_hs': [0, 1, 2, 3, 4],
    'hybridization': [0, 1, 2, 3, 4, 5],
}


def one_hot_encoding(value: int, choices: List[int]) -> List[int]:
    """Create one-hot encoding for a value.

    Args:
        value: Value to encode
        choices: List of possible values

    Returns:
        One-hot encoded list
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def get_atom_features(atom: Chem.Atom) -> np.ndarray:
    """Extract atom features for GNN.

    Args:
        atom: RDKit atom object

    Returns:
        Numpy array of atom features
    """
    features = []
    features += one_hot_encoding(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num'])
    features += one_hot_encoding(atom.GetTotalDegree(), ATOM_FEATURES['degree'])
    features += one_hot_encoding(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge'])
    features += one_hot_encoding(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag'])
    features += one_hot_encoding(atom.GetTotalNumHs(), ATOM_FEATURES['num_hs'])
    features += one_hot_encoding(int(atom.GetHybridization()), ATOM_FEATURES['hybridization'])
    features.append(atom.GetIsAromatic())
    features.append(atom.IsInRing())

    return np.array(features, dtype=np.float32)


def get_bond_features(bond: Chem.Bond) -> np.ndarray:
    """Extract bond features for GNN.

    Args:
        bond: RDKit bond object

    Returns:
        Numpy array of bond features
    """
    bond_type = bond.GetBondType()
    features = [
        bond_type == Chem.BondType.SINGLE,
        bond_type == Chem.BondType.DOUBLE,
        bond_type == Chem.BondType.TRIPLE,
        bond_type == Chem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
    ]

    return np.array(features, dtype=np.float32)


def mol_to_graph(smiles: str) -> Data:
    """Convert SMILES string to PyTorch Geometric graph.

    Args:
        smiles: SMILES representation of molecule

    Returns:
        PyTorch Geometric Data object

    Raises:
        ValueError: If SMILES cannot be parsed
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Add hydrogens for complete representation
    mol = Chem.AddHs(mol)

    # Extract atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
    x = torch.tensor(atom_features, dtype=torch.float)

    # Extract edges and edge features
    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices += [[i, j], [j, i]]

        bond_feat = get_bond_features(bond)
        edge_features += [bond_feat, bond_feat]

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def detect_functional_groups(mol: Chem.Mol, num_groups: int = 16) -> Tuple[List[List[int]], torch.Tensor]:
    """Detect functional groups in molecule using predefined SMARTS patterns.

    Args:
        mol: RDKit molecule object
        num_groups: Maximum number of functional groups to detect

    Returns:
        Tuple of (list of atom indices per group, group feature matrix)
    """
    # Common functional group SMARTS patterns
    functional_group_patterns = [
        ('hydroxyl', '[OH]'),
        ('carbonyl', '[CX3]=[OX1]'),
        ('carboxyl', '[CX3](=O)[OX1H0-,OX2H1]'),
        ('amine', '[NX3;H2,H1;!$(NC=O)]'),
        ('amide', '[NX3][CX3](=[OX1])'),
        ('ester', '[#6][CX3](=O)[OX2H0][#6]'),
        ('ether', '[OD2]([#6])[#6]'),
        ('aromatic', 'c'),
        ('alkene', '[CX3]=[CX3]'),
        ('alkyne', '[CX2]#C'),
        ('nitrile', '[NX1]#[CX2]'),
        ('nitro', '[$([NX3](=O)=O),$([NX3+](=O)[O-])]'),
        ('sulfoxide', '[$([SX3](=O)[#6]),$([SX3+]([O-])[#6])]'),
        ('halogen', '[F,Cl,Br,I]'),
        ('phosphate', '[$(P(=[OX1])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)])]'),
        ('thiol', '[SX2H]'),
    ]

    groups = []
    group_features = []

    for name, smarts in functional_group_patterns[:num_groups]:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue

        matches = mol.GetSubstructMatches(pattern)
        for match in matches:
            groups.append(list(match))
            # Simple one-hot encoding for group type
            feat = [0.0] * len(functional_group_patterns[:num_groups])
            feat[len(groups) - 1 % len(functional_group_patterns[:num_groups])] = 1.0
            group_features.append(feat)

    if not groups:
        # If no groups found, create a single group with all atoms
        groups = [list(range(mol.GetNumAtoms()))]
        group_features = [[1.0] + [0.0] * (len(functional_group_patterns[:num_groups]) - 1)]

    group_feature_tensor = torch.tensor(group_features, dtype=torch.float)

    return groups, group_feature_tensor


def compute_scaffold_complexity(smiles: str) -> float:
    """Compute molecular scaffold complexity score.

    Higher scores indicate more complex molecules with multiple rings,
    stereocenters, and diverse functional groups.

    Args:
        smiles: SMILES representation of molecule

    Returns:
        Complexity score (normalized to ~0-1 range)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0

    try:
        # Murcko scaffold
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)

        # Complexity factors
        num_rings = rdMolDescriptors.CalcNumRings(mol)
        num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        num_heteroatoms = rdMolDescriptors.CalcNumHeteroatoms(mol)
        num_stereocenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))

        # Bertz complexity
        bertz_ct = Descriptors.BertzCT(mol)

        # Combine factors (normalized heuristically)
        complexity = (
            0.3 * min(num_rings / 5.0, 1.0) +
            0.2 * min(num_aromatic_rings / 3.0, 1.0) +
            0.1 * min(num_rotatable_bonds / 10.0, 1.0) +
            0.1 * min(num_heteroatoms / 10.0, 1.0) +
            0.1 * min(num_stereocenters / 5.0, 1.0) +
            0.2 * min(bertz_ct / 1000.0, 1.0)
        )

        return float(complexity)

    except Exception as e:
        logger.warning(f"Error computing scaffold complexity for {smiles}: {e}")
        return 0.0


def scaffold_split(smiles_list: List[str],
                   labels: np.ndarray,
                   train_ratio: float = 0.8,
                   val_ratio: float = 0.1) -> Tuple[List[int], List[int], List[int]]:
    """Split dataset by molecular scaffolds.

    Args:
        smiles_list: List of SMILES strings
        labels: Array of labels
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    scaffolds = {}
    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            scaffold = "invalid"
        else:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)

        if scaffold not in scaffolds:
            scaffolds[scaffold] = []
        scaffolds[scaffold].append(idx)

    # Sort scaffolds by size
    scaffold_sets = list(scaffolds.values())
    scaffold_sets.sort(key=lambda x: len(x), reverse=True)

    # Split
    n = len(smiles_list)
    train_cutoff = int(train_ratio * n)
    val_cutoff = int((train_ratio + val_ratio) * n)

    train_idx, val_idx, test_idx = [], [], []

    for scaffold_set in scaffold_sets:
        if len(train_idx) + len(scaffold_set) <= train_cutoff:
            train_idx.extend(scaffold_set)
        elif len(train_idx) + len(val_idx) + len(scaffold_set) <= val_cutoff:
            val_idx.extend(scaffold_set)
        else:
            test_idx.extend(scaffold_set)

    return train_idx, val_idx, test_idx
