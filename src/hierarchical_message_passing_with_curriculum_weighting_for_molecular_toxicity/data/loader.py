"""Data loading utilities for molecular toxicity datasets."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data

# Try to import deepchem, but make it optional
try:
    import deepchem as dc
    DEEPCHEM_AVAILABLE = True
except ImportError:
    DEEPCHEM_AVAILABLE = False
    dc = None

from hierarchical_message_passing_with_curriculum_weighting_for_molecular_toxicity.data.preprocessing import (
    mol_to_graph,
    detect_functional_groups,
    compute_scaffold_complexity,
    scaffold_split,
)

logger = logging.getLogger(__name__)


class MoleculeDataset(Dataset):
    """PyTorch dataset for molecular graphs with toxicity labels.

    Args:
        smiles_list: List of SMILES strings
        labels: Array of toxicity labels
        complexities: Array of scaffold complexity scores
        num_groups: Number of functional groups to detect
        cache_dir: Directory to cache processed graphs
    """

    def __init__(
        self,
        smiles_list: List[str],
        labels: np.ndarray,
        complexities: Optional[np.ndarray] = None,
        num_groups: int = 16,
        cache_dir: Optional[str] = None,
    ):
        self.smiles_list = smiles_list
        self.labels = torch.tensor(labels, dtype=torch.float)
        self.num_groups = num_groups
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Compute complexities if not provided
        if complexities is None:
            logger.info("Computing scaffold complexities...")
            self.complexities = torch.tensor(
                [compute_scaffold_complexity(s) for s in smiles_list],
                dtype=torch.float
            )
        else:
            self.complexities = torch.tensor(complexities, dtype=torch.float)

        # Cache for processed graphs
        self._cache = {}

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a molecule graph with its label and complexity.

        Args:
            idx: Index of the molecule

        Returns:
            Dictionary containing graph data, label, and complexity
        """
        # Check cache
        if idx in self._cache:
            return self._cache[idx]

        smiles = self.smiles_list[idx]
        label = self.labels[idx]
        complexity = self.complexities[idx]

        try:
            # Convert to graph
            graph = mol_to_graph(smiles)

            # Detect functional groups
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol = Chem.AddHs(mol)
                groups, group_features = detect_functional_groups(mol, self.num_groups)
            else:
                groups = [[]]
                group_features = torch.zeros(1, self.num_groups)

            # Create data object
            data = {
                'x': graph.x,
                'edge_index': graph.edge_index,
                'edge_attr': graph.edge_attr,
                'y': label,
                'complexity': complexity,
                'groups': groups,
                'group_features': group_features,
                'smiles': smiles,
            }

            # Cache the processed data
            self._cache[idx] = data

            return data

        except Exception as e:
            logger.warning(f"Error processing molecule {idx} ({smiles}): {e}")
            # Return a dummy data object
            return {
                'x': torch.zeros(1, 155),
                'edge_index': torch.zeros(2, 0, dtype=torch.long),
                'edge_attr': torch.zeros(0, 6),
                'y': label,
                'complexity': complexity,
                'groups': [[]],
                'group_features': torch.zeros(1, self.num_groups),
                'smiles': smiles,
            }


def collate_fn(batch: List[Dict]) -> Dict[str, any]:
    """Custom collate function for batching molecular graphs.

    Args:
        batch: List of data dictionaries from MoleculeDataset

    Returns:
        Batched dictionary
    """
    batched_data = {
        'x': [],
        'edge_index': [],
        'edge_attr': [],
        'y': [],
        'complexity': [],
        'groups': [],
        'group_features': [],
        'batch': [],
        'smiles': [],
    }

    node_offset = 0
    for i, data in enumerate(batch):
        batched_data['x'].append(data['x'])
        batched_data['edge_attr'].append(data['edge_attr'])

        # Offset edge indices for batching
        edge_index = data['edge_index'] + node_offset
        batched_data['edge_index'].append(edge_index)

        batched_data['y'].append(data['y'])
        batched_data['complexity'].append(data['complexity'])

        # Store groups with offset
        offset_groups = [[idx + node_offset for idx in group] for group in data['groups']]
        batched_data['groups'].append(offset_groups)
        batched_data['group_features'].append(data['group_features'])

        # Batch assignment
        batch_idx = torch.full((data['x'].size(0),), i, dtype=torch.long)
        batched_data['batch'].append(batch_idx)

        batched_data['smiles'].append(data['smiles'])

        node_offset += data['x'].size(0)

    # Concatenate tensors
    batched_data['x'] = torch.cat(batched_data['x'], dim=0)
    batched_data['edge_index'] = torch.cat(batched_data['edge_index'], dim=1)
    batched_data['edge_attr'] = torch.cat(batched_data['edge_attr'], dim=0)
    batched_data['y'] = torch.stack(batched_data['y'])
    batched_data['complexity'] = torch.stack(batched_data['complexity'])
    batched_data['batch'] = torch.cat(batched_data['batch'])

    return batched_data


def load_tox21_dataset() -> Tuple[List[str], np.ndarray]:
    """Load Tox21 dataset from DeepChem.

    Returns:
        Tuple of (smiles_list, labels)
    """
    logger.info("Loading Tox21 dataset...")

    if not DEEPCHEM_AVAILABLE:
        logger.warning("DeepChem not available. Generating synthetic data for demonstration...")
        return generate_synthetic_data(num_samples=1000)

    try:
        # Load using DeepChem
        tasks, datasets, transformers = dc.molnet.load_tox21(featurizer='Raw')
        train_dataset, valid_dataset, test_dataset = datasets

        # Combine all splits
        all_smiles = []
        all_labels = []

        for dataset in [train_dataset, valid_dataset, test_dataset]:
            smiles = dataset.ids
            # Take the first task for binary classification
            labels = dataset.y[:, 0]

            # Filter out molecules with missing labels
            valid_mask = ~np.isnan(labels)
            all_smiles.extend(smiles[valid_mask])
            all_labels.append(labels[valid_mask])

        all_labels = np.concatenate(all_labels)

        logger.info(f"Loaded {len(all_smiles)} molecules from Tox21 dataset")
        return all_smiles, all_labels

    except Exception as e:
        logger.error(f"Error loading Tox21 dataset: {e}")
        logger.warning("Generating synthetic data for demonstration...")
        return generate_synthetic_data(num_samples=1000)


def generate_synthetic_data(num_samples: int = 1000) -> Tuple[List[str], np.ndarray]:
    """Generate synthetic molecular data for testing.

    Args:
        num_samples: Number of samples to generate

    Returns:
        Tuple of (smiles_list, labels)
    """
    # Simple SMILES for testing
    base_smiles = [
        'CCO',  # ethanol
        'CC(=O)O',  # acetic acid
        'c1ccccc1',  # benzene
        'CCN(CC)CC',  # triethylamine
        'CC(C)O',  # isopropanol
        'C1CCCCC1',  # cyclohexane
        'c1ccc(O)cc1',  # phenol
        'CC(=O)C',  # acetone
        'CCOC(=O)C',  # ethyl acetate
        'c1ccc(N)cc1',  # aniline
    ]

    smiles_list = []
    labels = []

    for i in range(num_samples):
        smiles = base_smiles[i % len(base_smiles)]
        smiles_list.append(smiles)

        # Generate label based on complexity
        complexity = compute_scaffold_complexity(smiles)
        label = 1 if complexity > 0.3 or np.random.rand() > 0.6 else 0
        labels.append(label)

    return smiles_list, np.array(labels)


def get_data_loaders(
    config: Dict,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders.

    Args:
        config: Configuration dictionary
        num_workers: Number of workers for data loading

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_config = config.get('data', {})
    dataset_name = data_config.get('dataset', 'tox21')
    split_type = data_config.get('split_type', 'scaffold')
    batch_size = config.get('training', {}).get('batch_size', 32)
    num_groups = config.get('model', {}).get('num_groups', 16)

    # Load dataset
    if dataset_name == 'tox21':
        smiles_list, labels = load_tox21_dataset()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Split dataset
    if split_type == 'scaffold':
        train_ratio = data_config.get('train_ratio', 0.8)
        val_ratio = data_config.get('val_ratio', 0.1)
        train_idx, val_idx, test_idx = scaffold_split(
            smiles_list, labels, train_ratio, val_ratio
        )
    else:
        # Random split
        n = len(smiles_list)
        indices = np.random.permutation(n)
        train_size = int(0.8 * n)
        val_size = int(0.1 * n)
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]

    logger.info(f"Split sizes - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Create datasets
    train_smiles = [smiles_list[i] for i in train_idx]
    train_labels = labels[train_idx]
    train_dataset = MoleculeDataset(train_smiles, train_labels, num_groups=num_groups)

    val_smiles = [smiles_list[i] for i in val_idx]
    val_labels = labels[val_idx]
    val_dataset = MoleculeDataset(val_smiles, val_labels, num_groups=num_groups)

    test_smiles = [smiles_list[i] for i in test_idx]
    test_labels = labels[test_idx]
    test_dataset = MoleculeDataset(test_smiles, test_labels, num_groups=num_groups)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
