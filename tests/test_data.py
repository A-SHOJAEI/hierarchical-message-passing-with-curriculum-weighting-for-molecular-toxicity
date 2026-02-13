"""Tests for data loading and preprocessing."""

import pytest
import numpy as np
import torch
from rdkit import Chem

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hierarchical_message_passing_with_curriculum_weighting_for_molecular_toxicity.data.preprocessing import (
    mol_to_graph,
    detect_functional_groups,
    compute_scaffold_complexity,
    scaffold_split,
)
from hierarchical_message_passing_with_curriculum_weighting_for_molecular_toxicity.data.loader import (
    MoleculeDataset,
)


class TestPreprocessing:
    """Test preprocessing functions."""

    def test_mol_to_graph(self, sample_smiles):
        """Test molecular graph conversion."""
        smiles = sample_smiles[0]
        graph = mol_to_graph(smiles)

        assert isinstance(graph.x, torch.Tensor)
        assert isinstance(graph.edge_index, torch.Tensor)
        assert isinstance(graph.edge_attr, torch.Tensor)
        assert graph.x.size(1) == 155  # atom feature dimension
        assert graph.edge_attr.size(1) == 6  # bond feature dimension

    def test_mol_to_graph_invalid(self):
        """Test invalid SMILES."""
        with pytest.raises(ValueError):
            mol_to_graph("INVALID_SMILES")

    def test_detect_functional_groups(self, sample_smiles):
        """Test functional group detection."""
        smiles = sample_smiles[0]
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)

        groups, group_features = detect_functional_groups(mol, num_groups=8)

        assert isinstance(groups, list)
        assert isinstance(group_features, torch.Tensor)
        assert len(groups) > 0
        assert group_features.size(0) == len(groups)

    def test_compute_scaffold_complexity(self, sample_smiles):
        """Test scaffold complexity computation."""
        for smiles in sample_smiles:
            complexity = compute_scaffold_complexity(smiles)
            assert 0.0 <= complexity <= 1.0

    def test_scaffold_split(self, sample_smiles, sample_labels):
        """Test scaffold splitting."""
        train_idx, val_idx, test_idx = scaffold_split(
            sample_smiles,
            sample_labels,
            train_ratio=0.6,
            val_ratio=0.2,
        )

        # Check no overlap
        assert len(set(train_idx) & set(val_idx)) == 0
        assert len(set(train_idx) & set(test_idx)) == 0
        assert len(set(val_idx) & set(test_idx)) == 0

        # Check all samples covered
        assert len(train_idx) + len(val_idx) + len(test_idx) == len(sample_smiles)


class TestMoleculeDataset:
    """Test MoleculeDataset."""

    def test_dataset_creation(self, sample_smiles, sample_labels):
        """Test dataset creation."""
        dataset = MoleculeDataset(sample_smiles, sample_labels, num_groups=8)

        assert len(dataset) == len(sample_smiles)

    def test_dataset_getitem(self, sample_smiles, sample_labels):
        """Test dataset indexing."""
        dataset = MoleculeDataset(sample_smiles, sample_labels, num_groups=8)

        data = dataset[0]

        assert 'x' in data
        assert 'edge_index' in data
        assert 'y' in data
        assert 'complexity' in data
        assert 'groups' in data
        assert 'smiles' in data

        assert isinstance(data['x'], torch.Tensor)
        assert isinstance(data['y'], torch.Tensor)
        assert isinstance(data['complexity'], torch.Tensor)
