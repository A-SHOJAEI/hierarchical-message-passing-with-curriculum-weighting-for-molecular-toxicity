"""Tests for training loop."""

import pytest
import torch
from torch.utils.data import DataLoader

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hierarchical_message_passing_with_curriculum_weighting_for_molecular_toxicity.models.model import (
    HierarchicalMolecularGNN,
)
from hierarchical_message_passing_with_curriculum_weighting_for_molecular_toxicity.training.trainer import (
    Trainer,
)
from hierarchical_message_passing_with_curriculum_weighting_for_molecular_toxicity.data.loader import (
    MoleculeDataset,
    collate_fn,
)


class TestTrainer:
    """Test trainer."""

    def test_trainer_creation(self, sample_config, device):
        """Test trainer creation."""
        model = HierarchicalMolecularGNN(sample_config)
        trainer = Trainer(model, sample_config, device)

        assert trainer is not None
        assert trainer.optimizer is not None
        assert trainer.criterion is not None

    def test_train_epoch(self, sample_config, sample_smiles, sample_labels, device):
        """Test single training epoch."""
        # Create small dataset
        dataset = MoleculeDataset(sample_smiles, sample_labels, num_groups=8)
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        # Create model and trainer
        model = HierarchicalMolecularGNN(sample_config)
        trainer = Trainer(model, sample_config, device)

        # Train one epoch
        loss, metrics = trainer.train_epoch(loader, epoch=1)

        assert loss > 0
        assert 'roc_auc' in metrics
        assert 'accuracy' in metrics

    def test_validate(self, sample_config, sample_smiles, sample_labels, device):
        """Test validation."""
        # Create small dataset
        dataset = MoleculeDataset(sample_smiles, sample_labels, num_groups=8)
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        # Create model and trainer
        model = HierarchicalMolecularGNN(sample_config)
        trainer = Trainer(model, sample_config, device)

        # Validate
        loss, metrics = trainer.validate(loader)

        assert loss > 0
        assert 'roc_auc' in metrics
        assert 'accuracy' in metrics

    def test_checkpoint_save_load(self, sample_config, device, tmp_path):
        """Test checkpoint saving and loading."""
        model = HierarchicalMolecularGNN(sample_config)
        trainer = Trainer(model, sample_config, device)

        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        trainer.save_checkpoint(checkpoint_path, epoch=1, val_loss=0.5, val_auc=0.8)

        assert checkpoint_path.exists()

        # Load checkpoint
        trainer.load_checkpoint(checkpoint_path)

        # Verify loaded
        checkpoint = torch.load(checkpoint_path)
        assert checkpoint['epoch'] == 1
        assert checkpoint['val_loss'] == 0.5
        assert checkpoint['val_auc'] == 0.8
