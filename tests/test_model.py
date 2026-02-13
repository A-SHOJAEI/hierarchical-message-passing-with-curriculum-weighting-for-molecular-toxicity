"""Tests for model architecture."""

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hierarchical_message_passing_with_curriculum_weighting_for_molecular_toxicity.models.model import (
    HierarchicalMolecularGNN,
    AtomLevelGNN,
    GroupLevelGNN,
)
from hierarchical_message_passing_with_curriculum_weighting_for_molecular_toxicity.models.components import (
    CurriculumWeightScheduler,
    FocalLoss,
    AttentionPooling,
    GroupMessagePassing,
    MultiScaleFusion,
)


class TestComponents:
    """Test custom components."""

    def test_curriculum_weight_scheduler(self):
        """Test curriculum weight scheduler."""
        scheduler = CurriculumWeightScheduler(
            start_epoch=5,
            warmup_epochs=3,
            max_weight=2.0,
            min_weight=0.5,
        )

        # Before start epoch
        scheduler.step(3)
        complexities = torch.tensor([0.2, 0.5, 0.8])
        weights = scheduler.compute_weights(complexities)
        assert torch.allclose(weights, torch.ones_like(complexities))

        # After start epoch
        scheduler.step(10)
        weights = scheduler.compute_weights(complexities)
        assert weights.min() >= 0.5
        assert weights.max() <= 2.0

    def test_focal_loss(self):
        """Test focal loss."""
        criterion = FocalLoss(alpha=0.25, gamma=2.0)

        inputs = torch.randn(4)
        targets = torch.tensor([0.0, 1.0, 0.0, 1.0])

        loss = criterion(inputs, targets)
        assert loss.item() > 0

    def test_attention_pooling(self):
        """Test attention pooling."""
        pooling = AttentionPooling(hidden_dim=64, num_heads=4)

        x = torch.randn(10, 64)  # 10 nodes
        batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])  # 3 graphs
        batch_size = 3

        output = pooling(x, batch, batch_size)

        assert output.size() == (3, 64)

    def test_group_message_passing(self):
        """Test group message passing."""
        gmp = GroupMessagePassing(group_dim=32, hidden_dim=64)

        group_features = torch.randn(4, 32)
        adjacency = torch.eye(4)

        output = gmp(group_features, adjacency)

        assert output.size() == (4, 32)

    def test_multi_scale_fusion(self):
        """Test multi-scale fusion."""
        fusion = MultiScaleFusion(atom_dim=64, group_dim=32, output_dim=64)

        atom_features = torch.randn(2, 64)
        group_features = torch.randn(2, 32)

        output = fusion(atom_features, group_features)

        assert output.size() == (2, 64)


class TestAtomLevelGNN:
    """Test atom-level GNN."""

    def test_forward(self):
        """Test forward pass."""
        gnn = AtomLevelGNN(
            input_dim=155,
            hidden_dim=64,
            num_layers=2,
            dropout=0.2,
            use_edge_features=True,
            edge_dim=6,
        )

        x = torch.randn(10, 155)
        edge_index = torch.randint(0, 10, (2, 20))
        edge_attr = torch.randn(20, 6)

        output = gnn(x, edge_index, edge_attr)

        assert output.size() == (10, 64)


class TestGroupLevelGNN:
    """Test group-level GNN."""

    def test_forward(self):
        """Test forward pass."""
        gnn = GroupLevelGNN(
            group_feature_dim=8,
            hidden_dim=32,
            atom_embedding_dim=32,
            num_layers=1,
            dropout=0.2,
        )

        atom_embeddings = torch.randn(10, 32)
        groups = [[0, 1, 2], [3, 4], [5, 6, 7, 8, 9]]
        group_features = torch.randn(3, 8)

        output = gnn(atom_embeddings, groups, group_features)

        assert output.size() == (3, 32)


class TestHierarchicalMolecularGNN:
    """Test hierarchical molecular GNN."""

    def test_model_creation(self, sample_config):
        """Test model creation."""
        model = HierarchicalMolecularGNN(sample_config)

        assert model is not None
        assert model.count_parameters() > 0

    def test_forward_with_hierarchy(self, sample_config, sample_batch):
        """Test forward pass with hierarchical components."""
        sample_config['hierarchical']['enable_group_level'] = True
        model = HierarchicalMolecularGNN(sample_config)

        output = model(sample_batch)

        assert output.size() == (2, 1)

    def test_forward_without_hierarchy(self, sample_config, sample_batch):
        """Test forward pass without hierarchical components."""
        sample_config['hierarchical']['enable_group_level'] = False
        model = HierarchicalMolecularGNN(sample_config)

        output = model(sample_batch)

        assert output.size() == (2, 1)

    def test_model_eval_mode(self, sample_config, sample_batch):
        """Test model in eval mode."""
        model = HierarchicalMolecularGNN(sample_config)
        model.eval()

        with torch.no_grad():
            output = model(sample_batch)

        assert output.size() == (2, 1)
