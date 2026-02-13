"""Pytest configuration and fixtures."""

import pytest
import torch
import numpy as np


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'model': {
            'hidden_dim': 64,
            'num_layers': 2,
            'num_groups': 8,
            'dropout': 0.2,
            'atom_feature_dim': 155,
            'edge_feature_dim': 6,
            'pooling': 'attention',
            'use_edge_features': True,
        },
        'hierarchical': {
            'enable_group_level': True,
            'group_embedding_dim': 32,
            'group_message_passing_layers': 1,
        },
        'curriculum': {
            'enable': True,
            'start_epoch': 5,
            'warmup_epochs': 3,
            'max_weight': 2.0,
            'min_weight': 0.5,
        },
        'training': {
            'num_epochs': 10,
            'batch_size': 4,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'gradient_clip': 1.0,
            'early_stopping_patience': 5,
        },
        'optimizer': {
            'type': 'adam',
            'betas': [0.9, 0.999],
        },
        'lr_scheduler': {
            'type': 'cosine',
            'min_lr': 1e-5,
        },
        'data': {
            'dataset': 'tox21',
            'split_type': 'scaffold',
            'num_workers': 0,
        },
        'logging': {
            'checkpoint_dir': 'models',
            'results_dir': 'results',
        },
        'seed': 42,
        'device': 'cpu',
        'mixed_precision': False,
    }


@pytest.fixture
def sample_smiles():
    """Sample SMILES strings for testing."""
    return [
        'CCO',  # ethanol
        'CC(=O)O',  # acetic acid
        'c1ccccc1',  # benzene
        'CCN(CC)CC',  # triethylamine
        'CC(C)O',  # isopropanol
    ]


@pytest.fixture
def sample_labels():
    """Sample labels for testing."""
    return np.array([0, 1, 0, 1, 0])


@pytest.fixture
def sample_batch():
    """Sample batch for testing."""
    # Create a simple batch with 2 molecules
    return {
        'x': torch.randn(10, 155),  # 10 atoms total
        'edge_index': torch.randint(0, 10, (2, 20)),
        'edge_attr': torch.randn(20, 6),
        'batch': torch.tensor([0] * 5 + [1] * 5),  # 5 atoms per molecule
        'y': torch.tensor([0.0, 1.0]),
        'complexity': torch.tensor([0.3, 0.7]),
        'groups': [
            [[0, 1], [2, 3]],  # molecule 0 groups
            [[0, 1, 2]],  # molecule 1 groups
        ],
        'group_features': [
            torch.randn(2, 8),
            torch.randn(1, 8),
        ],
        'smiles': ['CCO', 'CC(=O)O'],
    }


@pytest.fixture
def device():
    """Test device."""
    return torch.device('cpu')
