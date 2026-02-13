"""Hierarchical Message Passing with Curriculum Weighting for Molecular Toxicity.

Multi-scale molecular toxicity prediction using hierarchical graph neural networks
with adaptive curriculum learning.
"""

__version__ = "0.1.0"
__author__ = "Alireza Shojaei"

from hierarchical_message_passing_with_curriculum_weighting_for_molecular_toxicity.models.model import (
    HierarchicalMolecularGNN,
)
from hierarchical_message_passing_with_curriculum_weighting_for_molecular_toxicity.training.trainer import (
    Trainer,
)

__all__ = [
    "HierarchicalMolecularGNN",
    "Trainer",
]
