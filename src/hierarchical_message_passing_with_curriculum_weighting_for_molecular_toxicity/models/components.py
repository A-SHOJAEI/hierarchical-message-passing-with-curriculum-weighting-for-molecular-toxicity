"""Custom loss functions, layers, and training components."""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CurriculumWeightScheduler:
    """Adaptive curriculum weighting based on molecular complexity.

    Gradually increases the weight of complex molecules during training
    to improve model performance on challenging structures.

    Args:
        start_epoch: Epoch to start curriculum weighting
        warmup_epochs: Number of epochs to linearly increase weights
        max_weight: Maximum weight for complex molecules
        min_weight: Minimum weight for simple molecules
        schedule: Weight schedule type ('linear', 'exponential', 'cosine')
    """

    def __init__(
        self,
        start_epoch: int = 10,
        warmup_epochs: int = 5,
        max_weight: float = 3.0,
        min_weight: float = 0.5,
        schedule: str = 'linear',
    ):
        self.start_epoch = start_epoch
        self.warmup_epochs = warmup_epochs
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.schedule = schedule
        self.current_epoch = 0

    def step(self, epoch: int) -> None:
        """Update the current epoch.

        Args:
            epoch: Current training epoch
        """
        self.current_epoch = epoch

    def compute_weights(self, complexities: torch.Tensor) -> torch.Tensor:
        """Compute sample weights based on complexity scores.

        Args:
            complexities: Tensor of complexity scores [batch_size]

        Returns:
            Sample weights [batch_size]
        """
        if self.current_epoch < self.start_epoch:
            # No curriculum weighting before start epoch
            return torch.ones_like(complexities)

        # Compute progress in curriculum
        progress = min(
            (self.current_epoch - self.start_epoch) / self.warmup_epochs,
            1.0
        )

        if self.schedule == 'linear':
            weight_scale = progress
        elif self.schedule == 'exponential':
            weight_scale = 1.0 - torch.exp(-3.0 * progress)
        elif self.schedule == 'cosine':
            weight_scale = 0.5 * (1.0 - torch.cos(torch.pi * progress))
        else:
            weight_scale = progress

        # Map complexities to weights
        # Higher complexity -> higher weight
        weights = self.min_weight + (self.max_weight - self.min_weight) * complexities * weight_scale

        return weights


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.

    Focal loss down-weights easy examples and focuses on hard examples.

    Args:
        alpha: Weighting factor for positive class
        gamma: Focusing parameter (higher gamma = more focus on hard examples)
        reduction: Reduction method ('none', 'mean', 'sum')
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Predicted logits [batch_size] for binary classification
            targets: Ground truth labels [batch_size]
            weights: Optional sample weights [batch_size]

        Returns:
            Loss value
        """
        # Binary classification focal loss
        targets = targets.float()

        # Compute binary cross entropy with logits
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute probabilities
        probs = torch.sigmoid(inputs)

        # Compute p_t (probability of the true class)
        p_t = targets * probs + (1 - targets) * (1 - probs)

        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Compute focal loss
        focal_loss = alpha_t * focal_weight * bce_loss

        # Apply sample weights if provided
        if weights is not None:
            focal_loss = focal_loss * weights

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AttentionPooling(nn.Module):
    """Attention-based graph pooling layer.

    Uses learned attention weights to aggregate node features into
    a graph-level representation.

    Args:
        hidden_dim: Dimension of node features
        num_heads: Number of attention heads
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Perform attention pooling.

        Args:
            x: Node features [num_nodes, hidden_dim]
            batch: Batch assignment [num_nodes]
            batch_size: Number of graphs in batch

        Returns:
            Graph-level features [batch_size, hidden_dim]
        """
        # Multi-head attention
        Q = self.query(x).view(-1, self.num_heads, self.head_dim)
        K = self.key(x).view(-1, self.num_heads, self.head_dim)
        V = self.value(x).view(-1, self.num_heads, self.head_dim)

        # Compute attention scores
        attention_scores = torch.einsum('nhd,nhd->nh', Q, K) / (self.head_dim ** 0.5)

        # Apply attention per graph
        graph_features = []
        for i in range(batch_size):
            mask = (batch == i)
            if mask.sum() == 0:
                graph_features.append(torch.zeros(1, self.hidden_dim, device=x.device))
                continue

            graph_scores = attention_scores[mask]  # [num_nodes_in_graph, num_heads]
            graph_values = V[mask]  # [num_nodes_in_graph, num_heads, head_dim]

            # Softmax over nodes in the graph
            graph_attention = F.softmax(graph_scores, dim=0)  # [num_nodes_in_graph, num_heads]

            # Weighted sum
            graph_feat = torch.einsum('nh,nhd->hd', graph_attention, graph_values)
            graph_feat = graph_feat.reshape(1, -1)  # [1, hidden_dim]

            graph_features.append(graph_feat)

        graph_features = torch.cat(graph_features, dim=0)  # [batch_size, hidden_dim]
        graph_features = self.out_proj(graph_features)

        return graph_features


class GroupMessagePassing(nn.Module):
    """Message passing on functional group graph.

    Performs message passing between detected functional groups
    to capture higher-order chemical interactions.

    Args:
        group_dim: Dimension of group embeddings
        hidden_dim: Hidden dimension for message functions
    """

    def __init__(self, group_dim: int, hidden_dim: int):
        super().__init__()
        self.group_dim = group_dim
        self.hidden_dim = hidden_dim

        self.message_mlp = nn.Sequential(
            nn.Linear(2 * group_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, group_dim),
        )

        self.update_gru = nn.GRUCell(group_dim, group_dim)

    def forward(
        self,
        group_features: torch.Tensor,
        group_adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """Perform one step of group-level message passing.

        Args:
            group_features: Features of functional groups [num_groups, group_dim]
            group_adjacency: Adjacency matrix [num_groups, num_groups]

        Returns:
            Updated group features [num_groups, group_dim]
        """
        num_groups = group_features.size(0)

        # Compute messages
        messages = torch.zeros_like(group_features)

        for i in range(num_groups):
            neighbors = group_adjacency[i].nonzero(as_tuple=True)[0]
            if len(neighbors) == 0:
                continue

            # Aggregate messages from neighbors
            neighbor_features = group_features[neighbors]
            node_features = group_features[i].unsqueeze(0).expand(len(neighbors), -1)

            edge_features = torch.cat([node_features, neighbor_features], dim=-1)
            edge_messages = self.message_mlp(edge_features)

            # Sum messages
            messages[i] = edge_messages.sum(dim=0)

        # Update node features
        updated_features = self.update_gru(messages, group_features)

        return updated_features


class MultiScaleFusion(nn.Module):
    """Fuse atom-level and group-level representations.

    Uses attention to combine representations from different
    granularities (atoms and functional groups).

    Args:
        atom_dim: Dimension of atom features
        group_dim: Dimension of group features
        output_dim: Dimension of fused output
    """

    def __init__(self, atom_dim: int, group_dim: int, output_dim: int):
        super().__init__()
        self.atom_dim = atom_dim
        self.group_dim = group_dim
        self.output_dim = output_dim

        self.atom_proj = nn.Linear(atom_dim, output_dim)
        self.group_proj = nn.Linear(group_dim, output_dim)

        self.attention = nn.Sequential(
            nn.Linear(2 * output_dim, output_dim),
            nn.Tanh(),
            nn.Linear(output_dim, 2),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        atom_features: torch.Tensor,
        group_features: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse atom and group features.

        Args:
            atom_features: Atom-level graph features [batch_size, atom_dim]
            group_features: Group-level graph features [batch_size, group_dim]

        Returns:
            Fused features [batch_size, output_dim]
        """
        # Project to same dimension
        atom_proj = self.atom_proj(atom_features)
        group_proj = self.group_proj(group_features)

        # Compute attention weights
        combined = torch.cat([atom_proj, group_proj], dim=-1)
        attention_weights = self.attention(combined)  # [batch_size, 2]

        # Weighted combination
        fused = attention_weights[:, 0:1] * atom_proj + attention_weights[:, 1:2] * group_proj

        return fused
