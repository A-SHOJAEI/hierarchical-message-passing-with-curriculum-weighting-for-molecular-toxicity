"""Core hierarchical molecular GNN model."""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool

from hierarchical_message_passing_with_curriculum_weighting_for_molecular_toxicity.models.components import (
    AttentionPooling,
    GroupMessagePassing,
    MultiScaleFusion,
)

logger = logging.getLogger(__name__)


class AtomLevelGNN(nn.Module):
    """Atom-level graph neural network.

    Standard GNN for message passing on molecular graphs at the atom level.

    Args:
        input_dim: Dimension of input atom features
        hidden_dim: Hidden dimension for GNN layers
        num_layers: Number of GNN layers
        dropout: Dropout rate
        use_edge_features: Whether to use edge features
        edge_dim: Dimension of edge features
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 4,
        dropout: float = 0.2,
        use_edge_features: bool = True,
        edge_dim: int = 12,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_edge_features = use_edge_features

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(num_layers):
            if use_edge_features:
                conv = GATConv(
                    hidden_dim,
                    hidden_dim // 4,
                    heads=4,
                    dropout=dropout,
                    edge_dim=edge_dim,
                )
            else:
                conv = GCNConv(hidden_dim, hidden_dim)

            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through atom-level GNN.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]

        Returns:
            Node embeddings [num_nodes, hidden_dim]
        """
        # Input projection
        h = self.input_proj(x)
        h = F.relu(h)

        # GNN layers with residual connections
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            h_in = h

            if self.use_edge_features and edge_attr is not None:
                h = conv(h, edge_index, edge_attr=edge_attr)
            else:
                h = conv(h, edge_index)

            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            # Residual connection
            if i > 0:
                h = h + h_in

        return h


class GroupLevelGNN(nn.Module):
    """Group-level graph neural network.

    Operates on functional groups as nodes, performing higher-order
    message passing to capture interactions between chemical motifs.

    Args:
        group_feature_dim: Dimension of group features
        hidden_dim: Hidden dimension for group embeddings
        num_layers: Number of group message passing layers
        dropout: Dropout rate
    """

    def __init__(
        self,
        group_feature_dim: int,
        hidden_dim: int,
        atom_embedding_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.group_feature_dim = group_feature_dim
        self.hidden_dim = hidden_dim
        self.atom_embedding_dim = atom_embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Project group features
        self.input_proj = nn.Linear(group_feature_dim, hidden_dim)

        # Project atom embeddings to group dimension
        self.atom_to_group_proj = nn.Linear(atom_embedding_dim, hidden_dim)

        # Group message passing layers
        self.group_mps = nn.ModuleList([
            GroupMessagePassing(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        atom_embeddings: torch.Tensor,
        groups: List[List[int]],
        group_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through group-level GNN.

        Args:
            atom_embeddings: Atom embeddings from atom-level GNN [num_nodes, hidden_dim]
            groups: List of atom indices for each functional group
            group_features: Initial group features [num_groups, group_feature_dim]

        Returns:
            Group embeddings [num_groups, hidden_dim]
        """
        num_groups = len(groups)

        # Initialize group embeddings by pooling atom embeddings
        group_embeddings = []
        for group in groups:
            if len(group) == 0:
                group_embeddings.append(torch.zeros(1, self.hidden_dim, device=atom_embeddings.device))
            else:
                group_atoms = atom_embeddings[group]
                group_emb = group_atoms.mean(dim=0, keepdim=True)
                # Project to group dimension
                group_emb = self.atom_to_group_proj(group_emb)
                group_embeddings.append(group_emb)

        group_embeddings = torch.cat(group_embeddings, dim=0)  # [num_groups, hidden_dim]

        # Add learnable features (ensure group_features is on the same device)
        group_features = group_features.to(atom_embeddings.device)
        group_feat_proj = self.input_proj(group_features)
        group_embeddings = group_embeddings + group_feat_proj

        # Construct group adjacency (groups are connected if they share atoms or are close)
        group_adjacency = torch.zeros(num_groups, num_groups, device=atom_embeddings.device)
        for i in range(num_groups):
            for j in range(i + 1, num_groups):
                # Check if groups share atoms
                shared = set(groups[i]) & set(groups[j])
                if len(shared) > 0:
                    group_adjacency[i, j] = 1
                    group_adjacency[j, i] = 1

        # Group message passing
        for group_mp in self.group_mps:
            group_embeddings = group_mp(group_embeddings, group_adjacency)
            group_embeddings = F.dropout(group_embeddings, p=self.dropout, training=self.training)

        return group_embeddings


class HierarchicalMolecularGNN(nn.Module):
    """Hierarchical Molecular GNN with dual-granularity message passing.

    Combines atom-level and functional-group-level message passing
    for multi-scale molecular representation learning.

    Args:
        config: Model configuration dictionary
    """

    def __init__(self, config: Dict):
        super().__init__()

        # Extract config
        model_config = config.get('model', {})
        self.hidden_dim = model_config.get('hidden_dim', 256)
        self.num_layers = model_config.get('num_layers', 4)
        self.num_groups = model_config.get('num_groups', 16)
        self.dropout = model_config.get('dropout', 0.2)
        self.atom_feature_dim = model_config.get('atom_feature_dim', 155)
        self.edge_feature_dim = model_config.get('edge_feature_dim', 6)
        self.use_edge_features = model_config.get('use_edge_features', True)
        self.pooling = model_config.get('pooling', 'attention')

        hierarchical_config = config.get('hierarchical', {})
        self.enable_group_level = hierarchical_config.get('enable_group_level', True)
        self.group_embedding_dim = hierarchical_config.get('group_embedding_dim', 128)
        self.group_mp_layers = hierarchical_config.get('group_message_passing_layers', 2)

        # Atom-level GNN
        self.atom_gnn = AtomLevelGNN(
            input_dim=self.atom_feature_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            use_edge_features=self.use_edge_features,
            edge_dim=self.edge_feature_dim,
        )

        # Group-level GNN (optional)
        if self.enable_group_level:
            self.group_gnn = GroupLevelGNN(
                group_feature_dim=self.num_groups,
                hidden_dim=self.group_embedding_dim,
                atom_embedding_dim=self.hidden_dim,
                num_layers=self.group_mp_layers,
                dropout=self.dropout,
            )

            # Multi-scale fusion
            self.fusion = MultiScaleFusion(
                atom_dim=self.hidden_dim,
                group_dim=self.group_embedding_dim,
                output_dim=self.hidden_dim,
            )

        # Pooling layer
        if self.pooling == 'attention':
            self.pool = AttentionPooling(self.hidden_dim, num_heads=4)
        else:
            self.pool = None

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 4, 1),
        )

        logger.info(f"Initialized HierarchicalMolecularGNN with {self.count_parameters():,} parameters")

    def count_parameters(self) -> int:
        """Count number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, batch: Dict) -> torch.Tensor:
        """Forward pass through hierarchical GNN.

        Args:
            batch: Batch dictionary containing:
                - x: Node features [num_nodes, atom_feature_dim]
                - edge_index: Edge indices [2, num_edges]
                - edge_attr: Edge features [num_edges, edge_feature_dim]
                - batch: Batch assignment [num_nodes]
                - groups: List of functional groups per molecule
                - group_features: Group feature matrices

        Returns:
            Predictions [batch_size, 1]
        """
        x = batch['x']
        edge_index = batch['edge_index']
        edge_attr = batch.get('edge_attr', None)
        batch_idx = batch['batch']
        batch_size = batch_idx.max().item() + 1

        # Atom-level message passing
        atom_embeddings = self.atom_gnn(x, edge_index, edge_attr)

        # Pooling to graph-level
        if self.pool is not None:
            graph_embeddings_atom = self.pool(atom_embeddings, batch_idx, batch_size)
        else:
            graph_embeddings_atom = global_mean_pool(atom_embeddings, batch_idx)

        # Group-level processing
        if self.enable_group_level:
            # Process each graph separately for group-level GNN
            graph_embeddings_group = []

            for i in range(batch_size):
                mask = (batch_idx == i)
                atom_emb_graph = atom_embeddings[mask]

                # Get groups for this graph
                groups = batch['groups'][i]
                group_features = batch['group_features'][i]

                # Adjust group indices to local indexing
                local_groups = []
                node_mapping = torch.where(mask)[0]
                node_to_local = {node_mapping[j].item(): j for j in range(len(node_mapping))}

                for group in groups:
                    local_group = [node_to_local.get(node, -1) for node in group if node in node_to_local]
                    local_group = [idx for idx in local_group if idx >= 0]
                    local_groups.append(local_group)

                # Group-level GNN
                if len(local_groups) > 0:
                    group_emb = self.group_gnn(atom_emb_graph, local_groups, group_features)
                    # Pool group embeddings
                    graph_emb_group = group_emb.mean(dim=0, keepdim=True)
                else:
                    graph_emb_group = torch.zeros(1, self.group_embedding_dim, device=x.device)

                graph_embeddings_group.append(graph_emb_group)

            graph_embeddings_group = torch.cat(graph_embeddings_group, dim=0)

            # Fusion
            graph_embeddings = self.fusion(graph_embeddings_atom, graph_embeddings_group)
        else:
            graph_embeddings = graph_embeddings_atom

        # Prediction
        out = self.predictor(graph_embeddings)

        return out
