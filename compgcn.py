"""
Compositional Graph Convolutional Network (CompGCN) implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from .base import BaseGNN

# --------- CompGCNConv Layer ---------
class CompGCNConv(nn.Module):
    """
    Single CompGCN convolution layer.
    """
    def __init__(self, in_channels, out_channels, num_rels, opn='corr', act=lambda x: x, dropout=0.1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_rels = num_rels
        self.opn = opn
        self.act = act
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.rel_weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.rel_weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def comp(self, h, r, opn):
        if opn == 'corr':
            return self.circular_correlation(h, r)
        elif opn == 'sub':
            return h - r
        elif opn == 'mult':
            return h * r
        else:
            raise NotImplementedError(f"Composition operator {opn} not implemented.")

    def circular_correlation(self, h, r):
        fft_h = torch.fft.fft(h, dim=-1)
        fft_r = torch.fft.fft(r, dim=-1)
        conj_fft_h = torch.conj(fft_h)
        corr = torch.fft.ifft(conj_fft_h * fft_r, dim=-1).real
        return corr

    def forward(self, x, edge_index, edge_type, rel_embed):
        num_nodes = x.size(0)
        self_loop_edge = torch.arange(0, num_nodes, dtype=torch.long, device=x.device)
        self_loop_edge = self_loop_edge.unsqueeze(0).repeat(2, 1)
        self_loop_type = torch.full((num_nodes,), self.num_rels * 2, dtype=torch.long, device=x.device)
        edge_index = torch.cat([edge_index, self_loop_edge], dim=1)
        edge_type = torch.cat([edge_type, self_loop_type], dim=0)
        rel_embed = torch.cat([rel_embed, torch.zeros(1, rel_embed.size(1), device=x.device)], dim=0)
        h = x[edge_index[0]]
        r = rel_embed[edge_type]
        msg = self.comp(h, r, self.opn)
        out = torch.zeros_like(x)
        out = out.index_add(0, edge_index[1], msg)
        out = out @ self.weight
        if self.bias is not None:
            out = out + self.bias
        out = self.act(out)
        out = self.dropout(out)
        rel_embed = rel_embed @ self.rel_weight
        return out, rel_embed

# --------- CompGCN Model ---------
class CompGCN(BaseGNN):
    """
    Compositional Graph Convolutional Network (CompGCN) model.
    Inherits from BaseGNN.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CompGCN model.

        Args:
            config: Configuration dictionary containing model parameters.
                Required keys: input_dim, hidden_dim, output_dim, num_entities, num_relations
                Optional keys: num_layers, dropout, opn, etc.
        """
        super().__init__(config)
        self.num_entities = config['num_entities']
        self.num_relations = config['num_relations']
        self.opn = config.get('opn', 'corr')  # composition operator: 'corr', 'sub', 'mult'

        # Embedding layers
        self.entity_emb = nn.Embedding(self.num_entities, self.input_dim)
        self.relation_emb = nn.Embedding(self.num_relations * 2 + 1, self.input_dim)

        # Build CompGCNConv layers
        self.layers = nn.ModuleList()
        if self.num_layers == 1:
            layer_dims = [self.input_dim, self.output_dim]
        else:
            layer_dims = [self.input_dim] + self.hidden_dims[:self.num_layers-1] + [self.output_dim]

        for i in range(self.num_layers):
            act = F.relu if i < self.num_layers - 1 else (lambda x: x)
            self.layers.append(
                CompGCNConv(
                    layer_dims[i],
                    layer_dims[i + 1],
                    self.num_relations,
                    opn=self.opn,
                    act=act,
                    dropout=self.dropout
                )
            )
        self.reset_parameters()

    def reset_parameters(self):
        self.entity_emb.reset_parameters()
        self.relation_emb.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, edge_type, **kwargs):
        """
        Forward pass.

        Args:
            x: Node indices (ignored; uses embedding layer)
            edge_index: [2, num_edges]
            edge_type: [num_edges]
            **kwargs: extra arguments (unused)

        Returns:
            Entity embeddings, relation embeddings
        """
        ent = self.entity_emb.weight
        rel = self.relation_emb.weight
        for layer in self.layers:
            ent, rel = layer(ent, edge_index, edge_type, rel)
        return ent, rel

    def get_embeddings(self, edge_index, edge_type, **kwargs):
        """
        Returns entity embeddings after all CompGCN layers.
        """
        ent, rel = self.forward(None, edge_index, edge_type)
        return ent

    def decode(self, ent_emb, rel_emb, triples):
        """
        Score triples using DistMult.
        triples: tensor of shape [3, batch_size] (head, rel, tail)
        Returns: tensor of shape [batch_size]
        """
        h = ent_emb[triples[0]]
        r = rel_emb[triples[1]]
        t = ent_emb[triples[2]]
        return torch.sum(h * r * t, dim=-1)

# --------- Factory Function ---------
def create_compgcn(input_dim: int, hidden_dim, output_dim: int, num_entities: int, num_relations: int, **kwargs) -> CompGCN:
    """
    Convenience function to create a CompGCN model.

    Args:
        input_dim: Entity/relation embedding input dimension.
        hidden_dim: Hidden layer dimension(s). Can be:
                   - int: Single dimension for all hidden layers
                   - list/tuple: Different dimensions for each hidden layer
        output_dim: Output dimension
        num_entities: Number of entities in the graph
        num_relations: Number of (base) relations in the graph
        **kwargs: Additional configuration parameters (num_layers, dropout, opn, etc.)

    Returns:
        Configured CompGCN model

    Example:
        model = create_compgcn(64, 64, 64, num_entities=14541, num_relations=474, num_layers=2)
    """
    config = {
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'output_dim': output_dim,
        'num_entities': num_entities,
        'num_relations': num_relations,
        **kwargs
    }
    return CompGCN(config)
