#!/usr/bin/env python3
"""
graph autoencoder for anomaly detection. encoder: GCN/GraphSAGE/GIN/GAT, decoder: reconstructs node features.
anomaly score = reconstruction error. supports VGAE variant.
"""
from typing import Optional, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, SAGEConv, GINConv, GATConv,
    global_mean_pool, global_max_pool, global_add_pool
)


class GCNAnomaly(nn.Module):
    """
    graph autoencoder. encoder types: GCN, GraphSAGE, GIN, GAT. trains on background, uses reconstruction error as anomaly score.
    """
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        latent_dim: int = 16,
        num_layers: int = 3,
        dropout: float = 0.1,
        encoder_type: Literal["GCN", "GraphSAGE", "GIN", "GAT"] = "GCN",
        sage_agg: Literal["mean", "max", "lstm"] = "mean",
        variational: bool = False
    ):
        super().__init__()
        assert num_layers >= 2, "num_layers must be >= 2"
        self.dropout = dropout
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type
        self.variational = variational
        
        # encoder layers
        self.encoder_convs = nn.ModuleList()
        
        # first layer
        if encoder_type == "GCN":
            self.encoder_convs.append(GCNConv(in_channels, hidden_channels))
        elif encoder_type == "GraphSAGE":
            self.encoder_convs.append(SAGEConv(in_channels, hidden_channels, aggr=sage_agg))
        elif encoder_type == "GIN":
            # gin uses mlp
            mlp = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.encoder_convs.append(GINConv(mlp, train_eps=True))
        elif encoder_type == "GAT":
            self.encoder_convs.append(GATConv(in_channels, hidden_channels, heads=1, concat=False, dropout=dropout))
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        
        # middle layers
        for _ in range(num_layers - 2):
            if encoder_type == "GCN":
                self.encoder_convs.append(GCNConv(hidden_channels, hidden_channels))
            elif encoder_type == "GraphSAGE":
                self.encoder_convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=sage_agg))
            elif encoder_type == "GIN":
                mlp = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels)
                )
                self.encoder_convs.append(GINConv(mlp, train_eps=True))
            elif encoder_type == "GAT":
                self.encoder_convs.append(GATConv(hidden_channels, hidden_channels, heads=1, concat=False, dropout=dropout))
        
        # final encoder layer
        if encoder_type == "GCN":
            self.encoder_convs.append(GCNConv(hidden_channels, hidden_channels))
        elif encoder_type == "GraphSAGE":
            self.encoder_convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=sage_agg))
        elif encoder_type == "GIN":
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.encoder_convs.append(GINConv(mlp, train_eps=True))
        elif encoder_type == "GAT":
            self.encoder_convs.append(GATConv(hidden_channels, hidden_channels, heads=1, concat=False, dropout=dropout))
        
        # latent projection (vgae needs mean/logvar)
        if variational:
            self.encoder_proj_mean = nn.Linear(hidden_channels, latent_dim)
            self.encoder_proj_logvar = nn.Linear(hidden_channels, latent_dim)
        else:
            self.encoder_proj = nn.Linear(hidden_channels, latent_dim)
        
        # decoder: always gcn
        self.decoder_proj = nn.Linear(latent_dim, hidden_channels)
        self.decoder_convs = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.decoder_convs.append(GCNConv(hidden_channels, hidden_channels))
        self.decoder_convs.append(GCNConv(hidden_channels, in_channels))

    def encode(self, x, edge_index, batch):
        """encode graph to latent z."""
        for conv in self.encoder_convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # global pooling
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        g = global_mean_pool(x, batch)
        
        # project to latent
        if self.variational:
            z_mean = self.encoder_proj_mean(g)
            z_logvar = self.encoder_proj_logvar(g)
            # reparam trick
            if self.training:
                std = torch.exp(0.5 * z_logvar)
                eps = torch.randn_like(std)
                z = z_mean + eps * std
            else:
                z = z_mean
            return z, x, z_mean, z_logvar
        else:
            z = self.encoder_proj(g)
            return z, x

    def decode(self, z, edge_index, batch):
        """decode latent z back to node features."""
        # get num nodes
        if batch is None:
            num_nodes = edge_index.max().item() + 1 if edge_index.numel() > 0 else 1
            batch = z.new_zeros(num_nodes, dtype=torch.long)
        else:
            num_nodes = batch.size(0)
        
        # expand latent to each node
        z_expanded = z[batch] # [num_nodes, latent_dim]
        
        # project to hidden
        x = self.decoder_proj(z_expanded)
        x = F.relu(x)
        
        # decoder layers
        for conv in self.decoder_convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # final layer (no activation)
        x = self.decoder_convs[-1](x, edge_index)
        return x

    def forward(self, data):
        """encode then decode. returns x_recon, z (and z_mean, z_logvar for VGAE)."""
        x, edge_index, batch = data.x, data.edge_index, getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        
        # encode
        encode_result = self.encode(x, edge_index, batch)
        if self.variational:
            z, encoded_nodes, z_mean, z_logvar = encode_result
        else:
            z, encoded_nodes = encode_result
        
        # decode
        x_recon = self.decode(z, edge_index, batch)
        
        if self.variational:
            return x_recon, z, z_mean, z_logvar
        else:
            return x_recon, z


