# src/models/allognn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class StreamGAT(nn.Module):
    """Encodes structural topology using Graph Attention Networks."""
    def __init__(self, in_dim, hidden_dim, out_dim, n_heads=4, n_layers=3, edge_dim=4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(n_layers):
            self.layers.append(
                GATConv(hidden_dim, hidden_dim // n_heads, heads=n_heads, 
                        edge_dim=edge_dim, concat=True, dropout=0.1)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
            
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index, edge_attr):
        x = self.proj(x)
        for gat, norm in zip(self.layers, self.norms):
            residual = x
            x = gat(x, edge_index, edge_attr=edge_attr)
            x = self.dropout(x)
            x = norm(x + residual)
            x = F.gelu(x)
        return self.out_proj(x)

class BalancedGatingFusion(nn.Module):
    """
    Projects all streams to a common 128-dim space before gating.
    This prevents the sequence stream from dominating the soft attention weights.
    """
    def __init__(self, seq_dim=256, struct_dim=128, surf_dim=32, out_dim=256):
        super().__init__()
        # Project each stream to a balanced 128-dimensional representation
        self.seq_proj = nn.Linear(seq_dim, 128)
        self.struct_proj = nn.Linear(struct_dim, 128)
        self.surf_proj = nn.Linear(surf_dim, 128)
        
        # Gating network now operates on balanced [128 * 3] dimensions
        self.gate = nn.Sequential(
            nn.Linear(128 * 3, 128),
            nn.GELU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1)
        )
        self.out_proj = nn.Linear(128, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, seq_x, struct_x, surf_x):
        s = F.gelu(self.seq_proj(seq_x))
        t = F.gelu(self.struct_proj(struct_x))
        u = F.gelu(self.surf_proj(surf_x))
        
        # Shape: [N, 384] - completely balanced input
        concat = torch.cat([s, t, u], dim=-1)
        weights = self.gate(concat)
        
        # Blend representations in the 128-dim space
        fused = weights[:, 0:1] * s + weights[:, 1:2] * t + weights[:, 2:3] * u
        return self.norm(self.out_proj(fused)), weights

class AlloGNN(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.seq_proj = nn.Sequential(
            nn.Linear(1280, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )
        
        self.struct_gat = StreamGAT(in_dim=256, hidden_dim=256, out_dim=128, edge_dim=4)
        
        self.surf_proj = nn.Sequential(
            nn.Linear(5, 32),
            nn.LayerNorm(32),
            nn.GELU()
        )
        
        # Integrated Balanced Gating Fusion
        self.fusion = BalancedGatingFusion(seq_dim=256, struct_dim=128, surf_dim=32, out_dim=256)
        
        self.shared = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.binding_head = nn.Linear(128, 1)
        self.type_head = nn.Linear(128, 5)

    def forward(self, data):
        seq_emb = self.seq_proj(data.x_seq)
        struct_emb = self.struct_gat(seq_emb, data.edge_index, data.edge_attr)
        surf_emb = self.surf_proj(data.surf_x)
        
        fused, gate_weights = self.fusion(seq_emb, struct_emb, surf_emb)
        shared = self.shared(fused)
        
        return {
            'binding': torch.sigmoid(self.binding_head(shared).squeeze(-1)),
            'type': self.type_head(shared),
            'gate_weights': gate_weights
        }