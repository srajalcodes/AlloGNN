# run_ablations.py
import os
os.environ['PYG_HOME'] = 'data/pyg_cache'
os.environ['TORCH_GEOMETRIC_HOME'] = 'data/pyg_cache'
os.environ['TORCH_HOME'] = 'data/torch_cache'

import torch
import torch.nn as nn
from pathlib import Path
from torch_geometric.loader import DataLoader
from src.data.dataset import AlloDataset
from src.training.trainer import AlloGNNTrainer
from src.models.allognn import AlloGNN
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

# --- ABLATED ARCHITECTURES ---

class SeqOnlyAlloGNN(AlloGNN):
    """Ablation 1: Sequence Only"""
    def forward(self, data):
        seq_emb = self.seq_proj(data.x_seq)
        struct_emb = torch.zeros((data.num_nodes, 128), device=seq_emb.device)
        surf_emb = torch.zeros((data.num_nodes, 32), device=seq_emb.device)
        fused, _ = self.fusion(seq_emb, struct_emb, surf_emb)
        shared = self.shared(fused)
        return {
            'binding': torch.sigmoid(self.binding_head(shared).squeeze(-1)),
            'type': self.type_head(shared),
            'gate_weights': torch.zeros((data.num_nodes, 3), device=seq_emb.device)
        }

class StructSurfOnlyAlloGNN(AlloGNN):
    """Ablation 2: Structure & Surface Only (Zeros out Sequence)"""
    def forward(self, data):
        dummy_seq = torch.zeros_like(data.x_seq)
        seq_emb = self.seq_proj(dummy_seq)
        struct_emb = self.struct_gat(seq_emb, data.edge_index, data.edge_attr)
        surf_emb = self.surf_proj(data.surf_x)
        fused, _ = self.fusion(seq_emb, struct_emb, surf_emb)
        shared = self.shared(fused)
        return {
            'binding': torch.sigmoid(self.binding_head(shared).squeeze(-1)),
            'type': self.type_head(shared),
            'gate_weights': torch.zeros((data.num_nodes, 3), device=seq_emb.device)
        }

class SeqStructOnlyAlloGNN(AlloGNN):
    """Ablation 3: Sequence & Structure Only (Zeros out Surface)"""
    def forward(self, data):
        seq_emb = self.seq_proj(data.x_seq)
        struct_emb = self.struct_gat(seq_emb, data.edge_index, data.edge_attr)
        # Zero out the Surface stream
        surf_emb = torch.zeros((data.num_nodes, 32), device=seq_emb.device)
        fused, _ = self.fusion(seq_emb, struct_emb, surf_emb)
        shared = self.shared(fused)
        return {
            'binding': torch.sigmoid(self.binding_head(shared).squeeze(-1)),
            'type': self.type_head(shared),
            'gate_weights': torch.zeros((data.num_nodes, 3), device=seq_emb.device)
        }

class NoGatingAlloGNN(AlloGNN):
    """Ablation 4: Naive Concatenation (No Gate)"""
    def __init__(self):
        super().__init__()
        self.naive_fusion = nn.Sequential(
            nn.Linear(256 + 128 + 32, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )
        
    def forward(self, data):
        seq_emb = self.seq_proj(data.x_seq)
        struct_emb = self.struct_gat(seq_emb, data.edge_index, data.edge_attr)
        surf_emb = self.surf_proj(data.surf_x)
        concat = torch.cat([seq_emb, struct_emb, surf_emb], dim=-1)
        fused = self.naive_fusion(concat)
        shared = self.shared(fused)
        return {
            'binding': torch.sigmoid(self.binding_head(shared).squeeze(-1)),
            'type': self.type_head(shared),
            'gate_weights': torch.zeros((data.num_nodes, 3), device=seq_emb.device)
        }

def run_ablation(model_class, name, train_loader, val_loader, device):
    print(f"\n--- Running Ablation: {name} ---")
    model = model_class().to(device)
    
    # Optuna optimized parameters
    optimizer = AdamW(model.parameters(), lr=0.0008, weight_decay=1.2e-6)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=2)
    cosine = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[2])

    trainer = AlloGNNTrainer(
        model=model, optimizer=optimizer, scheduler=scheduler, device=device,
        pos_weight=8.26, gamma=1.06, alpha=1.0, beta=0.065
    )

    best_type4 = 0.0
    for epoch in range(1, 21):
        trainer.train_epoch(train_loader, epoch)
        metrics = trainer.evaluate(val_loader, epoch)
        if metrics.get('aupr_type4', 0.0) > best_type4:
            best_type4 = metrics.get('aupr_type4', 0.0)
            
    print(f"[{name}] Best Type IV AUPR: {best_type4:.4f}")
    return best_type4

def main():
    device = torch.device('cuda')
    
    train_dataset = AlloDataset("data/processed/graphs", "data/processed/embeddings/esm2_embeddings.h5", "data/processed/splits/train_ids.txt")
    val_dataset = AlloDataset("data/processed/graphs", "data/processed/embeddings/esm2_embeddings.h5", "data/processed/splits/val_ids.txt")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    results = {}
    
    results['Sequence-Only'] = run_ablation(SeqOnlyAlloGNN, "Sequence-Only", train_loader, val_loader, device)
    results['Struct+Surf-Only'] = run_ablation(StructSurfOnlyAlloGNN, "Struct+Surf-Only", train_loader, val_loader, device)
    results['Seq+Struct (No Surf)'] = run_ablation(SeqStructOnlyAlloGNN, "Seq+Struct (No Surf)", train_loader, val_loader, device)
    results['No-Gating (Concat)'] = run_ablation(NoGatingAlloGNN, "No-Gating (Concat)", train_loader, val_loader, device)
    
    print("\n================ ABLATION RESULTS ================")
    for k, v in results.items():
        print(f"{k:<25}: Type IV AUPR = {v:.4f}")
    print("--------------------------------------------------")
    print(f"{'Full AlloGNN (Baseline)':<25}: Type IV AUPR = 0.5900 (Bootstrap Mean)")
    print("==================================================")

if __name__ == "__main__":
    main()