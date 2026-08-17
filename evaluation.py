# evaluation.py
import os
os.environ['PYG_HOME'] = 'data/pyg_cache'
os.environ['TORCH_GEOMETRIC_HOME'] = 'data/pyg_cache'
os.environ['TORCH_HOME'] = 'data/torch_cache'

import torch
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

from src.data.dataset import AlloDataset
from src.models.allognn import AlloGNN

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load the gene-cold test dataset
    print("Loading gene-cold testing dataset (812 chains)...")
    test_dataset = AlloDataset(
        "data/processed/graphs",
        "data/processed/embeddings/esm2_embeddings.h5",
        "data/processed/splits/test_ids.txt"
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # 2. Initialize and load pre-trained AlloGNN weights
    print("Loading pre-trained AlloGNN model checkpoint...")
    model = AlloGNN().to(device)
    model.load_state_dict(torch.load("output/checkpoints/best_allo_20260811_104436.pt", map_location=device))
    model.eval()

    # 3. Collect predictions and ground-truths
    all_preds, all_labels, all_types = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch)
            all_preds.append(outputs['binding'].cpu())
            all_labels.append(batch.y.cpu())
            all_types.append(batch.site_type[batch.batch].cpu())
            
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    types = torch.cat(all_types).numpy()

    # 4. Generate Table II: Comparative Performance
    print("\n" + "="*60)
    print("   REPRODUCED TABLE II: COMPARATIVE PERFORMANCE (GENE-COLD TEST)")
    print("="*60)
    print(f"{'Target Site':<20} | {'AUROC':<10} | {'AUPR':<10}")
    print("-"*60)
    
    tmap = {
        0: 'Kinase Type I',
        1: 'Kinase Type I.5',
        2: 'Kinase Type II',
        3: 'Kinase Type III',
        4: 'Kinase Type IV'
    }
    
    for tid, tname in tmap.items():
        mask = (types == tid)
        if mask.sum() > 0 and labels[mask].sum() > 0:
            auroc = roc_auc_score(labels[mask], preds[mask])
            aupr = average_precision_score(labels[mask], preds[mask])
            print(f"{tname:<20} | {auroc:<10.3f} | {aupr:<10.3f}")
    print("="*60)

    # 5. Generate Table III: Residue-Level Metrics
    print("\n" + "="*70)
    print("   REPRODUCED TABLE III: RESIDUE-LEVEL METRICS (THRESHOLD = 0.5)")
    print("="*70)
    print(f"{'Target Site':<20} | {'Accuracy':<8} | {'Precision':<10} | {'Recall':<8} | {'F1-Score':<8} | {'MCC':<8}")
    print("-"*70)
    
    binary_preds = (preds >= 0.5).astype(int)
    
    # Print overall pooled metrics
    overall_acc = accuracy_score(labels, binary_preds)
    overall_prec = precision_score(labels, binary_preds)
    overall_rec = recall_score(labels, binary_preds)
    overall_f1 = f1_score(labels, binary_preds)
    overall_mcc = matthews_corrcoef(labels, binary_preds)
    print(f"{'Overall (Pooled)':<20} | {overall_acc:<8.3f} | {overall_prec:<10.3f} | {overall_rec:<8.3f} | {overall_f1:<8.3f} | {overall_mcc:<8.3f}")
    print("-"*70)
    
    # Print per-class metrics
    for tid, tname in tmap.items():
        mask = (types == tid)
        if mask.sum() > 0:
            l_sub = labels[mask]
            p_sub = binary_preds[mask]
            
            acc = accuracy_score(l_sub, p_sub)
            prec = precision_score(l_sub, p_sub)
            rec = recall_score(l_sub, p_sub)
            f1 = f1_score(l_sub, p_sub)
            mcc = matthews_corrcoef(l_sub, p_sub)
            print(f"{tname:<20} | {acc:<8.3f} | {prec:<10.3f} | {rec:<8.3f} | {f1:<8.3f} | {mcc:<8.3f}")
    print("="*70)

if __name__ == "__main__":
    main()