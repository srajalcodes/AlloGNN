# src/data/add_frustration_features.py
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MINIMAL_THRESH = 0.78
HIGH_THRESH = -1.00

def get_densities(z_scores):
    """Calculates fractions of minimal, neutral, and highly frustrated contacts."""
    if len(z_scores) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    z = np.array(z_scores)
    min_den = np.mean(z > MINIMAL_THRESH)
    neu_den = np.mean((z >= HIGH_THRESH) & (z <= MINIMAL_THRESH))
    high_den = np.mean(z < HIGH_THRESH)
    mean_z = np.mean(z)
    
    return min_den, neu_den, high_den, mean_z

def main():
    graph_dir = Path("data/processed/graphs")
    frust_dir = Path("data/processed/frustration")
    
    graph_files = [f for f in os.listdir(graph_dir) if f.endswith('.pt')]
    logging.info(f"Injecting frustration features into {len(graph_files)} graphs...")
    
    success = 0
    removed = 0
    
    for fname in graph_files:
        identifier = fname.replace('.pt', '')
        graph_path = graph_dir / fname
        
        conf_path = frust_dir / f"{identifier}_conf.csv"
        mut_path = frust_dir / f"{identifier}_mut.csv"
        
        # 1. If frustration failed (the 2 PDB limit errors), delete the graph
        if not conf_path.exists() or not mut_path.exists():
            os.remove(graph_path)
            removed += 1
            continue
            
        d = torch.load(graph_path)
        
        # Skip if already processed
        if hasattr(d, 'frust_x'):
            success += 1
            continue
            
        # Load DataFrames
        conf_df = pd.read_csv(conf_path)
        mut_df = pd.read_csv(mut_path)
        
        # Initialize node and edge feature tensors
        N = d.num_nodes
        E = d.edge_index.shape[1]
        
        frust_node_feats = np.zeros((N, 9), dtype=np.float32)
        frust_edge_feats = np.zeros((E, 3), dtype=np.float32)
        
        # Build fast lookup dictionaries for contacts
        # Assuming Frustratometer Res1 and Res2 roughly correspond to sequence indices
        # We use a tolerant approach: aggregate by standard index
        
        conf_dict = {}
        for _, row in conf_df.iterrows():
            # Frustratometer uses 1-based indexing, map to 0-based
            try:
                r1 = int(row['Res1']) - 1
                r2 = int(row['Res2']) - 1
                z = float(row['FrstIndex'])
                conf_dict.setdefault(r1, []).append(z)
                conf_dict.setdefault(r2, []).append(z)
                conf_dict[(r1, r2)] = z
                conf_dict[(r2, r1)] = z
            except (ValueError, KeyError):
                continue
                
        mut_dict = {}
        for _, row in mut_df.iterrows():
            try:
                r1 = int(row['Res1']) - 1
                r2 = int(row['Res2']) - 1
                z = float(row['FrstIndex'])
                mut_dict.setdefault(r1, []).append(z)
                mut_dict.setdefault(r2, []).append(z)
                mut_dict[(r1, r2)] = z
                mut_dict[(r2, r1)] = z
            except (ValueError, KeyError):
                continue

        # 2. Build Node Features
        for i in range(N):
            c_z = conf_dict.get(i, [])
            m_z = mut_dict.get(i, [])
            
            c_min, c_neu, c_high, c_mean = get_densities(c_z)
            m_min, m_neu, m_high, m_mean = get_densities(m_z)
            
            # The MAGIC FEATURE: Delta
            # High delta = stable structure but flexible evolution (Allosteric signature)
            delta = abs(c_mean - m_mean)
            
            frust_node_feats[i] = [
                c_min, c_neu, c_high, m_min, m_neu, m_high, c_mean, m_mean, delta
            ]
            
        # 3. Build Edge Features
        edges = d.edge_index.numpy()
        for e in range(E):
            u, v = edges[0, e], edges[1, e]
            
            c_edge_z = conf_dict.get((u, v), 0.0)
            m_edge_z = mut_dict.get((u, v), 0.0)
            e_delta = abs(c_edge_z - m_edge_z)
            
            frust_edge_feats[e] = [c_edge_z, m_edge_z, e_delta]

        # 4. Attach to PyG Data object and save
        d.frust_x = torch.tensor(frust_node_feats, dtype=torch.float32)
        d.frust_edge = torch.tensor(frust_edge_feats, dtype=torch.float32)
        
        torch.save(d, graph_path)
        success += 1
        
        if success % 200 == 0:
            logging.info(f"Injected features into {success} graphs...")
            
    logging.info(f"Done! Successfully injected features into {success} graphs. Removed {removed} bad graphs.")

if __name__ == "__main__":
    main()