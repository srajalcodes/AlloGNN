# src/data/graph_builder.py
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch_geometric.data import Data
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def compute_local_frames(ca_coords):
    """
    Computes a local, orthonormal coordinate frame (ex, ey, ez) per residue
    using sequential C-alpha neighbors along the backbone.
    """
    N = ca_coords.shape[0]
    ex = np.zeros((N, 3))
    ey = np.zeros((N, 3))
    ez = np.zeros((N, 3))
    
    for i in range(N):
        if 0 < i < N - 1:
            v1 = ca_coords[i+1] - ca_coords[i]
            v2 = ca_coords[i-1] - ca_coords[i]
        elif i == 0:
            v1 = ca_coords[1] - ca_coords[0]
            v2 = ca_coords[2] - ca_coords[0] if N > 2 else np.array([0.0, 0.0, 1.0])
        else: # i == N-1
            v1 = ca_coords[N-1] - ca_coords[N-2]
            v2 = ca_coords[N-1] - ca_coords[N-3] if N > 2 else np.array([0.0, 0.0, 1.0])
            
        # Gram-Schmidt Orthogonalization to build orthonormal basis
        v1_norm = np.linalg.norm(v1)
        e_x = v1 / (v1_norm + 1e-8)
        
        v2_proj = np.dot(v2, e_x) * e_x
        e_z = v2 - v2_proj
        e_z_norm = np.linalg.norm(e_z)
        e_z = e_z / (e_z_norm + 1e-8)
        
        e_y = np.cross(e_z, e_x)
        
        ex[i] = e_x
        ey[i] = e_y
        ez[i] = e_z
        
    return ex, ey, ez

def build_graph(ca_coords, cutoff=10.0):
    N = ca_coords.shape[0]

    # Calculate pairwise distances
    diff = ca_coords[:, None, :] - ca_coords[None, :, :]  # [N, N, 3]
    dists = np.sqrt((diff**2).sum(-1))  # [N, N]

    # Find edges within cutoff (excluding self-loops)
    adj_matrix = (dists <= cutoff) & (dists > 0)
    edge_index = np.vstack(np.where(adj_matrix))  # [2, E]

    # Compute local coordinate frames for all residues
    ex, ey, ez = compute_local_frames(ca_coords)

    n_edges = edge_index.shape[1]
    edge_attr = np.zeros((n_edges, 4), dtype=np.float32)

    for k in range(n_edges):
        u, v = edge_index[0, k], edge_index[1, k]
        
        # Global unit displacement vector from u to v
        global_dir = diff[u, v] / (dists[u, v] + 1e-8)
        
        # Project global vector onto the local orthonormal frame of source residue u
        dx_local = np.dot(global_dir, ex[u])
        dy_local = np.dot(global_dir, ey[u])
        dz_local = np.dot(global_dir, ez[u])
        
        edge_attr[k] = [dists[u, v] / cutoff, dx_local, dy_local, dz_local]

    return torch.tensor(edge_index, dtype=torch.long), torch.tensor(edge_attr, dtype=torch.float32)

def main():
    meta_path = Path("data/processed/structures/processed_metadata.csv")
    npz_dir = Path("data/processed/structures")
    out_dir = Path("data/processed/graphs")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(meta_path)
    type_map = {'type1': 0, 'type1.5': 1, 'type2': 2, 'type3': 3, 'type4': 4}

    logging.info("Building E(3)-Invariant PyTorch Geometric Graphs...")
    
    success = 0
    for idx, row in df.iterrows():
        pdb_id = row['pdb_id']
        chain_id = row['chain_id']
        inhib_type = type_map.get(row['inhibitor_type'], -1)
        identifier = f"{pdb_id}_{chain_id}"
        
        npz_path = npz_dir / f"{identifier}.npz"
        out_path = out_dir / f"{identifier}.pt"
        
        if not npz_path.exists():
            continue
            
        data = np.load(npz_path)
        ca_coords = data['ca_coords']
        labels = data['labels']
        
        edge_index, edge_attr = build_graph(ca_coords)
        
        graph_data = Data(
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(labels, dtype=torch.float32),
            pos=torch.tensor(ca_coords, dtype=torch.float32),
            site_type=torch.tensor(inhib_type, dtype=torch.long),
            identifier=identifier
        )
        
        torch.save(graph_data, out_path)
        success += 1
        
        if success % 200 == 0:
            logging.info(f"Built {success}/{len(df)} invariant graphs...")

    logging.info(f"Done! Saved {success} invariant PyTorch graphs to {out_dir}")

if __name__ == "__main__":
    main()