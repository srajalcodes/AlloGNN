"""
Custom PyTorch Geometric Dataset for loading multi-modal protein graphs.
"""
import h5py
import torch
from torch_geometric.data import Dataset
from pathlib import Path

class AlloDataset(Dataset):
    def __init__(self, graph_dir: str, emb_path: str, split_file: str):
        super().__init__()
        self.graph_dir = Path(graph_dir)
        self.emb_path = Path(emb_path)
        
        with open(split_file, 'r') as f:
            self.ids = f.read().splitlines()
            
        self.valid_ids = [i for i in self.ids if (self.graph_dir / f"{i}.pt").exists()]
        self.h5_file = None

    def len(self):
        return len(self.valid_ids)

    def get(self, idx):
        identifier = self.valid_ids[idx]
        graph_path = self.graph_dir / f"{identifier}.pt"
        
        data = torch.load(graph_path, weights_only=False)
        
        with h5py.File(self.emb_path, 'r') as f:
            esm_emb = torch.tensor(f[identifier][:], dtype=torch.float32)
            
        assert esm_emb.shape[0] == data.num_nodes, \
            f"Mismatch {identifier}: ESM2={esm_emb.shape[0]}, Graph={data.num_nodes}"
        
        data.x_seq = esm_emb
        
        if not hasattr(data, 'surf_x') or data.surf_x is None:
            data.surf_x = torch.zeros((data.num_nodes, 5), dtype=torch.float32)
            
        if not hasattr(data, 'site_type'):
            data.site_type = torch.tensor(-1, dtype=torch.long)
            
        return data

    def close(self):
        # BUG FIXED: h5py handles resources per call, but this prevents final-line AttributeError
        pass