# add_surface_features.py
import os
import torch
import numpy as np
from pathlib import Path
from Bio.PDB import MMCIFParser
from Bio.PDB.SASA import ShrakeRupley
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning

warnings.simplefilter('ignore', PDBConstructionWarning)

def main():
    print("=== Calculating Surface Features (SASA & B-factors) ===")
    
    graph_dir = Path("data/processed/graphs")
    cif_dir = Path("data/raw/kincore/cif")
    graph_files = [f for f in os.listdir(graph_dir) if f.endswith('.pt')]
    
    parser = MMCIFParser(QUIET=True)
    sr = ShrakeRupley()
    
    success = 0
    for i, fname in enumerate(graph_files):
        graph_path = graph_dir / fname
        d = torch.load(graph_path)
        
        # Skip if already calculated
        if hasattr(d, 'surf_x'):
            success += 1
            continue
            
        pdb_id, chain_id = fname.replace('.pt', '').split('_')
        cif_path = cif_dir / f"{pdb_id}.cif"
        
        surf_features = np.zeros((d.num_nodes, 5), dtype=np.float32)
        
        try:
            structure = parser.get_structure('p', cif_path)
            chain = structure[0][chain_id]
            sr.compute(structure, level="R") 
            
            res_idx = 0
            for res in chain.get_residues():
                if res.id[0] == ' ' and res.resname in [
                    'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
                    'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']:
                    
                    if res_idx >= d.num_nodes:
                        break
                        
                    sasa = res.sasa if hasattr(res, 'sasa') else 0.0
                    surf_features[res_idx, 0] = min(sasa / 200.0, 1.0)
                    
                    if 'CA' in res:
                        surf_features[res_idx, 1] = min(res['CA'].get_bfactor() / 100.0, 1.0)
                        
                    res_idx += 1
                    
            d.surf_x = torch.tensor(surf_features, dtype=torch.float32)
            torch.save(d, graph_path)
            success += 1
            
        except Exception as e:
            d.surf_x = torch.tensor(surf_features, dtype=torch.float32)
            torch.save(d, graph_path)
            
        if (i + 1) % 200 == 0:
            print(f"Processed {i + 1}/{len(graph_files)} graphs...")
            
    print(f"Complete. Added surf_x to {success} graphs.")

if __name__ == "__main__":
    main()