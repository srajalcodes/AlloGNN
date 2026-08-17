"""
Extracts sequences, 3D coordinates, and binding site labels from raw CIF files.
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from Bio.PDB import MMCIFParser, Selection
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning

# Suppress Biopython warnings for cleaner terminal output
warnings.simplefilter('ignore', PDBConstructionWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

AA_3_TO_1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

def extract_chain_data(cif_path, target_chain_id, cutoff=4.5):
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', cif_path)
    except Exception as e:
        return None, f"Failed to parse: {e}"

    model = structure[0]
    
    # 1. Identify protein chain and extract sequence/coords
    if target_chain_id not in model:
        return None, f"Chain {target_chain_id} not found."
        
    chain = model[target_chain_id]
    
    residues = []
    ca_coords = []
    seq = []
    
    for res in chain.get_residues():
        # Keep only standard amino acids (hetero flag is ' ')
        if res.id[0] == ' ' and res.resname in AA_3_TO_1:
            if 'CA' in res:
                residues.append(res)
                ca_coords.append(res['CA'].get_coord())
                seq.append(AA_3_TO_1[res.resname])
                
    if not residues:
        return None, "No standard residues with CA atoms found."

    ca_coords = np.array(ca_coords)
    sequence = ''.join(seq)
    
    # 2. Extract ligand coordinates (HETATMs not in solvent)
    ligand_coords = []
    solvent_names = {'HOH', 'WAT', 'DOD', 'SO4', 'PO4', 'CL', 'NA', 'K', 'MG', 'CA'}
    
    for res in model.get_residues():
        if res.id[0].startswith('H_') and res.resname not in solvent_names:
            for atom in res.get_atoms():
                if atom.element != 'H': # Exclude hydrogens
                    ligand_coords.append(atom.get_coord())
                    
    ligand_coords = np.array(ligand_coords)
    
    # 3. Calculate binding labels (distance < cutoff)
    labels = np.zeros(len(residues), dtype=int)
    
    if len(ligand_coords) > 0:
        for i, res in enumerate(residues):
            res_coords = np.array([atom.get_coord() for atom in res.get_atoms() if atom.element != 'H'])
            if len(res_coords) > 0:
                # Calculate pairwise distances between residue heavy atoms and all ligand heavy atoms
                dists = np.linalg.norm(res_coords[:, None, :] - ligand_coords[None, :, :], axis=2)
                if np.min(dists) <= cutoff:
                    labels[i] = 1

    return {
        'sequence': sequence,
        'ca_coords': ca_coords,
        'labels': labels,
        'num_binding_residues': int(np.sum(labels))
    }, "Success"

def main(args):
    cif_dir = Path(args.cif_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    meta_df = pd.read_csv(args.meta_path)
    
    # To store processed data
    processed_data = []
    
    logging.info(f"Processing {len(meta_df)} structures...")
    
    success_count = 0
    for idx, row in meta_df.iterrows():
        pdb_id = row['pdb_id'].lower()
        chain_id = row['chain_id']
        cif_path = cif_dir / f"{pdb_id}.cif"
        
        if not cif_path.exists():
            continue
            
        data, msg = extract_chain_data(cif_path, chain_id, args.cutoff)
        
        if data is not None:
            # Save coordinates and labels as NumPy archives
            np.savez_compressed(
                out_dir / f"{pdb_id}_{chain_id}.npz",
                ca_coords=data['ca_coords'],
                labels=data['labels']
            )
            
            processed_data.append({
                'pdb_id': pdb_id,
                'chain_id': chain_id,
                'inhibitor_type': row['inhibitor_type'],
                'sequence': data['sequence'],
                'seq_len': len(data['sequence']),
                'num_binding_residues': data['num_binding_residues']
            })
            success_count += 1
            
        if (idx + 1) % 100 == 0:
            logging.info(f"Processed {idx + 1}/{len(meta_df)}... (Success: {success_count})")
            
    # Save processed metadata
    processed_df = pd.DataFrame(processed_data)
    processed_df.to_csv(out_dir / "processed_metadata.csv", index=False)
    logging.info(f"Complete. Successfully processed {success_count} structures.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_path", type=str, default="data/processed/splits/subset_metadata.csv")
    parser.add_argument("--cif_dir", type=str, default="data/raw/kincore/cif")
    parser.add_argument("--out_dir", type=str, default="data/processed/structures")
    parser.add_argument("--cutoff", type=float, default=4.5)
    args = parser.parse_args()
    main(args)