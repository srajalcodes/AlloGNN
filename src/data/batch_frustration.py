# src/data/batch_frustration.py
import os
import subprocess
import multiprocessing
import logging
from pathlib import Path
from Bio.PDB import MMCIFParser, PDBIO
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning

warnings.simplefilter('ignore', PDBConstructionWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def cif_to_pdb(cif_path, pdb_path, chain_id):
    """Converts CIF to PDB, keeping only the target chain."""
    if os.path.exists(pdb_path):
        return True
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('p', cif_path)
        io = PDBIO()
        
        class ChainSelect:
            def accept_model(self, model): return True
            def accept_chain(self, chain): return chain.id == chain_id
            def accept_residue(self, residue): return residue.id[0] == ' '
            def accept_atom(self, atom): return True

        io.set_structure(structure)
        io.save(str(pdb_path), select=ChainSelect())
        return True
    except Exception as e:
        logging.error(f"Failed to convert {cif_path}: {e}")
        return False

def process_structure(args):
    cif_file, pdb_dir, out_dir, identifier = args
    pdb_id, chain_id = identifier.split('_')
    
    pdb_file = Path(pdb_dir) / f"{identifier}.pdb"
    
    # 1. Convert to PDB
    if not cif_to_pdb(cif_file, pdb_file, chain_id):
        return identifier, False
        
    # 2. Check if already computed
    conf_out = Path(out_dir) / f"{identifier}_conf.csv"
    mut_out = Path(out_dir) / f"{identifier}_mut.csv"
    if conf_out.exists() and mut_out.exists():
        return identifier, True

    # 3. Call R script
    cmd = [
        'Rscript', 'scripts/run_frustration.R',
        str(pdb_file), str(out_dir), identifier
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"Rscript failed for {identifier}:\n{result.stderr}")
        return identifier, False
        
    return identifier, True

def main():
    cif_dir = Path("data/raw/kincore/cif")
    pdb_dir = Path("data/raw/kincore/pdb")
    out_dir = Path("data/processed/frustration")
    graph_dir = Path("data/processed/graphs")
    
    pdb_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Only process structures that successfully made it into our PyG graphs
    graph_files = [f for f in os.listdir(graph_dir) if f.endswith('.pt')]
    
    tasks = []
    for fname in graph_files:
        identifier = fname.replace('.pt', '')
        pdb_id = identifier.split('_')[0]
        cif_file = cif_dir / f"{pdb_id}.cif"
        tasks.append((cif_file, pdb_dir, out_dir, identifier))

    # Use max CPU cores minus 2 (to keep system responsive)
    n_workers = max(1, multiprocessing.cpu_count() - 2)
    logging.info(f"Starting frustration calculation on {len(tasks)} structures using {n_workers} cores.")
    logging.info("This is CPU intensive and will take several hours. Leave it running.")
    
    success_count = 0
    with multiprocessing.Pool(n_workers) as pool:
        for i, (identifier, success) in enumerate(pool.imap_unordered(process_structure, tasks), 1):
            if success:
                success_count += 1
            if i % 50 == 0:
                logging.info(f"Processed {i}/{len(tasks)} structures (Success: {success_count})")

    logging.info(f"Completed. {success_count}/{len(tasks)} structures successfully analyzed.")

if __name__ == "__main__":
    main()