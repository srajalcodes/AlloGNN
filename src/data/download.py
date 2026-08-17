"""
Downloads and stratifies protein structures from RCSB PDB based on KinCoRe metadata.
"""

import os
import argparse
import logging
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def setup_directories(base_dir: str):
    dirs = [
        Path(base_dir) / "data" / "raw" / "kincore" / "cif",
        Path(base_dir) / "data" / "processed" / "splits"
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return dirs[0]

def stratify_dataset(metadata_path: str, output_path: str, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(metadata_path)
    df.columns = [col.lower().strip() for col in df.columns]
    
    # CHANGE THIS: Set all limits to None to take 100% of the raw data
    type_map = {
        'type1': None,    # Take all 7,584 available structures
        'type1.5': None,  # Take all 775 available structures
        'type2': None,    # Take all 550 available structures
        'type3': None,    # Take all 294 available structures
        'type4': None     # Take all 662 available structures
    }
    
    sampled_dfs = []
    for inhib_type, n_samples in type_map.items():
        subset = df[df['inhibitor_type'].str.lower() == inhib_type]
        if n_samples and len(subset) > n_samples:
            subset = subset.sample(n=n_samples, random_state=seed)
        sampled_dfs.append(subset)
        logging.info(f"Sampled {len(subset)} structures for {inhib_type}")
        
    stratified_df = pd.concat(sampled_dfs, ignore_index=True)
    stratified_df.to_csv(output_path, index=False)
    logging.info(f"Saved stratified metadata to {output_path}. Total: {len(stratified_df)}")
    
    return stratified_df

def download_cif(pdb_id: str, output_dir: Path) -> bool:
    pdb_id = pdb_id.lower()
    out_path = output_dir / f"{pdb_id}.cif"
    
    if out_path.exists():
        return True
        
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(out_path, 'wb') as f:
                f.write(response.content)
            return True
        return False
    except requests.RequestException:
        return False

def main(args):
    base_dir = Path(args.base_dir)
    cif_dir = setup_directories(base_dir)
    
    meta_path = base_dir / "data" / "raw" / "kincore" / "kincore_metadata.csv"
    subset_meta_path = base_dir / "data" / "processed" / "splits" / "subset_metadata.csv"
    
    if not meta_path.exists():
        logging.error(f"Metadata file not found at {meta_path}. Please download manually.")
        return
        
    df = stratify_dataset(meta_path, subset_meta_path)
    
    unique_pdbs = df['pdb_id'].unique()
    logging.info(f"Starting concurrent download of {len(unique_pdbs)} unique PDB structures.")
    
    success_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_pdb = {executor.submit(download_cif, pdb, cif_dir): pdb for pdb in unique_pdbs}
        
        for i, future in enumerate(as_completed(future_to_pdb), 1):
            pdb = future_to_pdb[future]
            try:
                if future.result():
                    success_count += 1
                if i % 100 == 0:
                    logging.info(f"Processed {i}/{len(unique_pdbs)} downloads.")
            except Exception as e:
                logging.error(f"Error downloading {pdb}: {str(e)}")
                
    logging.info(f"Download complete. Successfully retrieved {success_count}/{len(unique_pdbs)} structures.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlloGNN PDB Downloader")
    parser.add_argument("--base_dir", type=str, default=".", help="Project base directory")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent downloads")
    args = parser.parse_args()
    main(args)