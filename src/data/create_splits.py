"""
Creates gene-level train/val/test splits to prevent data leakage.
Fixed: Enforces strict 1-to-1 PDB-to-Gene mapping.
"""
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    meta_path = Path("data/processed/structures/processed_metadata.csv")
    tab_path = Path("data/raw/kincore/kincore_metadata.tab")
    out_dir = Path("data/processed/splits")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not meta_path.exists() or not tab_path.exists():
        logging.error("Metadata files missing.")
        return

    # Load processed structures
    processed = pd.read_csv(meta_path)
    processed['identifier'] = processed['pdb_id'] + "_" + processed['chain_id']
    
    # Load original TAB file which still has the Gene column
    raw_tab = pd.read_csv(tab_path, sep='\t', low_memory=False)
    
    # Extract just PDB and Gene
    gene_mapping = raw_tab[['pdb', 'Gene']].copy()
    gene_mapping.columns = ['pdb_id', 'gene']
    gene_mapping['pdb_id'] = gene_mapping['pdb_id'].str.lower()

    # BUG FIXED: Force exactly ONE gene per PDB to prevent multi-gene leakage
    gene_mapping = gene_mapping.drop_duplicates(subset=['pdb_id'], keep='first')

    # Merge to map each processed PDB to its Gene
    df = pd.merge(processed, gene_mapping, on='pdb_id', how='left')
    df['gene'] = df['gene'].fillna('unknown')
    
    # Safety Check: Drop any accidental identical chain duplicates
    df = df.drop_duplicates(subset=['identifier'], keep='first')

    # Shuffle unique genes to prevent alphabetical bias
    genes = pd.Series(df['gene'].unique()).sample(frac=1.0, random_state=42).reset_index(drop=True).values
    
    n_genes = len(genes)
    n_train = int(n_genes * 0.70)
    n_val = int(n_genes * 0.15)
    
    train_genes = set(genes[:n_train])
    val_genes = set(genes[n_train:n_train+n_val])
    test_genes = set(genes[n_train+n_val:])
    
    train_df = df[df['gene'].isin(train_genes)]
    val_df = df[df['gene'].isin(val_genes)]
    test_df = df[df['gene'].isin(test_genes)]
    
    # Save IDs
    for name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        ids = split_df['identifier'].tolist()
        out_path = out_dir / f"{name}_ids.txt"
        with open(out_path, 'w') as f:
            f.write('\n'.join(ids))
        logging.info(f"Saved {len(ids)} structures to {out_path}")

if __name__ == "__main__":
    main()