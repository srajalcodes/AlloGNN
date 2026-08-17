# src/data/esm2_embedder.py
import os
import torch
import pandas as pd
import numpy as np
import esm
import h5py
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
os.environ['TORCH_HOME'] = 'data/torch_cache'

def get_sliding_window_embedding(seq, model, batch_converter, device, max_len=1000, stride=500):
    """Embeds sequences longer than 1022 tokens using overlapping windows."""
    L = len(seq)
    full_emb = torch.zeros((L, 1280), dtype=torch.float32, device='cpu')
    counts = torch.zeros((L, 1), dtype=torch.float32, device='cpu')

    start = 0
    while start < L:
        end = min(start + max_len, L)
        if end == L and end - start < 50 and L > max_len:
            start = L - max_len
            
        chunk = seq[start:end]
        _, _, tokens = batch_converter([("chunk", chunk)])
        tokens = tokens.to(device)
        
        with torch.no_grad():
            res = model(tokens, repr_layers=[33], return_contacts=False)
        
        chunk_emb = res["representations"][33][0, 1:len(chunk)+1].cpu()
        
        full_emb[start:end] += chunk_emb
        counts[start:end] += 1
        
        if end == L:
            break
        start += stride
        
    full_emb /= counts
    return full_emb.numpy()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    logging.info("Loading ESM2-650M model...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    
    meta_path = Path("data/processed/structures/processed_metadata.csv")
    if not meta_path.exists():
        logging.error("Processed metadata not found. Run preprocess.py first.")
        return
        
    df = pd.read_csv(meta_path)
    out_dir = Path("data/processed/embeddings")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "esm2_embeddings.h5"
    
    success_count = 0
    with h5py.File(out_path, 'a') as h5f:
        for idx, row in df.iterrows():
            identifier = f"{row['pdb_id']}_{row['chain_id']}"
            seq = row['sequence']
            L = len(seq)
            
            # BUG FIXED: Detect existing corrupted/truncated embeddings and force repair
            if identifier in h5f:
                existing_len = h5f[identifier].shape[0]
                if existing_len == L:
                    success_count += 1
                    continue
                else:
                    logging.warning(f"Length mismatch for {identifier} (H5: {existing_len}, Meta: {L}). Repairing...")
                    del h5f[identifier]
            
            # Embed using appropriate strategy
            if L <= 1022:
                data = [(identifier, seq)]
                _, _, batch_tokens = batch_converter(data)
                batch_tokens = batch_tokens.to(device)
                
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33], return_contacts=False)
                token_reps = results["representations"][33][0, 1:L+1].cpu().numpy()
            else:
                logging.info(f"Embedding long sequence using sliding window: {identifier} (Length: {L})")
                token_reps = get_sliding_window_embedding(seq, model, batch_converter, device)
            
            h5f.create_dataset(identifier, data=token_reps, compression="gzip")
            success_count += 1
            
            if success_count % 100 == 0:
                logging.info(f"Processed {success_count}/{len(df)} embeddings...")
                
    logging.info(f"Complete! All embeddings validated and saved to {out_path}")

if __name__ == "__main__":
    main()