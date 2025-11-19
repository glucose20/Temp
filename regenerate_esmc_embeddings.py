#!/usr/bin/env python3
"""
Script to regenerate ESM-C embeddings with correct format.
Run this to fix the vec_dict issue where embeddings were stored as 2D instead of 1D.
"""

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import os
import sys

# Add code directory to path
sys.path.insert(0, './code')

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

def regenerate_esmc_embeddings(dataset='davis', model_name='esmc_300m'):
    """
    Regenerate ESM-C embeddings with correct format.
    
    Args:
        dataset: Dataset name ('davis', 'kiba', 'metz')
        model_name: ESM-C model variant
    """
    # Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_DIMS = {
        "esmc_300m": 960,
        "esmc_600m": 1152,
        "esmc_6b": 2560
    }
    EMBEDDING_DIM = MODEL_DIMS[model_name]
    
    print(f"{'='*60}")
    print(f"Regenerating ESM-C embeddings for {dataset.upper()}")
    print(f"Model: {model_name} (dim={EMBEDDING_DIM})")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}\n")
    
    # Load model
    print(f"Loading {model_name}...")
    model = ESMC.from_pretrained(model_name).to(DEVICE)
    model.eval()
    print("Model loaded!\n")
    
    # Load protein data
    df_dir = f'./data/dta-5fold-dataset/{dataset}/{dataset}_prots.csv'
    print(f"Loading proteins from: {df_dir}")
    
    df = pd.read_csv(df_dir)
    df.drop_duplicates(subset='prot_id', inplace=True)
    
    prot_ids = df['prot_id'].tolist()
    prot_seqs = df['prot_seq'].tolist()
    
    print(f"Found {len(prot_ids)} unique proteins\n")
    
    # Extract embeddings
    emb_dict = {}
    emb_mat_dict = {}
    length_dict = {}
    
    print("Extracting embeddings...")
    for idx in tqdm(range(len(prot_ids))):
        prot_id = str(prot_ids[idx])
        seq = prot_seqs[idx][:2048]  # ESM-C max length
        length_dict[prot_id] = len(seq)
        
        try:
            # Create protein object
            protein = ESMProtein(sequence=seq)
            protein_tensor = model.encode(protein)
            
            # Extract embeddings
            with torch.no_grad():
                logits_output = model.logits(
                    protein_tensor,
                    LogitsConfig(return_embeddings=True)
                )
            
            embeddings = logits_output.embeddings.cpu().numpy()
            
            # ✅ CRITICAL: Store mean embedding as 1D vector
            emb_dict[prot_id] = embeddings.mean(axis=0)  # Shape: (d_model,)
            
            # Store full matrix
            emb_mat_dict[prot_id] = embeddings  # Shape: (seq_len, d_model)
            
        except Exception as e:
            print(f"\nError processing {prot_id}: {e}")
            emb_dict[prot_id] = np.zeros(EMBEDDING_DIM)
            emb_mat_dict[prot_id] = np.zeros((len(seq), EMBEDDING_DIM))
    
    # Save embeddings
    dump_data = {
        "dataset": dataset,
        "vec_dict": emb_dict,
        "mat_dict": emb_mat_dict,
        "length_dict": length_dict,
        "model": model_name,
        "embedding_dim": EMBEDDING_DIM
    }
    
    output_file = f'./data/{dataset}/{dataset}_esmc_pretrain.pkl'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'wb') as f:
        pickle.dump(dump_data, f)
    
    print(f"\n{'='*60}")
    print(f"✅ Successfully saved to: {output_file}")
    print(f"Total proteins: {len(emb_dict)}")
    print(f"Embedding dimension: {EMBEDDING_DIM}")
    
    # Verify format
    first_prot_id = list(emb_dict.keys())[0]
    first_vec = emb_dict[first_prot_id]
    first_mat = emb_mat_dict[first_prot_id]
    
    print(f"\nVerification:")
    print(f"  First protein: {first_prot_id}")
    print(f"  Vec shape: {first_vec.shape} {'✅' if first_vec.ndim == 1 else '❌'}")
    print(f"  Mat shape: {first_mat.shape} {'✅' if first_mat.ndim == 2 else '❌'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Regenerate ESM-C embeddings')
    parser.add_argument('--dataset', type=str, default='davis', 
                       choices=['davis', 'kiba', 'metz'],
                       help='Dataset to process')
    parser.add_argument('--model', type=str, default='esmc_300m',
                       choices=['esmc_300m', 'esmc_600m', 'esmc_6b'],
                       help='ESM-C model variant')
    
    args = parser.parse_args()
    
    # Regenerate embeddings
    regenerate_esmc_embeddings(dataset=args.dataset, model_name=args.model)
