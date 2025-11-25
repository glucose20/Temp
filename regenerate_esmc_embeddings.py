#!/usr/bin/env python3
"""
Script to regenerate ESM-C embeddings with correct format.
Run this to fix the vec_dict issue where embeddings were stored as 2D instead of 1D.
Supports esmc_300m, esmc_600m (local), and esmc_6b (via Forge API).

Usage:
    # For esmc_300m or esmc_600m (local models)
    python regenerate_esmc_embeddings.py --dataset davis --model esmc_300m
    python regenerate_esmc_embeddings.py --dataset kiba --model esmc_600m
    
    # For esmc_6b (requires Forge API token)
    export ESM_API_KEY="your_token_here"
    python regenerate_esmc_embeddings.py --dataset metz --model esmc_6b
    
    # Or pass token directly
    python regenerate_esmc_embeddings.py --dataset davis --model esmc_6b --forge-token "your_token_here"
    
Get your Forge API token from: https://forge.evolutionaryscale.ai
"""

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import os
import sys

# Add code directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'code'))

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

def regenerate_esmc_embeddings(dataset='davis', model_name='esmc_300m', forge_token=None):
    """
    Regenerate ESM-C embeddings with correct format.
    
    Args:
        dataset: Dataset name ('davis', 'kiba', 'metz')
        model_name: ESM-C model variant
        forge_token: ESM Forge API token (required for esmc_6b)
    """
    # Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_DIMS = {
        "esmc_300m": 960,
        "esmc_600m": 1152,
        "esmc_6b": 2560  # Via Forge API
    }
    EMBEDDING_DIM = MODEL_DIMS[model_name]
    
    # Determine if using Forge API
    use_forge_api = (model_name == "esmc_6b")
    
    print(f"{'='*60}")
    print(f"Regenerating ESM-C embeddings for {dataset.upper()}")
    print(f"Model: {model_name} (dim={EMBEDDING_DIM})")
    print(f"Using Forge API: {use_forge_api}")
    print(f"Device: {DEVICE if not use_forge_api else 'Forge (server-side)'}")
    print(f"{'='*60}\n")
    
    # Load model
    if use_forge_api:
        if not forge_token:
            raise ValueError("Forge API token is required for esmc_6b. Set ESM_API_KEY environment variable or pass --forge-token")
        
        print(f"Connecting to ESM Forge API for {model_name}...")
        from esm.sdk.forge import ESM3ForgeInferenceClient
        
        model = ESM3ForgeInferenceClient(
            model="esmc-6b-2024-12",
            url="https://forge.evolutionaryscale.ai",
            token=forge_token
        )
        print("Connected to Forge API successfully!\n")
    else:
        print(f"Loading {model_name}...")
        model = ESMC.from_pretrained(model_name).to(DEVICE)
        model.eval()
        print("Model loaded!\n")
    
    # Load protein data
    # Use absolute path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df_dir = os.path.join(script_dir, 'data', 'dta-5fold-dataset', dataset, f'{dataset}_prots.csv')
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
    
    if use_forge_api:
        # Use Forge API with batch executor
        print("Using Forge Batch Executor (server-side GPU processing)...")
        from esm.sdk import batch_executor
        
        def embed_sequence(client, sequence, prot_id):
            """Helper function for batch processing"""
            try:
                seq = sequence[:2048]
                protein = ESMProtein(sequence=seq)
                protein_tensor = client.encode(protein)
                
                from esm.sdk.api import ESMProteinError
                if isinstance(protein_tensor, ESMProteinError):
                    raise protein_tensor
                
                output = client.logits(
                    protein_tensor,
                    LogitsConfig(sequence=True, return_embeddings=True)
                )
                
                # CRITICAL: Convert BFloat16 immediately after receiving from API
                embeddings = output.embeddings
                
                # Convert to torch tensor first if needed, then to float32
                if not isinstance(embeddings, torch.Tensor):
                    embeddings = torch.tensor(embeddings)
                
                if embeddings.dtype == torch.bfloat16:
                    embeddings = embeddings.to(torch.float32)
                
                # Now convert to numpy (Forge API runs on server, so CPU here is fine)
                embeddings = embeddings.cpu().numpy()
                
                return prot_id, embeddings, len(seq)
                
            except Exception as e:
                print(f"\nError processing {prot_id}: {e}")
                return prot_id, None, len(sequence[:2048])
        
        # Process in batches using Forge executor
        with batch_executor() as executor:
            # Prepare data for batch execution
            batch_data = [
                {'client': model, 'sequence': seq, 'prot_id': pid}
                for pid, seq in zip(prot_ids, prot_seqs)
            ]
            
            outputs = executor.execute_batch(
                user_func=embed_sequence,
                **{k: [d[k] for d in batch_data] for k in batch_data[0].keys()}
            )
        
        # Process outputs
        for prot_id, embeddings, seq_len in tqdm(outputs, desc="Processing results"):
            length_dict[prot_id] = seq_len
            
            if embeddings is not None:
                # Ensure float32 dtype
                if embeddings.dtype != np.float32:
                    embeddings = embeddings.astype(np.float32)
                
                # ✅ CRITICAL: Store mean embedding as 1D vector
                emb_dict[prot_id] = embeddings.mean(axis=0)  # Shape: (d_model,)
                emb_mat_dict[prot_id] = embeddings  # Shape: (seq_len, d_model)
            else:
                # Fallback to zeros
                emb_dict[prot_id] = np.zeros(EMBEDDING_DIM, dtype=np.float32)
                emb_mat_dict[prot_id] = np.zeros((seq_len, EMBEDDING_DIM), dtype=np.float32)
    
    else:
        # Local model processing - USE GPU!
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
                
                embeddings = logits_output.embeddings
                
                # Convert BFloat16 to Float32 if needed (still on GPU)
                if embeddings.dtype == torch.bfloat16:
                    embeddings = embeddings.to(torch.float32)
                
                # Only move to CPU at the last step before numpy conversion
                embeddings = embeddings.cpu().numpy()
                
                # ✅ CRITICAL: Store mean embedding as 1D vector
                emb_dict[prot_id] = embeddings.mean(axis=0)  # Shape: (d_model,)
                
                # Store full matrix
                emb_mat_dict[prot_id] = embeddings  # Shape: (seq_len, d_model)
                
            except Exception as e:
                print(f"\nError processing {prot_id}: {e}")
                emb_dict[prot_id] = np.zeros(EMBEDDING_DIM, dtype=np.float32)
                emb_mat_dict[prot_id] = np.zeros((len(seq), EMBEDDING_DIM), dtype=np.float32)
    
    # Save embeddings
    dump_data = {
        "dataset": dataset,
        "vec_dict": emb_dict,
        "mat_dict": emb_mat_dict,
        "length_dict": length_dict,
        "model": model_name,
        "embedding_dim": EMBEDDING_DIM
    }
    
    # Determine output filename based on model
    if model_name == "esmc_6b":
        output_filename = f'{dataset}_esmc_6b_pretrain.pkl'
    elif model_name == "esmc_600m":
        output_filename = f'{dataset}_esmc_600m_pretrain.pkl'
    else:
        output_filename = f'{dataset}_esmc_pretrain.pkl'
        
    output_file = os.path.join(script_dir, 'data', dataset, output_filename)
        
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
                       help='ESM-C model variant (esmc_6b requires Forge API)')
    parser.add_argument('--forge-token', type=str, default=None,
                       help='ESM Forge API token (for esmc_6b). Can also set ESM_API_KEY environment variable')
    
    args = parser.parse_args()
    
    # Get Forge token from args or environment
    forge_token = args.forge_token or os.environ.get('ESM_API_KEY')
    
    # Validate for 6B model
    if args.model == 'esmc_6b' and not forge_token:
        print("ERROR: esmc_6b requires ESM Forge API token.")
        print("Please provide token via --forge-token argument or ESM_API_KEY environment variable.")
        print("Get your token from: https://forge.evolutionaryscale.ai")
        sys.exit(1)
    
    # Regenerate embeddings
    regenerate_esmc_embeddings(dataset=args.dataset, model_name=args.model, forge_token=forge_token)
